"""Run benchmark jobs on fresh Modal containers; see docs/architecture.md#benchmark-tool."""

import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp
import modal

from .benchmark import (
    expand_jobs,
    file_inputs,
    function_name,
    gpu_matches,
    is_api_prompt,
    load_spec,
    modal_gpu,
    output_files,
    output_name,
    run_finished,
    select_jobs,
    summarize,
    unknown_nodes,
)
from .config import COMFY_COMMIT
from .helpers.http import get_json
from .modal_app import custom_nodes, image, models

PROBE = "/opt/benchmark_probe.py"
RECORDS = Path("/tmp/benchmark-records.jsonl")
OUTPUTS = Path("/tmp/comfy-output")

app = modal.App("comfyui-benchmark")


async def _run_job(prompt: dict, inputs: dict, job: dict, launched_at: float) -> dict:
    import os
    import subprocess
    import uuid
    from importlib import metadata

    from .runtime import BASE_URL, ComfyRuntime

    started = time.time()
    row = {**job, "queue_wait": started - launched_at,
           "clock_note": "queue_wait compares local and container clocks; approximate"}
    row["hardware"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not gpu_matches(job["gpu"], row["hardware"]):
        return {**row, "error": f"Requested {job['gpu']}, received {row['hardware']}"}
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    row["pip_freeze_sha256"] = hashlib.sha256("\n".join(packages).encode()).hexdigest()
    for name, data in inputs.items():
        target = Path("/tmp/comfy-input") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    RECORDS.write_text("")
    await models.reload.aio()
    runtime = ComfyRuntime()
    original_spawn = asyncio.create_subprocess_exec
    threads = str(job["cpu"])

    async def spawn(*args, **kwargs):
        if len(args) > 1 and args[1] == "main.py":
            args = (args[0], PROBE, *args[2:], *job["flags"])
            kwargs["env"] = {**os.environ, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
        return await original_spawn(*args, **kwargs)

    ready_started = time.perf_counter()
    try:
        asyncio.create_subprocess_exec = spawn
        try:
            await runtime.start()
        finally:
            asyncio.create_subprocess_exec = original_spawn
        container_to_ready = time.perf_counter() - ready_started
        client = str(uuid.uuid4())
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"{BASE_URL}/ws?clientId={client}") as ws:
                async with session.post(
                    f"{BASE_URL}/prompt", json={"prompt": prompt, "client_id": client}
                ) as response:
                    submitted = await response.json()
                    if response.status != 200:
                        return {**row, "error": json.dumps(submitted)}
                prompt_id = submitted["prompt_id"]
                async with asyncio.timeout(3000):
                    async for message in ws:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        if run_finished(json.loads(message.data), prompt_id):
                            break
            history = {}
            for _ in range(30):
                async with session.get(f"{BASE_URL}/history/{prompt_id}") as response:
                    history = (await response.json()).get(prompt_id) or {}
                if history:
                    break
                await asyncio.sleep(1)
        records = [json.loads(line) for line in RECORDS.read_text().splitlines()]
        summary = summarize(records, history.get("status") or {})
        outputs = [(name, path.read_bytes()) for name, path in output_files(history, OUTPUTS)]
        # ponytail: outputs return as bytes in the call result; switch to the jobs
        # volume if outputs exceed a few hundred MB per job.
        summary["seconds"] = {"queue_wait": row.pop("queue_wait"),
                              "container_to_ready": container_to_ready, **summary["seconds"]}
        return {**row, **summary, "outputs": outputs}
    finally:
        await runtime.stop()


def register(job):
    """Declare a function with this job's real GPU/CPU/memory before app.run()."""

    async def run_job(prompt: dict, inputs: dict, job: dict, launched_at: float) -> dict:
        return await _run_job(prompt, inputs, job, launched_at)

    return app.function(
        name=function_name(job),
        image=image.add_local_file(Path(__file__).with_name("benchmark_probe.py"), PROBE),
        volumes={"/models": models},
        gpu=modal_gpu(job["gpu"]),
        cpu=(job["cpu"], job["cpu"]),
        memory=job["memory_mib"],
        timeout=3600,
        scaledown_window=2,
        # Every job must start cold; never route a second call to a finished container.
        single_use_containers=True,
        # Names exist only for this run, so the container can't find them by import.
        serialized=True,
    )(run_job)


async def convert(workflows: list[Path], bridge_url: str) -> dict:
    """UI workflow JSON → API prompt, via the editor's own graphToPrompt."""
    prompts, pending = {}, {}
    for path in workflows:
        data = json.loads(path.read_text())
        if is_api_prompt(data):
            prompts[path] = data
        else:
            pending[path] = data
    if not pending:
        return prompts
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError("Install dev dependencies first: uv sync") from None
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            try:
                await page.goto(bridge_url, wait_until="networkidle", timeout=60000)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot open the editor at {bridge_url}; start it first: "
                    f"uv run comfy-bridge serve ({exc})"
                ) from None
            await page.wait_for_function("window.comfyAPI?.app?.app?.graph != null", timeout=60000)
            for path, workflow in pending.items():
                prompts[path] = await page.evaluate(
                    """async (workflow) => {
                        const app = window.comfyAPI.app.app;
                        await app.loadGraphData(workflow);
                        return (await app.graphToPrompt()).output;
                    }""",
                    workflow,
                )
        finally:
            await browser.close()
    return prompts


async def preflight(prompts: dict, bridge_url: str, explicit_inputs: dict | None = None) -> dict:
    """Unknown nodes or unfetchable inputs abort before any GPU is requested."""
    async with aiohttp.ClientSession() as session:
        try:
            object_info = await get_json(session, bridge_url + "/object_info")
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Bridge not reachable at {bridge_url}: {exc}") from None
        inputs = {}
        for path, prompt in prompts.items():
            unknown = unknown_nodes(prompt, object_info)
            if unknown:
                raise RuntimeError(
                    f"{path.name} uses nodes the deployment lacks: {', '.join(unknown)}. "
                    "Add them to custom_nodes.yaml, redeploy, then run comfy-bridge sync."
                )
            inputs[path] = dict(explicit_inputs or {})
            for value, name in sorted(file_inputs(prompt, object_info).items()):
                if name in inputs[path]:
                    inputs[path][value] = inputs[path][name]
                    continue
                async with session.get(
                    bridge_url + "/view", params={"filename": name, "type": "input"}
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"{path.name}: input file {name!r} is not in the editor's uploads"
                        )
                    inputs[path][value] = await response.read()
    return inputs


async def run_benchmark(args) -> int:
    spec = load_spec(args.spec)
    jobs = select_jobs(expand_jobs(spec), args.only)
    names = {job["workflow"]: job["name"] for job in jobs}
    workflows = sorted(names)
    prompts = await convert(workflows, args.bridge_url)
    explicit_inputs = {name: path.read_bytes() for name, path in spec["inputs"].items()}
    inputs = await preflight(prompts, args.bridge_url, explicit_inputs)
    folder = Path(".bridge/verification/benchmark") / time.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    provenance = {
        "spec": str(args.spec),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "only": [job["id"] for job in jobs] if args.only else None,
        "deployment": {
            "comfy_commit": COMFY_COMMIT,
            "custom_nodes": [{"name": n["name"], "revision": n["revision"]} for n in custom_nodes],
            "pip_freeze_sha256": None,
            "label": args.label or None,
        },
        "workflows": {
            names[path]: {
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompts[path], sort_keys=True).encode()
                ).hexdigest(),
                "inputs_sha256": {n: hashlib.sha256(d).hexdigest() for n, d in inputs[path].items()},
            }
            for path in workflows
        },
    }
    rows = []

    def save():
        rows.sort(key=lambda r: r["id"])
        (folder / "results.json").write_text(
            json.dumps({"provenance": provenance, "jobs": rows}, indent=2) + "\n"
        )

    async def one(job):
        meta = {**job, "workflow": job["name"]}
        print(f"Job {job['id']}: {job['gpu']} {job['cpu']}cpu {meta['workflow']}", flush=True)
        try:
            row = await workers[function_name(job)].remote.aio(prompts[job["workflow"]], inputs[job["workflow"]],
                                          meta, time.time())
        except Exception as exc:
            row = {**meta, "error": f"{type(exc).__name__}: {exc}"}
        saved = []
        for index, (remote_name, data) in enumerate(row.pop("outputs", []), 1):
            saved.append(output_name(job, index, remote_name))
            (folder / saved[-1]).write_bytes(data)
        row["outputs"] = saved
        if row.get("pip_freeze_sha256") and not provenance["deployment"]["pip_freeze_sha256"]:
            provenance["deployment"]["pip_freeze_sha256"] = row["pip_freeze_sha256"]
        rows.append(row)
        save()
        sampling = (row.get("seconds") or {}).get("sampling")
        print(f"Job {job['id']}: {row.get('error') or f'sampling {sampling:.1f}s'}", flush=True)

    shapes = {function_name(job): job for job in jobs}
    workers = {name: register(job) for name, job in shapes.items()}
    save()
    with modal.enable_output():
        async with app.run():
            await asyncio.gather(*(one(job) for job in jobs))
    print(f"Results: {folder / 'results.json'}")
    return 1 if any(row.get("error") for row in rows) else 0
