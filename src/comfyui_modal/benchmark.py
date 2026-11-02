"""Benchmark spec parsing, job expansion and result handling; no Modal imports."""

import json
from pathlib import Path, PurePosixPath

from .helpers.comfy_history import iter_output_items
from .helpers.paths import join_subfolder, safe_path
from .helpers.yaml_io import read_yaml

CONFIG_KEYS = {"gpu", "cpu", "flags", "memory_mib"}


def load_spec(path):
    path = Path(path)
    data = read_yaml(path, f"Cannot read benchmark spec {path}")
    if not isinstance(data, dict) or not data.get("workflows") or not data.get("configs"):
        raise RuntimeError("Benchmark spec needs non-empty 'workflows' and 'configs' lists")
    workflows = []
    for name in data["workflows"]:
        workflow = (path.parent / str(name)).resolve()
        if not workflow.is_file():
            raise RuntimeError(f"Workflow file not found: {name}")
        workflows.append(workflow)
    configs = []
    for config in data["configs"]:
        if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
            raise RuntimeError(f"Config keys must be among {sorted(CONFIG_KEYS)}: {config}")
        cpu, flags = config.get("cpu"), config.get("flags", [])
        memory = config.get("memory_mib", 32768)
        if not isinstance(config.get("gpu"), str):
            raise RuntimeError(f"Config needs a gpu string: {config}")
        if type(cpu) is not int or cpu < 1 or type(memory) is not int or memory < 1:
            raise RuntimeError(f"cpu and memory_mib must be positive integers: {config}")
        if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
            raise RuntimeError(f"flags must be a list of strings: {config}")
        configs.append({"gpu": config["gpu"], "cpu": cpu, "flags": flags, "memory_mib": memory})
    names = [str(name) for name in data["workflows"]]
    inputs = data.get("inputs", {})
    if not isinstance(inputs, dict):
        raise RuntimeError("inputs must map worker filenames to local file paths")
    files = {}
    for name, source in inputs.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise RuntimeError("Input destinations must be relative paths without traversal")
        if not isinstance(source, str) or not source:
            raise RuntimeError("Input sources must be non-empty file paths")
        local = path.parent / source
        # Reject protected paths before touching their contents, including links
        # and backup filenames. This command never loads private credentials.
        for candidate in (local, local.resolve()):
            if "credentials.yaml" in candidate.name or "credentials.yml" in candidate.name:
                raise RuntimeError("Credentials files cannot be benchmark inputs")
        if not local.is_file():
            raise RuntimeError(f"Benchmark input file not found: {source}")
        files[name] = local.resolve()
    return {"workflows": workflows, "names": names, "configs": configs, "inputs": files}


def expand_jobs(spec):
    workflows = list(zip(spec["workflows"], spec["names"], strict=True))
    pairs = [(c, w, n) for c in spec["configs"] for w, n in workflows]
    return [{"id": i, "workflow": w, "name": n, **c} for i, (c, w, n) in enumerate(pairs, 1)]


def select_jobs(jobs, only):
    if not only:
        return jobs
    by_id = {job["id"]: job for job in jobs}
    try:
        ids = [int(part) for part in only.split(",")]
    except ValueError:
        ids = [None]
    if any(i not in by_id for i in ids):
        raise RuntimeError(f"--only takes job ids in 1-{len(jobs)}, got {only!r}")
    return [by_id[i] for i in ids]


def modal_gpu(gpu):
    # H100 may be silently upgraded to H200 unless pinned.
    return "H100!" if gpu == "H100" else gpu


def function_name(job):
    """One Modal function per resource shape, so the dashboard shows the real GPU."""
    gpu = job["gpu"].lower().replace("!", "").replace("+", "plus")
    return f"benchmark-{gpu}-{job['cpu']}cpu-{job['memory_mib']}mib"


def gpu_matches(requested, hardware):
    return requested.split("-")[0].rstrip("!+").upper() in hardware.upper()


def is_api_prompt(data):
    return "nodes" not in data and all(
        isinstance(node, dict) and "class_type" in node for node in data.values()
    )


def file_inputs(prompt, object_info):
    """Map each upload-widget value in the prompt to its filename in the bridge's input folder."""
    found = {}
    for node in prompt.values():
        schema = object_info.get(node.get("class_type"), {}).get("input", {})
        specs = {**schema.get("required", {}), **schema.get("optional", {})}
        for name, value in node.get("inputs", {}).items():
            spec = specs.get(name)
            options = spec[1] if isinstance(spec, list) and len(spec) > 1 else {}
            uploads = isinstance(options, dict) and any(
                key.endswith("_upload") and flag for key, flag in options.items()
            )
            if uploads and isinstance(value, str):
                found[value] = value.removeprefix("./")
    return found


def unknown_nodes(prompt, known):
    """Node types the deployment lacks; the editor emits class_type None for ones it can't load."""
    missing = {n["class_type"] for n in prompt.values() if n.get("class_type")} - set(known)
    unloaded = [f"node {i} (unknown to the editor)" for i, n in prompt.items() if not n.get("class_type")]
    return sorted(missing) + unloaded


def output_name(job, index, remote_name):
    gpu = job["gpu"].rstrip("!+")
    suffix = Path(remote_name).suffix
    return f"{job['id']}_{gpu}_{job['cpu']}cpu_{Path(job['workflow']).stem}_{index}{suffix}"


def format_jobs(jobs):
    return "\n".join(
        f"{j['id']:>3}  {j['gpu']:<10} {j['cpu']}cpu  {','.join(j['flags']) or '-':<30} "
        f"{Path(j['workflow']).name}"
        for j in jobs
    )


def run_finished(event, prompt_id):
    """ComfyUI writes /history only after execution_success; executing node=None follows it."""
    data = event.get("data", {})
    return (
        event.get("type") == "executing"
        and data.get("node") is None
        and data.get("prompt_id") == prompt_id
    )


def output_files(history, root):
    """(filename, path) for each saved output; paths may not escape root."""
    return [
        (
            item["filename"],
            safe_path(root, join_subfolder(item.get("subfolder", ""), item["filename"])),
        )
        for item in iter_output_items(history)
        if item.get("type") == "output"
    ]


def summarize(records, status):
    """Turn probe JSONL records and ComfyUI history status into a result row."""
    seconds = {
        kind: sum(r["seconds"] for r in records if r["kind"] == kind)
        for kind in ("model_file_read", "weight_loading", "sampling", "vae_decode", "vae_encode")
    }
    stamps = {kind: data.get("timestamp") for kind, data in status.get("messages", [])}
    start, end = stamps.get("execution_start"), stamps.get("execution_success")
    seconds["graph_total"] = (end - start) / 1000 if start is not None and end is not None else None
    steps = [r for r in records if r["kind"] == "sampling_step"]
    stages = [r for r in records if r["kind"] == "sampling"]
    peaks = [r["peak_allocated_bytes"] for r in stages if "peak_allocated_bytes" in r]
    errors = [
        data.get("exception_message", "")
        for kind, data in status.get("messages", [])
        if kind == "execution_error"
    ]
    error = "; ".join(errors) or None
    if not error and status.get("status_str") != "success":
        error = status.get("status_str") or "no execution status"
    return {
        "seconds": seconds,
        "node_timings": [r for r in records if r["kind"] == "node_execution"],
        "stage_timings": records,
        "sampling_stages": [
            {
                "seconds": stage["seconds"],
                "steps": [s["seconds"] for s in steps if stage["start"] <= s["start"] <= stage["end"]],
            }
            for stage in stages
        ],
        "peak_vram_bytes": max(peaks) if peaks else None,
        "error": error,
    }


def merge(results):
    """Concatenate results files; each job is tagged with its pass and gets a unique id."""
    passes, merged = [], {"provenance": [], "jobs": []}
    for folder, data in results:
        provenance = data.get("provenance") or {}
        label = (provenance.get("deployment") or {}).get("label")
        name = label if label and label not in passes else folder
        passes.append(name)
        merged["provenance"].append({**provenance, "pass": name})
        merged["jobs"] += [{**job, "pass": name, "id": f"{name}:{job['id']}"} for job in data["jobs"]]
    return merged


def merge_files(paths, out):
    results = [(Path(p).parent.name, json.loads(Path(p).read_text())) for p in paths]
    Path(out).write_text(json.dumps(merge(results), indent=2) + "\n")
