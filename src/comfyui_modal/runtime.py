"""ComfyUI process and execution adapter. Imported on Modal, never loads Torch locally."""

import asyncio
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

import aiohttp

from .config import COMFY_COMMIT, DEPLOYMENT_ID
from .helpers.files import atomic_write_json
from .workflow import job_prompt

COMFY_ROOT = Path("/opt/ComfyUI")
BASE_URL = "http://127.0.0.1:8188"


async def scan_model_catalog():
    """Import ComfyUI's filename resolver without importing Torch or starting its server."""
    script = """
import json, os
from comfy.cli_args import args
args.models_directory = '/models'
import folder_paths
folders = {name: folder_paths.get_filename_list(name) for name in folder_paths.folder_names_and_paths}
print(json.dumps({'model_lists': folders, 'embeddings': [os.path.splitext(name)[0] for name in folders.get('embeddings', [])]}))
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        cwd=COMFY_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=60)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError("Model filename scan failed; try sync --full")
    result = json.loads(stdout)
    result["deployment_id"] = DEPLOYMENT_ID
    result["custom_nodes"] = json.loads(os.environ.get("COMFY_BRIDGE_CUSTOM_NODES", "[]"))
    return result


class ComfyRuntime:
    def __init__(self, cpu: bool = False):
        self.cpu = cpu
        self.process = None

    async def start(self):
        for directory in ("/models", "/jobs", "/tmp/comfy-input", "/tmp/comfy-output"):
            Path(directory).mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
            "--disable-auto-launch",
            "--offline",
            "--models-directory",
            "/models",
            "--input-directory",
            "/tmp/comfy-input",
            "--output-directory",
            "/tmp/comfy-output",
            "--preview-method",
            "latent2rgb",
        ]
        if self.cpu:
            command.append("--cpu")
        self.process = await asyncio.create_subprocess_exec(*command, cwd=COMFY_ROOT)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            for _ in range(240):
                if self.process.returncode is not None:
                    raise RuntimeError(f"ComfyUI exited at startup: {self.process.returncode}")
                try:
                    async with session.get(f"{BASE_URL}/system_stats") as response:
                        if response.status == 200:
                            return
                except (aiohttp.ClientError, TimeoutError):
                    pass
                await asyncio.sleep(1)
        raise TimeoutError("ComfyUI did not start within four minutes")

    async def stop(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def reload_models(self, models_volume):
        """Release ComfyUI's mmap handles before changing a live model Volume view."""
        await self.stop()
        await models_volume.reload.aio()
        await self.start()

    async def snapshot(self) -> dict:
        async with aiohttp.ClientSession() as session:
            result = {
                "deployment_id": DEPLOYMENT_ID,
                "comfy_commit": COMFY_COMMIT,
                "created_at": time.time(),
                "routes": {},
                "assets": {},
                "custom_nodes": json.loads(os.environ.get("COMFY_BRIDGE_CUSTOM_NODES", "[]")),
            }
            paths = [
                "/object_info",
                "/models",
                "/embeddings",
                "/features",
                "/system_stats",
                "/extensions",
                "/workflow_templates",
                "/node_replacements",
            ]
            for path in paths:
                async with session.get(BASE_URL + path) as response:
                    if response.status == 200:
                        result["routes"][path] = await response.json()
            for folder in result["routes"].get("/models", []):
                path = f"/models/{folder}"
                async with session.get(BASE_URL + path) as response:
                    if response.status == 200:
                        result["routes"][path] = await response.json()
            for path in result["routes"].get("/extensions", []):
                async with session.get(BASE_URL + path) as response:
                    response.raise_for_status()
                    result["assets"][path] = await response.text()
            # The bridge implements traditional queue/history and the modern jobs API.
            # Disable optional worker capabilities it does not implement locally.
            result["routes"]["/features"] = {}
            return result

    async def execute(self, job_id: str, payload: dict, state, jobs_volume) -> dict:
        from .worker_io import collect_artifacts

        job_dir = Path("/jobs") / job_id
        manifest = job_dir / "result.json"
        await jobs_volume.reload.aio()
        if manifest.exists():
            return json.loads(manifest.read_text())
        if await state.get.aio(f"cancel:{job_id}", False):
            return {
                "error": None,
                "artifacts": [],
                "history": {
                    "outputs": {},
                    "status": {
                        "status_str": "error",
                        "completed": True,
                        "messages": [["execution_interrupted", {"prompt_id": job_id}]],
                    },
                },
            }
        input_dir = job_dir / "input"
        if input_dir.exists():
            shutil.copytree(input_dir, Path("/tmp/comfy-input"), dirs_exist_ok=True)
        payload = job_prompt(payload, job_id)
        events, sequence, last_publish = [], 0, 0.0
        preview = None

        async def publish(event=None, force=False):
            nonlocal sequence, last_publish
            if event:
                sequence += 1
                events.append({"seq": sequence, "event": event})
                del events[:-128]
            now = time.monotonic()
            if force or now - last_publish >= 0.35:
                await state.put.aio(
                    job_id, {"events": events, "preview": preview, "updated_at": time.time()}
                )
                last_publish = now

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.ws_connect(
                f"{BASE_URL}/ws?clientId={job_id}", heartbeat=20, max_msg_size=8 * 1024 * 1024
            ) as ws:
                async with session.post(f"{BASE_URL}/prompt", json=payload) as response:
                    submitted = await response.json()
                    if response.status != 200:
                        return {"error": submitted, "history": None, "artifacts": []}
                remote_id = submitted["prompt_id"]
                finished = False
                while not finished:
                    try:
                        message = await asyncio.wait_for(ws.receive(), timeout=1)
                    except TimeoutError:
                        message = None
                    control = await state.get.aio(f"cancel:{job_id}", False)
                    if control:
                        async with session.post(
                            f"{BASE_URL}/interrupt", json={"prompt_id": remote_id}
                        ) as response:
                            response.raise_for_status()
                    if message and message.type == aiohttp.WSMsgType.TEXT:
                        event = message.json()
                        if event.get("data", {}).get("prompt_id") == remote_id:
                            event["data"]["prompt_id"] = job_id
                            await publish(event)
                    elif message and message.type == aiohttp.WSMsgType.BINARY:
                        # Keep just the latest preview; do not grow a persistent image event log.
                        if len(message.data) <= 1024 * 1024:
                            preview = {"seq": sequence + 1, "data": message.data}
                    elif message and message.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        raise ConnectionError("ComfyUI execution WebSocket disconnected")
                    async with session.get(f"{BASE_URL}/history/{remote_id}") as response:
                        history = (await response.json()).get(remote_id)
                    finished = history is not None
                    await publish(force=finished)
        history = copy.deepcopy(history)
        artifacts = collect_artifacts(history, job_dir)
        result = {"history": history, "artifacts": artifacts, "error": None}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest, result)
        await jobs_volume.commit.aio()
        return result
