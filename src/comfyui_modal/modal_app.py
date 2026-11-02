"""Deploy with: uv run modal deploy src/comfyui_modal/modal_app.py."""

import json
import os

import modal

from comfyui_modal.config import COMFY_COMMIT, MODEL_DOWNLOAD_CONCURRENCY
from comfyui_modal.custom_nodes import install_commands, load_custom_nodes
from comfyui_modal.deployment_config import load_deployment_config, modal_names

# Read public YAML only on the deployment machine. Carry the resolved value in
# the image environment so remote imports do not need local configuration files.
resources = (
    load_deployment_config(os.environ.get("COMFY_BRIDGE_CONFIG", "config.yaml"))
    if modal.is_local()
    else json.loads(os.environ["COMFY_BRIDGE_RESOURCES"])
)
gpu_type = resources["gpu_worker"]["gpu"]
names = modal_names(resources)

app = modal.App(names["app_name"])
models = modal.Volume.from_name(names["model_volume"], create_if_missing=True)
jobs = modal.Volume.from_name(names["job_volume"], create_if_missing=True)
state = modal.Dict.from_name(names["state_dict"], create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .run_commands(
        "git init /opt/ComfyUI",
        "git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git",
        f"git -C /opt/ComfyUI fetch --depth 1 origin {COMFY_COMMIT}",
        "git -C /opt/ComfyUI checkout FETCH_HEAD",
    )
    .uv_pip_install("huggingface-hub==2.3.0", extra_options="-r /opt/ComfyUI/requirements.txt")
    .env({"COMFY_BRIDGE_RESOURCES": json.dumps(resources)})
)

custom_nodes = []
if modal.is_local():
    manifest = os.path.join(
        os.path.dirname(os.environ.get("COMFY_BRIDGE_CONFIG", "config.yaml")), "custom_nodes.yaml"
    )
    custom_nodes = load_custom_nodes(manifest)
    for node in custom_nodes:
        image = image.run_commands(*install_commands(node))
        requirements = f"/opt/ComfyUI/custom_nodes/{node['name']}/requirements.txt"
        image = image.run_commands(
            f"if test -f {requirements}; then uv pip install --system -r {requirements}; fi"
        )
image = image.env({"COMFY_BRIDGE_CUSTOM_NODES": json.dumps(custom_nodes)}).add_local_python_source(
    "comfyui_modal"
)


async def export_editor_snapshot():
    from comfyui_modal.runtime import ComfyRuntime

    runtime = ComfyRuntime(cpu=True)
    try:
        await runtime.start()
        return await runtime.snapshot()
    finally:
        await runtime.stop()


CPU_FUNCTION = dict(
    image=image,
    volumes={"/models": models},
    cpu=resources["cpu_only"]["cpus"],
    memory=resources["cpu_only"]["memory_mib"],
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)


@app.function(**{**CPU_FUNCTION, "max_containers": MODEL_DOWNLOAD_CONCURRENCY}, timeout=3600)
async def install_model(download_id: str, request: dict):
    import uuid
    from pathlib import Path

    from comfyui_modal.model_downloads import CREDENTIAL_KEYS, ModelDownloadError, download_model

    await state.put.aio(f"download:call:{download_id}", modal.current_function_call_id())

    async def report(progress):
        if await state.get.aio(f"download:cancel:{download_id}", False):
            raise ModelDownloadError("Download cancelled")
        await state.put.aio(f"download:progress:{download_id}", progress)

    receipt = None
    try:
        await models.reload.aio()
        credentials = {key: os.environ.get(key) for key in CREDENTIAL_KEYS}
        receipt = await download_model(request, Path("/models"), report, credentials)
        await report(
            {"phase": "committing", "bytes": receipt["bytes"], "filename": receipt["filename"]}
        )
        if not receipt.get("skipped"):
            await models.commit.aio()
            await state.put.aio("models:revision", str(uuid.uuid4()))
        if not request.get("refresh_catalog", True):
            return {"error": None, "receipt": receipt, "snapshot": None}
        await report(
            {"phase": "refreshing", "bytes": receipt["bytes"], "filename": receipt["filename"]}
        )
        # Include models committed by parallel downloads since this container started.
        # ponytail: two snapshots can still land out of order; `sync` refreshes once at the end.
        await models.reload.aio()
        snapshot = await export_editor_snapshot()
        return {"error": None, "receipt": receipt, "snapshot": snapshot}
    except ModelDownloadError as exc:
        return {"error": str(exc), "receipt": receipt}
    except Exception as exc:
        return {
            "error": f"Model download or catalog refresh failed ({type(exc).__name__}). "
            "Check access to the file and retry the catalog refresh if the file was installed.",
            "receipt": receipt,
        }


@app.function(**CPU_FUNCTION, timeout=600)
async def refresh_models(full: bool = False):
    await models.reload.aio()
    if full:
        return await export_editor_snapshot()
    from comfyui_modal.runtime import scan_model_catalog

    return await scan_model_catalog()


@app.cls(
    image=image,
    gpu=gpu_type,
    cpu=(resources["gpu_worker"]["cpus"], resources["gpu_worker"]["cpus"]),
    memory=resources["gpu_worker"]["memory_mib"],
    volumes={"/models": models, "/jobs": jobs},
    min_containers=0,
    max_containers=1,
    scaledown_window=resources["gpu_worker"]["scaledown_window_seconds"],
    timeout=1800,
)
class GPUWorker:
    @modal.enter()
    async def start(self):
        from comfyui_modal.runtime import ComfyRuntime

        self.catalog_revision = await state.get.aio("models:revision")
        await models.reload.aio()
        self.runtime = ComfyRuntime()
        await self.runtime.start()

    @modal.method()
    async def run(self, job_id: str, payload: dict):
        import traceback

        await state.put.aio(f"call:{job_id}", modal.current_function_call_id())
        try:
            revision = await state.get.aio("models:revision")
            if revision != self.catalog_revision:
                # ComfyUI may retain mmap handles to weight files. Stop it before reloading
                # the Volume, then rebuild its filename caches on the next actual GPU job.
                await self.runtime.reload_models(models)
                self.catalog_revision = revision
            return await self.runtime.execute(job_id, payload, state, jobs)
        except Exception:
            return {"history": None, "artifacts": [], "error": traceback.format_exc()}

    @modal.exit()
    async def stop(self):
        if hasattr(self, "runtime"):
            await self.runtime.stop()
