"""Modal SDK access. No public GPU endpoint or browser credentials are required."""

import asyncio
from pathlib import Path

import modal

from .deployment_config import modal_names
from .helpers.modal_volume import list_volume_entries, retry_rate_limited
from .helpers.paths import safe_path


class SubmissionRejected(Exception):
    """The backend definitively refused the job, so no remote call exists to adopt."""


class ModalBackend:
    def __init__(self):
        names = modal_names()
        self.app_name = names["app_name"]
        self.worker = modal.Cls.from_name(self.app_name, "GPUWorker")()
        self.jobs = modal.Volume.from_name(names["job_volume"])
        self.state = modal.Dict.from_name(names["state_dict"])

    async def refresh_models(self, *, full=False):
        return await modal.Function.from_name(self.app_name, "refresh_models").remote.aio(full=full)

    async def submit_model_download(self, download_id, request):
        function = modal.Function.from_name(self.app_name, "install_model")
        if request.get("secret"):
            function = await asyncio.to_thread(
                function.with_options, secrets=[modal.Secret.from_name(request["secret"])]
            )
        request = {key: value for key, value in request.items() if key != "secret"}
        call = await function.spawn.aio(download_id, request)
        return call.object_id

    async def model_download_progress(self, download_id):
        return await self.state.get.aio(f"download:progress:{download_id}", {})

    async def recover_model_download(self, download_id):
        return await self.state.get.aio(f"download:call:{download_id}")

    async def cancel_model_download(self, download_id):
        await self.state.put.aio(f"download:cancel:{download_id}", True)

    async def submit(self, job_id: str, payload: dict, input_dir: Path):
        files = [path for path in input_dir.rglob("*") if path.is_file()]
        if files:
            async with self.jobs.batch_upload.aio(force=True) as upload:
                for path in files:
                    upload.put_file(str(path), f"/{job_id}/input/{path.relative_to(input_dir)}")
        try:
            call = await self.worker.run.spawn.aio(job_id, payload)
        except (
            modal.exception.AuthError,
            modal.exception.ConflictError,
            modal.exception.InvalidError,
            modal.exception.NotFoundError,
            modal.exception.PermissionDeniedError,
            modal.exception.ResourceExhaustedError,
        ) as exc:
            # Modal refused the request (e.g. spend limit), so no call was created.
            raise SubmissionRejected(f"Modal rejected the job: {exc}") from exc
        return call.object_id

    async def recover_call(self, job_id):
        return await self.state.get.aio(f"call:{job_id}")

    async def poll(self, call_id):
        try:
            return await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
        except (
            modal.exception.FunctionTimeoutError,
            modal.exception.OutputExpiredError,
            modal.exception.RemoteError,
            modal.exception.ExecutionError,
        ) as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "history": None, "artifacts": []}
        except (TimeoutError, modal.exception.TimeoutError):
            return None

    async def events(self, job_id):
        return await self.state.get.aio(job_id, {})

    async def cancel(self, job_id):
        await self.state.put.aio(f"cancel:{job_id}", True)
        # Cancellation is handled inside the active invocation and forwarded to ComfyUI.
        # This works without trying to route a second GPU call to a busy container.

    async def download(self, artifacts, root: Path):
        for artifact in artifacts:
            target = safe_path(root, artifact["local_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            try:
                with temporary.open("wb") as output:
                    async for chunk in self.jobs.read_file.aio(artifact["remote_path"]):
                        await asyncio.to_thread(output.write, chunk)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

    async def cleanup_jobs_volume(self, *, dry_run=False):
        from modal.volume import FileEntryType

        entries = await list_volume_entries(self.jobs)
        if entries is None:
            return {"files": 0, "bytes": 0}
        files = [entry for entry in entries if entry.type == FileEntryType.FILE]
        if not dry_run and entries:
            # Modal rejects removing '/'. Derive all top-level entries from the
            # single recursive listing and clear each subtree in bounded parallel.
            roots = {entry.path.lstrip("/").split("/", 1)[0] for entry in entries}
            if any(name in {"", ".", ".."} or "\\" in name for name in roots):
                raise RuntimeError("Unexpected entry in the job Volume")
            semaphore = asyncio.Semaphore(3)

            async def remove(name):
                async with semaphore:
                    try:
                        await retry_rate_limited(
                            self.jobs.remove_file.aio, f"/{name}", recursive=True
                        )
                    except (FileNotFoundError, modal.exception.NotFoundError):
                        pass

            await asyncio.gather(*(remove(name) for name in sorted(roots)))
        return {"files": len(files), "bytes": sum(entry.size for entry in files)}
