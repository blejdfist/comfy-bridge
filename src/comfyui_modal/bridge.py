"""Durable local scheduling and reconciliation, independent of the HTTP layer."""

import asyncio
import copy
import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from .backend import SubmissionRejected
from .config import MODEL_DOWNLOAD_CONCURRENCY
from .helpers.aio import cancel_and_wait
from .helpers.deployment import is_current_deployment, require_current_deployment
from .helpers.files import atomic_write_json
from .helpers.paths import safe_path
from .model_catalog import apply_model_lists
from .model_downloads import normalize_request
from .store import ACTIVE, TERMINAL, Store, execution_failure
from .worker_io import localize_artifacts
from .workflow import LOAD_IMAGE_NODES, validate_prompt

log = logging.getLogger(__name__)


class Bridge:
    def __init__(
        self,
        data_dir: Path,
        backend,
        output_dir: Path = Path("output"),
        workflows_dir: Path = Path("workflows"),
    ):
        self.root = data_dir.resolve()
        self.output_root = output_dir.resolve()
        self.workflows_root = workflows_dir.resolve()
        self.workflows_root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        for folder in ("input", "userdata/workflows", "jobs", "cache"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self.store = Store(self.root / "queue.sqlite3")
        self.sockets = {}
        self.task = None
        self.model_task = None
        self.last_error = None
        self.preview_seq = {}
        (self.root / "userdata/user.css").touch(exist_ok=True)
        snapshot = self.root / "cache/snapshot.json"
        self.snapshot = json.loads(snapshot.read_text()) if snapshot.exists() else None

    def save_snapshot(self, snapshot):
        require_current_deployment(snapshot)
        target = self.root / "cache/snapshot.json"
        atomic_write_json(target, snapshot)
        self.snapshot = snapshot

    async def refresh_models(self, *, full=False):
        full = full or not is_current_deployment(self.snapshot)
        catalog = await self.backend.refresh_models(full=bool(full))
        if not full and catalog.get("custom_nodes") != self.snapshot.get("custom_nodes"):
            catalog = await self.backend.refresh_models(full=True)
            full = True
        self.save_snapshot(catalog if full else apply_model_lists(self.snapshot, catalog))
        return "full" if full else "models"

    def enqueue_model_download(self, body):
        request = normalize_request(body)
        # Reattach to an unfinished download of the same file instead of queueing a duplicate.
        for item in self.store.model_downloads():
            if item["state"] not in TERMINAL and all(
                item["request"].get(key) == request[key] for key in ("url", "folder", "filename")
            ):
                return item["id"]
        download_id = str(uuid.uuid4())
        self.store.add_model_download(download_id, request)
        return download_id

    async def cancel_model_download(self, download_id):
        item = self.store.model_download(download_id)
        if not item or item["state"] in TERMINAL:
            return False
        if item["state"] != "pending":
            await self.backend.cancel_model_download(download_id)
        self.store.update_model_download(
            download_id, state="cancelling" if item["call_id"] else "cancelled"
        )
        return True

    async def download_tick(self):
        downloads = self.store.model_downloads()
        active = [item for item in downloads if item["state"] not in TERMINAL | {"pending"}]
        pending = [item for item in downloads if item["state"] == "pending"]
        active += pending[: max(0, MODEL_DOWNLOAD_CONCURRENCY - len(active))]
        for item in active:
            try:
                await self.advance_download(item)
            except Exception:
                # Keep one failing reconciliation from stalling the other downloads.
                log.warning("Model download %s reconciliation will retry", item["id"])

    async def advance_download(self, active):
        if active["state"] == "pending":
            self.store.update_model_download(active["id"], state="submitting")
            try:
                call_id = await self.backend.submit_model_download(active["id"], active["request"])
            except Exception:
                # Don't store a provider exception containing signed URLs or credential details.
                self.store.update_model_download(
                    active["id"],
                    state="unknown",
                    error="Submission outcome unknown. Waiting to recover the call; check the Modal Secret name.",
                )
                return
            self.store.update_model_download(active["id"], call_id=call_id, state="running")
            active = {**active, "call_id": call_id}
        if not active["call_id"]:
            call_id = await self.backend.recover_model_download(active["id"])
            if not call_id:
                return
            self.store.update_model_download(
                active["id"], call_id=call_id, state="running", error=None
            )
            active["call_id"] = call_id
        progress = await self.backend.model_download_progress(active["id"])
        if progress:
            self.store.update_model_download(active["id"], progress=progress)
        result = await self.backend.poll(active["call_id"])
        if result is None:
            return
        if result.get("error"):
            self.store.update_model_download(
                active["id"],
                state="cancelled" if active["state"] == "cancelling" else "failed",
                error=str(result["error"]),
                result=result.get("receipt"),
            )
            return
        if active["request"].get("refresh_catalog", True):
            try:
                self.save_snapshot(result["snapshot"])
            except ValueError:
                self.store.update_model_download(
                    active["id"],
                    state="failed",
                    result=result["receipt"],
                    error="Model installed, but local and remote runtime versions differ. "
                    "Restart the bridge with the current code and run "
                    "`comfy-bridge sync --full` to rebuild editor metadata.",
                )
                return
        self.store.update_model_download(
            active["id"], state="completed", result=result["receipt"], error=None
        )

    async def run_downloads(self):
        while True:
            try:
                await self.download_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("Model download reconciliation will retry")
                await asyncio.sleep(3)
            await asyncio.sleep(1)

    async def emit(self, event, client_id=None):
        for sid, sockets in list(self.sockets.items()):
            if client_id and sid != client_id:
                continue
            for socket in list(sockets):
                try:
                    if isinstance(event, bytes):
                        await socket.send_bytes(event)
                    else:
                        await socket.send_json(event)
                except (ConnectionError, RuntimeError):
                    sockets.discard(socket)

    def status(self):
        return {
            "exec_info": {
                "queue_remaining": sum(j["state"] not in TERMINAL for j in self.store.all())
            }
        }

    async def emit_status(self):
        await self.emit({"type": "status", "data": {"status": self.status()}})

    def enqueue(self, payload):
        if not is_current_deployment(self.snapshot):
            raise RuntimeError(
                "Run `uv run comfy-bridge sync` to install models and refresh metadata"
            )
        errors = validate_prompt(payload.get("prompt"), self.snapshot["routes"]["/object_info"])
        if errors:
            return None, errors
        job_id = (
            str(uuid.UUID(payload["prompt_id"])) if payload.get("prompt_id") else str(uuid.uuid4())
        )
        existing = self.store.get(job_id)
        if existing:
            if existing["payload"].get("prompt") != payload["prompt"]:
                raise ValueError("prompt_id already belongs to a different graph")
            return existing, {}
        payload = copy.deepcopy(payload)
        payload.setdefault("extra_data", {})["create_time"] = int(time.time() * 1000)
        input_root = self.root / "input"
        staged = self.root / "jobs" / job_id / "input"
        staged.mkdir(parents=True, exist_ok=True)
        for node in payload["prompt"].values():
            for name, value in node["inputs"].items():
                if not isinstance(value, str):
                    continue
                relative = value.removesuffix(" [input]")
                if "\n" in relative or any(len(part) > 255 for part in relative.split("/")):
                    continue  # Text prompts are not filesystem paths.
                try:
                    source = safe_path(input_root, relative)
                except ValueError:
                    if node["class_type"] in LOAD_IMAGE_NODES and name == "image":
                        raise ValueError("Upload input images through the local editor") from None
                    continue
                if source.is_file():
                    destination = safe_path(staged, relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                elif node["class_type"] in LOAD_IMAGE_NODES and name == "image":
                    raise ValueError(f"Input image is missing locally: {value}")
        print(f"Queued job {job_id}", flush=True)
        return self.store.add(job_id, payload, payload.get("front", False)), {}

    async def cancel(self, job_id):
        job = self.store.get(job_id)
        if not job or job["state"] in TERMINAL:
            return False
        if job["state"] == "pending":
            await self.finish(job, "cancelled", "Cancelled before submission")
        else:
            await self.backend.cancel(job_id)
            if not job["call_id"]:
                # The worker checks this cancellation flag before starting ComfyUI execution,
                # even if an ambiguous submission eventually arrives after local cancellation.
                await self.finish(
                    job, "cancelled", "Cancelled while submission outcome was unknown"
                )
            else:
                self.store.update(job_id, state="cancelling")
        await self.emit_status()
        return True

    async def finish(self, job, state, error=None, history=None):
        if history is None:
            history = {
                "prompt": self.store.queue_item(job),
                "outputs": {},
                "status": {
                    "status_str": "success" if state == "completed" else "error",
                    "completed": True,
                    "messages": [],
                },
            }
        # Keep local prompt ids, workflow metadata and ordering in history.
        history["prompt"] = self.store.queue_item(job)
        self.store.update(job["id"], state=state, history=history, error=error)
        detail = (execution_failure(history) or {}).get("exception_message") or error
        print(f"Job {job['id']} {state}" + (f": {detail}" if detail else ""), flush=True)
        sid = job["payload"].get("client_id")
        if state == "completed":
            for node_id, output in history.get("outputs", {}).items():
                await self.emit(
                    {
                        "type": "executed",
                        "data": {"prompt_id": job["id"], "node": node_id, "output": output},
                    },
                    sid,
                )
        event_type = {
            "completed": "execution_success",
            "cancelled": "execution_interrupted",
            "failed": "execution_error",
        }[state]
        await self.emit(
            {
                "type": event_type,
                "data": (execution_failure(history) if state == "failed" else None)
                or {
                    "prompt_id": job["id"],
                    "node_id": "",
                    "exception_message": error or "",
                    "exception_type": "RemoteExecution",
                    "traceback": [],
                    "executed": [],
                    "current_inputs": {},
                    "current_outputs": {},
                    "timestamp": int(time.time() * 1000),
                },
            },
            sid,
        )
        await self.emit({"type": "executing", "data": {"prompt_id": job["id"], "node": None}}, sid)
        await self.emit_status()

    async def tick(self):
        jobs = self.store.all()
        active = next((j for j in jobs if j["state"] in ACTIVE), None)
        if active is None:
            pending = sorted(
                (j for j in jobs if j["state"] == "pending"), key=lambda j: j["priority"]
            )
            if not pending:
                return
            active = pending[0]
            self.store.update(active["id"], state="submitting")
            await self.emit_status()
            try:
                call_id = await self.backend.submit(
                    active["id"], active["payload"], self.root / "jobs" / active["id"] / "input"
                )
            except SubmissionRejected as exc:
                await self.finish(active, "failed", str(exc))
                return
            except Exception as exc:
                # A network error may happen after Modal accepted the call. Never blindly resubmit.
                self.store.update(active["id"], state="unknown", error=str(exc))
                raise
            self.store.update(active["id"], call_id=call_id, state="running", error=None)
            print(f"Job {active['id']} started", flush=True)
            active = self.store.get(active["id"])
        if not active["call_id"]:
            call_id = await self.backend.recover_call(active["id"])
            if not call_id:
                error = (
                    f"Submission outcome unknown for {active['id']}. "
                    "Waiting to adopt the remote call; cancel this job to unblock queue."
                )
                if self.last_error != error:
                    print(error, flush=True)
                self.last_error = error
                return
            self.store.update(
                active["id"],
                call_id=call_id,
                state="cancelling" if active["state"] == "cancelling" else "running",
            )
            active = self.store.get(active["id"])
        if active["state"] == "cancelling":
            await self.backend.cancel(active["id"])
        snapshot = await self.backend.events(active["id"])
        sid = active["payload"].get("client_id")
        for entry in snapshot.get("events", []):
            if entry["seq"] <= active["event_seq"]:
                continue
            event = entry["event"]
            # Only advertise outputs/completion after the corresponding files are local.
            if event["type"] not in {
                "executed",
                "execution_success",
                "execution_error",
                "execution_interrupted",
            } and not (event["type"] == "executing" and event["data"].get("node") is None):
                await self.emit(event, sid)
            self.store.update(active["id"], event_seq=entry["seq"])
        preview = snapshot.get("preview")
        if preview and preview["seq"] != self.preview_seq.get(active["id"]):
            await self.emit(preview["data"], sid)
            self.preview_seq[active["id"]] = preview["seq"]
        result = active["result"] or await self.backend.poll(active["call_id"])
        if result is None:
            return
        if result.get("error"):
            await self.finish(active, "failed", json.dumps(result["error"]))
            return
        if result.get("local_output_layout") != 1:
            reserved = {
                artifact["local_path"]
                for job in self.store.all()
                if job["id"] != active["id"]
                and (job.get("result") or {}).get("local_output_layout") == 1
                for artifact in job["result"]["artifacts"]
            }
            result = localize_artifacts(result, self.output_root, reserved)
        self.store.update(active["id"], state="downloading", result=result)
        await self.backend.download(result["artifacts"], self.output_root)
        history = result["history"]
        status = history.get("status", {})
        messages = status.get("messages", [])
        cancelled = any(message[0] == "execution_interrupted" for message in messages)
        state = (
            "cancelled"
            if cancelled
            else ("completed" if status.get("status_str") == "success" else "failed")
        )
        failure = execution_failure(history)
        error = (
            None
            if state == "completed"
            else (failure.get("exception_message") if failure else json.dumps(status))
        )
        await self.finish(active, state, error, history)
        self.last_error = None

    async def run(self):
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                log.exception("Queue reconciliation failed; will retry without resubmitting")
                await asyncio.sleep(3)
            await asyncio.sleep(0.5)

    async def start(self):
        self.task = asyncio.create_task(self.run())
        self.model_task = asyncio.create_task(self.run_downloads())

    async def close(self):
        if self.model_task:
            await cancel_and_wait(self.model_task)
        if self.task:
            await cancel_and_wait(self.task)
        for sockets in self.sockets.values():
            for socket in list(sockets):
                await socket.close()
        self.store.close()
