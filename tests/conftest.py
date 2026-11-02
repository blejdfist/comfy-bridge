import copy
import json

import pytest
from workflow_fixtures import api_workflow

from comfyui_modal.bridge import Bridge
from comfyui_modal.config import DEPLOYMENT_ID


@pytest.fixture
def symlink():
    """Create a directory symlink, skipping where the OS refuses (Windows without Developer Mode)."""

    def create(link, target):
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"cannot create symlinks here: {exc}")

    return create


class FakeBackend:
    def __init__(self):
        self.submissions = []
        self.results = {}
        self.snapshots = {}
        self.calls = {}
        self.cancelled = []
        self.download_error = False
        self.submit_error = False
        self.model_submissions = []
        self.model_calls = {}
        self.model_cancelled = []

    async def submit_model_download(self, download_id, request):
        self.model_submissions.append((download_id, copy.deepcopy(request)))
        self.model_calls[download_id] = f"model-call-{download_id}"
        if self.submit_error:
            raise ConnectionError("Lost download submission response")
        return self.model_calls[download_id]

    async def recover_model_download(self, download_id):
        return self.model_calls.get(download_id)

    async def model_download_progress(self, download_id):
        return {}

    async def cancel_model_download(self, download_id):
        self.model_cancelled.append(download_id)

    async def submit(self, job_id, payload, input_dir):
        self.submissions.append((job_id, copy.deepcopy(payload), input_dir))
        self.calls[job_id] = f"call-{job_id}"
        if isinstance(self.submit_error, Exception):
            raise self.submit_error
        if self.submit_error:
            raise ConnectionError("Lost response after remote acceptance")
        return self.calls[job_id]

    async def recover_call(self, job_id):
        return self.calls.get(job_id)

    async def poll(self, call_id):
        return copy.deepcopy(self.results.get(call_id))

    async def events(self, job_id):
        return self.snapshots.get(job_id, {})

    async def cancel(self, job_id):
        self.cancelled.append(job_id)

    async def download(self, artifacts, root):
        if self.download_error:
            raise ConnectionError("Download interrupted")
        for artifact in artifacts:
            target = root / artifact["local_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"test-image")


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "cache").mkdir()
    schema = {
        node["class_type"]: {"input": {"required": {}}} for node in api_workflow("test").values()
    }
    schema["LoadImage"] = {"input": {"required": {"image": [["remote-only.png"], {}]}}}
    snapshot = {
        "deployment_id": DEPLOYMENT_ID,
        "assets": {},
        "routes": {
            "/object_info": schema,
            "/extensions": [],
            "/features": {},
            "/models": ["diffusion_models"],
            "/models/diffusion_models": ["krea2_turbo_int8_convrot.safetensors"],
        },
    }
    (tmp_path / "cache/snapshot.json").write_text(json.dumps(snapshot))
    return tmp_path


@pytest.fixture
def bridge(data_dir, backend):
    workflows = data_dir / "workflows"
    workflows.mkdir()
    instance = Bridge(data_dir, backend, output_dir=data_dir / "generated", workflows_dir=workflows)
    yield instance
    # HTTP app lifecycle may already have closed it.
    try:
        instance.store.close()
    except Exception:
        pass


def result_for(job_id):
    relative = f"{job_id}/output/{job_id}/Krea2_00001_.png"
    return {
        "error": None,
        "artifacts": [{"local_path": relative, "remote_path": relative}],
        "history": {
            "prompt": [],
            "outputs": {
                "29": {
                    "images": [
                        {
                            "filename": "Krea2_00001_.png",
                            "subfolder": relative.rsplit("/", 1)[0],
                            "type": "output",
                        }
                    ]
                }
            },
            "status": {"status_str": "success", "completed": True, "messages": []},
        },
    }
