import uuid

import pytest
from conftest import result_for
from workflow_fixtures import api_workflow

from comfyui_modal.backend import SubmissionRejected
from comfyui_modal.bridge import Bridge
from comfyui_modal.helpers.paths import safe_path
from comfyui_modal.store import public_job
from comfyui_modal.workflow import job_prompt


def payload(**extra):
    return {"prompt": api_workflow("a red cabin"), "client_id": "browser", **extra}


def test_default_outputs_follow_working_directory_not_data_directory(
    data_dir, backend, tmp_path, monkeypatch
):
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    instance = Bridge(data_dir, backend)
    try:
        assert instance.output_root == launch_dir / "output"
    finally:
        instance.store.close()


async def test_execution_is_serial_and_history_only_follows_download(bridge, backend):
    first, _ = bridge.enqueue(payload())
    second, _ = bridge.enqueue(payload())
    await bridge.tick()
    await bridge.tick()
    assert len(backend.submissions) == 1
    assert bridge.store.get(second["id"])["state"] == "pending"
    backend.results[f"call-{first['id']}"] = result_for(first["id"])
    backend.download_error = True
    with pytest.raises(ConnectionError):
        await bridge.tick()
    assert bridge.store.get(first["id"])["state"] == "downloading"
    assert not bridge.store.history()
    backend.download_error = False
    await bridge.tick()
    assert bridge.store.get(first["id"])["state"] == "completed"
    history = bridge.store.history()[first["id"]]
    assert history["prompt"][1] == first["id"]
    artifact = "Krea2_00001_.png"
    assert (bridge.output_root / artifact).read_bytes() == b"test-image"
    await bridge.tick()
    assert len(backend.submissions) == 2


async def test_flat_outputs_increment_names_and_keep_retry_mapping(bridge, backend):
    bridge.output_root.mkdir(parents=True)
    (bridge.output_root / "Krea2_00001_.png").write_bytes(b"earlier-image")
    first, _ = bridge.enqueue(payload())
    await bridge.tick()
    backend.results[f"call-{first['id']}"] = result_for(first["id"])
    backend.download_error = True
    with pytest.raises(ConnectionError):
        await bridge.tick()
    saved = bridge.store.get(first["id"])["result"]
    assert saved["artifacts"][0]["local_path"] == "Krea2_00002_.png"
    backend.download_error = False
    await bridge.tick()
    item = bridge.store.history()[first["id"]]["outputs"]["29"]["images"][0]
    assert item["filename"] == "Krea2_00002_.png"
    assert item["subfolder"] == ""
    second, _ = bridge.enqueue(payload())
    await bridge.tick()
    backend.results[f"call-{second['id']}"] = result_for(second["id"])
    await bridge.tick()
    assert (bridge.output_root / "Krea2_00001_.png").read_bytes() == b"earlier-image"
    assert (bridge.output_root / "Krea2_00002_.png").read_bytes() == b"test-image"
    assert (bridge.output_root / "Krea2_00003_.png").read_bytes() == b"test-image"
    assert all(path.is_file() for path in bridge.output_root.iterdir())


async def test_restart_adopts_call_without_duplicate_submission(bridge, backend, data_dir):
    job, _ = bridge.enqueue(payload())
    await bridge.tick()
    bridge.store.close()
    restarted = Bridge(
        data_dir, backend, output_dir=bridge.output_root, workflows_dir=bridge.workflows_root
    )
    try:
        backend.results[f"call-{job['id']}"] = result_for(job["id"])
        await restarted.tick()
        assert len(backend.submissions) == 1
        assert restarted.store.get(job["id"])["state"] == "completed"
    finally:
        restarted.store.close()


async def test_rejected_submission_fails_job_and_unblocks_queue(bridge, backend):
    job, _ = bridge.enqueue(payload())
    second, _ = bridge.enqueue(payload())
    backend.submit_error = SubmissionRejected("spend limit reached")
    await bridge.tick()
    assert bridge.store.get(job["id"])["state"] == "failed"
    assert bridge.store.get(job["id"])["error"] == "spend limit reached"
    backend.submit_error = False
    await bridge.tick()
    assert backend.submissions[-1][0] == second["id"]
    assert bridge.store.get(second["id"])["state"] == "running"
    # Fields the editor's zod schema requires; a missing one empties its whole queue panel.
    entry = public_job(bridge.store.get(job["id"]))
    assert "workflow_id" not in entry
    assert entry["execution_error"].keys() >= {
        "current_inputs",
        "current_outputs",
        "exception_message",
        "exception_type",
        "node_id",
        "node_type",
        "traceback",
    }


async def test_ambiguous_submission_recovers_call_without_resubmit(bridge, backend):
    job, _ = bridge.enqueue(payload())
    backend.submit_error = True
    with pytest.raises(ConnectionError):
        await bridge.tick()
    assert bridge.store.get(job["id"])["state"] == "unknown"
    await bridge.tick()
    assert len(backend.submissions) == 1
    assert bridge.store.get(job["id"])["call_id"] == f"call-{job['id']}"


async def test_pending_cancel_does_not_wake_gpu(bridge, backend):
    job, _ = bridge.enqueue(payload())
    assert await bridge.cancel(job["id"])
    await bridge.tick()
    assert not backend.submissions
    assert bridge.store.get(job["id"])["state"] == "cancelled"


async def test_running_cancel_uses_control_channel(bridge, backend):
    job, _ = bridge.enqueue(payload())
    await bridge.tick()
    await bridge.cancel(job["id"])
    assert backend.cancelled == [job["id"]]
    assert bridge.store.get(job["id"])["state"] == "cancelling"


async def test_remote_validation_failure_becomes_terminal_history(bridge, backend):
    job, _ = bridge.enqueue(payload())
    await bridge.tick()
    backend.results[f"call-{job['id']}"] = {"error": {"node_errors": {"3": "bad input"}}}
    await bridge.tick()
    assert bridge.store.get(job["id"])["state"] == "failed"
    assert job["id"] in bridge.store.history()


def test_unknown_nodes_rejected_before_gpu_submission(bridge):
    body = payload()
    body["prompt"]["10"]["class_type"] = "MissingCustomNode"
    job, errors = bridge.enqueue(body)
    assert job is None
    assert "10" in errors
    assert not bridge.store.all()


def test_stable_prompt_id_is_idempotent(bridge):
    body = payload(prompt_id=str(uuid.uuid4()))
    first, _ = bridge.enqueue(body)
    again, _ = bridge.enqueue(body)
    assert first["id"] == again["id"]
    assert len(bridge.store.all()) == 1
    body["prompt"]["6"]["inputs"]["text"] = "a different graph"
    with pytest.raises(ValueError):
        bridge.enqueue(body)


def test_input_files_are_snapshotted_at_acceptance(bridge):
    source = bridge.root / "input/photo.png"
    source.write_bytes(b"original")
    body = payload()
    body["prompt"]["50"] = {"class_type": "LoadImage", "inputs": {"image": "photo.png"}}
    job, _ = bridge.enqueue(body)
    source.write_bytes(b"changed")
    assert (bridge.root / "jobs" / job["id"] / "input/photo.png").read_bytes() == b"original"


def test_path_traversal_is_rejected(tmp_path):
    for name in ("../private", "/etc/passwd", "..\\private"):
        with pytest.raises(ValueError):
            safe_path(tmp_path, name)


def test_symlink_escape_is_rejected(tmp_path, symlink):
    root = tmp_path / "root"
    root.mkdir()
    symlink(root / "escape", tmp_path)
    with pytest.raises(ValueError):
        safe_path(root, "escape/private")


def test_remote_rewrite_preserves_original_workflow_and_isolates_outputs():
    original = payload()
    original["prompt"]["30"] = {
        "class_type": "SaveVideo",
        "inputs": {"filename_prefix": "video/MiniMax_H3", "video": ["28", 0], "format": "mp4"},
    }
    changed = job_prompt(original, "job-a")
    assert changed["prompt"]["29"]["inputs"]["filename_prefix"] == "job-a/Krea2"
    assert changed["prompt"]["30"]["inputs"]["filename_prefix"] == "job-a/MiniMax_H3"
    assert original["prompt"]["29"]["inputs"]["filename_prefix"] == "Krea2"
    assert original["prompt"]["30"]["inputs"]["filename_prefix"] == "video/MiniMax_H3"


async def test_output_events_wait_for_local_files(bridge, backend):
    received = []

    async def emit(event, client_id=None):
        received.append(event)

    bridge.emit = emit
    job, _ = bridge.enqueue(payload())
    await bridge.tick()
    backend.snapshots[job["id"]] = {
        "events": [
            {"seq": 1, "event": {"type": "progress", "data": {"value": 1, "max": 8}}},
            {"seq": 2, "event": {"type": "executed", "data": {"node": "29"}}},
            {"seq": 3, "event": {"type": "execution_success", "data": {}}},
        ]
    }
    await bridge.tick()
    assert "progress" in [e["type"] for e in received]
    assert "executed" not in [e["type"] for e in received]
    backend.results[f"call-{job['id']}"] = result_for(job["id"])
    await bridge.tick()
    assert "executed" in [e["type"] for e in received]


def test_metadata_mismatch_blocks_submission(bridge):
    bridge.snapshot["deployment_id"] = "different-version"
    with pytest.raises(RuntimeError):
        bridge.enqueue(payload())


def test_detailed_prompts_are_not_treated_as_input_filenames(bridge):
    body = payload()
    body["prompt"]["6"]["inputs"]["text"] = "A richly detailed cinematic scene. " * 50
    job, errors = bridge.enqueue(body)
    assert job is not None
    assert not errors


async def test_unknown_submission_can_be_cancelled_to_unblock_queue(bridge, backend):
    job, _ = bridge.enqueue(payload())
    bridge.store.update(job["id"], state="unknown")
    await bridge.cancel(job["id"])
    assert backend.cancelled == [job["id"]]
    assert bridge.store.get(job["id"])["state"] == "cancelled"


def test_krea_encoder_mismatch_is_rejected_before_gpu_submission(bridge):
    body = payload()
    body["prompt"]["11"]["inputs"]["type"] = "lumina2"
    # Reproduce the submitted graph's model-sampling wrapper.
    body["prompt"]["69"] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": ["10", 0], "shift": 3},
    }
    body["prompt"]["3"]["inputs"]["model"] = ["69", 0]
    bridge.snapshot["routes"]["/object_info"]["ModelSamplingAuraFlow"] = {}
    job, errors = bridge.enqueue(body)
    assert job is None
    assert errors["11"]["errors"][0]["type"] == "krea2_encoder_type_mismatch"
    assert not bridge.store.all()


def test_unrelated_text_encoder_is_not_rejected(bridge):
    body = payload()
    body["prompt"]["99"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "lumina2"},
    }
    job, errors = bridge.enqueue(body)
    assert job is not None
    assert not errors


async def test_remote_error_keeps_node_id_type_message_and_traceback(bridge, backend):
    received = []

    async def emit(event, client_id=None):
        received.append(event)

    bridge.emit = emit
    job, _ = bridge.enqueue(payload())
    await bridge.tick()
    failure = {
        "prompt_id": job["id"],
        "node_id": "70",
        "node_type": "KSampler",
        "exception_type": "ValueError",
        "exception_message": "Use CLIPLoader type 'krea2'.",
        "traceback": ["original remote traceback"],
        "executed": ["62", "67"],
    }
    backend.results[f"call-{job['id']}"] = {
        "error": None,
        "artifacts": [],
        "history": {
            "outputs": {},
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [
                    ["execution_start", {"prompt_id": job["id"]}],
                    ["execution_error", failure],
                ],
            },
        },
    }
    await bridge.tick()
    saved = bridge.store.get(job["id"])
    assert saved["error"] == failure["exception_message"]
    event = next(event for event in received if event["type"] == "execution_error")
    assert event["data"] == failure
    assert public_job(saved)["execution_error"] == failure
