import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from modal.volume import FileEntryType

from comfyui_modal.backend import ModalBackend
from comfyui_modal.cleanup import check_cleanup_idle, cleanup
from comfyui_modal.helpers.modal_volume import retry_rate_limited
from comfyui_modal.store import Store


@pytest.fixture(autouse=True)
def job_volume_name():
    """cleanup() names the Volume from config.yaml; keep tests off the project's file."""
    with patch("comfyui_modal.cleanup.modal_names", return_value={"job_volume": "app-jobs"}):
        yield


def add_job(store, state, artifacts=True):
    job_id = str(uuid.uuid4())
    store.add(job_id, {})
    store.update(
        job_id,
        state=state,
        result={"artifacts": [{"local_path": "image.png"}]} if artifacts else {},
    )
    return job_id


def test_idle_check_allows_terminal_jobs(tmp_path):
    store = Store(tmp_path / "queue.sqlite3")
    for state in ("completed", "failed", "cancelled"):
        add_job(store, state)
    check_cleanup_idle(tmp_path)
    store.close()


async def test_unfinished_jobs_block_volume_deletion(tmp_path):
    store = Store(tmp_path / "queue.sqlite3")
    add_job(store, "running")
    backend = SimpleNamespace(cleanup_jobs_volume=AsyncMock())
    with pytest.raises(RuntimeError):
        await cleanup(SimpleNamespace(data_dir=tmp_path, dry_run=False), backend)
    backend.cleanup_jobs_volume.assert_not_awaited()
    store.close()


async def test_cleanup_runs_without_local_history(tmp_path):
    backend = SimpleNamespace(
        cleanup_jobs_volume=AsyncMock(return_value={"files": 3, "bytes": 1024**2})
    )
    await cleanup(SimpleNamespace(data_dir=tmp_path, dry_run=False), backend)
    backend.cleanup_jobs_volume.assert_awaited_once_with(dry_run=False)


async def test_dry_run_is_available_while_jobs_are_active(tmp_path):
    store = Store(tmp_path / "queue.sqlite3")
    add_job(store, "running")
    backend = SimpleNamespace(cleanup_jobs_volume=AsyncMock(return_value={"files": 2, "bytes": 0}))
    await cleanup(SimpleNamespace(data_dir=tmp_path, dry_run=True), backend)
    backend.cleanup_jobs_volume.assert_awaited_once_with(dry_run=True)
    store.close()


@pytest.mark.parametrize("dry_run", [False, True])
async def test_backend_clears_all_root_entries_with_one_listing(dry_run):
    backend = ModalBackend.__new__(ModalBackend)
    entries = [
        SimpleNamespace(path="job/artifacts/image.png", type=FileEntryType.FILE, size=123),
        SimpleNamespace(path="job", type=FileEntryType.DIRECTORY, size=999),
        SimpleNamespace(path="orphan/input/upload.png", type=FileEntryType.FILE, size=10),
        SimpleNamespace(path="job/result.json", type=FileEntryType.FILE, size=20),
    ]
    listdir, remove = AsyncMock(return_value=entries), AsyncMock()
    backend.jobs = SimpleNamespace(
        listdir=SimpleNamespace(aio=listdir), remove_file=SimpleNamespace(aio=remove)
    )
    result = await backend.cleanup_jobs_volume(dry_run=dry_run)
    assert result == {"files": 3, "bytes": 153}
    if dry_run:
        remove.assert_not_awaited()
    else:
        assert {call.args[0] for call in remove.call_args_list} == {"/job", "/orphan"}
        assert all(call.kwargs == {"recursive": True} for call in remove.call_args_list)


async def test_backend_repeated_cleanup_is_safe():
    backend = ModalBackend.__new__(ModalBackend)
    remove = AsyncMock()
    backend.jobs = SimpleNamespace(
        listdir=SimpleNamespace(aio=AsyncMock(side_effect=FileNotFoundError)),
        remove_file=SimpleNamespace(aio=remove),
    )
    assert await backend.cleanup_jobs_volume() == {"files": 0, "bytes": 0}
    remove.assert_not_awaited()


async def test_volume_rate_limits_are_retried_then_bounded():
    from modal.exception import ResourceExhaustedError

    limited = ResourceExhaustedError("rate limit")
    with patch("comfyui_modal.helpers.modal_volume.asyncio.sleep", new_callable=AsyncMock) as sleep:
        recovers = AsyncMock(side_effect=[limited, limited, []])
        assert await retry_rate_limited(recovers, "/job/artifacts", recursive=True) == []
        assert sleep.await_count == 2
        with pytest.raises(ResourceExhaustedError):
            await retry_rate_limited(AsyncMock(side_effect=limited), "/job", recursive=True)
