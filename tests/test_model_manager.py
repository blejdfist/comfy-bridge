import copy
from unittest.mock import AsyncMock

import pytest

from comfyui_modal.bridge import Bridge
from comfyui_modal.server import create_app


def request():
    return {
        "url": "https://civitai.red/models/1?modelVersionId=2",
        "folder": "loras",
        "secret": "civitai",
    }


def download_result(bridge):
    snapshot = copy.deepcopy(bridge.snapshot)
    snapshot["routes"]["/models/loras"] = ["style.safetensors"]
    return {"error": None, "receipt": {"path": "loras/style.safetensors"}, "snapshot": snapshot}


async def test_download_completes_and_refreshes_catalog(bridge, backend):
    download_id = bridge.enqueue_model_download(request())
    await bridge.download_tick()
    assert len(backend.model_submissions) == 1
    backend.results[f"model-call-{download_id}"] = download_result(bridge)
    await bridge.download_tick()
    assert bridge.store.model_downloads()[0]["state"] == "completed"
    assert bridge.snapshot["routes"]["/models/loras"] == ["style.safetensors"]


async def test_batch_download_completes_without_catalog_snapshot(bridge, backend):
    download_id = bridge.enqueue_model_download({**request(), "refresh_catalog": False})
    await bridge.download_tick()
    assert backend.model_submissions[0][1]["refresh_catalog"] is False
    backend.results[f"model-call-{download_id}"] = {
        "error": None,
        "receipt": {"path": "loras/style.safetensors"},
        "snapshot": None,
    }
    await bridge.download_tick()
    assert bridge.store.model_downloads()[0]["state"] == "completed"


async def test_download_recovers_after_local_restart_without_resubmission(
    bridge, backend, data_dir
):
    download_id = bridge.enqueue_model_download(request())
    await bridge.download_tick()
    bridge.store.close()
    restarted = Bridge(
        data_dir,
        backend,
        output_dir=data_dir / "generated",
        workflows_dir=data_dir / "workflows",
    )
    try:
        backend.results[f"model-call-{download_id}"] = download_result(restarted)
        await restarted.download_tick()
        assert restarted.store.model_downloads()[0]["state"] == "completed"
        assert len(backend.model_submissions) == 1
    finally:
        restarted.store.close()


async def test_catalog_version_mismatch_finishes_with_receipt(bridge, backend):
    first = bridge.enqueue_model_download(request())
    await bridge.download_tick()
    result = download_result(bridge)
    result["snapshot"]["deployment_id"] = "different-runtime"
    backend.results[f"model-call-{first}"] = result
    await bridge.download_tick()
    failed = bridge.store.model_downloads()[0]
    assert failed["state"] == "failed"
    assert failed["result"] == result["receipt"]


async def test_duplicate_request_reattaches_until_download_finishes(bridge, backend):
    first = bridge.enqueue_model_download(request())
    assert bridge.enqueue_model_download(request()) == first
    await bridge.download_tick()
    backend.results[f"model-call-{first}"] = {"error": "remote error", "receipt": None}
    await bridge.download_tick()
    assert bridge.enqueue_model_download(request()) != first


async def test_downloads_run_in_parallel_up_to_limit(bridge, backend, monkeypatch):
    monkeypatch.setattr("comfyui_modal.bridge.MODEL_DOWNLOAD_CONCURRENCY", 2)
    ids = [
        bridge.enqueue_model_download({**request(), "url": f"https://example.com/{n}.safetensors"})
        for n in range(3)
    ]
    await bridge.download_tick()
    assert [submitted for submitted, _ in backend.model_submissions] == ids[:2]
    backend.results[f"model-call-{ids[0]}"] = {"error": "remote error", "receipt": None}
    await bridge.download_tick()  # finishes the first; its slot is reused on the next tick
    await bridge.download_tick()
    assert [submitted for submitted, _ in backend.model_submissions] == ids


async def test_ambiguous_download_submission_recovers_call_id(bridge, backend):
    bridge.enqueue_model_download(request())
    backend.submit_error = True
    await bridge.download_tick()
    assert bridge.store.model_downloads()[0]["state"] == "unknown"
    await bridge.download_tick()
    assert bridge.store.model_downloads()[0]["state"] == "running"
    assert len(backend.model_submissions) == 1


async def test_pending_download_cancels_without_cloud_submission(bridge, backend):
    download_id = bridge.enqueue_model_download(request())
    await bridge.cancel_model_download(download_id)
    await bridge.download_tick()
    assert bridge.store.model_downloads()[0]["state"] == "cancelled"
    assert not backend.model_submissions


async def test_running_download_cancellation_is_reconciled(bridge, backend):
    download_id = bridge.enqueue_model_download(request())
    await bridge.download_tick()
    await bridge.cancel_model_download(download_id)
    backend.results[f"model-call-{download_id}"] = {"error": "Download cancelled", "receipt": None}
    await bridge.download_tick()
    assert bridge.store.model_downloads()[0]["state"] == "cancelled"
    assert backend.model_cancelled == [download_id]


async def test_remote_download_error_marks_download_failed(bridge, backend):
    download_id = bridge.enqueue_model_download(request())
    await bridge.download_tick()
    backend.results[f"model-call-{download_id}"] = {"error": "remote error", "receipt": None}
    await bridge.download_tick()
    assert bridge.store.model_downloads()[0]["state"] == "failed"
    assert bridge.store.model_downloads()[0]["error"] == "remote error"


@pytest.mark.parametrize("full", [False, True])
async def test_catalog_refresh_route(bridge, backend, aiohttp_client, full):
    if full:
        catalog = download_result(bridge)["snapshot"]
    else:
        catalog = {
            "deployment_id": bridge.snapshot["deployment_id"],
            "model_lists": {"loras": ["style.safetensors"]},
        }
    backend.refresh_models = AsyncMock(return_value=catalog)
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    response = await client.post("/bridge/models/refresh", params={"full": "1"} if full else {})
    assert response.status == 200
    assert (await response.json())["mode"] == ("full" if full else "models")
    catalog = await (await client.get("/bridge/models/catalog")).json()
    assert catalog["models"]["loras"] == ["style.safetensors"]
    backend.refresh_models.assert_awaited_once_with(full=full)


async def test_missing_cache_automatically_bootstraps_full_snapshot(bridge, backend):
    snapshot = download_result(bridge)["snapshot"]
    bridge.snapshot = None
    backend.refresh_models = AsyncMock(return_value=snapshot)
    assert await bridge.refresh_models() == "full"
    backend.refresh_models.assert_awaited_once_with(full=True)
    assert bridge.snapshot == snapshot


async def test_incompatible_cache_automatically_rebuilds_metadata(bridge, backend):
    snapshot = download_result(bridge)["snapshot"]
    bridge.snapshot["deployment_id"] = "older-comfyui"
    backend.refresh_models = AsyncMock(return_value=snapshot)
    assert await bridge.refresh_models() == "full"
    backend.refresh_models.assert_awaited_once_with(full=True)


async def test_civitai_url_with_token_is_not_persisted(bridge, aiohttp_client):
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    body = request()
    body["url"] += "&token=private-key"
    response = await client.post("/bridge/model-downloads", json=body)
    assert response.status == 400
    assert not bridge.store.model_downloads()
