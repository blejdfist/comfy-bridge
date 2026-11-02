import io
import json

from aiohttp import FormData
from PIL import Image
from workflow_fixtures import api_workflow

from comfyui_modal.server import create_app


async def test_view_serves_configured_outputs_and_legacy_history(bridge, aiohttp_client):
    bridge.output_root.mkdir(parents=True)
    (bridge.output_root / "new.png").write_bytes(b"new-image")
    legacy = bridge.root / "output"
    legacy.mkdir()
    (legacy / "old.png").write_bytes(b"old-image")
    (legacy / "new.png").write_bytes(b"outdated-image")
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    assert await (await client.get("/view?filename=new.png&type=output")).read() == b"new-image"
    assert await (await client.get("/view?filename=old.png&type=output")).read() == b"old-image"
    assert (await client.get("/view?filename=missing.png&type=output")).status == 404


async def test_editor_and_metadata_work_without_a_gpu(bridge, aiohttp_client):
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    assert (await client.get("/")).status == 200
    response = await client.get("/api/object_info")
    assert "KSampler" in await response.json()
    response = await client.get("/api/models/diffusion_models")
    assert await response.json() == ["krea2_turbo_int8_convrot.safetensors"]


async def test_websocket_queue_status_and_prompt_submission(bridge, aiohttp_client):
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    async with client.ws_connect("/ws?clientId=browser") as ws:
        message = await ws.receive_json()
        assert message["data"]["sid"] == "browser"
        await ws.send_json({"type": "feature_flags", "data": {}})
        assert (await ws.receive_json())["type"] == "feature_flags"
        response = await client.post("/api/prompt", json={"prompt": api_workflow("cabin")})
        body = await response.json()
        assert response.status == 200
        assert body["prompt_id"]
        status = await ws.receive_json()
        assert status["data"]["status"]["exec_info"]["queue_remaining"] == 1
        queue = await (await client.get("/api/queue")).json()
        assert queue["queue_pending"][0][1] == body["prompt_id"]
        jobs = await (await client.get("/api/jobs?status=pending")).json()
        assert jobs["pagination"]["total"] == 1


async def test_settings_survive_requests(bridge, aiohttp_client):
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    response = await client.post("/api/settings", json={"Comfy.ColorPalette": "dark"})
    assert response.status == 200
    assert (await (await client.get("/api/settings")).json())["Comfy.ColorPalette"] == "dark"


async def test_upload_populates_local_loadimage_widget(bridge, aiohttp_client):
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    form = FormData()
    form.add_field("image", buffer.getvalue(), filename="test.png", content_type="image/png")
    response = await client.post("/api/upload/image", data=form)
    assert response.status == 200
    assert (await response.json())["name"] == "test.png"
    schema = await (await client.get("/object_info/LoadImage")).json()
    assert schema["LoadImage"]["input"]["required"]["image"][0] == ["test.png"]
    response = await client.get("/view?filename=test.png&type=input")
    assert await response.read() == buffer.getvalue()


async def test_cross_origin_cannot_submit_paid_jobs(bridge, aiohttp_client):
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    response = await client.post(
        "/prompt", json={"prompt": api_workflow("cabin")}, headers={"Origin": "https://example.com"}
    )
    assert response.status == 403
    assert not bridge.store.all()


async def test_bad_krea_encoder_returns_actionable_local_validation(bridge, aiohttp_client):
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    prompt = api_workflow("cabin")
    prompt["11"]["inputs"]["type"] = "lumina2"
    response = await client.post("/api/prompt", json={"prompt": prompt})
    assert response.status == 400
    assert "11" in (await response.json())["node_errors"]
    assert not bridge.store.all()


async def test_no_metadata_returns_actionable_error(tmp_path, backend, aiohttp_client):
    from comfyui_modal.bridge import Bridge

    bridge = Bridge(
        tmp_path, backend, output_dir=tmp_path / "output", workflows_dir=tmp_path / "workflows"
    )
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    response = await client.post("/prompt", json={"prompt": api_workflow("cabin")})
    assert response.status == 503


async def test_workflow_files_are_discovered_live_and_saved_in_source_folder(
    bridge, aiohttp_client
):
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    nested = bridge.workflows_root / "portraits"
    nested.mkdir()
    graph = {"nodes": [], "links": [], "version": 0.4}
    source = nested / "new.json"
    source.write_text(json.dumps(graph))
    (nested / "api.json").write_text(json.dumps(api_workflow("cabin")))
    (nested / "broken.json").write_text("{not json")
    response = await client.get("/api/v2/userdata?dir=workflows")
    assert "workflows/portraits/new.json" in {item["path"] for item in await response.json()}
    response = await client.get("/templates/index.json")
    names = {item["name"] for item in (await response.json())[0]["templates"]}
    assert "portraits/new" in names
    assert not names & {"portraits/api", "portraits/broken"}
    for name in ("api", "broken"):
        assert (await client.get(f"/templates/portraits/{name}.json")).status == 404
    response = await client.get("/templates/portraits/new.json")
    assert await response.json() == graph
    response = await client.get("/api/userdata/workflows%2Fportraits%2Fnew.json")
    assert await response.json() == graph
    graph["revision"] = 2
    response = await client.post("/api/userdata/workflows%2Fportraits%2Fnew.json", json=graph)
    assert response.status == 200
    assert json.loads(source.read_text()) == graph
    assert not (bridge.root / "userdata/workflows/portraits/new.json").exists()
    source.unlink()
    response = await client.get("/templates/index.json")
    assert "portraits/new" not in {item["name"] for item in (await response.json())[0]["templates"]}


async def test_workflow_move_and_full_info_use_virtual_paths(bridge, aiohttp_client):
    client = await aiohttp_client(create_app(bridge, run_queue=False))
    response = await client.post(
        "/api/userdata/workflows%2Ftest.json?full_info=true", data=b'{"nodes":[]}'
    )
    assert (await response.json())["path"] == "workflows/test.json"
    response = await client.post(
        "/api/userdata/workflows%2Ftest.json/move/workflows%2Frenamed.json"
    )
    assert response.status == 200
    assert await response.json() == "workflows/renamed.json"
    assert (bridge.workflows_root / "renamed.json").exists()
    response = await client.get("/api/v2/userdata")
    assert "workflows/renamed.json" in {item["path"] for item in await response.json()}
