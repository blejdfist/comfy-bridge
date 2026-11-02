from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from comfyui_modal.bridge_client import run_model_command


@pytest.mark.parametrize("fails", [False, True])
async def test_download_client_polls_until_terminal_state(aiohttp_server, monkeypatch, fails):
    states = iter(
        [
            {
                "state": "running",
                "progress": {"phase": "downloading", "bytes": 1024**2, "total": 2 * 1024**2},
            },
            {"state": "failed", "error": "synthetic failure"}
            if fails
            else {"state": "completed", "result": {"path": "loras/style.safetensors"}},
        ]
    )
    bodies = []

    async def enqueue(request):
        bodies.append(await request.json())
        return web.json_response({"id": "test-download"}, status=202)

    async def progress(_):
        return web.json_response(next(states))

    app = web.Application()
    app.router.add_post("/bridge/model-downloads", enqueue)
    app.router.add_get("/bridge/model-downloads/test-download", progress)
    server = await aiohttp_server(app)
    args = SimpleNamespace(
        model_command="download",
        url="https://example.com/style.safetensors",
        folder="loras",
        filename="style.safetensors",
        secret=None,
        file_id=None,
        no_wait=False,
        bridge_url=str(server.make_url("")),
    )
    monkeypatch.setattr("comfyui_modal.bridge_client.asyncio.sleep", AsyncMock())
    if fails:
        with pytest.raises(RuntimeError):
            await run_model_command(args)
    else:
        assert await run_model_command(args) == {"path": "loras/style.safetensors"}
    assert bodies[0]["refresh_catalog"] is True and bodies[0]["skip_existing"] is False
