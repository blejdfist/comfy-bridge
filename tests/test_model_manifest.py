from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from comfyui_modal.model_manifest import (
    credential_values,
    load_yaml,
    manifest_entries,
    sync_manifest,
    unlisted_files,
)


def document(**overrides):
    return {
        "version": 1,
        "models": [
            {
                "url": "https://huggingface.co/example/model/resolve/main/a.safetensors",
                "folder": "loras",
                "filename": "a.safetensors",
                **overrides,
            }
        ],
    }


def test_manifest_normalizes_and_rejects_credentials_and_duplicate_destinations():
    entries = manifest_entries(document(credential="civitai"))
    assert entries[0][1] == "civitai"
    assert entries[0][0]["secret"] is None
    with pytest.raises(RuntimeError):
        manifest_entries(document(CIVITAI_API_TOKEN="synthetic"))
    duplicate = document()
    duplicate["models"] *= 2
    with pytest.raises(RuntimeError):
        manifest_entries(duplicate)


def test_credential_validation_does_not_echo_values():
    value = {"version": 1, "credentials": {"civitai": {"CIVITAI_API_TOKEN": "synthetic"}}}
    assert credential_values(value, {"civitai"}) == value["credentials"]
    value["credentials"]["civitai"] = {"synthetic-secret-in-key": "synthetic-secret-in-value"}
    with pytest.raises(RuntimeError) as exc:
        credential_values(value, {"civitai"})
    assert "synthetic-secret" not in str(exc.value)


@pytest.mark.parametrize("version", ["true", "1.0"])
def test_manifest_version_must_be_integer_one(version):
    with patch("pathlib.Path.read_text", return_value=f"version: {version}\n"):
        with pytest.raises(RuntimeError, match="version: 1"):
            load_yaml("unused")


def test_yaml_parser_errors_do_not_echo_source():
    with patch(
        "pathlib.Path.read_text", return_value="version: 1\ncredentials: [synthetic-secret: ["
    ):
        with pytest.raises(RuntimeError) as exc:
            load_yaml("unused", private=True)
    assert "synthetic-secret" not in str(exc.value)
    assert exc.value.__suppress_context__


@pytest.mark.asyncio
async def test_sync_skips_installed_files_without_reading_credentials(aiohttp_server):
    from aiohttp import web

    app = web.Application()
    app.router.add_get(
        "/bridge/models/catalog",
        lambda _: web.json_response({"models": {"loras": ["a.safetensors"]}}),
    )

    refreshes = []

    async def refresh(request):
        refreshes.append(dict(request.query))
        return web.json_response({"ready": True})

    app.router.add_post("/bridge/models/refresh", refresh)
    server = await aiohttp_server(app)
    args = SimpleNamespace(
        manifest="unused",
        credentials="never-read",
        bridge_url=str(server.make_url("")),
        full=True,
    )
    with patch(
        "comfyui_modal.model_manifest.load_yaml", return_value=document(credential="civitai")
    ) as loader:
        await sync_manifest(args)
    loader.assert_called_once_with("unused")
    assert refreshes == [{"full": "1"}]


@pytest.mark.asyncio
async def test_sync_only_queues_secret_names(aiohttp_server):
    from aiohttp import web

    app = web.Application()
    app.router.add_get("/bridge/models/catalog", lambda _: web.json_response({"models": {}}))
    server = await aiohttp_server(app)
    args = SimpleNamespace(
        manifest="unused", credentials="synthetic", bridge_url=str(server.make_url(""))
    )
    values = {
        "version": 1,
        "credentials": {"civitai": {"CIVITAI_API_TOKEN": "synthetic-private-token"}},
    }
    create = AsyncMock()
    with (
        patch(
            "comfyui_modal.model_manifest.load_yaml",
            side_effect=[document(credential="civitai"), values],
        ),
        patch(
            "modal.Secret",
            SimpleNamespace(
                objects=SimpleNamespace(
                    create=SimpleNamespace(aio=create), delete=SimpleNamespace(aio=AsyncMock())
                )
            ),
        ),
        patch(
            "comfyui_modal.model_manifest.modal_names",
            return_value={"app_name": "app"},
        ),
        patch("comfyui_modal.bridge_client.run_model_command", new_callable=AsyncMock) as manage,
    ):
        await sync_manifest(args)
    assert create.call_args.args[1] == values["credentials"]["civitai"]
    request = manage.call_args_list[0].args[0]
    assert create.call_args.args[0] == request.secret == "app-civitai"
    assert "synthetic-private-token" not in repr(vars(request))


@pytest.mark.asyncio
@pytest.mark.parametrize("download_fails", [False, True])
async def test_sync_refreshes_once_after_downloads_even_on_failure(aiohttp_server, download_fails):
    from aiohttp import web

    async def catalog(_):
        return web.json_response({"models": {}})

    app = web.Application()
    app.router.add_get("/bridge/models/catalog", catalog)
    server = await aiohttp_server(app)
    args = SimpleNamespace(
        manifest="unused", credentials="never-read", bridge_url=str(server.make_url(""))
    )
    manifest = document()
    manifest["models"].append({**manifest["models"][0], "filename": "b.safetensors"})
    events = []

    async def manage(command):
        events.append(command.model_command)
        if command.model_command == "download":
            assert command.refresh_catalog is False
            assert command.skip_existing is True
            if download_fails:
                raise RuntimeError("download failed")
            return {"skipped": command.filename == "b.safetensors"}

    with (
        patch("comfyui_modal.model_manifest.load_yaml", return_value=manifest),
        patch("comfyui_modal.bridge_client.run_model_command", side_effect=manage),
    ):
        if download_fails:
            with pytest.raises(RuntimeError):
                await sync_manifest(args)
            assert events == ["download", "download", "refresh"]
        else:
            await sync_manifest(args)
            assert events == ["download", "download", "refresh"]


def test_rm_selects_only_unlisted_model_files():
    entries = manifest_entries(
        {
            "version": 1,
            "models": [
                {
                    "url": "https://example.com/a.safetensors",
                    "folder": "vae",
                    "filename": "sub/a.safetensors",
                }
            ],
        }
    )
    paths = [
        "vae/sub/a.safetensors",
        "vae/old.safetensors",
        "vae/.123.b.safetensors.partial",
        ".bridge-downloads/receipt.json",
        "notes.txt",
    ]
    assert unlisted_files(paths, entries) == ["vae/old.safetensors"]
