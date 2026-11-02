import json
import struct
from unittest.mock import AsyncMock

import pytest

from comfyui_modal.model_downloads import (
    ModelDownloadError,
    auth_headers,
    checked_filename,
    checked_url,
    download_model,
    normalize_request,
    open_download,
    resolve_source,
)


def safetensors():
    header = json.dumps({"weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + b"abcd"


async def test_sync_skips_existing_remote_file_before_network_access(tmp_path):
    (tmp_path / "loras").mkdir()
    (tmp_path / "loras/existing.safetensors").write_bytes(b"existing")
    receipt = await download_model(
        {
            "url": "https://civitai.red/models/1",
            "folder": "loras",
            "filename": "existing.safetensors",
            "skip_existing": True,
        },
        tmp_path,
        AsyncMock(),
        session=Session({}),
    )
    assert receipt["skipped"] is True
    assert receipt["bytes"] == 8
    assert (tmp_path / "loras/existing.safetensors").read_bytes() == b"existing"


class Response:
    def __init__(self, data=b"", status=200, headers=None):
        self.data, self.status = data, status
        self.headers = {"Content-Type": "application/octet-stream", **(headers or {})}
        self.content_type = self.headers["Content-Type"].split(";")[0]
        self.content_length = int(self.headers.get("Content-Length", len(data)))
        self.content = self

    async def iter_chunked(self, size):
        for offset in range(0, len(self.data), size):
            yield self.data[offset : offset + size]

    def release(self):
        pass


class Session:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses[url]


def json_response(value):
    return Response(json.dumps(value).encode(), headers={"Content-Type": "application/json"})


async def test_civitai_page_resolves_version_file_and_auth():
    host = "civitai.com"
    version = {
        "id": 7,
        "baseModel": "Krea 2",
        "trainedWords": ["ink style"],
        "files": [
            {
                "id": 81,
                "name": "style.safetensors",
                "primary": True,
                "type": "Model",
                "metadata": {"format": "SafeTensor"},
                "downloadUrl": f"https://{host}/api/download/models/7?fileId=81",
            }
        ],
    }
    session = Session({f"https://{host}/api/v1/model-versions/7": json_response(version)})
    source = await resolve_source(
        session,
        normalize_request(
            {"url": f"https://{host}/models/2/style?modelVersionId=7", "folder": "loras"}
        ),
        {"CIVITAI_API_TOKEN": "private-key"},
    )
    assert source["filename"] == "style.safetensors"
    assert source["base_model"] == "Krea 2"
    assert source["trained_words"] == ["ink style"]
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer private-key"


async def test_direct_civitai_download_honors_file_id_in_url():
    files = [
        {"id": 1, "name": "first.safetensors", "primary": True},
        {"id": 2, "name": "second.safetensors", "primary": False},
    ]
    session = Session(
        {"https://civitai.red/api/v1/model-versions/7": json_response({"id": 7, "files": files})}
    )
    source = await resolve_source(
        session,
        normalize_request(
            {"url": "https://civitai.red/api/download/models/7?fileId=2", "folder": "loras"}
        ),
        {},
    )
    assert source["filename"] == "second.safetensors"


async def test_model_page_without_version_uses_default_published_version():
    session = Session(
        {
            "https://civitai.red/api/v1/models/2": json_response(
                {"modelVersions": [{"id": 7, "files": [{"id": 1, "name": "style.safetensors"}]}]}
            )
        }
    )
    source = await resolve_source(
        session, normalize_request({"url": "https://civitai.red/models/2", "folder": "loras"}), {}
    )
    assert source["version_id"] == 7


async def test_download_streams_verifies_and_publishes_a_remote_model(tmp_path):
    data = safetensors()
    session = Session({"https://weights.example/style.safetensors": Response(data)})
    updates = []

    async def report(progress):
        updates.append(progress)

    result = await download_model(
        {"url": "https://weights.example/style.safetensors", "folder": "loras"},
        tmp_path,
        report,
        session=session,
    )
    assert (tmp_path / "loras/style.safetensors").read_bytes() == data
    assert result["bytes"] == len(data)
    assert result["path"] == "loras/style.safetensors"
    assert not list(tmp_path.rglob("*.partial"))
    assert list((tmp_path / ".bridge-downloads").glob("*.json"))
    assert updates[-1]["phase"] == "verifying"
    assert updates[-1]["bytes"] == len(safetensors())


async def test_html_disguised_as_model_is_rejected(tmp_path):
    session = Session(
        {"https://weights.example/style.ckpt": Response(b"<!DOCTYPE html><html>Login</html>")}
    )
    with pytest.raises(ModelDownloadError):
        await download_model(
            {"url": "https://weights.example/style.ckpt", "folder": "loras"},
            tmp_path,
            AsyncMock(),
            session=session,
        )
    assert not (tmp_path / "loras/style.ckpt").exists()
    assert not list(tmp_path.rglob("*.partial"))


async def test_truncated_download_is_discarded(tmp_path):
    session = Session(
        {
            "https://weights.example/style.safetensors": Response(
                safetensors(), headers={"Content-Length": "100000"}
            )
        }
    )
    with pytest.raises(ModelDownloadError):
        await download_model(
            {"url": "https://weights.example/style.safetensors", "folder": "loras"},
            tmp_path,
            AsyncMock(),
            session=session,
        )
    assert not (tmp_path / "loras/style.safetensors").exists()
    assert not list(tmp_path.rglob("*.partial"))


async def test_existing_model_is_not_overwritten(tmp_path):
    (tmp_path / "loras").mkdir()
    (tmp_path / "loras/style.safetensors").write_bytes(b"existing")
    session = Session({"https://weights.example/style.safetensors": Response(safetensors())})
    report = AsyncMock()
    with pytest.raises(ModelDownloadError):
        await download_model(
            {"url": "https://weights.example/style.safetensors", "folder": "loras"},
            tmp_path,
            report,
            session=session,
        )
    assert (tmp_path / "loras/style.safetensors").read_bytes() == b"existing"
    # Rejected before the transfer starts, not after downloading the whole file.
    assert [call.args[0]["phase"] for call in report.await_args_list] == ["resolving"]


async def test_civitai_token_is_not_forwarded_to_download_cdn():
    session = Session(
        {
            "https://civitai.red/api/download/models/7": Response(
                status=302,
                headers={"Location": "https://cdn.example/model.safetensors?signature=private"},
            ),
            "https://cdn.example/model.safetensors?signature=private": Response(safetensors()),
        }
    )
    await open_download(
        session, "https://civitai.red/api/download/models/7", {"CIVITAI_API_TOKEN": "private-key"}
    )
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer private-key"
    assert "Authorization" not in session.calls[1][1]["headers"]


@pytest.mark.parametrize(
    "url", ["http://example.com/model.safetensors", "https://user:pw@example.com/model"]
)
def test_invalid_or_credential_bearing_urls_are_rejected(url):
    with pytest.raises(ModelDownloadError):
        checked_url(url)


@pytest.mark.parametrize("name", ["../evil.safetensors", "x.py", "x\\y.safetensors"])
def test_invalid_destination_filenames_are_rejected(name):
    with pytest.raises(ModelDownloadError):
        checked_filename(name)


@pytest.mark.parametrize(
    "name", ["../evil.safetensors", "a//b.safetensors", "/abs.safetensors", "a\\b.safetensors"]
)
def test_subfolder_filenames_reject_traversal_and_backslashes(name):
    assert checked_filename("qwen-image/vae.safetensors", subfolders=True)
    with pytest.raises(ModelDownloadError):
        checked_filename(name, subfolders=True)


def test_tokens_are_scoped_to_their_service():
    credentials = {"CIVITAI_TOKEN": "civi", "HF_TOKEN": "hf"}
    assert (
        auth_headers("https://civitai.red/api", "civitai.red", credentials)["Authorization"]
        == "Bearer civi"
    )
    assert (
        auth_headers("https://huggingface.co/file", "huggingface.co", credentials)["Authorization"]
        == "Bearer hf"
    )
    assert "Authorization" not in auth_headers(
        "https://cdn.example/file", "civitai.red", credentials
    )
