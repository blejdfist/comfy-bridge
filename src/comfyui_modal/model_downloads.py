"""Resolve model links and stream weights directly into the remote model Volume."""

import asyncio
import hashlib
import json
import re
import struct
import time
import uuid
from email.message import Message
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit, urlunsplit

import aiohttp

from .helpers.paths import safe_path

CIVITAI_HOSTS = {"civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"}
MODEL_FOLDERS = (
    "loras",
    "diffusion_models",
    "checkpoints",
    "text_encoders",
    "vae",
    "embeddings",
    "controlnet",
    "upscale_models",
)
MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
RESUME_ATTEMPTS = 5


class ModelDownloadError(ValueError):
    """A safe-to-display error which never includes a signed URL or credentials."""


def checked_url(url):
    if not isinstance(url, str) or any(char.isspace() for char in url):
        raise ModelDownloadError("Paste a complete HTTPS URL without spaces")
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ModelDownloadError("Model downloads require an HTTPS URL")
    if parts.username or parts.password:
        raise ModelDownloadError("Use a Modal Secret instead of credentials in the URL")
    host = parts.hostname.lower()
    if host in CIVITAI_HOSTS and any(
        key.lower() in {"token", "api_key", "apikey"} for key in parse_qs(parts.query)
    ):
        raise ModelDownloadError("Remove the token from the Civitai URL and select a Modal Secret")
    return urlunsplit(parts._replace(fragment=""))


def checked_filename(name, *, subfolders=False):
    """A model filename; with ``subfolders``, optionally under ``/``-separated directories,
    matching how ComfyUI lists nested models (``qwen-image/qwen_image_vae.safetensors``)."""
    parts = name.split("/") if subfolders and isinstance(name, str) else [name]
    if (
        not isinstance(name, str)
        or len(name) > 240
        or any(
            not part
            or part.startswith(".")
            or any(char in part for char in ("/", "\\", "\x00", "\r", "\n"))
            for part in parts
        )
    ):
        raise ModelDownloadError(
            "Filename must be a model filename, optionally under plain subfolders"
            if subfolders
            else "Filename must be a single model filename without directories"
        )
    if Path(name).suffix.lower() not in MODEL_SUFFIXES:
        raise ModelDownloadError("Choose a model weight file, such as a .safetensors file")
    return name


def normalize_request(body):
    url = checked_url(body.get("url", ""))
    folder = body.get("folder")
    if folder not in MODEL_FOLDERS:
        raise ModelDownloadError("Choose a supported ComfyUI model folder")
    filename = checked_filename(body["filename"], subfolders=True) if body.get("filename") else None
    secret = body.get("secret") or None
    if secret and not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", secret):
        raise ModelDownloadError("Enter the name of an existing Modal Secret")
    file_id = int(body["file_id"]) if body.get("file_id") else None
    if file_id is not None and file_id <= 0:
        raise ModelDownloadError("Civitai file ID must be positive")
    for flag in ("refresh_catalog", "skip_existing"):
        if flag in body and type(body[flag]) is not bool:
            raise ModelDownloadError(f"{flag} must be a boolean")
    return {
        "url": url,
        "folder": folder,
        "filename": filename,
        "secret": secret,
        "file_id": file_id,
        "refresh_catalog": body.get("refresh_catalog", True),
        "skip_existing": body.get("skip_existing", False),
    }


CREDENTIAL_KEYS = (
    "CIVITAI_API_TOKEN",
    "CIVITAI_API_KEY",
    "CIVITAI_TOKEN",
    "HF_TOKEN",
    "MODEL_DOWNLOAD_TOKEN",
)


def auth_headers(url, initial_host, credentials):
    host = urlsplit(url).hostname.lower()
    token = None
    if host in CIVITAI_HOSTS:
        token = (
            credentials.get("CIVITAI_API_TOKEN")
            or credentials.get("CIVITAI_API_KEY")
            or credentials.get("CIVITAI_TOKEN")
        )
    elif host == "huggingface.co":
        token = credentials.get("HF_TOKEN")
    elif host == initial_host:
        token = credentials.get("MODEL_DOWNLOAD_TOKEN")
    headers = {"User-Agent": "comfyui-modal-bridge/0.1", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def open_download(session, url, credentials, start=0):
    initial_host = urlsplit(url).hostname.lower()
    for _ in range(9):
        url = checked_url(url)
        headers = auth_headers(url, initial_host, credentials)
        if start:
            headers["Range"] = f"bytes={start}-"
        response = await session.get(url, headers=headers, allow_redirects=False)
        if response.status in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.release()
            if not location:
                raise ModelDownloadError("Download redirect has no destination")
            url = urljoin(url, location)
            continue
        if response.status >= 400:
            status = response.status
            response.release()
            if status in {401, 403}:
                raise ModelDownloadError(
                    f"Host returned HTTP {status}. Select a Modal Secret with your Civitai API key "
                    "and ensure your account has download access to this file."
                )
            raise ModelDownloadError(f"Model host returned HTTP {status}")
        return response
    raise ModelDownloadError("Too many download redirects")


async def api_json(session, url, credentials):
    response = await open_download(session, url, credentials)
    try:
        if response.content_type != "application/json":
            raise ModelDownloadError("Civitai returned a web page instead of API metadata")
        data = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            data.extend(chunk)
            if len(data) > 16 * 1024 * 1024:
                raise ModelDownloadError("Civitai API response is unexpectedly large")
        return json.loads(data)
    finally:
        response.release()


async def resolve_source(session, request, credentials):
    parts = urlsplit(request["url"])
    if parts.hostname.lower() not in CIVITAI_HOSTS:
        # Hugging Face browser links are made into raw file links.
        path = (
            parts.path.replace("/blob/", "/resolve/", 1)
            if parts.hostname == "huggingface.co"
            else parts.path
        )
        return {"url": urlunsplit(parts._replace(path=path)), "filename": None}
    query = parse_qs(parts.query)
    origin = f"https://{parts.hostname}"
    version_id = query.get("modelVersionId", [None])[0]
    match = re.fullmatch(
        r"/(?:api/download/models|api/v1/model-versions|model-versions)/(\d+)/?", parts.path
    )
    if match:
        version_id = match.group(1)
    if version_id:
        if not str(version_id).isdigit():
            raise ModelDownloadError("Invalid Civitai model version ID")
        version = await api_json(
            session, f"{origin}/api/v1/model-versions/{version_id}", credentials
        )
    else:
        match = re.match(r"^/models/(\d+)(?:/|$)", parts.path)
        if not match:
            raise ModelDownloadError("Use a Civitai model page, version link or download link")
        model = await api_json(session, f"{origin}/api/v1/models/{match.group(1)}", credentials)
        versions = model.get("modelVersions", [])
        if not versions:
            raise ModelDownloadError("This Civitai model has no downloadable versions")
        version = versions[0]
        version_id = str(version["id"])
    files = [
        file
        for file in version.get("files", [])
        if Path(file.get("name", "")).suffix.lower() in MODEL_SUFFIXES
    ]
    file_id = request.get("file_id") or int(query.get("fileId", [0])[0])
    if file_id:
        files = [file for file in files if file.get("id") == file_id]
    for key in ("type", "format", "fp", "size"):
        if key in query:
            wanted = query[key][0].lower()
            files = [
                file
                for file in files
                if str(
                    file.get("type") if key == "type" else file.get("metadata", {}).get(key, "")
                ).lower()
                == wanted
            ]
    if not files:
        raise ModelDownloadError(
            "No matching Civitai weight file; check the version and file selection"
        )
    files.sort(
        key=lambda file: (
            not bool(file.get("primary")),
            not file["name"].lower().endswith(".safetensors"),
        )
    )
    file = files[0]
    return {
        "url": file.get("downloadUrl") or f"{origin}/api/download/models/{version_id}",
        "filename": file["name"],
        "version_id": int(version_id),
        "file_id": file.get("id"),
        "base_model": version.get("baseModel"),
        "trained_words": version.get("trainedWords", []),
    }


def verify_weight_file(path):
    with path.open("rb") as file:
        prefix = file.read(256)
        if prefix.lstrip().lower().startswith((b"<!doctype", b"<html", b"<?xml", b'{"error"')):
            raise ModelDownloadError("The host returned a web/error page instead of model weights")
        if not prefix:
            raise ModelDownloadError("Downloaded model file is empty")
        if path.name.lower().endswith(".safetensors.partial"):
            if len(prefix) < 8:
                raise ModelDownloadError("Invalid safetensors file header")
            length = struct.unpack("<Q", prefix[:8])[0]
            size = path.stat().st_size
            if length < 2 or length > min(100 * 1024 * 1024, size - 8):
                raise ModelDownloadError("Invalid safetensors file header")
            file.seek(8)
            try:
                header = json.loads(file.read(length))
            except (ValueError, UnicodeDecodeError):
                raise ModelDownloadError("Invalid safetensors metadata") from None
            if not isinstance(header, dict):
                raise ModelDownloadError("Invalid safetensors metadata")
            tensors = [entry for key, entry in header.items() if key != "__metadata__"]
            if not tensors or any(
                not isinstance(entry, dict)
                or not isinstance(entry.get("data_offsets"), list)
                or len(entry["data_offsets"]) != 2
                or not all(isinstance(offset, int) for offset in entry["data_offsets"])
                or not 0
                <= entry["data_offsets"][0]
                <= entry["data_offsets"][1]
                <= size - 8 - length
                for entry in tensors
            ):
                raise ModelDownloadError(
                    "Safetensors tensor offsets do not match the downloaded file"
                )


async def download_model(request, root: Path, report, credentials=None, session=None):
    """Streaming transfer with verification and publication only after the complete file exists."""
    request = normalize_request(request)
    if request["skip_existing"] and request["filename"]:
        target = safe_path(root, f"{request['folder']}/{request['filename']}")
        if target.is_file():
            return {
                "path": f"{request['folder']}/{request['filename']}",
                "folder": request["folder"],
                "filename": request["filename"],
                "bytes": target.stat().st_size,
                "skipped": True,
            }
    credentials = credentials or {}
    if session is None:
        async with aiohttp.ClientSession(
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=3300, sock_connect=30, sock_read=120),
        ) as session:
            try:
                return await download_model(request, root, report, credentials, session)
            except (aiohttp.ClientError, TimeoutError) as exc:
                raise ModelDownloadError(
                    f"Model transfer failed ({type(exc).__name__}); retry the download"
                ) from None
    await report({"phase": "resolving", "bytes": 0})
    source = await resolve_source(session, request, credentials)
    response = await open_download(session, source["url"], credentials)
    temporary = None
    try:
        if response.content_type in {"text/html", "application/json", "application/xml"}:
            raise ModelDownloadError(
                "The URL points to a web page or API response, not model weights"
            )
        disposition = Message()
        disposition["content-disposition"] = response.headers.get("Content-Disposition", "")
        # The request filename was validated above; names reported by the host stay flat.
        filename = request["filename"] or checked_filename(
            source.get("filename")
            or disposition.get_filename()
            or Path(urlsplit(source["url"]).path).name
        )
        target = safe_path(root, f"{request['folder']}/{filename}")
        if target.exists():
            raise ModelDownloadError(
                "That filename already exists remotely; choose another filename"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{uuid.uuid4().hex}.{target.name}.partial")
        received, last_update = 0, 0.0
        total = response.content_length
        with temporary.open("wb") as output:
            for attempt in range(RESUME_ATTEMPTS + 1):
                try:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        await asyncio.to_thread(output.write, chunk)
                        received += len(chunk)
                        if time.monotonic() - last_update >= 0.5:
                            await report(
                                {
                                    "phase": "downloading",
                                    "bytes": received,
                                    "total": total,
                                    "filename": filename,
                                }
                            )
                            last_update = time.monotonic()
                    break
                except (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError, TimeoutError):
                    # Hosts and CDNs drop long transfers; continue from the bytes on disk.
                    response.release()
                    if attempt == RESUME_ATTEMPTS:
                        raise
                    await asyncio.sleep(2**attempt)
                    # Re-resolve from the source URL so expired signed CDN links are refreshed.
                    response = await open_download(session, source["url"], credentials, received)
                    if response.status != HTTPStatus.PARTIAL_CONTENT:
                        await asyncio.to_thread(output.seek, 0)
                        await asyncio.to_thread(output.truncate)
                        received = 0
                    elif not response.headers.get("Content-Range", "").startswith(
                        f"bytes {received}-"
                    ):
                        raise ModelDownloadError("Model host resumed at the wrong offset")
        if total is not None and received != total:
            raise ModelDownloadError(
                "Model download was truncated; the incomplete file was discarded"
            )
        await report(
            {"phase": "verifying", "bytes": received, "total": total, "filename": filename}
        )
        await asyncio.to_thread(verify_weight_file, temporary)
        if target.exists():
            raise ModelDownloadError(
                "That filename was installed by another download; choose another filename"
            )
        temporary.replace(target)
        receipt = {
            "path": f"{request['folder']}/{filename}",
            "filename": filename,
            "folder": request["folder"],
            "bytes": received,
            "source": urlunsplit(urlsplit(request["url"])._replace(query="", fragment="")),
            **{
                key: source[key]
                for key in ("version_id", "file_id", "base_model", "trained_words")
                if key in source
            },
        }
        receipt_dir = root / ".bridge-downloads"
        receipt_dir.mkdir(exist_ok=True)
        (receipt_dir / f"{hashlib.sha256(receipt['path'].encode()).hexdigest()}.json").write_text(
            json.dumps(receipt)
        )
        return receipt
    finally:
        response.release()
        if temporary:
            temporary.unlink(missing_ok=True)
