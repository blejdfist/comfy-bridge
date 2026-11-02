"""ComfyUI-compatible local HTTP/WebSocket surface for the bundled editor."""

import asyncio
import io
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import comfyui_frontend_package
from aiohttp import web
from PIL import Image

from .bridge import Bridge
from .config import DEPLOYMENT_ID, FRONTEND_VERSION
from .helpers.paths import join_subfolder, safe_path
from .model_downloads import MODEL_FOLDERS
from .store import ACTIVE, TERMINAL, public_job
from .workflow import LOAD_IMAGE_NODES
from .workflow_files import file_info, user_file, user_relative, workflow_templates

log = logging.getLogger(__name__)
BRIDGE = web.AppKey("bridge", Bridge)
STATIC = Path(comfyui_frontend_package.__file__).parent / "static"


@web.middleware
async def guard(request, handler):
    # The local API can spend money on a GPU: prevent drive-by requests from other sites.
    origin = request.headers.get("Origin")
    if request.headers.get("Sec-Fetch-Site") == "cross-site" or (
        origin and urlsplit(origin).netloc != request.host
    ):
        raise web.HTTPForbidden(text="Cross-origin access is disabled")
    host = request.url.host
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise web.HTTPForbidden(text="Only localhost access is supported")
    try:
        return await handler(request)
    except (ValueError, KeyError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def websocket(request, bridge):
    sid = request.query.get("clientId") or str(uuid.uuid4())
    socket = web.WebSocketResponse(heartbeat=30, max_msg_size=1024 * 1024)
    await socket.prepare(request)
    bridge.sockets.setdefault(sid, set()).add(socket)
    try:
        await socket.send_json({"type": "status", "data": {"status": bridge.status(), "sid": sid}})
        async for message in socket:
            if message.type == aiohttp.WSMsgType.TEXT:
                data = message.json()
                if data.get("type") == "feature_flags":
                    await socket.send_json({"type": "feature_flags", "data": {}})
    finally:
        bridge.sockets.get(sid, set()).discard(socket)
    return socket


async def upload(request, bridge):
    reader = await request.multipart()
    image, filename, fields = None, None, {}
    while part := await reader.next():
        if part.name == "image":
            filename = part.filename
            image = await part.read(decode=False)
        else:
            fields[part.name] = await part.text()
    if not filename or image is None:
        raise ValueError("An image file is required")
    if fields.get("type", "input") != "input":
        raise ValueError("Uploads must have type=input")
    # Decode once so invalid image uploads fail locally rather than during paid execution.
    await asyncio.to_thread(lambda: Image.open(io.BytesIO(image)).verify())
    subfolder = fields.get("subfolder", "")
    root = bridge.root / "input"
    target = safe_path(root, join_subfolder(subfolder, filename))
    target.parent.mkdir(parents=True, exist_ok=True)
    if fields.get("overwrite", "false").lower() not in {"true", "1"}:
        index, original = 1, target
        while target.exists():
            target = original.with_stem(f"{original.stem} ({index})")
            index += 1
    target.write_bytes(image)
    return web.json_response(
        {
            "name": target.name,
            "subfolder": target.parent.relative_to(root).as_posix(),
            "type": "input",
        }
    )


async def userdata(request, bridge, path):
    root = bridge.root / "userdata"
    if path in {"/userdata", "/v2/userdata"}:
        relative = request.query.get("dir", "")
        directory = user_file(bridge, relative)
        if not directory.is_dir():
            return web.json_response([])
        recurse = request.query.get("recurse") == "true" or path.startswith("/v2/")
        files = sorted(directory.rglob("*") if recurse else directory.iterdir())
        if not relative and recurse:
            files = [p for p in files if not p.is_relative_to(root / "workflows")]
            files += [bridge.workflows_root, *sorted(bridge.workflows_root.rglob("*"))]
        files = [p for p in files if user_file(bridge, user_relative(bridge, p)).exists()]
        if path.startswith("/v2/"):
            return web.json_response(
                [
                    {
                        "name": p.name,
                        "path": user_relative(bridge, p),
                        "type": "directory" if p.is_dir() else "file",
                        **(
                            {"size": p.stat().st_size, "modified": p.stat().st_mtime}
                            if p.is_file()
                            else {}
                        ),
                    }
                    for p in files
                ]
            )
        if request.query.get("full_info") == "true":
            return web.json_response([file_info(p, directory) for p in files if p.is_file()])
        names = [p.relative_to(directory).as_posix() for p in files if p.is_file()]
        if request.query.get("split") == "true":
            return web.json_response([[name, *name.split("/")] for name in names])
        return web.json_response(names)
    relative = path.removeprefix("/userdata/")
    if "/move/" in relative and request.method == "POST":
        source, destination = relative.split("/move/", 1)
        target = user_file(bridge, destination)
        if target.exists() and request.query.get("overwrite", "false") != "true":
            raise web.HTTPConflict()
        target.parent.mkdir(parents=True, exist_ok=True)
        user_file(bridge, source).rename(target)
        return web.json_response(destination)
    target = user_file(bridge, relative)
    if request.method == "GET":
        if not target.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(target)
    if request.method == "DELETE":
        target.unlink(missing_ok=True)
        return web.Response(status=204)
    if request.method == "POST":
        if target.exists() and request.query.get("overwrite") == "false":
            raise web.HTTPConflict()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(await request.read())
        temporary.replace(target)
        value = (
            {**file_info(target, target.parent), "path": relative}
            if request.query.get("full_info") == "true"
            else relative
        )
        return web.json_response(value)
    raise web.HTTPMethodNotAllowed(request.method, ["GET", "POST", "DELETE"])


async def jobs_api(request, bridge, path):
    suffix = path.removeprefix("/jobs")
    if request.method == "POST":
        if suffix == "/cancel":
            ids = (await request.json()).get("job_ids", [])
        elif suffix.endswith("/cancel"):
            ids = [suffix.strip("/").removesuffix("/cancel")]
        else:
            raise web.HTTPNotFound()
        results = [await bridge.cancel(job_id) for job_id in ids]
        return web.json_response({"cancelled": results[0] if len(results) == 1 else results})
    if suffix:
        job = bridge.store.get(suffix.strip("/"))
        if not job:
            raise web.HTTPNotFound()
        return web.json_response(public_job(job, include_outputs=True))
    items = [public_job(j) for j in bridge.store.all()]
    if statuses := request.query.get("status"):
        items = [j for j in items if j["status"] in statuses.split(",")]
    if workflow_id := request.query.get("workflow_id"):
        items = [j for j in items if j.get("workflow_id") == workflow_id]
    items.sort(key=lambda j: j["create_time"], reverse=request.query.get("sort_order") != "asc")
    offset = max(0, int(request.query.get("offset", 0)))
    limit = max(1, int(request.query.get("limit", 100)))
    return web.json_response(
        {
            "jobs": items[offset : offset + limit],
            "pagination": {
                "offset": offset,
                "limit": limit,
                "total": len(items),
                "has_more": offset + limit < len(items),
            },
        }
    )


async def dispatch(request):
    bridge = request.app[BRIDGE]
    original = request.path
    path = original.removeprefix("/api") if original.startswith("/api/") else original
    method = request.method
    if path == "/bridge/models/catalog":
        routes = bridge.snapshot["routes"] if bridge.snapshot else {}
        return web.json_response(
            {
                "folders": list(MODEL_FOLDERS),
                "models": {folder: routes.get(f"/models/{folder}", []) for folder in MODEL_FOLDERS},
                "custom_nodes": bridge.snapshot.get("custom_nodes") if bridge.snapshot else None,
            }
        )
    if path == "/bridge/models/refresh" and method == "POST":
        mode = await bridge.refresh_models(full=request.query.get("full") == "1")
        return web.json_response({"ready": True, "mode": mode})
    if path == "/bridge/model-downloads" and method == "POST":
        download_id = bridge.enqueue_model_download(await request.json())
        return web.json_response({"id": download_id}, status=202)
    if path.startswith("/bridge/model-downloads/"):
        item = bridge.store.model_download(path.removeprefix("/bridge/model-downloads/"))
        if not item:
            raise web.HTTPNotFound()
        return web.json_response(item)
    if path == "/extensions":
        return web.json_response(bridge.snapshot["routes"].get(path, []) if bridge.snapshot else [])
    if path == "/ws":
        return await websocket(request, bridge)
    if path == "/bridge/status":
        return web.json_response(
            {
                "ready": bool(bridge.snapshot),
                "deployment_id": DEPLOYMENT_ID,
                "queue": bridge.status(),
                "last_error": bridge.last_error,
                "frontend_version": FRONTEND_VERSION,
            }
        )
    if path == "/user.css":
        target = user_file(bridge, "user.css")
        return (
            web.FileResponse(target)
            if target.exists()
            else web.Response(text="", content_type="text/css")
        )
    if path == "/i18n":
        return web.json_response({})
    if path == "/experiment/models":
        return web.json_response([])
    if path == "/templates/index_logo.json":
        return web.json_response({})
    if path.startswith("/templates/index"):
        return web.json_response(
            [
                {
                    "moduleName": "default",
                    "title": "Workflows",
                    "type": "image",
                    "isEssential": True,
                    "templates": workflow_templates(bridge),
                }
            ]
        )
    if path.startswith("/templates/") and path.endswith(".json"):
        name = path.removeprefix("/templates/").removesuffix(".json")
        if name not in {item["name"] for item in workflow_templates(bridge)}:
            raise web.HTTPNotFound()
        return web.FileResponse(safe_path(bridge.workflows_root, name + ".json"))
    if path == "/prompt":
        if method == "GET":
            return web.json_response(bridge.status())
        try:
            job, errors = bridge.enqueue(await request.json())
        except RuntimeError as exc:
            print(f"Rejected job: {exc}", flush=True)
            return web.json_response({"error": str(exc)}, status=503)
        except ValueError as exc:
            print(f"Rejected job: {exc}", flush=True)
            raise
        if errors:
            message = next(iter(errors.values()))["errors"][0]["message"]
            print(f"Rejected job: {message}", flush=True)
            return web.json_response(
                {
                    "error": {
                        "type": "prompt_validation_failed",
                        "message": message,
                    },
                    "node_errors": errors,
                },
                status=400,
            )
        await bridge.emit_status()
        return web.json_response(
            {"prompt_id": job["id"], "number": job["number"], "node_errors": {}}
        )
    if path == "/queue":
        if method == "GET":
            return web.json_response(bridge.store.queue())
        body = await request.json()
        ids = (
            [j["id"] for j in bridge.store.all() if j["state"] == "pending"]
            if body.get("clear")
            else body.get("delete", [])
        )
        for job_id in ids:
            job = bridge.store.get(job_id)
            if job and job["state"] == "pending":
                await bridge.cancel(job_id)
        return web.json_response({})
    if path == "/interrupt":
        body = await request.json() if request.can_read_body else {}
        for job in bridge.store.all():
            if job["state"] in ACTIVE and (
                not body.get("prompt_id") or body["prompt_id"] == job["id"]
            ):
                await bridge.cancel(job["id"])
        return web.json_response({})
    if path == "/history" or path.startswith("/history/"):
        if method == "GET":
            history = bridge.store.history(
                path.removeprefix("/history/") if path != "/history" else None
            )
            if max_items := request.query.get("max_items"):
                history = dict(list(history.items())[-max(0, int(max_items)) :])
            return web.json_response(history)
        body = await request.json()
        ids = (
            [j["id"] for j in bridge.store.all() if j["state"] in TERMINAL]
            if body.get("clear")
            else body.get("delete", [])
        )
        for job_id in ids:
            bridge.store.delete(job_id)
        return web.json_response({})
    if path == "/jobs" or path.startswith("/jobs/"):
        return await jobs_api(request, bridge, path)
    if path.startswith("/userdata") or path == "/v2/userdata":
        return await userdata(request, bridge, path)
    if path.startswith("/settings"):
        target = user_file(bridge, "comfy.settings.json")
        settings = json.loads(target.read_text()) if target.exists() else {}
        key = path.removeprefix("/settings/")
        if method == "GET":
            return web.json_response(settings if path == "/settings" else settings.get(key))
        body = await request.json()
        if path == "/settings":
            settings.update(body)
        else:
            settings[key] = body
        target.write_text(json.dumps(settings))
        return web.json_response({})
    if path == "/users":
        return web.json_response({"storage": "server", "migrated": True})
    if path == "/upload/image":
        return await upload(request, bridge)
    if path == "/view":
        kind = request.query.get("type", "output")
        if kind not in {"input", "output", "temp"}:
            raise ValueError("Unknown file type")
        root = bridge.root / "input" if kind == "input" else bridge.output_root
        relative = join_subfolder(request.query.get("subfolder", ""), request.query["filename"])
        target = safe_path(root, relative)
        if kind != "input" and not target.is_file():
            # Keep images from older bridge versions accessible through saved history.
            target = safe_path(bridge.root / "output", relative)
        if not target.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(target)
    if path == "/free":
        return web.json_response({})  # No local models; the worker expires automatically.
    if path in {"/subgraphs", "/global_subgraphs"}:
        return web.json_response([])
    if path in {"/workflow_templates", "/node_replacements"}:
        return web.json_response({})
    if path == "/internal/folder_paths":
        routes = bridge.snapshot["routes"] if bridge.snapshot else {}
        return web.json_response(
            {k.removeprefix("/models/"): [] for k in routes if k.startswith("/models/")}
        )
    if path == "/object_info" or path.startswith("/object_info/"):
        if not bridge.snapshot:
            return web.json_response({"error": "Run uv run comfy-bridge sync first"}, status=503)
        schema = json.loads(json.dumps(bridge.snapshot["routes"]["/object_info"]))
        # Input widgets must list local uploads, not the provisioning container's files.
        images = [
            p.relative_to(bridge.root / "input").as_posix()
            for p in (bridge.root / "input").rglob("*")
            if p.is_file()
        ]
        for name in LOAD_IMAGE_NODES:
            if name in schema:
                schema[name]["input"]["required"]["image"][0] = images
        if path != "/object_info":
            name = path.removeprefix("/object_info/")
            schema = {name: schema[name]} if name in schema else {}
        return web.json_response(schema)
    if bridge.snapshot:
        if path in bridge.snapshot["routes"]:
            return web.json_response(bridge.snapshot["routes"][path])
        if path in bridge.snapshot.get("assets", {}):
            return web.Response(
                text=bridge.snapshot["assets"][path], content_type="text/javascript"
            )
    if path in {"/features", "/models", "/embeddings"}:
        return web.json_response({} if path == "/features" else [])
    if method == "GET" and not original.startswith("/api/"):
        target = safe_path(STATIC, original.lstrip("/") or "index.html")
        if target.is_file():
            return web.FileResponse(target, headers={"Cache-Control": "no-cache"})
    log.debug("Unsupported editor route: %s %s", method, original)
    raise web.HTTPNotFound()


def create_app(bridge: Bridge, run_queue=True):
    app = web.Application(middlewares=[guard], client_max_size=100 * 1024 * 1024)
    app[BRIDGE] = bridge
    app.router.add_route("*", "/{tail:.*}", dispatch)

    async def lifecycle(app):
        if run_queue:
            await bridge.start()
        yield
        await bridge.close()

    app.cleanup_ctx.append(lifecycle)
    mimetypes.add_type("text/javascript", ".js")
    return app
