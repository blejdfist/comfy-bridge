"""Client for the running bridge's model-manager HTTP API, with download progress output."""

import asyncio
import json
import time

import aiohttp
from tqdm import tqdm

from .helpers.http import get_json, json_or_raise


class DownloadProgress:
    """One tqdm bar per download; several can stack via ``position``.

    Finished bars clear their row (``leave=False``) and print a result line above the
    remaining bars. A kept bar at position 0 prints a newline on close, shifting the
    rows of every other bar so they overwrite each other.
    """

    def __init__(self, filename=None, position=0):
        self.filename = filename or "model"
        self.state = "pending"
        self.bar = tqdm(
            total=None,
            desc=self.filename,
            position=position,
            leave=False,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        )

    def update(self, progress, state):
        filename = progress.get("filename") or self.filename
        # Render remote filenames as a single line, including on interactive terminals.
        self.bar.set_description_str(
            "".join(char if char.isprintable() else "?" for char in filename), refresh=False
        )
        total = progress.get("total")
        if total and total > 0:
            self.bar.total = total
        self.bar.n = progress.get("bytes", self.bar.n)
        self.state = progress.get("phase", state)
        self.bar.set_postfix_str(self.state)

    def __enter__(self):
        return self

    def __exit__(self, error_type, *_):
        state = "failed" if error_type else self.state
        size = tqdm.format_sizeof(self.bar.n, "B", 1024) if self.bar.n else ""
        self.bar.close()
        tqdm.write(f"{self.bar.desc}: {state} {size}".rstrip())


async def run_model_command(args):
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if args.model_command == "download":
            body = {
                key: getattr(args, key)
                for key in ("url", "folder", "filename", "secret", "file_id")
            }
            body["refresh_catalog"] = getattr(args, "refresh_catalog", True)
            body["skip_existing"] = getattr(args, "skip_existing", False)
            async with session.post(
                args.bridge_url + "/bridge/model-downloads", json=body
            ) as response:
                result = await json_or_raise(response, 202)
            download_id = result["id"]
            deadline = time.monotonic() + 3700
            with DownloadProgress(args.filename, getattr(args, "position", 0)) as display:
                while time.monotonic() < deadline:
                    item = await get_json(
                        session, args.bridge_url + "/bridge/model-downloads/" + download_id
                    )
                    if item["state"] == "completed":
                        display.update(
                            {},
                            "skipped" if (item.get("result") or {}).get("skipped") else "completed",
                        )
                        break
                    if item["state"] in {"failed", "cancelled", "unknown"}:
                        raise RuntimeError(item["error"] or item["state"])
                    display.update(item.get("progress") or {}, item["state"])
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError(
                        "Stopped waiting; the download remains visible in the model manager"
                    )
            if body["refresh_catalog"]:
                print(json.dumps(item["result"], indent=2))
                print("Model lists refreshed. Refresh the editor to see the new file.")
            return item["result"]
        elif args.model_command == "refresh":
            params = {"full": "1"} if getattr(args, "full", False) else {}
            async with session.post(
                args.bridge_url + "/bridge/models/refresh", params=params
            ) as response:
                await json_or_raise(response, 200)
                print("Model catalog refreshed. Refresh the editor's node definitions.")
        else:
            catalog = await get_json(session, args.bridge_url + "/bridge/models/catalog")
            print(json.dumps(catalog, indent=2))
