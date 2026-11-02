"""Local editor, remote execution, model management, and deployment commands."""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import aiohttp
from aiohttp import web

from .backend import ModalBackend
from .bridge import Bridge
from .bridge_client import run_model_command
from .helpers.comfy_history import last_execution_error

DEFAULT_DATA_DIR = Path(".bridge")
DEFAULT_PORT = 8288
DEFAULT_BRIDGE_URL = f"http://127.0.0.1:{DEFAULT_PORT}"


def summarize_error(error):
    """Extract the useful message from persisted remote exception records."""
    if not error:
        return ""
    try:
        record = json.loads(error)
    except (ValueError, TypeError):
        record = error
    if isinstance(record, dict):
        failure = last_execution_error(record.get("messages", []))
        if failure is not None:
            record = failure.get("exception_message") or error
        else:
            record = record.get("exception_message") or record.get("error") or error
    lines = str(record).strip().splitlines()
    message = lines[-1] if lines and lines[0].startswith("Traceback") else " ".join(lines)
    return message if len(message) <= 240 else message[:237] + "..."


def deploy(args):
    from .deployment_config import load_deployment_config

    config_path = args.config.resolve()
    settings = load_deployment_config(config_path)
    print(
        f"Deploying {settings['app_name']}: {settings['gpu_worker']['gpu']} worker with {settings['gpu_worker']['cpus']} CPUs / "
        f"{settings['gpu_worker']['memory_mib']} MiB RAM; CPU-only functions with "
        f"{settings['cpu_only']['cpus']} CPUs / {settings['cpu_only']['memory_mib']} MiB RAM.",
        flush=True,
    )
    environment = os.environ.copy()
    environment["COMFY_BRIDGE_CONFIG"] = str(config_path)
    return subprocess.run(
        [sys.executable, "-m", "modal", "deploy", "-m", "comfyui_modal.modal_app"],
        env=environment,
        check=False,
    ).returncode


def main():
    parser = argparse.ArgumentParser(description="Local ComfyUI editor backed by Modal GPU jobs")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Public Modal app and resource configuration",
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    deployment = sub.add_parser(
        "deploy", help="Deploy all Modal functions using public YAML settings"
    )
    deployment.add_argument("--config", type=Path, default=argparse.SUPPRESS)
    teardown_parser = sub.add_parser(
        "teardown", help="Stop the Modal app and delete its Volumes and state; keep local files"
    )
    teardown_parser.add_argument("--config", type=Path, default=argparse.SUPPRESS)
    teardown_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    cleanup_parser = sub.add_parser(
        "cleanup", help="Clear all files from the remote job Volume; keep local files and models"
    )
    cleanup_parser.add_argument(
        "--dry-run", action="store_true", help="List output counts and bytes without deleting"
    )
    serve = sub.add_parser("serve", help="Serve the editor and process the durable local queue")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--output-dir", type=Path, default=Path("output"))
    status = sub.add_parser("status", help="Show saved local job history, without polling Modal")
    status.add_argument("--details", action="store_true", help="Show full recorded errors")
    info = sub.add_parser("info", help="Show cached model and custom-node inventory; no live scan")
    info.set_defaults(model_command="info")
    info.add_argument("--bridge-url", default=DEFAULT_BRIDGE_URL)
    sync = sub.add_parser(
        "sync", help="Install models from YAML; load credentials only when needed"
    )
    sync.add_argument("--manifest", type=Path, default=Path("models.yaml"))
    sync.add_argument("--credentials", type=Path, default=Path("credentials.yaml"))
    sync.add_argument("--bridge-url", default=DEFAULT_BRIDGE_URL)
    sync.add_argument(
        "--full", action="store_true", help="Rebuild all editor metadata after installing models"
    )
    sync.add_argument(
        "--rm",
        action="store_true",
        help="After all downloads succeed, delete remote models not listed in the manifest",
    )
    sync.add_argument("-y", "--yes", action="store_true", help="Skip the --rm confirmation")
    args = parser.parse_args()
    os.environ["COMFY_BRIDGE_CONFIG"] = str(args.config.resolve())
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "deploy":
            raise SystemExit(deploy(args))
        elif args.command == "teardown":
            from .teardown import teardown

            raise SystemExit(teardown(args))
        elif args.command == "cleanup":
            from .cleanup import cleanup

            asyncio.run(cleanup(args, ModalBackend()))
        elif args.command == "sync":
            from .model_manifest import sync_manifest

            asyncio.run(sync_manifest(args))
        elif args.command == "info":
            asyncio.run(run_model_command(args))
        elif args.command == "status":
            from .store import Store

            store = Store(args.data_dir / "queue.sqlite3")
            try:
                for job in store.all():
                    error = job["error"] or ""
                    print(
                        job["id"], job["state"], error if args.details else summarize_error(error)
                    )
            finally:
                store.close()
        else:
            # SQLite's exclusive lock is cross-platform and released when the process exits.
            lock = sqlite3.connect(args.data_dir / "bridge.lock", timeout=0, isolation_level=None)
            with closing(lock):
                try:
                    lock.execute("BEGIN EXCLUSIVE")
                except sqlite3.OperationalError:
                    raise RuntimeError(
                        "Another bridge process is using this data directory"
                    ) from None
                from .server import create_app

                bridge = Bridge(args.data_dir, ModalBackend(), output_dir=args.output_dir)
                if not bridge.snapshot:
                    print(
                        "Editor metadata missing. In another terminal, run `uv run comfy-bridge sync` before queueing."
                    )
                web.run_app(create_app(bridge), host="127.0.0.1", port=args.port)
    except (RuntimeError, TimeoutError, aiohttp.ClientError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
