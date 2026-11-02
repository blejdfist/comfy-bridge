"""Stop the Modal app and delete its remote storage; local files are never touched."""

import asyncio
import subprocess
import sys

import modal

from .deployment_config import modal_names
from .helpers.modal_volume import list_volume_entries


def format_bytes(size):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


async def volume_usage(name):
    from modal.volume import FileEntryType

    entries = await list_volume_entries(modal.Volume.from_name(name))
    if entries is None:
        return None
    files = [entry for entry in entries if entry.type == FileEntryType.FILE]
    return len(files), sum(entry.size for entry in files)


async def inventory(names):
    """One row per remote resource; `found` is False when it is already gone."""
    try:
        await modal.App.lookup.aio(names["app_name"])
        app_found = True
    except modal.exception.NotFoundError:
        app_found = False
    try:
        entries = await modal.Dict.from_name(names["state_dict"]).len.aio()
    except modal.exception.NotFoundError:
        entries = None
    secrets = sorted(
        secret.name
        for secret in await modal.Secret.objects.list.aio()
        if secret.name.startswith(names["app_name"] + "-")
    )
    models, jobs = await asyncio.gather(
        volume_usage(names["model_volume"]), volume_usage(names["job_volume"])
    )
    rows = [("stop", "App", names["app_name"], app_found, "deployed functions and containers", None)]
    for name, usage, purpose in (
        (names["model_volume"], models, "model weights"),
        (names["job_volume"], jobs, "job inputs, outputs and manifests"),
    ):
        rows.append(("delete", "Volume", name, usage is not None, purpose, usage))
    rows.append(("delete", "Dict", names["state_dict"], entries is not None, "worker state", entries))
    rows += [("delete", "Secret", name, True, "download credentials", None) for name in secrets]
    return rows


def describe(rows):
    """Render the plan as an aligned table with a total of the data deleted."""
    lines = []
    total_files = total_bytes = 0
    for action, kind, name, found, purpose, usage in rows:
        if not found:
            size = "not found, skipped"
        elif kind == "Volume":
            files, size_bytes = usage
            total_files, total_bytes = total_files + files, total_bytes + size_bytes
            size = f"{format_bytes(size_bytes):>10}  {files:>6,} files"
        elif kind == "Dict":
            size = f"{usage:>10,}  entries"
        elif kind == "Secret":
            size = ""
        else:
            size = "deployed"
        verb = f"Will {action}" if found else "Skip"
        lines.append(f"  {verb:<12} {kind:<7} {name:<34} {size:<26} {purpose}")
    lines += [
        "",
        f"  Data deleted: {format_bytes(total_bytes)} in {total_files:,} files",
        "  Kept:         local files (data dir, output/, credentials.yaml)",
    ]
    return "\n".join(lines)


def teardown(args):
    from .deployment_config import load_deployment_config

    names = modal_names(load_deployment_config(args.config.resolve()))
    print(f"Inspecting Modal resources for {names['app_name']}…", flush=True)
    rows = asyncio.run(inventory(names))
    print(f"\nTeardown plan for {names['app_name']}\n\n{describe(rows)}\n")
    if not any(row[3] for row in rows):
        print("Nothing to tear down.")
        return 0
    print("This permanently deletes the data above and cannot be undone.")
    if not args.yes and input(f"Type {names['app_name']} to confirm: ") != names["app_name"]:
        print("Aborted; nothing was changed.")
        return 1
    modal_cli = [sys.executable, "-m", "modal"]
    failed = 0
    for _, kind, name, found, _, _ in rows:
        if not found:
            continue
        command = ["app", "stop", "-y"] if kind == "App" else [kind.lower(), "delete", "-y"]
        failed |= subprocess.run([*modal_cli, *command, name], check=False).returncode
    print("Teardown failed; rerun to retry the rest." if failed else "Teardown complete.")
    return 1 if failed else 0
