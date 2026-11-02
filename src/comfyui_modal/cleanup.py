"""Clear the remote job Volume without touching local files or model weights."""

import sqlite3

from .deployment_config import modal_names


def check_cleanup_idle(data_dir):
    database = data_dir.resolve() / "queue.sqlite3"
    if not database.is_file():
        return
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        active = connection.execute(
            "SELECT count(*) FROM jobs WHERE state NOT IN ('completed','failed','cancelled')"
        ).fetchone()[0]
    finally:
        connection.close()
    if active:
        raise RuntimeError("Cannot clear the job Volume while the local queue has unfinished jobs")


async def cleanup(args, backend):
    if not args.dry_run:
        check_cleanup_idle(args.data_dir)
    print(
        f"{'Inspecting' if args.dry_run else 'Clearing'} Modal job Volume {modal_names()['job_volume']}…",
        flush=True,
    )
    result = await backend.cleanup_jobs_volume(dry_run=args.dry_run)
    action = "Would remove" if args.dry_run else "Removed"
    print(
        f"{action} {result['files']} files ({result['bytes'] / 1024**2:.2f} MiB) "
        "from the entire job Volume, including inputs, outputs, and result manifests. "
        "Local files, history, and model weights are unchanged."
    )
