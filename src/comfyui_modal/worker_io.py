"""Output persistence, isolated from Modal for integration tests."""

import copy
import re
import shutil
from pathlib import Path, PurePosixPath

from .helpers.comfy_history import iter_output_items
from .helpers.paths import join_subfolder, safe_path


def localize_artifacts(result, output_root, reserved_paths=()):
    """Persistable flat local names, independent of remote job namespaces."""
    result = copy.deepcopy(result)
    occupied = set(reserved_paths)
    renamed = {}
    for artifact in result["artifacts"]:
        original = artifact["local_path"]
        safe_path(output_root, original)
        if original in renamed:
            artifact["local_path"] = renamed[original]
            continue
        name = Path(original).name
        candidate = name
        match = re.fullmatch(r"(.*)_([0-9]+)_", Path(name).stem)
        prefix = match[1] if match else Path(name).stem
        counter = int(match[2]) if match else 0
        width = max(5, len(match[2])) if match else 5
        while candidate in occupied or safe_path(output_root, candidate).exists():
            counter += 1
            candidate = f"{prefix}_{counter:0{width}d}_{Path(name).suffix}"
        occupied.add(candidate)
        renamed[original] = candidate
        artifact["local_path"] = candidate
    for item in iter_output_items(result["history"]):
        original = join_subfolder(item.get("subfolder", ""), item["filename"])
        if original in renamed:
            item["filename"] = renamed[original]
            item["subfolder"] = ""
    result["local_output_layout"] = 1
    return result


def collect_artifacts(
    history: dict,
    job_dir: Path,
    output_root=Path("/tmp/comfy-output"),
    temp_root=Path("/opt/ComfyUI/temp"),
) -> list[dict]:
    artifacts = []
    seen = set()
    for item in iter_output_items(history):
        kind = item.get("type", "output")
        root = output_root if kind == "output" else temp_root if kind == "temp" else None
        if root is None:
            continue
        relative = join_subfolder(item.get("subfolder", ""), item["filename"])
        source = safe_path(root, relative)
        # A unique local namespace makes filenames safe across jobs and worker restarts.
        subfolder = f"{job_dir.name}/{kind}"
        filename = source.name
        target = safe_path(job_dir / "artifacts", f"{kind}/{relative}")
        key = (kind, relative)
        if key not in seen:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            artifacts.append(
                {
                    # Modal mounts are symlinks into /__modal/volumes. safe_path
                    # resolves those links, so compare against the resolved job
                    # root and construct the Volume-relative path explicitly.
                    "remote_path": f"{job_dir.name}/{target.relative_to(job_dir.resolve()).as_posix()}",
                    "local_path": f"{subfolder}/{relative}",
                    "type": kind,
                }
            )
            seen.add(key)
        item["filename"] = filename
        nested = str(PurePosixPath(relative).parent)
        item["subfolder"] = subfolder + (f"/{nested}" if nested != "." else "")
        item["type"] = kind
    return artifacts
