"""Map the editor's virtual userdata paths onto the workflows source directory."""

import json
from pathlib import Path

from .helpers.paths import safe_path


def user_file(bridge, relative):
    # Validate the virtual path before mapping the workflows mount.
    safe_path(bridge.root / "userdata", relative)
    parts = Path(relative).parts
    if parts and parts[0] == "workflows":
        return safe_path(bridge.workflows_root, "/".join(parts[1:]))
    return safe_path(bridge.root / "userdata", relative)


def user_relative(bridge, path):
    if path.is_relative_to(bridge.workflows_root):
        return (Path("workflows") / path.relative_to(bridge.workflows_root)).as_posix()
    return path.relative_to(bridge.root / "userdata").as_posix()


def workflow_templates(bridge):
    templates = []
    for path in sorted(bridge.workflows_root.rglob("*.json")):
        # API payloads are downloadable user files, but cannot open as editor graphs.
        if not path.resolve().is_relative_to(bridge.workflows_root):
            continue
        try:
            graph = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
            name = path.relative_to(bridge.workflows_root).with_suffix("").as_posix()
            templates.append(
                {
                    "name": name,
                    "title": path.stem.replace("_", " "),
                    "openSource": True,
                    "description": name,
                    "tags": [],
                }
            )
    return templates


def file_info(path, root):
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "created": stat.st_ctime,
    }
