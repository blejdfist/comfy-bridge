"""Public, pinned custom-node manifest used only while building the image."""

import re
import shlex
from pathlib import Path

from .helpers.yaml_io import read_yaml


def load_custom_nodes(path=Path("custom_nodes.yaml")):
    path = Path(path)
    if not path.exists():
        return []
    document = read_yaml(path, "Cannot read custom_nodes.yaml")
    if (
        not isinstance(document, dict)
        or set(document) != {"version", "nodes"}
        or type(document["version"]) is not int
        or document["version"] != 1
        or not isinstance(document["nodes"], list)
    ):
        raise RuntimeError("custom_nodes.yaml requires version: 1 and a nodes list")
    names = set()
    for node in document["nodes"]:
        if isinstance(node, dict) and (
            not isinstance(node.get("revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", node["revision"])
        ):
            raise RuntimeError(
                f"Custom node {node.get('name', '<unnamed>')!r}: revision must be a full "
                "40-character commit SHA; branch names such as 'main' are not supported. "
                "Copy a commit SHA from the repository's commit history."
            )
        if (
            not isinstance(node, dict)
            or set(node) != {"name", "repository", "revision"}
            or not isinstance(node["name"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", node["name"])
            or not isinstance(node["repository"], str)
            or not re.fullmatch(
                r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", node["repository"]
            )
            or node["name"] in names
        ):
            raise RuntimeError(
                "Each custom node needs a unique safe name, public GitHub HTTPS repository, and full commit SHA revision"
            )
        names.add(node["name"])
    return document["nodes"]


def install_commands(node):
    directory = shlex.quote("/opt/ComfyUI/custom_nodes/" + node["name"])
    return [
        f"git init {directory}",
        f"git -C {directory} remote add origin {shlex.quote(node['repository'])}",
        f"git -C {directory} fetch --depth 1 origin {node['revision']}",
        f"git -C {directory} checkout --detach FETCH_HEAD",
    ]
