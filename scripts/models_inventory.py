"""List models in a locally running ComfyUI and models.yaml, and which side has each.

    uv run python scripts/models_inventory.py
    uv run python scripts/models_inventory.py | grep " local "

Entries are matched on folder and relative path, so a nested local file corresponds to a
models.yaml filename such as `qwen-image/qwen_image_vae.safetensors`.
"""

import argparse
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path, PureWindowsPath

from comfyui_modal.model_downloads import MODEL_FOLDERS
from comfyui_modal.model_manifest import load_yaml, manifest_entries

ICONS = {"both": "✅", "local": "💻", "remote": "📄"}
LEGEND = {"both": "in both", "local": "local ComfyUI only", "remote": "models.yaml only"}


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def relative_name(name):
    """ComfyUI lists nested files as 'Sub\\file' on Windows and 'Sub/file' elsewhere."""
    return PureWindowsPath(name).as_posix()


def inventory(local, manifest):
    """Rows of (id, status, folder, filename, local name) over both sides, sorted."""
    names = {(folder, relative_name(n)): n for folder, files in local.items() for n in files}
    rows = []
    keys = sorted(names.keys() | manifest, key=lambda k: (k[0], names.get(k, k[1]).lower()))
    for index, key in enumerate(keys, start=1):
        if key in names and key in manifest:
            status = "both"
        else:
            status = "local" if key in names else "remote"
        rows.append((index, status, *key, names.get(key)))
    return rows


def display_width(text):
    """Terminal columns: emoji and other wide characters take two."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def print_table(rows):
    headers = ("ID", "STATUS", "FOLDER", "PATH")
    table = [
        headers,
        *((row[0], f"{ICONS[row[1]]} {row[1]}", row[2], row[4] or row[3]) for row in rows),
    ]
    width = [max(display_width(str(row[i])) for row in table) for i in range(4)]
    for row in table:
        cells = (
            str(v) + " " * (w - display_width(str(v))) for v, w in zip(row, width, strict=True)
        )
        print("  ".join(cells).rstrip())
    print("\n" + "   ".join(f"{ICONS[s]} {s}: {text}" for s, text in LEGEND.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("models.yaml"))
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Emoji fail in cp1252 when output is piped.

    try:
        local = {f: get_json(f"{args.comfy_url}/models/{f}") for f in MODEL_FOLDERS}
    except (OSError, ValueError) as exc:
        sys.exit(f"No ComfyUI model API at {args.comfy_url} ({exc})")
    manifest = set()
    if args.manifest.exists():
        entries = manifest_entries(load_yaml(args.manifest))
        manifest = {(request["folder"], request["filename"]) for request, _ in entries}
    print_table(inventory(local, manifest))


if __name__ == "__main__":
    main()
