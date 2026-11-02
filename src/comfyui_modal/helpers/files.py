"""File writes shared by the bridge and the remote worker."""

import json
from pathlib import Path


def atomic_write_json(path: Path, value) -> None:
    """Replace the file in one step so readers never see a partial document."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)
