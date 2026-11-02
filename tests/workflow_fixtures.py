"""Synthetic API-format payload for fake-backend tests; independent of workflows/ files."""

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "api_workflow.json"


def api_workflow(text, seed=42, width=1024, height=1024):
    graph = json.loads(FIXTURE.read_text())
    graph["6"]["inputs"]["text"] = text
    graph["3"]["inputs"]["seed"] = seed
    graph["5"]["inputs"].update(width=width, height=height)
    return graph
