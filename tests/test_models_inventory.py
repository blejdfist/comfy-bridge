import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "models_inventory.py"
spec = importlib.util.spec_from_file_location("models_inventory", SCRIPT)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_inventory_matches_nested_local_names_by_relative_path():
    rows = inventory.inventory(
        {"loras": ["Sub\\a.safetensors", "x.safetensors"]},
        {("loras", "Sub/a.safetensors"), ("loras", "a.safetensors"), ("vae", "b.safetensors")},
    )
    assert [(r[0], r[1], r[3]) for r in rows] == [
        (1, "remote", "a.safetensors"),
        (2, "both", "Sub/a.safetensors"),
        (3, "local", "x.safetensors"),
        (4, "remote", "b.safetensors"),
    ]
