import copy

import pytest

from comfyui_modal.config import DEPLOYMENT_ID
from comfyui_modal.model_catalog import apply_model_lists


def test_filename_refresh_updates_loaders_preserves_schema_and_removes_stale_files():
    snapshot = {
        "deployment_id": DEPLOYMENT_ID,
        "assets": {"extension.js": "cached script"},
        "routes": {
            "/models/loras": ["old.safetensors"],
            "/models/vae": ["old-vae.safetensors"],
            "/models/checkpoints": [],
            "/object_info": {
                "LoraLoader": {
                    "input": {"required": {"lora_name": [["old.safetensors"], {"tooltip": "keep"}]}}
                },
                "VAELoader": {
                    "input": {"required": {"vae_name": [["old-vae.safetensors", "pixel_space"]]}}
                },
                "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [[]]}}},
                "AnotherLoader": {"input": {"optional": {"style": [["old.safetensors"]]}}},
                "Sampler": {
                    "input": {"required": {"sampler": [["euler", "heun"]], "number": [[1, 2]]}}
                },
            },
        },
    }
    before = copy.deepcopy(snapshot)
    catalog = {
        "deployment_id": DEPLOYMENT_ID,
        "model_lists": {
            "loras": ["new.safetensors"],
            "vae": ["new-vae.safetensors"],
            "checkpoints": ["checkpoint.safetensors"],
        },
        "embeddings": ["embedding"],
    }
    updated = apply_model_lists(snapshot, catalog)
    nodes = updated["routes"]["/object_info"]
    assert nodes["LoraLoader"]["input"]["required"]["lora_name"] == [
        ["new.safetensors"],
        {"tooltip": "keep"},
    ]
    assert nodes["VAELoader"]["input"]["required"]["vae_name"][0] == [
        "new-vae.safetensors",
        "pixel_space",
    ]
    assert nodes["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0] == [
        "checkpoint.safetensors"
    ]
    assert nodes["AnotherLoader"]["input"]["optional"]["style"][0] == ["new.safetensors"]
    assert nodes["Sampler"] == snapshot["routes"]["/object_info"]["Sampler"]
    assert updated["assets"] == snapshot["assets"]
    assert updated["routes"]["/models/loras"] == ["new.safetensors"]
    assert updated["routes"]["/embeddings"] == ["embedding"]
    assert snapshot == before


def test_wrong_runtime_catalog_is_rejected():
    with pytest.raises(ValueError):
        apply_model_lists({}, {"deployment_id": "different"})
