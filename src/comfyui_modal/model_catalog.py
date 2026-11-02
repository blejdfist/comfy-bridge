"""Refresh filename choices while retaining cached ComfyUI schemas and extensions."""

import copy
import time

from .helpers.deployment import require_current_deployment

FIELD_FOLDERS = {
    "ckpt_name": "checkpoints",
    "unet_name": "diffusion_models",
    "clip_name": "text_encoders",
    "clip_name1": "text_encoders",
    "clip_name2": "text_encoders",
    "clip_name3": "text_encoders",
    "clip_name4": "text_encoders",
    "vae_name": "vae",
    "lora_name": "loras",
    "control_net_name": "controlnet",
    "clip_vision": "clip_vision",
    "style_model_name": "style_models",
    "gligen_name": "gligen",
}
CLASS_FOLDERS = {
    "UpscaleModelLoader": "upscale_models",
    "CLIPVisionLoader": "clip_vision",
    "StyleModelLoader": "style_models",
    "HypernetworkLoader": "hypernetworks",
}


def apply_model_lists(snapshot, catalog):
    require_current_deployment(catalog)
    updated = copy.deepcopy(snapshot)
    routes = updated["routes"]
    folders = catalog["model_lists"]
    old_folders = {folder: set(routes.get(f"/models/{folder}", [])) for folder in folders}
    for class_name, node in routes["/object_info"].items():
        for section in node.get("input", {}).values():
            if not isinstance(section, dict):
                continue
            for field, spec in section.items():
                if not isinstance(spec, list) or not spec or not isinstance(spec[0], list):
                    continue
                choices = spec[0]
                if not all(isinstance(choice, str) for choice in choices):
                    continue
                folder = FIELD_FOLDERS.get(field)
                if folder is None and field == "model_name":
                    folder = CLASS_FOLDERS.get(class_name)
                if folder is None:
                    matches = [
                        name for name, files in old_folders.items() if files.intersection(choices)
                    ]
                    if len(matches) == 1:
                        folder = matches[0]
                if folder in folders:
                    extras = [choice for choice in choices if choice not in old_folders[folder]]
                    spec[0] = list(dict.fromkeys([*folders[folder], *extras]))
    routes["/models"] = list(folders)
    for folder, files in folders.items():
        routes[f"/models/{folder}"] = files
    routes["/embeddings"] = catalog.get("embeddings", [])
    updated["created_at"] = time.time()
    return updated
