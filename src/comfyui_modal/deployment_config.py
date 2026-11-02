"""Public, local deployment settings, separate from model credentials."""

import json
import math
import os
import re
from pathlib import Path

from .helpers.yaml_io import read_yaml

DEFAULT_GPU = "L40S"
DEFAULT_APP_NAME = "comfyui-krea2-bridge"
GPU_TYPES = {
    "T4",
    "L4",
    "A10",
    "L40S",
    "A100",
    "A100-40GB",
    "A100-80GB",
    "RTX-PRO-6000",
    "H100",
    "H100!",
    "H200",
    "B200",
    "B200+",
    "B300",
}


def load_deployment_config(path=Path("config.yaml")):
    """Resolve public container settings at deployment time."""
    document = read_yaml(path, "Cannot read deployment config.yaml", missing={"version": 1})
    if (
        not isinstance(document, dict)
        or set(document) - {"version", "app_name", "cpu_only", "gpu_worker"}
        or type(document.get("version")) is not int
        or document.get("version") != 1
    ):
        raise RuntimeError(
            "Deployment config.yaml requires version: 1 with cpu_only and/or gpu_worker settings; "
            "put the GPU setting under gpu_worker.gpu"
        )
    app_name = document.get("app_name", DEFAULT_APP_NAME)
    if (
        not isinstance(app_name, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,56}", app_name)
        or re.fullmatch(r"ap-[A-Za-z0-9]{22}", app_name)
    ):
        raise RuntimeError(
            "app_name must be 1–57 letters, digits, underscores or hyphens, starting with a letter or digit, and must not resemble a Modal App ID"
        )
    result = {"version": 1, "app_name": app_name}
    for name, memory in (("cpu_only", 8192), ("gpu_worker", 32768)):
        resources = document.get(name, {})
        allowed = {"cpus", "memory_mib", "gpu", "scaledown_window_seconds"} if name == "gpu_worker" else {"cpus", "memory_mib"}
        if not isinstance(resources, dict) or set(resources) - allowed:
            raise RuntimeError(f"Unsupported container settings in {name}")
        cpus = resources.get("cpus", 4)
        memory_mib = resources.get("memory_mib", memory)
        if type(cpus) not in {int, float} or not math.isfinite(cpus) or cpus < 0.125:
            raise RuntimeError(f"{name}.cpus must be a finite number of at least 0.125")
        if type(memory_mib) is not int or memory_mib < 1:
            raise RuntimeError(f"{name}.memory_mib must be a positive integer")
        result[name] = {"cpus": cpus, "memory_mib": memory_mib}
        if name == "gpu_worker":
            gpu = resources.get("gpu", DEFAULT_GPU)
            if not isinstance(gpu, str) or gpu not in GPU_TYPES:
                raise RuntimeError("Unsupported gpu_worker.gpu; choose a single Modal GPU type")
            result[name]["gpu"] = gpu
            idle = resources.get("scaledown_window_seconds", 60)
            if type(idle) is not int or not 2 <= idle <= 3600:
                raise RuntimeError("gpu_worker.scaledown_window_seconds must be an integer from 2 to 3600")
            result[name]["scaledown_window_seconds"] = idle
    return result


def load_runtime_config():
    """Remote containers use baked settings; local commands read public YAML."""
    if "COMFY_BRIDGE_RESOURCES" in os.environ:
        return json.loads(os.environ["COMFY_BRIDGE_RESOURCES"])
    return load_deployment_config(os.environ.get("COMFY_BRIDGE_CONFIG", "config.yaml"))


def modal_names(settings=None):
    name = (settings if settings is not None else load_runtime_config())["app_name"]
    return {
        "app_name": name,
        "model_volume": f"{name}-models",
        "job_volume": f"{name}-jobs",
        "state_dict": f"{name}-state",
    }
