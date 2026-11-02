import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from comfyui_modal.deployment_config import load_deployment_config


@pytest.mark.parametrize(
    "text",
    [
        "version: 1\ngpu_worker:\n  gpu: imaginary\n",
        "version: 1\ngpu_worker:\n  gpu: [L4, L40S]\n",
        "version: 1\ngpu: L4\n",
        "version: 2\n",
        "version: true\n",
        "gpu_worker: {}\n",
        "[broken",
    ],
)
def test_bad_config_fails_before_resource_selection(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(RuntimeError):
        load_deployment_config(path)


def test_container_defaults_keep_previous_allocations(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("version: 1\n")
    settings = load_deployment_config(tmp_path / "missing.yaml")
    assert settings == load_deployment_config(path)
    assert settings["app_name"] == "comfyui-krea2-bridge"
    assert settings["cpu_only"] == {"cpus": 4, "memory_mib": 8192}
    assert settings["gpu_worker"] == {
        "gpu": "L40S",
        "cpus": 4,
        "memory_mib": 32768,
        "scaledown_window_seconds": 60,
    }


def test_gpu_cannot_be_assigned_to_cpu_only_container(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("version: 1\ncpu_only:\n  gpu: L4\n")
    with pytest.raises(RuntimeError):
        load_deployment_config(path)


def test_both_container_classes_have_independent_resources(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "version: 1\ncpu_only:\n  cpus: 0.5\n  memory_mib: 4096\ngpu_worker:\n  gpu: L4\n  cpus: 8\n  memory_mib: 65536\n"
    )
    settings = load_deployment_config(path)
    assert settings["cpu_only"] == {"cpus": 0.5, "memory_mib": 4096}
    gpu_worker = settings["gpu_worker"]
    assert (gpu_worker["gpu"], gpu_worker["cpus"], gpu_worker["memory_mib"]) == ("L4", 8, 65536)


@pytest.mark.parametrize(
    "resource",
    [
        "cpus: true",
        "cpus: 0",
        "cpus: .nan",
        "cpus: .inf",
        "memory_mib: 1.5",
        "memory_mib: false",
        "memory_mib: 0",
    ],
)
def test_invalid_resources_are_rejected(tmp_path, resource):
    path = tmp_path / "config.yaml"
    path.write_text(f"version: 1\ngpu_worker:\n  {resource}\n")
    with pytest.raises(RuntimeError):
        load_deployment_config(path)


def test_deploy_launches_installed_module_with_selected_config_and_propagates_status(tmp_path):
    from comfyui_modal.cli import deploy

    path = tmp_path / "custom.yaml"
    path.write_text("version: 1\ngpu_worker:\n  gpu: L4\n  cpus: 2\n")
    with patch(
        "comfyui_modal.cli.subprocess.run", return_value=SimpleNamespace(returncode=7)
    ) as run:
        assert deploy(SimpleNamespace(config=path)) == 7
    assert run.call_args.args[0] == [
        sys.executable,
        "-m",
        "modal",
        "deploy",
        "-m",
        "comfyui_modal.modal_app",
    ]
    assert run.call_args.kwargs["env"]["COMFY_BRIDGE_CONFIG"] == str(path.resolve())
    assert run.call_args.kwargs["check"] is False


def test_deploy_invalid_config_never_calls_modal(tmp_path):
    from comfyui_modal.cli import deploy

    path = tmp_path / "bad.yaml"
    path.write_text("version: 1\ngpu_worker:\n  gpu: invalid\n")
    with patch("comfyui_modal.cli.subprocess.run") as run, pytest.raises(RuntimeError):
        deploy(SimpleNamespace(config=path))
    run.assert_not_called()


def test_decorators_use_configured_resources_at_deploy_time(tmp_path, monkeypatch):
    import importlib.util

    import modal

    from comfyui_modal import modal_app

    path = tmp_path / "config.yaml"
    path.write_text(
        "version: 1\ncpu_only:\n  cpus: 1\n  memory_mib: 4096\ngpu_worker:\n  gpu: L4\n"
        "  cpus: 8\n  memory_mib: 65536\n  scaledown_window_seconds: 300\n"
    )
    monkeypatch.setenv("COMFY_BRIDGE_CONFIG", str(path))
    definitions = {}

    def decorator(self, **kwargs):
        def register(value):
            definitions[value.__name__] = kwargs
            return value

        return register

    spec = importlib.util.spec_from_file_location("resource_test_app", Path(modal_app.__file__))
    module = importlib.util.module_from_spec(spec)
    with (
        patch.object(modal, "is_local", return_value=True),
        patch.object(modal.App, "function", decorator),
        patch.object(modal.App, "cls", decorator),
    ):
        spec.loader.exec_module(module)
    for name in ("install_model", "refresh_models"):
        assert definitions[name]["cpu"] == 1
        assert definitions[name]["memory"] == 4096
        assert "gpu" not in definitions[name]
    assert definitions["GPUWorker"]["gpu"] == "L4"
    assert definitions["GPUWorker"]["cpu"] == (8, 8)
    assert definitions["GPUWorker"]["memory"] == 65536
    assert definitions["GPUWorker"]["scaledown_window"] == 300


@pytest.mark.parametrize("value", ["1", "3601", "60.5"])
def test_invalid_scaledown_window_is_rejected(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(f"version: 1\ngpu_worker:\n  scaledown_window_seconds: {value}\n")
    with pytest.raises(RuntimeError):
        load_deployment_config(path)


def test_app_name_controls_all_backend_resources(tmp_path, monkeypatch):
    from comfyui_modal.backend import ModalBackend
    from comfyui_modal.deployment_config import modal_names

    path = tmp_path / "custom.yaml"
    path.write_text("version: 1\napp_name: studio-test\n")
    monkeypatch.delenv("COMFY_BRIDGE_RESOURCES", raising=False)
    monkeypatch.setenv("COMFY_BRIDGE_CONFIG", str(path))
    assert modal_names() == {
        "app_name": "studio-test",
        "model_volume": "studio-test-models",
        "job_volume": "studio-test-jobs",
        "state_dict": "studio-test-state",
    }
    with (
        patch("modal.Cls.from_name"),
        patch("modal.Volume.from_name") as volume,
        patch("modal.Dict.from_name") as state,
    ):
        ModalBackend()
    volume.assert_called_once_with("studio-test-jobs")
    state.assert_called_once_with("studio-test-state")


@pytest.mark.parametrize("name", ["../models", 123, "x" * 58, "ap-" + "a" * 22])
def test_invalid_app_name_fails_before_deployment(tmp_path, name):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "app_name": name}))
    with pytest.raises(RuntimeError):
        load_deployment_config(path)


@pytest.mark.parametrize("global_option", [True, False])
def test_cli_deploy_accepts_same_config_before_or_after_command(
    tmp_path, monkeypatch, global_option
):
    import os

    from comfyui_modal.cli import main

    path = tmp_path / "custom.yaml"
    path.write_text("version: 1\napp_name: another-studio\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMFY_BRIDGE_CONFIG", "unused")
    arguments = (
        ["--config", str(path), "deploy"] if global_option else ["deploy", "--config", str(path)]
    )
    monkeypatch.setattr(sys, "argv", ["comfy-bridge", *arguments])
    with (
        patch("comfyui_modal.cli.deploy", return_value=0) as deploy,
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 0
    assert deploy.call_args.args[0].config == path
    assert os.environ["COMFY_BRIDGE_CONFIG"] == str(path.resolve())
