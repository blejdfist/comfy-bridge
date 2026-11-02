import json
from pathlib import Path

import pytest
import yaml

from comfyui_modal.benchmark import (
    expand_jobs,
    file_inputs,
    function_name,
    gpu_matches,
    is_api_prompt,
    load_spec,
    merge,
    modal_gpu,
    output_files,
    output_name,
    run_finished,
    select_jobs,
    summarize,
    unknown_nodes,
)


def write_spec(tmp_path, workflows=("a.json",), configs=({"gpu": "H100", "cpu": 1},)):
    for name in workflows:
        (tmp_path / name).write_text("{}")
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump({"workflows": list(workflows), "configs": list(configs)}))
    return path


def test_expand_is_config_major(tmp_path):
    spec = write_spec(
        tmp_path,
        workflows=["a.json", "b.json"],
        configs=[
            {"gpu": "H100", "cpu": 1},
            {"gpu": "H100", "cpu": 2},
            {"gpu": "L40S", "cpu": 1},
            {"gpu": "L40S", "cpu": 2},
        ],
    )
    jobs = expand_jobs(load_spec(spec))
    assert len(jobs) == 8
    assert [(j["id"], j["gpu"], j["cpu"], j["workflow"].name) for j in jobs[:3]] == [
        (1, "H100", 1, "a.json"),
        (2, "H100", 1, "b.json"),
        (3, "H100", 2, "a.json"),
    ]
    assert jobs[0]["flags"] == []
    assert jobs[0]["workflow"] == tmp_path / "a.json"
    assert [j["name"] for j in jobs[:2]] == ["a.json", "b.json"]


def test_same_stem_in_two_folders_keeps_distinct_names(tmp_path):
    for folder in ("x", "y"):
        (tmp_path / folder).mkdir()
    spec = write_spec(tmp_path, workflows=["x/flow.json", "y/flow.json"])
    assert [j["name"] for j in expand_jobs(load_spec(spec))] == ["x/flow.json", "y/flow.json"]


@pytest.mark.parametrize(
    "configs",
    [
        [],
        [{"gpu": "H100", "cpu": 0}],
        [{"gpu": "H100", "cpu": True}],
        [{"gpu": "H100", "cpu": 1, "flags": "--highvram"}],
        [{"gpu": "H100", "cpu": 1, "ram": 1}],
    ],
)
def test_bad_spec_fails(tmp_path, configs):
    with pytest.raises(RuntimeError):
        load_spec(write_spec(tmp_path, configs=configs))


def test_missing_workflow_file_fails(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump({"workflows": ["nope.json"], "configs": [{"gpu": "L4", "cpu": 1}]}))
    with pytest.raises(RuntimeError):
        load_spec(path)


def test_explicit_inputs_resolve_relative_to_spec(tmp_path):
    spec = write_spec(tmp_path)
    asset = tmp_path / "saved.latent"
    asset.write_bytes(b"synthetic tensor file")
    data = yaml.safe_load(spec.read_text())
    data["inputs"] = {"fixtures/video.latent": "saved.latent"}
    spec.write_text(yaml.safe_dump(data))
    assert load_spec(spec)["inputs"] == {"fixtures/video.latent": asset.resolve()}


@pytest.mark.parametrize("inputs", [[], {"../escape": "saved.latent"},
                                   {"/absolute": "saved.latent"},
                                   {"a\\b": "saved.latent"},
                                   {"file": 42}, {"file": "missing.latent"},
                                   {"file": "credentials.yaml.backup"},
                                   {"file": "test.credentials.yml"}])
def test_invalid_explicit_inputs_fail_before_launch(tmp_path, inputs):
    spec = write_spec(tmp_path)
    # Sources exist, so each case can only fail on its own check.
    for name in ("saved.latent", "credentials.yaml.backup", "test.credentials.yml"):
        (tmp_path / name).write_bytes(b"synthetic")
    data = yaml.safe_load(spec.read_text())
    data["inputs"] = inputs
    spec.write_text(yaml.safe_dump(data))
    with pytest.raises(RuntimeError):
        load_spec(spec)


async def test_explicit_inputs_reach_every_workflow_and_satisfy_uploads(
    aiohttp_server, tmp_path, monkeypatch
):
    import sys

    from aiohttp import web

    import comfyui_modal

    # Re-import modal_app against a missing config in tmp_path, never the project's YAML.
    # setitem/setattr first so the originals are restored (or removed) after the test.
    monkeypatch.setenv("COMFY_BRIDGE_CONFIG", str(tmp_path / "config.yaml"))
    for name in ("modal_app", "benchmark_app"):
        monkeypatch.setitem(sys.modules, f"comfyui_modal.{name}", None)
        del sys.modules[f"comfyui_modal.{name}"]
        monkeypatch.setattr(comfyui_modal, name, None, raising=False)
        delattr(comfyui_modal, name)
    from comfyui_modal.benchmark_app import preflight

    schema = {
        "AnyFileLoader": {"input": {"required": {"file": ["STRING", {"file_upload": True}]}}},
        "OtherLoader": {"input": {"required": {"file": [["fixture.bin"]]}}},
    }
    app = web.Application()

    async def object_info(request):
        return web.json_response(schema)

    app.router.add_get('/object_info', object_info)
    # No /view route: fetching an explicitly supplied file would fail this test.
    server = await aiohttp_server(app)
    prompt = {
        "1": {"class_type": "AnyFileLoader", "inputs": {"file": "./image.png"}},
        "2": {"class_type": "OtherLoader", "inputs": {"file": "fixture.bin"}},
    }
    files = {"image.png": b"synthetic image", "fixture.bin": b"synthetic fixture"}
    results = await preflight({Path('a.json'): prompt, Path('b.json'): prompt},
                              str(server.make_url('')).rstrip('/'), files)
    for uploaded in results.values():
        assert uploaded['fixture.bin'] == files['fixture.bin']
        assert uploaded['./image.png'] == files['image.png']
    assert './image.png' not in files


def test_select_only_and_unknown_id():
    jobs = [{"id": i} for i in range(1, 9)]
    assert [j["id"] for j in select_jobs(jobs, "3,7")] == [3, 7]
    assert select_jobs(jobs, "") == jobs
    with pytest.raises(RuntimeError):
        select_jobs(jobs, "9")


def test_gpu_pin_and_match():
    assert modal_gpu("H100") == "H100!" and modal_gpu("L40S") == "L40S"
    assert gpu_matches("H100", "NVIDIA H100 80GB HBM3, 81559 MiB, 570.1")
    assert not gpu_matches("H100", "NVIDIA H200, 143771 MiB, 570.1")
    assert gpu_matches("A100-80GB", "NVIDIA A100-SXM4-80GB, 81920 MiB, 570.1")


def test_api_prompt_detection():
    assert is_api_prompt({"3": {"class_type": "KSampler", "inputs": {}}})
    assert not is_api_prompt({"nodes": [], "links": []})


def test_file_inputs_uses_upload_widgets_and_strips_dot_slash():
    object_info = {
        "LoadImage": {"input": {"required": {"image": [["cat.png"], {"image_upload": True}]}}},
        "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": True}]}}},
    }
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "./cat.png"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat.png"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "missing.png"}},
    }
    assert file_inputs(prompt, object_info) == {"./cat.png": "cat.png", "missing.png": "missing.png"}


def test_output_names_unique_across_same_stem():
    a = {"id": 1, "gpu": "H100", "cpu": 1, "workflow": Path("x/flow.json")}
    b = {**a, "id": 2, "workflow": Path("y/flow.json")}
    assert output_name(a, 1, "out_0001.mp4").endswith(".mp4")
    assert output_name(a, 1, "v.mp4") != output_name(b, 1, "v.mp4")


def test_summarize_sums_and_attaches_steps():
    records = [
        {"kind": "model_file_read", "seconds": 2.0},
        {"kind": "model_file_read", "seconds": 1.0},
        {"kind": "weight_loading", "seconds": 3.0},
        {"kind": "sampling_step", "step": 0, "start": 10.0, "end": 11.0, "seconds": 1.0},
        {"kind": "sampling", "start": 10.0, "end": 12.0, "seconds": 2.0, "peak_allocated_bytes": 5},
        {"kind": "sampling_step", "step": 0, "start": 20.0, "end": 20.5, "seconds": 0.5},
        {"kind": "sampling", "start": 20.0, "end": 21.0, "seconds": 1.0, "peak_allocated_bytes": 7},
        {"kind": "vae_decode", "seconds": 0.5},
    ]
    status = {
        "status_str": "success",
        "messages": [["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 9000}]],
    }
    row = summarize(records, status)
    assert row["seconds"]["model_file_read"] == 3.0 and row["seconds"]["graph_total"] == 8.0
    assert row["seconds"]["sampling"] == 3.0 and row["seconds"]["vae_encode"] == 0.0
    assert row["sampling_stages"] == [{"seconds": 2.0, "steps": [1.0]}, {"seconds": 1.0, "steps": [0.5]}]
    assert row["peak_vram_bytes"] == 7 and row["error"] is None


def test_summarize_reports_execution_error():
    status = {"status_str": "error", "messages": [["execution_error", {"exception_message": "OOM"}]]}
    row = summarize([], status)
    assert row["error"] == "OOM" and row["seconds"]["graph_total"] is None


def test_merge_prefixes_ids_and_disambiguates_labels():
    a = {"provenance": {"deployment": {"label": "dense"}}, "jobs": [{"id": 1}]}
    b = {"provenance": {"deployment": {"label": "dense"}}, "jobs": [{"id": 1}]}
    merged = merge([("20261027-1", a), ("20261027-2", b)])
    assert [j["id"] for j in merged["jobs"]] == ["dense:1", "20261027-2:1"]
    assert [j["pass"] for j in merged["jobs"]] == ["dense", "20261027-2"]
    assert len(merged["provenance"]) == 2


def test_probe_records_sampling_steps_and_keeps_callback(tmp_path, monkeypatch):
    import sys
    import types

    from comfyui_modal import benchmark_probe

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(benchmark_probe, "RECORDS", tmp_path / "records.jsonl")
    seen = []

    def sample(self, model, sigmas, extra, callback, noise):
        for step in range(3):
            callback(step, None, None, 3)
        return "latent"

    wrapped = benchmark_probe.measured(sample, "sampling")
    assert wrapped(None, None, None, None, lambda *a: seen.append(a[0]), None) == "latent"
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["sampling_step"] * 3 + ["sampling"]
    assert seen == [0, 1, 2]
    assert len(summarize(rows, {"status_str": "success"})["sampling_stages"][0]["steps"]) == 3


def test_unknown_nodes_names_missing_and_unconverted_nodes():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "2": {"class_type": "NoSuchNode", "inputs": {}},
        "7": {"class_type": None, "inputs": {}},
    }
    unknown = unknown_nodes(prompt, {"KSampler"})
    assert len(unknown) == 2 and unknown[0] == "NoSuchNode" and "7" in unknown[1]
    assert unknown_nodes({"1": prompt["1"]}, {"KSampler"}) == []


async def test_probe_attributes_nested_stages_and_restores_node_context(tmp_path, monkeypatch):
    import sys
    from enum import Enum
    from types import SimpleNamespace

    from comfyui_modal import benchmark_probe as probe

    class Result(Enum):
        SUCCESS = 1

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(probe, 'RECORDS', tmp_path / 'records.jsonl')
    prompt = SimpleNamespace(get_node=lambda node_id: {'class_type': 'AnyNode'})
    load = probe.measured(lambda: 'weights', 'weight_loading')

    async def execute(server, dynprompt, caches, node_id):
        assert load() == 'weights'
        return Result.SUCCESS, None, None

    result = await probe.measured_node(execute)(None, prompt, None, '7')
    assert result[0] is Result.SUCCESS
    assert probe.active_node.get() is None
    rows = [json.loads(line) for line in probe.RECORDS.read_text().splitlines()]
    stage, node = rows
    assert stage['node_id'] == node['node_id'] == '7'
    assert stage['class_type'] == node['class_type'] == 'AnyNode'
    assert node['start'] <= stage['start'] <= stage['end'] <= node['end']
    assert node['execution_result'] == 'SUCCESS'
    assert summarize(rows, {'status_str': 'success'})['node_timings'] == [node]

    async def fail(*args):
        raise RuntimeError('synthetic failure')

    with pytest.raises(RuntimeError):
        await probe.measured_node(fail)(None, prompt, None, '8')
    assert probe.active_node.get() is None
    last = json.loads(probe.RECORDS.read_text().splitlines()[-1])
    assert last['execution_result'] == 'EXCEPTION' and last['node_id'] == '8'


def test_run_finished_waits_for_history_write():
    # ComfyUI writes history after execution_success; only executing node=None follows it.
    assert not run_finished({"type": "execution_success", "data": {"prompt_id": "p"}}, "p")
    assert not run_finished({"type": "execution_error", "data": {"prompt_id": "p"}}, "p")
    assert not run_finished({"type": "executing", "data": {"node": "3", "prompt_id": "p"}}, "p")
    assert not run_finished({"type": "executing", "data": {"node": None, "prompt_id": "q"}}, "p")
    assert run_finished({"type": "executing", "data": {"node": None, "prompt_id": "p"}}, "p")


def test_function_name_shows_resources_and_ignores_flags():
    job = {"gpu": "H100", "cpu": 2, "memory_mib": 32768, "flags": []}
    assert function_name({**job, "flags": ["--highvram"]}) == function_name(job)
    shapes = [job, {**job, "gpu": "L40S"}, {**job, "cpu": 4}, {**job, "memory_mib": 65536}]
    assert len({function_name(j) for j in shapes}) == len(shapes)
    assert "+" not in function_name({**job, "gpu": "B200+"})
    assert function_name({**job, "gpu": "B200+"}) != function_name({**job, "gpu": "B200"})


def test_output_files_stay_inside_output_root(tmp_path):
    def history(*items):
        return {"outputs": {"9": {"images": list(items)}}}

    saved = {"filename": "a.png", "subfolder": "x", "type": "output"}
    assert output_files(history(saved, {"type": "output"}, {**saved, "type": "temp"}), tmp_path) == [
        ("a.png", (tmp_path / "x" / "a.png").resolve())
    ]
    with pytest.raises(ValueError):
        output_files(history({**saved, "subfolder": "../.."}), tmp_path)


def test_unreadable_spec_fails_without_echoing_contents(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_bytes(b"workflows: [\xff secret")
    with pytest.raises(RuntimeError) as error:
        load_spec(path)
    assert "secret" not in str(error.value)
