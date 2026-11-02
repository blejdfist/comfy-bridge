import json
import sys

from comfyui_modal.cli import main, summarize_error


def test_benchmark_merge_parses_and_runs(monkeypatch, tmp_path):
    a, b = tmp_path / "a" / "results.json", tmp_path / "b" / "results.json"
    for path in (a, b):
        path.parent.mkdir()
        path.write_text('{"provenance": {}, "jobs": [{"id": 1}]}')
    out = tmp_path / "m.json"
    # main() writes COMFY_BRIDGE_CONFIG; setenv first so monkeypatch restores it.
    monkeypatch.setenv("COMFY_BRIDGE_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["comfy-bridge", "--data-dir", str(tmp_path / "d"), "--config", str(tmp_path / "config.yaml"),
         "benchmark", "merge", str(a), str(b), "-o", str(out)],
    )
    main()
    assert [j["id"] for j in json.loads(out.read_text())["jobs"]] == ["a:1", "b:1"]


def test_summarize_error_skips_malformed_execution_errors():
    record = {
        "messages": [
            ["execution_error", {"exception_message": "out of memory"}],
            ["execution_error", "not a payload"],
        ]
    }
    assert summarize_error(json.dumps(record)) == "out of memory"
