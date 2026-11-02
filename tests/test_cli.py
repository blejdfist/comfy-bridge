import json

from comfyui_modal.cli import summarize_error


def test_summarize_error_skips_malformed_execution_errors():
    record = {
        "messages": [
            ["execution_error", {"exception_message": "out of memory"}],
            ["execution_error", "not a payload"],
        ]
    }
    assert summarize_error(json.dumps(record)) == "out of memory"
