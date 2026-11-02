"""Read Modal's control-plane counters without invoking or waking a GPU worker."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import modal

from comfyui_modal.config import modal_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    worker = modal.Cls.from_name(modal_names()["app_name"], "GPUWorker")()
    stats = worker.run.get_current_stats()
    result = {"observed_at": time.time(), **dataclasses.asdict(stats)}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
