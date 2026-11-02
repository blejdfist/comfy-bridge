from comfyui_modal.worker_io import collect_artifacts


def test_outputs_survive_worker_shutdown_and_history_points_to_local_files(tmp_path):
    output = tmp_path / "comfy-output"
    (output / "job-1").mkdir(parents=True)
    (output / "job-1/image.png").write_bytes(b"png-file")
    job_dir = tmp_path / "jobs/job-1"
    history = {
        "outputs": {
            "29": {"images": [{"filename": "image.png", "subfolder": "job-1", "type": "output"}]}
        }
    }
    artifacts = collect_artifacts(history, job_dir, output_root=output)
    assert len(artifacts) == 1
    assert (job_dir / "artifacts/output/job-1/image.png").read_bytes() == b"png-file"
    item = history["outputs"]["29"]["images"][0]
    assert f"{item['subfolder']}/{item['filename']}" == artifacts[0]["local_path"]


def test_duplicate_output_references_are_copied_once(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "image.png").write_bytes(b"png")
    history = {
        "outputs": {
            "1": {"images": [{"filename": "image.png", "type": "output"}]},
            "2": {"images": [{"filename": "image.png", "type": "output"}]},
        }
    }
    artifacts = collect_artifacts(history, tmp_path / "jobs/job-a", output_root=output)
    assert len(artifacts) == 1


def test_modal_symlink_mount_keeps_remote_paths_volume_relative(tmp_path, symlink):
    volume_root = tmp_path / "actual-volume"
    volume_root.mkdir()
    mount = tmp_path / "jobs"
    symlink(mount, volume_root)
    output = tmp_path / "output"
    output.mkdir()
    (output / "image.png").write_bytes(b"png")
    history = {"outputs": {"29": {"images": [{"filename": "image.png", "type": "output"}]}}}
    artifacts = collect_artifacts(history, mount / "job-a", output_root=output)
    assert artifacts[0]["remote_path"] == "job-a/artifacts/output/image.png"
    assert (volume_root / artifacts[0]["remote_path"]).read_bytes() == b"png"
