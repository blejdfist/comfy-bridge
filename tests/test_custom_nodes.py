import pytest

from comfyui_modal.custom_nodes import load_custom_nodes


def test_missing_manifest_installs_nothing(tmp_path):
    assert load_custom_nodes(tmp_path / "missing.yaml") == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "../escape"),
        ("repository", "https://github.com/a/b;touch /tmp/no"),
        ("revision", "main"),
    ],
)
def test_reject_invalid_nodes(tmp_path, field, value):
    import yaml

    node = {"name": "Example", "repository": "https://github.com/a/b", "revision": "a" * 40}
    node[field] = value
    path = tmp_path / "nodes.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "nodes": [node]}))
    with pytest.raises(RuntimeError):
        load_custom_nodes(path)


@pytest.mark.asyncio
async def test_changed_remote_nodes_force_full_metadata_refresh(bridge, backend):
    from unittest.mock import AsyncMock

    from comfyui_modal.config import DEPLOYMENT_ID

    bridge.snapshot["custom_nodes"] = []
    rebuilt = {"deployment_id": DEPLOYMENT_ID, "custom_nodes": [{"name": "KJNodes"}]}
    backend.refresh_models = AsyncMock(
        side_effect=[{"custom_nodes": rebuilt["custom_nodes"]}, rebuilt]
    )
    assert await bridge.refresh_models() == "full"
    assert [call.kwargs for call in backend.refresh_models.call_args_list] == [
        {"full": False},
        {"full": True},
    ]
    assert bridge.snapshot == rebuilt
