"""Workflow validation and job-specific output rewriting."""

import copy

LOAD_IMAGE_NODES = ("LoadImage", "LoadImageMask")


def validate_prompt(prompt: dict, schema: dict) -> dict:
    """Fast local checks; ComfyUI remains authoritative for graph/type validation."""
    errors = {}
    if not isinstance(prompt, dict) or not prompt:
        raise ValueError("prompt must be a non-empty API-format graph")
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise ValueError(f"Node {node_id} must have class_type and inputs")
        kind = node.get("class_type")
        if not isinstance(kind, str) or kind not in schema:
            errors[node_id] = {
                "class_type": kind,
                "errors": [
                    {
                        "type": "invalid_node_type",
                        "message": f"Node unavailable remotely: {kind}",
                        "details": "Refresh metadata or deploy a runtime with this node.",
                    }
                ],
                "dependent_outputs": [],
            }
    if errors:
        return errors

    def ancestors(values, *, conditioning=False):
        pending, seen = list(values), set()
        while pending:
            value = pending.pop()
            if not isinstance(value, list) or len(value) != 2:
                continue
            node_id = str(value[0])
            if node_id in seen or node_id not in prompt:
                continue
            seen.add(node_id)
            node = prompt[node_id]
            yield node_id, node
            if conditioning and node["class_type"] == "CLIPTextEncode":
                # The text may come from a separate LLM with its own encoder.
                # Only the CLIP input supplies this image-conditioning encoder.
                pending.append(node["inputs"].get("clip"))
            else:
                pending.extend(node["inputs"].values())

    for consumer in prompt.values():
        inputs = consumer["inputs"]
        model_nodes = list(ancestors([inputs.get("model")]))
        if not any(
            node["class_type"] == "UNETLoader"
            and str(node["inputs"].get("unet_name", "")).split("/")[-1].lower().startswith("krea2_")
            for _, node in model_nodes
        ):
            continue
        # Follow the consumer's conditioning branch, including through model-sampling
        # wrappers and conditioning transforms; don't reject unrelated graph branches.
        for node_id, node in ancestors(
            (value for name, value in inputs.items() if name != "model"), conditioning=True
        ):
            if node["class_type"] == "CLIPLoader" and node["inputs"].get("type") != "krea2":
                received = node["inputs"].get("type", "unset")
                message = (
                    f"Krea 2 requires CLIPLoader type 'krea2'; node {node_id} uses '{received}'."
                )
                errors[node_id] = {
                    "class_type": "CLIPLoader",
                    "dependent_outputs": [],
                    "errors": [
                        {
                            "type": "krea2_encoder_type_mismatch",
                            "message": message,
                            "details": f"Set type to 'krea2' on Load CLIP node {node_id} before queueing.",
                        }
                    ],
                }
    return errors


def job_prompt(payload: dict, job_id: str) -> dict:
    payload = copy.deepcopy(payload)
    payload["prompt_id"] = job_id
    payload["client_id"] = job_id
    # Native image and video artifacts are isolated by stable job id on the worker.
    for node in payload["prompt"].values():
        if node.get("class_type") in {"SaveImage", "SaveVideo"}:
            prefix = node["inputs"].get("filename_prefix", "ComfyUI")
            if isinstance(prefix, str):
                node["inputs"]["filename_prefix"] = f"{job_id}/{prefix.split('/')[-1]}"
    return payload
