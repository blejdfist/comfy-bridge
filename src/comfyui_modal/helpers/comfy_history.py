"""Walk ComfyUI history entries."""


def iter_output_items(history: dict):
    """Yield every file entry (a dict with a filename) in a history's node outputs."""
    for outputs in history.get("outputs", {}).values():
        for entries in outputs.values():
            if not isinstance(entries, list):
                continue
            for item in entries:
                if isinstance(item, dict) and "filename" in item:
                    yield item


def last_execution_error(messages):
    """The payload of the most recent well-formed execution_error status message."""
    for event in reversed(messages):
        if (
            isinstance(event, (list, tuple))
            and len(event) == 2
            and event[0] == "execution_error"
            and isinstance(event[1], dict)
        ):
            return event[1]
    return None
