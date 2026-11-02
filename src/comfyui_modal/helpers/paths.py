"""Path utilities shared by the bridge and the remote worker."""

from pathlib import Path, PurePosixPath


def join_subfolder(subfolder: str, filename: str) -> str:
    """ComfyUI's subfolder/filename pair as one relative POSIX path."""
    return str(PurePosixPath(subfolder) / filename)


def safe_path(root: Path, name: str) -> Path:
    """Resolve a client-supplied relative path, including existing symlinks."""
    if "\\" in name or "\x00" in name or Path(name).is_absolute():
        raise ValueError("Expected a relative path")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Path escapes its storage directory")
    return target
