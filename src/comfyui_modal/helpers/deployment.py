"""Checks that remote metadata matches the local deployment pins."""

from ..config import DEPLOYMENT_ID


def is_current_deployment(snapshot) -> bool:
    return bool(snapshot) and snapshot.get("deployment_id") == DEPLOYMENT_ID


def require_current_deployment(data) -> None:
    if data["deployment_id"] != DEPLOYMENT_ID:
        raise ValueError("Remote deployment differs from local pins; deploy this version first")
