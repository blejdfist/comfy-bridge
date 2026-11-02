"""Shared deployment contract. Change these pins together with the editor dependency."""

COMFY_COMMIT = "7cebca52227c63b4180ceed448cbd7ad22903b6d"
FRONTEND_VERSION = "1.57.1"
# Model downloads run in separate CPU containers, each writing its own Volume files.
MODEL_DOWNLOAD_CONCURRENCY = 3
DEPLOYMENT_ID = f"{COMFY_COMMIT}:{FRONTEND_VERSION}:manifest-v2"
