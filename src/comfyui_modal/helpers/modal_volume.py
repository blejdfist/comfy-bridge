"""Modal Volume calls that tolerate API rate limits."""

import asyncio

import modal


async def retry_rate_limited(operation, *args, **kwargs):
    """Volume listing/removal is idempotent; back off on Modal's API rate limit."""
    for attempt in range(6):
        try:
            return await operation(*args, **kwargs)
        except modal.exception.ResourceExhaustedError:
            if attempt == 5:
                raise
            await asyncio.sleep(0.5 * 2**attempt)


async def list_volume_entries(volume):
    """Every entry in a Volume, or None when the Volume does not exist."""
    try:
        return await retry_rate_limited(volume.listdir.aio, "/", recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
