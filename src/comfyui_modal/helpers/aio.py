"""asyncio task utilities."""

import asyncio
import contextlib


async def cancel_and_wait(task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
