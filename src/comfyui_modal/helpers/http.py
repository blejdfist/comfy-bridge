"""aiohttp client utilities for the bridge's HTTP API."""


async def get_json(session, url, **kwargs):
    async with session.get(url, **kwargs) as response:
        response.raise_for_status()
        return await response.json()


async def json_or_raise(response, expected_status):
    """The response body, or its `error` field raised as RuntimeError."""
    result = await response.json()
    if response.status != expected_status:
        raise RuntimeError(result.get("error", result))
    return result
