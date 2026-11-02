"""Declarative model configuration; credentials are read only by explicit user CLI use."""

import asyncio
import re
import sys
from urllib.parse import parse_qs, urlsplit

from .deployment_config import modal_names
from .helpers.http import get_json
from .helpers.yaml_io import read_yaml
from .model_downloads import CREDENTIAL_KEYS, MODEL_FOLDERS, normalize_request


def load_yaml(path, *, private=False):
    value = read_yaml(path, "Cannot read credentials YAML" if private else "Cannot read model YAML")
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise RuntimeError("YAML configuration must be a mapping with version: 1")
    return value


def manifest_entries(document):
    if set(document) != {"version", "models"} or not isinstance(document["models"], list):
        raise RuntimeError("Model YAML requires version and a models list")
    entries = []
    targets = set()
    for item in document["models"]:
        if not isinstance(item, dict) or set(item) - {
            "name",
            "url",
            "folder",
            "filename",
            "file_id",
            "credential",
        }:
            raise RuntimeError("Unknown model field; API keys belong in credentials YAML")
        if not item.get("filename"):
            raise RuntimeError("Each model needs a filename for repeatable synchronization")
        alias = item.get("credential")
        if alias is not None and (
            not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", alias)
        ):
            raise RuntimeError("Invalid credential alias")
        try:
            request = normalize_request(item)
            if any(
                key.lower() in {"token", "api_key", "apikey", "access_token"}
                for key in parse_qs(urlsplit(request["url"]).query)
            ):
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise RuntimeError(
                "Invalid model entry; use HTTPS URLs without embedded API keys"
            ) from None
        target = (request["folder"], request["filename"])
        if target in targets:
            raise RuntimeError("Duplicate model destination in manifest")
        targets.add(target)
        entries.append((request, alias))
    return entries


def credential_values(document, aliases):
    credentials = document.get("credentials")
    if set(document) != {"version", "credentials"} or not isinstance(credentials, dict):
        raise RuntimeError("Credentials YAML requires version and a credentials mapping")
    result = {}
    for alias in aliases:
        values = credentials.get(alias)
        if (
            not isinstance(values, dict)
            or not values
            or any(
                key not in CREDENTIAL_KEYS
                or not isinstance(value, str)
                or not value.strip()
                or value == "REPLACE_ME"
                for key, value in values.items()
            )
        ):
            raise RuntimeError("Missing or invalid credential values; check credentials YAML")
        result[alias] = values
    return result


def unlisted_files(paths, entries):
    """Model files on the Volume that no manifest entry names, skipping in-progress partials."""
    keep = {f"{request['folder']}/{request['filename']}" for request, _ in entries}
    return sorted(
        path
        for path in paths
        if path.split("/", 1)[0] in MODEL_FOLDERS
        and "/" in path
        and not path.rsplit("/", 1)[-1].startswith(".")
        and path not in keep
    )


async def remove_unlisted(entries, *, yes=False):
    """Delete remote model files missing from the manifest; runs only after downloads succeed."""
    import hashlib

    import modal

    from .helpers.modal_volume import list_volume_entries, retry_rate_limited

    volume = modal.Volume.from_name(modal_names()["model_volume"])
    listing = await list_volume_entries(volume) or []
    sizes = {
        entry.path.lstrip("/"): entry.size
        for entry in listing
        if entry.type == modal.volume.FileEntryType.FILE
    }
    doomed = unlisted_files(sizes, entries)
    if not doomed:
        print("No remote models outside the manifest.")
        return
    print("Remote models not in the manifest:")
    for path in doomed:
        print(f"  {path} ({sizes[path] / 1024**3:.1f} GiB)")
    if not yes and input(f"Delete these {len(doomed)} files from Modal? [y/N] ").lower() != "y":
        print("Kept remote models.")
        return
    for path in doomed:
        await retry_rate_limited(volume.remove_file.aio, path)
        receipt = f".bridge-downloads/{hashlib.sha256(path.encode()).hexdigest()}.json"
        if receipt in sizes:
            await retry_rate_limited(volume.remove_file.aio, receipt)
    print(f"Deleted {len(doomed)} remote models.")


async def sync_manifest(args):
    import argparse

    import aiohttp
    import modal

    from .bridge_client import run_model_command

    entries = manifest_entries(load_yaml(args.manifest))
    async with aiohttp.ClientSession() as session:
        catalog = await get_json(session, args.bridge_url + "/bridge/models/catalog")
    installed = catalog.get("models", {})
    pending = [
        (request, alias)
        for request, alias in entries
        if request["filename"] not in installed.get(request["folder"], [])
    ]
    downloaded, skipped = 0, len(entries) - len(pending)
    try:
        aliases = {alias for _, alias in pending if alias}
        secrets = {}
        if aliases:
            values = credential_values(load_yaml(args.credentials, private=True), aliases)
            app_name = modal_names()["app_name"]
            for alias in sorted(aliases):
                # Prefixed: Secret names are workspace-wide and teardown deletes by prefix.
                name = f"{app_name}-{alias}"
                try:
                    # create(allow_existing=True) keeps stale values, so replace the Secret.
                    await modal.Secret.objects.delete.aio(name, allow_missing=True)
                    await modal.Secret.objects.create.aio(name, values[alias])
                except Exception:
                    raise RuntimeError(
                        "Could not provision download credentials in Modal"
                    ) from None
                secrets[alias] = name
            print(
                f"Stored download credentials in Modal Secrets {', '.join(secrets.values())} "
                "(values are never printed)."
            )
        if pending:
            print(f"Downloading {len(pending)} models", flush=True)
        # Queue everything at once; the bridge caps how many run remotely at a time.
        results = await asyncio.gather(
            *(
                run_model_command(
                    argparse.Namespace(
                        **{
                            **request,
                            "secret": secrets.get(alias),
                            "refresh_catalog": False,
                            "skip_existing": True,
                        },
                        model_command="download",
                        bridge_url=args.bridge_url,
                        position=index,
                    )
                )
                for index, (request, alias) in enumerate(pending)
            ),
            return_exceptions=True,
        )
        failures = []
        for (request, _), result in zip(pending, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(result)
                print(
                    f"{request['folder']}/{request['filename']} failed: {result}",
                    file=sys.stderr,
                )
            elif isinstance(result, dict) and result.get("skipped"):
                skipped += 1
            else:
                downloaded += 1
        if failures:
            raise failures[0]
        if getattr(args, "rm", False):
            await remove_unlisted(entries, yes=getattr(args, "yes", False))
    finally:
        original_error = sys.exception()
        full = getattr(args, "full", False)
        print(
            "Rebuilding full editor metadata…" if full else "Refreshing model catalog…", flush=True
        )
        try:
            await run_model_command(
                argparse.Namespace(model_command="refresh", bridge_url=args.bridge_url, full=full)
            )
        except Exception:
            if original_error is None:
                raise
            print(
                "Final catalog refresh also failed; run `comfy-bridge sync` after resolving the download error.",
                file=sys.stderr,
            )
    print(f"Model manifest synchronized: {downloaded} downloaded, {skipped} already installed.")
