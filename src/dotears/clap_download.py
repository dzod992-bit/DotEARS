"""Explicit download of the official CLAP checkpoint, with pinned revision and size/hash checks."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from dotears.errors import DownloadError

REPOSITORY = "laion/clap-htsat-unfused"
FILES = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def download_clap(destination: Path) -> str:
    """Download only official weights/config/tokenizer files; no login or remote Python code."""
    endpoint = f"https://huggingface.co/api/models/{REPOSITORY}?blobs=true"
    with urllib.request.urlopen(endpoint, timeout=60) as response:
        metadata = json.loads(response.read(2 * 1024**2))
    revision = metadata.get("sha", "")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise DownloadError("CLAP repository did not provide a full commit revision.")
    entries = {item["rfilename"]: item for item in metadata.get("siblings", [])}
    total = 0
    for name in FILES:
        entry = entries.get(name)
        if entry is None:
            raise DownloadError(f"Required official CLAP file is missing: {name}")
        expected_size = entry.get("size")
        if not isinstance(expected_size, int) or not 0 < expected_size <= 700 * 1024**2:
            raise DownloadError(f"Unexpected CLAP file size: {name}")
        total += expected_size
        if total > 768 * 1024**2:
            raise DownloadError("CLAP model exceeds the 768 MiB preparation budget.")
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{name}"
        digest = hashlib.sha256()
        written = 0
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            (destination / name).open("wb") as output,
        ):
            while chunk := response.read(1024**2):
                written += len(chunk)
                if written > expected_size:
                    raise DownloadError(f"CLAP download exceeded declared size: {name}")
                digest.update(chunk)
                output.write(chunk)
        if written != expected_size:
            raise DownloadError(f"Incomplete CLAP download: {name}")
        expected_hash = entry.get("lfs", {}).get("sha256")
        if name == "pytorch_model.bin" and not expected_hash:
            raise DownloadError("Official CLAP weight SHA-256 is unavailable.")
        if expected_hash and digest.hexdigest() != expected_hash:
            raise DownloadError(f"CLAP SHA-256 mismatch: {name}")
    return f"https://huggingface.co/{REPOSITORY}/tree/{revision}"
