"""Explicit one-time AudioSep download; pinned weights/source/tokenizer, no remote execution."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

from dotears.audiosep import PINNED_FILES, SOURCE_REVISION, file_digest, verify_prepared_audiosep

CHECKPOINT_REVISION = "0807a713283035d8b0d0628f1b8e3d6580459f7b"
TOKENIZER_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
TOKENIZER_FILES = {
    "config.json": "ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9",
    "vocab.json": "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "tokenizer_config.json": "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
}


def download(url: str, path: Path, expected: str, limit: int) -> dict:
    if path.exists():
        if file_digest(path) != expected:
            raise ValueError(f"Existing file differs from the pinned artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        first = partial.stat().st_size if partial.exists() else 0
        if first > limit:
            raise ValueError("Partial download exceeds the file budget.")
        headers = {"User-Agent": "DotEARS-explicit-local-model-preparation"}
        if first:
            headers["Range"] = f"bytes={first}-"
        started, last_report = time.monotonic(), first
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=60
        ) as response:
            if first and response.status != 206:
                first = 0
            total = first
            with partial.open("ab" if first else "wb") as handle:
                while chunk := response.read(1024**2):
                    total += len(chunk)
                    if total > limit or time.monotonic() - started > 1800:
                        raise ValueError("Model preparation exceeded its byte/time budget.")
                    handle.write(chunk)
                    if total - last_report >= 128 * 1024**2:
                        print(
                            json.dumps({"file": path.name, "downloaded_mib": total // 1024**2}),
                            flush=True,
                        )
                        last_report = total
        if file_digest(partial) != expected:
            raise ValueError(f"Publisher/pinned SHA-256 mismatch: {path.name}")
        if path.exists():
            raise FileExistsError("Another process created the preparation target.")
        partial.rename(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": expected, "url": url}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify an existing preparation without network access",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if args.check_only:
        verify_prepared_audiosep(root)
        print(json.dumps({"status": "ready", "path": str(root), "network_used": False}))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise ValueError("At least 2 GiB free space is required for preparation.")
    manifest = {
        "schema_version": "dotears.audiosep-preparation/1",
        "github_revision": SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "repository": "https://github.com/Audio-AGI/AudioSep",
        "files": [],
    }
    specifications = {
        **PINNED_FILES,
        "source/LICENSE": "f38c0f36eebc626636decda302388054a78c0e89af440d3936ba626e0db05ead",
        "source/config/audiosep_base.yaml": (
            "e7e2e1a089d1de5b58ee0ddeae978f5c8a4649ae0ddea724301363a1427f7f52"
        ),
    }
    for relative, expected in specifications.items():
        if relative.startswith("source/"):
            url = f"https://raw.githubusercontent.com/Audio-AGI/AudioSep/{SOURCE_REVISION}/{relative[7:]}"
            limit = 2 * 1024**2
        else:
            url = f"https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/{CHECKPOINT_REVISION}/checkpoint/{relative}?download=true"
            limit = 1264844076
        record = download(url, root / relative, expected, limit)
        record["path"] = relative
        manifest["files"].append(record)
    tokenizer = {
        "repository": "FacebookAI/roberta-base",
        "revision": TOKENIZER_REVISION,
        "learned_weights_downloaded": False,
        "files": [],
    }
    for name, expected in TOKENIZER_FILES.items():
        url = f"https://huggingface.co/FacebookAI/roberta-base/resolve/{TOKENIZER_REVISION}/{name}"
        record = download(url, root / "tokenizer" / name, expected, 2 * 1024**2)
        record["path"] = name
        tokenizer["files"].append(record)
    (root / "tokenizer/manifest.json").write_text(json.dumps(tokenizer, indent=2), "utf-8")
    (root / "preparation.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    verify_prepared_audiosep(root)
    print(json.dumps({"status": "prepared", "path": str(root), "checkpoint_bytes": 1264844076}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
