"""Explicit pinned CED-Mini float preparation; no downloaded code is executed here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_audiosep import download

from dotears.audiosep import file_digest
from dotears.ced import CED_REVISION, FLOAT_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    files = {
        **FLOAT_FILES,
        "README.md": "0fbd9dd64fffc22211d63b22689e38a81446d5e37db6148e3ac42667d3c22882",
    }
    records = []
    for name, expected in files.items():
        path = root / name
        if args.check_only:
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError("Missing/modified pinned CED artifact: " + name)
            records.append({"path": name, "sha256": expected, "bytes": path.stat().st_size})
        else:
            url = f"https://huggingface.co/mispeech/ced-mini/resolve/{CED_REVISION}/{name}"
            record = download(
                url, path, expected, 40_000_000 if name.endswith("safetensors") else 100_000
            )
            records.append({**record, "path": name})
    manifest = {
        "schema_version": "dotears.ced-preparation/1",
        "repository": "mispeech/ced-mini",
        "revision": CED_REVISION,
        "license": "Apache-2.0",
        "files": records,
        "inference_precision": "float32",
        "quantized_onnx_not_selected": True,
        "downloaded_code_executed_during_preparation": False,
    }
    if not args.check_only:
        (root / "preparation.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    print(
        json.dumps(
            {
                "status": "ready",
                "network_used": not args.check_only,
                "path": str(root),
                "files_verified": len(records),
                "precision": "float32",
            }
        )
    )


if __name__ == "__main__":
    main()
