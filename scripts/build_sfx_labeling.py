"""Build a standalone offline SFX labeling page from a prepared WAV manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import wave
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

LABEL_SET_VERSION = "motion-sfx-listening-v1"


def build_dataset(manifest_path: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("schema_version") != "dotears.sfx-review-manifest/1":
        raise ValueError("Unsupported SFX manifest format")
    export_filename = manifest.get("export_filename", "dotears-sfx-labels.json")
    if not isinstance(export_filename, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,80}\.json", export_filename
    ):
        raise ValueError("Expected a plain JSON export filename")
    rows = manifest["samples"]
    if not 1 <= len(rows) <= 200:
        raise ValueError("Expected between 1 and 200 audio samples")
    samples, seen = [], set()
    for row in rows:
        path = (manifest_path.parent / row["audio_path"]).resolve()
        if not path.is_relative_to(manifest_path.parent) or not path.is_file():
            raise ValueError("Preview must be inside the manifest's directory")
        if path.stat().st_size > 8 * 1024**2:
            raise ValueError("A single audio preview exceeds 8 MiB")
        raw = path.read_bytes()
        audio_hash = hashlib.sha256(raw).hexdigest()
        if audio_hash != row["audio_sha256"]:
            raise ValueError(f"Preview hash mismatch: {path}")
        source_hash = row["source_sha256"]
        if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise ValueError("Invalid source SHA-256")
        identity = f"sfx-{source_hash[:20]}"
        if identity in seen:
            raise ValueError("Duplicate source audio in SFX manifest")
        seen.add(identity)
        with wave.open(str(path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
            if audio.getsampwidth() != 2 or audio.getnchannels() not in (1, 2):
                raise ValueError("Expected 16-bit mono/stereo WAV preview")
            if not 0.01 <= duration <= 30:
                raise ValueError("Audio preview must be between 0.01 and 30 seconds")
        if row["source"]["license"] != "CC0-1.0":
            raise ValueError("This clean review set expects confirmed CC0 source metadata")
        source_url = urlsplit(row["source"]["page"])
        if source_url.scheme != "https" or not source_url.hostname:
            raise ValueError("Source credit must reference an HTTPS page")
        samples.append(
            {
                "id": identity,
                "source_sha256": source_hash,
                "audio_sha256": audio_hash,
                "group_id": "group-" + hashlib.sha256(row["family"].encode()).hexdigest()[:16],
                "duration_seconds": round(duration, 6),
                "preview_transform": row["preview_transform"],
                "audio": "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii"),
            }
        )
    seed = str(manifest.get("shuffle_seed", "dotears-sfx-v1"))
    samples.sort(key=lambda row: hashlib.sha256((seed + row["id"]).encode()).digest())
    binding = [
        {key: row[key] for key in ("id", "source_sha256", "audio_sha256", "group_id")}
        for row in samples
    ]
    dataset_id = hashlib.sha256(
        json.dumps({"label_set": LABEL_SET_VERSION, "samples": binding}, sort_keys=True).encode()
    ).hexdigest()[:24]
    credits = sorted(
        {(row["source"]["author"], row["source"]["page"], row["source"]["license"]) for row in rows}
    )
    return {
        "dataset_id": dataset_id,
        "label_set_version": LABEL_SET_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "title": manifest.get("title", "DotEARS · Разметка SFX"),
        "export_filename": export_filename,
        "samples": samples,
        "credits": [
            {"author": author, "page": page, "license": license_name}
            for author, page, license_name in credits
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = build_dataset(args.manifest)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("sfx_labeling.html").read_text("utf-8")
    if template.count("__SFX_DATA__") != 1:
        raise ValueError("Expected one dataset placeholder")
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("__SFX_DATA__", payload), "utf-8")
    print(
        json.dumps(
            {
                "output": str(destination),
                "dataset_id": data["dataset_id"],
                "samples": len(data["samples"]),
                "bytes": destination.stat().st_size,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
