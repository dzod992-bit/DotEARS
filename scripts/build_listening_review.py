"""Build an offline listening worksheet from existing reports and matching local media.

The manifest is a JSON array of {title, report, video} objects. Paths are relative to
the manifest. Media is embedded in the generated HTML; nothing is uploaded or analyzed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def starter_ids(attacks: list[dict], sync: list[dict]) -> list[str]:
    """Sample distinct excerpt windows; overlap is not semantic sound deduplication."""
    target = min(12, len(attacks))
    selected: list[dict] = []

    def add(event: dict) -> None:
        if len(selected) >= target or any(e["id"] == event["id"] for e in selected):
            return
        for previous in selected:
            overlap = max(
                0.0, min(previous["end"], event["end"]) - max(previous["start"], event["start"])
            )
            shorter = min(previous["end"] - previous["start"], event["end"] - event["start"])
            if shorter > 0 and overlap / shorter > 0.8:
                return
        selected.append(event)

    for event in sorted(attacks, key=lambda e: -e["score"])[:4]:
        add(event)
    for candidate in sync[:3]:
        for event in attacks:
            if event["id"] == candidate.get("event_id"):
                add(event)
    if attacks:
        for i in range(target):
            add(attacks[round(i * (len(attacks) - 1) / max(1, target - 1))])
        # Fill any rejected windows from across the remaining timeline, rather than
        # concentrating replacements at the beginning. A small pool may yield <12.
        while len(selected) < target:
            remaining = [e for e in attacks if all(e["id"] != s["id"] for s in selected)]
            remaining.sort(
                key=lambda e: min(abs(e["peak_time"] - s["peak_time"]) for s in selected),
                reverse=True,
            )
            before = len(selected)
            for event in remaining:
                add(event)
                if len(selected) > before:
                    break
            if len(selected) == before:
                break
    selected_ids = {e["id"] for e in selected}
    return [e["id"] for e in attacks if e["id"] in selected_ids]


def build_reference(item: dict, parent: Path) -> dict:
    report_path = (parent / item["report"]).resolve()
    video_path = (parent / item["video"]).resolve()
    report = json.loads(report_path.read_text("utf-8"))
    with video_path.open("rb") as source:
        source_sha = hashlib.file_digest(source, "sha256").hexdigest()
    if source_sha != report["source"]["sha256"]:
        raise ValueError(f"Video hash does not match the report: {video_path}")
    attacks = []
    for event in report["events"]:
        if event.get("category") != "audio_attack" or not event.get("clip_path"):
            continue
        peak = event["peak_time"]
        if not isinstance(peak, (int, float)) or not math.isfinite(peak):
            raise ValueError(f"Invalid peak time: {event['id']}")
        clip = (report_path.parent / event["clip_path"]).resolve()
        clip.relative_to(report_path.parent)
        attacks.append(
            {
                "id": event["id"],
                "peak_time": peak,
                "start": event["clip_start"],
                "end": event["clip_end"],
                "score": event["score"],
                "audio": data_uri(clip, "audio/wav"),
            }
        )
    attacks.sort(key=lambda event: event["peak_time"])
    if not attacks:
        raise ValueError(f"No attack clips in {report_path}")
    sync = report.get("sync_candidates", [])
    return {
        "title": item["title"],
        "analysis_id": report["analysis_id"],
        "engine_version": report["engine_version"],
        "source_sha256": source_sha,
        "report_path": str(report_path),
        "duration": report["source"]["duration_seconds"],
        "video": data_uri(video_path, "video/mp4"),
        "attacks": attacks,
        "starter_ids": starter_ids(attacks, sync),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    items = json.loads(manifest.read_text("utf-8"))
    references = [build_reference(item, manifest.parent) for item in items]
    ids = [ref["analysis_id"] for ref in references]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate analysis_id in manifest")
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "references": references,
    }
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("listening_review.html").read_text("utf-8")
    if template.count("__REVIEW_DATA__") != 1:
        raise ValueError("Expected one data placeholder")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REVIEW_DATA__", encoded), "utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "references": len(references),
                "clips": sum(len(ref["attacks"]) for ref in references),
                "starter_clips": sum(len(ref["starter_ids"]) for ref in references),
                "bytes": output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
