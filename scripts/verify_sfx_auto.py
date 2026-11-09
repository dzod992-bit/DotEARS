"""Audit automatic SFX evidence and optional fresh-model numerical reproduction."""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from dotears.audiosep import file_digest  # noqa: E402
from dotears.sfx_auto import (  # noqa: E402
    FEATURE_NAMES,
    QUERIES,
    CandidateEngine,
    decision,
    focus_window,
    read_policy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--live-check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    features_path = root / "calibration/features.jsonl"
    rows = [json.loads(line) for line in features_path.read_text("utf-8").splitlines()]
    policy_path = root / "calibration/policy.json"
    policy = read_policy(policy_path)
    scan = json.loads(args.scan.read_text("utf-8"))
    calibration = json.loads((root / "calibration/calibration.json").read_text("utf-8"))
    audit_paths = [
        Path("validation-output/sfx-labeled-eval-2026-10-28/audit.json"),
        Path("validation-output/sfx-reference-eval-2026-10-28/round2-review/audit.json"),
    ]
    human_sources = {}
    for audit_path in audit_paths:
        audit = json.loads(audit_path.read_text("utf-8"))
        for source_row in audit["rows"]:
            if source_row.get("primary_evaluation_eligible", True):
                human_sources[source_row["id"]] = source_row
    clean = json.loads(
        Path("validation-output/sfx-events-2026-10-29/corpus.json").read_text("utf-8")
    )["clean_sources"]
    if len(clean) != 30:
        raise ValueError("Unexpected human source set.")
    for item in clean:
        human = human_sources[item["id"]]
        if (
            item["label"] != human["label"]
            or item["family"] != human["family"]
            or item["source_sha256"] != human["source_sha256"]
            or file_digest(Path(item["source_path"])) != item["source_sha256"]
            or file_digest(Path(item["audio_path"])) != item["audio_sha256"]
        ):
            raise ValueError("Human class/source binding changed.")
    if file_digest(features_path) != policy["features_sha256"]:
        raise ValueError("Feature evidence changed after policy fit.")
    if file_digest(policy_path) != scan["policy_sha256"]:
        raise ValueError("Policy changed after scan.")
    if scan["models"] != policy["models"] or scan["queries"] != QUERIES:
        raise ValueError("Scan used incompatible models or queries.")
    lookup = {row["id"]: row for row in rows}
    for fold in calibration["folds"]:
        queries = [lookup[key] for key in fold["query_ids"]]
        train = [lookup[key] for key in fold["training_ids"]]
        if (
            {row["family"] for row in queries} & {row["family"] for row in train}
            or {row["source_sha256"] for row in queries} & {row["source_sha256"] for row in train}
            or {row["audio_sha256"] for row in queries} & {row["audio_sha256"] for row in train}
        ):
            raise ValueError("Source-family fold leakage.")
    verified_estimates = 0
    for row in rows:
        if file_digest(Path(row["audio_path"])) != row["audio_sha256"]:
            raise ValueError("Changed calibration input.")
        if set(row["candidates"]) != set(QUERIES) or row["models"] != policy["models"]:
            raise ValueError("Missing/wrong fixed query.")
        if row["human_label"] != "background":
            human = human_sources[row["source_id"]]
            if (
                row["human_label"] != human["label"]
                or row["family"] != human["family"]
                or row["source_sha256"] != human["source_sha256"]
            ):
                raise ValueError("Calibration label differs from the human source label.")
        for label, evidence in row["candidates"].items():
            if evidence["query"] != QUERIES[label] or len(evidence["features"]) != len(
                FEATURE_NAMES
            ):
                raise ValueError("Wrong prompt/features.")
            if file_digest(Path(evidence["estimate_path"])) != evidence["estimate_sha256"]:
                raise ValueError("Changed separator estimate.")
            verified_estimates += 1
    source = Path(scan["source_report_path"])
    if file_digest(source) != scan["source_report_sha256"]:
        raise ValueError("Original report changed.")
    original = json.loads(source.read_text("utf-8"))
    native = (source.parent / original["audio"]["native_path"]).resolve()
    if (
        not native.is_relative_to(source.parent.resolve())
        or file_digest(native) != scan["native_audio_sha256"]
    ):
        raise ValueError("Original native audio changed.")
    expected = {row["id"]: row for row in original["events"] if row["category"] == "audio_attack"}
    if len(scan["events"]) != len(expected) or set(expected) != {
        row["event_id"] for row in scan["events"]
    }:
        raise ValueError("Missing or duplicated video attacks.")
    verified_focus = 0
    for row in scan["events"]:
        if row["peak_time"] != expected[row["event_id"]]["peak_time"]:
            raise ValueError("Changed peak time.")
        if decision(policy, row["feature_evidence"])["label"] != row["label"]:
            raise ValueError("Stored type disagrees with frozen policy.")
        mixture = args.scan.parent / row["mixture_focus_path"]
        if file_digest(mixture) != row["mixture_focus_sha256"]:
            raise ValueError("Changed mixture focus.")
        mix, rate = sf.read(mixture, dtype="float32")
        if rate != 32000 or mix.ndim != 1 or len(mix) > 22401:
            raise ValueError("Invalid raw focus window.")
        verified_focus += 1
        for evidence in row["feature_evidence"].values():
            full_path = Path(evidence["estimate_path"])
            full, rate = sf.read(full_path, dtype="float32")
            focus = args.scan.parent / evidence["focus_audio_path"]
            values, focus_rate = sf.read(focus, dtype="float32")
            if (
                file_digest(full_path) != evidence["estimate_sha256"]
                or file_digest(focus) != evidence["focus_audio_sha256"]
                or rate != focus_rate
                or rate != 32000
                or not np.array_equal(
                    values, focus_window(full, row["peak_time"] - row["context_start"])
                )
            ):
                raise ValueError("Focus must be the exact unnormalized separator samples.")
            verified_focus += 1
    live = None
    if args.live_check:

        def reject(*_args, **_kwargs):
            raise RuntimeError("Verification attempted a network connection.")

        socket.create_connection = reject
        socket.socket.connect = reject
        model = CandidateEngine(
            Path("models/audiosep"),
            Path("models/clap-htsat-unfused"),
            root / "verification-live-cache",
            4,
        )
        query = next(row for row in rows if row["split"] == "A" and row["kind"] == "mixed")
        waveform, rate = sf.read(query["audio_path"], dtype="float32")
        if rate != 32000:
            raise ValueError("Invalid live verification input rate.")
        actual = model.analyze_window(waveform, query["peak_relative"])
        error = max(
            float(
                np.max(
                    np.abs(
                        np.array(actual[label]["features"]) - query["candidates"][label]["features"]
                    )
                )
            )
            for label in QUERIES
        )
        if error > 2e-5:
            raise ValueError("Fresh model run did not reproduce calibration features.")
        live = {
            "case_id": query["id"],
            "all_three_queries": True,
            "maximum_feature_absolute_error": error,
            "fresh_cache": str(model.cache),
            "network_blocked": True,
        }
    result = {
        "passed": True,
        "created_at": datetime.now(UTC).isoformat(),
        "family_folds_without_leakage": len(calibration["folds"]),
        "human_source_bindings_verified": len(clean),
        "human_audit_sha256": {str(path): file_digest(path) for path in audit_paths},
        "calibration_windows": len(rows),
        "verified_query_estimates": verified_estimates,
        "video_attacks": len(scan["events"]),
        "verified_video_focus_files": verified_focus,
        "original_report_sha256": file_digest(source),
        "original_native_sha256": file_digest(native),
        "policy_sha256": file_digest(policy_path),
        "scan_sha256": file_digest(args.scan),
        "policy_frozen_before_video": datetime.fromisoformat(policy["created_at"])
        < datetime.fromisoformat(scan["created_at"]),
        "fresh_model_reproduction": live,
        "code_sha256": {
            str(path): file_digest(path)
            for path in [
                Path("src/dotears/sfx_auto.py"),
                Path("scripts/calibrate_sfx_auto.py"),
                Path("scripts/compare_sfx_auto.py"),
                Path("scripts/verify_sfx_auto.py"),
            ]
        },
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False), "utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
