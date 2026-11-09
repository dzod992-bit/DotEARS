"""Build, measure and freeze a small local abstention policy without the user's video.

Run prepare, features and fit separately. A uses music/speech A; validation B uses
different backgrounds, with the same foreground sources. This is a development
check, not an independent, representative test of commercial motion sound design.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

import numpy as np  # noqa: E402

from dotears.audiosep import PINNED_FILES, file_digest  # noqa: E402
from dotears.sfx import PROMPTS  # noqa: E402
from dotears.sfx_auto import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    POLICY_SCHEMA,
    QUERIES,
    CandidateEngine,
    decision,
    json_hash,
    read_policy,
)
from dotears.sfx_events import balanced_source_weights  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def prepare(root: Path, a: Path, b: Path) -> None:
    import soundfile as sf

    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        raise ValueError("Do not overwrite a frozen automatic-SFX protocol.")
    rows = []
    for split, source in (("A", a), ("B", b)):
        corpus = json.loads((source / "corpus.json").read_text("utf-8"))
        cases = corpus["cases"]
        chosen = {}
        for case in cases:
            use = case["kind"] in {
                "clean_preservation",
                "other_control",
                "background_control",
                "silence_control",
            }
            use |= case["kind"] == "known_mixture" and case["snr_db"] == 0
            if use:
                chosen.setdefault(case["input"]["path"], case)
        for case in chosen.values():
            path = Path(case["input"]["path"])
            if file_digest(path) != case["input"]["sha256"]:
                raise ValueError("Changed prepared calibration input.")
            kind = case["kind"]
            identity = path.stem
            background = case.get("background", "none")
            family = case.get("family", "background-" + background)
            label = case.get("human_label", "background")
            row = {
                "id": split + "-" + identity,
                "split": split,
                "kind": "mixed"
                if kind == "known_mixture"
                else "clean"
                if kind in {"clean_preservation", "other_control"}
                else "background",
                "audio_path": str(path.resolve()),
                "audio_sha256": file_digest(path),
                "human_label": label,
                "family": family,
                "source_sha256": case.get("source_sha256", file_digest(path)),
                "source_id": case.get("source_id", identity),
                "peak_relative": 1.0,
                "background": background,
            }
            rows.append(row)
        # Also mix ALL eight human 'other' sounds with backgrounds. A wrong query
        # is measured on these too; no oracle class chooses a separator call.
        clean_others = [r for r in rows if r["split"] == split and r["human_label"] == "other"]
        sources = json.loads(
            Path("validation-output/sfx-events-2026-10-29/corpus.json").read_text("utf-8")
        )["clean_sources"]
        source_order = {row["id"]: index for index, row in enumerate(sources)}
        for row in clean_others:
            target, rate = sf.read(row["audio_path"], dtype="float64")
            nonzero = np.flatnonzero(np.abs(target) > 1e-8)
            first, last = max(32000, int(nonzero[0])), int(nonzero[-1]) + 1
            for background in ("music", "speech", "music_speech"):
                index = source_order[row["source_id"]] % 4
                bed, bed_rate = sf.read(
                    source / "inputs" / f"background-{background}-{index}.wav", dtype="float64"
                )
                if rate != 32000 or bed_rate != rate or bed.shape != target.shape:
                    raise ValueError("Mismatched full-window calibration audio.")
                foreground = target * (
                    np.sqrt(np.mean(bed[first:last] ** 2))
                    / max(np.sqrt(np.mean(target[first:last] ** 2)), 1e-12)
                )
                mixed = foreground + bed
                mixed *= min(1.0, 0.9 / max(np.max(np.abs(mixed)), 1e-12))
                identity = f"{split}-mix-{row['source_id']}-{background}-other"
                path = root / (identity + ".wav")
                sf.write(path, mixed.astype(np.float32), rate, subtype="FLOAT")
                rows.append(
                    {
                        **row,
                        "id": identity,
                        "audio_path": str(path),
                        "audio_sha256": file_digest(path),
                        "kind": "mixed",
                        "background": background,
                    }
                )
    if Counter(row["split"] for row in rows) != {"A": 133, "B": 133}:
        raise ValueError("Expected 133 windows per background set, including other mixtures.")
    save(root / "corpus.json", {"rows": rows, "sample_rate": 32000})
    protocol = {
        "schema_version": "dotears.sfx-auto-calibration-protocol/1",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_sha256": file_digest(root / "corpus.json"),
        "queries": QUERIES,
        "query_selection": "global choice from preceding AudioSep study, never chosen per event",
        "feature_names": FEATURE_NAMES,
        "feature_version": FEATURE_VERSION,
        "all_three_queries_for_every_window": True,
        "video_in_training_or_threshold_selection": False,
        "training": "A only; binary logistic C=1, source-balanced, weighted standardization",
        "threshold_selection": {
            "data": "A source-family-held-out predictions",
            "min_precision": 0.95,
            "min_true_positives": 3,
            "min_positive_families": 2,
            "objective": "maximum true positives subject to constraints",
        },
        "validation": {
            "data": "B mixed/background only, no duplicated clean views",
            "min_precision": 0.9,
            "min_true_positives": 3,
            "min_positive_families": 2,
            "max_background_false_accepts": 0,
            "max_blade_false_accepts": 0,
            "class_disabling": "monotonic fixed point; no threshold refit on B",
        },
        "veto": {
            "min_rms": 1e-5,
            "min_estimate_energy_db": -40,
            "estimated_target_must_beat_background": True,
            "ambiguous_excess_margin": 0.5,
        },
        "limits": {"seconds": 2400, "rss_bytes": 5 * 1024**3, "threads": 4},
        "scope": (
            "30 human-rated clean sources, 10 families; "
            "synthetic 0dB mixtures not separately listened"
        ),
        "validation_limitation": "B has different backgrounds but the same foreground sources as A",
        "foreground_generalization_diagnostic": (
            "A folds exclude all views of the held-out source family"
        ),
        "calibration_source_directories": {"A": str(a), "B": str(b)},
    }
    save(root / "protocol.json", protocol)
    print(json.dumps({"prepared": len(rows), "protocol_frozen": True}))


def block_network():
    def reject(*_args, **_kwargs):
        raise RuntimeError("Automatic SFX inference attempted network access.")

    socket.create_connection = reject
    socket.socket.connect = reject


def features(root: Path, audiosep: Path, clap: Path, cache: Path) -> None:
    import psutil
    import soundfile as sf

    block_network()
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    if file_digest(root / "corpus.json") != protocol["corpus_sha256"]:
        raise ValueError("Calibration corpus changed after freeze.")
    rows = json.loads((root / "corpus.json").read_text("utf-8"))["rows"]
    started, peak_rss = time.perf_counter(), 0
    engine = CandidateEngine(audiosep, clap, cache, 4)
    # Import only hash-verified, matching outputs from completed offline studies.
    previous = Path("validation-output/audiosep-2026-10-29")
    imported = 0
    for directory in (previous, previous / "query-variants"):
        result = json.loads((directory / "results.json").read_text("utf-8"))
        protocol_path = directory / "protocol.json"
        if result["protocol_sha256"] != file_digest(protocol_path):
            raise ValueError("Previous separation protocol changed.")
        if (
            directory == previous
            and result["model_load"]["checkpoint_sha256"]
            != PINNED_FILES["audiosep_base_4M_steps.ckpt"]
        ):
            raise ValueError("Previous estimates used another checkpoint.")
        for row in result["rows"]:
            if row["prompt"] not in QUERIES.values() or row["kind"] == "real_diagnostic":
                continue
            original, output = Path(row["input"]["path"]), Path(row["output_path"])
            if (
                file_digest(original) != row["input"]["sha256"]
                or file_digest(output) != row["output_sha256"]
            ):
                raise ValueError("Previous input or estimate changed.")
            waveform, rate = sf.read(original, dtype="float32")
            if rate != 32000:
                raise ValueError("Incompatible previous sample rate.")
            engine.seed_verified_estimate(waveform, row["prompt"], output)
            imported += 1
    progress = root / "features.jsonl"
    completed = {}
    if progress.exists():
        for line in progress.read_text("utf-8").splitlines():
            item = json.loads(line)
            if (
                item["protocol_sha256"] != file_digest(root / "protocol.json")
                or item["models"] != engine.identity
            ):
                raise ValueError("Partial features belong to a different experiment.")
            completed[item["id"]] = item
    for index, row in enumerate(rows, 1):
        if row["id"] in completed:
            continue
        if file_digest(Path(row["audio_path"])) != row["audio_sha256"]:
            raise ValueError("Calibration audio changed.")
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        if (
            time.perf_counter() - started > protocol["limits"]["seconds"]
            or peak_rss > protocol["limits"]["rss_bytes"]
        ):
            raise RuntimeError(
                "Resource budget reached; verified progress/cache retained for resume."
            )
        waveform, rate = sf.read(row["audio_path"], dtype="float32")
        if rate != 32000:
            raise ValueError("Calibration audio must be 32kHz mono.")
        # Only PCM and a constant peak enter inference; labels remain outside the engine.
        evidence = engine.analyze_window(waveform, row["peak_relative"])
        result = {
            **row,
            "candidates": evidence,
            "models": engine.identity,
            "protocol_sha256": file_digest(root / "protocol.json"),
        }
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        if index % 5 == 0:
            print(
                json.dumps(
                    {
                        "windows": index,
                        "total": len(rows),
                        "seconds": round(time.perf_counter() - started, 1),
                        "counters": engine.counters,
                    }
                ),
                flush=True,
            )
    save(
        root / "feature-run.json",
        {
            "completed": len(rows),
            "models": engine.identity,
            "seconds": round(time.perf_counter() - started, 3),
            "peak_sampled_rss": peak_rss,
            "counters": engine.counters,
            "verified_previous_estimates_imported": imported,
            "network_connections_blocked": True,
            "features_sha256": file_digest(progress),
        },
    )


def fit_head(rows, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    x = np.array([row["candidates"][label]["features"] for row in rows])
    y = np.array([row["human_label"] == label for row in rows])
    if len(set(y)) != 2:
        raise ValueError("A held-out fold has no positive or negative training examples.")
    weight = balanced_source_weights(rows)
    for value in (False, True):
        weight[y == value] *= len(rows) / 2 / weight[y == value].sum()
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(x, sample_weight=weight)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4)
        model.fit(scaler.transform(x), y, sample_weight=weight)
    if max(model.n_iter_) >= 1000:
        raise ValueError("A calibration head did not converge.")
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "enabled": True,
        "threshold": -1e100,
        "score_scale": 1.0,
    }


def score(head, row, label):
    value = np.array(row["candidates"][label]["features"])
    return float(
        ((value - head["mean"]) / head["scale"]) @ head["coefficients"] + head["intercept"]
    )


def eligible(row, label):
    c = row["candidates"][label]
    return (
        c["raw_rms"] >= 1e-5
        and c["estimate_rms"] >= 1e-5
        and c["energy_ratio_db"] >= -40
        and c["estimated_target_cosine"] >= c["estimated_best_control_cosine"]
    )


def statistics(rows, labels, target):
    accepts = [(row, label) for row, label in zip(rows, labels, strict=True) if label == target]
    true = [row for row, _ in accepts if row["human_label"] == target]
    false = [row for row, _ in accepts if row["human_label"] != target]
    return {
        "accepted": len(accepts),
        "true_positives": len(true),
        "false_positives": len(false),
        "precision": len(true) / len(accepts) if accepts else None,
        "positive_families": sorted({row["family"] for row in true}),
        "background_false_accepts": sum(row["human_label"] == "background" for row in false),
        "blade_false_accepts": sum("starninjas" in row["family"].lower() for row in false),
        "false_accept_ids": [row["id"] for row in false],
        "true_accept_ids": [row["id"] for row in true],
    }


def fit(root):
    if (root / "policy.json").exists():
        raise ValueError("A frozen policy exists; do not refit it.")
    run = json.loads((root / "feature-run.json").read_text("utf-8"))
    if file_digest(root / "features.jsonl") != run["features_sha256"]:
        raise ValueError("Completed feature evidence changed.")
    all_rows = [
        json.loads(line) for line in (root / "features.jsonl").read_text("utf-8").splitlines()
    ]
    rows = [row for row in all_rows if row["split"] == "A"]
    validation = [
        row
        for row in all_rows
        if row["split"] == "B" and row["kind"] != "clean" and not row["id"].endswith("silence")
    ]
    if len(rows) != 133 or len(validation) != 102 or len(all_rows) != 266:
        raise ValueError("Incomplete calibration features.")
    oof = {label: np.full(len(rows), np.nan) for label in QUERIES}
    folds = []
    for family in sorted({row["family"] for row in rows}):
        query = [i for i, row in enumerate(rows) if row["family"] == family]
        hashes = {rows[i]["source_sha256"] for i in query}
        audio_hashes = {rows[i]["audio_sha256"] for i in query}
        train = [
            row
            for row in rows
            if row["family"] != family
            and row["source_sha256"] not in hashes
            and row["audio_sha256"] not in audio_hashes
        ]
        for label in QUERIES:
            head = fit_head(train, label)
            for i in query:
                oof[label][i] = score(head, rows[i], label)
        folds.append(
            {
                "held_out_family": family,
                "query_ids": [rows[i]["id"] for i in query],
                "training_ids": [r["id"] for r in train],
                "source_and_audio_hash_exclusion": True,
            }
        )
    heads, calibration = {}, {}
    for label in QUERIES:
        scores = oof[label]
        if not np.isfinite(scores).all():
            raise ValueError("Incomplete family-held-out predictions.")
        options = []
        for threshold in sorted(set(scores)):
            selected = [
                label if value >= threshold and eligible(row, label) else "unknown"
                for row, value in zip(rows, scores, strict=True)
            ]
            stats = statistics(rows, selected, label)
            if (
                stats["precision"] is not None
                and stats["precision"] >= 0.95
                and stats["true_positives"] >= 3
                and len(stats["positive_families"]) >= 2
            ):
                options.append((stats["true_positives"], stats["precision"], threshold, stats))
        best = max(options, key=lambda value: value[:3]) if options else None
        head = fit_head(rows, label)
        head.update(
            threshold=best[2] if best else None,
            enabled=bool(best),
            score_scale=max(float(np.subtract(*np.percentile(scores, [75, 25]))), 1.0),
        )
        heads[label] = head
        calibration[label] = {
            "passed": bool(best),
            "statistics": best[3] if best else None,
            "oof_scores": scores.tolist(),
            "reason": "constraints_satisfied"
            if best
            else "no_threshold_met_family_precision_support_constraints",
        }
    policy = {
        "schema_version": POLICY_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "feature_names": FEATURE_NAMES,
        "feature_version": FEATURE_VERSION,
        "prompt_sha256": json_hash(PROMPTS),
        "queries": QUERIES,
        "models": run["models"],
        "heads": heads,
        "status": "small_set_local_policy_not_general_validation",
        "calibration_protocol_sha256": file_digest(root / "protocol.json"),
        "features_sha256": run["features_sha256"],
        "video_used_for_fitting": False,
    }
    validation_rounds = []
    for _ in range(4):
        predictions = [decision(policy, row["candidates"])["label"] for row in validation]
        round_stats, disable = {}, []
        for label in QUERIES:
            stats = statistics(validation, predictions, label)
            passed = (
                stats["precision"] is not None
                and stats["precision"] >= 0.90
                and stats["true_positives"] >= 3
                and len(stats["positive_families"]) >= 2
                and stats["background_false_accepts"] == stats["blade_false_accepts"] == 0
            )
            round_stats[label] = {
                **stats,
                "passed": passed,
                "enabled_before": heads[label]["enabled"],
            }
            if heads[label]["enabled"] and not passed:
                disable.append(label)
        validation_rounds.append(round_stats)
        if not disable:
            break
        for label in disable:
            heads[label]["enabled"] = False
    policy["local_gate_evidence"] = {
        "A_family_oof_thresholds": calibration,
        "B_validation_rounds": validation_rounds,
        "validation_has_shared_foreground_sources": True,
    }
    save(root / "policy.json", policy)
    read_policy(root / "policy.json")
    save(
        root / "calibration.json",
        {
            "folds": folds,
            "A_rows": [r["id"] for r in rows],
            "B_rows": [r["id"] for r in validation],
            "calibration": calibration,
            "validation_rounds": validation_rounds,
            "final_enabled_classes": [label for label in QUERIES if heads[label]["enabled"]],
            "policy_sha256": file_digest(root / "policy.json"),
            "frozen_before_video_evaluation": True,
        },
    )
    print(
        json.dumps(
            {
                "enabled": {label: heads[label]["enabled"] for label in QUERIES},
                "policy_sha256": file_digest(root / "policy.json"),
                "validation": validation_rounds[-1],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "features", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a", type=Path)
    parser.add_argument("--b", type=Path)
    parser.add_argument("--audiosep", type=Path, default=Path("models/audiosep"))
    parser.add_argument("--clap", type=Path, default=Path("models/clap-htsat-unfused"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/sfx-auto-cache"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "prepare":
        prepare(root, args.a.resolve(), args.b.resolve())
    elif args.phase == "features":
        features(root, args.audiosep, args.clap, args.cache)
    else:
        fit(root)


if __name__ == "__main__":
    main()
