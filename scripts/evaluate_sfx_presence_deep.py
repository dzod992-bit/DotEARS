"""Latent raw-audio model selection on calibration authors, then a fresh-author test."""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

import numpy as np  # noqa: E402
from evaluate_sfx_presence import fit_head, save, score, stats  # noqa: E402

from dotears.audiosep import file_digest  # noqa: E402
from dotears.sfx_events import balanced_source_weights  # noqa: E402
from dotears.sfx_presence import FEATURE_NAMES as BASE_NAMES  # noqa: E402
from dotears.sfx_presence import TYPES  # noqa: E402
from dotears.sfx_presence_deep import (  # noqa: E402
    FEATURE_NAMES,
    SCHEMA,
    VERSION,
    LatentSFXEngine,
    decide,
    read_policy,
)


def features(root):
    import psutil
    import soundfile as sf

    def reject(*a, **k):
        raise RuntimeError("No network during latent SFX inference.")

    socket.create_connection = reject
    socket.socket.connect = reject
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    if file_digest(root / "corpus.json") != protocol["corpus_sha256"]:
        raise ValueError("Corpus changed.")
    rows = json.loads((root / "corpus.json").read_text("utf-8"))["rows"]
    engine = LatentSFXEngine(
        Path("models/ced-mini"),
        Path("models/clap-htsat-unfused"),
        Path(".cache/sfx-presence-cache"),
        4,
    )
    started = time.perf_counter()
    peak = 0
    path = root / "features.jsonl"
    previous = set()
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            row = json.loads(line)
            if row["models"] != engine.identity or row["protocol_sha256"] != file_digest(
                root / "protocol.json"
            ):
                raise ValueError("Partial features changed identity.")
            previous.add(row["id"])
    for index, row in enumerate(rows, 1):
        if row["id"] in previous:
            continue
        peak = max(peak, psutil.Process().memory_info().rss)
        if time.perf_counter() - started > 2400 or peak > 5 * 1024**3:
            raise RuntimeError("Resource budget reached; cache retained.")
        if file_digest(Path(row["audio_path"])) != row["audio_sha256"]:
            raise ValueError("Input changed.")
        audio, rate = sf.read(row["audio_path"], dtype="float32")
        if rate != 32000:
            raise ValueError("Wrong input rate.")
        evidence = engine.analyze(audio, row["peak_relative"])
        result = {
            **row,
            "evidence": evidence,
            "models": engine.identity,
            "protocol_sha256": file_digest(root / "protocol.json"),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        if index % 40 == 0:
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
            "features_sha256": file_digest(path),
            "seconds": time.perf_counter() - started,
            "sampled_peak_rss": peak,
            "network_blocked": True,
            "counters": engine.counters,
        },
    )


def weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2)]])


def fit_variant(rows, label, regularization, variant):
    if variant == "semantic":
        reduced = [
            {
                **r,
                "evidence": {
                    **r["evidence"],
                    "features": r["evidence"]["features"][: len(BASE_NAMES)],
                },
            }
            for r in rows
        ]
        head = fit_head(reduced, label, regularization)
        extra = len(FEATURE_NAMES) - len(BASE_NAMES)
        head["mean"] += [0.0] * extra
        head["scale"] += [1.0] * extra
        head["coefficients"] += [0.0] * extra
    else:
        head = fit_head(rows, label, regularization)
    head["representation"] = variant
    return head


def fit(root, evaluate_test=True):
    if (root / "policy.json").exists():
        raise ValueError("Do not refit a frozen policy.")
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    run = json.loads((root / "feature-run.json").read_text("utf-8"))
    if file_digest(root / "features.jsonl") != run["features_sha256"]:
        raise ValueError("Features changed.")
    rows = [json.loads(line) for line in (root / "features.jsonl").read_text("utf-8").splitlines()]
    train = [r for r in rows if r["split"] == "train"]
    calib = [r for r in rows if r["split"] == "calibration"]
    weights = balanced_source_weights(train)
    candidates = []
    folds = []
    for variant in ("semantic", "latent"):
        for regularization in (0.1, 1.0, 10.0):
            oof = {label: np.full(len(train), np.nan) for label in TYPES}
            for author in sorted({r["author_group"] for r in train}):
                indices = [i for i, r in enumerate(train) if r["author_group"] == author]
                hashes = {train[i]["source_sha256"] for i in indices}
                audio_hashes = {train[i]["audio_sha256"] for i in indices}
                selected = [
                    r
                    for r in train
                    if r["author_group"] != author
                    and r["source_sha256"] not in hashes
                    and r["audio_sha256"] not in audio_hashes
                ]
                for label in TYPES:
                    head = fit_variant(selected, label, regularization, variant)
                    for i in indices:
                        oof[label][i] = score(head, train[i])
                if variant == "semantic" and regularization == 0.1:
                    folds.append(
                        {
                            "author": author,
                            "training_ids": [r["id"] for r in selected],
                            "query_ids": [train[i]["id"] for i in indices],
                        }
                    )
            heads = {}
            for label in TYPES:
                values = oof[label]
                if not np.isfinite(values).all():
                    raise ValueError("Incomplete OOF predictions.")
                y = np.array([r["label"] == label for r in train])
                center = weighted_median(values[~y], weights[~y])
                pos = weighted_median(values[y], weights[y])
                heads[label] = fit_variant(train, label, regularization, variant)
                heads[label].update(
                    competition_center=center, competition_scale=max(pos - center, 1.0)
                )
            supports = np.column_stack(
                [
                    (oof[label] - heads[label]["competition_center"])
                    / heads[label]["competition_scale"]
                    for label in TYPES
                ]
            )
            rankings = np.argsort(-supports, axis=1, kind="stable")
            winners = [
                TYPES[order[0]]
                if supports[i, order[0]] - supports[i, order[1]] >= 0.15
                and train[i]["evidence"]["rms"] >= 1e-5
                and train[i]["evidence"]["raw_target_scores"][TYPES[order[0]]] > 0
                else "unknown"
                for i, order in enumerate(rankings)
            ]
            oof_stats = {}
            for label in TYPES:
                options = []
                for threshold in sorted(set(oof[label])):
                    predictions = [
                        label if winner == label and value >= threshold else "unknown"
                        for winner, value in zip(winners, oof[label], strict=True)
                    ]
                    result = stats(train, predictions, label)
                    if (
                        result["precision"] is not None
                        and result["precision"] >= 0.95
                        and result["true_positives"] >= 5
                        and len(result["positive_authors"]) >= 2
                        and result["positive_source_count"] >= 3
                        and result["background_false"] == 0
                        and result["blade_false"] == 0
                    ):
                        options.append(
                            (result["true_positives"], result["precision"], threshold, result)
                        )
                best = max(options, key=lambda v: v[:3]) if options else None
                heads[label].update(enabled=bool(best), threshold=best[2] if best else None)
                oof_stats[label] = {
                    "passed": bool(best),
                    "statistics": best[3] if best else None,
                    "scores": oof[label].tolist(),
                }
            policy = {
                "schema_version": SCHEMA,
                "version": VERSION,
                "feature_names": FEATURE_NAMES,
                "models": run["models"],
                "heads": heads,
                "created_at": datetime.now(UTC).isoformat(),
                "video_used_for_fitting": False,
                "protocol_sha256": file_digest(root / "protocol.json"),
                "feature_evidence_sha256": run["features_sha256"],
                "decision_rule": "raw_audio_heads_compete_including_disabled_types",
                "margin": 0.15,
                "status": "provisional_author_caption_and_human_reference_training",
            }
            predictions = [decide(policy, r["evidence"])["label"] for r in calib]
            checks = {}
            for label in TYPES:
                check = stats(calib, predictions, label)
                passed = (
                    heads[label]["enabled"]
                    and check["precision"] is not None
                    and check["precision"] >= 0.9
                    and check["background_false"] == 0
                    and check["primary_coverage"] is not None
                    and check["primary_coverage"] >= 0.2
                )
                checks[label] = {**check, "passed": passed}
                if not passed:
                    heads[label]["enabled"] = False
            enabled = [label for label in TYPES if heads[label]["enabled"]]
            quality = (
                len(enabled),
                sum(checks[label]["primary_coverage"] for label in enabled),
                sum(checks[label]["true_positives"] for label in enabled),
                -abs(np.log10(regularization)),
                variant == "semantic",
            )
            candidates.append(
                {
                    "policy": policy,
                    "representation": variant,
                    "C": regularization,
                    "quality": quality,
                    "oof": oof_stats,
                    "calibration": checks,
                }
            )
            print(
                json.dumps(
                    {
                        "candidate": variant,
                        "C": regularization,
                        "enabled_after_calibration": enabled,
                        "primary_true": {label: checks[label]["primary_true"] for label in TYPES},
                    }
                ),
                flush=True,
            )
    chosen = max(candidates, key=lambda item: item["quality"])
    # Test gates may only disable a copy; retain calibration candidates unchanged.
    policy = copy.deepcopy(chosen["policy"])
    save(root / "selected-before-test.json", policy)
    if not evaluate_test:
        save(
            root / "candidates-only.json",
            {
                "folds": folds,
                "candidates": candidates,
                "selection_used_test_data": False,
                "selected": {"representation": chosen["representation"], "C": chosen["C"]},
            },
        )
        print(
            json.dumps(
                {
                    "calibration_selection": {
                        "representation": chosen["representation"],
                        "C": chosen["C"],
                    },
                    "test_not_evaluated": True,
                }
            ),
            flush=True,
        )
        return
    checks = {}
    for split in ("test", "regression"):
        selected = [r for r in rows if r["split"] == split]
        predictions = [decide(policy, r["evidence"])["label"] for r in selected]
        checks[split] = {label: stats(selected, predictions, label) for label in TYPES}
        if split == "test":
            for label, result in checks[split].items():
                passed = (
                    result["precision"] is not None
                    and result["precision"] >= 0.9
                    and result["background_false"] == 0
                    and result["primary_coverage"] is not None
                    and result["primary_coverage"] >= 0.2
                )
                result["passed"] = passed
                result["enabled_before"] = policy["heads"][label]["enabled"]
                if policy["heads"][label]["enabled"] and not passed:
                    policy["heads"][label]["enabled"] = False
    policy["fresh_author_test"] = checks["test"]
    save(root / "policy.json", policy)
    read_policy(root / "policy.json")
    save(
        root / "evaluation.json",
        {
            "folds": folds,
            "candidates": candidates,
            "selected": {"representation": chosen["representation"], "C": chosen["C"]},
            "checks": checks,
            "enabled": [label for label in TYPES if policy["heads"][label]["enabled"]],
            "selected_before_test_sha256": file_digest(root / "selected-before-test.json"),
            "policy_sha256": file_digest(root / "policy.json"),
            "source_label_caveat": protocol["label_scope"],
        },
    )
    print(
        json.dumps(
            {
                "enabled": [label for label in TYPES if policy["heads"][label]["enabled"]],
                "selected": {"variant": chosen["representation"], "C": chosen["C"]},
                "fresh_author_test": checks["test"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("features", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "features":
        features(root)
    else:
        fit(root)


if __name__ == "__main__":
    main()
