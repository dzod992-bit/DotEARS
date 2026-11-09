"""Offline broad-UI mixture training; select on calibration, then test once unchanged."""

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
from evaluate_sfx_presence import fit_head, save, stats  # noqa: E402
from evaluate_sfx_presence_deep import weighted_median  # noqa: E402

from dotears.audiosep import file_digest  # noqa: E402
from dotears.sfx_contrast import (  # noqa: E402
    FEATURE_NAMES,
    SCHEMA,
    VERSION,
    ContrastSFXEngine,
    decide,
    read_policy,
)
from dotears.sfx_events import balanced_source_weights  # noqa: E402
from dotears.sfx_presence_deep import score_head  # noqa: E402
from dotears.sfx_taxonomy import UI_TAXONOMY, UI_TYPES  # noqa: E402


def features(root):
    import psutil
    import soundfile as sf

    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    if file_digest(root / "corpus.json") != protocol["corpus_sha256"]:
        raise ValueError("Frozen mixture corpus changed.")
    rows = json.loads((root / "corpus.json").read_text("utf-8"))["rows"]
    engine = ContrastSFXEngine(
        Path("models/ced-mini"),
        Path("models/clap-htsat-unfused"),
        Path(".cache/sfx-contrast-cache"),
        4,
    )
    path = root / "features.jsonl"
    previous = set()
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            row = json.loads(line)
            if row["models"] != engine.identity or row["protocol_sha256"] != file_digest(
                root / "protocol.json"
            ):
                raise ValueError("Partial features have another identity.")
            previous.add(row["id"])
    started, peak = time.perf_counter(), 0
    for i, row in enumerate(rows, 1):
        if row["id"] in previous:
            continue
        peak = max(peak, psutil.Process().memory_info().rss)
        if time.perf_counter() - started > 2400 or peak > 5 * 1024**3:
            raise RuntimeError("Feature resource budget reached; progress retained.")
        if file_digest(Path(row["audio_path"])) != row["audio_sha256"]:
            raise ValueError("Mixture changed.")
        audio, rate = sf.read(row["audio_path"], dtype="float32")
        if rate != 32000:
            raise ValueError("Wrong mixture sample rate.")
        evidence = engine.analyze(audio, row["peak_relative"])
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        **row,
                        "evidence": evidence,
                        "models": engine.identity,
                        "protocol_sha256": file_digest(root / "protocol.json"),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        if i % 50 == 0:
            print(
                json.dumps(
                    {
                        "windows": i,
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
            "counters": engine.counters,
            "network_blocked": True,
        },
    )


def indices_for(variant):
    semantic = list(range(1024, len(FEATURE_NAMES)))
    if variant == "semantic":
        return semantic
    if variant == "raw_embedding":
        return list(range(512)) + list(range(1024, 1036)) + list(range(1048, len(FEATURE_NAMES)))
    return list(range(len(FEATURE_NAMES)))


def fit_variant(rows, label, regularization, variant, learner="linear"):
    indices = indices_for(variant)
    weights = source_weights(rows)
    if learner == "rbf":
        from evaluate_sfx_presence_rbf import fit_rbf

        return fit_rbf(
            rows, label, regularization, variant, active_indices=indices, sample_weights=weights
        )
    reduced = [
        {
            **r,
            "evidence": {
                **r["evidence"],
                "features": [r["evidence"]["features"][i] for i in indices],
            },
        }
        for r in rows
    ]
    head = fit_head(reduced, label, regularization, sample_weights=weights)
    head.update(
        kind="linear", feature_indices=indices, feature_count=len(indices), representation=variant
    )
    return head


def source_weights(rows):
    return balanced_source_weights(
        rows, label_key="label", allowed_labels=set(UI_TYPES) | {"other", "background"}
    )


def numeric_head(head):
    """Prepare numeric parameters once instead of reparsing large RBF lists per query."""
    result = dict(head)
    for key in ("mean", "scale", "coefficients", "support_vectors", "dual_coefficients"):
        if key in result:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def numeric_policy(policy):
    return {**policy, "heads": {k: numeric_head(h) for k, h in policy["heads"].items()}}


def fold_indices(rows):
    # GroupKFold keeps every source/augmentation of an author together.
    from sklearn.model_selection import GroupKFold

    # Background views can contain identical silence or short repeated music. Keep
    # all background controls together so these PCM identities cannot cross folds.
    authors = [
        "background-controls" if r["label"] == "background" else r["author_group"] for r in rows
    ]
    return list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), groups=authors))


def gate(check):
    return (
        check["precision"] is not None
        and check["precision"] >= 0.9
        and check["background_false"] == 0
        and check["primary_coverage"] is not None
        and check["primary_coverage"] >= 0.2
        and check["positive_source_count"] >= 2
    )


def fit(root):
    if (root / "selected-before-test.json").exists():
        raise ValueError("Do not refit a frozen mixture model selection.")
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    run = json.loads((root / "feature-run.json").read_text("utf-8"))
    if file_digest(root / "features.jsonl") != run["features_sha256"]:
        raise ValueError("Mixture features changed.")
    rows = [json.loads(line) for line in (root / "features.jsonl").read_text("utf-8").splitlines()]
    amendment_path = root / "curation-before-fitting.json"
    amendment = json.loads(amendment_path.read_text("utf-8")) if amendment_path.exists() else {}
    excluded = set(amendment.get("exclude_train_source_ids", []))
    if excluded and file_digest(root / "sources.json") != amendment.get("source_catalog_sha256"):
        raise ValueError("Curation belongs to another source catalog.")
    if excluded and (
        amendment.get("test_metrics_consulted") is not False
        or amendment.get("new_positive_labels_assigned") is not False
    ):
        raise ValueError("Curation must precede fitting and preserve unresolved original labels.")
    train = [r for r in rows if r["split"] == "train" and r["source_id"] not in excluded]
    quarantined = [r for r in rows if r["split"] == "train" and r["source_id"] in excluded]
    if excluded != {r["source_id"] for r in quarantined}:
        raise ValueError("Curation may only quarantine existing training sources.")
    calibration = [r for r in rows if r["split"] == "calibration"]
    splits = fold_indices(train)
    folds = []
    for a, b in splits:
        for key in ("author_group", "source_sha256", "audio_sha256"):
            if {train[i][key] for i in a} & {train[i][key] for i in b}:
                raise ValueError("Author/source/audio leaked across a fitting fold.")
        folds.append(
            {"train_ids": [train[i]["id"] for i in a], "query_ids": [train[i]["id"] for i in b]}
        )
    candidates = []
    weights = source_weights(train)
    plan_path = root / "learners-before-fitting.json"
    plan = (
        json.loads(plan_path.read_text("utf-8")) if plan_path.exists() else {"learners": ["linear"]}
    )
    if any(kind not in ("linear", "rbf") for kind in plan["learners"]):
        raise ValueError("Unsupported mixture classifier candidate.")
    for variant, learner in [
        (v, k) for v in protocol["feature_candidates"] for k in plan["learners"]
    ]:
        for regularization in protocol["C_values"]:
            oof = {label: np.full(len(train), np.nan) for label in UI_TYPES}
            for a, b in splits:
                for label in UI_TYPES:
                    head = fit_variant(
                        [train[i] for i in a], label, regularization, variant, learner
                    )
                    scoring_head = numeric_head(head)
                    for i in b:
                        oof[label][i] = score_head(scoring_head, train[i]["evidence"]["features"])
            heads = {}
            for label in UI_TYPES:
                values = oof[label]
                if not np.isfinite(values).all():
                    raise ValueError("Incomplete grouped predictions.")
                y = np.array([r["label"] == label for r in train])
                center = weighted_median(values[~y], weights[~y])
                positive = weighted_median(values[y], weights[y])
                heads[label] = fit_variant(train, label, regularization, variant, learner)
                heads[label].update(
                    competition_center=center, competition_scale=max(positive - center, 1.0)
                )
            supports = np.column_stack(
                [
                    (oof[label] - heads[label]["competition_center"])
                    / heads[label]["competition_scale"]
                    for label in UI_TYPES
                ]
            )
            ranks = np.argsort(-supports, axis=1, kind="stable")
            winners = [
                UI_TYPES[order[0]]
                if supports[i, order[0]] - supports[i, order[1]] >= 0.15
                and train[i]["evidence"]["rms"] >= 1e-5
                and train[i]["evidence"]["raw_target_scores"][UI_TYPES[order[0]]] > 0
                else "unknown"
                for i, order in enumerate(ranks)
            ]
            training_checks = {}
            for label in UI_TYPES:
                options = []
                for threshold in sorted(set(oof[label])):
                    predictions = [
                        label if winner == label and score >= threshold else "unknown"
                        for winner, score in zip(winners, oof[label], strict=True)
                    ]
                    check = stats(train, predictions, label)
                    if (
                        check["precision"] is not None
                        and check["precision"] >= 0.9
                        and check["true_positives"] >= 5
                        and len(check["positive_authors"]) >= 3
                        and check["background_false"] == 0
                        and check["blade_false"] == 0
                    ):
                        options.append(
                            (check["true_positives"], check["precision"], threshold, check)
                        )
                best = max(options, key=lambda v: v[:3]) if options else None
                heads[label].update(enabled=bool(best), threshold=best[2] if best else None)
                training_checks[label] = best[3] if best else None
            policy = {
                "schema_version": SCHEMA,
                "version": VERSION,
                "label_taxonomy": UI_TAXONOMY,
                "feature_names": FEATURE_NAMES,
                "models": run["models"],
                "heads": heads,
                "created_at": datetime.now(UTC).isoformat(),
                "protocol_sha256": file_digest(root / "protocol.json"),
                "features_sha256": run["features_sha256"],
                "video_used_for_fitting": False,
                "status": "broad_ui_source_labeled_mixture_research",
                "curation_sha256": file_digest(amendment_path) if amendment_path.exists() else None,
                "learner_plan_sha256": file_digest(plan_path) if plan_path.exists() else None,
            }
            scoring_policy = numeric_policy(policy)
            predictions = [decide(scoring_policy, r["evidence"])["label"] for r in calibration]
            checks = {label: stats(calibration, predictions, label) for label in UI_TYPES}
            for label, check in checks.items():
                check["passed"] = heads[label]["enabled"] and gate(check)
                if not check["passed"]:
                    heads[label]["enabled"] = False
            enabled = [label for label in UI_TYPES if heads[label]["enabled"]]
            quality = (
                len(enabled),
                sum(checks[k]["primary_coverage"] for k in enabled),
                sum(checks[k]["true_positives"] for k in enabled),
                regularization == 1.0,
                -protocol["feature_candidates"].index(variant),
                learner == "linear",
            )
            candidates.append(
                {
                    "policy": policy,
                    "variant": variant,
                    "learner": learner,
                    "C": regularization,
                    "quality": quality,
                    "oof": training_checks,
                    "calibration": checks,
                }
            )
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "learner": learner,
                        "C": regularization,
                        "enabled": enabled,
                        "calibration": {
                            k: {
                                key: v[key]
                                for key in (
                                    "true_positives",
                                    "false_positives",
                                    "primary_true",
                                    "primary_count",
                                )
                            }
                            for k, v in checks.items()
                        },
                    }
                ),
                flush=True,
            )
    chosen = max(candidates, key=lambda r: r["quality"])
    policy = copy.deepcopy(chosen["policy"])
    policy["selection"] = {
        "variant": chosen["variant"],
        "C": chosen["C"],
        "learner": chosen["learner"],
        "test_used": False,
    }
    save(root / "selected-before-test.json", policy)
    # There is one chosen candidate. Other candidates' test performance is never evaluated.
    test = [r for r in rows if r["split"] == "test"]
    predictions = [decide(policy, r["evidence"])["label"] for r in test]
    checks = {label: stats(test, predictions, label) for label in UI_TYPES}
    for label, check in checks.items():
        check["enabled_before"] = policy["heads"][label]["enabled"]
        check["passed"] = check["enabled_before"] and gate(check)
        if not check["passed"]:
            policy["heads"][label]["enabled"] = False
    policy["test_gate"] = checks
    save(root / "policy.json", policy)
    read_policy(root / "policy.json")
    result = {
        "candidates": candidates,
        "folds": folds,
        "selected": policy["selection"],
        "checks": checks,
        "enabled": [k for k, v in policy["heads"].items() if v["enabled"]],
        "policy_sha256": file_digest(root / "policy.json"),
        "selected_before_test_sha256": file_digest(root / "selected-before-test.json"),
        "test_decisions": [
            {"id": r["id"], "label": r["label"], "decision": p}
            for r, p in zip(test, predictions, strict=True)
        ],
        "effective_training_windows": len(train),
        "scope_ambiguous_regression": [
            {
                "id": r["id"],
                "original_label": r["label"],
                "decision": decide(policy, r["evidence"])["label"],
            }
            for r in quarantined
        ],
    }
    save(root / "evaluation.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k in ("enabled", "selected", "checks")}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("features", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    def reject(*_a, **_k):
        raise RuntimeError("No network during mixture inference or fitting.")

    socket.create_connection = reject
    socket.socket.connect = reject
    (features if args.phase == "features" else fit)(args.root.resolve())


if __name__ == "__main__":
    main()
