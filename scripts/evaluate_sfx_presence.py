"""Offline raw-presence features, author-held-out calibration and independent test."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

import numpy as np  # noqa: E402

from dotears.audiosep import file_digest  # noqa: E402
from dotears.sfx_events import balanced_source_weights  # noqa: E402
from dotears.sfx_presence import (  # noqa: E402
    FEATURE_NAMES,
    SCHEMA,
    TYPES,
    VERSION,
    RawSFXEngine,
    decide,
    read_policy,
)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def features(root, ced, clap, cache):
    import psutil
    import soundfile as sf

    def reject(*_args, **_kwargs):
        raise RuntimeError("No network in raw SFX inference.")

    socket.create_connection = reject
    socket.socket.connect = reject
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    if file_digest(root / "corpus.json") != protocol["corpus_sha256"]:
        raise ValueError("Frozen corpus changed.")
    rows = json.loads((root / "corpus.json").read_text("utf-8"))["rows"]
    started = time.perf_counter()
    peak = 0
    engine = RawSFXEngine(ced, clap, cache, 4)
    path = root / "features.jsonl"
    previous = {}
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            row = json.loads(line)
            if row["models"] != engine.identity or row["protocol_sha256"] != file_digest(
                root / "protocol.json"
            ):
                raise ValueError("Partial features do not match the frozen experiment.")
            previous[row["id"]] = row
    for index, row in enumerate(rows, 1):
        if row["id"] in previous:
            continue
        peak = max(peak, psutil.Process().memory_info().rss)
        if time.perf_counter() - started > 2400 or peak > 5 * 1024**3:
            raise RuntimeError("Resource budget reached; progress retained for resume.")
        if file_digest(Path(row["audio_path"])) != row["audio_sha256"]:
            raise ValueError("Prepared input changed.")
        values, rate = sf.read(row["audio_path"], dtype="float32")
        if rate != 32000 or values.shape != (96000,):
            raise ValueError("Invalid prepared audio.")
        evidence = engine.analyze(values, row["peak_relative"])
        result = {
            **row,
            "evidence": evidence,
            "models": engine.identity,
            "protocol_sha256": file_digest(root / "protocol.json"),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        if index % 20 == 0:
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
            "network_blocked": True,
            "seconds": time.perf_counter() - started,
            "sampled_peak_rss": peak,
            "counters": engine.counters,
        },
    )


def fit_head(rows, label, regularization=1.0, sample_weights=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    x = np.array([row["evidence"]["features"] for row in rows])
    y = np.array([r["label"] == label for r in rows])
    if len(set(y)) != 2:
        raise ValueError("Both positive/negative sources are required.")
    weight = (
        balanced_source_weights(rows)
        if sample_weights is None
        else np.array(sample_weights, dtype=float, copy=True)
    )
    source_units = len({r["source_sha256"] for r in rows})
    for value in (False, True):
        weight[y == value] *= source_units / 2 / weight[y == value].sum()
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(x, sample_weight=weight)
        model = LogisticRegression(C=regularization, solver="lbfgs", max_iter=1000, tol=1e-4)
        model.fit(scaler.transform(x), y, sample_weight=weight)
    if model.n_iter_.max() >= 1000:
        raise ValueError("Head did not converge.")
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "threshold": -1e100,
        "enabled": True,
        "training_source_units": source_units,
        "C": regularization,
    }


def score(head, row):
    x = np.array(row["evidence"]["features"])
    return float(((x - head["mean"]) / head["scale"]) @ head["coefficients"] + head["intercept"])


def eligible(row, label):
    raw = row["evidence"]["raw_target_scores"]
    order = sorted(raw, key=lambda k: -raw[k])
    return (
        order[0] == label
        and raw[label] > 0
        and raw[label] > raw[order[1]]
        and row["evidence"]["rms"] >= 1e-5
    )


def stats(rows, predictions, label):
    yes = [r for r, p in zip(rows, predictions, strict=True) if p == label]
    true = [r for r in yes if r["label"] == label]
    false = [r for r in yes if r["label"] != label]
    primary = [r for r in rows if r["label"] == label and r.get("snr_db") == 0]
    primary_objects = {id(r) for r in primary}
    primary_true = [
        r for r, p in zip(rows, predictions, strict=True) if id(r) in primary_objects and p == label
    ]
    return {
        "accepted": len(yes),
        "true_positives": len(true),
        "false_positives": len(false),
        "precision": len(true) / len(yes) if yes else None,
        "positive_authors": sorted({r["author_group"] for r in true}),
        "positive_source_count": len({r["source_sha256"] for r in true}),
        "background_false": sum(r["label"] == "background" for r in false),
        "blade_false": sum(r["author_group"] == "starninjas" for r in false),
        "primary_true": len(primary_true),
        "primary_count": len(primary),
        "primary_coverage": len(primary_true) / len(primary) if primary else None,
        "false_ids": [r["id"] for r in false],
        "true_ids": [r["id"] for r in true],
    }


def fit(root, candidate_plan=None):
    if (root / "policy.json").exists():
        raise ValueError("Do not refit a frozen policy.")
    run = json.loads((root / "feature-run.json").read_text("utf-8"))
    if file_digest(root / "features.jsonl") != run["features_sha256"]:
        raise ValueError("Features changed.")
    rows = [json.loads(line) for line in (root / "features.jsonl").read_text("utf-8").splitlines()]
    train = [r for r in rows if r["split"] == "train"]
    if candidate_plan is not None:
        return fit_candidates(root, run, rows, train, candidate_plan)
    folds = []
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
            head = fit_head(selected, label)
            for i in indices:
                oof[label][i] = score(head, train[i])
        folds.append(
            {
                "author": author,
                "training_ids": [r["id"] for r in selected],
                "query_ids": [train[i]["id"] for i in indices],
            }
        )
    heads = {}
    calibration = {}
    for label in TYPES:
        values = oof[label]
        if not np.isfinite(values).all():
            raise ValueError("Incomplete held-author predictions.")
        options = []
        for threshold in sorted(set(values)):
            predictions = [
                label if v >= threshold and eligible(r, label) else "unknown"
                for r, v in zip(train, values, strict=True)
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
                options.append((result["true_positives"], result["precision"], threshold, result))
        best = max(options, key=lambda x: x[:3]) if options else None
        heads[label] = fit_head(train, label)
        heads[label].update(enabled=bool(best), threshold=best[2] if best else None)
        calibration[label] = {
            "passed": bool(best),
            "statistics": best[3] if best else None,
            "oof_scores": values.tolist(),
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
        "status": "research_author_split_weak_label_validation",
    }
    evaluations = {}
    for split in ("calibration", "test"):
        held = [r for r in rows if r["split"] == split]
        predictions = [decide(policy, r["evidence"])["label"] for r in held]
        split_stats = {}
        for label in TYPES:
            result = stats(held, predictions, label)
            passed = (
                result["precision"] is not None
                and result["precision"] >= 0.9
                and result["background_false"] == 0
                and (
                    result["primary_true"] >= 1
                    if split == "calibration"
                    else result["primary_coverage"] is not None
                    and result["primary_coverage"] >= 0.2
                )
            )
            split_stats[label] = {
                **result,
                "passed": passed,
                "enabled_before": heads[label]["enabled"],
            }
            if heads[label]["enabled"] and not passed:
                heads[label]["enabled"] = False
        evaluations[split] = split_stats
    policy["gates"] = {
        "training_author_oof": calibration,
        "independent_author_evaluations": evaluations,
    }
    save(root / "policy.json", policy)
    read_policy(root / "policy.json")
    save(
        root / "evaluation.json",
        {
            "folds": folds,
            "calibration": calibration,
            "heldout": evaluations,
            "enabled": [label for label in TYPES if heads[label]["enabled"]],
            "policy_sha256": file_digest(root / "policy.json"),
            "weak_author_caption_labels_not_human_listening": True,
        },
    )
    print(
        json.dumps(
            {"enabled": {label: heads[label]["enabled"] for label in TYPES}, "heldout": evaluations}
        )
    )


def fit_candidates(root, run, rows, train, plan_path):
    """Select C on calibration authors only; test authors never select model/threshold."""
    plan = json.loads(plan_path.read_text("utf-8"))
    if plan["parent_protocol_sha256"] != file_digest(root / "protocol.json"):
        raise ValueError("Candidate plan belongs to another frozen corpus protocol.")
    choices = plan["C_values"]
    if choices != [0.1, 1.0, 10.0]:
        raise ValueError("Unsupported predeclared candidate grid.")
    held = [r for r in rows if r["split"] == "calibration"]
    candidates = {label: [] for label in TYPES}
    folds = []
    for regularization in choices:
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
                head = fit_head(selected, label, regularization)
                for i in indices:
                    oof[label][i] = score(head, train[i])
            if regularization == choices[0]:
                folds.append(
                    {
                        "author": author,
                        "training_ids": [r["id"] for r in selected],
                        "query_ids": [train[i]["id"] for i in indices],
                    }
                )
        for label in TYPES:
            values = oof[label]
            if not np.isfinite(values).all():
                raise ValueError("Incomplete OOF scores.")
            options = []
            for threshold in sorted(set(values)):
                predictions = [
                    label if v >= threshold and eligible(r, label) else "unknown"
                    for r, v in zip(train, values, strict=True)
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
            best = max(options, key=lambda value: value[:3]) if options else None
            head = fit_head(train, label, regularization)
            head.update(enabled=bool(best), threshold=best[2] if best else None)
            predictions = [
                label
                if head["enabled"] and eligible(r, label) and score(head, r) >= head["threshold"]
                else "unknown"
                for r in held
            ]
            check = stats(held, predictions, label)
            passed = (
                head["enabled"]
                and check["precision"] is not None
                and check["precision"] >= 0.9
                and check["background_false"] == 0
                and check["primary_true"] >= 1
            )
            candidates[label].append(
                {
                    "head": head,
                    "C": regularization,
                    "oof_passed": bool(best),
                    "oof_statistics": best[3] if best else None,
                    "oof_scores": values.tolist(),
                    "calibration": check,
                    "passed": passed,
                }
            )
        print(
            json.dumps(
                {
                    "candidate_C": regularization,
                    "calibration": {
                        label: {
                            "passed": candidates[label][-1]["passed"],
                            "statistics": candidates[label][-1]["calibration"],
                        }
                        for label in TYPES
                    },
                }
            ),
            flush=True,
        )
    heads = {}
    selection = {}
    for label in TYPES:
        qualified = [item for item in candidates[label] if item["passed"]]
        if qualified:
            best = max(
                qualified,
                key=lambda item: (
                    item["calibration"]["primary_true"],
                    item["calibration"]["precision"],
                    item["calibration"]["true_positives"],
                    -abs(np.log10(item["C"])),
                ),
            )
        else:
            best = candidates[label][1]
        heads[label] = dict(best["head"])
        heads[label]["enabled"] = bool(qualified)
        selection[label] = {
            "C": best["C"],
            "enabled": bool(qualified),
            "reason": "passed_author_oof_and_calibration" if qualified else "no_candidate_passed",
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
        "candidate_plan_sha256": file_digest(plan_path),
        "feature_evidence_sha256": run["features_sha256"],
        "status": "research_author_split_weak_label_validation",
        "selection": selection,
    }
    save(root / "selected-before-test.json", policy)
    test = [r for r in rows if r["split"] == "test"]
    predictions = [decide(policy, r["evidence"])["label"] for r in test]
    heldout = {}
    for label in TYPES:
        result = stats(test, predictions, label)
        passed = (
            result["precision"] is not None
            and result["precision"] >= 0.9
            and result["background_false"] == 0
            and result["primary_coverage"] is not None
            and result["primary_coverage"] >= 0.2
        )
        heldout[label] = {**result, "passed": passed, "enabled_before": heads[label]["enabled"]}
        if heads[label]["enabled"] and not passed:
            heads[label]["enabled"] = False
    policy["gates"] = {"independent_author_test": heldout}
    save(root / "policy.json", policy)
    read_policy(root / "policy.json")
    save(
        root / "evaluation.json",
        {
            "folds": folds,
            "candidates": candidates,
            "selection": selection,
            "independent_test": heldout,
            "enabled": [label for label in TYPES if heads[label]["enabled"]],
            "selected_before_test_sha256": file_digest(root / "selected-before-test.json"),
            "policy_sha256": file_digest(root / "policy.json"),
            "weak_author_caption_labels_not_human_listening": True,
        },
    )
    print(
        json.dumps(
            {
                "enabled": {label: heads[label]["enabled"] for label in TYPES},
                "independent_test": heldout,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("features", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ced", type=Path, default=Path("models/ced-mini"))
    parser.add_argument("--clap", type=Path, default=Path("models/clap-htsat-unfused"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/sfx-presence-cache"))
    parser.add_argument("--candidate-plan", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "features":
        features(root, args.ced, args.clap, args.cache)
    else:
        fit(root, args.candidate_plan)


if __name__ == "__main__":
    main()
