"""Calibrate frozen heads on calibration authors; keep the final test in a separate phase."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from evaluate_sfx_contrast import gate, numeric_head, numeric_policy, source_weights
from evaluate_sfx_presence import stats
from evaluate_sfx_presence_deep import weighted_median

from dotears.audiosep import file_digest
from dotears.sfx_contrast import decide, read_policy
from dotears.sfx_presence_deep import score_head
from dotears.sfx_taxonomy import UI_TYPES


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def select(root, output):
    if output.exists():
        raise ValueError("Use a new calibration output folder.")
    plan = json.loads((root / "posthoc-plan-before-test.json").read_text("utf-8"))
    if plan["normalizations"] != ["original_oof", "calibration_medians"]:
        raise ValueError("Unsupported frozen calibration plan.")
    output.mkdir(parents=True)
    run = json.loads((root / "feature-run.json").read_text("utf-8"))
    if file_digest(root / "features.jsonl") != run["features_sha256"]:
        raise ValueError("Calibration features changed after extraction.")
    # Access only candidate weights and calibration data, never the previous test checks.
    candidates = json.loads((root / "evaluation.json").read_text("utf-8"))["candidates"]
    rows = [json.loads(line) for line in (root / "features.jsonl").read_text("utf-8").splitlines()]
    calibration = [r for r in rows if r["split"] == "calibration"]
    weights = source_weights(calibration)
    best, summaries = None, []
    for index, candidate in enumerate(candidates):
        if candidate["policy"]["models"] != run["models"]:
            raise ValueError("Candidate and calibration backbones differ.")
        numeric_heads = {k: numeric_head(h) for k, h in candidate["policy"]["heads"].items()}
        raw = {
            label: np.array(
                [score_head(numeric_heads[label], r["evidence"]["features"]) for r in calibration]
            )
            for label in UI_TYPES
        }
        for normalization in plan["normalizations"]:
            policy = copy.deepcopy(candidate["policy"])
            if normalization == "calibration_medians":
                for label in UI_TYPES:
                    positive = np.array([r["label"] == label for r in calibration])
                    center = weighted_median(raw[label][~positive], weights[~positive])
                    pos = weighted_median(raw[label][positive], weights[positive])
                    policy["heads"][label].update(
                        competition_center=center, competition_scale=max(pos - center, 0.1)
                    )
            supports = np.column_stack(
                [
                    (raw[label] - policy["heads"][label]["competition_center"])
                    / policy["heads"][label]["competition_scale"]
                    for label in UI_TYPES
                ]
            )
            ranks = np.argsort(-supports, axis=1, kind="stable")
            winners = [
                UI_TYPES[order[0]]
                if supports[i, order[0]] - supports[i, order[1]] >= 0.15
                and calibration[i]["evidence"]["rms"] >= 1e-5
                and calibration[i]["evidence"]["raw_target_scores"][UI_TYPES[order[0]]] > 0
                else "unknown"
                for i, order in enumerate(ranks)
            ]
            checks = {}
            for label in UI_TYPES:
                options = []
                for threshold in sorted(set(raw[label])):
                    predictions = [
                        label if w == label and value >= threshold else "unknown"
                        for w, value in zip(winners, raw[label], strict=True)
                    ]
                    check = stats(calibration, predictions, label)
                    if (
                        gate(check)
                        and check["true_positives"] >= 3
                        and len(check["positive_authors"]) >= 2
                    ):
                        options.append(
                            (check["true_positives"], check["precision"], threshold, check)
                        )
                chosen = max(options, key=lambda v: v[:3]) if options else None
                policy["heads"][label].update(
                    enabled=bool(chosen), threshold=chosen[2] if chosen else None
                )
                checks[label] = chosen[3] if chosen else None
            enabled = [label for label in UI_TYPES if policy["heads"][label]["enabled"]]
            quality = (
                len(enabled),
                sum(checks[k]["primary_coverage"] for k in enabled),
                sum(checks[k]["true_positives"] for k in enabled),
                candidate["C"] == 1.0,
                candidate["learner"] == "linear",
                normalization == "original_oof",
                candidate["variant"] == "semantic",
            )
            summary = {
                "candidate_index": index,
                "variant": candidate["variant"],
                "C": candidate["C"],
                "learner": candidate["learner"],
                "normalization": normalization,
                "quality": quality,
                "enabled": enabled,
                "calibration_fit": checks,
            }
            summaries.append(summary)
            if best is None or quality > best[0]:
                best = (quality, policy, summary)
            print(
                json.dumps(
                    {k: v for k, v in summary.items() if k not in ("quality", "calibration_fit")}
                ),
                flush=True,
            )
    policy = best[1]
    policy.update(
        created_at=datetime.now(UTC).isoformat(),
        posthoc_selection=best[2],
        posthoc_plan_sha256=file_digest(root / "posthoc-plan-before-test.json"),
        calibration_data_used_for_parameter_fitting=True,
        status="posthoc_calibration_not_yet_tested",
        previous_test_results_consulted=False,
        posthoc_normalization_definition={
            "negative_center": "source-balanced negative median on calibration data",
            "scale": "max(positive median - negative median, 0.1)",
            "original_oof_variant": "Preserve its original normalizers.",
        },
        calibration_code_sha256=file_digest(Path(__file__)),
    )
    save(output / "selected-before-test.json", policy)
    read_policy(output / "selected-before-test.json")
    save(
        output / "calibration-selection.json",
        {
            "selected": best[2],
            "candidates": summaries,
            "parent_candidate_file_sha256": file_digest(root / "evaluation.json"),
            "fresh_test_performance_consulted": False,
            "video_features_consulted": False,
        },
    )
    print(json.dumps({"selected": {k: v for k, v in best[2].items() if k != "calibration_fit"}}))


def test(root, output):
    policy_path = output / "selected-before-test.json"
    policy = json.loads(policy_path.read_text("utf-8"))
    rows = [json.loads(line) for line in (root / "features.jsonl").read_text("utf-8").splitlines()]
    held = [r for r in rows if r["split"] == "test"]
    scoring_policy = numeric_policy(policy)
    predictions = [decide(scoring_policy, r["evidence"])["label"] for r in held]
    checks = {label: stats(held, predictions, label) for label in UI_TYPES}
    for label, check in checks.items():
        check["enabled_before"] = policy["heads"][label]["enabled"]
        check["passed"] = check["enabled_before"] and gate(check)
        if not check["passed"]:
            policy["heads"][label]["enabled"] = False
    policy.update(
        test_gate=checks,
        status="test_gated_posthoc_calibrated_model",
        selected_before_test_sha256=file_digest(policy_path),
    )
    save(output / "policy.json", policy)
    read_policy(output / "policy.json")
    result = {
        "checks": checks,
        "enabled": [k for k, v in policy["heads"].items() if v["enabled"]],
        "test_decisions": [
            {"id": r["id"], "label": r["label"], "decision": p}
            for r, p in zip(held, predictions, strict=True)
        ],
        "policy_sha256": file_digest(output / "policy.json"),
        "thresholds_or_weights_changed_after_test": False,
    }
    save(output / "evaluation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "test_decisions"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "test"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (select if args.phase == "select" else test)(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
