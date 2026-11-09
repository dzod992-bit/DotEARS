"""Experimental recognition from all query-free component views; never oracle-selected."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dotears.errors import ConfigurationError
from dotears.sfx import PROMPTS
from dotears.sfx_presence_deep import score_head, validate_heads
from dotears.sfx_taxonomy import UI_TAXONOMY, UI_TYPES

SCHEMA = "dotears.sfx-component-policy/1"
VERSION = "fuss-clap-clean-heads-v1"
FEATURE_NAMES = [f"clap_audio:{i}" for i in range(512)] + [f"clap_cosine:{k}" for k in PROMPTS]


def read_policy(path: Path):
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema_version") != SCHEMA
        or policy.get("version") != VERSION
        or policy.get("label_taxonomy") != UI_TAXONOMY
        or policy.get("feature_names") != FEATURE_NAMES
        or set(policy.get("heads", {})) != set(UI_TYPES)
    ):
        raise ConfigurationError("Unsupported component-feature policy.")
    validate_heads(
        {
            "heads": {
                old: policy["heads"][new]
                for old, new in zip(("whoosh", "notification", "ui_click"), UI_TYPES, strict=True)
            }
        },
        len(FEATURE_NAMES),
    )
    return policy


def window_scores(policy, case):
    if len(case["streams"]) != 4:
        raise ConfigurationError("Four query-free component views are required.")
    for stream in case["streams"]:
        values = np.asarray(stream["features"], dtype=float)
        if (
            values.shape != (len(FEATURE_NAMES),)
            or not np.isfinite(values).all()
            or not math.isfinite(stream["rms"])
            or stream["rms"] < 0
            or not math.isfinite(stream["energy_fraction"])
            or stream["energy_fraction"] < 0
            or set(stream["text_scores"]) != set(UI_TYPES)
            or not all(math.isfinite(v) for v in stream["text_scores"].values())
        ):
            raise ConfigurationError("Invalid component feature evidence.")
    result = {}
    for label in UI_TYPES:
        eligible = []
        for index, stream in enumerate(case["streams"]):
            if (
                stream["rms"] >= 1e-5
                and stream["energy_fraction"] >= 1e-4
                and stream["text_scores"][label] > 0
            ):
                eligible.append((score_head(policy["heads"][label], stream["features"]), index))
        best = max(eligible) if eligible else (-100.0, None)
        result[label] = {"raw_score": best[0], "stream_index": best[1]}
    return result


def decide(policy, case):
    if set(case["raw_target_scores"]) != set(UI_TYPES) or not all(
        math.isfinite(v) for v in case["raw_target_scores"].values()
    ):
        raise ConfigurationError("All finite raw-audio type scores are required.")
    candidates = window_scores(policy, case)
    for label, candidate in candidates.items():
        head = policy["heads"][label]
        candidate["support"] = (candidate["raw_score"] - head["competition_center"]) / head[
            "competition_scale"
        ]
        candidate["threshold"] = head["threshold"]
    ranked = sorted(UI_TYPES, key=lambda k: -candidates[k]["support"])
    best = ranked[0]
    for label, candidate in candidates.items():
        head = policy["heads"][label]
        reasons = []
        if label != best or candidates[best]["support"] - candidates[ranked[1]]["support"] < 0.15:
            reasons.append("ambiguous_or_competing_component_types")
        if not head["enabled"]:
            reasons.append("type_failed_independent_gate")
        if candidate["stream_index"] is None:
            reasons.append("no_eligible_component_evidence")
        if not math.isfinite(case["rms"]) or case["rms"] < 1e-5:
            reasons.append("quiet_input")
        if case["raw_target_scores"][label] <= 0:
            reasons.append("nonpositive_raw_audio_similarity")
        if head["threshold"] is None or candidate["raw_score"] < head["threshold"]:
            reasons.append("below_frozen_threshold")
        candidate["rejection_reasons"] = reasons
    label = best if not candidates[best]["rejection_reasons"] else "unknown"
    return {
        "label": label,
        "confidence": None,
        "candidates": candidates,
        "status": "provisional_component_feature_type" if label != "unknown" else "abstained",
        "component_views_are_not_isolated_effects": True,
        "scores_are_probabilities": False,
    }
