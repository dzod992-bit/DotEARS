"""Broad short-signal family presence, with function/notification subtype kept unknown."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dotears.errors import ConfigurationError
from dotears.sfx_presence_deep import score_head, validate_heads

SCHEMA = "dotears.sfx-signal-policy/1"
TAXONOMY = "dotears.sfx-types/3-hierarchical"


def parent_label(label):
    return "ui_sound" if label in ("notification", "ui_sound", "ui_click", "ui_tone") else label


def read_policy(path: Path):
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema_version") != SCHEMA
        or policy.get("label_taxonomy") != TAXONOMY
        or policy.get("feature_source") not in ("contrast", "components")
    ):
        raise ConfigurationError("Unsupported broad signal-family policy.")
    dimensions = 1075 if policy["feature_source"] == "contrast" else 524
    # Reuse frozen numerical validation, without changing legacy semantic vocabularies.
    validate_heads(
        {"heads": dict.fromkeys(("whoosh", "notification", "ui_click"), policy["head"])}, dimensions
    )
    return policy


def evidence_score(policy, case):
    if policy["feature_source"] == "contrast":
        values = np.asarray(case["evidence"]["features"], dtype=float)
        if values.shape != (1075,) or not np.isfinite(values).all():
            raise ConfigurationError("Invalid broad signal raw features.")
        return score_head(policy["head"], values), None
    options = []
    for i, stream in enumerate(case["streams"]):
        if (
            stream["rms"] >= 1e-5
            and stream["energy_fraction"] >= 1e-4
            and max(stream["text_scores"]["notification"], stream["text_scores"]["ui_sound"]) > 0
        ):
            options.append((score_head(policy["head"], stream["features"]), i))
    return max(options) if options else (-100.0, None)


def decide(policy, case):
    score, index = evidence_score(policy, case)
    head = policy["head"]
    raw = case["raw_target_scores"]
    reasons = []
    if not head["enabled"]:
        reasons.append("family_failed_heldout_gate")
    if head["threshold"] is None or score < head["threshold"]:
        reasons.append("below_frozen_threshold")
    if not math.isfinite(case["rms"]) or case["rms"] < 1e-5:
        reasons.append("quiet_input")
    if max(raw["notification"], raw["ui_sound"]) <= 0:
        reasons.append("nonpositive_raw_signal_similarity")
    if policy["feature_source"] == "components" and index is None:
        reasons.append("no_eligible_component")
    return {
        "label": "ui_sound" if not reasons else "unknown",
        "function_role": "unknown",
        "notification_subtype": "unknown",
        "confidence": None,
        "score": score,
        "stream_index": index,
        "rejection_reasons": reasons,
        "recognized_scope": "short_click_or_tonal_signal_family_only",
        "role_inferred_from_audio": False,
        "scores_are_probabilities": False,
    }


def merge_families(whoosh_decision, signal_decision):
    families = []
    if whoosh_decision["label"] == "whoosh":
        families.append("whoosh")
    if signal_decision["label"] == "ui_sound":
        families.append("ui_sound")
    label = families[0] if len(families) == 1 else "unknown"
    return {
        "label": label,
        "function_role": "unknown",
        "notification_subtype": "unknown",
        "confidence": None,
        "possible_families": families,
        "reason": "competing_sound_families"
        if len(families) > 1
        else "passed_family_check"
        if families
        else "family_evidence_insufficient",
        "whoosh_decision": whoosh_decision,
        "signal_decision": signal_decision,
        "human_confirmed": False,
        "scores_are_probabilities": False,
    }
