"""Frozen learned sound-family classifiers over the local audio features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dotears.errors import ConfigurationError
from dotears.sfx_contrast import FEATURE_NAMES
from dotears.sfx_hierarchy import numeric_head
from dotears.sfx_presence_deep import score_head, validate_heads
from dotears.sound_expansion import FAMILIES

SCHEMA = "dotears.sound-expansion/3"
TARGETS = ("keyboard", "glass_break", "zipper", "dog_bark")
DIMENSIONS = len(FEATURE_NAMES) + 2 * len(FAMILIES)


def features(evidence: dict, vectors: np.ndarray) -> np.ndarray:
    raw = np.asarray(evidence["features"], dtype=float)
    if raw.shape != (len(FEATURE_NAMES),) or not np.isfinite(raw).all():
        raise ConfigurationError("Invalid input to the sound-family classifier.")
    similarities = raw[:1024].reshape(2, 512) @ vectors.T
    return np.concatenate((raw, similarities.ravel()))


def read_policy(path: Path, models: dict) -> dict:
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema") != SCHEMA
        or policy.get("models") != models
        or policy.get("families") != list(FAMILIES)
        or not policy.get("heads")
        or not set(policy["heads"]) <= set(TARGETS)
    ):
        raise ConfigurationError("Incompatible learned sound-family policy.")
    vectors = np.asarray(policy["text_vectors"], dtype=float)
    if vectors.shape != (len(FAMILIES), 512) or not np.isfinite(vectors).all():
        raise ConfigurationError("Invalid sound-family text vectors.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ConfigurationError("Sound-family text vectors must have unit length.")
    heads = {}
    for name, head in policy["heads"].items():
        validate_heads(
            {"heads": dict.fromkeys(("whoosh", "notification", "ui_click"), head)}, DIMENSIONS
        )
        if not 0 <= head["semantic_floor"] <= 1 or type(head["proposal_enabled"]) is not bool:
            raise ConfigurationError("Invalid sound-family semantic gate.")
        if head["enabled"] and not head["proposal_enabled"]:
            raise ConfigurationError("A family was enabled without its calibrated proposal.")
        heads[name] = numeric_head(head)
    return {**policy, "text_vectors": vectors, "heads": heads}


def semantic_support(vector: np.ndarray, family: str) -> float:
    index = list(FAMILIES).index(family)
    return float(
        max(vector[len(FEATURE_NAMES) + index], vector[len(FEATURE_NAMES) + len(FAMILIES) + index])
    )


def decide(policy: dict, evidence: dict, legacy_label: str) -> dict:
    if legacy_label != "unknown" or evidence["rms"] < 1e-5:
        return {}
    vector = features(evidence, policy["text_vectors"])
    proposals = []
    for name, head in policy["heads"].items():
        if not head["proposal_enabled"] or semantic_support(vector, name) < head["semantic_floor"]:
            continue
        score = score_head(head, vector)
        if score >= head["threshold"]:
            proposals.append((name, score))
    if len(proposals) != 1:
        return {}
    name, score = proposals[0]
    return (
        {"expanded_label": name, "sound_family_score": score}
        if policy["heads"][name]["enabled"]
        else {}
    )
