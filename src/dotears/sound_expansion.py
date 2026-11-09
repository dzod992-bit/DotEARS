"""Calibrated additional sound families on the existing raw/contrast audio embeddings."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dotears.errors import ConfigurationError

SCHEMA = "dotears.sound-expansion/1"
CONFIRMED_SCHEMA = "dotears.sound-expansion/2"
CONFIRMATION_TAGS = {
    "keyboard": ("Computer keyboard", "Typing"),
    "glass_break": ("Glass", "Shatter", "Breaking"),
    "zipper": ("Zipper (clothing)",),
}
FAMILIES = {
    "footstep": (
        "Шаги",
        "footstep sounds",
        (
            "Footsteps from a person walking",
            "Footfalls while walking",
            "A shoe hitting the ground while walking",
        ),
    ),
    "knock": (
        "Стук",
        "a knock on a surface",
        (
            "Knocking on a door or solid surface",
            "A short dry knock",
            "A hand knocking on a hard surface",
        ),
    ),
    "clap": (
        "Хлопок ладонями",
        "a hand clap",
        ("Hands clapping together", "A single sharp hand clap", "The sound of clapping hands"),
    ),
    "keyboard": (
        "Клавиатура",
        "keyboard keystrokes",
        (
            "Computer keyboard typing",
            "Keys being pressed on a computer keyboard",
            "A short sequence of keyboard keystrokes",
        ),
    ),
    "glass_break": (
        "Разбитое стекло",
        "a glass shatter",
        ("Glass breaking", "A sharp glass shatter", "Glass breaking into small pieces"),
    ),
    "zipper": (
        "Застёжка-молния",
        "the sound of a zipper moving",
        ("A clothing zipper being pulled", "A short zipper sound", "A zipper opening or closing"),
    ),
    "thud": (
        "Глухой удар",
        "a low thudding impact",
        ("A dull thudding impact", "A low thump from a collision", "A short heavy muffled thud"),
    ),
    "dog_bark": (
        "Лай",
        "a dog bark",
        ("A dog barking", "A short dog bark", "The sound of a barking dog"),
    ),
}
VIEWS = ("raw", "balanced", "agreement", "raw_family", "balanced_family")
DISTRACTORS = ("speech", "music", "music_and_speech", "noise", "whoosh", "ui_click", "ui_tone")


def read_policy(path: Path, models: dict) -> dict:
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema") not in {SCHEMA, CONFIRMED_SCHEMA}
        or policy.get("models") != models
        or policy.get("families") != list(FAMILIES)
        or set(policy.get("heads", {})) != set(FAMILIES)
    ):
        raise ConfigurationError("Sound-family policy is incompatible with the installed features.")
    vectors = np.asarray(policy["text_vectors"], dtype=float)
    if vectors.shape != (len(FAMILIES), 512) or not np.isfinite(vectors).all():
        raise ConfigurationError("Invalid sound-family text vectors.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise ConfigurationError("Sound-family text vectors must have unit length.")
    for head in policy["heads"].values():
        if (
            head["view"] not in VIEWS
            or type(head["enabled"]) is not bool
            or not math.isfinite(head["threshold"])
            or not 0 <= head["threshold"] <= 1
            or not math.isfinite(head["margin"])
            or not 0 <= head["margin"] <= 1
        ):
            raise ConfigurationError("Invalid sound-family decision thresholds.")
    if policy["schema"] == CONFIRMED_SCHEMA:
        gates = policy["confirmation_thresholds"]
        if (
            set(gates) != set(CONFIRMATION_TAGS)
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in gates.values())
            or any(
                head["enabled"] and label not in gates for label, head in policy["heads"].items()
            )
        ):
            raise ConfigurationError("Invalid independent sound-family confirmation thresholds.")
    return {**policy, "text_vectors": vectors}


def scores(vectors: np.ndarray, evidence: dict) -> dict[str, dict[str, tuple[float, float]]]:
    """Return cosine and lead over rival sounds for each family and view."""
    embedding = np.asarray(evidence["features"][:1024], dtype=float).reshape(2, 512)
    target = embedding @ vectors.T
    background = np.array(
        [
            [evidence[f"{name}_cosines"][label] for label in DISTRACTORS]
            for name in ("raw", "contrast")
        ]
    )
    joint = np.column_stack((target, background))
    balanced = joint.mean(axis=0)
    views = (
        joint[0],
        balanced,
        joint.min(axis=0),
        joint[0, : len(FAMILIES)],
        balanced[: len(FAMILIES)],
    )
    return {
        name: {
            family: (float(values[i]), float(values[i] - np.max(np.delete(values, i))))
            for i, family in enumerate(FAMILIES)
        }
        for name, values in zip(VIEWS, views, strict=True)
    }


def decide(policy: dict, evidence: dict, legacy_label: str) -> str:
    if legacy_label != "unknown" or evidence["rms"] < 1e-5:
        return legacy_label
    measurements = scores(policy["text_vectors"], evidence)
    accepted = []
    for family, head in policy["heads"].items():
        cosine, margin = measurements[head["view"]][family]
        proposal_enabled = (
            family in CONFIRMATION_TAGS
            if policy.get("schema") == CONFIRMED_SCHEMA
            else head["enabled"]
        )
        if proposal_enabled and cosine >= head["threshold"] and margin >= head["margin"]:
            accepted.append(family)
    # A disabled family must not expose a different candidate from an ambiguous window.
    return (
        accepted[0] if len(accepted) == 1 and policy["heads"][accepted[0]]["enabled"] else "unknown"
    )


def confirmation_score(ced, audio: np.ndarray, family: str) -> float:
    """Run CED on the same raw focus. Its broad tags must corroborate CLAP."""
    import librosa

    values = librosa.resample(audio, orig_sr=32000, target_sr=16000)
    if len(values) < 2048:
        values = np.pad(values, (0, 2048 - len(values)))
    predicted = ced.predict(values)
    tags = [float(predicted[ced.labels.index(label)]) for label in CONFIRMATION_TAGS[family]]
    if family == "glass_break":
        glass, shatter, breaking = tags
        return min(glass, max(shatter, breaking))
    return max(tags)
