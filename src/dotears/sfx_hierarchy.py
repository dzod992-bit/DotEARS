"""Compose validated acoustic families while keeping unresolved function roles unknown."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dotears.audiosep import file_digest
from dotears.errors import ConfigurationError
from dotears.sfx import local_artifact
from dotears.sfx_contrast import FEATURE_NAMES
from dotears.sfx_contrast import decide as whoosh_decide
from dotears.sfx_contrast import read_policy as read_whoosh_policy
from dotears.sfx_signal import TAXONOMY, merge_families
from dotears.sfx_signal import decide as signal_decide
from dotears.sfx_signal import read_policy as read_signal_policy

SCHEMA = "dotears.sfx-presence-policy/5"
VERSION = "whoosh-and-broad-signal-families-v1"


def numeric_head(head):
    result = dict(head)
    for key in ("mean", "scale", "coefficients", "support_vectors", "dual_coefficients"):
        if key in result:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def read_policy(path: Path):
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema_version") != SCHEMA
        or policy.get("version") != VERSION
        or policy.get("label_taxonomy") != TAXONOMY
        or policy.get("feature_names") != FEATURE_NAMES
    ):
        raise ConfigurationError("Unsupported hierarchical SFX profile.")
    children = {}
    for key, loader in (("whoosh", read_whoosh_policy), ("signal", read_signal_policy)):
        reference = policy["detectors"][key]
        child_path = local_artifact(path.resolve().parent, reference["path"])
        if file_digest(child_path) != reference["sha256"]:
            raise ConfigurationError("A hierarchical child policy changed.")
        children[key] = loader(child_path)
    if (
        children["signal"]["feature_source"] != "contrast"
        or children["whoosh"]["models"] != children["signal"]["models"]
        or policy["models"] != children["whoosh"]["models"]
    ):
        raise ConfigurationError("Hierarchical policies need identical raw feature backbones.")
    if any(h["enabled"] for k, h in children["whoosh"]["heads"].items() if k != "whoosh"):
        raise ConfigurationError("Use the dedicated whoosh profile for the whoosh branch.")
    children["whoosh"]["heads"] = {
        k: numeric_head(h) for k, h in children["whoosh"]["heads"].items()
    }
    children["signal"]["head"] = numeric_head(children["signal"]["head"])
    policy["_runtime"] = children
    return policy


def decide(policy, evidence):
    whoosh = whoosh_decide(policy["_runtime"]["whoosh"], evidence)
    signal = signal_decide(
        policy["_runtime"]["signal"],
        {
            "evidence": evidence,
            "rms": evidence["rms"],
            "raw_target_scores": evidence["raw_target_scores"],
        },
    )
    merged = merge_families(whoosh, signal)
    return {
        **merged,
        "status": "provisional_sound_family" if merged["label"] != "unknown" else "abstained",
        "role_classification_enabled": False,
        "label_taxonomy": TAXONOMY,
        "temporal_scope": "presence_in_mixture_focus_window",
        "effect_onset_seconds": None,
    }


def build(whoosh_path: Path, signal_path: Path, output: Path):
    if output.exists():
        raise ConfigurationError("Use a new folder for the hierarchical SFX profile.")
    whoosh, signal = read_whoosh_policy(whoosh_path), read_signal_policy(signal_path)
    if signal["feature_source"] != "contrast" or whoosh["models"] != signal["models"]:
        raise ConfigurationError("The two family profiles used different features.")
    data = {"whoosh": whoosh_path.read_bytes(), "signal": signal_path.read_bytes()}
    hashes = {"whoosh": file_digest(whoosh_path), "signal": file_digest(signal_path)}
    output.mkdir(parents=True)
    for name, value in data.items():
        (output / (name + ".json")).write_bytes(value)
        if file_digest(output / (name + ".json")) != hashes[name]:
            raise ConfigurationError("Source policy changed during profile composition.")
    policy = {
        "schema_version": SCHEMA,
        "version": VERSION,
        "label_taxonomy": TAXONOMY,
        "created_at": datetime.now(UTC).isoformat(),
        "models": whoosh["models"],
        "feature_names": FEATURE_NAMES,
        "detectors": {
            name: {"path": name + ".json", "sha256": value} for name, value in hashes.items()
        },
        "output_definitions": {
            "whoosh": "Sweeping/rushing acoustic family.",
            "ui_sound": "Short click/beep/chime family, including notification-like signals.",
            "function_role": (
                "Unknown: notification, confirmation or interface intent is not established."
            ),
        },
        "notification_subtype_enabled": False,
        "human_annotations_used_in_inference": False,
        "merge_rule": "One accepted family returns its name; competing families remain unknown.",
    }
    path = output / "policy.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(policy, handle, ensure_ascii=False, indent=2, allow_nan=False)
    read_policy(path)
    return path
