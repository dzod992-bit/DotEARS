import json

import numpy as np
import pytest

from dotears.errors import ConfigurationError
from dotears.sfx_contrast import FEATURE_NAMES
from dotears.sfx_contrast import SCHEMA as WHOOSH_SCHEMA
from dotears.sfx_contrast import VERSION as WHOOSH_VERSION
from dotears.sfx_hierarchy import build, decide, numeric_head, read_policy
from dotears.sfx_presence_deep import score_head
from dotears.sfx_signal import SCHEMA as SIGNAL_SCHEMA
from dotears.sfx_signal import TAXONOMY
from dotears.sfx_taxonomy import UI_TAXONOMY, UI_TYPES


def child_policies(folder):
    def head(intercept, enabled):
        return {
            "mean": [0.0] * 1075,
            "scale": [1.0] * 1075,
            "coefficients": [0.0] * 1075,
            "intercept": intercept,
            "threshold": 1.0,
            "enabled": enabled,
            "competition_center": 0.0,
            "competition_scale": 1.0,
        }

    whoosh = {
        "schema_version": WHOOSH_SCHEMA,
        "version": WHOOSH_VERSION,
        "label_taxonomy": UI_TAXONOMY,
        "feature_names": FEATURE_NAMES,
        "models": {"identity": "same-frozen-features"},
        "heads": {
            label: head(4 if label == "whoosh" else 0, label == "whoosh") for label in UI_TYPES
        },
    }
    signal = {
        "schema_version": SIGNAL_SCHEMA,
        "label_taxonomy": TAXONOMY,
        "feature_source": "contrast",
        "models": whoosh["models"],
        "head": head(3, True),
    }
    paths = (folder / "whoosh.json", folder / "signal.json")
    for path, value in zip(paths, (whoosh, signal), strict=True):
        path.write_text(json.dumps(value), "utf-8")
    return paths


def test_composed_profile_binds_children_and_keeps_competing_families_unknown(tmp_path):
    paths = child_policies(tmp_path)
    profile = build(*paths, tmp_path / "composed")
    policy = read_policy(profile)
    evidence = {
        "features": [0.0] * 1075,
        "rms": 0.1,
        "raw_target_scores": dict.fromkeys(UI_TYPES, 0.3),
    }
    result = decide(policy, evidence)
    assert result["label"] == "unknown"
    assert result["reason"] == "competing_sound_families"
    assert result["function_role"] == result["notification_subtype"] == "unknown"
    assert not result["role_classification_enabled"]
    assert result["temporal_scope"] == "presence_in_mixture_focus_window"
    assert result["effect_onset_seconds"] is None
    assert (profile.parent / "whoosh.json").read_bytes() == paths[0].read_bytes()
    child = profile.parent / "signal.json"
    child.write_text(child.read_text("utf-8") + " ", "utf-8")
    with pytest.raises(ConfigurationError, match="child policy changed"):
        read_policy(profile)


def test_composition_refuses_different_feature_backbones(tmp_path):
    whoosh, signal = child_policies(tmp_path)
    data = json.loads(signal.read_text("utf-8"))
    data["models"] = {"identity": "different"}
    signal.write_text(json.dumps(data), "utf-8")
    with pytest.raises(ConfigurationError, match="different features"):
        build(whoosh, signal, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_precompiling_rbf_parameters_preserves_scores_and_serialized_head():
    head = {
        "kind": "rbf",
        "feature_count": 2,
        "feature_indices": [2, 0],
        "mean": [0.5, -0.1],
        "scale": [0.8, 2.0],
        "gamma": 0.7,
        "support_vectors": [[1.0, 0.2], [-0.3, 0.8]],
        "dual_coefficients": [0.6, -0.4],
        "intercept": -0.2,
    }
    frozen = json.dumps(head)
    features = np.array([0.2, 0.4, -0.8])
    assert score_head(numeric_head(head), features) == score_head(head, features)
    assert json.dumps(head) == frozen
