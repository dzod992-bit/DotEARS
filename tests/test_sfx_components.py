import copy

import pytest

from dotears.errors import ConfigurationError
from dotears.sfx_components import FEATURE_NAMES, decide
from dotears.sfx_taxonomy import UI_TYPES


def fixture():
    heads = {
        k: {
            "mean": [0.0] * len(FEATURE_NAMES),
            "scale": [1.0] * len(FEATURE_NAMES),
            "coefficients": [float(j == i) for j in range(len(FEATURE_NAMES))],
            "intercept": 0.0,
            "competition_center": 0.0,
            "competition_scale": 1.0,
            "enabled": True,
            "threshold": 1.0,
        }
        for i, k in enumerate(UI_TYPES)
    }
    features = [0.0] * len(FEATURE_NAMES)
    features[1] = 4.0
    stream = {
        "features": features,
        "rms": 0.004,
        "energy_fraction": 0.001,
        "text_scores": dict.fromkeys(UI_TYPES, 0.3),
    }
    case = {
        "streams": [copy.deepcopy(stream) for _ in range(4)],
        "rms": 0.1,
        "raw_target_scores": dict.fromkeys(UI_TYPES, 0.2),
    }
    return {"heads": heads}, case


def test_quiet_component_features_require_raw_evidence_and_never_normalize_audio():
    policy, case = fixture()
    before = copy.deepcopy(case)
    assert decide(policy, case)["label"] == "notification"
    assert case == before
    case["raw_target_scores"]["notification"] = -0.1
    assert decide(policy, case)["label"] == "unknown"
    case = before
    for stream in case["streams"]:
        stream["energy_fraction"] = 1e-8
    assert decide(policy, case)["label"] == "unknown"


def test_disabled_component_winner_cannot_fall_back_and_nan_is_rejected():
    policy, case = fixture()
    policy["heads"]["notification"]["enabled"] = False
    assert decide(policy, case)["label"] == "unknown"
    case["streams"][0]["features"][0] = float("nan")
    with pytest.raises(ConfigurationError):
        decide(policy, case)
