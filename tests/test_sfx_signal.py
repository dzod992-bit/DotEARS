import numpy as np

from dotears.sfx_signal import decide, merge_families, parent_label


def test_signal_family_membership_preserves_unknown_function_and_original_types():
    assert parent_label("notification") == "ui_sound"
    assert parent_label("ui_tone") == "ui_sound"
    assert parent_label("whoosh") == "whoosh"
    policy = {
        "feature_source": "contrast",
        "head": {
            "mean": [0.0] * 1075,
            "scale": [1.0] * 1075,
            "coefficients": [0.0] * 1075,
            "intercept": 3.0,
            "threshold": 2.0,
            "enabled": True,
        },
    }
    case = {
        "evidence": {"features": np.zeros(1075)},
        "rms": 0.1,
        "raw_target_scores": {"notification": 0.4, "ui_sound": 0.1, "whoosh": -0.1},
    }
    result = decide(policy, case)
    assert result["label"] == "ui_sound"
    assert result["function_role"] == "unknown" and result["notification_subtype"] == "unknown"
    policy["head"]["enabled"] = False
    assert decide(policy, case)["label"] == "unknown"


def test_competing_families_remain_unknown_without_reinterpreting_either_as_a_role():
    whoosh = {"label": "whoosh"}
    signal = {"label": "ui_sound"}
    unknown = {"label": "unknown"}
    assert merge_families(whoosh, signal)["label"] == "unknown"
    assert merge_families(whoosh, unknown)["label"] == "whoosh"
    assert merge_families(unknown, signal)["label"] == "ui_sound"
    assert merge_families(unknown, signal)["function_role"] == "unknown"
