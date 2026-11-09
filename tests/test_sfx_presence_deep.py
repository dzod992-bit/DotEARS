import numpy as np
import pytest

from dotears.errors import ConfigurationError
from dotears.sfx_presence import TYPES
from dotears.sfx_presence_deep import FEATURE_NAMES, decide, score_head


def policy(scores=(5.0, 3.0, 1.0)):
    return {
        "heads": {
            label: {
                "mean": [0.0] * len(FEATURE_NAMES),
                "scale": [1.0] * len(FEATURE_NAMES),
                "coefficients": [0.0] * len(FEATURE_NAMES),
                "intercept": score,
                "competition_center": 0.0,
                "competition_scale": 1.0,
                "enabled": True,
                "threshold": 1.0,
            }
            for label, score in zip(TYPES, scores, strict=True)
        }
    }


def evidence(raw=None):
    return {
        "features": [0.0] * len(FEATURE_NAMES),
        "rms": 0.1,
        "raw_target_scores": raw or {"whoosh": 0.1, "notification": 0.2, "ui_click": 0.15},
    }


def test_learned_raw_head_can_resolve_a_text_prompt_ranking_error():
    frozen = policy((1.0, 2.0, 5.0))
    result = decide(frozen, evidence({"whoosh": 0.3, "notification": 0.2, "ui_click": 0.28}))
    assert result["label"] == "ui_click"
    assert result["confidence"] is None and not result["human_confirmed"]


def test_disabled_winning_head_never_unmasks_another_named_type():
    frozen = policy()
    frozen["heads"]["whoosh"]["enabled"] = False
    assert decide(frozen, evidence())["label"] == "unknown"
    frozen = policy((1.0, 5.0, 4.0))
    frozen["heads"]["notification"]["enabled"] = False
    assert decide(frozen, evidence())["label"] == "unknown"


def test_negative_original_similarity_vetoes_whoosh_and_does_not_fall_back():
    assert (
        decide(policy(), evidence({"whoosh": -0.055, "notification": 0.215, "ui_click": -0.008}))[
            "label"
        ]
        == "unknown"
    )


def test_close_raw_heads_and_invalid_features_abstain_or_fail():
    assert decide(policy((5.0, 4.9, 1.0)), evidence())["label"] == "unknown"
    invalid = evidence()
    invalid["features"][0] = np.nan
    with pytest.raises(ConfigurationError):
        decide(policy(), invalid)


def test_exported_rbf_matches_sklearn_and_ignores_unused_latent_dimensions():
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    rng = np.random.default_rng(9531)
    x = rng.normal(size=(40, 8))
    y = x[:, 0] * x[:, 1] > 0
    scaler = StandardScaler().fit(x)
    model = SVC(C=3.0, kernel="rbf", gamma="scale").fit(scaler.transform(x), y)
    head = {
        "kind": "rbf",
        "feature_count": 8,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "support_vectors": model.support_vectors_.tolist(),
        "dual_coefficients": model.dual_coef_[0].tolist(),
        "gamma": float(model._gamma),
        "intercept": float(model.intercept_[0]),
    }
    query = rng.normal(size=(5, 8))
    expected = model.decision_function(scaler.transform(query))
    actual = [score_head(head, np.r_[item, np.full(30, 999.0)]) for item in query]
    assert actual == pytest.approx(expected, abs=1e-11)
