import numpy as np
import pytest

from dotears.errors import ConfigurationError
from dotears.sfx_contrast import FEATURE_NAMES, decide, spectral_contrast
from dotears.sfx_taxonomy import UI_TYPES


def test_query_free_contrast_suppresses_stable_background_and_preserves_short_attack():
    times = np.arange(96000) / 32000
    music = 0.1 * np.sin(2 * np.pi * 400 * times)
    effect = np.zeros(len(times))
    effect[32000:33600] = np.random.default_rng(438).normal(0, 0.12, 1600)
    values = music + effect
    result = spectral_contrast(values, 1.0)
    assert result.shape == values.shape and np.isfinite(result).all()
    assert np.sqrt(np.mean(result[:16000] ** 2)) < 0.4 * np.sqrt(np.mean(values[:16000] ** 2))
    assert np.sqrt(np.mean(result[32000:33600] ** 2)) > 0.5 * np.sqrt(
        np.mean(effect[32000:33600] ** 2)
    )
    assert np.max(np.abs(spectral_contrast(values * 0.3, 1.0) - result * 0.3)) < 2e-8
    assert np.array_equal(spectral_contrast(np.zeros(96000), 1.0), np.zeros(96000))


def test_contrast_handles_audio_boundaries_and_rejects_invalid_context():
    values = np.random.default_rng(73).normal(size=16000)
    for peak in (0.0, 0.5):
        assert spectral_contrast(values, peak).shape == values.shape
    for peak in (-1.0, float("nan"), 10.0):
        with pytest.raises(ConfigurationError):
            spectral_contrast(values, peak)


def policy(scores):
    return {
        "heads": {
            label: {
                "mean": [0.0] * len(FEATURE_NAMES),
                "scale": [1.0] * len(FEATURE_NAMES),
                "coefficients": [0.0] * len(FEATURE_NAMES),
                "intercept": score,
                "threshold": 1.0,
                "competition_center": 0.0,
                "competition_scale": 1.0,
                "enabled": True,
            }
            for label, score in zip(UI_TYPES, scores, strict=True)
        }
    }


def evidence():
    return {
        "features": [0.0] * len(FEATURE_NAMES),
        "rms": 0.1,
        "raw_target_scores": {"whoosh": 0.1, "notification": 0.2, "ui_sound": 0.3},
    }


def test_broad_ui_competes_with_notifications_and_disabled_winner_does_not_fall_back():
    frozen = policy((1.0, 3.0, 5.0))
    assert decide(frozen, evidence())["label"] == "ui_sound"
    frozen["heads"]["ui_sound"]["enabled"] = False
    assert decide(frozen, evidence())["label"] == "unknown"
    assert decide(policy((1.0, 5.0, 4.9)), evidence())["label"] == "unknown"
    quiet = evidence()
    quiet["rms"] = 0
    assert decide(policy((5.0, 1.0, 2.0)), quiet)["label"] == "unknown"
    negative = evidence()
    negative["raw_target_scores"]["ui_sound"] = -0.1
    assert decide(policy((1.0, 2.0, 5.0)), negative)["label"] == "unknown"
