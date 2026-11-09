import numpy as np
import pytest

from dotears.errors import ConfigurationError
from dotears.separation import output_energy_db, reference_metrics


def test_perfect_separation_improves_error_and_retains_target_amplitude():
    rng = np.random.default_rng(321)
    target, background = rng.normal(0, 0.1, (2, 32000))
    result = reference_metrics(target, target + background, target)
    assert result["sdr_improvement_db"] > 100
    assert result["target_projection_scale"] == pytest.approx(1)
    assert result["target_projection_gain_db"] == pytest.approx(0)


def test_muting_everything_cannot_masquerade_as_successful_separation():
    t = np.arange(32000) / 32000
    target = 0.2 * np.sin(2 * np.pi * 300 * t)
    mixture = target + 0.2 * np.sin(2 * np.pi * 1500 * t)
    muted = reference_metrics(target, mixture, np.zeros_like(target))
    assert muted["output_si_sdr_db"] is None
    assert muted["target_projection_gain_db"] == -300
    assert muted["output_sdr_db"] == pytest.approx(0)
    tiny = reference_metrics(target, mixture, target * 0.001)
    assert tiny["target_projection_gain_db"] == pytest.approx(-60)
    assert tiny["output_sdr_db"] < 0.02


def test_metrics_do_not_secretly_align_a_delayed_output():
    rng = np.random.default_rng(500)
    target = rng.normal(0, 0.1, 32000)
    result = reference_metrics(target, target, np.roll(target, 100))
    assert result["output_si_sdr_db"] < -20
    assert result["alignment"] == "zero_lag_no_posthoc_shift_or_gain_normalization"


def test_no_target_controls_measure_output_energy_without_inventing_a_reference():
    signal = np.ones(32000) * 0.1
    assert output_energy_db(signal, signal * 0.1) == pytest.approx(-20)
    assert output_energy_db(np.zeros(32), np.zeros(32)) is None
    with pytest.raises(ConfigurationError):
        reference_metrics(np.zeros(32), np.ones(32), np.zeros(32))
    with pytest.raises(ConfigurationError):
        output_energy_db(np.zeros(32), np.ones(31))
    with pytest.raises(ConfigurationError):
        output_energy_db(np.ones(32), np.full(32, np.nan))
