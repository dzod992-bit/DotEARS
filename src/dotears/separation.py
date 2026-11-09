"""Signal-level separation measurements; no semantic labels are inferred here."""

from __future__ import annotations

import math

import numpy as np

from dotears.errors import ConfigurationError


def _signal(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ConfigurationError("Separation metrics require non-empty finite mono vectors.")
    return value


def _ratio_db(numerator: float, denominator: float) -> float:
    return 10 * math.log10(max(numerator, 1e-30) / max(denominator, 1e-30))


def output_energy_db(mixture: np.ndarray, estimate: np.ndarray) -> float | None:
    mixture, estimate = _signal(mixture), _signal(estimate)
    if mixture.shape != estimate.shape:
        raise ConfigurationError("Input and separated waveform lengths must match.")
    energy = float(mixture @ mixture)
    if energy < 1e-20:
        return None
    return _ratio_db(float(estimate @ estimate), energy)


def reference_metrics(target: np.ndarray, mixture: np.ndarray, estimate: np.ndarray) -> dict:
    """Zero-lag SI-SDR plus scale-dependent errors and target retention.

    Target/mixture are known construction components. Never use these measurements as
    ground truth for a real clip whose clean foreground is unavailable. SI-SDR alone can
    reward near-muted outputs, so projection gain and scale-dependent SDR are reported too.
    """
    target, mixture, estimate = (_signal(value) for value in (target, mixture, estimate))
    if not target.shape == mixture.shape == estimate.shape:
        raise ConfigurationError("Reference, mixture and estimate lengths must match.")
    reference = target - target.mean()
    ref_energy = float(reference @ reference)
    if ref_energy < 1e-20:
        raise ConfigurationError("A zero-energy target cannot define a separation benchmark.")

    def measurements(value):
        centered = value - value.mean()
        gain = float(reference @ centered) / ref_energy
        residual = centered - gain * reference
        si_sdr = (
            _ratio_db(float((gain * reference) @ (gain * reference)), float(residual @ residual))
            if float(centered @ centered) >= 1e-20
            else None
        )
        error = value - target
        return si_sdr, _ratio_db(float(target @ target), float(error @ error)), gain

    input_si, input_sdr, _ = measurements(mixture)
    output_si, output_sdr, gain = measurements(estimate)
    return {
        "input_si_sdr_db": input_si,
        "output_si_sdr_db": output_si,
        "si_sdr_improvement_db": output_si - input_si
        if output_si is not None and input_si is not None
        else None,
        "input_sdr_db": input_sdr,
        "output_sdr_db": output_sdr,
        "sdr_improvement_db": output_sdr - input_sdr,
        "target_projection_scale": gain,
        "target_projection_gain_db": 20 * math.log10(max(abs(gain), 1e-15)),
        "output_to_input_energy_db": output_energy_db(mixture, estimate),
        "alignment": "zero_lag_no_posthoc_shift_or_gain_normalization",
    }
