"""Query-free 8ms/1ms attack morphology for short UI clicks and competing sounds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dotears.errors import ConfigurationError
from dotears.sfx_presence_deep import FEATURE_NAMES as DEEP_NAMES
from dotears.sfx_presence_deep import LatentSFXEngine, validate_heads

VERSION = "raw-multiscale-microtransient-v3"
SCHEMA = "dotears.sfx-presence-policy/3"
BANDS = (0, 250, 500, 1000, 2000, 4000, 8000, 12000, 16001)
MICRO_NAMES = [
    f"micro_band:{i}:{name}"
    for i in range(8)
    for name in (
        "excess_db",
        "core_fraction",
        "prerise_fraction",
        "late_fraction",
        "width80",
        "center",
    )
]
MICRO_NAMES += [
    "micro_core5",
    "micro_core10",
    "micro_core20",
    "micro_core40",
    "micro_width80",
    "micro_width50",
    "micro_decay",
    "micro_rise",
    "micro_flatness",
    "micro_centroid",
    "micro_periodicity",
    "micro_zcr",
]
FEATURE_NAMES = DEEP_NAMES + MICRO_NAMES


def micro_features(waveform: np.ndarray, peak: float) -> list[float]:
    audio = np.asarray(waveform, dtype=np.float64)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not 0 <= peak <= len(audio) / 32000:
        raise ConfigurationError("Invalid raw transient context.")
    before, after = 5120, 7680
    center = round(peak * 32000)
    first, last = max(0, center - before), min(len(audio), center + after)
    local = np.zeros(before + after)
    offset = first - (center - before)
    local[offset : offset + last - first] = audio[first:last]
    padded = np.pad(local, (128, 128))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 256)[::32]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(256), axis=1)) ** 2
    times = np.arange(len(frames)) * 0.001 - 0.16
    freq = np.fft.rfftfreq(256, 1 / 32000)
    prefix = (times >= -0.15) & (times <= -0.04)
    baseline = np.median(spectrum[prefix], axis=0)
    residual = np.maximum(spectrum - baseline, 0)
    near = np.abs(times) <= 0.025
    core = np.abs(times) <= 0.01

    def width(power, q):
        total = power.sum()
        if total < 1e-20:
            return 0.0
        cumulative = np.cumsum(power) / total
        return float(
            (
                times[min(np.searchsorted(cumulative, (1 + q) / 2), len(times) - 1)]
                - times[min(np.searchsorted(cumulative, (1 - q) / 2), len(times) - 1)]
            )
            / 0.4
        )

    features = []
    for low, high in zip(BANDS[:-1], BANDS[1:], strict=True):
        indices = (freq >= low) & (freq < high)
        power = residual[:, indices].sum(1)
        total = max(power.sum(), 1e-20)
        original = spectrum[:, indices].sum(1)
        base = baseline[indices].sum()
        features += [
            float(
                np.clip(10 * np.log10(max(original[near].max(), 1e-20) / max(base, 1e-20)), -60, 60)
            ),
            float(power[core].sum() / total),
            float(power[(times >= -0.1) & (times <= -0.02)].sum() / total),
            float(power[(times >= 0.02) & (times <= 0.16)].sum() / total),
            width(power, 0.8),
            float((power * times).sum() / total / 0.4),
        ]
    power = residual.sum(1)
    total = max(power.sum(), 1e-20)
    features += [
        float(power[np.abs(times) <= span].sum() / total) for span in (0.005, 0.01, 0.02, 0.04)
    ]
    features += [width(power, 0.8), width(power, 0.5)]

    def ratio(numerator, denominator):
        return float(
            np.clip(
                np.log10(max(power[numerator].sum(), 1e-20) / max(power[denominator].sum(), 1e-20)),
                -6,
                6,
            )
        )

    features += [
        ratio((times >= 0.02) & (times <= 0.1), (times >= 0) & (times <= 0.02)),
        ratio((times >= -0.08) & (times <= -0.02), (times >= -0.02) & (times <= 0)),
    ]
    core_spectrum = spectrum[np.abs(times) <= 0.008].mean(0) + 1e-20
    features += [
        float(np.exp(np.mean(np.log(core_spectrum))) / np.mean(core_spectrum)),
        float((core_spectrum * freq).sum() / core_spectrum.sum() / 16000),
    ]
    short = local[before - 640 : before + 640].copy()
    short -= short.mean()
    corr = np.correlate(short, short, mode="full")[len(short) - 1 :]
    periodic = float(np.max(corr[32:321]) / max(corr[0], 1e-20))
    features += [periodic, float(np.mean(np.diff(np.signbit(short)) != 0))]
    if len(features) != len(MICRO_NAMES) or not np.isfinite(features).all():
        raise ConfigurationError("Invalid microtransient features.")
    return features


class MicroSFXEngine:
    def __init__(self, ced: Path, clap: Path, cache: Path, threads=4):
        self.latent = LatentSFXEngine(ced, clap, cache, threads)
        self.identity = {"latent_backbone": self.latent.identity, "micro_version": VERSION}
        self.counters = self.latent.counters

    def analyze(self, waveform: np.ndarray, peak: float) -> dict:
        base = self.latent.analyze(waveform, peak)
        return {**base, "features": base["features"] + micro_features(waveform, peak)}


def read_policy(path: Path) -> dict:
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema_version") != SCHEMA
        or policy.get("version") != VERSION
        or policy.get("feature_names") != FEATURE_NAMES
    ):
        raise ConfigurationError("Unsupported microtransient presence policy.")
    validate_heads(policy, len(FEATURE_NAMES))
    return policy
