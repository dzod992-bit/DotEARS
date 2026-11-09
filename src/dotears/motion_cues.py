"""Search for localized motion-sound shapes between the main attack markers."""

from __future__ import annotations

import numpy as np

from dotears.ui_acoustics import classify_form, short_ui_transient, spectrum


def localized_resonance(values: np.ndarray, rate: int, peak: float) -> dict:
    """Check that a muted impulse has a short, locally distinct midrange body."""
    times, frequencies, power = spectrum(values, rate, 0.010, 0.001)
    near = np.abs(times - peak) < 0.004
    flanks = (np.abs(times - peak) >= 0.019) & (np.abs(times - peak) <= 0.051)
    middle = np.flatnonzero((frequencies >= 400) & (frequencies < 2000))
    if near.sum() < 4 or flanks.sum() < 18 or not middle.size:
        return {}
    floor = np.quantile(power[flanks], 0.65, axis=0).clip(1e-7, None)
    observed = np.median(power[near], axis=0)
    excess = np.maximum(observed - floor, 0)
    index = middle[np.argmax(excess[middle])]
    core = slice(index - 1, index + 2)
    if excess[core].sum() <= 0:
        return {}
    return {
        "hz": round(float(frequencies[index])),
        "contrast_db": round(float(10 * np.log10(observed[core].sum() / floor[core].sum())), 1),
        "excess_fraction": round(float(excess[core].sum() / excess.sum()), 3),
    }


def transient_proposals(values: np.ndarray, rate: int):
    """Sweep the mixed track, using a fine local check for each plausible impulse."""
    duration = len(values) / rate
    found = []
    for anchor in np.arange(0.055, duration, 0.036):
        first = max(0, round((anchor - 0.115) * rate))
        last = min(len(values), round((anchor + 0.26) * rate))
        window = values[first:last]
        form, probes = short_ui_transient(window, rate, anchor - first / rate)
        if form == "unknown":
            continue
        probe = probes[form]
        peak = probe["relative_peak_seconds"]
        time = first / rate + peak
        if found and abs(found[-1]["time"] - time) < 0.009:
            continue
        classic, evidence = classify_form(window, rate, peak)
        if classic in {"beep", "chime"}:
            continue
        agrees = (
            classic == form and abs(evidence["impulse"]["relative_peak_seconds"] - peak) <= 0.016
        )
        resonance = localized_resonance(window, rate, peak) if form == "tap_like" else {}
        found.append(
            {
                "time": time,
                "form": form,
                "impulse": probe,
                "classic_agrees": agrees,
                "resonance": resonance,
            }
        )
    return found


def supported_transient(proposal: dict) -> bool:
    """Require agreement across measurements or a particularly distinct local shape."""
    impulse = proposal["impulse"]
    resonance = proposal["resonance"]
    if proposal["classic_agrees"]:
        return True
    if proposal["form"] == "click_like":
        return (
            impulse["contrast_db"] >= 26
            and impulse["width_ms"] <= 6
            and impulse["sharpness_db"] >= 5
        )
    return (
        resonance.get("contrast_db", 0) >= 18
        and resonance["excess_fraction"] >= 0.64
        and impulse["width_ms"] <= 15
    )


def pitch_glide(values: np.ndarray, rate: int, peak: float) -> dict | None:
    """Find a short moving spectral ridge supported across adjacent frames."""
    times, frequencies, power = spectrum(values, rate, 0.024, 0.005)
    flank = (times < peak - 0.06) | (times > peak + 0.26)
    near = np.flatnonzero((times >= peak - 0.055) & (times <= peak + 0.24))
    bins = np.flatnonzero((frequencies >= 400) & (frequencies <= 6200))
    bins = bins[(bins >= 2) & (bins + 2 < len(frequencies))]
    if flank.sum() < 10 or len(near) < 12 or not bins.size:
        return None
    floor = np.quantile(power[flank], 0.70, axis=0).clip(1e-7, None)
    residual = np.maximum(power[near] - floor, 0)
    strongest = bins[np.argmax(residual[:, bins], axis=1)]
    core = strongest[:, None] + np.arange(-1, 2)
    weights = np.take_along_axis(residual, core, axis=1)
    excess = weights.sum(axis=1)
    raw = np.take_along_axis(power[near], core, axis=1).sum(axis=1)
    fraction = excess / np.maximum(residual.sum(axis=1), 1e-7)
    pitches = (weights * frequencies[core]).sum(axis=1) / np.maximum(excess, 1e-7)
    good = (
        (raw >= 16 * floor[core].sum(axis=1))
        & (fraction >= 0.54)
        & (raw >= 0.24 * power[near].sum(axis=1))
    )
    # A real glide persists with small frequency steps. One loud frame cannot set its direction.
    step = np.abs(np.diff(np.log2(np.maximum(pitches, 1))))
    splits = np.flatnonzero(~good[1:] | ~good[:-1] | (step > 0.18)) + 1
    for group in sorted(np.split(np.arange(len(near)), splits), key=len, reverse=True):
        if len(group) < 13 or not good[group[0]]:
            continue
        part = np.log2(pitches[group])
        stamps = times[near[group]]
        if abs(stamps[0] - peak) > 0.055 or stamps[-1] > peak + 0.23:
            continue
        movement = np.median(part[-3:]) - np.median(part[:3])
        if not 0.50 <= abs(movement) <= 3.0 or abs(np.corrcoef(stamps, part)[0, 1]) < 0.96:
            continue
        if np.mean(np.diff(part) * np.sign(movement) >= -0.015) < 0.85:
            continue
        return {
            "direction": "up" if movement > 0 else "down",
            "start_hz": round(float(2 ** np.median(part[:3]))),
            "end_hz": round(float(2 ** np.median(part[-3:]))),
            "span_ms": round(1000 * float(stamps[-1] - stamps[0])),
        }
    return None
