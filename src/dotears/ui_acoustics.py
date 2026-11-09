"""Conservative acoustic forms for a window already tagged as a UI signal."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def spectrum(values: np.ndarray, rate: int, width: float, hop: float):
    size, step = round(width * rate), round(hop * rate)
    starts = np.arange(0, len(values) - size + 1, step)
    samples = values[starts[:, None] + np.arange(size)]
    power = np.mean(
        np.abs(np.fft.rfft(samples * np.hanning(size)[None, :, None], axis=1)) ** 2, axis=2
    )
    return (starts + size / 2) / rate, np.fft.rfftfreq(size, 1 / rate), power


def tonal_lines(values: np.ndarray, rate: int, peak: float) -> list[dict]:
    times, frequencies, power = spectrum(values, rate, 0.032, 0.005)
    center = (times >= peak - 0.085) & (times <= peak + 0.165)
    reference = (times < peak - 0.075) | (times > peak + 0.27)
    if center.sum() < 5 or reference.sum() < 7:
        return []
    floor = np.quantile(power[reference], 0.62, axis=0) + 1e-9
    prominence = np.max(power[center], axis=0) / floor
    possible = np.flatnonzero((frequencies >= 620) & (frequencies <= 6500))
    candidates = possible[np.argsort(-prominence[possible])[:36]]
    found = []
    for k in candidates:
        if k < 6 or k + 6 >= power.shape[1] or prominence[k] < 12:
            continue
        if any(abs(k - row["bin"]) <= 3 for row in found):
            continue
        core = power[:, k - 1 : k + 2].sum(axis=1)
        side = np.column_stack((power[:, k - 6 : k - 2], power[:, k + 3 : k + 7]))
        local = 3 * np.median(side, axis=1)
        track = np.maximum(core - local, 0)
        baseline = np.quantile(track[reference], 0.65)
        start = int(np.argmax(np.where(center, track, -1)))
        if track[start] < max(0.0001, 15 * baseline):
            continue
        valid = (
            (track >= max(0.11 * track[start], 5 * baseline))
            & (core >= 3.6 * local)
            & (power[:, k] >= 0.35 * np.max(power[:, k - 2 : k + 3], axis=1))
        )
        if not valid[start]:
            continue
        first = last = start
        while first and valid[first - 1]:
            first -= 1
        while last + 1 < len(times) and valid[last + 1]:
            last += 1
        span = times[last] - times[first] + 0.005
        if (
            span < 0.045
            or span > 0.32
            or first == 0
            or last + 1 == len(times)
            or not peak - 0.095 <= times[first] <= peak + 0.09
        ):
            continue
        part = track[first : last + 1]
        quarter = max(2, len(part) // 4)
        front, tail = np.median(part[:quarter]), np.median(part[-quarter:])
        purity = float(np.median(part / np.maximum(power[first : last + 1].sum(axis=1), 1e-9)))
        found.append(
            {
                "bin": int(k),
                "hz": round(float(frequencies[k])),
                "start": float(times[first]),
                "span_ms": round(1000 * span),
                "prominence_db": float(10 * np.log10(track[start] / max(baseline, 1e-9))),
                "decay_db": float(10 * np.log10(max(front, 1e-9) / max(tail, 1e-9))),
                "purity": purity,
                "energy": float(np.sum(part)),
            }
        )
    return sorted(found, key=lambda row: -row["energy"])


def transient(values: np.ndarray, rate: int, peak: float) -> dict:
    times, frequencies, power = spectrum(values, rate, 0.008, 0.002)
    center = np.abs(times - peak) <= 0.022
    reference = (times < peak - 0.043) | (times > peak + 0.095)
    if center.sum() < 3 or reference.sum() < 7:
        return {}
    floor = np.quantile(power[reference], 0.55, axis=0)
    excess = np.maximum(power - floor, 0)
    combined = excess[:, frequencies >= 250].sum(axis=1)
    index = int(np.argmax(np.where(center, combined, -1)))
    top = combined[index]
    if top < 1e-5:
        return {}
    surround = np.quantile(combined[reference], 0.85)
    high = excess[:, frequencies >= 2300].sum(axis=1)
    high_reference = np.quantile(high[reference], 0.85)
    mid = combined - high
    mid_reference = np.quantile(mid[reference], 0.85)
    first = last = index
    while first and combined[first - 1] >= 0.32 * top:
        first -= 1
    while last + 1 < len(times) and combined[last + 1] >= 0.32 * top:
        last += 1
    return {
        "relative_peak_seconds": float(times[index]),
        "width_ms": round(1000 * (times[last] - times[first] + 0.008)),
        "contrast_db": float(10 * np.log10(top / max(surround, 1e-9))),
        "high_contrast_db": float(
            10 * np.log10(max(high[index], 1e-9) / max(high_reference, 1e-9))
        ),
        "mid_contrast_db": float(10 * np.log10(max(mid[index], 1e-9) / max(mid_reference, 1e-9))),
        "high_fraction": float(excess[index, frequencies >= 2300].sum() / top),
    }


def short_ui_transient(values: np.ndarray, rate: int, peak: float) -> tuple[str, dict]:
    """Compare fast upper-band clicks and muted midrange taps against local flanks."""
    times, frequencies, power = spectrum(values, rate, 0.004, 0.001)
    step = max(1, round(rate * 0.001)) / rate
    before = np.rint(np.arange(-0.046, -0.018, 0.003) / step).astype(int)
    after = np.rint(np.arange(0.019, 0.047, 0.003) / step).astype(int)
    candidates = np.flatnonzero(np.abs(times - peak) <= 0.032)
    candidates = candidates[(candidates >= -before[0]) & (candidates < len(times) - after[-1])]
    if not candidates.size:
        return "unknown", {}

    edges = [250, 500, 900, 1500, 2300, 3600, 5600, 8500, 12000]
    bands = np.column_stack(
        [
            power[:, (frequencies >= low) & (frequencies < high)].sum(axis=1)
            for low, high in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    floor = np.maximum(
        np.quantile(bands[candidates[:, None] + before], 0.65, axis=1),
        np.quantile(bands[candidates[:, None] + after], 0.65, axis=1),
    ).clip(1e-6, None)
    probes = {}
    accepted = []
    for name, selection in (("click_like", slice(4, None)), ("tap_like", slice(0, 4))):
        track = bands[:, selection].sum(axis=1)
        background = floor[:, selection].sum(axis=1)
        ratio = track[candidates] / np.maximum(background, 0.20 * np.quantile(background, 0.70))
        best = int(np.argmax(ratio))
        index = candidates[best]
        baseline = background[best]
        local = np.maximum(track - baseline, 0)
        top = local[index]
        if top <= 0:
            continue
        level = max(0.20 * top, 0.35 * baseline)
        first = last = index
        while first > index - 32 and local[first - 1] >= level:
            first -= 1
        while last < index + 32 and local[last + 1] >= level:
            last += 1
        width = float((last - first + 1) * step * 1000)
        excess = np.maximum(bands[index] - floor[best], 0)
        share = excess / excess.sum()
        contrast = float(10 * np.log10(ratio[best]))
        contrast_by_band = 10 * np.log10(np.maximum(bands[index], 1e-9) / floor[best])
        sharpness = float(10 * np.log10(top / np.mean(local[index - 6 : index + 10])))
        time = float(times[index])
        probes[name] = {
            "relative_peak_seconds": time,
            "width_ms": round(width, 1),
            "rise_ms": round(float((index - first) * step * 1000), 1),
            "fall_ms": round(float((last - index) * step * 1000), 1),
            "contrast_db": round(contrast, 1),
            "high_excess_fraction": round(float(share[4:].sum()), 3),
            "sharpness_db": round(sharpness, 1),
        }
        if abs(time - peak) > 0.024:
            continue
        if (
            name == "click_like"
            and 1.9 <= width <= 12.1
            and contrast >= 14
            and share[4:].sum() >= 0.65
            and sharpness >= 3
            and np.count_nonzero(contrast_by_band[4:] > 6) >= 2
        ):
            accepted.append(name)
        if (
            name == "tap_like"
            and 3.9 <= width <= 20.1
            and share[4:].sum() <= 0.20
            and share[1:4].sum() >= 0.65
            and share[0] <= 0.35
            and bands[index, 4:].sum() <= 0.35 * bands[index].sum()
            and np.max(contrast_by_band[1:4]) >= 15
        ):
            accepted.append(name)
    return accepted[0] if len(accepted) == 1 else "unknown", probes


def repeats_on_beat(path: Path, clip_start: float, bpm: float, cue: float, form: str) -> bool:
    """Reject a cue when closely matching waveforms recur at the same musical subdivision."""
    from scipy.signal import butter, correlate, sosfiltfilt

    if not 40 <= bpm <= 240:
        return False
    center = clip_start + cue
    with wave.open(str(path), "rb") as source:
        rate, channels = source.getframerate(), source.getnchannels()
        if source.getsampwidth() != 2 or rate < 8000:
            return False
        limits = (2300, min(11000, 0.46 * rate)) if form == "click_like" else (250, 2300)
        filter_ = butter(4, limits, btype="bandpass", fs=rate, output="sos")
        half = round(rate * (0.005 if form == "click_like" else 0.008))
        radius = round(rate * 0.045)
        drift = round(rate * 0.016)

        def band_window(time: float):
            first = round(time * rate) - radius
            if first < 0 or first + 2 * radius > source.getnframes():
                return None
            source.setpos(first)
            samples = np.frombuffer(source.readframes(2 * radius), dtype="<i2")
            return sosfiltfilt(filter_, samples.reshape(-1, channels), axis=0)

        base = band_window(center)
        if base is None:
            return False
        template = base[radius - half : radius + half]
        if np.sum(template.mean(axis=1) ** 2) >= 0.5 * np.max(np.sum(template**2, axis=0)):
            channel = None
            template = template.mean(axis=1)
        else:
            channel = int(np.argmax(np.sum(template**2, axis=0)))
            template = template[:, channel]
        energy = float(template @ template)
        if energy < 1:
            return False
        matches = 0
        for subdivision in (*range(-36, -1), *range(2, 37)):
            other = band_window(center + subdivision * 30 / bpm)
            if other is None:
                continue
            search = other[radius - half - drift : radius + half + drift]
            search = search.mean(axis=1) if channel is None else search[:, channel]
            product = correlate(search, template, mode="valid")
            cumulative = np.r_[0.0, np.cumsum(search**2)]
            power = cumulative[len(template) :] - cumulative[: -len(template)]
            scores = product / np.sqrt(energy * np.maximum(power, 1))
            if np.any((scores >= 0.88) & (power >= 0.18 * energy)):
                matches += 1
                if matches >= 3:
                    return True
    return False


def classify_form(values: np.ndarray, rate: int, peak: float) -> tuple[str, dict]:
    """Distinguish shapes rather than guessing a notification or a physical action."""
    lines = tonal_lines(values, rate, peak)
    impulse = transient(values, rate, peak)
    evidence = {"scope": "localized_cue_in_mixed_window", "lines": lines[:4], "impulse": impulse}
    if lines:
        main = lines[0]
        companions = []
        for other in lines[1:]:
            ratio = max(main["hz"], other["hz"]) / min(main["hz"], other["hz"])
            if (
                abs(main["start"] - other["start"]) <= 0.04
                and abs(ratio - round(ratio)) > 0.10
                and other["energy"] >= 0.10 * main["energy"]
            ):
                companions.append(other)
        if (
            companions
            and main["span_ms"] >= 75
            and main["decay_db"] >= 3.5
            and any(row["decay_db"] >= 2.5 for row in companions)
        ):
            return "chime", evidence
        if (
            not companions
            and main["span_ms"] >= 70
            and abs(main["decay_db"]) <= 3
            and main["purity"] >= 0.30
            and abs(main["start"] - peak) <= 0.05
        ):
            return "beep", evidence
        if (
            not companions
            and 45 <= main["span_ms"] <= 170
            and main["decay_db"] >= 4
            and main["purity"] >= 0.30
            and main["prominence_db"] >= 22
            and abs(main["start"] - peak) <= 0.06
        ):
            return "chime", evidence
    if impulse.get("contrast_db", 0) >= 5:
        if (
            impulse["width_ms"] <= 16
            and impulse["high_fraction"] >= 0.59
            and impulse["high_contrast_db"] >= 8
        ):
            return "click_like", evidence
        if (
            10 <= impulse["width_ms"] <= 30
            and impulse["high_fraction"] <= 0.21
            and impulse["mid_contrast_db"] >= 10
        ):
            return "tap_like", evidence
    return "unknown", evidence


def has_ring_support(raw_evidence: dict) -> bool:
    """Require tonal UI context in both audio views before proposing a standalone ring."""
    for view in ("raw_cosines", "contrast_cosines"):
        scores = raw_evidence.get(view, {})
        ring = scores.get("notification", -1.0)
        if (
            ring < 0.16
            or ring - max(scores.get("ui_tone", 1.0), scores.get("ui_click", 1.0)) < 0.04
        ):
            return False
    return True
