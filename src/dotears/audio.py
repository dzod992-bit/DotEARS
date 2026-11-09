"""Librosa-based observations, without pretending DSP identifies sound semantics."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return samples.reshape(-1, channels).mean(axis=1).astype(np.float32) / 32768, rate


def perceive(path: Path, profile: str, max_events: int) -> dict[str, Any]:
    import librosa

    y, rate = read_wav(path)
    duration = len(y) / rate
    hop = max(1, round(rate * 0.01))
    n_fft = 1024
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    peak = float(np.max(np.abs(y)))
    envelope = librosa.onset.onset_strength(y=y, sr=rate, hop_length=hop, n_fft=n_fft)
    silence = peak < 10 ** (-55 / 20)
    frames = (
        np.array([], dtype=int)
        if silence
        else librosa.onset.onset_detect(
            onset_envelope=envelope,
            sr=rate,
            hop_length=hop,
            backtrack=False,
            delta=0.07 if profile == "motion" else 0.12,
            wait=10 if profile == "motion" else 15,
        )
    )
    # A cutoff relative to the strongest attack keeps numerical low-level artifacts out.
    top = max(float(np.max(envelope)), 1e-9)
    frames = np.array(
        [f for f in frames if envelope[f] >= 0.08 * top and f * hop / rate < duration],
        dtype=int,
    )
    strongest = sorted(frames, key=lambda f: float(envelope[f]), reverse=True)[:max_events]
    events = []
    for index, frame in enumerate(sorted(strongest), 1):
        timestamp = min(float(frame * hop / rate), duration)
        if timestamp >= duration:
            continue
        events.append(
            {
                "id": f"attack_{index:03d}",
                "start": max(0.0, timestamp - 0.03),
                "end": min(duration, timestamp + 0.12),
                "peak_time": timestamp,
                "category": "audio_attack",
                "label": "unknown",
                "confidence": None,
                "score": round(float(envelope[frame]) / top, 6),
                "score_kind": "relative_onset_strength_not_probability",
                "method": "librosa_spectral_flux",
                "clip_path": None,
                "mixture_status": "mixed_track_not_source_separated",
                "overlap": "unknown",
            }
        )
    rhythm: dict[str, Any] = {
        "bpm": None,
        "bpm_alternatives": [],
        "beat_times": [],
        "method": "librosa_beat_track",
        "is_music_evidence": False,
        "caveat": "Rhythmic candidates may be speech or SFX; short clips have tempo ambiguity.",
    }
    if len(frames) >= 4 and duration >= 2:
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=rate,
            hop_length=hop,
            trim=False,
        )
        bpm = float(np.asarray(tempo).reshape(-1)[0])
        if bpm > 0:
            rhythm.update(
                {
                    "bpm": round(bpm, 3),
                    "bpm_alternatives": [round(v, 3) for v in (bpm / 2, bpm * 2) if 40 <= v <= 240],
                    "beat_times": [
                        round(float(b * hop / rate), 4) for b in beats if b * hop / rate < duration
                    ],
                }
            )
    stride = max(1, round(0.1 * rate / hop))
    energies = [
        {
            "time": round(i * hop / rate, 4),
            "rms_dbfs": round(float(20 * np.log10(max(float(rms[i]), 1e-8))), 2),
        }
        for i in range(0, len(rms), stride)
        if i * hop / rate < duration
    ]
    changes = []
    for before, after in zip(energies, energies[1:], strict=False):
        delta = after["rms_dbfs"] - before["rms_dbfs"]
        if abs(delta) >= 6 and max(after["rms_dbfs"], before["rms_dbfs"]) > -55:
            changes.append({"time": after["time"], "delta_db": round(delta, 2)})
    return {
        "events": events,
        "rhythm": rhythm,
        "energy": energies,
        "energy_changes": changes,
        "duration_seconds": duration,
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-8)), 2),
        "silence": silence,
        "onset_candidates_total": len(frames),
    }
