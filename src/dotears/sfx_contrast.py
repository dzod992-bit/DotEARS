"""Broad UI SFX presence from raw and query-free spectral-contrast audio views."""

from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.signal import istft, stft

from dotears.errors import ConfigurationError, MediaError
from dotears.sfx import PROMPTS
from dotears.sfx_auto import json_hash, pcm_hash
from dotears.sfx_presence import CED_TAGS, RawSFXEngine, dsp_features, window
from dotears.sfx_presence_deep import score_head, validate_heads
from dotears.sfx_taxonomy import UI_TAXONOMY, UI_TYPES

VERSION = "raw-and-spectral-contrast-clap-embedding-v1"
SCHEMA = "dotears.sfx-presence-policy/4"
FEATURE_NAMES = (
    [f"raw_clap_embedding:{i}" for i in range(512)]
    + [f"contrast_clap_embedding:{i}" for i in range(512)]
    + [f"{view}_clap_cosine:{label}" for view in ("raw", "contrast") for label in PROMPTS]
    + [f"raw_ced_logit:{tag}" for tag in CED_TAGS]
    + [f"raw_dsp:{i}" for i in range(10)]
)


def spectral_contrast(values: np.ndarray, peak: float) -> np.ndarray:
    """A fixed numerical view, not a separated or semantic SFX estimate."""
    audio = np.asarray(values, dtype=np.float64)
    if (
        audio.ndim != 1
        or not 4096 <= len(audio) <= 128000
        or not np.isfinite(audio).all()
        or not math.isfinite(peak)
        or not 0 <= peak <= len(audio) / 32000
    ):
        raise ConfigurationError("Invalid SFX contrast context.")
    _, times, spectrum = stft(audio, fs=32000, nperseg=1024, noverlap=896)
    power = np.abs(spectrum) ** 2
    outside = (times < peak - 0.2) | (times > peak + 0.65)
    # At boundaries use the available remainder. No target word or label is involved.
    background = np.quantile(power[:, outside] if outside.sum() >= 8 else power, 0.25, axis=1)
    gain = np.sqrt(np.maximum(power - background[:, None], 0) / np.maximum(power, 1e-20))
    gain = np.maximum(gain, 0.15)
    _, reconstructed = istft(spectrum * gain, fs=32000, nperseg=1024, noverlap=896)
    result = reconstructed[: len(audio)].astype(np.float32)
    if result.shape != audio.shape or not np.isfinite(result).all():
        raise MediaError("Invalid spectral-contrast view.")
    return result


class ContrastSFXEngine:
    def __init__(self, ced: Path, clap: Path, cache: Path, threads=4):
        self.base = RawSFXEngine(ced, clap, cache, threads)
        self.identity = {
            "backbone": self.base.identity,
            "feature_version": VERSION,
            "label_taxonomy": UI_TAXONOMY,
        }
        self.counters = {"windows": 0, "cache_hits": 0, "ced_forwards": 0, "clap_forwards": 0}

    def analyze(self, waveform: np.ndarray, peak: float) -> dict:
        import librosa

        values = np.asarray(waveform, dtype=np.float32)
        contrast = spectral_contrast(values, peak)
        key = json_hash([pcm_hash(values), peak, self.identity])
        path = self.base.cache / (key + "-contrast.json")
        self.counters["windows"] += 1
        if path.exists():
            result = json.loads(path.read_text("utf-8"))
            if (
                result.get("input_key") != key
                or len(result.get("features", [])) != len(FEATURE_NAMES)
                or not np.isfinite(result["features"]).all()
            ):
                raise MediaError("Invalid contrast-feature cache.")
            self.counters["cache_hits"] += 1
            return result
        embeddings, similarities = [], []
        for view in (values, contrast):
            focus = window(view, peak, 0.15, 0.55)
            audio = librosa.resample(focus, orig_sr=32000, target_sr=48000)
            inputs = self.base.processor(
                audio=audio, sampling_rate=48000, return_tensors="pt", padding="repeatpad"
            )
            with self.base.torch.inference_mode():
                encoded = self.base.model.get_audio_features(**inputs)
                encoded = self.base.torch.nn.functional.normalize(encoded, dim=-1)[0].cpu().numpy()
            if encoded.shape != (512,):
                raise MediaError("Unexpected CLAP embedding dimension.")
            embeddings.append(encoded)
            similarities.append(encoded @ self.base.text.T)
            self.counters["clap_forwards"] += 1
        short = window(values, peak, 0.15, 0.55)
        audio = librosa.resample(short, orig_sr=32000, target_sr=16000)
        if len(audio) < 2048:
            audio = np.pad(audio, (0, 2048 - len(audio)))
        tags = self.base.ced.predict(audio)[self.base.tag_indices]
        self.counters["ced_forwards"] += 1
        clipped = np.clip(tags, 1e-7, 1 - 1e-7)
        features = np.r_[
            *embeddings,
            *similarities,
            np.log(clipped / (1 - clipped)),
            dsp_features(short, window(values, peak, 0.35, 1.05), min(0.15, peak)),
        ]
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise MediaError("Invalid contrast-presence features.")
        raw = dict(zip(PROMPTS, similarities[0].tolist(), strict=True))
        result = {
            "input_key": key,
            "features": features.tolist(),
            "raw_target_scores": {
                "whoosh": raw["whoosh"],
                "notification": raw["notification"],
                "ui_sound": max(raw["ui_click"], raw["ui_tone"]),
            },
            "raw_cosines": raw,
            "contrast_cosines": dict(zip(PROMPTS, similarities[1].tolist(), strict=True)),
            "raw_ced_scores": tags.tolist(),
            "rms": float(np.sqrt(np.mean(short.astype(np.float64) ** 2))),
            "contrast_is_not_isolated_effect": True,
        }
        # A stopped scan must not leave a half-written entry for the next run.
        pending = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            pending.write_text(json.dumps(result, allow_nan=False), "utf-8")
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        return result


def read_policy(path: Path) -> dict:
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema_version") != SCHEMA
        or policy.get("version") != VERSION
        or policy.get("label_taxonomy") != UI_TAXONOMY
        or policy.get("feature_names") != FEATURE_NAMES
        or set(policy.get("heads", {})) != set(UI_TYPES)
    ):
        raise ConfigurationError("Unsupported broad UI contrast policy.")
    # Reuse numerical validation without changing the frozen legacy type vocabulary.
    validate_heads(
        {
            "heads": {
                old: policy["heads"][new]
                for old, new in zip(("whoosh", "notification", "ui_click"), UI_TYPES, strict=True)
            }
        },
        len(FEATURE_NAMES),
    )
    return policy


def decide(policy: dict, evidence: dict) -> dict:
    x = np.asarray(evidence["features"], dtype=float)
    raw = evidence["raw_target_scores"]
    if (
        x.shape != (len(FEATURE_NAMES),)
        or not np.isfinite(x).all()
        or set(raw) != set(UI_TYPES)
        or not all(math.isfinite(v) for v in raw.values())
    ):
        raise ConfigurationError("Invalid broad UI contrast evidence.")
    candidates = {}
    for label in UI_TYPES:
        head = policy["heads"][label]
        score = score_head(head, x)
        candidates[label] = {
            "raw_score": score,
            "threshold": head["threshold"],
            "support": (score - head["competition_center"]) / head["competition_scale"],
        }
    order = sorted(UI_TYPES, key=lambda label: -candidates[label]["support"])
    best = order[0]
    for label in UI_TYPES:
        head, candidate = policy["heads"][label], candidates[label]
        reasons = []
        if label != best or candidates[best]["support"] - candidates[order[1]]["support"] < 0.15:
            reasons.append("ambiguous_or_competing_raw_audio_heads")
        if not head["enabled"]:
            reasons.append("type_failed_independent_gate")
        if raw[label] <= 0:
            reasons.append("nonpositive_raw_audio_similarity")
        if not math.isfinite(evidence["rms"]) or evidence["rms"] < 1e-5:
            reasons.append("insufficient_audio_energy")
        if head["threshold"] is None or candidate["raw_score"] < head["threshold"]:
            reasons.append("below_frozen_threshold")
        candidate["rejection_reasons"] = reasons
    label = best if not candidates[best]["rejection_reasons"] else "unknown"
    return {
        "label": label,
        "confidence": None,
        "status": "provisional_model_type" if label != "unknown" else "abstained",
        "reason": "raw_and_contrast_presence_check",
        "candidates": candidates,
        "scores_are_probabilities": False,
        "human_confirmed": False,
    }
