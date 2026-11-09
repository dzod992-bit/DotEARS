"""Raw-audio CED/CLAP presence evidence, without query-conditioned generation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from dotears.ced import FLOAT_FILES, LocalCED
from dotears.core import model_fingerprint
from dotears.errors import ConfigurationError, DependencyError, MediaError
from dotears.sfx import PROMPTS
from dotears.sfx_auto import json_hash, pcm_hash

TYPES = ("whoosh", "notification", "ui_click")
CED_TAGS = (
    "Whoosh",
    "Chime",
    "Ding",
    "Ding-dong",
    "Beep",
    "Ringtone",
    "Clicking",
    "Sound effect",
    "Speech",
    "Music",
    "Electronic music",
    "Synthesizer",
    "Noise",
    "Wind",
    "Tap",
    "Thump",
    "Finger snapping",
)
VIEWS = ((0.15, 0.55), (0.35, 1.05), (1.0, 2.0))
FEATURE_NAMES = (
    [f"ced_logit:{view}:{tag}" for view in range(3) for tag in CED_TAGS]
    + [f"clap:{view}:{label}" for view in range(2) for label in PROMPTS]
    + [
        "log_rms",
        "log_wide_rms",
        "log_energy_contrast",
        "log_crest",
        "centroid",
        "flatness",
        "high_fraction",
        "low_fraction",
        "zcr",
        "core_fraction",
    ]
)
VERSION = "raw-ced-float-clap-multiscale-v1"
SCHEMA = "dotears.sfx-presence-policy/1"


def window(values: np.ndarray, peak: float, before: float, after: float, rate=32000):
    first, last = (
        max(0, round((peak - before) * rate)),
        min(len(values), round((peak + after) * rate)),
    )
    if last <= first:
        raise ConfigurationError("Empty raw SFX focus.")
    return values[first:last]


def dsp_features(short: np.ndarray, wide: np.ndarray, peak_in_short: float) -> list[float]:
    short, wide = short.astype(np.float64), wide.astype(np.float64)
    rms = float(np.sqrt(np.mean(short**2)))
    wrms = float(np.sqrt(np.mean(wide**2)))
    spectrum = np.abs(np.fft.rfft(short * np.hanning(len(short)))) ** 2
    spectrum += 1e-20
    frequencies = np.fft.rfftfreq(len(short), 1 / 32000)
    total = spectrum.sum()
    first, last = (
        max(0, round((peak_in_short - 0.04) * 32000)),
        min(len(short), round((peak_in_short + 0.04) * 32000)),
    )
    return [
        float(np.log10(max(rms, 1e-12))),
        float(np.log10(max(wrms, 1e-12))),
        float(np.log10(max(rms, 1e-12) / max(wrms, 1e-12))),
        float(np.log10(max(float(np.max(np.abs(short))), 1e-12) / max(rms, 1e-12))),
        float((spectrum * frequencies).sum() / total / 16000),
        float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)),
        float(spectrum[frequencies > 4000].sum() / total),
        float(spectrum[frequencies < 500].sum() / total),
        float(np.mean(np.diff(np.signbit(short)) != 0)),
        float(np.sum(short[first:last] ** 2) / max(np.sum(short**2), 1e-20)),
    ]


class RawSFXEngine:
    def __init__(self, ced: Path, clap: Path, cache: Path, threads=4):
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
            raise ConfigurationError("Raw SFX CPU threads must be between 1 and 16.")
        os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise DependencyError(
                "Raw SFX needs its prepared CPU torch/transformers environment."
            ) from exc

        self.torch, self.cache = torch, cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.ced = LocalCED(ced, threads)
        if set(CED_TAGS) - set(self.ced.labels):
            raise ConfigurationError("CED labels differ from the frozen raw feature set.")
        self.tag_indices = [self.ced.labels.index(label) for label in CED_TAGS]
        self.identity = {
            "ced_float_files": FLOAT_FILES,
            "clap_fingerprint": model_fingerprint(str(clap.resolve())),
            "prompt_sha256": json_hash(PROMPTS),
            "feature_version": VERSION,
        }
        self.model = (
            ClapModel.from_pretrained(
                str(clap.resolve()), local_files_only=True, use_safetensors=False, weights_only=True
            )
            .cpu()
            .eval()
        )
        self.processor = ClapProcessor.from_pretrained(str(clap.resolve()), local_files_only=True)
        text = self.processor(
            text=[v for values in PROMPTS.values() for v in values],
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            encoded = self.model.get_text_features(**text).reshape(len(PROMPTS), 3, -1).mean(1)
            self.text = torch.nn.functional.normalize(encoded, dim=-1).cpu().numpy()
        self.counters = {"windows": 0, "cache_hits": 0, "ced_forwards": 0, "clap_forwards": 0}

    def analyze(self, waveform: np.ndarray, peak: float) -> dict:
        import librosa

        waveform = np.asarray(waveform, dtype=np.float32)
        if (
            waveform.ndim != 1
            or not 4096 <= len(waveform) <= 320000
            or not np.isfinite(waveform).all()
            or not math.isfinite(peak)
            or not 0 <= peak <= len(waveform) / 32000
        ):
            raise ConfigurationError("Raw SFX needs finite mono 32kHz context and a valid peak.")
        key = json_hash([pcm_hash(waveform), peak, self.identity, VERSION])
        dest = self.cache / (key + ".json")
        self.counters["windows"] += 1
        if dest.exists():
            result = json.loads(dest.read_text("utf-8"))
            if result.get("input_key") != key:
                raise MediaError("Mismatched raw-feature cache.")
            self.counters["cache_hits"] += 1
            return result
        features, ced_scores, clap_scores = [], [], []
        for before, after in VIEWS:
            audio = window(waveform, peak, before, after)
            audio16 = librosa.resample(audio, orig_sr=32000, target_sr=16000)
            if len(audio16) < 2048:
                audio16 = np.pad(audio16, (0, 2048 - len(audio16)))
            scores = self.ced.predict(audio16)
            selected = scores[self.tag_indices]
            clipped = np.clip(selected, 1e-7, 1 - 1e-7)
            features.extend(np.log(clipped / (1 - clipped)).tolist())
            ced_scores.append(selected.tolist())
            self.counters["ced_forwards"] += 1
        for before, after in VIEWS[:2]:
            audio = librosa.resample(
                window(waveform, peak, before, after), orig_sr=32000, target_sr=48000
            )
            inputs = self.processor(
                audio=audio, sampling_rate=48000, return_tensors="pt", padding="repeatpad"
            )
            with self.torch.inference_mode():
                encoded = self.model.get_audio_features(**inputs)
                encoded = self.torch.nn.functional.normalize(encoded, dim=-1)[0].cpu().numpy()
            scores = encoded @ self.text.T
            clap_scores.append(scores.tolist())
            features.extend(scores.tolist())
            self.counters["clap_forwards"] += 1
        short, wide = [window(waveform, peak, *v) for v in VIEWS[:2]]
        features.extend(dsp_features(short, wide, min(0.15, peak)))
        # Disabled classes still compete here; never fall back to the next enabled type.
        target_scores = {
            label: float(clap_scores[0][list(PROMPTS).index(label)]) for label in TYPES
        }
        result = {
            "input_key": key,
            "features": features,
            "ced_scores": ced_scores,
            "clap_scores": clap_scores,
            "raw_target_scores": target_scores,
            "rms": float(np.sqrt(np.mean(short.astype(np.float64) ** 2))),
            "class_names_are_not_confirmation": True,
        }
        if len(features) != len(FEATURE_NAMES) or not np.isfinite(features).all():
            raise MediaError("Invalid raw-presence features.")
        dest.write_text(json.dumps(result, allow_nan=False), "utf-8")
        return result


def read_policy(path: Path) -> dict:
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema_version") != SCHEMA
        or policy.get("feature_names") != FEATURE_NAMES
        or policy.get("version") != VERSION
        or set(policy.get("heads", {})) != set(TYPES)
    ):
        raise ConfigurationError("Unsupported raw-presence policy.")
    for head in policy["heads"].values():
        vectors = [np.asarray(head[key], dtype=float) for key in ("mean", "scale", "coefficients")]
        if (
            any(v.shape != (len(FEATURE_NAMES),) for v in vectors)
            or not all(np.isfinite(v).all() for v in vectors)
            or np.any(vectors[1] <= 0)
            or not math.isfinite(head["intercept"])
            or not isinstance(head["enabled"], bool)
            or (head["threshold"] is not None and not math.isfinite(head["threshold"]))
            or (head["enabled"] and head["threshold"] is None)
        ):
            raise ConfigurationError("Invalid raw-presence head.")
    return policy


def decide(policy: dict, evidence: dict) -> dict:
    x = np.asarray(evidence["features"], dtype=float)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ConfigurationError("Invalid raw-presence evidence.")
    raw = evidence["raw_target_scores"]
    if set(raw) != set(TYPES) or not all(math.isfinite(v) for v in raw.values()):
        raise ConfigurationError("All raw target competitors are required.")
    order = sorted(raw, key=lambda key: -raw[key])
    best = order[0]
    heads = {}
    for label in TYPES:
        head = policy["heads"][label]
        score = float(
            ((x - head["mean"]) / head["scale"]) @ head["coefficients"] + head["intercept"]
        )
        reasons = []
        if not head["enabled"]:
            reasons.append("type_failed_independent_gate")
        if label != best or raw[best] <= 0 or raw[best] - raw[order[1]] <= 0:
            reasons.append("raw_audio_supports_another_type_or_is_ambiguous")
        if not math.isfinite(evidence["rms"]) or evidence["rms"] < 1e-5:
            reasons.append("insufficient_audio_energy")
        if head["threshold"] is None or score < head["threshold"]:
            reasons.append("below_frozen_threshold")
        heads[label] = {
            "raw_score": score,
            "threshold": head["threshold"],
            "rejection_reasons": reasons,
        }
    label = best if not heads[best]["rejection_reasons"] else "unknown"
    return {
        "label": label,
        "confidence": None,
        "status": "provisional_model_type" if label != "unknown" else "abstained",
        "reason": "passed_raw_consensus_and_presence_gate"
        if label != "unknown"
        else "raw_presence_checks_failed",
        "candidates": heads,
        "scores_are_probabilities": False,
        "human_confirmed": False,
    }
