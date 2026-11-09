"""Raw latent presence: learned audio heads choose the type, all classes compete."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dotears.errors import ConfigurationError, MediaError
from dotears.sfx_auto import json_hash
from dotears.sfx_presence import FEATURE_NAMES as BASE_NAMES
from dotears.sfx_presence import TYPES, RawSFXEngine, window

VERSION = "raw-ced-latent-clap-v2"
SCHEMA = "dotears.sfx-presence-policy/2"
FEATURE_NAMES = (
    BASE_NAMES
    + [f"ced_short:{i}" for i in range(256)]
    + [f"ced_short_minus_context:{i}" for i in range(256)]
)


class LatentSFXEngine:
    def __init__(self, ced: Path, clap: Path, cache: Path, threads=4):
        self.base = RawSFXEngine(ced, clap, cache, threads)
        self.identity = {"raw_backbone": self.base.identity, "latent_version": VERSION}
        self.counters = self.base.counters
        self.counters["latent_forwards"] = 0
        self.counters["latent_cache_hits"] = 0

    def augment(self, base: dict, waveform: np.ndarray, peak: float) -> dict:
        import librosa

        key = json_hash([base["input_key"], self.identity, VERSION])
        path = self.base.cache / (key + "-latent.json")
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            if data["input_key"] != key:
                raise MediaError("Wrong latent cache key.")
            latent = np.asarray(data["latent"], dtype=float)
            self.counters["latent_cache_hits"] += 1
        else:
            embeddings = []
            for before, after in ((0.15, 0.55), (1.0, 2.0)):
                audio = librosa.resample(
                    window(waveform, peak, before, after), orig_sr=32000, target_sr=16000
                )
                if len(audio) < 2048:
                    audio = np.pad(audio, (0, 2048 - len(audio)))
                embeddings.append(self.base.ced.embedding(audio))
                self.counters["latent_forwards"] += 1
            latent = np.r_[embeddings[0], embeddings[0] - embeddings[1]]
            path.write_text(
                json.dumps({"input_key": key, "latent": latent.tolist()}, allow_nan=False), "utf-8"
            )
        if latent.shape != (512,) or not np.isfinite(latent).all():
            raise MediaError("Invalid CED latent evidence.")
        return {
            **base,
            "base_features": base["features"],
            "features": base["features"] + latent.tolist(),
            "latent_input_key": key,
        }

    def analyze(self, waveform: np.ndarray, peak: float) -> dict:
        return self.augment(self.base.analyze(waveform, peak), waveform, peak)


def read_policy(path: Path) -> dict:
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema_version") != SCHEMA
        or policy.get("version") != VERSION
        or policy.get("feature_names") != FEATURE_NAMES
        or set(policy.get("heads", {})) != set(TYPES)
    ):
        raise ConfigurationError("Unsupported latent-presence policy.")
    validate_heads(policy, len(FEATURE_NAMES))
    return policy


def validate_heads(policy: dict, total_dimensions: int) -> None:
    if set(policy.get("heads", {})) != set(TYPES):
        raise ConfigurationError("All presence heads are required.")
    for head in policy["heads"].values():
        dimensions = head.get("feature_count", total_dimensions)
        indices = head.get("feature_indices")
        if indices is not None:
            if (
                len(indices) != dimensions
                or len(set(indices)) != dimensions
                or any(
                    not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < total_dimensions
                    for i in indices
                )
            ):
                raise ConfigurationError("Invalid presence feature indices.")
        elif dimensions not in (len(BASE_NAMES), len(FEATURE_NAMES), total_dimensions):
            raise ConfigurationError("Invalid presence feature dimension.")
        vectors = [np.asarray(head[k], dtype=float) for k in ("mean", "scale")]
        if head.get("kind", "linear") == "linear":
            vectors.append(np.asarray(head["coefficients"], dtype=float))
        elif head.get("kind") == "rbf":
            support = np.asarray(head["support_vectors"], dtype=float)
            dual = np.asarray(head["dual_coefficients"], dtype=float)
            if (
                support.ndim != 2
                or support.shape[1] != dimensions
                or not 1 <= len(support) <= 2000
                or dual.shape != (len(support),)
                or not np.isfinite(support).all()
                or not np.isfinite(dual).all()
                or not math.isfinite(head["gamma"])
                or head["gamma"] <= 0
            ):
                raise ConfigurationError("Invalid RBF presence support parameters.")
        else:
            raise ConfigurationError("Unsupported presence classifier.")
        if (
            any(v.shape != (dimensions,) for v in vectors)
            or not all(np.isfinite(v).all() for v in vectors)
            or np.any(vectors[1] <= 0)
            or not all(
                math.isfinite(head[k])
                for k in ("intercept", "competition_center", "competition_scale")
            )
            or head["competition_scale"] <= 0
            or not isinstance(head["enabled"], bool)
            or (head["threshold"] is not None and not math.isfinite(head["threshold"]))
            or (head["enabled"] and head["threshold"] is None)
        ):
            raise ConfigurationError("Invalid latent presence head.")


def score_head(head: dict, features: np.ndarray) -> float:
    dimensions = head.get("feature_count", len(head["mean"]))
    values = np.asarray(features, dtype=float)
    active = (
        values[head["feature_indices"]]
        if head.get("feature_indices") is not None
        else values[:dimensions]
    )
    x = (active - head["mean"]) / head["scale"]
    if head.get("kind", "linear") == "linear":
        return float(x @ head["coefficients"] + head["intercept"])
    support = np.asarray(head["support_vectors"])
    kernel = np.exp(-head["gamma"] * np.sum((support - x) ** 2, axis=1))
    return float(kernel @ head["dual_coefficients"] + head["intercept"])


def decide(policy: dict, evidence: dict) -> dict:
    x = np.asarray(evidence["features"], dtype=float)
    if x.shape != (len(policy.get("feature_names", FEATURE_NAMES)),) or not np.isfinite(x).all():
        raise ConfigurationError("Invalid latent presence features.")
    raw = evidence["raw_target_scores"]
    if set(raw) != set(TYPES) or not all(math.isfinite(v) for v in raw.values()):
        raise ConfigurationError("Every raw target score is required.")
    heads = {}
    for label in TYPES:
        h = policy["heads"][label]
        score = score_head(h, x)
        support = (score - h["competition_center"]) / h["competition_scale"]
        heads[label] = {
            "raw_score": score,
            "normalized_raw_support": support,
            "threshold": h["threshold"],
        }
    order = sorted(TYPES, key=lambda label: -heads[label]["normalized_raw_support"])
    best = order[0]
    ambiguous = (
        heads[best]["normalized_raw_support"] - heads[order[1]]["normalized_raw_support"] < 0.15
    )
    for label in TYPES:
        h = policy["heads"][label]
        row = heads[label]
        reasons = []
        if label != best or ambiguous:
            reasons.append("raw_audio_heads_disagree_or_support_another_type")
        if not h["enabled"]:
            reasons.append("type_failed_independent_gate")
        if raw[label] <= 0:
            reasons.append("negative_independent_text_similarity")
        if not math.isfinite(evidence["rms"]) or evidence["rms"] < 1e-5:
            reasons.append("insufficient_audio_energy")
        if h["threshold"] is None or row["raw_score"] < h["threshold"]:
            reasons.append("below_frozen_threshold")
        row["rejection_reasons"] = reasons
    label = best if not heads[best]["rejection_reasons"] else "unknown"
    return {
        "label": label,
        "confidence": None,
        "status": "provisional_model_type" if label != "unknown" else "abstained",
        "reason": "passed_raw_audio_head_consensus"
        if label != "unknown"
        else "raw_presence_checks_failed",
        "candidates": heads,
        "scores_are_probabilities": False,
        "human_confirmed": False,
    }
