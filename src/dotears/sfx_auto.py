"""Three fixed local separator queries, evidence features and conservative decisions.

The separator's conditioning string is never a label. A separately frozen, locally
evaluated policy may accept a provisional type; otherwise the result is unknown.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dotears.audiosep import PINNED_FILES, file_digest
from dotears.core import model_fingerprint
from dotears.errors import ConfigurationError, DependencyError, LimitError, MediaError
from dotears.sfx import CONTROL_LABELS, PROMPTS, local_artifact, read_window
from dotears.storage import temporary_directory

QUERIES = {
    "whoosh": PROMPTS["whoosh"][2],
    "notification": PROMPTS["notification"][0],
    "ui_click": PROMPTS["ui_click"][0],
}
FEATURE_VERSION = "raw-and-separated-focus-v1"
FEATURE_NAMES = (
    [f"raw:{label}" for label in PROMPTS]
    + [f"estimate:{label}" for label in PROMPTS]
    + [f"delta:{label}" for label in PROMPTS]
    + ["log10_raw_rms", "log10_estimate_rms", "energy_ratio_db", "embedding_cosine"]
)
POLICY_SCHEMA = "dotears.sfx-auto-policy/1"


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pcm_hash(values: np.ndarray, rate: int = 32000) -> str:
    values = np.asarray(values, dtype="<f4")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ConfigurationError("Audio cache requires finite mono PCM.")
    digest = hashlib.sha256(f"float32-le/mono/{rate}/{len(values)}:".encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def focus_window(values: np.ndarray, peak: float, rate: int = 32000) -> np.ndarray:
    if not math.isfinite(peak) or not 0 <= peak <= len(values) / rate:
        raise ConfigurationError("Candidate peak falls outside its audio context.")
    first = max(0, round((peak - 0.15) * rate))
    last = min(len(values), round((peak + 0.55) * rate))
    if last <= first:
        raise ConfigurationError("Empty candidate focus window.")
    return values[first:last]


def read_policy(path: Path) -> dict:
    policy = json.loads(path.read_text("utf-8"))
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or policy.get("queries") != QUERIES
        or policy.get("feature_names") != FEATURE_NAMES
        or policy.get("feature_version") != FEATURE_VERSION
        or policy.get("prompt_sha256") != json_hash(PROMPTS)
        or set(policy.get("heads", {})) != set(QUERIES)
    ):
        raise ConfigurationError("Unsupported or incompatible automatic SFX policy.")
    for head in policy["heads"].values():
        mean, scale, coef = [np.asarray(head[k]) for k in ("mean", "scale", "coefficients")]
        numbers = np.r_[mean, scale, coef, head["intercept"], head["score_scale"]]
        if (
            any(v.shape != (len(FEATURE_NAMES),) for v in (mean, scale, coef))
            or not np.isfinite(numbers).all()
            or np.any(scale <= 0)
            or head["score_scale"] <= 0
            or not isinstance(head["enabled"], bool)
            or (head["threshold"] is not None and not math.isfinite(head["threshold"]))
            or (head["enabled"] and head["threshold"] is None)
        ):
            raise ConfigurationError("Invalid automatic SFX policy parameters.")
    return policy


def decision(policy: dict, candidates: dict) -> dict:
    """No probabilities, no forced winner, no acceptance of a disabled class."""
    if set(candidates) != set(QUERIES):
        raise ConfigurationError("Automatic SFX decisions require all three query results.")
    evidence, passed = {}, []
    for label in QUERIES:
        row, head = candidates[label], policy["heads"][label]
        features = np.asarray(row["features"], dtype=np.float64)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ConfigurationError("Invalid automatic SFX evidence features.")
        measurements = [
            row[key]
            for key in (
                "raw_rms",
                "estimate_rms",
                "energy_ratio_db",
                "estimated_target_cosine",
                "estimated_best_control_cosine",
            )
        ]
        if not all(math.isfinite(v) for v in measurements) or min(measurements[:2]) < 0:
            raise ConfigurationError("Invalid automatic SFX energy or similarity measurements.")
        score = float(
            ((features - head["mean"]) / head["scale"]) @ head["coefficients"] + head["intercept"]
        )
        rejection = []
        if not head["enabled"]:
            rejection.append("class_failed_calibration_or_validation_gate")
        if row["raw_rms"] < 1e-5 or row["estimate_rms"] < 1e-5:
            rejection.append("insufficient_audio_energy")
        if row["energy_ratio_db"] < -40:
            rejection.append("estimated_component_too_weak")
        if row["estimated_target_cosine"] < row["estimated_best_control_cosine"]:
            rejection.append("estimated_background_match_dominates")
        threshold = head["threshold"]
        if threshold is None or score < threshold:
            rejection.append("below_frozen_rejection_threshold")
        support = None if threshold is None else (score - threshold) / head["score_scale"]
        evidence[label] = {
            "raw_score": score,
            "threshold": threshold,
            "normalized_excess": support,
            "accepted_candidate": not rejection,
            "rejection_reasons": rejection,
        }
        if not rejection:
            passed.append((support, label))
    passed.sort(reverse=True)
    ambiguous = len(passed) > 1 and passed[0][0] - passed[1][0] < 0.5
    label = passed[0][1] if passed and not ambiguous else "unknown"
    return {
        "label": label,
        "confidence": None,
        "status": "provisional_model_type" if label != "unknown" else "abstained",
        "reason": (
            "multiple_candidates_have_similar_support"
            if ambiguous
            else "no_candidate_passed_all_checks"
            if not passed
            else "passed_local_policy_checks"
        ),
        "candidates": evidence,
        "scores_are_probabilities": False,
        "human_confirmed": False,
    }


class CandidateEngine:
    """The inference API accepts audio and a peak, never a known sound label."""

    def __init__(self, audiosep: Path, clap: Path, cache: Path, threads: int = 4):
        os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise DependencyError(
                "Automatic SFX needs the isolated local AudioSep/CLAP environment."
            ) from exc

        from dotears.audiosep import LocalAudioSep

        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
            raise ConfigurationError("Automatic SFX CPU threads must be between 1 and 16.")
        if not (clap / "pytorch_model.bin").is_file():
            raise ConfigurationError("Prepare the official local CLAP model before automatic SFX.")
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.torch, self.cache = torch, cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.clap_sha = model_fingerprint(str(clap.resolve()))
        self.model = (
            ClapModel.from_pretrained(
                str(clap.resolve()), local_files_only=True, use_safetensors=False, weights_only=True
            )
            .to("cpu")
            .eval()
        )
        self.processor = ClapProcessor.from_pretrained(str(clap.resolve()), local_files_only=True)
        self.separator = LocalAudioSep(audiosep, threads=threads)
        self.identity = {
            "audiosep_checkpoint_sha256": PINNED_FILES["audiosep_base_4M_steps.ckpt"],
            "clap_fingerprint": self.clap_sha,
            "prompt_sha256": json_hash(PROMPTS),
        }
        texts = [text for values in PROMPTS.values() for text in values]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.inference_mode():
            encoded = self.model.get_text_features(**inputs).reshape(len(PROMPTS), 3, -1)
            self.text_features = torch.nn.functional.normalize(encoded.mean(dim=1), dim=-1)
        self.counters = {"separations": 0, "separation_cache_hits": 0, "embeddings": 0}

    def embedding(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import librosa

        audio = librosa.resample(waveform, orig_sr=32000, target_sr=48000).astype(np.float32)
        key = json_hash([pcm_hash(audio, 48000), self.clap_sha, "repeatpad", FEATURE_VERSION])
        path = self.cache / ("embedding-" + key + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                feature = cached["embedding"]
            if feature.shape != (512,) or not np.isfinite(feature).all():
                raise MediaError("Invalid cached CLAP embedding.")
        else:
            inputs = self.processor(
                audio=audio, sampling_rate=48000, return_tensors="pt", padding="repeatpad"
            )
            with self.torch.inference_mode():
                feature = self.model.get_audio_features(**inputs)
                feature = self.torch.nn.functional.normalize(feature, dim=-1)[0].cpu().numpy()
            np.savez(path, embedding=feature)
            self.counters["embeddings"] += 1
        scores = feature @ self.text_features.cpu().numpy().T
        return feature, scores

    def separation_key(self, waveform: np.ndarray, prompt: str) -> str:
        return json_hash([pcm_hash(waveform), prompt, self.identity["audiosep_checkpoint_sha256"]])

    def seed_verified_estimate(self, waveform: np.ndarray, prompt: str, output: Path) -> None:
        """Caller must verify frozen experiment provenance before importing old estimates."""
        import soundfile as sf

        values, rate = sf.read(output, dtype="float32")
        if rate != 32000 or values.shape != waveform.shape or not np.isfinite(values).all():
            raise MediaError("Incompatible cached separator estimate.")
        path = self.cache / ("separation-" + self.separation_key(waveform, prompt) + ".wav")
        if not path.exists():
            sf.write(path, values, 32000, subtype="FLOAT")
            (path.with_suffix(".json")).write_text(
                json.dumps({"sha256": file_digest(path), "imported_from": str(output)}), "utf-8"
            )

    def separate(self, waveform: np.ndarray, prompt: str) -> tuple[np.ndarray, Path]:
        import soundfile as sf

        path = self.cache / ("separation-" + self.separation_key(waveform, prompt) + ".wav")
        metadata = path.with_suffix(".json")
        if path.exists() and metadata.exists():
            if file_digest(path) != json.loads(metadata.read_text("utf-8"))["sha256"]:
                raise MediaError("A cached separator estimate changed.")
            values, rate = sf.read(path, dtype="float32")
            if rate != 32000 or values.shape != waveform.shape or not np.isfinite(values).all():
                raise MediaError("Invalid cached separator output.")
            self.counters["separation_cache_hits"] += 1
        else:
            values = self.separator.separate(waveform, prompt)
            sf.write(path, values, 32000, subtype="FLOAT")
            metadata.write_text(json.dumps({"sha256": file_digest(path)}), "utf-8")
            self.counters["separations"] += 1
        return values, path

    def analyze_window(self, waveform: np.ndarray, peak: float) -> dict:
        from dotears.separation import output_energy_db

        waveform = np.asarray(waveform, dtype=np.float32)
        if (
            waveform.ndim != 1
            or not 4096 <= len(waveform) <= 96000
            or not np.isfinite(waveform).all()
        ):
            raise ConfigurationError("Automatic SFX context must be finite mono, 0.128–3 seconds.")
        raw = focus_window(waveform, peak)
        raw_feature, raw_scores = self.embedding(raw)
        raw_rms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))
        results = {}
        labels = list(PROMPTS)
        controls = [labels.index(label) for label in CONTROL_LABELS]
        for label, prompt in QUERIES.items():
            estimate, path = self.separate(waveform, prompt)
            focus = focus_window(estimate, peak)
            feature, scores = self.embedding(focus)
            rms = float(np.sqrt(np.mean(focus.astype(np.float64) ** 2)))
            ratio = output_energy_db(raw, focus)
            # Silence is explicitly vetoed; finite placeholders only feed the linear head.
            ratio = -300.0 if ratio is None else ratio
            values = np.r_[
                raw_scores,
                scores,
                scores - raw_scores,
                np.log10(max(raw_rms, 1e-12)),
                np.log10(max(rms, 1e-12)),
                np.clip(ratio, -120, 120),
                float(raw_feature @ feature),
            ]
            results[label] = {
                "features": values.tolist(),
                "raw_rms": raw_rms,
                "estimate_rms": rms,
                "energy_ratio_db": ratio,
                "estimated_target_cosine": float(scores[labels.index(label)]),
                "estimated_best_control_cosine": float(max(scores[controls])),
                "estimate_path": str(path),
                "estimate_sha256": file_digest(path),
                "query": prompt,
            }
        return results


def scan_report(
    report_path: Path,
    audiosep: Path,
    clap: Path,
    policy_path: Path,
    output: Path,
    cache: Path,
    threads: int = 4,
    timeout: float = 1800,
) -> dict:
    """Automatically check every existing attack; preserve the original report unchanged."""
    output, report_path = output.resolve(), report_path.resolve()
    if output.exists():
        raise ConfigurationError("Automatic SFX output exists; choose a new directory.")
    if cache.resolve().is_relative_to(output):
        raise ConfigurationError("Choose a model cache outside the new scan output directory.")
    if not math.isfinite(timeout) or not 30 <= timeout <= 7200:
        raise ConfigurationError("Automatic SFX time limit must be 30–7200 seconds.")
    policy = read_policy(policy_path)
    policy_sha = file_digest(policy_path)
    report_sha = file_digest(report_path)
    report = json.loads(report_path.read_text("utf-8"))
    attacks = [row for row in report["events"] if row["category"] == "audio_attack"]
    if (
        not 1 <= len(attacks) <= 200
        or len({row["id"] for row in attacks}) != len(attacks)
        or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", row["id"]) for row in attacks)
    ):
        raise LimitError("Automatic SFX requires 1–200 unique, safely named attack events.")
    native = local_artifact(report_path.parent, report["audio"]["native_path"])
    offset, duration = [
        report["audio"][key] for key in ("timeline_offset_seconds", "duration_seconds")
    ]
    if not all(math.isfinite(v) for v in (offset, duration)) or duration < 0.128:
        raise ConfigurationError("Invalid report audio timeline.")
    for event in attacks:
        if (
            not math.isfinite(event["peak_time"])
            or not 0 <= event["peak_time"] - offset <= duration
        ):
            raise ConfigurationError("Attack peak falls outside the native report audio.")
    import librosa
    import soundfile as sf

    started = time.perf_counter()
    engine = CandidateEngine(audiosep, clap, cache, threads)
    if policy["models"] != engine.identity:
        raise ConfigurationError(
            "Automatic SFX policy was evaluated with different model artifacts."
        )
    native_sha = file_digest(native)
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary_directory(prefix="dotears-auto-", dir=output.parent) as name:
        stage, events = Path(name), []
        (stage / "audio").mkdir()
        for event in attacks:
            if time.perf_counter() - started > timeout:
                raise LimitError(
                    "Automatic SFX time budget reached; verified model cache is retained."
                )
            peak = event["peak_time"] - offset
            first = max(0.0, min(peak - 1.0, duration - 3.0))
            last = min(duration, first + 3.0)
            waveform, rate = read_window(native, first, last)
            first = round(first * rate) / rate
            waveform = librosa.resample(waveform, orig_sr=rate, target_sr=32000).astype(np.float32)
            evidence = engine.analyze_window(waveform, min(peak - first, len(waveform) / 32000))
            result = decision(policy, evidence)
            relative = f"audio/{event['id']}-mixture.wav"
            sf.write(stage / relative, focus_window(waveform, peak - first), 32000, subtype="FLOAT")
            for label in QUERIES:
                estimate, _ = sf.read(evidence[label]["estimate_path"], dtype="float32")
                dest = f"audio/{event['id']}-{label}.wav"
                sf.write(stage / dest, focus_window(estimate, peak - first), 32000, subtype="FLOAT")
                evidence[label]["focus_audio_path"] = dest
                evidence[label]["focus_audio_sha256"] = file_digest(stage / dest)
            events.append(
                {
                    "event_id": event["id"],
                    "peak_time": event["peak_time"],
                    "context_start": first + offset,
                    "context_end": first + len(waveform) / 32000 + offset,
                    "focus_start": first + max(0, peak - first - 0.15) + offset,
                    "focus_end": first + min(len(waveform) / 32000, peak - first + 0.55) + offset,
                    "mixture_focus_path": relative,
                    "mixture_focus_sha256": file_digest(stage / relative),
                    **result,
                    "feature_evidence": evidence,
                }
            )
            print(
                f"SFX {len(events)}/{len(attacks)}: {event['id']} = {result['label']}",
                flush=True,
                file=sys.stderr,
            )
        if file_digest(report_path) != report_sha or file_digest(native) != native_sha:
            raise MediaError("Source report or native audio changed during the automatic scan.")
        if file_digest(policy_path) != policy_sha:
            raise ConfigurationError("The frozen decision policy changed during the scan.")
        result = {
            "schema_version": "dotears.sfx-auto-report/1",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "experimental_local_automatic_check",
            "analysis_id": report["analysis_id"],
            "source_sha256": report["source"]["sha256"],
            "source_report_sha256": report_sha,
            "source_report_path": str(report_path),
            "native_audio_sha256": native_sha,
            "policy_path": str(policy_path.resolve()),
            "policy_sha256": policy_sha,
            "models": engine.identity,
            "queries": QUERIES,
            "human_annotations_used_in_inference": False,
            "score_kind": "uncalibrated_linear_evidence_not_probability",
            "events": events,
            "counters": engine.counters,
            "seconds": round(time.perf_counter() - started, 3),
            "limitations": [
                "Accepted types are provisional and calibrated on a small convenience set.",
                "Unknown does not mean no effect. Attack windows may contain multiple sounds.",
                "Only existing DSP attacks are checked; this is not continuous event recall.",
                "A UI tone is not necessarily a UI click. Estimates may contain music or speech.",
            ],
        }
        (stage / "sfx-auto.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), "utf-8"
        )
        if output.exists():
            raise ConfigurationError("Automatic SFX output was created concurrently.")
        stage.rename(output)
    return {
        "sfx_report_path": str(output / "sfx-auto.json"),
        "events": len(events),
        "accepted": sum(row["label"] != "unknown" for row in events),
        "seconds": result["seconds"],
    }
