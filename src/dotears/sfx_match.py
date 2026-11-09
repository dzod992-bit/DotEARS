"""Conservative waveform identity matching against a locally reviewed SFX library."""

from __future__ import annotations

import json
import math
import re
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt

from dotears.audiosep import file_digest
from dotears.errors import ConfigurationError, LimitError, MediaError
from dotears.sfx import local_artifact, read_window
from dotears.sfx_taxonomy import UI_DEFINITIONS, UI_TAXONOMY, UI_TYPES
from dotears.storage import temporary_directory

RATE = 32000
SCHEMA = "dotears.reference-match-policy/1"
BROAD_SCHEMA = "dotears.reference-match-policy/2"
VERSION = "highpass-waveform-ncc-v1"
LABELS = ("whoosh", "notification", "ui_click", "other")


def filtered(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim != 1 or len(values) < 32 or not np.isfinite(values).all():
        raise ConfigurationError("Invalid reference-match audio.")
    return sosfiltfilt(butter(3, 300, fs=RATE, btype="highpass", output="sos"), values)


def make_template(audio: np.ndarray, peak: float) -> tuple[np.ndarray, int]:
    values = filtered(audio)
    energy = np.cumsum(values**2)
    if energy[-1] < 1e-12:
        raise ConfigurationError("Silent reference template.")
    first = max(0, int(np.searchsorted(energy, 0.005 * energy[-1])) - 64)
    last = min(len(values), int(np.searchsorted(energy, 0.995 * energy[-1])) + 65)
    result = values[first:last].copy()
    result -= result.mean()
    offset = round(peak * RATE) - first
    if not 64 <= len(result) <= 80000 or not 0 <= offset < len(result):
        raise ConfigurationError("Reference exceeds the bounded waveform-match scope.")
    return result.astype(np.float32), offset


class ReferenceMatcher:
    def __init__(self, policy_path: Path):
        if policy_path.stat().st_size > 512000:
            raise ConfigurationError("Reference policy exceeds its size limit.")
        self.policy = json.loads(policy_path.read_text("utf-8"))
        policy = self.policy
        if (
            policy.get("schema_version") not in (SCHEMA, BROAD_SCHEMA)
            or policy.get("version") != VERSION
        ):
            raise ConfigurationError("Unsupported reference-match policy.")
        self.broad_ui = policy["schema_version"] == BROAD_SCHEMA
        if self.broad_ui and policy.get("label_taxonomy") != UI_TAXONOMY:
            raise ConfigurationError("The broad UI policy needs its explicit label taxonomy.")
        self.labels = (*UI_TYPES, "other") if self.broad_ui else LABELS
        entries = policy.get("templates", [])
        if not 1 <= len(entries) <= 100 or len({e["id"] for e in entries}) != len(entries):
            raise ConfigurationError("Invalid reference template catalog.")
        bank = local_artifact(policy_path.resolve().parent, policy["bank_path"])
        if bank.stat().st_size > 24 * 1024**2 or file_digest(bank) != policy["bank_sha256"]:
            raise ConfigurationError("Reference bank changed or exceeds its limit.")
        with zipfile.ZipFile(bank) as archive:
            if (
                len(archive.infolist()) > 100
                or sum(e.file_size for e in archive.infolist()) > 21 * 1024**2
            ):
                raise ConfigurationError("Expanded reference bank exceeds its limit.")
        self.templates = []
        with np.load(bank, allow_pickle=False) as arrays:
            if sum(arrays[e["id"]].nbytes for e in entries) > 20 * 1024**2:
                raise ConfigurationError("Decoded reference bank exceeds its limit.")
            for entry in entries:
                values = np.asarray(arrays[entry["id"]], dtype=np.float64)
                if (
                    not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", entry["id"])
                    or entry["label"] not in self.labels
                    or values.ndim != 1
                    or not 64 <= len(values) <= 80000
                    or not np.isfinite(values).all()
                    or not 0 <= entry["peak_offset"] < len(values)
                    or not isinstance(entry["enabled"], bool)
                    or not 0.65 <= entry["threshold"] <= 1.01
                ):
                    raise ConfigurationError("Invalid reference-match template.")
                values = values - values.mean()
                norm = float(np.linalg.norm(values))
                if norm < 1e-9:
                    raise ConfigurationError("Reference template has no usable energy.")
                self.templates.append((entry, values / norm))

    def scores(self, audio: np.ndarray, peak: float) -> list[dict]:
        if not math.isfinite(peak) or not 0 <= peak <= len(audio) / RATE or len(audio) > 4 * RATE:
            raise ConfigurationError("Invalid reference-match context or peak.")
        values = filtered(audio)
        results = []
        for entry, template in self.templates:
            desired = round(peak * RATE) - entry["peak_offset"]
            start = max(0, desired - 3840)
            end = min(len(values) - len(template), desired + 3840)
            score, found = 0.0, None
            if end >= start:
                local = values[start : end + len(template)]
                numerator = correlate(local, template, mode="valid", method="fft")
                prefix = np.r_[0.0, np.cumsum(local)]
                power = np.r_[0.0, np.cumsum(local**2)]
                n = len(template)
                sums = prefix[n:] - prefix[:-n]
                variance = np.maximum(power[n:] - power[:-n] - sums**2 / n, 1e-20)
                scores = np.clip(np.abs(numerator) / np.sqrt(variance), 0, 1)
                energies = np.sqrt(variance / n)
                scores[energies < 1e-5] = 0
                index = int(np.argmax(scores))
                score = float(scores[index])
                found = (start + index + entry["peak_offset"]) / RATE
            results.append(
                {
                    "reference_id": entry["id"],
                    "label": entry["label"],
                    "score": score,
                    "matched_peak_relative": found,
                    "enabled": entry["enabled"],
                    "threshold": entry["threshold"],
                }
            )
        return results

    @staticmethod
    def decide(scores: list[dict]) -> dict:
        ranked = sorted(scores, key=lambda r: -r["score"])
        best = ranked[0]
        second = max((r["score"] for r in ranked if r["label"] != best["label"]), default=0)
        reasons = []
        if best["label"] == "other":
            reasons.append("reviewed_negative_reference_wins")
        if not best["enabled"]:
            reasons.append("reference_failed_independent_mixture_gate")
        if best["score"] < best["threshold"]:
            reasons.append("below_frozen_identity_threshold")
        if best["score"] - second < 0.05:
            reasons.append("ambiguous_reference_type")
        label = best["label"] if not reasons else "unknown"
        return {
            "label": label,
            "confidence": None,
            "status": "known_reference_pattern" if label != "unknown" else "abstained",
            "scope": "known_waveform_only_not_unseen_effect_classification",
            "best_reference_id": best["reference_id"],
            "reference_score": best["score"],
            "matched_peak_relative": best["matched_peak_relative"],
            "rejection_reasons": reasons,
            "scores_are_probabilities": False,
            "human_confirmed_in_video": False,
            "candidates": ranked,
        }

    def analyze(self, audio: np.ndarray, peak: float) -> dict:
        return self.decide(self.scores(audio, peak))


def scan_reference_report(report_path: Path, policy_path: Path, output: Path) -> dict:
    import librosa

    output = output.resolve()
    if output.exists():
        raise ConfigurationError("Reference-match output exists; choose a new directory.")
    report_path = report_path.resolve()
    original = file_digest(report_path)
    policy_hash = file_digest(policy_path)
    matcher = ReferenceMatcher(policy_path)
    report = json.loads(report_path.read_text("utf-8"))
    native = local_artifact(report_path.parent, report["audio"]["native_path"])
    native_hash = file_digest(native)
    events = [e for e in report["events"] if e["category"] == "audio_attack"]
    if (
        not 1 <= len(events) <= 200
        or len({e["id"] for e in events}) != len(events)
        or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", e["id"]) for e in events)
    ):
        raise LimitError("Reference matching needs 1–200 safely named, unique attacks.")
    offset, duration = [report["audio"][k] for k in ("timeline_offset_seconds", "duration_seconds")]
    if not all(math.isfinite(v) for v in (offset, duration)) or duration < 0.128:
        raise ConfigurationError("Invalid reference-match timeline.")
    started = time.perf_counter()
    results = []
    for event in events:
        peak = event["peak_time"] - offset
        if not math.isfinite(peak) or not 0 <= peak <= duration:
            raise ConfigurationError("Reference-match peak is outside the audio.")
        first = max(0.0, min(peak - 1.0, duration - 3.0))
        audio, rate = read_window(native, first, min(duration, first + 3.0))
        first = round(first * rate) / rate
        audio = librosa.resample(audio, orig_sr=rate, target_sr=RATE).astype(np.float32)
        decision = matcher.analyze(audio, min(peak - first, len(audio) / RATE))
        results.append(
            {
                "event_id": event["id"],
                "peak_time": event["peak_time"],
                "context_start": first + offset,
                **decision,
            }
        )
    if (
        file_digest(report_path) != original
        or file_digest(native) != native_hash
        or file_digest(policy_path) != policy_hash
    ):
        raise MediaError("Original input or reference policy changed during matching.")
    result = {
        "schema_version": "dotears.reference-match-report/2"
        if matcher.broad_ui
        else "dotears.reference-match-report/1",
        "label_taxonomy": UI_TAXONOMY if matcher.broad_ui else "dotears.sfx-types/1-narrow-click",
        "label_definitions": UI_DEFINITIONS if matcher.broad_ui else None,
        "version": VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "analysis_id": report["analysis_id"],
        "source_sha256": report["source"]["sha256"],
        "source_report_path": str(report_path),
        "source_report_sha256": original,
        "native_audio_sha256": native_hash,
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": policy_hash,
        "bank_sha256": matcher.policy["bank_sha256"],
        "video_annotations_used_in_inference": False,
        "human_reviewed_library_labels": True,
        "scope": "known_waveform_only_not_unseen_effect_classification",
        "events": results,
        "seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "Matching validates known waveforms, not arbitrary new SFX.",
            "Pitch/time changes, codecs, masking, or missing attacks may prevent a match.",
            "Unknown is not evidence that an effect is absent.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary_directory(prefix="dotears-match-", dir=output.parent) as name:
        stage = Path(name)
        (stage / "sfx-match.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), "utf-8"
        )
        if output.exists():
            raise ConfigurationError("Reference-match output appeared concurrently.")
        stage.rename(output)
    return {
        "sfx_report_path": str(output / "sfx-match.json"),
        "events": len(results),
        "accepted": sum(e["label"] != "unknown" for e in results),
        "seconds": result["seconds"],
    }
