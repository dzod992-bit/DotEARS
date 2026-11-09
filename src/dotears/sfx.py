"""Experimental local CLAP hypotheses. These rankings are not confirmed SFX labels."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dotears.core import digest_file, model_fingerprint
from dotears.errors import ConfigurationError, DependencyError, LimitError, MediaError

# Frozen before the first model run. User notes are never used as text prompts.
PROMPTS = {
    "whoosh": [
        "The sound of a short whoosh transition effect.",
        "A swooshing sound of air rushing past.",
        "A fast sweeping swish sound effect.",
    ],
    "notification": [
        "The sound of a notification chime.",
        "A bright electronic message alert sound.",
        "A short bell-like notification sound effect.",
    ],
    "ui_click": [
        "The sound of clicking a user interface button.",
        "A short digital click or tick sound effect.",
        "A soft computer interface tap sound.",
    ],
    "ui_tone": [
        "The sound of an electronic user interface tone.",
        "A short synthetic beep or blip sound effect.",
        "A clean digital confirmation sound.",
    ],
    "pop": [
        "The sound of a short pop sound effect.",
        "A soft bubble popping sound.",
        "A playful plop sound effect.",
    ],
    "impact": [
        "The sound of a deep impact sound effect.",
        "A short heavy thump or hit sound.",
        "A dramatic cinematic boom sound effect.",
    ],
    "riser": [
        "The sound of a rising sweep sound effect.",
        "A rising electronic tone building tension.",
        "A gradually increasing riser transition sound.",
    ],
    "glitch": [
        "The sound of a digital glitch sound effect.",
        "A short distorted electronic stutter sound.",
        "A computer error buzzing and crackling sound.",
    ],
    "music": [
        "The sound of instrumental music playing.",
        "A musical note or drum beat in background music.",
        "A rhythmic musical accent.",
    ],
    "speech": [
        "The sound of a person speaking.",
        "A spoken word in a human voice.",
        "The sound of narration or conversation.",
    ],
    "music_and_speech": [
        "The sound of a person talking over background music.",
        "Narration with instrumental music playing underneath.",
        "A human voice and rhythmic music together.",
    ],
    "noise": [
        "The sound of background noise or hiss.",
        "A quiet recording with no distinct sound event.",
        "The sound of ambient room noise.",
    ],
}
CONTROL_LABELS = {"music", "speech", "music_and_speech", "noise"}


def rank_scores(scores: np.ndarray, labels: list[str]) -> dict:
    scores = np.asarray(scores)
    if scores.shape != (len(labels),) or not np.isfinite(scores).all():
        raise MediaError("CLAP returned invalid similarity scores.")
    if np.max(np.abs(scores)) > 1.00001:
        raise MediaError("CLAP cosine similarity is outside [-1, 1].")
    ranked = sorted(
        [
            {"label": label, "cosine_similarity": round(float(score), 6)}
            for label, score in zip(labels, scores, strict=True)
        ],
        key=lambda item: -item["cosine_similarity"],
    )
    best = ranked[0]
    return {
        "label": "unknown",
        "confidence": None,
        "status": "experimental_hypotheses_only",
        "proposed_label": best["label"] if best["label"] not in CONTROL_LABELS else None,
        "best_match": best,
        "top_sfx_hypotheses": [e for e in ranked if e["label"] not in CONTROL_LABELS][:3],
        "best_control_match": next(e for e in ranked if e["label"] in CONTROL_LABELS),
        "margin_to_second": round(best["cosine_similarity"] - ranked[1]["cosine_similarity"], 6),
        "rankings": ranked,
        "score_kind": "uncalibrated_clap_cosine_similarity_not_probability",
        "abstention_reason": "No validated SFX decision threshold or held-out evaluation yet.",
    }


def local_artifact(folder: Path, relative: str) -> Path:
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ConfigurationError("Missing report artifact or path outside its report folder.")
    return path


def read_window(wav: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= start + 10:
        raise ConfigurationError("CLAP window must be finite, non-empty and at most 10 seconds.")
    with wave.open(str(wav), "rb") as audio:
        rate, channels = audio.getframerate(), audio.getnchannels()
        if channels not in (1, 2) or audio.getsampwidth() != 2 or not 8000 <= rate <= 192000:
            raise MediaError("SFX probe requires native 16-bit mono/stereo PCM WAV.")
        first = min(round(start * rate), audio.getnframes())
        last = min(round(end * rate), audio.getnframes())
        if last <= first:
            raise MediaError("SFX window falls outside the decoded audio.")
        audio.setpos(first)
        samples = np.frombuffer(audio.readframes(last - first), dtype="<i2")
    return samples.reshape(-1, channels).mean(axis=1).astype(np.float32) / 32768, rate


def classify_sfx(
    report_path: Path,
    model_path: Path,
    event_ids: list[str] | None = None,
    window_mode: str = "excerpt",
    context_seconds: float = 3.0,
    padding: str = "repeatpad",
    threads: int = 4,
) -> dict:
    if window_mode not in {"excerpt", "context"} or padding not in {"repeatpad", "pad"}:
        raise ConfigurationError("Unsupported CLAP window or padding mode.")
    if not math.isfinite(context_seconds) or not 0.2 <= context_seconds <= 10:
        raise ConfigurationError("CLAP context must be between 0.2 and 10 seconds.")
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 32:
        raise ConfigurationError("CLAP thread count must be an integer between 1 and 32.")
    report_path, model_path = report_path.resolve(), model_path.resolve()
    if not (model_path / "pytorch_model.bin").is_file():
        raise ConfigurationError("Prepare the official local CLAP model before running the probe.")
    report = json.loads(report_path.read_text("utf-8"))
    attacks = [e for e in report["events"] if e["category"] == "audio_attack"]
    if event_ids is not None:
        requested = set(event_ids)
        if requested - {e["id"] for e in attacks}:
            raise ConfigurationError("Requested event ID is absent from the report's attacks.")
        attacks = [e for e in attacks if e["id"] in requested]
    if not 1 <= len(attacks) <= 200:
        raise LimitError("SFX probe requires between 1 and 200 existing attack events.")
    native = local_artifact(report_path.parent, report["audio"]["native_path"])
    offset = report["audio"]["timeline_offset_seconds"]
    duration = report["audio"]["duration_seconds"]
    windows = []
    for event in attacks:
        peak = event["peak_time"] - offset
        if window_mode == "excerpt":
            start, end = event["clip_start"] - offset, event["clip_end"] - offset
        else:
            start = max(0.0, peak - context_seconds / 2)
            end = min(duration, peak + context_seconds / 2)
        windows.append((event, start, end))
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    try:
        import librosa
        import torch
        from transformers import ClapModel, ClapProcessor
    except ImportError as exc:
        raise DependencyError(
            "SFX probe requires the optional CPU torch/transformers packages."
        ) from exc
    started = time.perf_counter()
    try:
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        model = (
            ClapModel.from_pretrained(
                str(model_path),
                local_files_only=True,
                use_safetensors=False,
                weights_only=True,
            )
            .to("cpu")
            .eval()
        )
        processor = ClapProcessor.from_pretrained(str(model_path), local_files_only=True)
        labels = list(PROMPTS)
        texts = [text for values in PROMPTS.values() for text in values]
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        with torch.inference_mode():
            features = model.get_text_features(**text_inputs)
            means = features.reshape(len(labels), 3, -1).mean(dim=1)
            text_features = torch.nn.functional.normalize(means, dim=-1)
        results = []
        for event, start, end in windows:
            waveform, rate = read_window(native, start, end)
            waveform = librosa.resample(waveform, orig_sr=rate, target_sr=48000)
            inputs = processor(
                audio=waveform,
                sampling_rate=48000,
                return_tensors="pt",
                padding=padding,
            )
            with torch.inference_mode():
                audio_features = model.get_audio_features(**inputs)
                audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
                scores = (audio_features @ text_features.T)[0].cpu().numpy()
            results.append(
                {
                    "event_id": event["id"],
                    "peak_time": event["peak_time"],
                    "window_start": start + offset,
                    "window_end": end + offset,
                    **rank_scores(scores, labels),
                }
            )
    except (ConfigurationError, MediaError):
        raise
    except Exception as exc:
        raise MediaError(f"Local CLAP SFX probe failed: {exc}") from exc
    return {
        "schema_version": "dotears.sfx-hypotheses/1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "experimental_not_validated_for_automatic_labels",
        "analysis_id": report["analysis_id"],
        "source_sha256": report["source"]["sha256"],
        "source_report_sha256": digest_file(report_path),
        "source_report_path": str(report_path),
        "model_path": str(model_path),
        "model_sha256": model_fingerprint(str(model_path)),
        "prompt_version": "motion-sfx-v1",
        "prompt_sha256": hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest(),
        "prompts": PROMPTS,
        "method": "clap_mean_of_three_text_embeddings_cosine_similarity",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "device": "cpu",
        "threads": threads,
        "window_mode": window_mode,
        "context_seconds": context_seconds,
        "padding": padding,
        "events": results,
        "inference_seconds": round(time.perf_counter() - started, 3),
        "caveats": [
            "Rankings depend on the candidate descriptions and the context window.",
            "Mixed audio may hide effects behind speech or music.",
            "Notification, UI tone and musical chime can be acoustically ambiguous.",
            "No score is a calibrated probability; original attack labels remain unknown.",
        ],
    }
