"""Optional CPU model backends. Analysis accepts prepared local models only."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from dotears.audio import read_wav
from dotears.errors import ConfigurationError, DependencyError, MediaError


def transcribe(wav: Path, model_path: Path, language: str | None, threads: int) -> dict[str, Any]:
    if not model_path.is_dir():
        raise ConfigurationError("ASR requires an existing local CTranslate2 model directory.")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise DependencyError("Transcription requires: pip install 'dotears[asr]'") from exc
    try:
        model = WhisperModel(
            str(model_path),
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            local_files_only=True,
        )
        segments, info = model.transcribe(
            str(wav),
            language=language,
            beam_size=3,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        result = [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
                "avg_logprob": float(s.avg_logprob),
                "no_speech_prob": float(s.no_speech_prob),
            }
            for s in segments
        ]
        return {
            "status": "completed",
            "language": info.language,
            "language_probability": float(info.language_probability),
            "segments": result,
            "method": "faster_whisper_cpu_int8",
            "model_path": str(model_path),
            "caveat": "ASR text and speech boundaries may be wrong; verify against the audio.",
        }
    except Exception as exc:
        raise MediaError(f"Local transcription failed: {exc}") from exc


def score_intervals(
    scores: np.ndarray, labels: list[str], duration: float, threshold: float
) -> list[dict]:
    """Merge overlapping YAMNet patches by class, preserving its original vocabulary."""
    events = []
    if scores.ndim != 2 or scores.shape[1] != len(labels) or not np.isfinite(scores).all():
        raise MediaError("YAMNet returned invalid class scores.")
    for column, label in enumerate(labels):
        if label == "Silence":
            continue
        active = np.flatnonzero(scores[:, column] >= threshold)
        groups: list[list[int]] = []
        for frame in active.tolist():
            if groups and frame == groups[-1][-1] + 1:
                groups[-1].append(frame)
            else:
                groups.append([frame])
        for group in groups:
            start, end = group[0] * 0.48, min(duration, group[-1] * 0.48 + 0.975)
            if start >= duration:
                continue
            category = (
                "speech" if label == "Speech" else "music" if label == "Music" else "sound_event"
            )
            events.append(
                {
                    "start": start,
                    "end": end,
                    "category": category,
                    "label": label,
                    "confidence": None,
                    "score": float(np.max(scores[group, column])),
                    "score_kind": "uncalibrated_yamnet_sigmoid_not_probability",
                    "method": "yamnet_audioset",
                    "clip_path": None,
                    "mixture_status": "mixed_track_not_source_separated",
                    "overlap": "unknown",
                    "temporal_resolution_seconds": 0.48,
                }
            )
    return sorted(events, key=lambda e: (e["start"], -e["score"]))


def classify(wav: Path, model_path: Path, threshold: float, threads: int) -> dict[str, Any]:
    if not (model_path / "saved_model.pb").is_file():
        raise ConfigurationError(
            "YAMNet requires a local official TensorFlow SavedModel directory."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise DependencyError("YAMNet requires: pip install 'dotears[yamnet]'") from exc
    try:
        tf.config.set_visible_devices([], "GPU")
        # TensorFlow may already be initialized by the embedding application.
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass
        import librosa

        waveform, rate = read_wav(wav)
        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=16000)
        duration = len(waveform) / 16000
        model = tf.saved_model.load(str(model_path))
        class_map = Path(model.class_map_path().numpy().decode("utf-8"))
        with class_map.open(encoding="utf-8", newline="") as handle:
            labels = [row["display_name"] for row in csv.DictReader(handle)]
        if len(labels) != 521:
            raise ValueError("Expected 521 AudioSet classes")
        # Bound intermediate tensors even for the longest allowed clip.
        chunks = []
        for begin in range(0, max(1, math.ceil(duration / 9.6))):
            first = begin * 153600  # 20 patch hops: preserve the global 0.48 s grid.
            part = waveform[first : first + 153600 + 15600]
            if len(part) < 15600:
                part = np.pad(part, (0, 15600 - len(part)))
            scores, _, _ = model(tf.constant(part, dtype=tf.float32))
            chunks.append(scores.numpy()[:20])
        events = score_intervals(np.concatenate(chunks), labels, duration, threshold)
        return {
            "status": "completed",
            "events": events,
            "model_path": str(model_path),
            "threshold": threshold,
            "method": "yamnet_audioset",
            "caveat": "AudioSet labels are broad; motion SFX semantics need review.",
        }
    except Exception as exc:
        raise MediaError(f"Local YAMNet classification failed: {exc}") from exc
