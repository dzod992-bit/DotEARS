"""Explicit, query-conditioned extraction of one existing attack context."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dotears.core import digest_file
from dotears.errors import ConfigurationError, DependencyError, MediaError
from dotears.separation import output_energy_db
from dotears.sfx import local_artifact, read_window
from dotears.storage import temporary_directory


def separate_report_window(
    report_path: Path,
    event_id: str,
    prompt: str,
    model_path: Path,
    output: Path,
    threads: int = 4,
) -> dict:
    """Keep the raw window and an unconfirmed estimate; never rewrite the analysis."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1000:
        raise ConfigurationError("Provide a non-empty target description, at most 1000 characters.")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
        raise ConfigurationError("CPU threads must be an integer between 1 and 16.")
    output, report_path = output.resolve(), report_path.resolve()
    if output.exists():
        raise ConfigurationError("Separation output exists; choose a new directory.")
    report = json.loads(report_path.read_text("utf-8"))
    event = next(
        (e for e in report["events"] if e["id"] == event_id and e["category"] == "audio_attack"),
        None,
    )
    if event is None:
        raise ConfigurationError("Choose an existing audio_attack event ID from this report.")
    native = local_artifact(report_path.parent, report["audio"]["native_path"])
    offset = report["audio"]["timeline_offset_seconds"]
    duration = report["audio"]["duration_seconds"]
    peak = event["peak_time"] - offset
    if not all(math.isfinite(v) for v in (duration, peak, offset)) or not 0 <= peak <= duration:
        raise ConfigurationError("Invalid native audio timeline in the report.")
    first = max(0.0, min(peak - 1.0, duration - 3.0))
    last = min(duration, first + 3.0)
    waveform, rate = read_window(native, first, last)
    actual_first = round(first * rate) / rate
    if len(waveform) / rate < 0.128:
        raise ConfigurationError("AudioSep needs at least 0.128 seconds of native audio.")
    import librosa
    import soundfile as sf

    from dotears.audiosep import LocalAudioSep

    waveform = librosa.resample(waveform, orig_sr=rate, target_sr=32000).astype(np.float32)
    try:
        model = LocalAudioSep(model_path, threads=threads)
        estimate = model.separate(waveform, prompt)
    except (ConfigurationError, DependencyError, MediaError):
        raise
    except ImportError as exc:
        raise DependencyError(
            "Install the optional AudioSep CPU dependencies in its isolated environment."
        ) from exc
    except Exception as exc:
        raise MediaError(f"Local AudioSep extraction failed: {exc}") from exc
    result = {
        "schema_version": "dotears.separated-window/1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "experimental_query_conditioned_estimate",
        "label": "unknown",
        "confidence": None,
        "query_is_not_sound_recognition": True,
        "query": prompt,
        "event_id": event_id,
        "analysis_id": report["analysis_id"],
        "source_sha256": report["source"]["sha256"],
        "source_report_path": str(report_path),
        "source_report_sha256": digest_file(report_path),
        "native_audio_sha256": digest_file(native),
        "window_start": actual_first + offset,
        "window_end": actual_first + len(waveform) / 32000 + offset,
        "sample_rate": 32000,
        "channels": 1,
        "sample_format": "float32",
        "output_amplitude_normalized": False,
        "model": "AudioSep",
        "model_load": model.load_evidence,
        "device": "cpu",
        "threads": threads,
        "mixed_window_path": "mixture.wav",
        "estimated_component_path": "estimated.wav",
        "output_to_input_energy_db": output_energy_db(waveform, estimate),
        "caveats": [
            "The description specifies what to attempt to retain; "
            "it does not confirm a sound was present.",
            "The estimate can retain background, alter or remove the requested sound.",
            "Local evaluation was most promising for notification-like sounds; "
            "whooshes were unreliable.",
            "The original mixed audio and analysis report are preserved.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary_directory(prefix="dotears-separate-", dir=output.parent) as name:
        stage = Path(name)
        sf.write(stage / "mixture.wav", waveform, 32000, subtype="FLOAT")
        sf.write(stage / "estimated.wav", estimate, 32000, subtype="FLOAT")
        result["mixed_window_sha256"] = digest_file(stage / "mixture.wav")
        result["estimated_component_sha256"] = digest_file(stage / "estimated.wav")
        with (stage / "separation.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        if output.exists():
            raise ConfigurationError(
                "Separation output was created concurrently; choose a new directory."
            )
        stage.rename(output)
    result["separation_report_path"] = str(output / "separation.json")
    return result
