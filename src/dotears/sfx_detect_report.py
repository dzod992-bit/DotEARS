"""Optional raw SFX presence sidecar; no query-conditioned estimate becomes a label."""

from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dotears.audiosep import file_digest
from dotears.errors import ConfigurationError, LimitError, MediaError
from dotears.sfx import PROMPTS, local_artifact, read_window
from dotears.sfx_presence import (
    CED_TAGS,
    FEATURE_NAMES,
    RawSFXEngine,
    decide,
    read_policy,
    window,
)
from dotears.sound_classifier import decide as decide_expanded_sound
from dotears.sound_classifier import read_policy as read_expansion_policy
from dotears.storage import temporary_directory


def scan_raw_report(
    report_path: Path,
    ced: Path,
    clap: Path,
    policy_path: Path,
    output: Path,
    cache: Path,
    threads: int = 4,
    timeout: float = 1800,
    expansion_path: Path | None = None,
) -> dict:
    output, report_path = output.resolve(), report_path.resolve()
    if output.exists():
        raise ConfigurationError("SFX output exists; choose a new directory.")
    if cache.resolve().is_relative_to(output):
        raise ConfigurationError("Use a model cache outside the new output directory.")
    if not math.isfinite(timeout) or not 30 <= timeout <= 7200:
        raise ConfigurationError("SFX time limit must be 30–7200 seconds.")
    schema = json.loads(policy_path.read_text("utf-8")).get("schema_version")
    if schema == "dotears.sfx-presence-policy/5":
        from dotears.sfx_contrast import ContrastSFXEngine
        from dotears.sfx_hierarchy import decide as hierarchy_decide
        from dotears.sfx_hierarchy import read_policy as read_hierarchy_policy

        policy = read_hierarchy_policy(policy_path)
        engine_class, decide_function = ContrastSFXEngine, hierarchy_decide
    elif schema == "dotears.sfx-presence-policy/4":
        from dotears.sfx_contrast import ContrastSFXEngine
        from dotears.sfx_contrast import decide as contrast_decide
        from dotears.sfx_contrast import read_policy as read_contrast_policy

        policy = read_contrast_policy(policy_path)
        engine_class, decide_function = ContrastSFXEngine, contrast_decide
    elif schema == "dotears.sfx-presence-policy/3":
        from dotears.sfx_micro import MicroSFXEngine
        from dotears.sfx_micro import read_policy as read_micro_policy
        from dotears.sfx_presence_deep import decide as micro_decide

        policy = read_micro_policy(policy_path)
        engine_class, decide_function = MicroSFXEngine, micro_decide
    elif schema == "dotears.sfx-presence-policy/2":
        from dotears.sfx_presence_deep import LatentSFXEngine
        from dotears.sfx_presence_deep import decide as latent_decide
        from dotears.sfx_presence_deep import read_policy as read_latent_policy

        policy = read_latent_policy(policy_path)
        engine_class, decide_function = LatentSFXEngine, latent_decide
    else:
        policy = read_policy(policy_path)
        engine_class, decide_function = RawSFXEngine, decide
    policy_sha = file_digest(policy_path)
    source_sha = file_digest(report_path)
    report = json.loads(report_path.read_text("utf-8"))
    events = [e for e in report["events"] if e["category"] == "audio_attack"]
    if (
        not 1 <= len(events) <= 200
        or len({e["id"] for e in events}) != len(events)
        or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", e["id"]) for e in events)
    ):
        raise LimitError("SFX presence needs 1–200 unique, safely named attack events.")
    native = local_artifact(report_path.parent, report["audio"]["native_path"])
    offset, duration = [report["audio"][k] for k in ("timeline_offset_seconds", "duration_seconds")]
    if not all(math.isfinite(v) for v in (offset, duration)) or duration < 0.128:
        raise ConfigurationError("Invalid report audio timeline.")
    for event in events:
        if (
            not math.isfinite(event["peak_time"])
            or not 0 <= event["peak_time"] - offset <= duration
        ):
            raise ConfigurationError("Attack peak is outside the report audio.")
    import librosa
    import soundfile as sf

    native_sha = file_digest(native)
    started = time.perf_counter()
    engine = engine_class(ced, clap, cache, threads)
    if engine.identity != policy["models"]:
        raise ConfigurationError("Presence policy used different models or feature definitions.")
    expansion = None
    expansion_meta = {"status": "unavailable"}
    if expansion_path is not None:
        try:
            expansion_sha = file_digest(expansion_path)
            candidate_policy = read_expansion_policy(expansion_path, engine.identity)
            if (
                schema != "dotears.sfx-presence-policy/5"
                or candidate_policy["legacy_policy_sha256"] != policy_sha
                or not any(head["enabled"] for head in candidate_policy["heads"].values())
            ):
                raise ConfigurationError("The expansion needs the verified profile and baseline.")
            expansion = candidate_policy
            expansion_meta = {
                "status": "completed",
                "policy_sha256": expansion_sha,
                "enabled_families": [k for k, h in expansion["heads"].items() if h["enabled"]],
                "method": "learned_clap_ced_audio_features",
            }
        except (OSError, ValueError, KeyError, ConfigurationError) as exc:
            print(f"Sound expansion unavailable: {exc}", file=sys.stderr)
            expansion_meta = {"status": "failed"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary_directory(prefix="dotears-presence-", dir=output.parent) as name:
        stage = Path(name)
        (stage / "audio").mkdir()
        results = []
        for event in events:
            if time.perf_counter() - started > timeout:
                raise LimitError("Presence scan time limit reached; feature cache retained.")
            peak = event["peak_time"] - offset
            first = max(0.0, min(peak - 1.0, duration - 3.0))
            last = min(duration, first + 3.0)
            audio, rate = read_window(native, first, last)
            first = round(first * rate) / rate
            values = librosa.resample(audio, orig_sr=rate, target_sr=32000).astype(np.float32)
            relative = min(peak - first, len(values) / 32000)
            evidence = engine.analyze(values, relative)
            decision = decide_function(policy, evidence)
            path = f"audio/{event['id']}-mixture.wav"
            focus = window(values, relative, 0.15, 0.55)
            sf.write(stage / path, focus, 32000, subtype="FLOAT")
            additional = (
                decide_expanded_sound(expansion, evidence, decision["label"])
                if expansion is not None
                else {}
            )
            results.append(
                {
                    "event_id": event["id"],
                    "peak_time": event["peak_time"],
                    "context_start": first + offset,
                    "context_end": first + len(values) / 32000 + offset,
                    "focus_start": first + max(0, relative - 0.15) + offset,
                    "focus_end": first + min(len(values) / 32000, relative + 0.55) + offset,
                    "mixture_focus_path": path,
                    "mixture_focus_sha256": file_digest(stage / path),
                    **decision,
                    **additional,
                    "raw_evidence": evidence,
                }
            )
            print(
                f"SFX raw {len(results)}/{len(events)}: {event['id']} = {decision['label']}",
                file=sys.stderr,
                flush=True,
            )
        if file_digest(report_path) != source_sha or file_digest(native) != native_sha:
            raise MediaError("Original report/native audio changed during the scan.")
        if file_digest(policy_path) != policy_sha:
            raise ConfigurationError("Presence policy changed during the scan.")
        if expansion is not None and file_digest(expansion_path) != expansion_meta["policy_sha256"]:
            raise ConfigurationError("Sound expansion policy changed during the scan.")
        result = {
            "schema_version": "dotears.sfx-presence-report/1",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "experimental_raw_audio_presence",
            "analysis_id": report["analysis_id"],
            "source_sha256": report["source"]["sha256"],
            "source_report_path": str(report_path),
            "source_report_sha256": source_sha,
            "native_audio_sha256": native_sha,
            "policy_path": str(policy_path.resolve()),
            "policy_sha256": policy_sha,
            "models": engine.identity,
            "feature_names": policy.get("feature_names", FEATURE_NAMES),
            "label_taxonomy": policy.get("label_taxonomy", "dotears.sfx-types/1-narrow-click"),
            "ced_tags": CED_TAGS,
            "clap_labels": list(PROMPTS),
            "human_annotations_used_in_inference": False,
            "separation_used_for_type_decision": False,
            "events": results,
            "counters": engine.counters,
            "sound_expansion": expansion_meta,
            "seconds": round(time.perf_counter() - started, 3),
            "limitations": [
                "Types are provisional; scores are not probabilities.",
                "Background speech/music may mask a sound; unknown is not absence.",
                "Only existing DSP attacks are checked; "
                "neighboring windows may describe one event.",
                "A tone alone may not distinguish a notification from other UI feedback.",
                "Spectral contrast, when used, is a numerical view, not an isolated effect.",
                "Some independent source labels come from authors, not human listening review.",
            ],
        }
        (stage / "sfx-auto.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), "utf-8"
        )
        if output.exists():
            raise ConfigurationError("SFX output was created concurrently.")
        stage.rename(output)
    return {
        "sfx_report_path": str(output / "sfx-auto.json"),
        "events": len(results),
        "accepted": sum(e["label"] != "unknown" for e in results),
        "seconds": result["seconds"],
    }
