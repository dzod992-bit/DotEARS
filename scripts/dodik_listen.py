"""Turn a local audio or video file into one review for Dodik's Windows task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from collections import Counter
from functools import partial
from html import escape
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import numpy as np

from dotears.core import digest_file, reference_lock
from dotears.errors import DotEarsError
from dotears.motion_cues import supported_transient, transient_proposals
from dotears.sfx import local_artifact
from dotears.sound_description import NAMES, describe, describe_transient, read_clip
from dotears.storage import temporary_directory
from dotears.ui_acoustics import has_ring_support, repeats_on_beat

PROJECT = Path(__file__).resolve().parents[1]
POLICY = PROJECT / "validation-output/sfx-mixtures-2026-10-30/hierarchy-policy/policy.json"
SFX_PYTHON = PROJECT / ".cache/sfx-venv/Scripts/python.exe"
MODELS = PROJECT / "models"
LABELS = {"whoosh": "Свуш", "ui_sound": "Короткий UI-сигнал", "unknown": "Не определён"}


def worker(python: Path, args: list[str], log: Path, timeout: int) -> dict:
    result = subprocess.run(
        [str(python), "-m", "dotears", *args],
        cwd=PROJECT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    log.write_text(result.stderr, "utf-8")
    if result.returncode:
        lines = result.stderr.strip().splitlines()
        detail = lines[-1][:1000] if lines else "No diagnostic from DotEARS."
        raise ValueError(f"{args[0]} returned {result.returncode}: {detail}")
    return json.loads(result.stdout)


def check_effects(report: dict, stage: Path, output: Path) -> tuple[str, dict]:
    attacks = {
        event["id"]: event for event in report["events"] if event["category"] == "audio_attack"
    }
    if not attacks:
        return "skipped_no_attacks", {}
    ced, clap = MODELS / "ced-mini", MODELS / "clap-htsat-unfused"
    if not all(path.exists() for path in (SFX_PYTHON, POLICY, ced, clap)):
        return "unavailable", {}
    # Serialize writers to the persistent feature cache for this local library.
    with reference_lock(output / ".effect-cache.lock"):
        result = worker(
            SFX_PYTHON,
            [
                "sfx-auto",
                report["report_path"],
                "--ced",
                str(ced),
                "--clap",
                str(clap),
                "--policy",
                str(POLICY),
                "--output",
                str(stage / "effects"),
                "--cache",
                str(output / "effect-cache"),
                "--timeout",
                "420",
            ],
            stage / "effects.log",
            timeout=450,
        )
    raw = json.loads(Path(result["sfx_report_path"]).read_text("utf-8"))
    events = {event["event_id"]: event for event in raw["events"]}
    if (
        raw["source_report_sha256"] != digest_file(Path(report["report_path"]))
        or raw["policy_sha256"] != digest_file(POLICY)
        or raw["source_sha256"] != report["source"]["sha256"]
        or raw["label_taxonomy"] != "dotears.sfx-types/3-hierarchical"
        or len(raw["events"]) != len(attacks)
        or events.keys() != attacks.keys()
        or any(
            event["peak_time"] != attacks[key]["peak_time"] or event["label"] not in LABELS
            for key, event in events.items()
        )
    ):
        raise ValueError("Effect results do not match this analysis and policy.")
    return "completed", events


def scan_motion_cues(
    full_audio: Path, audio: dict, bpm: float | None, attacks: list, stage: Path, final: Path
) -> list[dict]:
    values, rate = read_clip(full_audio, max_seconds=121)
    offset = audio.get("timeline_offset_seconds", 0)
    covered = [
        event["ui_transient_peak_seconds"] or event["candidate_peak_seconds"]
        for event in attacks
        if event["generation_prompt"]
    ]
    cues = []
    for proposal in transient_proposals(values, rate):
        peak = proposal["time"]
        time = peak + offset
        if not supported_transient(proposal) or any(abs(time - other) < 0.032 for other in covered):
            continue
        if (
            bpm
            and audio["has_music"]
            and repeats_on_beat(full_audio, 0, bpm, peak, proposal["form"])
        ):
            continue
        description, prompt = describe_transient(proposal)
        first = max(0, round((peak - 0.15) * rate))
        last = min(len(values), round((peak + 0.55) * rate))
        relative = Path("motion-clips") / f"cue-{len(cues) + 1:03d}.wav"
        path = stage / relative
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as clip:
            clip.setparams((values.shape[1], 2, rate, 0, "NONE", "not compressed"))
            clip.writeframes(np.rint(values[first:last] * 32768).astype("<i2").tobytes())
        cues.append(
            {
                "time_seconds": round(time, 6),
                "effect_family": "unknown",
                "shape_hint": proposal["form"],
                "description_ru": description,
                "acoustic_evidence": proposal,
                "clip_path": str(final / relative),
                "clip_seconds": [first / rate + offset, last / rate + offset],
                "generation_prompt": prompt,
            }
        )
        covered.append(time)
    return cues


def review(report: dict, effects: dict, summary: dict, stage: Path, final: Path) -> dict:
    folder = Path(report["report_path"]).parent
    audio = report["audio"]
    full_audio = local_artifact(folder, audio["native_path"]) if audio["present"] else None
    bpm = (report.get("rhythm") or {}).get("bpm")
    intervals = [e for e in report["events"] if e["category"] in {"speech", "music"}]
    backgrounds = intervals + [
        {**segment, "category": "speech"} for segment in report["transcription"]["segments"]
    ]
    backgrounds_checked = report["classification"]["status"] == "completed"
    attacks = []
    for event in report["events"]:
        if event["category"] != "audio_attack":
            continue
        prediction = effects.get(event["id"], {})
        family = prediction.get("label", "unknown")
        clip_path = local_artifact(folder, event["clip_path"])
        background_in_clip = sorted(
            {
                interval["category"]
                for interval in backgrounds
                if interval["start"] < event["clip_end"] and interval["end"] > event["clip_start"]
            }
        )
        rhythmic_cue = None
        if full_audio and bpm and (audio["has_music"] or "music" in background_in_clip):
            rhythmic_cue = partial(
                repeats_on_beat,
                full_audio,
                event["clip_start"] - audio.get("timeline_offset_seconds", 0),
                bpm,
            )
        description = describe(
            clip_path,
            event["peak_time"] - event["clip_start"],
            family,
            background_in_clip if background_in_clip or backgrounds_checked else None,
            ring_context=has_ring_support(prediction.get("raw_evidence", {})),
            rhythmic_cue=rhythmic_cue,
        )
        transient = (description["ui_form_evidence"] or {}).get("short_transient", {})
        prompt = description["generation_prompt"]
        cue_form = (
            transient["candidate"]
            if prompt and prompt["basis"] == "observed_short_transient_candidate"
            else description["ui_sound_form"]
        )
        probe = transient.get("probes", {}).get(cue_form)
        attacks.append(
            {
                "id": event["id"],
                "candidate_peak_seconds": event["peak_time"],
                "effect_family": family,
                "baseline_effect_family": family,
                "effect_role": "unknown",
                "effect_onset_seconds": None,
                "evidence_window_seconds": [prediction["focus_start"], prediction["focus_end"]]
                if prediction
                else None,
                "clip_path": str(clip_path),
                "clip_seconds": [event["clip_start"], event["clip_end"]],
                "relative_attack_strength": event["score"],
                "overlap": event["overlap"],
                "ui_transient_peak_seconds": (
                    round(event["clip_start"] + probe["relative_peak_seconds"], 6)
                    if probe
                    else None
                ),
                **description,
            }
        )
    motion_cues = (
        scan_motion_cues(full_audio, audio, bpm, attacks, stage, final) if full_audio else []
    )
    summary["attacks"] = len(attacks)
    summary["effect_windows"] = dict(Counter(event["effect_family"] for event in attacks))
    summary["generation_drafts"] = sum(event["generation_prompt"] is not None for event in attacks)
    summary["motion_cue_count"] = len(motion_cues)
    summary["ui_forms"] = dict(
        Counter(event["ui_sound_form"] for event in attacks if event["effect_family"] == "ui_sound")
    )
    return {
        "schema": "dotears.dodik-review/1",
        **summary,
        "source_path": report.get("request_source", report["source"])["path"],
        "source_sha256": report["source"]["sha256"],
        "duration_seconds": report["source"]["duration_seconds"],
        "has_music": audio["has_music"],
        "has_speech": audio["has_speech"],
        "full_audio_path": str(full_audio) if full_audio else None,
        "audio_intervals": [
            {key: event[key] for key in ("start", "end", "category", "label")}
            for event in intervals
        ],
        "transcription": report["transcription"],
        "attack_detection": report["attack_detection"],
        "attack_windows": attacks,
        "motion_cues": motion_cues,
        "visual_events": report["visual_events"],
        "sync_candidates": report["sync_candidates"],
        "warnings": report["warnings"]
        + [
            "Provisional effect families describe mixed windows and can repeat across windows.",
            "Candidate peaks do not establish effect onsets or editing intent.",
            "Acoustic descriptions measure the mixed WAV near each candidate. "
            "Generation prompts are drafts for creating a similar isolated sound.",
        ],
    }


def markdown(data: dict, folder: Path) -> str:
    def link(path: str) -> str:
        return quote(Path(os.path.relpath(path, folder)).as_posix())

    def detected(value: bool | None) -> str:
        return {True: "есть признаки", False: "модель не отметила", None: "не определено"}[value]

    lines = [
        "# DotEARS · разбор дорожки",
        "",
        f"Длительность {data['duration_seconds']:.3f} с. Статус `{data['status']}`.",
        f"Речь — {detected(data['has_speech'])}. Музыка — {detected(data['has_music'])}.",
        f"[Полный отчёт]({link(data['report_path'])}) · [JSON для агента](review.json)",
    ]
    if data["full_audio_path"]:
        lines.append(f"[Полная аудиодорожка]({link(data['full_audio_path'])})")
    if data["status"] == "no_audio":
        lines += ["", "В файле нет звуковой дорожки."]
    for name, status in data["stages"].items():
        lines.append(f"- `{name}` · `{status}`")
    lines += [
        "",
        "## Речь",
        "",
        "Автоматическая расшифровка. Текст из записи может содержать ошибки.",
    ]
    for segment in data["transcription"]["segments"]:
        speech = escape(" ".join(segment["text"].split())).replace("[", "\\[")
        lines.append(f"- {segment['start']:.3f}–{segment['end']:.3f} с · {speech}")
    lines += [
        "",
        "## Звуковые акценты",
        "",
        "Метки предварительные и описывают звук в окне. Фрагменты содержат смешанную дорожку.",
        "Точное начало и назначение эффекта могут оставаться неизвестными.",
        "",
        "| Пик кандидата, с | Описание | Фрагмент |",
        "| ---: | --- | --- |",
    ]
    named = [event for event in data["attack_windows"] if event["generation_prompt"] is not None]
    unknown = sorted(
        (event for event in data["attack_windows"] if event["generation_prompt"] is None),
        key=lambda event: -event["relative_attack_strength"],
    )
    for event in sorted(named + unknown[:8], key=lambda event: event["candidate_peak_seconds"]):
        lines.append(
            f"| {event['candidate_peak_seconds']:.3f} | {event['description_ru']} "
            f"| [WAV]({link(event['clip_path'])}) |"
        )
    examples = {}
    for event in sorted(named, key=lambda item: -item["relative_attack_strength"]):
        if event["generation_prompt"]:
            key = (
                event["ui_sound_form"]
                if event["ui_sound_form"] != "unknown"
                else event["effect_family"]
            )
            basis = event["generation_prompt"]["basis"]
            if basis == "observed_ringing_candidate":
                key = "ring_candidate"
            if basis == "observed_short_transient_candidate":
                form = event["ui_form_evidence"]["short_transient"]["candidate"]
                key = "click_candidate" if form == "click_like" else "tap_candidate"
            if basis == "observed_pitch_glide":
                key = (
                    "Тональный подъём"
                    if event["motion_shape"]["direction"] == "up"
                    else "Тональное понижение"
                )
            examples.setdefault(key, event)
    if examples:
        lines += ["", "## Черновики для ElevenLabs", "", "По одному примеру на тип звука."]
        for family, event in sorted(examples.items()):
            time = event.get("ui_transient_peak_seconds") or event["candidate_peak_seconds"]
            lines += [
                "",
                f"**{NAMES.get(family, family)}** · {time:.3f} с",
                "",
                event["generation_prompt"]["text"],
            ]
    if data["motion_cues"]:
        lines += [
            "",
            "## Дополнительные короткие звуки",
            "",
            "Поиск по частотным полосам. Названия и происхождение кандидатов неизвестны.",
        ]
        for cue in data["motion_cues"]:
            lines += [
                "",
                f"**{cue['time_seconds']:.3f} с** · {cue['description_ru']} "
                f"[WAV]({link(cue['clip_path'])})",
                "",
                cue["generation_prompt"]["text"],
            ]
    detection = data["attack_detection"]
    if detection["status"] == "completed":
        lines += [
            "",
            f"В JSON {data['attacks']} кандидатов. Лимит отсёк {detection['omitted_by_limit']}.",
        ]
    lines += [
        "",
        f"Смены кадра — {len(data['visual_events'])}. "
        f"Совпадения по времени со звуком — {len(data['sync_candidates'])}.",
        "Их таймкоды и интервалы речи и музыки доступны в JSON.",
    ]
    return "\n".join(lines) + "\n"


def listen(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"review-{uuid4().hex[:12]}"
    with temporary_directory(prefix=".review-", dir=output) as temporary:
        stage = Path(temporary)
        command = [
            "analyze",
            str(source),
            "--profile",
            "motion",
            "--max-events",
            "200",
            "--output",
            str(output / "library"),
        ]
        for flag, model in (("--yamnet-model", "yamnet"), ("--asr-model", "whisper-tiny")):
            model_dir = MODELS / model
            if model_dir.is_dir():
                command += [flag, str(model_dir)]
        report = worker(Path(sys.executable), command, stage / "analysis.log", timeout=300)
        try:
            effect_status, effects = check_effects(report, stage, output)
        except (DotEarsError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            (stage / "effect-error.txt").write_text(str(exc), "utf-8")
            effect_status, effects = "failed", {}
        stages = {
            "speech": report["transcription"]["status"],
            "audio_tags": report["classification"]["status"],
            "effect_families": effect_status,
        }
        status = "no_audio"
        if report["audio"]["present"]:
            available = {"completed", "skipped_no_attacks"}
            status = (
                "completed" if all(value in available for value in stages.values()) else "partial"
            )
        summary = {
            "status": status,
            "stages": stages,
            "sound_profile": "ui_motion",
            "analysis_id": report["analysis_id"],
            "cache_hit": report["cache_hit"],
            "report_path": report["report_path"],
            "review_path": str(final / "review.md"),
            "review_json_path": str(final / "review.json"),
            "effect_report_path": str(final / "effects/sfx-auto.json") if effects else None,
        }
        data = review(report, effects, summary, stage, final)
        (stage / "review.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        (stage / "review.md").write_text(markdown(data, final), "utf-8")
        stage.rename(final)
    return summary


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing local video or audio file")
    parser.add_argument("--output", type=Path, default=PROJECT / "validation-output/dodik-mvp")
    args = parser.parse_args()
    try:
        source = args.source.expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("Source must be an existing local audio or video file.")
        result = listen(source, args.output.expanduser().resolve())
    except (DotEarsError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(
            json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
