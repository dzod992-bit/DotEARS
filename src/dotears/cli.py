from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path

from dotears.core import AnalysisOptions, analyze, default_library
from dotears.errors import DotEarsError
from dotears.network import proxy_summary


def doctor() -> dict:
    return {
        "python": sys.version.split()[0],
        "cpu_supported": True,
        "gpu_required": False,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "optional_packages": {
            name: importlib.util.find_spec(name) is not None
            for name in (
                "yt_dlp",
                "faster_whisper",
                "tensorflow",
                "mcp",
                "torch",
                "transformers",
                "torchlibrosa",
            )
        },
        "default_library": str(default_library()),
        "analysis_network_policy": "local_models_only",
        "dot_integration": "documented_local_task_route_not_a_live_connection_check",
        "network": proxy_summary(),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="dotears", description="Give Your AI Agent Ears")
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("analyze", "extract"):
        command = commands.add_parser(name)
        command.add_argument("source", help="Local media file or public video HTTP(S) URL")
        command.add_argument("--output", type=Path, help="Reference library root")
        command.add_argument("--profile", choices=("motion", "general"), default="motion")
        command.add_argument("--max-duration", type=float, default=120)
        command.add_argument("--max-size-mb", type=float, default=200)
        command.add_argument("--sample-rate", type=int, default=22050)
        command.add_argument("--channels", type=int, choices=(1, 2), default=1)
        command.add_argument("--max-events", type=int, default=60)
        command.add_argument("--clip-before", type=float, default=0.15)
        command.add_argument("--clip-after", type=float, default=0.55)
        command.add_argument("--no-visual", action="store_true")
        command.add_argument("--no-original", action="store_true")
        command.add_argument("--scene-threshold", type=float, default=0.3)
        command.add_argument("--sync-tolerance", type=float, default=0.15)
        command.add_argument("--timeout", type=float, default=120)
        command.add_argument("--yamnet-model", type=Path)
        command.add_argument("--classifier-threshold", type=float, default=0.3)
        command.add_argument("--asr-model", type=Path)
        command.add_argument("--language", help="Optional ISO code, e.g. ru or en")
        command.add_argument("--threads", type=int, default=4)
    commands.add_parser("doctor")
    clip = commands.add_parser("clip", help="Cut a decoded PCM WAV at audio-relative times")
    clip.add_argument("source", type=Path)
    clip.add_argument("destination", type=Path)
    clip.add_argument("--start", required=True, type=float)
    clip.add_argument("--end", required=True, type=float)
    model = commands.add_parser("prepare-model", help="Explicit one-time free model download")
    model.add_argument("kind", choices=("yamnet", "whisper", "clap"))
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--size", choices=("tiny", "base", "small"), default="tiny")
    sfx = commands.add_parser(
        "sfx", help="Experimental local SFX hypotheses from an existing report"
    )
    sfx.add_argument("report", type=Path)
    sfx.add_argument("--model", type=Path, required=True)
    sfx.add_argument("--output", type=Path, required=True)
    sfx.add_argument("--event-id", action="append")
    sfx.add_argument("--window", choices=("excerpt", "context"), default="excerpt")
    sfx.add_argument("--context-seconds", type=float, default=3)
    sfx.add_argument("--padding", choices=("repeatpad", "pad"), default="repeatpad")
    sfx.add_argument("--threads", type=int, default=4)
    separate = commands.add_parser(
        "separate", help="Experimental AudioSep estimate for one existing attack context"
    )
    separate.add_argument("report", type=Path)
    separate.add_argument("--event-id", required=True)
    separate.add_argument("--prompt", required=True)
    separate.add_argument("--model", type=Path, required=True)
    separate.add_argument("--output", type=Path, required=True)
    separate.add_argument("--threads", type=int, default=4)
    auto = commands.add_parser(
        "sfx-auto", help="Check all attacks with three fixed local queries and a frozen policy"
    )
    auto.add_argument("report", type=Path)
    auto_models = auto.add_mutually_exclusive_group(required=True)
    auto_models.add_argument("--audiosep", type=Path, help="Legacy query-conditioned research mode")
    auto_models.add_argument("--ced", type=Path, help="Raw-audio presence mode with float CED/CLAP")
    auto.add_argument("--clap", type=Path, required=True)
    auto.add_argument("--policy", type=Path, required=True)
    auto.add_argument("--output", type=Path, required=True)
    auto.add_argument("--cache", type=Path, required=True)
    auto.add_argument("--sound-expansion-policy", type=Path, help="Verified extra sound families")
    auto.add_argument("--threads", type=int, default=4)
    auto.add_argument("--timeout", type=float, default=1800)
    match = commands.add_parser("sfx-match", help="Match locally reviewed, known SFX waveforms")
    match.add_argument("report", type=Path)
    match.add_argument("--policy", type=Path, required=True)
    match.add_argument("--output", type=Path, required=True)
    server = commands.add_parser("serve", help="Local stdio MCP server, no HTTP listener")
    server.add_argument("--media-root", type=Path, required=True)
    server.add_argument("--output", type=Path, required=True)
    server.add_argument(
        "--allow-urls", action="store_true", help="Explicitly enable public URL input"
    )
    server.add_argument("--yamnet-model", type=Path)
    server.add_argument("--asr-model", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command in {"analyze", "extract"}:
            options = AnalysisOptions(
                profile=args.profile,
                max_duration=args.max_duration,
                max_bytes=int(args.max_size_mb * 1024**2),
                sample_rate=args.sample_rate,
                channels=args.channels,
                max_events=args.max_events,
                clip_before=args.clip_before,
                clip_after=args.clip_after,
                visual=not args.no_visual,
                scene_threshold=args.scene_threshold,
                sync_tolerance=args.sync_tolerance,
                preserve_original=not args.no_original,
                extract_only=args.command == "extract",
                operation_timeout=args.timeout,
                yamnet_model=str(args.yamnet_model.resolve()) if args.yamnet_model else None,
                classifier_threshold=args.classifier_threshold,
                asr_model=str(args.asr_model.resolve()) if args.asr_model else None,
                language=args.language,
                threads=args.threads,
            )
            result = analyze(args.source, args.output, options)
        elif args.command == "clip":
            from dotears.media import cut_wav

            if args.destination.exists():
                raise DotEarsError("Clip destination already exists; choose a new path.")
            if not args.source.is_file():
                raise DotEarsError(f"WAV file does not exist: {args.source}")
            result = {
                "path": str(args.destination.resolve()),
                **cut_wav(args.source, args.destination, args.start, args.end),
            }
        elif args.command == "sfx":
            from dotears.sfx import classify_sfx

            if args.output.exists():
                raise DotEarsError("SFX output already exists; choose a new path.")
            result = classify_sfx(
                args.report,
                args.model,
                args.event_id,
                args.window,
                args.context_seconds,
                args.padding,
                args.threads,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            result["sfx_report_path"] = str(args.output.resolve())
        elif args.command == "separate":
            from dotears.separate_report import separate_report_window

            result = separate_report_window(
                args.report, args.event_id, args.prompt, args.model, args.output, args.threads
            )
        elif args.command == "sfx-auto":
            if args.ced:
                from dotears.sfx_detect_report import scan_raw_report

                result = scan_raw_report(
                    args.report,
                    args.ced,
                    args.clap,
                    args.policy,
                    args.output,
                    args.cache,
                    args.threads,
                    args.timeout,
                    args.sound_expansion_policy,
                )
            else:
                if args.sound_expansion_policy is not None:
                    raise DotEarsError("Sound expansion requires --ced.")
                from dotears.sfx_auto import scan_report

                result = scan_report(
                    args.report,
                    args.audiosep,
                    args.clap,
                    args.policy,
                    args.output,
                    args.cache,
                    args.threads,
                    args.timeout,
                )
        elif args.command == "sfx-match":
            from dotears.sfx_match import scan_reference_report

            result = scan_reference_report(args.report, args.policy, args.output)
        elif args.command == "prepare-model":
            from dotears.prepare import prepare

            result = prepare(args.kind, args.output, args.size)
        else:
            from dotears.integrations.mcp_server import serve

            serve(args.media_root, args.output, args.allow_urls, args.yamnet_model, args.asr_model)
            return 0
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except DotEarsError as exc:
        print(
            json.dumps({"error": {"code": exc.code, "message": str(exc)}}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {"error": {"code": "io_or_value_error", "message": str(exc)}}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 2
