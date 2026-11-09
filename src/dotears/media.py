"""Bounded local FFmpeg operations; commands never use a shell."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

from dotears.errors import DependencyError, LimitError, MediaError

FORMATS = "mov,matroska,webm,avi,wav,mp3,flac,ogg,aac"


def executable(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise DependencyError(f"{name} is not on PATH. Install FFmpeg and restart your terminal.")
    return found


def run(args: list[str], timeout: float = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError as exc:
        raise DependencyError(f"Executable not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise LimitError(f"Media operation exceeded {timeout:g} seconds.") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"Media operation failed: {exc.stderr[-1500:]}") from exc


def input_args(path: Path) -> list[str]:
    return ["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS, "-i", str(path)]


def probe(path: Path, max_duration: float, max_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        raise MediaError(f"Input file does not exist: {path}")
    if path.stat().st_size > max_bytes:
        raise LimitError(f"Input exceeds {max_bytes} bytes.")
    result = run(
        [
            executable("ffprobe"),
            "-v",
            "error",
            *input_args(path),
            "-show_format",
            "-show_streams",
            "-of",
            "json",
        ],
        30,
    )
    try:
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("nonpositive duration")
    except (KeyError, ValueError, TypeError) as exc:
        raise MediaError("Media has no finite positive duration.") from exc
    if duration > max_duration:
        raise LimitError(f"Duration {duration:.3f}s exceeds {max_duration:g}s.")
    audio = next((s for s in data.get("streams", []) if s["codec_type"] == "audio"), None)
    video = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), None)
    if not audio and not video:
        raise MediaError("No audio or video streams found.")
    if audio and (int(audio.get("channels", 0)) > 8 or int(audio.get("sample_rate", 0)) > 192000):
        raise LimitError("At most 8 audio channels and 192 kHz are supported.")
    start = float(data["format"].get("start_time", 0))
    offset = max(0.0, float(audio.get("start_time", start)) - start) if audio else 0.0
    return {
        "duration_seconds": duration,
        "size_bytes": path.stat().st_size,
        "audio_stream": audio,
        "video_stream": video,
        "audio_offset_seconds": offset,
        "timeline_origin": "container_start",
    }


def extract_audio(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = 22050,
    channels: int = 1,
    max_duration: float = 120,
    timeout: float = 120,
) -> float:
    run(
        [
            executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            *input_args(source),
            "-map",
            "0:a:0",
            "-vn",
            "-t",
            str(max_duration + 1),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-map_metadata",
            "-1",
            str(destination),
        ],
        timeout,
    )
    with wave.open(str(destination), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    if duration <= 0:
        raise MediaError("Audio stream decoded to an empty waveform.")
    if duration > max_duration + 0.05:
        raise LimitError("Decoded audio exceeds the duration limit.")
    return duration


def preserve_audio(source: Path, destination: Path, timeout: float = 120) -> None:
    """Remux the first audio stream, preserving compressed packets without re-encoding."""
    run(
        [
            executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            *input_args(source),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "copy",
            "-map_metadata",
            "-1",
            str(destination),
        ],
        timeout,
    )


def cut_wav(source: Path, destination: Path, start: float, end: float) -> dict[str, float]:
    """Sample-accurate cutting of decoded PCM; preserve its channels and rate."""
    if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
        raise MediaError("Clip requires finite 0 <= start < end.")
    with wave.open(str(source), "rb") as wav:
        rate = wav.getframerate()
        first, last = round(start * rate), round(end * rate)
        if last > wav.getnframes() or first >= last:
            raise MediaError("Clip interval lies outside the decoded audio.")
        wav.setpos(first)
        frames = wav.readframes(last - first)
        params = wav.getparams()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return {"start": first / rate, "end": last / rate}
