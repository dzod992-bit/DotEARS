from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def ffmpeg():
    found = shutil.which("ffmpeg")
    if not found or not shutil.which("ffprobe"):
        pytest.skip("Integration tests require FFmpeg and ffprobe on PATH")
    return found


def write_signal(path: Path, duration: float = 4, rate: int = 44100, silent: bool = False):
    rng = np.random.default_rng(739)
    signal = np.zeros(round(duration * rate), dtype=np.float64)
    if not silent:
        for onset in (0.5, 1.5, 2.5, 3.5):
            length = round(0.08 * rate)
            t = np.arange(length) / rate
            pulse = 0.7 * np.sin(2 * np.pi * 120 * t) + 0.2 * rng.standard_normal(length)
            pulse *= np.exp(-t * 55)
            first = round(onset * rate)
            if first + length <= len(signal):
                signal[first : first + length] += pulse
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        stereo = np.repeat(np.clip(signal, -1, 1)[:, None], 2, axis=1)
        wav.writeframes((stereo * 32767).astype("<i2").tobytes())
    return signal


@pytest.fixture(scope="session")
def media_files(tmp_path_factory, ffmpeg):
    root = tmp_path_factory.mktemp("media")
    wav = root / "pulses.wav"
    write_signal(wav)
    video = root / "reference.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=30:d=1.5",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=160x90:r=30:d=2.5",
            "-i",
            str(wav),
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a:0",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    mute = root / "mute.mp4"
    subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y", "-i", str(video), "-an", "-c:v", "copy", str(mute)],
        check=True,
        capture_output=True,
    )
    return {"root": root, "wav": wav, "video": video, "mute": mute}
