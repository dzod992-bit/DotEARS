"""Generate original synthetic test media; no third-party audio is used."""

from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np

from dotears.storage import temporary_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("private-media/demo.mp4"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    with temporary_directory(prefix=".dotears-demo-", dir=args.output.parent) as temporary:
        wav = Path(temporary) / "signal.wav"
        rate, duration = 44100, 8
        samples = np.zeros(duration * rate)
        rng = np.random.default_rng(739)
        for onset in np.arange(0.5, duration, 0.5):
            t = np.arange(round(0.1 * rate)) / rate
            pulse = 0.65 * np.sin(2 * np.pi * 100 * t) + 0.15 * rng.standard_normal(len(t))
            first = round(onset * rate)
            samples[first : first + len(t)] += pulse * np.exp(-t * 50)
        with wave.open(str(wav), "wb") as handle:
            handle.setparams((2, 2, rate, 0, "NONE", "not compressed"))
            handle.writeframes(np.repeat((samples * 32767).astype("<i2"), 2).tobytes())
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "lavfi",
                "-i",
                "color=black:s=320x180:r=30:d=3.5",
                "-f",
                "lavfi",
                "-i",
                "color=white:s=320x180:r=30:d=4.5",
                "-i",
                str(wav),
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map",
                "[v]",
                "-map",
                "2:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-shortest",
                str(args.output.resolve()),
            ],
            check=True,
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
