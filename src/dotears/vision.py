"""Scene-change observations from FFmpeg's scene score and actual frame PTS."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dotears.media import executable, input_args, run


def scene_changes(path: Path, threshold: float, timeout: float, max_duration: float) -> list[dict]:
    result = run(
        [
            executable("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "info",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            *input_args(path),
            "-t",
            str(max_duration + 1),
            "-an",
            "-vf",
            f"scale=160:90,select='gt(scene,{threshold})',showinfo",
            "-fps_mode",
            "vfr",
            "-f",
            "null",
            "-",
        ],
        timeout,
    )
    timestamps = sorted(
        {float(value) for value in re.findall(r"pts_time:([\d.eE+\-]+)", result.stderr)}
    )
    return [
        {
            "id": f"visual_{i:03d}",
            "time": value,
            "type": "abrupt_frame_change",
            "method": "ffmpeg_scene_score",
            "threshold": threshold,
            "interpretation": "Observable image change; action semantics and causality unknown.",
        }
        for i, value in enumerate(timestamps, 1)
        if 0 <= value <= max_duration
    ]


def synchronize(
    events: list[dict], visual: list[dict], tolerance: float = 0.15
) -> list[dict[str, Any]]:
    result = []
    for event in events:
        if "peak_time" not in event:
            continue
        for frame in visual:
            delta = event["peak_time"] - frame["time"]
            if abs(delta) <= tolerance:
                result.append(
                    {
                        "event_id": event["id"],
                        "visual_event_id": frame["id"],
                        "delta_seconds": round(delta, 6),
                        "tolerance_seconds": tolerance,
                        "interpretation": "temporal_coincidence_not_causation",
                    }
                )
    return result
