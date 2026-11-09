from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def test_doctor_json_and_gpu_independence():
    result = subprocess.run(
        [sys.executable, "-m", "dotears", "doctor"], capture_output=True, check=True
    )
    report = json.loads(result.stdout.decode("utf-8"))
    assert report["cpu_supported"] and not report["gpu_required"]


@pytest.mark.integration
def test_cli_utf8_source_path_and_json_error(media_files, tmp_path):
    source = tmp_path / "Референс 🎧.mp4"
    source.write_bytes(media_files["video"].read_bytes())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dotears",
            "extract",
            str(source),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "ascii"},
        check=True,
    )
    report = json.loads(result.stdout.decode("utf-8"))
    assert report["source"]["path"] == str(source.resolve())
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dotears",
            "extract",
            str(tmp_path / "absent.mp4"),
        ],
        capture_output=True,
    )
    assert failed.returncode == 2
    assert json.loads(failed.stderr.decode("utf-8"))["error"]["code"] == "invalid_media"
