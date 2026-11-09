from __future__ import annotations

import json
import socket
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from dotears import AnalysisOptions, analyze
from dotears.errors import ConfigurationError, DependencyError, LimitError, MediaError
from dotears.media import cut_wav, extract_audio, probe


@pytest.mark.integration
def test_full_cpu_pipeline_schema_timestamps_and_native_clips(media_files, tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Local analysis attempted network access")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    result = analyze(media_files["video"], tmp_path)
    assert result["audio"]["present"]
    assert result["audio"]["has_music"] is None
    assert result["audio"]["has_speech"] is None
    assert result["source"]["duration_seconds"] == pytest.approx(4, abs=0.05)
    assert not result["cache_hit"]
    folder = Path(result["report_path"]).parent
    with wave.open(str(folder / "analysis.wav"), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels()) == (22050, 1)
    attacks = result["events"]
    assert 3 <= len(attacks) <= 8
    for expected in (0.5, 1.5, 2.5, 3.5):
        assert min(abs(e["peak_time"] - expected) for e in attacks) < 0.1
    for event in attacks:
        assert event["label"] == "unknown" and event["confidence"] is None
        assert event["mixture_status"] == "mixed_track_not_source_separated"
        with wave.open(str(folder / event["clip_path"]), "rb") as clip:
            assert (clip.getframerate(), clip.getnchannels()) == (44100, 2)
            assert clip.getnframes() / clip.getframerate() == pytest.approx(
                event["clip_end"] - event["clip_start"],
                abs=1 / 44100,
            )
    assert any(abs(v["time"] - 1.5) < 0.04 for v in result["visual_events"])
    assert result["sync_candidates"]
    assert all(
        s["interpretation"] == "temporal_coincidence_not_causation"
        for s in result["sync_candidates"]
    )
    stored = json.loads((folder / "report.json").read_text("utf-8"))
    schema = json.loads((Path(__file__).parents[1] / "schemas/report-0.1.json").read_text("utf-8"))
    jsonschema.validate(stored, schema)
    assert (folder / "report.md").is_file()
    assert (folder / "original.mka").stat().st_size > 0


@pytest.mark.integration
def test_attack_limit_is_visible_in_saved_report_and_cache(media_files, tmp_path):
    full = analyze(media_files["video"], tmp_path, AnalysisOptions(max_events=200, visual=False))
    limited_options = AnalysisOptions(max_events=2, visual=False)
    limited = analyze(media_files["video"], tmp_path, limited_options)
    counts = limited["attack_detection"]
    assert counts["status"] == "completed"
    assert counts["candidates_total"] == full["attack_detection"]["returned"]
    assert counts["candidates_total"] >= 4
    assert counts["returned"] == counts["selected_by_limit"] == len(limited["events"]) == 2
    assert counts["omitted_by_limit"] == counts["candidates_total"] - 2
    assert counts["removed_by_timeline"] == 0
    assert all(e["label"] == "unknown" for e in limited["events"])
    folder = Path(limited["report_path"]).parent
    assert f"{counts['omitted_by_limit']} omitted by limit 2" in (folder / "report.md").read_text(
        "utf-8"
    )
    stored = json.loads((folder / "report.json").read_text("utf-8"))
    schema = json.loads((Path(__file__).parents[1] / "schemas/report-0.1.json").read_text("utf-8"))
    jsonschema.validate(stored, schema)
    cached = analyze(media_files["video"], tmp_path, limited_options)
    assert cached["cache_hit"] and cached["attack_detection"] == counts


@pytest.mark.integration
def test_dedup_options_and_content(media_files, tmp_path):
    options = AnalysisOptions(extract_only=True)
    first = analyze(media_files["video"], tmp_path, options)
    assert first["attack_detection"]["status"] == "not_requested"
    assert first["attack_detection"]["candidates_total"] is None
    duplicate = tmp_path / "renamed.mp4"
    duplicate.write_bytes(media_files["video"].read_bytes())
    second = analyze(duplicate, tmp_path, options)
    assert second["cache_hit"] and first["analysis_id"] == second["analysis_id"]
    other = analyze(duplicate, tmp_path, replace(options, sample_rate=16000))
    assert other["analysis_id"] != first["analysis_id"]


@pytest.mark.integration
def test_no_audio_is_valid_report(media_files, tmp_path):
    result = analyze(media_files["mute"], tmp_path)
    assert result["audio"]["present"] is False
    assert result["events"] == []
    assert result["audio"]["analysis_path"] is None
    assert result["attack_detection"]["status"] == "no_audio"
    assert result["attack_detection"]["returned"] is None
    assert "no audio" in " ".join(result["warnings"])


@pytest.mark.integration
def test_extract_duration_and_exact_sample_cut(media_files, tmp_path):
    output = tmp_path / "audio.wav"
    duration = extract_audio(media_files["video"], output, sample_rate=16000)
    assert duration == pytest.approx(4, abs=0.05)
    clip = tmp_path / "clip.wav"
    interval = cut_wav(output, clip, 0.25, 0.75)
    with wave.open(str(clip), "rb") as wav:
        assert wav.getnframes() == 8000
        assert wav.getframerate() == 16000
    assert interval == {"start": 0.25, "end": 0.75}


@pytest.mark.integration
def test_missing_and_invalid_media(ffmpeg, tmp_path):
    with pytest.raises(MediaError, match="does not exist"):
        analyze(tmp_path / "absent.mp4", tmp_path / "out")
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_text("This is not a video.")
    with pytest.raises(MediaError):
        analyze(corrupt, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr("dotears.media.shutil.which", lambda _: None)
    with pytest.raises(DependencyError, match="ffmpeg"):
        analyze(tmp_path / "file.mp4", tmp_path)


@pytest.mark.integration
def test_limits_reject_before_analysis(media_files):
    with pytest.raises(LimitError, match="Duration"):
        probe(media_files["video"], 1, 200 * 1024**2)
    with pytest.raises(LimitError, match="bytes"):
        probe(media_files["video"], 10, 100)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_duration": float("nan")},
        {"max_events": 0},
        {"sample_rate": 22050.5},
        {"operation_timeout": float("inf")},
        {"clip_before": -1},
        {"profile": "invented"},
        {"yamnet_model": "missing-model"},
        {"channels": 3},
    ],
)
def test_invalid_config(changes):
    with pytest.raises(ConfigurationError):
        AnalysisOptions(**changes).validate()


@pytest.mark.integration
@pytest.mark.parametrize("start,end", [(-1, 1), (2, 1), (0, 50), (float("nan"), 1)])
def test_invalid_cut(media_files, tmp_path, start, end):
    with pytest.raises(MediaError):
        cut_wav(media_files["wav"], tmp_path / "bad.wav", start, end)


@pytest.mark.integration
def test_transaction_cleanup_on_failure(media_files, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise MediaError("forced analysis failure")

    monkeypatch.setattr("dotears.core.audio.perceive", fail)
    with pytest.raises(MediaError, match="forced"):
        analyze(media_files["video"], tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.integration
def test_delayed_audio_uses_video_timeline(media_files, tmp_path, ffmpeg):
    source = tmp_path / "delayed.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(media_files["mute"]),
            "-itsoffset",
            "0.4",
            "-i",
            str(media_files["wav"]),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    report = analyze(source, tmp_path / "out", AnalysisOptions(visual=False))
    assert report["audio"]["timeline_offset_seconds"] == pytest.approx(0.4, abs=0.05)
    for time in (0.9, 1.9, 2.9, 3.9):
        assert min(abs(e["peak_time"] - time) for e in report["events"]) < 0.1
