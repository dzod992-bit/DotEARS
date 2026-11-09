import hashlib
import json
import wave

import numpy as np
import pytest
import soundfile as sf

from dotears.errors import ConfigurationError
from dotears.separate_report import separate_report_window


def prepare_report(root):
    native = root / "native.wav"
    samples = (np.sin(np.arange(44100 * 4) / 44100 * 2 * np.pi * 500) * 10000).astype("<i2")
    with wave.open(str(native), "wb") as wav:
        wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        wav.writeframes(samples.tobytes())
    path = root / "report.json"
    path.write_text(
        json.dumps(
            dict(
                analysis_id="test-analysis",
                source=dict(sha256="a" * 64),
                audio=dict(
                    native_path="native.wav", duration_seconds=4.0, timeline_offset_seconds=0.0
                ),
                events=[
                    dict(id="attack_001", category="audio_attack", peak_time=2.0, label="unknown")
                ],
            )
        ),
        "utf-8",
    )
    return path


def test_separate_keeps_original_report_and_unconfirmed_query_provenance(tmp_path, monkeypatch):
    report = prepare_report(tmp_path)
    before = report.read_bytes()

    class FakeSeparator:
        def __init__(self, folder, threads):
            self.load_evidence = {"checkpoint_sha256": "b" * 64}

        def separate(self, signal, prompt):
            assert len(signal) == 96000 and prompt == "notification chime"
            return signal * 0.5

    monkeypatch.setattr("dotears.audiosep.LocalAudioSep", FakeSeparator)
    output = tmp_path / "separated"
    result = separate_report_window(
        report, "attack_001", "notification chime", tmp_path / "model", output
    )
    assert report.read_bytes() == before
    assert result["label"] == "unknown" and result["query_is_not_sound_recognition"]
    assert result["window_start"] == 1 and result["window_end"] == 4
    source, rate = sf.read(output / "mixture.wav")
    estimate, _ = sf.read(output / "estimated.wav")
    assert rate == 32000 and np.allclose(estimate, source * 0.5)
    saved = json.loads((output / "separation.json").read_text("utf-8"))
    assert saved["source_report_sha256"] == hashlib.sha256(before).hexdigest()
    with pytest.raises(ConfigurationError, match="output exists"):
        separate_report_window(
            report, "attack_001", "notification chime", tmp_path / "model", output
        )


def test_separation_rejects_native_path_escape_and_unknown_event_before_model_load(
    tmp_path, monkeypatch
):
    report = prepare_report(tmp_path)

    def fail_if_model_loaded(*a, **k):
        raise AssertionError("Model must not be loaded for an invalid request")

    monkeypatch.setattr("dotears.audiosep.LocalAudioSep", fail_if_model_loaded)
    with pytest.raises(ConfigurationError, match="existing audio_attack"):
        separate_report_window(
            report, "not-an-event", "chime", tmp_path / "model", tmp_path / "out"
        )
    data = json.loads(report.read_text("utf-8"))
    data["audio"]["native_path"] = "../outside.wav"
    report.write_text(json.dumps(data), "utf-8")
    with pytest.raises(ConfigurationError, match="outside its report folder"):
        separate_report_window(report, "attack_001", "chime", tmp_path / "model", tmp_path / "out")
