from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from dotears.audiosep import file_digest
from dotears.errors import ConfigurationError
from dotears.sfx_match import (
    RATE,
    SCHEMA,
    VERSION,
    ReferenceMatcher,
    make_template,
    scan_reference_report,
)


def bank(tmp_path):
    rng = np.random.default_rng(17)
    audio = np.zeros(3 * RATE, dtype=np.float32)
    audio[RATE : RATE + 640] = rng.normal(0, 0.15, 640) * np.exp(-np.arange(640) / 200)
    template, offset = make_template(audio, 1.005)
    np.savez(tmp_path / "bank.npz", click=template)
    policy = {
        "schema_version": SCHEMA,
        "version": VERSION,
        "bank_path": "bank.npz",
        "bank_sha256": file_digest(tmp_path / "bank.npz"),
        "templates": [
            {
                "id": "click",
                "label": "ui_click",
                "peak_offset": offset,
                "enabled": True,
                "threshold": 0.8,
            }
        ],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    return path, audio


def test_known_waveform_survives_gain_polarity_offset_and_independent_background(tmp_path):
    policy, source = bank(tmp_path)
    matcher = ReferenceMatcher(policy)
    shifted = np.roll(source, 321)
    noise = np.random.default_rng(212).normal(0, 0.003, len(source))
    result = matcher.analyze(-0.4 * shifted + noise, 1.005)
    assert result["label"] == "ui_click" and result["reference_score"] > 0.9
    assert abs(result["matched_peak_relative"] - (1.005 + 321 / RATE)) < 1 / RATE
    assert result["confidence"] is None
    assert result["scope"] == "known_waveform_only_not_unseen_effect_classification"
    assert not result["human_confirmed_in_video"]


def test_unrelated_waveform_silence_and_out_of_tolerance_remain_unknown(tmp_path):
    policy, source = bank(tmp_path)
    matcher = ReferenceMatcher(policy)
    noise = np.random.default_rng(23).normal(0, 0.1, len(source))
    for values in (noise, source * 0, np.roll(source, RATE // 2)):
        assert matcher.analyze(values, 1.005)["label"] == "unknown"
    with pytest.raises(ConfigurationError):
        matcher.analyze(noise, float("nan"))


def test_disabled_or_wrong_type_winner_cannot_fall_back_to_enabled_reference(tmp_path):
    policy, source = bank(tmp_path)
    matcher = ReferenceMatcher(policy)
    scores = matcher.scores(source, 1.005)
    scores[0]["enabled"] = False
    scores.append(
        {**scores[0], "reference_id": "wrong", "enabled": True, "label": "whoosh", "score": 0.81}
    )
    assert matcher.decide(scores)["label"] == "unknown"
    scores[0].update(enabled=True, label="other")
    assert matcher.decide(scores)["label"] == "unknown"
    scores[0].update(label="ui_click")
    scores[1]["score"] = 0.97
    assert matcher.decide(scores)["label"] == "unknown"


def test_reference_bank_tampering_is_rejected(tmp_path):
    policy, _ = bank(tmp_path)
    with (tmp_path / "bank.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ConfigurationError, match="changed"):
        ReferenceMatcher(policy)


def test_reference_cli_report_preserves_original_audio_and_binds_timeline(tmp_path):
    policy, audio = bank(tmp_path)
    native = tmp_path / "native.wav"
    with wave.open(str(native), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((audio * 30000).astype("<i2").tobytes())
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "analysis_id": "test",
                "source": {"sha256": "source"},
                "audio": {
                    "native_path": "native.wav",
                    "timeline_offset_seconds": 10.0,
                    "duration_seconds": 3.0,
                },
                "events": [{"id": "attack_001", "category": "audio_attack", "peak_time": 11.005}],
            }
        )
    )
    originals = (report.read_bytes(), native.read_bytes())
    result = scan_reference_report(report, policy, tmp_path / "out")
    data = json.loads((tmp_path / "out/sfx-match.json").read_text("utf-8"))
    assert result["accepted"] == 1
    assert data["events"][0]["label"] == "ui_click"
    assert data["events"][0]["peak_time"] == 11.005
    assert data["source_sha256"] == "source" and data["analysis_id"] == "test"
    assert not data["video_annotations_used_in_inference"]
    assert originals == (report.read_bytes(), native.read_bytes())
    with pytest.raises(ConfigurationError, match="exists"):
        scan_reference_report(report, policy, tmp_path / "out")
