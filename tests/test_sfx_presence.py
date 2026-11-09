from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from dotears.errors import ConfigurationError
from dotears.sfx_detect_report import scan_raw_report
from dotears.sfx_presence import FEATURE_NAMES, SCHEMA, TYPES, VERSION, decide, read_policy


def policy():
    return {
        "schema_version": SCHEMA,
        "version": VERSION,
        "feature_names": FEATURE_NAMES,
        "models": {"test": "model"},
        "heads": {
            label: {
                "mean": [0.0] * len(FEATURE_NAMES),
                "scale": [1.0] * len(FEATURE_NAMES),
                "coefficients": [0.0] * len(FEATURE_NAMES),
                "intercept": 10.0,
                "threshold": 1.0,
                "enabled": True,
            }
            for label in TYPES
        },
    }


def evidence(scores=None):
    return {
        "features": [0.0] * len(FEATURE_NAMES),
        "rms": 0.1,
        "raw_target_scores": scores or {"whoosh": 0.1, "notification": 0.3, "ui_click": 0.05},
    }


def test_disabled_raw_winner_cannot_be_replaced_by_a_strong_enabled_whoosh():
    frozen = policy()
    frozen["heads"]["notification"]["enabled"] = False
    frozen["heads"]["whoosh"]["intercept"] = 1000.0
    result = decide(frozen, evidence({"whoosh": -0.055, "notification": 0.215, "ui_click": -0.008}))
    assert result["label"] == "unknown"
    assert (
        "raw_audio_supports_another_type_or_is_ambiguous"
        in result["candidates"]["whoosh"]["rejection_reasons"]
    )
    frozen["heads"]["notification"]["enabled"] = True
    result = decide(frozen, evidence())
    assert result["label"] == "notification"
    assert result["confidence"] is None and not result["human_confirmed"]


@pytest.mark.parametrize(
    "scores",
    [
        {"whoosh": 0.2, "notification": 0.2, "ui_click": 0.1},
        {"whoosh": -0.1, "notification": -0.2, "ui_click": -0.3},
    ],
)
def test_ambiguous_or_negative_raw_support_stays_unknown(scores):
    assert decide(policy(), evidence(scores))["label"] == "unknown"


def test_quiet_audio_and_below_threshold_remain_unknown():
    item = evidence()
    item["rms"] = 0
    assert decide(policy(), item)["label"] == "unknown"
    frozen = policy()
    frozen["heads"]["notification"]["threshold"] = 11
    assert decide(frozen, evidence())["label"] == "unknown"
    with pytest.raises(ConfigurationError):
        decide(frozen, evidence({"whoosh": 0.2}))


def test_policy_rejects_different_features_and_nonfinite_or_invalid_parameters(tmp_path):
    path = tmp_path / "policy.json"
    frozen = policy()
    path.write_text(json.dumps(frozen))
    assert read_policy(path)["version"] == VERSION
    frozen["heads"]["whoosh"]["scale"][0] = 0
    path.write_text(json.dumps(frozen))
    with pytest.raises(ConfigurationError):
        read_policy(path)
    frozen = policy()
    frozen["feature_names"] = FEATURE_NAMES[:-1]
    path.write_text(json.dumps(frozen))
    with pytest.raises(ConfigurationError):
        read_policy(path)


def test_raw_cli_sidecar_preserves_timeline_source_and_all_three_competitors(tmp_path, monkeypatch):
    source = tmp_path / "native.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(32000)
        handle.writeframes(np.full(96000, 1000, dtype="<i2").tobytes())
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "analysis_id": "a",
                "source": {"sha256": "source"},
                "audio": {
                    "native_path": "native.wav",
                    "timeline_offset_seconds": 10.0,
                    "duration_seconds": 3.0,
                },
                "events": [
                    {"id": f"attack_{i:03}", "category": "audio_attack", "peak_time": p}
                    for i, p in enumerate((10.0, 11.0, 13.0), 1)
                ],
            }
        )
    )
    before = report.read_bytes(), source.read_bytes()
    frozen = policy()
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(frozen))
    calls = []

    class Engine:
        identity = frozen["models"]
        counters = {"mock": 0}

        def __init__(self, *_args):
            pass

        def analyze(self, audio, peak):
            calls.append(peak)
            return evidence()

    monkeypatch.setattr("dotears.sfx_detect_report.RawSFXEngine", Engine)
    result = scan_raw_report(report, tmp_path, tmp_path, path, tmp_path / "out", tmp_path / "cache")
    output = json.loads(Path(result["sfx_report_path"]).read_text("utf-8"))
    assert len(output["events"]) == 3 and calls == [0.0, 1.0, 3.0]
    assert all(
        r["label"] == "notification" and set(r["candidates"]) == set(TYPES)
        for r in output["events"]
    )
    assert not output["separation_used_for_type_decision"]
    assert before == (report.read_bytes(), source.read_bytes())
    with pytest.raises(ConfigurationError, match="exists"):
        scan_raw_report(report, tmp_path, tmp_path, path, tmp_path / "out", tmp_path / "cache")
    frozen["models"] = {"wrong": "model"}
    path.write_text(json.dumps(frozen))
    with pytest.raises(ConfigurationError, match="different models"):
        scan_raw_report(report, tmp_path, tmp_path, path, tmp_path / "new", tmp_path / "cache")
    assert not (tmp_path / "new").exists()
