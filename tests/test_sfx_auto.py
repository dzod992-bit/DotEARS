from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from dotears.errors import ConfigurationError
from dotears.sfx import PROMPTS
from dotears.sfx_auto import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    POLICY_SCHEMA,
    QUERIES,
    CandidateEngine,
    decision,
    focus_window,
    json_hash,
    pcm_hash,
    read_policy,
    scan_report,
)


def policy():
    return {
        "schema_version": POLICY_SCHEMA,
        "queries": QUERIES,
        "feature_names": FEATURE_NAMES,
        "feature_version": FEATURE_VERSION,
        "prompt_sha256": json_hash(PROMPTS),
        "heads": {
            label: {
                "mean": [0.0] * len(FEATURE_NAMES),
                "scale": [1.0] * len(FEATURE_NAMES),
                "coefficients": [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
                "intercept": 0.0,
                "threshold": 1.0,
                "score_scale": 1.0,
                "enabled": True,
            }
            for label in QUERIES
        },
    }


def candidates(scores=(0.0, 0.0, 0.0)):
    return {
        label: {
            "features": [score] + [0.0] * (len(FEATURE_NAMES) - 1),
            "raw_rms": 0.1,
            "estimate_rms": 0.1,
            "energy_ratio_db": 0.0,
            "estimated_target_cosine": 0.3,
            "estimated_best_control_cosine": 0.2,
        }
        for label, score in zip(QUERIES, scores, strict=True)
    }


def test_rejection_and_ambiguity_preserve_unknown_without_fake_confidence():
    assert decision(policy(), candidates())["label"] == "unknown"
    ambiguous = decision(policy(), candidates((2.0, 1.9, 0.0)))
    assert ambiguous["label"] == "unknown"
    assert ambiguous["reason"] == "multiple_candidates_have_similar_support"
    passed = decision(policy(), candidates((3.0, 1.1, 0.0)))
    assert passed["label"] == "whoosh"
    assert passed["confidence"] is None
    assert passed["status"] == "provisional_model_type"
    assert not passed["human_confirmed"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_rms", 0.0),
        ("estimate_rms", 0.0),
        ("energy_ratio_db", -41.0),
        ("estimated_best_control_cosine", 0.8),
    ],
)
def test_weak_or_background_like_estimate_cannot_accept_even_a_high_score(field, value):
    rows = candidates((100.0, 0.0, 0.0))
    rows["whoosh"][field] = value
    assert decision(policy(), rows)["label"] == "unknown"


def test_disabled_class_is_never_unmasked_as_a_named_sound():
    frozen = policy()
    frozen["heads"]["whoosh"]["enabled"] = False
    assert decision(frozen, candidates((100.0, 0.0, 0.0)))["label"] == "unknown"
    with pytest.raises(ConfigurationError):
        decision(frozen, {"whoosh": candidates()["whoosh"]})


def test_policy_rejects_incompatible_prompts_and_nonfinite_or_invalid_parameters(tmp_path):
    path = tmp_path / "policy.json"
    frozen = policy()
    path.write_text(json.dumps(frozen))
    assert read_policy(path)["queries"] == QUERIES
    frozen["queries"] = {**QUERIES, "whoosh": "invented prompt"}
    path.write_text(json.dumps(frozen))
    with pytest.raises(ConfigurationError):
        read_policy(path)
    frozen = policy()
    frozen["heads"]["notification"]["scale"][0] = 0
    path.write_text(json.dumps(frozen))
    with pytest.raises(ConfigurationError):
        read_policy(path)


def test_audio_keys_bind_samples_rate_and_shape_and_focus_respects_edge_peaks():
    signal = np.arange(32000, dtype=np.float32) / 32000
    assert pcm_hash(signal) != pcm_hash(signal, 48000)
    assert pcm_hash(signal) != pcm_hash(signal[:-1])
    changed = signal.copy()
    changed[100] += 0.001
    assert pcm_hash(changed) != pcm_hash(signal)
    assert len(focus_window(signal, 0.0)) == 17600
    assert len(focus_window(signal, 1.0)) == 4800
    with pytest.raises(ConfigurationError):
        focus_window(signal, 1.1)


def test_engine_checks_all_three_queries_without_accepting_a_ground_truth_label(monkeypatch):
    engine = CandidateEngine.__new__(CandidateEngine)
    calls = []

    def separate(waveform, prompt):
        calls.append(prompt)
        return waveform.copy(), Path("mock.wav")

    engine.separate = separate
    engine.embedding = lambda _: (np.ones(512) / np.sqrt(512), np.zeros(len(PROMPTS)))
    monkeypatch.setattr("dotears.sfx_auto.file_digest", lambda _: "mock-sha")
    result = engine.analyze_window(np.full(96000, 0.1, dtype=np.float32), 1.0)
    assert calls == list(QUERIES.values())
    assert set(result) == set(QUERIES)
    assert all(len(row["features"]) == len(FEATURE_NAMES) for row in result.values())
    with pytest.raises(TypeError):
        engine.analyze_window(np.ones(96000), 1.0, human_label="whoosh")


def test_scan_preserves_source_and_uses_exact_policy_and_all_events(tmp_path, monkeypatch):
    import soundfile as sf

    original = tmp_path / "native.wav"
    with wave.open(str(original), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(32000)
        audio.writeframes(np.full(96000, 1000, dtype="<i2").tobytes())
    estimate = tmp_path / "estimate.wav"
    sf.write(estimate, np.full(96000, 0.03), 32000, subtype="FLOAT")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "analysis_id": "test",
                "source": {"sha256": "source-hash"},
                "audio": {
                    "native_path": "native.wav",
                    "timeline_offset_seconds": 10.0,
                    "duration_seconds": 3.0,
                },
                "events": [
                    {"id": f"attack_{i:03}", "category": "audio_attack", "peak_time": peak}
                    for i, peak in enumerate((10.0, 11.0, 13.0), 1)
                ],
            }
        )
    )
    before = report.read_bytes(), original.read_bytes()
    frozen = policy()
    frozen["models"] = {"mock": "model-hash"}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(frozen))
    calls = []

    class MockEngine:
        identity = frozen["models"]
        counters = {"mock": 0}

        def __init__(self, *_args):
            pass

        def analyze_window(self, _waveform, peak):
            calls.append(peak)
            evidence = candidates((3.0, 0.0, 0.0))
            for row in evidence.values():
                row["estimate_path"] = str(estimate)
            return evidence

    monkeypatch.setattr("dotears.sfx_auto.CandidateEngine", MockEngine)
    output = tmp_path / "scan"
    result = scan_report(report, tmp_path, tmp_path, policy_path, output, tmp_path / "cache")
    written = json.loads((output / "sfx-auto.json").read_text("utf-8"))
    assert result["events"] == 3
    assert calls == [0.0, 1.0, 3.0]
    assert all(row["label"] == "whoosh" for row in written["events"])
    assert [row["event_id"] for row in written["events"]] == [
        "attack_001",
        "attack_002",
        "attack_003",
    ]
    assert before == (report.read_bytes(), original.read_bytes())
    with pytest.raises(ConfigurationError, match="exists"):
        scan_report(report, tmp_path, tmp_path, policy_path, output, tmp_path / "cache")
    frozen["models"] = {"different": "model-hash"}
    policy_path.write_text(json.dumps(frozen))
    with pytest.raises(ConfigurationError, match="different model"):
        scan_report(report, tmp_path, tmp_path, policy_path, tmp_path / "new", tmp_path / "cache")
    assert not (tmp_path / "new").exists()
