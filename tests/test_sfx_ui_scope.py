from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from test_sfx_match import bank

from dotears.audiosep import file_digest
from dotears.errors import ConfigurationError
from dotears.sfx_match import BROAD_SCHEMA, RATE, VERSION, ReferenceMatcher, make_template
from dotears.sfx_taxonomy import UI_TAXONOMY, broad_ui_label


def migration():
    spec = importlib.util.spec_from_file_location(
        "update_ui_scope", Path(__file__).resolve().parents[1] / "scripts/update_sfx_ui_scope.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.migrate


def test_broad_ui_membership_preserves_notifications_and_negative_reviews():
    for label in ("ui_click", "ui_tone", "ui_sound"):
        assert broad_ui_label(label) == "ui_sound"
    for label in ("notification", "other", "whoosh", "background", "unknown"):
        assert broad_ui_label(label) == label
    with pytest.raises(ConfigurationError):
        broad_ui_label("unreviewed")


def test_scope_migration_preserves_bank_scores_thresholds_and_disabled_patterns(tmp_path):
    source, audio = bank(tmp_path)
    original = json.loads(source.read_text())
    original["templates"].append(
        {**original["templates"][0], "id": "disabled", "label": "notification", "enabled": False}
    )
    # Reuse the exact waveform as a conflicting role to ensure it cannot be renamed to UI.
    arrays = np.load(tmp_path / "bank.npz", allow_pickle=False)
    np.savez(tmp_path / "bank.npz", click=arrays["click"], disabled=arrays["click"])
    original["bank_sha256"] = file_digest(tmp_path / "bank.npz")
    source.write_text(json.dumps(original))
    before = source.read_bytes(), (tmp_path / "bank.npz").read_bytes()
    old = ReferenceMatcher(source)
    target = migration()(source, tmp_path / "broad")
    new = ReferenceMatcher(target)
    assert new.broad_ui and new.policy["label_taxonomy"] == UI_TAXONOMY
    assert new.policy["templates"][0]["label"] == "ui_sound"
    assert new.policy["templates"][1]["label"] == "notification"
    assert new.policy["templates"][1]["enabled"] is False
    assert new.policy["bank_sha256"] == original["bank_sha256"]
    assert [r["score"] for r in old.scores(audio, 1.005)] == [
        r["score"] for r in new.scores(audio, 1.005)
    ]
    assert new.analyze(audio, 1.005)["label"] == "unknown"
    assert before == (source.read_bytes(), (tmp_path / "bank.npz").read_bytes())


def test_broad_policy_needs_explicit_taxonomy_and_accepts_ui_label(tmp_path):
    source, audio = bank(tmp_path)
    policy = json.loads(source.read_text())
    policy["schema_version"] = BROAD_SCHEMA
    policy["templates"][0]["label"] = "ui_sound"
    source.write_text(json.dumps(policy))
    with pytest.raises(ConfigurationError, match="taxonomy"):
        ReferenceMatcher(source)
    policy["label_taxonomy"] = UI_TAXONOMY
    source.write_text(json.dumps(policy))
    assert ReferenceMatcher(source).analyze(audio, 1.005)["label"] == "ui_sound"


def test_identical_beep_with_two_roles_remains_unknown():
    scores = [
        {
            "reference_id": "beep-ui",
            "label": "ui_sound",
            "score": 0.99,
            "matched_peak_relative": 1.0,
            "threshold": 0.65,
            "enabled": True,
        },
        {
            "reference_id": "beep-alert",
            "label": "notification",
            "score": 0.99,
            "matched_peak_relative": 1.0,
            "threshold": 0.65,
            "enabled": True,
        },
    ]
    result = ReferenceMatcher.decide(scores)
    assert result["label"] == "unknown"
    assert "ambiguous_reference_type" in result["rejection_reasons"]


def test_known_tonal_confirmation_is_allowed_in_broad_ui_scope(tmp_path):
    # A tonal feedback signal is intentionally unlike the old noise-click fixture.
    times = np.arange(4800) / RATE
    tone = 0.2 * np.sin(2 * np.pi * (900 * times + 2300 * times**2)) * np.hanning(len(times))
    source = np.zeros(3 * RATE, dtype=np.float32)
    source[RATE : RATE + len(tone)] = tone
    values, offset = make_template(source, 1.075)
    np.savez(tmp_path / "bank.npz", tone=values)
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": BROAD_SCHEMA,
                "version": VERSION,
                "label_taxonomy": UI_TAXONOMY,
                "bank_path": "bank.npz",
                "bank_sha256": file_digest(tmp_path / "bank.npz"),
                "templates": [
                    {
                        "id": "tone",
                        "label": "ui_sound",
                        "peak_offset": offset,
                        "enabled": True,
                        "threshold": 0.8,
                    }
                ],
            }
        )
    )
    background = np.random.default_rng(402).normal(0, 0.002, len(source))
    result = ReferenceMatcher(path).analyze(source + background, 1.075)
    assert result["label"] == "ui_sound" and result["reference_score"] > 0.95
