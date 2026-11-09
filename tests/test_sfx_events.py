import numpy as np
import pytest

from dotears.errors import ConfigurationError
from dotears.sfx_events import (
    balanced_source_weights,
    fit_linear_head,
    predict_linear_head,
    suppress_prefix_background,
    training_indices,
)


def test_foreground_mask_preserves_length_silence_and_finite_amplitude():
    silence = np.zeros(11200)
    assert np.array_equal(suppress_prefix_background(silence), silence)
    with pytest.raises(ConfigurationError):
        suppress_prefix_background(np.array([np.nan] * 11200))
    with pytest.raises(ConfigurationError):
        suppress_prefix_background(silence, sample_rate=8000)


def test_prefix_mask_reduces_steady_background_while_retaining_a_later_transient():
    t = np.arange(11200) / 16000
    background = 0.1 * np.sin(2 * np.pi * 500 * t)
    pulse = np.zeros_like(t)
    pulse[2400:3200] = 0.5 * np.sin(2 * np.pi * 3000 * t[:800]) * np.hanning(800)
    output = suppress_prefix_background(background + pulse)

    def rms(x):
        return np.sqrt(np.mean(x**2))

    assert rms(output[1000:1500]) < rms(background[1000:1500]) * 0.3
    assert rms(output[2500:3100]) > rms(pulse[2500:3100]) * 0.7
    assert len(output) == len(t) and np.max(np.abs(output)) <= 1


def test_all_synthetic_views_of_query_family_and_source_are_excluded():
    rows = [
        dict(family="A", source_sha256="source1", audio_sha256="clean"),
        dict(family="A", source_sha256="source2", audio_sha256="mix1"),
        dict(family="B", source_sha256="source1", audio_sha256="mix2"),
        dict(family="B", source_sha256="source3", audio_sha256="clean"),
        dict(family="B", source_sha256="source4", audio_sha256="different"),
    ]
    assert training_indices(rows, rows[0]).tolist() == [4]


def test_augmentation_does_not_change_total_class_or_family_weight():
    rows = [
        dict(human_label="whoosh", family="a", source_sha256="1"),
        dict(human_label="whoosh", family="b", source_sha256="2"),
        dict(human_label="ui_click", family="c", source_sha256="3"),
    ]
    many = rows + [rows[0].copy() for _ in range(9)]
    weights = balanced_source_weights(many)
    assert weights[
        [i for i, r in enumerate(many) if r["human_label"] == "whoosh"]
    ].sum() == pytest.approx(weights[2])
    assert weights[[i for i, r in enumerate(many) if r["family"] == "a"]].sum() == pytest.approx(
        weights[1]
    )


def test_explicit_broad_taxonomy_weights_do_not_require_claiming_human_role_labels():
    rows = [
        dict(label="ui_sound", family="a", source_sha256="1"),
        dict(label="notification", family="b", source_sha256="2"),
    ]
    weights = balanced_source_weights(
        rows, label_key="label", allowed_labels={"ui_sound", "notification"}
    )
    assert weights.tolist() == pytest.approx([1.0, 1.0])
    assert all("human_label" not in row for row in rows)


def test_exported_linear_head_predicts_without_pickle_and_keeps_labels_unconfirmed():
    rng = np.random.default_rng(43)
    features, rows = [], []
    for i, label in enumerate(("whoosh", "ui_click", "background")):
        for j in range(4):
            features.append(np.eye(3)[i] * 5 + rng.normal(0, 0.1, 3))
            rows.append(
                dict(
                    human_label=label,
                    family=label,
                    source_sha256=f"{i}-{j}",
                    audio_sha256=f"{i}-{j}",
                )
            )
    features = np.array(features)
    head = fit_linear_head(features, rows)
    result = predict_linear_head(head, features)
    assert [r["best_match"]["label"] for r in result] == [r["human_label"] for r in rows]
    assert all(r["label"] == "unknown" and r["confidence"] is None for r in result)
    head["scale"][0] = 0
    with pytest.raises(ConfigurationError):
        predict_linear_head(head, features)
