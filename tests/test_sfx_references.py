import numpy as np
import pytest

from dotears.errors import ConfigurationError, MediaError
from dotears.sfx_references import rank_references


def reference(identity, label, group):
    return {"id": identity, "label": label, "group_id": group, "source_sha256": identity * 64}


def test_query_family_and_identical_source_cannot_vote_for_itself():
    rows = [
        reference("a", "whoosh", "pack1"),
        reference("b", "whoosh", "pack1"),
        reference("c", "notification", "pack2"),
        reference("d", "ui_click", "pack3"),
    ]
    result = rank_references(
        np.array([1, 0]),
        np.array([[1, 0], [1, 0], [0, 1], [1, 0]]),
        rows,
        excluded_groups={"pack1"},
        excluded_source_hashes={"d" * 64},
    )
    assert result["available_labels"] == ["notification"]
    assert [row["id"] for row in result["nearest_references"]] == ["c"]
    assert result["label"] == "unknown" and result["confidence"] is None


def test_duplicating_a_variant_does_not_outweigh_another_source_family():
    rows = [reference("a", "whoosh", "pack1"), reference("b", "whoosh", "pack2")]
    first = rank_references(np.array([1, 0]), np.array([[1, 0], [0, 1]]), rows)
    repeated = rank_references(
        np.array([1, 0]),
        np.array([[1, 0], [0, 1], [1, 0]]),
        rows + [reference("c", "whoosh", "pack1")],
    )
    assert first["best_match"]["cosine_similarity"] == repeated["best_match"]["cosine_similarity"]
    assert repeated["best_match"]["reference_families"] == 2


def test_missing_positive_class_remains_unavailable_and_empty_bank_abstains():
    rows = [reference("a", "whoosh", "only-family")]
    result = rank_references(
        np.array([1, 0]), np.array([[1, 0]]), rows, excluded_groups={"only-family"}
    )
    assert result["best_match"] is None and result["rankings"] == []
    assert result["available_labels"] == [] and result["label"] == "unknown"


def test_background_examples_can_win_without_becoming_a_confirmed_sfx_label():
    rows = [reference("a", "notification", "tones"), reference("b", "background", "music")]
    result = rank_references(np.array([0, 1]), np.array([[1, 0], [0, 1]]), rows)
    assert result["best_match"]["label"] == "background"
    assert result["label"] == "unknown" and result["confidence"] is None


@pytest.mark.parametrize(
    "query,matrix", [([np.nan, 1], [[1, 0]]), ([1, 0], [[0, 0]]), ([1, 0], [[1, np.inf]])]
)
def test_broken_embeddings_are_rejected(query, matrix):
    with pytest.raises(MediaError):
        rank_references(np.array(query), np.array(matrix), [reference("a", "whoosh", "p")])


def test_duplicate_or_unknown_reference_labels_are_rejected():
    row = reference("a", "whoosh", "p")
    with pytest.raises(ConfigurationError):
        rank_references(np.array([1, 0]), np.array([[1, 0], [0, 1]]), [row, row])
    with pytest.raises(ConfigurationError):
        rank_references(np.array([1, 0]), np.array([[1, 0]]), [reference("a", "uncertain", "p")])
