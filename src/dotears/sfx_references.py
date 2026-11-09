"""Experimental audio-reference ranking; no calibrated classifier decisions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from dotears.errors import ConfigurationError, MediaError

REFERENCE_LABELS = {"whoosh", "notification", "ui_click", "other", "background"}
METHOD = "clap_equal_family_then_class_centroid_cosine_v1"


def unit_vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise MediaError("Expected a non-empty finite embedding matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise MediaError("Cannot normalize a zero audio embedding.")
    return values / norms


def rank_references(
    query: np.ndarray,
    embeddings: np.ndarray,
    references: list[dict],
    *,
    excluded_groups: set[str] | None = None,
    excluded_source_hashes: set[str] | None = None,
) -> dict:
    """Rank class prototypes with equal family weight; exclude before aggregation.

    Callers evaluating an existing source must exclude its entire family and source hash.
    Returned scores are audio similarities, never probabilities or confirmed event labels.
    """
    matrix = unit_vectors(embeddings)
    query = np.asarray(query, dtype=np.float64)
    if query.shape != (matrix.shape[1],) or len(references) != len(matrix):
        raise ConfigurationError("Reference IDs and embedding dimensions do not match.")
    query = unit_vectors(query[None])[0]
    excluded_groups = excluded_groups or set()
    excluded_source_hashes = excluded_source_hashes or set()
    groups = defaultdict(list)
    eligible = []
    seen = set()
    for index, row in enumerate(references):
        if (
            row.get("label") not in REFERENCE_LABELS
            or not row.get("group_id")
            or not row.get("source_sha256")
            or not row.get("id")
            or row["id"] in seen
        ):
            raise ConfigurationError("Invalid or duplicate labeled audio reference.")
        seen.add(row["id"])
        if row["group_id"] in excluded_groups or row["source_sha256"] in excluded_source_hashes:
            continue
        groups[(row["label"], row["group_id"])].append(matrix[index])
        eligible.append((row, matrix[index]))
    by_label = defaultdict(list)
    for (label, _group), vectors in groups.items():
        by_label[label].append(unit_vectors(np.mean(vectors, axis=0)[None])[0])
    rankings = []
    for label, vectors in by_label.items():
        centroid = unit_vectors(np.mean(vectors, axis=0)[None])[0]
        rankings.append(
            {
                "label": label,
                "cosine_similarity": round(float(query @ centroid), 6),
                "reference_families": len(vectors),
                "reference_examples": sum(row["label"] == label for row, _ in eligible),
            }
        )
    rankings.sort(key=lambda row: (-row["cosine_similarity"], row["label"]))
    nearest = sorted(
        [
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "label": row["label"],
                "cosine_similarity": round(float(query @ vector), 6),
            }
            for row, vector in eligible
        ],
        key=lambda row: (-row["cosine_similarity"], row["id"]),
    )[:3]
    return {
        "label": "unknown",
        "confidence": None,
        "status": "experimental_reference_hypotheses_only",
        "method": METHOD,
        "score_kind": "uncalibrated_audio_cosine_similarity_not_probability",
        "best_match": rankings[0] if rankings else None,
        "rankings": rankings,
        "nearest_references": nearest,
        "excluded_groups": sorted(excluded_groups),
        "excluded_source_hashes": sorted(excluded_source_hashes),
        "available_labels": sorted(by_label),
        "abstention_reason": "No independently validated automatic-label decision rule.",
    }
