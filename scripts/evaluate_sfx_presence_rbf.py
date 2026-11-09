"""Bounded nonlinear raw-audio candidates; export only NumPy inference parameters."""

from __future__ import annotations

import argparse

import evaluate_sfx_presence_deep as evaluator
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from dotears.errors import MediaError
from dotears.sfx_events import balanced_source_weights
from dotears.sfx_presence import FEATURE_NAMES as BASE_NAMES
from dotears.sfx_presence_deep import FEATURE_NAMES, score_head


def fit_rbf(rows, label, regularization, variant, active_indices=None, sample_weights=None):
    dimensions = len(BASE_NAMES) if variant == "semantic" else len(FEATURE_NAMES)
    if active_indices is not None:
        dimensions = len(active_indices)
        x = np.array(
            [[r["evidence"]["features"][i] for i in active_indices] for r in rows], dtype=float
        )
    else:
        x = np.array([r["evidence"]["features"][:dimensions] for r in rows], dtype=float)
    y = np.array([r["label"] == label for r in rows])
    weights = (
        balanced_source_weights(rows)
        if sample_weights is None
        else np.array(sample_weights, dtype=float, copy=True)
    )
    source_units = len({r["source_sha256"] for r in rows})
    for value in (False, True):
        weights[y == value] *= source_units / 2 / weights[y == value].sum()
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(x, sample_weight=weights)
        model = SVC(
            C=regularization, kernel="rbf", gamma="scale", tol=1e-4, max_iter=10000, cache_size=128
        )
        model.fit(scaler.transform(x), y, sample_weight=weights)
    if model.fit_status_ != 0:
        raise MediaError("RBF presence candidate did not converge.")
    result = {
        "kind": "rbf",
        "feature_count": dimensions,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "support_vectors": model.support_vectors_.tolist(),
        "dual_coefficients": model.dual_coef_[0].tolist(),
        "gamma": float(model._gamma),
        "intercept": float(model.intercept_[0]),
        "threshold": -1e100,
        "enabled": True,
        "representation": variant,
        "C": regularization,
        "training_source_units": source_units,
    }
    if active_indices is not None:
        result["feature_indices"] = active_indices
    return result


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    evaluator.fit_variant = fit_rbf
    evaluator.score = lambda head, row: score_head(head, row["evidence"]["features"])
    evaluator.fit(args.root.resolve(), evaluate_test=False)


if __name__ == "__main__":
    main()
