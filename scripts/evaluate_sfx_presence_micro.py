"""Compare local attack-morphology additions using calibration authors only."""

from __future__ import annotations

import argparse
from pathlib import Path

import evaluate_sfx_presence_deep as evaluator
from evaluate_sfx_presence import fit_head
from evaluate_sfx_presence_rbf import fit_rbf

from dotears.sfx_micro import FEATURE_NAMES, SCHEMA, VERSION, read_policy
from dotears.sfx_presence import FEATURE_NAMES as BASE_NAMES
from dotears.sfx_presence_deep import FEATURE_NAMES as DEEP_NAMES
from dotears.sfx_presence_deep import score_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("linear", "rbf"), required=True)
    args = parser.parse_args()

    def fit_variant(rows, label, regularization, variant):
        indices = (
            list(range(len(BASE_NAMES))) + list(range(len(DEEP_NAMES), len(FEATURE_NAMES)))
            if variant == "semantic"
            else list(range(len(FEATURE_NAMES)))
        )
        if args.kind == "rbf":
            return fit_rbf(rows, label, regularization, variant, active_indices=indices)
        reduced = [
            {
                **r,
                "evidence": {
                    **r["evidence"],
                    "features": [r["evidence"]["features"][i] for i in indices],
                },
            }
            for r in rows
        ]
        head = fit_head(reduced, label, regularization)
        head.update(
            kind="linear",
            feature_count=len(indices),
            feature_indices=indices,
            representation=variant,
        )
        return head

    evaluator.FEATURE_NAMES = FEATURE_NAMES
    evaluator.SCHEMA = SCHEMA
    evaluator.VERSION = VERSION
    evaluator.fit_variant = fit_variant
    evaluator.score = lambda head, row: score_head(head, row["evidence"]["features"])
    evaluator.read_policy = read_policy
    evaluator.fit(args.root.resolve(), evaluate_test=False)


if __name__ == "__main__":
    main()
