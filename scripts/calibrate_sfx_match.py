"""Build a known-waveform bank and validate it on explicitly prepared local mixtures.

This is source-identity validation, not unseen-source semantic classification.
Use new output folders; calibration and test corpus manifests must be frozen first.
"""

from __future__ import annotations

import argparse
import copy
import json
import socket
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from dotears.audiosep import file_digest
from dotears.sfx_auto import pcm_hash
from dotears.sfx_match import BROAD_SCHEMA, SCHEMA, VERSION, ReferenceMatcher, make_template
from dotears.sfx_taxonomy import UI_DEFINITIONS, UI_TAXONOMY, UI_TYPES, broad_ui_label


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def prepare(root, catalog, ui_scope="narrow"):
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Prepare requires an empty new output folder.")
    sources = [
        r
        for r in json.loads(catalog.read_text("utf-8"))["sources"]
        if r["label_origin"] == "human_high_medium"
    ]
    if not 1 <= len(sources) <= 100:
        raise ValueError("Need 1–100 explicitly reviewed source references.")
    arrays, entries = {}, []
    for source in sources:
        if file_digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("Reviewed source changed.")
        audio, rate = sf.read(source["prepared_effect_path"], dtype="float32")
        if rate != 32000 or len(audio) != 96000:
            raise ValueError("Use prepared 32kHz/3s effects with the acoustic peak at 1s.")
        values, offset = make_template(audio, 1.0)
        arrays[source["id"]] = values
        entries.append(
            {
                "id": source["id"],
                "label": broad_ui_label(source["label"])
                if ui_scope == "broad"
                else source["label"],
                "original_review_label": source["label"],
                "source_sha256": source["sha256"],
                "source_path": source["path"],
                "label_origin": source["label_origin"],
                "peak_offset": offset,
                "enabled": True,
                "threshold": 0.65,
            }
        )
    np.savez(root / "bank.npz", **arrays)
    save(
        root / "protocol.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_catalog_sha256": file_digest(catalog),
            "scope": "Known reviewed waveforms on new backgrounds; foreground overlap intentional.",
            "threshold": "max(0.65, maximum wrong-type calibration NCC+0.03), capped1.01",
            "competition": "All templates including disabled/other; cross-type margin0.05",
            "enable": "At least two correct 0dB mixtures from two background kinds",
            "test_gate": "precision>=0.9, zero background/nonbank false, 0dB coverage>=0.2",
            "video_used_for_fitting": False,
        },
    )
    save(
        root / "initial-policy.json",
        {
            "schema_version": BROAD_SCHEMA if ui_scope == "broad" else SCHEMA,
            "label_taxonomy": UI_TAXONOMY
            if ui_scope == "broad"
            else "dotears.sfx-types/1-narrow-click",
            "label_definitions": UI_DEFINITIONS if ui_scope == "broad" else None,
            "version": VERSION,
            "bank_path": "bank.npz",
            "bank_sha256": file_digest(root / "bank.npz"),
            "templates": entries,
            "protocol_sha256": file_digest(root / "protocol.json"),
            "created_at": datetime.now(UTC).isoformat(),
            "scope": "Known waveforms only",
        },
    )


def apply_policy(matcher, rows, policy):
    entries = {e["id"]: e for e in policy["templates"]}
    results = []
    for row in rows:
        scores = [
            {
                **s,
                "threshold": entries[s["reference_id"]]["threshold"],
                "enabled": entries[s["reference_id"]]["enabled"],
            }
            for s in row["scores"]
        ]
        results.append(
            {**{k: v for k, v in row.items() if k != "scores"}, "decision": matcher.decide(scores)}
        )
    return results


def fit(root, calibration, test, extra_features):
    matcher = ReferenceMatcher(root / "initial-policy.json")
    known = {e["id"] for e in matcher.policy["templates"]}
    extra = (
        [json.loads(line) for line in extra_features.read_text("utf-8").splitlines()]
        if extra_features
        else []
    )

    def collect(path, split):
        rows = [r for r in json.loads(path.read_text("utf-8"))["rows"] if r["split"] == split]
        rows += [
            {**r, "label": "background", "nonbank_source": True}
            for r in extra
            if r["split"] == split and r["source_id"] not in known
        ]
        results = []
        unique_pcm = {}
        for row in rows:
            if matcher.broad_ui:
                row = {
                    **row,
                    "original_source_label": row["label"],
                    "label": broad_ui_label(row["label"]),
                }
            if file_digest(Path(row["audio_path"])) != row["audio_sha256"]:
                raise ValueError("Frozen mixture changed.")
            audio, rate = sf.read(row["audio_path"], dtype="float32")
            if rate != 32000:
                raise ValueError("Use 32kHz prepared mixtures.")
            fingerprint = pcm_hash(audio)
            if fingerprint in unique_pcm:
                if unique_pcm[fingerprint] != row["label"]:
                    raise ValueError("Contradictory labels on identical waveform PCM.")
                continue
            unique_pcm[fingerprint] = row["label"]
            results.append(
                {
                    **{k: v for k, v in row.items() if k != "evidence"},
                    "scores": matcher.scores(audio, row["peak_relative"]),
                }
            )
        save(root / (split + "-scores.json"), results)
        return results

    calibration_rows = collect(calibration, "calibration")
    policy = copy.deepcopy(matcher.policy)
    for template in policy["templates"]:
        negative = [
            next(s["score"] for s in r["scores"] if s["reference_id"] == template["id"])
            for r in calibration_rows
            if r["label"] != template["label"]
        ]
        threshold = min(1.01, max(0.65, max(negative, default=0) + 0.03))
        template["threshold"] = threshold
        positives = []
        for row in calibration_rows:
            scores = copy.deepcopy(row["scores"])
            next(s for s in scores if s["reference_id"] == template["id"])["threshold"] = threshold
            result = matcher.decide(scores)
            if (
                row["source_id"] == template["id"]
                and row.get("snr_db") == 0
                and result["label"] == template["label"]
                and result["best_reference_id"] == template["id"]
            ):
                positives.append(row)
        template["enabled"] = (
            template["label"] != "other"
            and len(positives) >= 2
            and len({r["background"] for r in positives}) >= 2
        )
    policy["created_at"] = datetime.now(UTC).isoformat()
    save(root / "selected-before-test.json", policy)
    test_rows = collect(test, "test")
    decisions = apply_policy(matcher, test_rows, policy)
    checks = {}
    for label in UI_TYPES if matcher.broad_ui else ("whoosh", "notification", "ui_click"):
        accepted = [r for r in decisions if r["decision"]["label"] == label]
        true = [r for r in accepted if r["label"] == label and not r.get("nonbank_source")]
        false = [r for r in accepted if r not in true]
        primary = [r for r in decisions if r["label"] == label and r.get("snr_db") == 0]
        primary_true = sum(r["decision"]["label"] == label for r in primary)
        precision = len(true) / len(accepted) if accepted else None
        bg_false = sum(r["label"] == "background" for r in false)
        coverage = primary_true / len(primary) if primary else 0
        passed = precision is not None and precision >= 0.9 and bg_false == 0 and coverage >= 0.2
        checks[label] = {
            "accepted": len(accepted),
            "true_positives": len(true),
            "false_positives": len(false),
            "precision": precision,
            "background_or_nonbank_false": bg_false,
            "primary_true": primary_true,
            "primary_count": len(primary),
            "primary_coverage": coverage,
            "passed": passed,
        }
    final = copy.deepcopy(policy)
    for entry in final["templates"]:
        if entry["label"] == "other" or not checks[entry["label"]]["passed"]:
            entry["enabled"] = False
    final.update(status="known_patterns_only_pending_full_length_integration", test_gate=checks)
    save(root / "policy.json", final)
    save(
        root / "evaluation.json",
        {
            "checks": checks,
            "decisions": decisions,
            "policy_sha256": file_digest(root / "policy.json"),
            "calibration_corpus_sha256": file_digest(calibration),
            "test_corpus_sha256": file_digest(test),
            "active_templates": dict(
                Counter(e["label"] for e in final["templates"] if e["enabled"])
            ),
            "scope": "Known foregrounds; new backgrounds; not unseen-source recognition",
        },
    )
    print(json.dumps(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--extra-features", type=Path)
    parser.add_argument("--ui-scope", choices=("narrow", "broad"), default="narrow")
    args = parser.parse_args()

    def reject(*_a, **_k):
        raise RuntimeError("Reference calibration is offline.")

    socket.create_connection = reject
    socket.socket.connect = reject
    if args.phase == "prepare":
        if not args.catalog:
            parser.error("prepare needs --catalog")
        prepare(args.root.resolve(), args.catalog, args.ui_scope)
    else:
        if not args.calibration or not args.test:
            parser.error("fit needs --calibration and --test frozen corpus manifests")
        fit(args.root.resolve(), args.calibration, args.test, args.extra_features)


if __name__ == "__main__":
    main()
