"""Freeze multi-source SFX mixtures, stems and disjoint author/background splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from dotears.audiosep import file_digest
from dotears.sfx_auto import pcm_hash
from dotears.sfx_contrast import FEATURE_NAMES, VERSION
from dotears.sfx_taxonomy import UI_TAXONOMY


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def decode(path):
    info = sf.info(path)
    if info.channels not in (1, 2) or not 8000 <= info.samplerate <= 192000 or info.duration > 180:
        raise ValueError("Source is outside bounded mono/stereo decode scope.")
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    return librosa.resample(audio.mean(1), orig_sr=rate, target_sr=32000)


def build(root, catalog_path):
    if root.exists():
        raise ValueError("Use a new mixture output folder.")
    catalog = json.loads(catalog_path.read_text("utf-8"))
    sources, backgrounds = catalog["sources"], catalog["backgrounds"]
    source_authors = {
        split: {r["author_group"] for r in sources if r["split"] == split}
        for split in ("train", "calibration", "test")
    }
    for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        if source_authors[a] & source_authors[b]:
            raise ValueError("Foreground authors cross data splits.")
        if {r["sha256"] for r in backgrounds if r["split"] == a} & {
            r["sha256"] for r in backgrounds if r["split"] == b
        }:
            raise ValueError("A background recording crosses data splits.")
    root.mkdir(parents=True)
    for name in ("audio", "effects", "beds"):
        (root / name).mkdir()
    decoded_beds = {}
    for row in backgrounds:
        if file_digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("Background source changed.")
        decoded_beds[row["id"]] = decode(Path(row["path"]))
        if rms(decoded_beds[row["id"]]) < 1e-6:
            raise ValueError("Background source is silent.")
    rows, source_receipts, seen_pcm = [], [], {}

    def bed_for(split, kind, seed):
        rng = np.random.default_rng(seed)
        parts, choices = [], []
        for role in ("music", "speech") if kind == "music_speech" else (kind,):
            candidates = [r for r in backgrounds if r["split"] == split and r["role"] == role]
            if not candidates:
                raise ValueError("Need music and speech backgrounds in every split.")
            chosen = candidates[int(rng.integers(len(candidates)))]
            values = decoded_beds[chosen["id"]]
            repeat = len(values) < 96000
            values = np.resize(values, max(96000, len(values)))
            first = int(rng.integers(len(values) - 96000 + 1))
            part = values[first : first + 96000].copy()
            part *= 0.06 / max(rms(part), 1e-10)
            parts.append(part)
            choices.append(
                {
                    "source_id": chosen["id"],
                    "source_sha256": chosen["sha256"],
                    "first_frame": first,
                    "repeated_if_short": repeat,
                }
            )
        values = sum(parts)
        values *= 0.08 / max(rms(values), 1e-10)
        return values, choices

    def add(identity, values, split, label, author, source_id, source_sha, peak, **extra):
        if extra.get("kind") == "background":
            gain = min(1.0, 0.9 / max(float(np.max(np.abs(values))), 1e-12))
            values = values * gain
            extra["background_output_gain"] = gain
        path = root / "audio" / (identity + ".wav")
        sf.write(path, values.astype(np.float32), 32000, subtype="FLOAT")
        rows.append(
            {
                "id": identity,
                "split": split,
                "label": label,
                "family": author,
                "author_group": author,
                "source_id": source_id,
                "source_sha256": source_sha,
                "audio_path": str(path),
                "audio_sha256": file_digest(path),
                "peak_relative": peak,
                **extra,
            }
        )

    for source in sources:
        if source["label"] not in ("whoosh", "notification", "ui_sound", "other"):
            raise ValueError("Unsupported mixture source label.")
        path = Path(source["path"])
        if file_digest(path) != source["sha256"]:
            raise ValueError("SFX source changed.")
        audio = decode(path)
        fingerprint = pcm_hash(audio.astype(np.float32))
        if fingerprint in seen_pcm:
            prior = seen_pcm[fingerprint]
            if prior["split"] != source["split"] or prior["label"] != source["label"]:
                raise ValueError("Identical source PCM crosses labels or splits.")
            continue
        seen_pcm[fingerprint] = source
        energy = np.convolve(audio**2, np.ones(320) / 320, mode="same")
        peak = int(np.argmax(energy))
        first, last = max(0, peak - 32000), min(len(audio), peak + 64000)
        effect = np.zeros(96000)
        offset = 32000 - (peak - first)
        effect[offset : offset + last - first] = audio[first:last]
        if rms(effect) < 1e-7:
            raise ValueError("Silent SFX source.")
        effect *= min(1.0, 0.8 / max(np.max(np.abs(effect)), 1e-12))
        effect_path = root / "effects" / (source["id"] + ".wav")
        sf.write(effect_path, effect, 32000, subtype="FLOAT")
        actual_effect, _ = sf.read(effect_path, dtype="float64")
        source_receipts.append(
            {
                **source,
                "canonical_pcm_sha256": fingerprint,
                "effect_path": str(effect_path),
                "effect_sha256": file_digest(effect_path),
                "source_duration": len(audio) / 32000,
                "source_peak_seconds": peak / 32000,
                "retained_energy_fraction": float(
                    np.sum(audio[first:last] ** 2) / np.sum(audio**2)
                ),
            }
        )
        active = np.abs(actual_effect) > max(np.max(np.abs(actual_effect)) * 0.01, 1e-7)
        seed = int(source["sha256"][:8], 16)
        common = {
            "label_origin": source["label_origin"],
            "label_scope": (
                "Known inserted excerpt with a source-level label; "
                "mixture audibility not independently human-rated."
            ),
            "effect_path": str(effect_path),
            "effect_sha256": file_digest(effect_path),
        }
        add(
            source["split"] + "-" + source["id"] + "-clean",
            actual_effect,
            source["split"],
            source["label"],
            source["author_group"],
            source["id"],
            source["sha256"],
            1.0,
            kind="clean",
            **common,
        )
        for j, kind in enumerate(("music", "speech", "music_speech")):
            levels = (0, (-6 if (seed + j) % 2 else 6)) if source["split"] == "train" else (0, -6)
            for snr in levels:
                identity = f"{source['split']}-{source['id']}-{kind}-{snr:+d}"
                for attempt in range(64):
                    bed, choices = bed_for(
                        source["split"], kind, seed + j * 101 + (snr + 6) * 7919 + attempt * 104729
                    )
                    if rms(bed[active]) >= 0.02:
                        break
                else:
                    raise ValueError("Could not place an effect on an active background interval.")
                bed_path = root / "beds" / (identity + ".wav")
                sf.write(bed_path, bed, 32000, subtype="FLOAT")
                bed, _ = sf.read(bed_path, dtype="float64")
                gain = rms(bed[active]) * 10 ** (snr / 20) / rms(actual_effect[active])
                mixed = actual_effect * gain + bed
                joint = min(1.0, 0.9 / max(np.max(np.abs(mixed)), 1e-12))
                jitter = float(
                    np.random.default_rng(seed + j + (snr + 6) * 31).uniform(-0.06, 0.06)
                )
                add(
                    identity,
                    mixed * joint,
                    source["split"],
                    source["label"],
                    source["author_group"],
                    source["id"],
                    source["sha256"],
                    1.0 + jitter,
                    kind="mixed",
                    background=kind,
                    snr_db=snr,
                    true_effect_peak_relative=1.0,
                    candidate_jitter_seconds=jitter,
                    bed_path=str(bed_path),
                    bed_sha256=file_digest(bed_path),
                    bed_sources=choices,
                    effect_gain=gain,
                    background_placement_attempts=attempt + 1,
                    joint_gain=joint,
                    active_sample_count=int(active.sum()),
                    **common,
                )
    for split in ("train", "calibration", "test"):
        for kind in ("music", "speech", "music_speech"):
            for i in range(12):
                values, choices = bed_for(split, kind, 847913 + i * 73)
                identity = f"{split}-background-{kind}-{i:02}"
                add(
                    identity,
                    values,
                    split,
                    "background",
                    "background-" + split + "-" + kind,
                    identity,
                    pcm_hash(values.astype(np.float32)),
                    0.8 + (i % 5) * 0.1,
                    kind="background",
                    bed_sources=choices,
                )
        add(
            split + "-silence",
            np.zeros(96000),
            split,
            "background",
            "background-" + split,
            split + "-silence",
            "silence",
            1.0,
            kind="background",
        )
    save(root / "sources.json", {"sources": source_receipts, "backgrounds": backgrounds})
    save(root / "corpus.json", {"rows": rows, "sample_rate": 32000})
    save(
        root / "protocol.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "label_taxonomy": UI_TAXONOMY,
            "version": VERSION,
            "feature_names": FEATURE_NAMES,
            "source_catalog_sha256": file_digest(catalog_path),
            "corpus_sha256": file_digest(root / "corpus.json"),
            "feature_candidates": ["semantic", "raw_embedding", "raw_and_contrast"],
            "C_values": [0.1, 1.0],
            "training_validation": (
                "Five disjoint whole-author folds; source-balanced standardization and heads."
            ),
            "threshold_gate": {
                "precision_min": 0.9,
                "true_min": 5,
                "positive_authors_min": 3,
                "background_false_max": 0,
                "blade_false_max": 0,
            },
            "calibration_and_test_gate": {
                "precision_min": 0.9,
                "background_false_max": 0,
                "primary_0db_recovery_min": 0.2,
                "positive_sources_min": 2,
            },
            "selection": (
                "Calibration only: passing class count, macro 0dB recovery, TP; "
                "prefer C=1 then simpler features."
            ),
            "competition": (
                "All heads including disabled; OOF median support normalization; "
                "margin0.15; positive raw text cosine."
            ),
            "private_video_used_for_fitting": False,
            "private_video_is_known_regression": True,
            "source_labels": (
                "Prior human/author-caption labels plus FSD50K human acoustic labels "
                "with explicitly recorded author role metadata."
            ),
            "mixtures_human_listening_verified": False,
            "label_scope": "Known component insertion, source-level semantic labels.",
            "counts": dict(Counter(r["split"] for r in rows)),
            "source_counts": dict(Counter(r["split"] for r in source_receipts)),
            "author_sets": {k: sorted(v) for k, v in source_authors.items()},
        },
    )
    print(
        json.dumps(
            {
                "windows": len(rows),
                "sources": len(source_receipts),
                "counts": dict(Counter(r["split"] for r in rows)),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.catalog.resolve())


if __name__ == "__main__":
    main()
