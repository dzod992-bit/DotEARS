"""Freeze author-separated raw-SFX experiments from an explicit local source catalog."""

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
from dotears.sfx_presence import CED_TAGS, FEATURE_NAMES, VERSION


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def decode(path):
    info = sf.info(path)
    if info.channels not in (1, 2) or not 8000 <= info.samplerate <= 192000 or info.duration > 60:
        raise ValueError("Source exceeds bounded decode scope.")
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    return librosa.resample(audio.mean(1), orig_sr=rate, target_sr=32000)


def rms(values):
    return float(np.sqrt(np.mean(values**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        raise ValueError("Do not overwrite a frozen experiment.")
    sources = json.loads(args.catalog.read_text("utf-8"))["sources"]
    rows = []
    dedup = {}
    sources_kept = []
    for source in sources:
        path = Path(source["path"])
        actual = file_digest(path)
        if source["sha256"] and source["sha256"] != actual:
            raise ValueError("Source file changed.")
        source["sha256"] = actual
        audio = decode(path)
        if not len(audio) or rms(audio) < 1e-6:
            raise ValueError("Empty/silent source.")
        fingerprint = pcm_hash(audio.astype(np.float32))
        if fingerprint in dedup:
            if dedup[fingerprint]["label"] != source["label"]:
                raise ValueError("Contradictory labels on identical PCM sources.")
            if dedup[fingerprint]["split"] != source["split"]:
                raise ValueError("Identical source crosses author splits.")
            continue
        dedup[fingerprint] = source
        # A deterministic acoustic peak, independent of semantic class or user video.
        envelope = np.convolve(audio**2, np.ones(320) / 320, mode="same")
        peak = int(np.argmax(envelope))
        effect = np.zeros(96000, dtype=np.float64)
        original_first = max(0, peak - 32000)
        original_last = min(len(audio), peak + 64000)
        target_first = 32000 - (peak - original_first)
        effect[target_first : target_first + original_last - original_first] = audio[
            original_first:original_last
        ]
        effect *= min(1.0, 0.9 / max(np.max(np.abs(effect)), 1e-12))
        source["canonical_pcm_sha256"] = fingerprint
        source["original_duration"] = len(audio) / 32000
        source["source_peak_seconds"] = peak / 32000
        source["retained_energy_fraction"] = float(
            np.sum(audio[original_first:original_last] ** 2) / np.sum(audio**2)
        )
        source["prepared_effect_path"] = str(root / "effects" / (source["id"] + ".wav"))
        Path(source["prepared_effect_path"]).parent.mkdir(exist_ok=True)
        sf.write(source["prepared_effect_path"], effect, 32000, subtype="FLOAT")
        sources_kept.append(source)
    author_sets = {
        split: {r["author_group"] for r in sources_kept if r["split"] == split}
        for split in ("train", "calibration", "test")
    }
    if any(
        author_sets[a] & author_sets[b]
        for a, b in [("train", "calibration"), ("train", "test"), ("calibration", "test")]
    ):
        raise ValueError("Authors cross the frozen train/calibration/test split.")
    base = Path("validation-output/sfx-reference-eval-2026-10-28/sources")
    music_a = base / "cinameng-music/music.wav"
    music_b = base / "drakzlin-music/files/Battle.wav"
    beds = {
        "train": (decode(music_a), decode(base / "local-tts/speech.wav")),
        "calibration": (
            decode(music_a),
            decode(root.parent / "backgrounds/librispeech-speech.wav"),
        ),
        "test": (decode(music_b), decode(root.parent / "backgrounds/voices-speech.wav")),
    }

    def bed_for(split, kind, seed):
        rng = np.random.default_rng(seed)
        parts = []
        for values in beds[split]:
            if len(values) < 96000:
                values = np.tile(values, 1 + 96000 // len(values))
            first = int(rng.integers(0, len(values) - 96000 + 1))
            part = values[first : first + 96000].copy()
            part *= 0.08 / max(rms(part), 1e-9)
            parts.append(part)
        combined = (
            parts[0] if kind == "music" else parts[1] if kind == "speech" else parts[0] + parts[1]
        )
        return combined * 0.08 / max(rms(combined), 1e-9)

    def add(identity, values, split, label, author, source_id, source_sha, kind, **extra):
        path = root / "audio" / (identity + ".wav")
        path.parent.mkdir(exist_ok=True)
        sf.write(path, values.astype(np.float32), 32000, subtype="FLOAT")
        rows.append(
            {
                "id": identity,
                "split": split,
                "label": label,
                "family": author,
                "author_group": author,
                "human_label": label,
                "source_id": source_id,
                "source_sha256": source_sha,
                "audio_path": str(path),
                "audio_sha256": file_digest(path),
                "kind": kind,
                "peak_relative": 1.0,
                **extra,
            }
        )

    for source in sources_kept:
        effect, _ = sf.read(source["prepared_effect_path"], dtype="float64")
        prefix = source["split"] + "-" + source["id"]
        common = {
            "label_origin": source["label_origin"],
            "original_duration": source["original_duration"],
            "retained_energy_fraction": source["retained_energy_fraction"],
        }
        add(
            prefix + "-clean",
            effect,
            source["split"],
            source["label"],
            source["author_group"],
            source["id"],
            source["sha256"],
            "clean",
            **common,
        )
        active = np.abs(effect) > max(np.max(np.abs(effect)) * 0.01, 1e-7)
        levels = (-6, 0, 6) if source["split"] == "train" else (-6, 0)
        for background in ("music", "speech", "music_speech"):
            for snr in levels:
                seed = int(source["sha256"][:8], 16) + (snr + 6) * 123 + sum(map(ord, background))
                bed = bed_for(source["split"], background, seed)
                gain = rms(bed[active]) * 10 ** (snr / 20) / max(rms(effect[active]), 1e-9)
                mixed = effect * gain + bed
                mixed *= min(1.0, 0.9 / max(np.max(np.abs(mixed)), 1e-12))
                add(
                    prefix + f"-{background}-{snr:+d}",
                    mixed,
                    source["split"],
                    source["label"],
                    source["author_group"],
                    source["id"],
                    source["sha256"],
                    "mixed",
                    background=background,
                    snr_db=snr,
                    **common,
                )
    for split in ("train", "calibration", "test"):
        for background in ("music", "speech", "music_speech"):
            for i in range(4):
                bed = bed_for(split, background, 9943 + i * 99)
                identity = f"{split}-background-{background}-{i}"
                add(
                    identity,
                    bed,
                    split,
                    "background",
                    "background-" + split + "-" + background,
                    identity,
                    pcm_hash(bed.astype(np.float32)),
                    "background",
                    background=background,
                )
        add(
            split + "-silence",
            np.zeros(96000),
            split,
            "background",
            "background-" + split + "-silence",
            split + "-silence",
            "silence",
            "background",
        )
    save(
        root / "source-catalog.json",
        {"sources": sources_kept, "author_sets": {k: sorted(v) for k, v in author_sets.items()}},
    )
    save(root / "corpus.json", {"rows": rows, "sample_rate": 32000})
    save(
        root / "protocol.json",
        {
            "schema_version": "dotears.raw-presence-protocol/1",
            "created_at": datetime.now(UTC).isoformat(),
            "version": VERSION,
            "feature_names": FEATURE_NAMES,
            "ced_tags": CED_TAGS,
            "corpus_sha256": file_digest(root / "corpus.json"),
            "catalog_sha256": file_digest(root / "source-catalog.json"),
            "head": (
                "binary weighted logistic C=1, weighted standardization; frozen foundation weights"
            ),
            "threshold": {
                "data": "train leave-whole-author-out predictions",
                "min_precision": 0.95,
                "min_true_positives": 5,
                "min_positive_authors": 2,
                "max_background_false": 0,
                "max_blade_false": 0,
                "objective": "maximum TP subject to constraints",
            },
            "calibration_gate": {
                "min_precision": 0.9,
                "max_background_false": 0,
                "minimum_primary_0db_true": 1,
            },
            "test_gate": {
                "min_precision": 0.9,
                "max_background_false": 0,
                "min_primary_0db_coverage": 0.2,
            },
            "raw_gate": (
                "highest short-focus CLAP target must be the candidate, positive and untied; "
                "disabled classes still compete"
            ),
            "whole_source_preserved": False,
            "label_scope": (
                "known source label on strongest-energy source excerpt; "
                "mixed audibility not independently rated"
            ),
            "source_labels": (
                "30 original human sources plus explicitly marked author-caption weak labels"
            ),
            "all_foreground_author_splits_disjoint": True,
            "exact_pcm_source_duplicates_removed": True,
            "calibration_music_shared_with_training": True,
            "test_music_and_speech_not_in_training": True,
            "private_video_used_in_training": False,
            "video_is_known_regression_not_blind_test": True,
            "counts": dict(Counter(row["split"] for row in rows)),
            "author_sets": {k: sorted(v) for k, v in author_sets.items()},
            "resource_limit": {"seconds": 2400, "rss_bytes": 5 * 1024**3, "threads": 4},
        },
    )
    print(
        json.dumps(
            {
                "windows": len(rows),
                "sources": len(sources_kept),
                "splits": dict(Counter(row["split"] for row in rows)),
                "authors": {k: len(v) for k, v in author_sets.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
