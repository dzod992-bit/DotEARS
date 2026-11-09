"""Verify frozen source identity, disjoint splits and exact mixture component accounting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from dotears.audiosep import file_digest
from dotears.sfx_auto import pcm_hash


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    if file_digest(root / "corpus.json") != protocol["corpus_sha256"]:
        raise ValueError("Frozen corpus changed.")
    catalog = json.loads((root / "sources.json").read_text("utf-8"))
    sources, backgrounds = catalog["sources"], catalog["backgrounds"]
    rows = json.loads((root / "corpus.json").read_text("utf-8"))["rows"]
    by_source = {r["id"]: r for r in sources}
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate window identifiers.")
    for collection, key in (
        (sources, "author_group"),
        (sources, "sha256"),
        (sources, "canonical_pcm_sha256"),
        (backgrounds, "sha256"),
    ):
        for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
            if {r[key] for r in collection if r["split"] == a} & {
                r[key] for r in collection if r["split"] == b
            }:
                raise ValueError("Source or recording identity crosses experiment splits.")
    effects = {}
    for row in sources:
        if file_digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("Original source changed.")
        if file_digest(Path(row["effect_path"])) != row["effect_sha256"]:
            raise ValueError("Prepared source changed.")
        audio, rate = sf.read(row["effect_path"], dtype="float64")
        if rate != 32000 or audio.shape != (96000,):
            raise ValueError("Wrong prepared source format.")
        effects[row["id"]] = audio
    for row in backgrounds:
        if file_digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("Background recording changed.")
    max_error, max_snr_error, peak = 0.0, 0.0, 0.0
    for row in rows:
        path = Path(row["audio_path"])
        if file_digest(path) != row["audio_sha256"]:
            raise ValueError("Prepared mixture changed.")
        audio, rate = sf.read(path, dtype="float64")
        if rate != 32000 or audio.shape != (96000,) or not np.isfinite(audio).all():
            raise ValueError("Invalid mixture waveform.")
        peak = max(peak, float(np.max(np.abs(audio))))
        if row["kind"] == "mixed":
            source = by_source[row["source_id"]]
            if source["label"] != row["label"] or source["split"] != row["split"]:
                raise ValueError("Mixture label/source split changed.")
            effect = effects[row["source_id"]]
            if file_digest(Path(row["bed_path"])) != row["bed_sha256"]:
                raise ValueError("Background component changed.")
            bed, rate = sf.read(row["bed_path"], dtype="float64")
            actual = (effect * row["effect_gain"] + bed) * row["joint_gain"]
            max_error = max(max_error, float(np.max(np.abs(audio - actual))))
            active = np.abs(effect) > max(np.max(np.abs(effect)) * 0.01, 1e-7)
            achieved = 10 * np.log10(
                np.sum((effect[active] * row["effect_gain"]) ** 2) / np.sum(bed[active] ** 2)
            )
            if not np.isfinite(achieved) or row["effect_gain"] <= 0:
                raise ValueError("A source was lost on a silent background interval.")
            max_snr_error = max(max_snr_error, abs(float(achieved) - row["snr_db"]))
            if abs(row["peak_relative"] - 1.0 - row["candidate_jitter_seconds"]) > 1e-9:
                raise ValueError("Candidate peak jitter changed.")
    if max_error > 1e-7 or max_snr_error > 1e-6 or peak > 1.0:
        raise ValueError("Mixture accounting, nominal SNR or no-clipping check failed.")
    seen_pcm = {}
    for row in rows:
        audio, _ = sf.read(row["audio_path"], dtype="float32")
        if row["kind"] != "background":
            key = pcm_hash(audio)
            if key in seen_pcm and seen_pcm[key]["split"] != row["split"]:
                raise ValueError("Identical foreground mixture PCM crosses splits.")
            seen_pcm[key] = row
    return {
        "passed": True,
        "sources": len(sources),
        "background_recordings": len(backgrounds),
        "windows": len(rows),
        "mixed_windows": sum(r["kind"] == "mixed" for r in rows),
        "source_authors": len({r["author_group"] for r in sources}),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "max_component_reconstruction_error": max_error,
        "max_component_snr_error_db": max_snr_error,
        "peak_amplitude": peak,
        "no_foreground_or_background_recording_leaks": True,
        "note": "Components are verified; mixture perceptual labels were not human-rated.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
