"""Prepare a frozen AudioSep study from the existing human-labeled SFX corpus."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from dotears.audiosep import PINNED_FILES, file_digest, verify_prepared_audiosep
from dotears.sfx import PROMPTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--music", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--audiosep", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        raise ValueError("Choose a new preparation directory; do not overwrite frozen protocols.")
    preparation = verify_prepared_audiosep(args.audiosep)
    source_corpus = json.loads(args.corpus.read_text("utf-8"))
    sources = source_corpus["clean_sources"]
    queries = {label: PROMPTS[label][0] for label in ("whoosh", "notification", "ui_click")}
    rate, source_rate, frames, onset = 32000, 48000, 144000, 48000

    def decode(path):
        info = sf.info(path)
        if info.channels not in (1, 2) or info.frames * info.channels * 8 > 64 * 1024**2:
            raise ValueError("Source decode exceeds the preparation budget.")
        values, original_rate = sf.read(path, dtype="float64", always_2d=True)
        mono = values.mean(axis=1)
        return librosa.resample(mono, orig_sr=original_rate, target_sr=source_rate)

    def rms(value):
        return float(np.sqrt(np.mean(value**2)))

    def to_native(value):
        return librosa.resample(value, orig_sr=source_rate, target_sr=rate).astype(np.float32)

    def save(name, value):
        path = root / "inputs" / (name + ".wav")
        path.parent.mkdir(exist_ok=True)
        if not np.isfinite(value).all() or len(value) != 96000:
            raise ValueError("Prepared separation windows must be finite, mono and three seconds.")
        sf.write(path, value, rate, subtype="FLOAT")
        return {"path": str(path), "sha256": file_digest(path)}

    music, speech = decode(args.music), decode(args.speech)
    if min(len(music), len(speech)) < frames:
        raise ValueError("Both background recordings must contain at least three seconds.")
    speech_starts = []
    for start in range(0, len(speech) - frames, 4800):
        window = speech[start : start + frames]
        if (
            rms(window[onset : onset + 960]) > 0.001
            and rms(window) > 0.005
            and all(abs(start - s) >= frames for s in speech_starts)
        ):
            speech_starts.append(start)
            if len(speech_starts) == 4:
                break
    if len(speech_starts) != 4:
        raise ValueError("Fixed energy rule could not select four non-overlapping speech windows.")
    beds = {}
    music_starts = []
    for index in range(4):
        start = round((len(music) - frames) * index / 3)
        music_starts.append(start)
        m, s = (
            music[start : start + frames],
            speech[speech_starts[index] : speech_starts[index] + frames],
        )
        m, s = m / rms(m) * 0.08, s / rms(s) * 0.08
        for label, values in (("music", m), ("speech", s), ("music_speech", m + s)):
            beds[(label, index)] = values / rms(values) * 0.08

    cases = []

    def add(identity, input_info, query_label, kind, target=None, **extra):
        cases.append(
            {
                "id": identity,
                "kind": kind,
                "input": input_info,
                "query_label": query_label,
                "prompt": queries[query_label],
                "target": target,
                **extra,
            }
        )

    for index, row in enumerate(sources):
        if file_digest(Path(row["audio_path"])) != row["audio_sha256"]:
            raise ValueError("Human-reviewed preview changed.")
        if file_digest(Path(row["source_path"])) != row["source_sha256"]:
            raise ValueError("Original source audio changed.")
        preview = decode(row["audio_path"])
        # Remove only the old worksheet's known artificial 150/250ms padding.
        effect = preview[7200:-12000]
        if not 0 < len(effect) < 96000 or rms(effect) < 1e-6:
            raise ValueError("Invalid full source effect duration or energy.")
        metadata = {
            "source_id": row["id"],
            "source_sha256": row["source_sha256"],
            "family": row["family"],
            "human_label": row["label"],
        }
        clean = np.zeros(frames)
        clean[onset : onset + len(effect)] = effect
        clean *= min(1.0, 0.90 / max(np.max(np.abs(clean)), 1e-12))
        clean_info = save("clean-" + row["id"], to_native(clean))
        support = {"support_first": 32000, "support_last": 32000 + round(len(effect) * 2 / 3)}
        if row["label"] not in queries:
            for label in queries:
                add(f"other-{row['id']}-{label}", clean_info, label, "other_control", **metadata)
            continue
        add(
            "clean-" + row["id"],
            clean_info,
            row["label"],
            "clean_preservation",
            target=clean_info,
            **metadata,
            **support,
        )
        for background in ("music", "speech", "music_speech"):
            bed_index = index % 4
            bed = beds[(background, bed_index)]
            bed_rms = rms(bed[onset : onset + len(effect)])
            if bed_rms < 1e-8:
                raise ValueError("Silent background on effect support.")
            for snr in (6, 0, -6):
                gain = bed_rms * 10 ** (snr / 20) / rms(effect)
                foreground = np.zeros(frames)
                foreground[onset : onset + len(effect)] = effect * gain
                mixed = foreground + bed
                common_gain = min(1.0, 0.90 / max(np.max(np.abs(mixed)), 1e-12))
                actual_snr = 20 * np.log10(rms(foreground[onset : onset + len(effect)]) / bed_rms)
                if abs(actual_snr - snr) > 1e-8:
                    raise ValueError("Component SNR mismatch.")
                identity = f"mix-{row['id']}-{background}-{snr:+d}"
                mixture_info = save(identity, to_native(mixed * common_gain))
                target_info = save(identity + "-target", to_native(foreground * common_gain))
                add(
                    identity,
                    mixture_info,
                    row["label"],
                    "known_mixture",
                    target=target_info,
                    **metadata,
                    **support,
                    background=background,
                    background_index=bed_index,
                    snr_db=snr,
                    achieved_component_snr_db=float(actual_snr),
                    joint_gain=common_gain,
                    conditioning="oracle_known_class_description; not automatic event recognition",
                )
    for (background, index), bed in beds.items():
        info = save(f"background-{background}-{index}", to_native(bed))
        for label in queries:
            add(
                f"background-{background}-{index}-{label}",
                info,
                label,
                "background_control",
                background=background,
            )
    silence_info = save("silence", np.zeros(96000, dtype=np.float32))
    for label in queries:
        add("silence-" + label, silence_info, label, "silence_control")

    # Real video contexts are qualitative only: surrounding 3s can contain several events.
    pilot = json.loads(
        Path("validation-output/sfx-pilot-2026-10-27/excerpt.json").read_text("utf-8")
    )
    report_path = Path(pilot["source_report_path"])
    report = json.loads(report_path.read_text("utf-8"))
    native = decode(report_path.parent / report["audio"]["native_path"])
    event_lookup = {row["id"]: row for row in report["events"]}
    for event_id in ("attack_001", "attack_017", "attack_028", "attack_051"):
        event = event_lookup[event_id]
        peak = event["peak_time"] - report["audio"]["timeline_offset_seconds"]
        first = min(max(0, round((peak - 1) * source_rate)), len(native) - frames)
        info = save("real-" + event_id, to_native(native[first : first + frames]))
        for label in queries:
            add(
                f"real-{event_id}-{label}",
                info,
                label,
                "real_diagnostic",
                event_id=event_id,
                window_start=first / source_rate,
                window_end=first / source_rate + 3,
                source_sha256=report["source"]["sha256"],
                reference_available=False,
            )
    corpus = {"schema_version": "dotears.separation-corpus/1", "sample_rate": rate, "cases": cases}
    (root / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=2), "utf-8")
    code = [
        "src/dotears/audiosep.py",
        "src/dotears/separation.py",
        "scripts/evaluate_separation.py",
    ]
    protocol = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": "AudioSep ResUNet30 + its paired text encoder",
        "source_corpus_sha256": file_digest(args.corpus),
        "corpus_sha256": file_digest(root / "corpus.json"),
        "pinned_model_files": PINNED_FILES,
        "github_revision": preparation["github_revision"],
        "checkpoint_revision": preparation["checkpoint_revision"],
        "code_sha256": {name: file_digest(Path(name)) for name in code},
        "queries": queries,
        "queries_fitted": False,
        "model_training": False,
        "threads": 4,
        "input_duration_seconds": 3,
        "complete_source_effect_preserved": True,
        "inputs_and_outputs_float_no_amplitude_normalization_of_outputs": True,
        "no_lag_or_gain_alignment_in_metrics": True,
        "primary_metric": "zero-lag SI-SDR improvement on the overlapping effect support",
        "secondary_metrics": [
            "full-window SI-SDR",
            "scale-dependent SDR",
            "target projection gain",
            "no-insertion output energy",
        ],
        "correct_prompt_is_oracle_upper_bound_not_semantic_detection": True,
        "music_source_sha256": file_digest(args.music),
        "speech_source_sha256": file_digest(args.speech),
        "music_start_frames_48k": music_starts,
        "speech_start_frames_48k": speech_starts,
        "label_scope": (
            "known inserted clean human class; mixed audibility not independently reviewed"
        ),
        "dataset_scope": (
            "small convenience set; several physical swishes; "
            "not representative editing-SFX benchmark"
        ),
        "research_gate": {
            "per_class_median_support_si_sdri_min_db": 3,
            "primary_snr_db": 0,
            "per_class_gain_retained_fraction_min": 0.8,
            "gain_range_db": [-6, 6],
            "background_energy_attenuation_db": -15,
            "background_attenuated_fraction_min": 0.9,
            "clean_preservation_median_sdr_min_db": 10,
            "silence_peak_max": 1e-6,
        },
        "runtime_limits": {"maximum_seconds": 1800, "maximum_rss_bytes": 5 * 1024**3},
        "automatic_labels_enabled": False,
        "case_count": len(cases),
    }
    with (root / "protocol.json").open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"cases": len(cases), "queries": queries, "protocol_frozen": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
