"""Offline, resumable AudioSep evaluation with known components and no-insertion controls."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    if (root / "results.json").exists():
        raise ValueError("Completed results exist; use a different experiment directory.")
    os.environ.update(
        USE_TF="0",
        USE_FLAX="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )

    def reject_network(*_args, **_kwargs):
        raise RuntimeError("Separation inference attempted a network connection.")

    socket.create_connection = reject_network
    socket.socket.connect = reject_network
    import numpy as np
    import psutil
    import soundfile as sf

    from dotears.audiosep import LocalAudioSep, file_digest
    from dotears.separation import output_energy_db, reference_metrics

    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    if file_digest(root / "corpus.json") != protocol["corpus_sha256"]:
        raise ValueError("Prepared corpus changed after protocol freeze.")
    for path, expected in protocol["code_sha256"].items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"Experiment code changed: {path}")
    corpus = json.loads((root / "corpus.json").read_text("utf-8"))
    cases = corpus["cases"]
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Duplicate separation case IDs.")
    for row in cases:
        if not re.fullmatch(r"[A-Za-z0-9_.+\-]{1,180}", row["id"]):
            raise ValueError("Unsafe separation case ID.")
        if file_digest(Path(row["input"]["path"])) != row["input"]["sha256"]:
            raise ValueError("Prepared input changed.")
        if row["target"] and file_digest(Path(row["target"]["path"])) != row["target"]["sha256"]:
            raise ValueError("Known target changed.")
    started, stop, peak = time.perf_counter(), threading.Event(), [0]
    process = psutil.Process()

    def monitor_memory():
        while not stop.wait(0.1):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=monitor_memory, daemon=True)
    sampler.start()
    model = LocalAudioSep(args.model, threads=protocol["threads"])
    parity = model.verify_text_forward(list(protocol["queries"].values()))
    singleton_parity = model.verify_text_forward([next(iter(protocol["queries"].values()))])
    progress_path = root / "progress.jsonl"
    previous = {}
    if progress_path.exists():
        for line in progress_path.read_text("utf-8").splitlines():
            entry = json.loads(line)
            if entry["protocol_sha256"] != file_digest(root / "protocol.json"):
                raise ValueError("Partial progress belongs to another protocol.")
            previous[entry["id"]] = entry
    destination = root / "outputs"
    destination.mkdir(exist_ok=True)
    results = []
    for index, case in enumerate(cases, 1):
        if case["id"] in previous:
            prior = previous[case["id"]]
            if file_digest(Path(prior["output_path"])) != prior["output_sha256"]:
                raise ValueError("A previously separated output changed.")
            results.append(prior)
            continue
        if (
            time.perf_counter() - started > protocol["runtime_limits"]["maximum_seconds"]
            or peak[0] > protocol["runtime_limits"]["maximum_rss_bytes"]
        ):
            raise RuntimeError(
                "Separation resource budget reached; verified progress is saved for resume."
            )
        waveform, rate = sf.read(case["input"]["path"], dtype="float32")
        if rate != 32000 or waveform.shape != (96000,):
            raise ValueError("Prepared windows must be three-second 32kHz mono WAVs.")
        call_started = time.perf_counter()
        estimate = model.separate(waveform, case["prompt"])
        output_path = destination / (case["id"] + ".wav")
        # Float WAV preserves actual output levels, including possible peaks above unity.
        sf.write(output_path, estimate, rate, subtype="FLOAT")
        result = {
            **case,
            "output_path": str(output_path),
            "output_sha256": file_digest(output_path),
            "output_peak": float(np.max(np.abs(estimate))),
            "output_rms": float(np.sqrt(np.mean(estimate.astype(np.float64) ** 2))),
            "output_energy_relative_db": output_energy_db(waveform, estimate),
            "separation_seconds": round(time.perf_counter() - call_started, 4),
            "protocol_sha256": file_digest(root / "protocol.json"),
            "label": "unknown",
            "confidence": None,
            "status": "estimated_component_not_confirmed_isolated_sfx",
        }
        if case["target"]:
            target, _ = sf.read(case["target"]["path"], dtype="float32")
            result["full_window_metrics"] = reference_metrics(target, waveform, estimate)
            first, last = case["support_first"], case["support_last"]
            result["support_metrics"] = reference_metrics(
                target[first:last], waveform[first:last], estimate[first:last]
            )
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        results.append(result)
        if index % 10 == 0:
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(cases),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    stop.set()
    sampler.join()

    def median(values):
        finite = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(finite)) if finite else None

    def summarize(rows):
        si = [r["support_metrics"]["si_sdr_improvement_db"] for r in rows]
        gains = [r["support_metrics"]["target_projection_gain_db"] for r in rows]
        lo, hi = protocol["research_gate"]["gain_range_db"]
        return {
            "cases": len(rows),
            "source_count": len({r["source_id"] for r in rows}),
            "median_support_si_sdri_db": median(si),
            "positive_si_sdri_fraction": sum(v is not None and v > 0 for v in si) / len(rows),
            "median_support_sdri_db": median(
                [r["support_metrics"]["sdr_improvement_db"] for r in rows]
            ),
            "median_target_gain_db": median(gains),
            "gain_retained_fraction": sum(
                lo <= v <= hi and r["support_metrics"]["target_projection_scale"] > 0
                for v, r in zip(gains, rows, strict=True)
            )
            / len(rows),
        }

    gate = protocol["research_gate"]
    primary, grouped, checks = {}, {}, {}
    for label in protocol["queries"]:
        group = [r for r in results if r["kind"] == "known_mixture" and r["query_label"] == label]
        primary[label] = summarize([r for r in group if r["snr_db"] == gate["primary_snr_db"]])
        grouped[label] = {
            background: {
                snr: summarize(
                    [r for r in group if r["background"] == background and r["snr_db"] == snr]
                )
                for snr in (6, 0, -6)
            }
            for background in ("music", "speech", "music_speech")
        }
        clean = [
            r for r in results if r["kind"] == "clean_preservation" and r["query_label"] == label
        ]
        clean_sdr = median([r["full_window_metrics"]["output_sdr_db"] for r in clean])
        primary[label]["clean_preservation_median_sdr_db"] = clean_sdr
        checks[label] = {
            "overlap_improvement": primary[label]["median_support_si_sdri_db"] is not None
            and primary[label]["median_support_si_sdri_db"]
            >= gate["per_class_median_support_si_sdri_min_db"],
            "effect_retained": primary[label]["gain_retained_fraction"]
            >= gate["per_class_gain_retained_fraction_min"],
            "clean_preserved": clean_sdr is not None
            and clean_sdr >= gate["clean_preservation_median_sdr_min_db"],
        }
    background = [r for r in results if r["kind"] == "background_control"]
    attenuated = sum(
        r["output_energy_relative_db"] <= gate["background_energy_attenuation_db"]
        for r in background
    )
    background_check = attenuated / len(background) >= gate["background_attenuated_fraction_min"]
    silence_check = (
        max(r["output_peak"] for r in results if r["kind"] == "silence_control")
        <= gate["silence_peak_max"]
    )
    passed = (
        all(all(values.values()) for values in checks.values())
        and background_check
        and silence_check
    )
    summary = {
        "primary_0db": primary,
        "by_background_and_snr": grouped,
        "class_gate_checks": checks,
        "background": {
            "cases": len(background),
            "sufficiently_attenuated": attenuated,
            "median_output_energy_relative_db": median(
                [r["output_energy_relative_db"] for r in background]
            ),
            "gate_passed": background_check,
        },
        "silence_gate_passed": silence_check,
        "research_gate_passed": passed,
    }
    output = {
        "schema_version": "dotears.separation-evaluation/1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "model_load": model.load_evidence,
        "text_forward_parity": parity,
        "singleton_text_parity": singleton_parity,
        "protocol_sha256": file_digest(root / "protocol.json"),
        "summary": summary,
        "seconds": round(time.perf_counter() - started, 3),
        "sampled_peak_process_rss": peak[0],
        "network_connections_blocked": True,
        "threads": protocol["threads"],
        "automatic_labels_enabled": False,
        "case_count": len(results),
        "rows": results,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "torchlibrosa", "librosa")
        },
    }
    with (root / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
    (root / "summary.json").write_text(
        json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2), "utf-8"
    )
    print(json.dumps({k: v for k, v in output.items() if k not in {"rows", "summary"}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
