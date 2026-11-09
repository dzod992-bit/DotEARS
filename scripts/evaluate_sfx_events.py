"""Evaluate short-event YAMNet transfer heads with whole-family exclusion, entirely offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import socket
import threading
import time
import wave
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--yamnet", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    if (root / "results.json").exists():
        raise ValueError("Experiment output exists; prepare a separate run directory.")
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    if digest(root / "corpus.json") != protocol["corpus_sha256"]:
        raise ValueError("Corpus changed after protocol freeze.")
    for name, expected in protocol["code_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Experiment code changed after protocol freeze: {name}")
    corpus = json.loads((root / "corpus.json").read_text("utf-8"))
    for path, expected in protocol["source_annotation_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Original human annotations changed after protocol freeze.")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[name] = str(protocol["threads"])

    def reject_network(*_args, **_kwargs):
        raise RuntimeError("SFX event experiment attempted a network connection.")

    socket.create_connection = reject_network
    socket.socket.connect = reject_network
    started = time.perf_counter()
    import librosa
    import numpy as np
    import psutil
    import tensorflow as tf

    from dotears.core import model_fingerprint
    from dotears.sfx import read_window
    from dotears.sfx_events import (
        fit_linear_head,
        predict_linear_head,
        suppress_prefix_background,
        training_indices,
    )

    if model_fingerprint(str(args.yamnet.resolve())) != protocol["yamnet_sha256"]:
        raise ValueError("Prepared YAMNet differs from the frozen model hash.")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(protocol["threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = tf.saved_model.load(str(args.yamnet.resolve()))
    with (args.yamnet / "assets/yamnet_class_map.csv").open(encoding="utf-8") as handle:
        original_labels = [row["display_name"] for row in csv.DictReader(handle)]
    process, stop, peak = psutil.Process(), threading.Event(), [0]

    def sample_memory():
        while not stop.wait(0.1):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    entries = corpus["training"] + corpus["queries"]
    unique = {}
    for row in entries:
        if digest(Path(row["audio_path"])) != row["audio_sha256"]:
            raise ValueError(f"Changed audio: {row['id']}")
        unique.setdefault(row["audio_sha256"], row)
    features, diagnostics = {}, {}
    conditions = protocol["conditions"]
    probe_dir = root / "foreground-diagnostics"
    probe_dir.mkdir(exist_ok=True)
    for number, (audio_hash, row) in enumerate(unique.items(), 1):
        path = Path(row["audio_path"])
        with wave.open(str(path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        samples, rate = read_window(path, 0, duration)
        samples = librosa.resample(samples, orig_sr=rate, target_sr=16000)
        samples *= min(1.0, 1 / max(float(np.max(np.abs(samples))), 1.0))
        filtered = suppress_prefix_background(samples)
        diagnostics[audio_hash] = {
            "input_rms": float(np.sqrt(np.mean(samples**2))),
            "filtered_rms": float(np.sqrt(np.mean(filtered**2))),
            "sample_count": len(samples),
        }
        for condition, signal in zip(conditions, (samples, filtered), strict=True):
            # YAMNet's first patch needs 0.975s including its analysis frame extent.
            padded = np.pad(signal, (0, max(0, 15600 - len(signal))))
            scores, embeddings, _ = model(tf.constant(padded, dtype=tf.float32))
            vector = embeddings.numpy().mean(axis=0)
            if vector.shape != (1024,) or not np.isfinite(vector).all():
                raise ValueError("Invalid YAMNet short-event embedding.")
            features[(audio_hash, condition)] = vector
            ranking = np.argsort(-scores.numpy().mean(axis=0))[:3]
            diagnostics[audio_hash][condition] = [
                {
                    "label": original_labels[i],
                    "uncalibrated_score": float(scores.numpy()[:, i].mean()),
                }
                for i in ranking
            ]
        if row["kind"] == "real_diagnostic":
            with wave.open(str(probe_dir / (row["event_id"] + ".wav")), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(np.round(filtered * 32767).astype("<i2").tobytes())
        if number % 100 == 0:
            print(json.dumps({"embedded": number, "unique_inputs": len(unique)}), flush=True)
    keys = sorted(unique)
    np.savez_compressed(
        root / "features.npz",
        hashes=np.array(keys),
        **{
            condition: np.stack([features[(key, condition)] for key in keys])
            for condition in conditions
        },
    )
    heads_dir = root / "heads"
    heads_dir.mkdir(exist_ok=True)
    predictions, fit_audit = [], []
    train = corpus["training"]
    for condition in conditions:
        train_features = np.stack([features[(row["audio_sha256"], condition)] for row in train])
        fold_models = {}
        for query in corpus["queries"]:
            indices = training_indices(train, query)
            fold_key = hashlib.sha256(indices.tobytes()).hexdigest()[:16]
            if fold_key not in fold_models:
                selected = [train[i] for i in indices]
                head = fit_linear_head(train_features[indices], selected)
                fold_models[fold_key] = head
                filename = f"{condition}-{fold_key}.json"
                head.update(
                    feature_extractor="yamnet_1024",
                    preprocessing=condition,
                    yamnet_sha256=protocol["yamnet_sha256"],
                    protocol_sha256=digest(root / "protocol.json"),
                )
                dump(heads_dir / filename, head)
                fit_audit.append(
                    {
                        "condition": condition,
                        "fold": fold_key,
                        "head_path": filename,
                        "training_ids": [row["id"] for row in selected],
                        "excluded_training_count": len(train) - len(selected),
                        "training_families": head["training_families"],
                        "iterations": head["iterations"],
                    }
                )
            head = fold_models[fold_key]
            assert not query.get("family") or query["family"] not in head["training_families"]
            if query.get("source_sha256"):
                assert query["source_sha256"] not in head["training_source_hashes"]
            assert query["audio_sha256"] not in head["training_audio_hashes"]
            prediction = predict_linear_head(
                head, features[(query["audio_sha256"], condition)][None]
            )[0]
            predictions.append({**query, "condition": condition, "fold": fold_key, **prediction})
        print(json.dumps({"condition": condition, "trained_folds": len(fold_models)}), flush=True)
    stop.set()
    sampler.join()
    metrics = {}
    targets = {"whoosh", "notification", "ui_click"}
    for condition in conditions:
        rows = [row for row in predictions if row["condition"] == condition]

        def summarize(group):
            out = {}
            for label in ("whoosh", "notification", "ui_click", "other", "background"):
                selected = [row for row in group if row.get("human_label") == label]
                out[label] = {
                    "examples": len(selected),
                    "first_matches": sum(row["best_match"]["label"] == label for row in selected),
                    "first_names": dict(Counter(row["best_match"]["label"] for row in selected)),
                }
            return out

        clean = summarize([row for row in rows if row["kind"] == "clean"])
        mixtures = {}
        for background in ("music", "speech", "music_speech"):
            mixtures[background] = {}
            for snr in (6, 0, -6):
                mixtures[background][snr] = summarize(
                    [
                        row
                        for row in rows
                        if row["kind"] == "mixture"
                        and row["background"] == background
                        and row["snr_db"] == snr
                    ]
                )
        no_effect = [row for row in rows if row["kind"] == "background"]
        blade = [
            row for row in rows if row["kind"] == "clean" and row["family"] == "starninjas-sword"
        ]
        zero = summarize([row for row in rows if row["kind"] == "mixture" and row["snr_db"] == 0])
        bg_false = sum(row["best_match"]["label"] in targets for row in no_effect)
        blade_false = sum(row["best_match"]["label"] == "whoosh" for row in blade)
        gate = protocol["research_gate"]
        gate_checks = {
            "clean_targets": all(
                clean[label]["first_matches"] / clean[label]["examples"]
                >= gate["clean_target_min_fraction"]
                for label in targets
            ),
            "zero_db_targets": all(
                zero[label]["first_matches"] / zero[label]["examples"]
                >= gate["mixture_0db_each_target_min_fraction"]
                for label in targets
            ),
            "background_rejection": bg_false <= gate["background_target_first_max"],
            "blade_rejection": blade_false <= gate["blade_whoosh_first_max"],
        }
        metrics[condition] = {
            "clean": clean,
            "mixtures": mixtures,
            "all_0db_mixtures": zero,
            "background_examples": len(no_effect),
            "background_target_first": bg_false,
            "background_first_names": dict(
                Counter(row["best_match"]["label"] for row in no_effect)
            ),
            "blade_examples": len(blade),
            "blade_whoosh_first": blade_false,
            "research_gate_checks": gate_checks,
            "research_gate_passed": all(gate_checks.values()),
        }
    result = {
        "schema_version": "dotears.sfx-event-evaluation/1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed_research_only",
        "protocol_sha256": digest(root / "protocol.json"),
        "network_connections_blocked": True,
        "base_weights_frozen": True,
        "linear_heads_trained": True,
        "seconds": round(time.perf_counter() - started, 3),
        "sampled_peak_process_rss": peak[0],
        "unique_inputs": len(unique),
        "metrics": metrics,
        "fits": fit_audit,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("tensorflow", "scikit-learn", "librosa")
        },
        "automatic_labels_enabled": False,
        "rows": predictions,
    }
    dump(root / "results.json", result)
    dump(root / "audio-diagnostics.json", diagnostics)
    dump(root / "summary.json", {k: v for k, v in result.items() if k not in {"rows", "fits"}})
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"rows", "fits", "metrics"}}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
