"""Run a frozen, offline CLAP audio-reference experiment on a prepared local corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import threading
import time
import wave
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    if (root / "results.json").exists():
        raise ValueError("Results already exist; prepare a separate experiment directory.")
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    corpus_path = root / "corpus.json"
    if digest(corpus_path) != protocol["corpus_sha256"]:
        raise ValueError("Corpus changed after the protocol was frozen.")
    corpus = json.loads(corpus_path.read_text("utf-8"))
    if digest(Path(corpus["catalog_path"])) != protocol["catalog_sha256"]:
        raise ValueError("Human reference catalog changed after protocol freeze.")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    os.environ["OMP_NUM_THREADS"] = str(protocol["threads"])

    def reject_network(*_args, **_kwargs):
        raise RuntimeError("The offline SFX experiment attempted a network connection.")

    socket.create_connection = reject_network
    socket.socket.connect = reject_network
    started = time.perf_counter()
    import numpy as np
    import psutil
    import torch
    from transformers import ClapModel, ClapProcessor

    from dotears.core import model_fingerprint
    from dotears.sfx import PROMPTS, rank_scores, read_window
    from dotears.sfx_references import METHOD, rank_references

    if model_fingerprint(str(args.model.resolve())) != protocol["model_sha256"]:
        raise ValueError("CLAP model differs from the frozen protocol.")
    if (
        hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest()
        != protocol["prompt_sha256"]
    ):
        raise ValueError("Text baseline prompts differ from the frozen protocol.")
    if METHOD != protocol["method"]:
        raise ValueError("Audio-reference method differs from the frozen protocol.")
    process, stop, peak = psutil.Process(), threading.Event(), [0]

    def sample_memory():
        while not stop.wait(0.1):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    torch.set_num_threads(protocol["threads"])
    torch.set_num_interop_threads(1)
    model = (
        ClapModel.from_pretrained(
            str(args.model.resolve()),
            local_files_only=True,
            use_safetensors=False,
            weights_only=True,
        )
        .to("cpu")
        .eval()
    )
    processor = ClapProcessor.from_pretrained(str(args.model.resolve()), local_files_only=True)
    labels = list(PROMPTS)
    texts = [text for entries in PROMPTS.values() for text in entries]
    with torch.inference_mode():
        text = model.get_text_features(**processor(text=texts, padding=True, return_tensors="pt"))
        text = torch.nn.functional.normalize(text.reshape(len(labels), 3, -1).mean(dim=1), dim=-1)
        text = text.cpu().numpy()
    cache = {}

    def embed(row):
        path = Path(row["audio_path"])
        if digest(path) != row["audio_sha256"]:
            raise ValueError(f"Input audio changed: {path}")
        key = row["audio_sha256"]
        if key not in cache:
            with wave.open(str(path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            samples, rate = read_window(path, 0, duration)
            if rate != 48000:
                raise ValueError("Prepared corpus must use 48 kHz PCM.")
            inputs = processor(
                audio=samples, sampling_rate=rate, return_tensors="pt", padding=protocol["padding"]
            )
            with torch.inference_mode():
                vector = torch.nn.functional.normalize(model.get_audio_features(**inputs), dim=-1)
            cache[key] = vector[0].cpu().numpy()
        return cache[key]

    references = corpus["references"]
    embeddings = np.stack([embed(row) for row in references])
    predictions = []
    for number, row in enumerate(corpus["cases"], 1):
        vector = embed(row)
        exclude = row["kind"] in {"clean_preview", "clean_context", "mixture"}
        reference = rank_references(
            vector,
            embeddings,
            references,
            excluded_groups={row["group_id"]} if exclude else set(),
            excluded_source_hashes={row["source_sha256"]} if exclude else set(),
        )
        predictions.append(
            {
                "case_id": row["id"],
                "reference": reference,
                "text": rank_scores(vector @ text.T, labels),
            }
        )
        if number % 25 == 0:
            print(json.dumps({"processed": number, "total": len(corpus["cases"])}), flush=True)
    # Join evaluation metadata only after inference. Reference labels build prototypes;
    # query labels never select a prototype, threshold, prediction or preprocessing condition.
    metadata = {row["id"]: row for row in corpus["cases"]}
    results = [{**metadata[row["case_id"]], **row} for row in predictions]
    for row in results:
        row["reference_class_representable"] = (
            row.get("human_label") in row["reference"]["available_labels"]
            if "human_label" in row
            else None
        )
    stop.set()
    sampler.join()
    result = {
        "schema_version": "dotears.sfx-reference-evaluation/1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed_hypotheses_only",
        "protocol_sha256": digest(root / "protocol.json"),
        "protocol": protocol,
        "network_connections_blocked": True,
        "seconds": round(time.perf_counter() - started, 3),
        "sampled_peak_process_rss": peak[0],
        "unique_audio_embeddings": len(cache),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "rows": results,
    }
    with (root / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    keys = sorted(cache)
    np.savez_compressed(
        root / "embeddings.npz",
        hashes=np.array(keys),
        embeddings=np.stack([cache[key] for key in keys]),
    )
    print(
        json.dumps({key: value for key, value in result.items() if key not in {"rows", "protocol"}})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
