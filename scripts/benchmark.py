"""Measure actual fresh-process wall time, artifact bytes and sampled process-tree RSS."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yamnet-model", type=Path)
    parser.add_argument("--asr-model", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "dotears",
        "analyze",
        args.source,
        "--profile",
        "motion",
        "--output",
        str(root / "library"),
    ]
    for flag, value in (("--yamnet-model", args.yamnet_model), ("--asr-model", args.asr_model)):
        if value:
            command += [flag, str(value.resolve())]
    started, peak_rss, samples = time.perf_counter(), 0, 0
    with (root / "cli.json").open("wb") as out, (root / "stderr.log").open("wb") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        observed = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [observed, *observed.children(recursive=True)]
                rss = 0
                for child in processes:
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                peak_rss = max(peak_rss, rss)
                samples += 1
            except psutil.Error:
                pass
            time.sleep(0.05)
        process.wait()
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise SystemExit((root / "stderr.log").read_text("utf-8", errors="replace"))
    report = json.loads((root / "cli.json").read_text("utf-8"))
    reference = Path(report["report_path"]).parent
    result = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "source": str(Path(args.source).resolve()),
        "wall_seconds": round(elapsed, 6),
        "sampled_peak_process_tree_rss_bytes": peak_rss,
        "sampling_interval_seconds": 0.05,
        "samples": samples,
        "memory_caveat": "Sampled RSS sum, not exact/private peak; shared pages may count twice.",
        "cache_hit": report["cache_hit"],
        "pipeline_metrics": report["metrics"],
        "artifact_bytes_including_reports": sum(
            p.stat().st_size for p in reference.rglob("*") if p.is_file()
        ),
        "events": len(report["events"]),
        "clips": sum(bool(e["clip_path"]) for e in report["events"]),
        "report_path": str(reference / "report.json"),
    }
    (root / "benchmark.json").write_text(json.dumps(result, indent=2), "utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
