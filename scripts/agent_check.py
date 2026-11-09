"""Run a fresh CLI analysis and a cached repeat from an agent's local task.

The caller's identity must be verified in the agent's own task history. This
script does not claim that execution originated from Dot.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-models", action="store_true", help="DSP-only smoke test")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Source is not an existing local file: {source}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session = args.output.resolve() / f"run-{stamp}-{uuid4().hex[:6]}"
    command = [
        sys.executable,
        "-m",
        "dotears",
        "analyze",
        str(source),
        "--profile",
        "motion",
        "--output",
        str(session / "library"),
    ]
    if not args.no_models:
        for flag, relative in (
            ("--yamnet-model", "models/yamnet"),
            ("--asr-model", "models/whisper-tiny"),
        ):
            model = project / relative
            if not model.is_dir():
                parser.error(f"Prepared model missing: {model}. No download was attempted.")
            command += [flag, str(model)]
    session.mkdir(parents=True)
    calls = []
    for number in (1, 2):
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                timeout=300,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "OPENBLAS_NUM_THREADS": "4",
                    "OMP_NUM_THREADS": "4",
                },
            )
        except subprocess.TimeoutExpired:
            print(
                json.dumps({"error": "Analysis exceeded 300 seconds.", "session": str(session)}),
                file=sys.stderr,
            )
            return 2
        (session / f"call-{number}.stderr.log").write_text(result.stderr, "utf-8")
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            return 2
        report = json.loads(result.stdout)
        if report["cache_hit"] != (number == 2):
            raise RuntimeError("Expected a fresh first call and a cached second call.")
        calls.append(
            {
                "call": number,
                "cache_hit": report["cache_hit"],
                "seconds": round(time.perf_counter() - started, 6),
                "report_path": report["report_path"],
                "analysis_id": report["analysis_id"],
                "events": len(report["events"]),
                "clips": sum(bool(e.get("clip_path")) for e in report["events"]),
            }
        )
    if calls[0]["analysis_id"] != calls[1]["analysis_id"]:
        raise RuntimeError("Repeat call returned a different reference.")
    summary = {
        "status": "passed",
        "source": str(source),
        "calls": calls,
        "summary_path": str(session / "agent-check.json"),
        "caller_identity": "Not attested; verify the invoking agent's task history.",
        "next_step": "Read report.json/report.md and correlate with actual visual observations.",
    }
    (session / "agent-check.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
