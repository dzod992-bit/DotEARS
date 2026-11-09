"""Downloads run in a disposable, time- and disk-bounded child process."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotears.errors import DependencyError, DownloadError, LimitError, ProxyConnectionError
from dotears.storage import temporary_directory


def is_url(value: str) -> bool:
    return urlsplit(value).scheme.lower() in {"http", "https"}


@contextmanager
def acquire(source: str, max_duration: float, max_bytes: int, timeout: float):
    if not is_url(source):
        # Windows drive letters are local paths; all other URI schemes are rejected.
        if "://" in source:
            raise DownloadError("Only public HTTP(S) video URLs or local files are supported.")
        yield (
            Path(source).expanduser().resolve(),
            {"type": "local", "path": str(Path(source).resolve())},
        )
        return
    parts = urlsplit(source)
    if not parts.hostname or parts.username or parts.password:
        raise DownloadError("A video URL must have a hostname and no embedded credentials.")
    if not importlib.util.find_spec("yt_dlp"):
        raise DependencyError("URL input requires: pip install 'dotears[download]'")
    with temporary_directory(prefix="dotears-download-") as directory:
        root = Path(directory)
        command = [
            sys.executable,
            "-m",
            "dotears.download_worker",
            source,
            str(root),
            str(max_duration),
            str(max_bytes),
        ]
        # Redirect to files to avoid a full pipe deadlocking the monitor.
        with (root / "worker.log").open("wb") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            deadline = time.monotonic() + timeout
            try:
                while child.poll() is None:
                    if time.monotonic() > deadline:
                        raise LimitError("Video download timed out. Try a local file.")
                    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
                    if total > max_bytes:
                        raise LimitError(
                            "Download exceeded the size budget. Try a smaller local file."
                        )
                    time.sleep(0.1)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
        if child.returncode:
            detail = (root / "worker.log").read_text(encoding="utf-8", errors="replace")[-1500:]
            try:
                failure = json.loads((root / "error.json").read_text("utf-8"))
            except (OSError, ValueError):
                failure = {}
            if isinstance(failure, dict):
                if failure.get("code") == "proxy_connection_failed":
                    raise ProxyConnectionError(
                        "Cannot fetch the video URL through the configured proxy. "
                        "Check the calling host's network permission and proxy availability; "
                        "run dotears doctor for redacted proxy settings. "
                        "No analysis started and video availability is unknown."
                    )
                if isinstance(failure.get("message"), str):
                    detail = failure["message"][-1500:]
            raise DownloadError(
                f"URL acquisition failed before analysis; an authorized local file is a fallback. "
                f"{detail}"
            )
        try:
            metadata: dict[str, Any] = json.loads((root / "result.json").read_text("utf-8"))
            path = Path(metadata.pop("download_path")).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("invalid download path")
        except (OSError, ValueError, KeyError) as exc:
            raise DownloadError("Downloader did not return a valid media file.") from exc
        yield path, {"type": "url", "url": source, **metadata}
