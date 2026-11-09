from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dotears.acquisition import acquire
from dotears.errors import DependencyError, DownloadError, LimitError, ProxyConnectionError


def test_url_dependency_missing(monkeypatch):
    monkeypatch.setattr("dotears.acquisition.importlib.util.find_spec", lambda _: None)
    with pytest.raises(DependencyError):
        with acquire("https://example.com/video", 120, 10000, 1):
            pass


def test_worker_failure_returns_local_file_fallback(monkeypatch):
    real = subprocess.Popen
    temporary = []

    def fake(command, **kwargs):
        temporary.append(Path(command[4]))
        return real([sys.executable, "-c", "raise SystemExit(1)"], **kwargs)

    monkeypatch.setattr("dotears.acquisition.subprocess.Popen", fake)
    with pytest.raises(DownloadError, match="local file"):
        with acquire("https://example.com/video", 120, 10000, 2):
            pass
    assert temporary and not temporary[0].exists()


def test_download_timeout_kills_worker_and_cleans_up(monkeypatch):
    real = subprocess.Popen
    workers, temporary = [], []

    def fake(command, **kwargs):
        temporary.append(Path(command[4]))
        child = real([sys.executable, "-c", "import time; time.sleep(10)"], **kwargs)
        workers.append(child)
        return child

    monkeypatch.setattr("dotears.acquisition.subprocess.Popen", fake)
    with pytest.raises(LimitError, match="timed out"):
        with acquire("https://example.com/video", 120, 10000, 0.1):
            pass
    assert workers[0].poll() is not None
    assert not temporary[0].exists()


def test_proxy_failure_is_distinct_and_does_not_remove_inherited_proxy(monkeypatch):
    import os

    real = subprocess.Popen
    folders = []
    proxy = "http://127.0.0.1:9"
    monkeypatch.setenv("HTTPS_PROXY", proxy)

    def fake(command, **kwargs):
        assert "env" not in kwargs
        assert os.environ["HTTPS_PROXY"] == proxy
        root = Path(command[4])
        folders.append(root)
        (root / "error.json").write_text(
            '{"code":"proxy_connection_failed","message":"internal proxy diagnostic"}',
            encoding="utf-8",
        )
        return real([sys.executable, "-c", "raise SystemExit(2)"], **kwargs)

    monkeypatch.setattr("dotears.acquisition.subprocess.Popen", fake)
    with pytest.raises(ProxyConnectionError, match="availability is unknown"):
        with acquire("https://example.com/video", 60, 10000, 2):
            pass
    assert os.environ["HTTPS_PROXY"] == proxy
    assert not folders[0].exists()


@pytest.mark.parametrize("source", ["ftp://example.com/video", "https://user:pass@example.com/v"])
def test_unsupported_or_credentialed_url(source):
    with pytest.raises(DownloadError):
        with acquire(source, 120, 10000, 1):
            pass
