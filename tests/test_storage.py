from pathlib import Path

import pytest

from dotears.storage import temporary_directory


def test_workspace_is_usable_and_removed(tmp_path):
    with temporary_directory(prefix="work-", dir=tmp_path) as name:
        folder = Path(name)
        assert folder.parent == tmp_path
        (folder / "result.txt").write_text("result", encoding="utf-8")
        assert (folder / "result.txt").read_text("utf-8") == "result"
    assert not folder.exists()


def test_workspace_is_cleaned_after_error(tmp_path):
    with pytest.raises(RuntimeError, match="test failure"):
        with temporary_directory(prefix="work-", dir=tmp_path) as name:
            folder = Path(name)
            (folder / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("test failure")
    assert not folder.exists()


def test_atomic_publish_survives_workspace_exit(tmp_path):
    final = tmp_path / "published"
    with temporary_directory(prefix="work-", dir=tmp_path) as name:
        folder = Path(name)
        (folder / "report.json").write_text("{}", encoding="utf-8")
        folder.rename(final)
    assert not folder.exists()
    assert (final / "report.json").read_text("utf-8") == "{}"


def test_transient_windows_sharing_lock_retries_only_owned_workspace(tmp_path, monkeypatch):
    from dotears import storage

    folder = tmp_path / "owned"
    folder.mkdir()
    (folder / "worker.log").write_text("log")
    real = storage.shutil.rmtree
    calls = []
    delays = []

    def sharing_then_remove(path):
        calls.append(path)
        if len(calls) < 3:
            error = PermissionError("busy")
            error.winerror = 32
            raise error
        real(path)

    monkeypatch.setattr(storage.shutil, "rmtree", sharing_then_remove)
    monkeypatch.setattr(storage.time, "sleep", delays.append)
    storage._remove_windows_workspace(folder, tmp_path)
    assert calls == [folder] * 3 and delays == [0.05, 0.1] and not folder.exists()


def test_windows_access_denied_is_not_retried_or_hidden(tmp_path, monkeypatch):
    from dotears import storage

    folder = tmp_path / "owned"
    folder.mkdir()
    calls = []

    def denied(path):
        calls.append(path)
        error = PermissionError("denied")
        error.winerror = 5
        raise error

    monkeypatch.setattr(storage.shutil, "rmtree", denied)
    with pytest.raises(PermissionError, match="denied"):
        storage._remove_windows_workspace(folder, tmp_path)
    assert calls == [folder] and folder.exists()
