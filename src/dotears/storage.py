"""Temporary workspaces that respect the caller's existing directory access."""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _remove_windows_workspace(path: Path, parent: Path) -> None:
    """Retry only transient Windows sharing locks, without altering any access rules."""
    for attempt in range(5):
        # Refuse a replaced directory/junction before every recursive cleanup attempt.
        if path.parent != parent or path.resolve() != path:
            raise RuntimeError("Temporary workspace target was replaced or escaped its parent.")
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (32, 33) or attempt == 4:
                raise
            time.sleep(0.05 * (2**attempt))


@contextmanager
def temporary_directory(*, prefix: str, dir: str | Path | None = None) -> Iterator[str]:
    """Use inherited Windows access; retain standard private temp directories on POSIX.

    Windows treats mkdir(mode=0o700) specially: it installs an owner-only ACL.
    A restricted local-agent token may then lose access even inside its allowed
    workspace. Other mkdir modes inherit the parent's existing ACL on Windows.
    This helper changes no existing ACL, permission profile or system setting.
    """
    if os.name != "nt":
        with tempfile.TemporaryDirectory(prefix=prefix, dir=dir) as name:
            yield name
        return
    if not prefix or any(char in prefix for char in "/\\:"):
        raise ValueError("A temporary-directory prefix must be a plain filename prefix.")
    parent = Path(dir if dir is not None else tempfile.gettempdir()).resolve()
    for _ in range(100):
        path = parent / f"{prefix}{secrets.token_hex(12)}"
        try:
            path.mkdir(mode=0o755)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("Could not allocate a unique temporary workspace.")
    try:
        yield str(path)
    finally:
        # A caller can atomically rename staging into its final destination.
        if path.parent != parent:
            raise RuntimeError("Temporary workspace is outside its expected parent.")
        _remove_windows_workspace(path, parent)
