"""Explicit one-time model downloads. No media is ever uploaded."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from dotears.errors import ConfigurationError, DependencyError, DownloadError
from dotears.storage import temporary_directory

YAMNET_URL = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"


def unpack_model(archive: Path, target: Path) -> None:
    """Extract regular files/directories only, with traversal and expansion bounds."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if sum(m.size for m in members) > 80 * 1024**2 or len(members) > 1000:
            raise DownloadError("Model archive exceeds extraction budget.")
        for member in members:
            destination = (target / member.name).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise DownloadError("Unsafe path in model archive.")
            if not member.isdir() and not member.isfile():
                raise DownloadError("Model archive contains a link or special file.")
        for member in members:
            destination = target / member.name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise DownloadError("Unreadable model archive member.")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)


def prepare(kind: str, destination: Path, size: str = "tiny") -> dict:
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise ConfigurationError(f"Model destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary_directory(prefix=".dotears-model-", dir=destination.parent) as temporary:
        root = Path(temporary)
        model = root / "model"
        model.mkdir()
        if kind == "yamnet":
            archive = root / "yamnet.tar.gz"
            try:
                with (
                    urllib.request.urlopen(YAMNET_URL, timeout=60) as response,
                    archive.open("wb") as out,
                ):
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 32 * 1024**2:
                            raise DownloadError("YAMNet download exceeded 32 MiB.")
                        out.write(chunk)
                unpack_model(archive, model)
            except (OSError, tarfile.TarError) as exc:
                raise DownloadError(
                    f"Cannot download official YAMNet; install manually: {exc}"
                ) from exc
            if not (model / "saved_model.pb").is_file():
                raise DownloadError("YAMNet archive did not contain a SavedModel.")
            source = YAMNET_URL
        elif kind == "whisper":
            if size not in {"tiny", "base", "small"}:
                raise ConfigurationError("Whisper size must be tiny, base or small (multilingual).")
            try:
                from faster_whisper.utils import download_model
            except ImportError as exc:
                raise DependencyError(
                    "Model preparation requires: pip install 'dotears[asr]'"
                ) from exc
            try:
                download_model(size, output_dir=str(model), use_auth_token=False)
            except Exception as exc:
                raise DownloadError(f"Whisper model download failed: {exc}") from exc
            source = f"https://huggingface.co/Systran/faster-whisper-{size}"
        elif kind == "clap":
            from dotears.clap_download import download_clap

            try:
                source = download_clap(model)
            except (OSError, ValueError) as exc:
                raise DownloadError(f"Official CLAP model download failed: {exc}") from exc
        else:
            raise ConfigurationError("Unknown model kind.")
        manifest = {
            "model": kind,
            "source": source,
            "license": "Apache-2.0" if kind in {"yamnet", "clap"} else "MIT",
            "files": [],
            "weights_bytes": 0,
        }
        for path in sorted(p for p in model.rglob("*") if p.is_file()):
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            manifest["files"].append(
                {
                    "path": path.relative_to(model).as_posix(),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
            )
            manifest["weights_bytes"] += path.stat().st_size
        (model / "dotears-model.json").write_text(json.dumps(manifest, indent=2), "utf-8")
        model.rename(destination)
    return {"path": str(destination), **manifest}
