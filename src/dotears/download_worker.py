"""Internal yt-dlp worker. No cookies, authentication, plugins or DRM bypass."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotears.network import is_proxy_failure


def failure_payload(error: BaseException) -> dict:
    if is_proxy_failure(error):
        return {
            "code": "proxy_connection_failed",
            "message": "Cannot fetch the video URL through the configured proxy. "
            "Check the calling host's network permission and proxy availability. "
            "Run dotears doctor for redacted proxy settings. No analysis started; "
            "the video's availability has not been established.",
        }
    return {"code": "download_failed", "message": str(error)[-1500:]}


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    from yt_dlp import YoutubeDL
    from yt_dlp.globals import plugin_dirs
    from yt_dlp.utils import DownloadError

    url, directory, duration, size = sys.argv[1:]
    root, limit, budget = Path(directory), float(duration), int(size)
    plugin_dirs.value = []

    def reject(info, *, incomplete=False):
        if incomplete:
            return None
        if info.get("is_live") or info.get("duration") is None:
            return "Live streams and unknown-duration downloads are unsupported; use a local file."
        if float(info["duration"]) > limit:
            return "Duration exceeds the configured limit."
        if info.get("has_drm"):
            return "DRM media is unsupported."
        return None

    options = {
        "outtmpl": str(root / "source.%(ext)s"),
        "noplaylist": True,
        "extract_flat": "in_playlist",
        "format": "best[height<=720]/best",
        "max_filesize": budget,
        "match_filter": reject,
        "socket_timeout": 15,
        "retries": 1,
        "fragment_retries": 1,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "cachedir": False,
        "overwrites": False,
        "hls_prefer_native": True,
        "external_downloader": {"default": "native"},
    }
    # The Python API does not load CLI config files. No browser-cookie lookup is enabled.
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
        if not info or info.get("_type") in {"playlist", "multi_video"}:
            raise DownloadError("A single video is required.")
        reason = reject(info)
        if reason:
            raise DownloadError(reason)
        info = downloader.extract_info(url, download=True)
        if not info:
            raise DownloadError("No downloadable media was returned.")
        path = Path(downloader.prepare_filename(info))
        if not path.is_file():
            raise DownloadError("Download was skipped or did not produce a file.")
        result = {
            "download_path": str(path),
            "platform": info.get("extractor_key"),
            "video_id": info.get("id"),
            "title": info.get("title"),
            "author": info.get("uploader"),
            "webpage_url": info.get("webpage_url"),
        }
        (root / "result.json").write_text(json.dumps(result, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = failure_payload(exc)
        # The parent uses this bounded structured result instead of decoding duplicated logs.
        (Path(sys.argv[2]) / "error.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        print(payload["message"], file=sys.stderr)
        raise SystemExit(2) from None
