"""Read-only network diagnostics; never remove or override inherited proxy settings."""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping
from urllib.parse import urlsplit


def proxy_summary(proxies: Mapping[str, str] | None = None) -> dict:
    configured = urllib.request.getproxies() if proxies is None else proxies
    endpoints = {}
    for name, value in configured.items():
        if name.lower() in {"no", "no_proxy"} or not value:
            continue
        try:
            parsed = urlsplit(value if "://" in value else "http://" + value)
            endpoints[name] = {
                "scheme": parsed.scheme,
                "host": parsed.hostname,
                "port": parsed.port,
                "credentials_present": bool(parsed.username or parsed.password),
            }
        except ValueError:
            endpoints[name] = {"configured": True, "endpoint_parseable": False}
    return {
        "inspection": "configuration_only_no_connection_attempt",
        "proxies": endpoints,
        "no_proxy_configured": bool(configured.get("no") or configured.get("no_proxy")),
        "note": "URL downloads need host-authorized network access. Proxy settings are preserved.",
    }


def is_proxy_failure(error: BaseException) -> bool:
    """Recognize wrapped yt-dlp/requests proxy failures without requiring either package."""
    pending, seen = [error], set()
    while pending and len(seen) < 32:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if type(current).__name__ == "ProxyError":
            return True
        for attribute in ("__cause__", "__context__", "cause", "reason"):
            wrapped = getattr(current, attribute, None)
            if isinstance(wrapped, BaseException):
                pending.append(wrapped)
        info = getattr(current, "exc_info", None)
        if isinstance(info, tuple) and len(info) > 1 and isinstance(info[1], BaseException):
            pending.append(info[1])
    # Some extractors preserve only a rendered error after crossing a handler boundary.
    message = str(error).lower()
    return "unable to connect to proxy" in message or "proxyerror(" in message
