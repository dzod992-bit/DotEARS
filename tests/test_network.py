import json

from dotears.download_worker import failure_payload
from dotears.network import is_proxy_failure, proxy_summary


def test_proxy_diagnostics_redact_credentials_and_no_proxy_domains():
    result = proxy_summary(
        {
            "https": "http://private-user:secret-password@127.0.0.1:9",
            "no": "internal.private.example",
        }
    )
    rendered = json.dumps(result)
    for private in ("private-user", "secret-password", "internal.private.example"):
        assert private not in rendered
    assert result["proxies"]["https"]["host"] == "127.0.0.1"
    assert result["proxies"]["https"]["port"] == 9
    assert result["proxies"]["https"]["credentials_present"]
    assert result["no_proxy_configured"]
    assert result["inspection"] == "configuration_only_no_connection_attempt"


def test_proxy_inspection_does_not_connect_or_change_configuration(monkeypatch):
    configured = {"https": "http://127.0.0.1:9"}
    monkeypatch.setattr("dotears.network.urllib.request.getproxies", lambda: configured.copy())
    result = proxy_summary()
    assert configured == {"https": "http://127.0.0.1:9"}
    assert result["proxies"]["https"]["port"] == 9


def test_wrapped_proxy_failure_is_classified_and_sanitized():
    ProxyError = type("ProxyError", (Exception,), {})
    inner = ProxyError("sensitive proxy diagnostic")
    wrapped = RuntimeError("Unable to acquire video")
    wrapped.exc_info = (ProxyError, inner, None)
    assert is_proxy_failure(wrapped)
    payload = failure_payload(wrapped)
    assert payload["code"] == "proxy_connection_failed"
    assert "sensitive" not in payload["message"]
    assert "availability has not been established" in payload["message"]


def test_media_unavailable_is_not_misclassified_as_proxy():
    error = RuntimeError("The video is unavailable")
    assert not is_proxy_failure(error)
    assert failure_payload(error)["code"] == "download_failed"
