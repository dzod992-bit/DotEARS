import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


def script_module():
    path = Path(__file__).parents[1] / "scripts/prepare_audiosep.py"
    spec = importlib.util.spec_from_file_location("prepare_audiosep_download_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resumed_model_download_checks_the_complete_hash_before_publication(tmp_path, monkeypatch):
    module = script_module()
    path = tmp_path / "model.bin"
    path.with_suffix(".bin.part").write_bytes(b"abc")

    class Response(io.BytesIO):
        status = 206

    def response(request, timeout):
        assert request.get_header("Range") == "bytes=3-"
        return Response(b"def")

    monkeypatch.setattr(module.urllib.request, "urlopen", response)
    result = module.download(
        "https://example.invalid/model", path, hashlib.sha256(b"abcdef").hexdigest(), 6
    )
    assert path.read_bytes() == b"abcdef"
    assert result["bytes"] == 6 and not path.with_suffix(".bin.part").exists()


@pytest.mark.parametrize(
    "content,limit,message", [(b"bad hash", 8, "SHA-256"), (b"too large", 3, "byte/time budget")]
)
def test_unverified_or_oversized_model_is_never_published(
    tmp_path, monkeypatch, content, limit, message
):
    module = script_module()

    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response(content))
    path = tmp_path / "model.bin"
    with pytest.raises(ValueError, match=message):
        module.download(
            "https://example.invalid/model", path, hashlib.sha256(b"trusted").hexdigest(), limit
        )
    assert not path.exists()
