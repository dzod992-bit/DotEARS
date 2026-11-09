import hashlib
import json

import pytest

from dotears import audiosep
from dotears.errors import ConfigurationError


def fixture_preparation(root, monkeypatch):
    (root / "preparation.json").write_text(
        json.dumps({"github_revision": audiosep.SOURCE_REVISION}), "utf-8"
    )
    checkpoint = root / "checkpoint"
    checkpoint.write_bytes(b"safe test placeholder, not a real checkpoint")
    monkeypatch.setattr(
        audiosep,
        "PINNED_FILES",
        {"checkpoint": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
    )
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    return tokenizer


def test_modified_model_code_is_rejected_before_torch_or_source_execution(tmp_path, monkeypatch):
    fixture_preparation(tmp_path, monkeypatch)
    source = tmp_path / "untrusted.py"
    marker = tmp_path / "should-not-exist"
    source.write_text(f"open({str(marker)!r}, 'w').write('executed')", "utf-8")
    monkeypatch.setitem(
        audiosep.PINNED_FILES, "untrusted.py", hashlib.sha256(b"expected original code").hexdigest()
    )
    with pytest.raises(ConfigurationError, match="modified AudioSep"):
        audiosep.LocalAudioSep(tmp_path)
    assert not marker.exists()


def test_tokenizer_manifest_cannot_escape_the_prepared_model_directory(tmp_path, monkeypatch):
    tokenizer = fixture_preparation(tmp_path, monkeypatch)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    (tokenizer / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "../outside.json",
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                ]
            }
        ),
        "utf-8",
    )
    with pytest.raises(ConfigurationError, match="tokenizer changed"):
        audiosep.verify_prepared_audiosep(tmp_path)
