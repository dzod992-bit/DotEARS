from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest

from dotears.clap_download import FILES, download_clap
from dotears.errors import ConfigurationError, DownloadError, MediaError
from dotears.prepare import prepare
from dotears.sfx import CONTROL_LABELS, PROMPTS, local_artifact, rank_scores, read_window


def test_sfx_rankings_keep_uncalibrated_hypotheses_separate_from_confirmed_labels():
    labels = list(PROMPTS)
    scores = np.zeros(len(labels))
    scores[labels.index("whoosh")] = 0.7
    scores[labels.index("speech")] = 0.69
    result = rank_scores(scores, labels)
    assert result["best_match"]["label"] == "whoosh"
    assert result["proposed_label"] == "whoosh"
    assert result["label"] == "unknown" and result["confidence"] is None
    assert result["margin_to_second"] == pytest.approx(0.01)
    assert not any(e["label"] in CONTROL_LABELS for e in result["top_sfx_hypotheses"])
    scores[labels.index("music")] = 0.9
    result = rank_scores(scores, labels)
    assert result["proposed_label"] is None
    assert result["best_control_match"]["label"] == "music"


@pytest.mark.parametrize("values", [[np.nan] * 12, [2.0] * 12, [0.0] * 11])
def test_invalid_sfx_scores_are_rejected(values):
    with pytest.raises(MediaError):
        rank_scores(np.array(values), list(PROMPTS))


def fake_model_response(monkeypatch, corrupt=False, revision="a" * 40):
    files = {name: (b"weights" if name == "pytorch_model.bin" else b"{}") for name in FILES}
    entries = [{"rfilename": name, "size": len(value)} for name, value in files.items()]
    for entry in entries:
        if entry["rfilename"] == "pytorch_model.bin":
            entry["lfs"] = {"sha256": hashlib.sha256(files[entry["rfilename"]]).hexdigest()}
    metadata = {"sha": revision, "siblings": entries}
    urls = []

    def response(url, timeout):
        urls.append(url)
        if "/api/models/" in url:
            return io.BytesIO(json.dumps(metadata).encode())
        assert f"/resolve/{revision}/" in url
        value = files[url.rsplit("/", 1)[1]]
        if corrupt and url.endswith("pytorch_model.bin"):
            value = b"corrupt"
        return io.BytesIO(value)

    monkeypatch.setattr("dotears.clap_download.urllib.request.urlopen", response)
    return urls


def test_clap_download_pins_files_to_one_revision_and_verifies_weight_hash(tmp_path, monkeypatch):
    urls = fake_model_response(monkeypatch)
    source = download_clap(tmp_path)
    assert source.endswith("a" * 40)
    assert {p.name for p in tmp_path.iterdir()} == set(FILES)
    assert len(urls) == 1 + len(FILES)


def test_bad_clap_download_is_not_published_as_a_prepared_model(tmp_path, monkeypatch):
    fake_model_response(monkeypatch, corrupt=True)
    destination = tmp_path / "clap"
    with pytest.raises(DownloadError, match="SHA-256"):
        prepare("clap", destination)
    assert list(tmp_path.iterdir()) == []


def test_clap_download_requires_full_commit_revision(tmp_path, monkeypatch):
    fake_model_response(monkeypatch, revision="main")
    with pytest.raises(DownloadError, match="revision"):
        download_clap(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_sfx_artifacts_cannot_escape_the_report_folder(tmp_path):
    with pytest.raises(ConfigurationError):
        local_artifact(tmp_path, "../outside.wav")
    with pytest.raises(ConfigurationError):
        read_window(tmp_path / "absent.wav", 0, 20)
