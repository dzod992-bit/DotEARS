"""Opt-in real-weight checks; no network or implicit downloads in the test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dotears.models import classify, transcribe


@pytest.mark.model
def test_offline_yamnet_real_weights(media_files, tmp_path):
    directory = os.environ.get("DOTEARS_TEST_YAMNET")
    if not directory:
        pytest.skip("Set DOTEARS_TEST_YAMNET to prepared local SavedModel")
    from dotears.media import extract_audio

    wav = tmp_path / "test.wav"
    extract_audio(media_files["video"], wav)
    result = classify(wav, Path(directory), 0.3, 2)
    assert result["status"] == "completed"
    for event in result["events"]:
        assert 0 <= event["start"] < event["end"] <= 4.1
        assert 0.3 <= event["score"] <= 1
        assert event["confidence"] is None


@pytest.mark.model
def test_offline_whisper_real_weights(media_files):
    directory = os.environ.get("DOTEARS_TEST_ASR")
    if not directory:
        pytest.skip("Set DOTEARS_TEST_ASR to prepared local CTranslate2 model")
    result = transcribe(media_files["wav"], Path(directory), "en", 2)
    assert result["status"] == "completed"
    assert result["method"] == "faster_whisper_cpu_int8"
    # Percussive tones have no ground-truth speech; do not invent an ASR accuracy claim.
    assert isinstance(result["segments"], list)
