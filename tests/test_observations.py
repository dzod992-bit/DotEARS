from __future__ import annotations

import numpy as np
import pytest
from conftest import write_signal

from dotears.audio import perceive
from dotears.errors import MediaError
from dotears.models import score_intervals


def test_silence_has_no_invented_events(tmp_path):
    wav = tmp_path / "silence.wav"
    write_signal(wav, silent=True)
    result = perceive(wav, "motion", 10)
    assert result["silence"] and result["events"] == []
    assert result["rhythm"]["bpm"] is None


def test_yamnet_intervals_preserve_vocabulary_and_overlap():
    scores = np.array([[0.8, 0.5, 0], [0.7, 0.6, 0], [0.1, 0.2, 0.9], [0.8, 0.1, 0]])
    events = score_intervals(scores, ["Speech", "Music", "Silence"], 2.1, 0.3)
    speech = [e for e in events if e["label"] == "Speech"]
    assert len(speech) == 2
    assert speech[0]["start"] == 0 and speech[0]["end"] == pytest.approx(1.455)
    assert speech[1]["start"] == pytest.approx(1.44) and speech[1]["end"] == 2.1
    assert speech[0]["confidence"] is None
    assert not any(e["label"] == "Silence" for e in events)


def test_bad_classifier_output_is_rejected():
    with pytest.raises(MediaError):
        score_intervals(np.array([[np.nan]]), ["Speech"], 1, 0.3)
