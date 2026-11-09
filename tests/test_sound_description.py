import wave

import numpy as np
import pytest
from scipy.signal import butter, chirp, sosfilt

from dotears.sound_description import describe
from dotears.ui_acoustics import repeats_on_beat


@pytest.mark.parametrize(("frequency", "band"), [(180, "low"), (4300, "high")])
def test_acoustic_draft_uses_actual_pcm_and_keeps_mixed_attributes_separate(
    tmp_path, frequency, band
):
    rate = 16000
    time = np.arange(int(0.7 * rate)) / rate
    pulse = 0.35 * np.sin(2 * np.pi * frequency * time) * np.exp(-(((time - 0.15) / 0.017) ** 2))
    audio = pulse + 0.009 * np.sin(2 * np.pi * 700 * time)
    # Inverted stereo channels still contain a real accent on both channels.
    stereo = np.column_stack((audio, -audio))
    clip = tmp_path / "accent.wav"
    with wave.open(str(clip), "wb") as wav:
        wav.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(np.round(stereo * 32767).astype("<i2").tobytes())

    described = describe(clip, 0.15, "ui_sound", [])
    assert described["acoustic_evidence"]["dominant_band"] == band
    assert described["acoustic_evidence"]["strong_local_accent"]
    assert abs(described["acoustic_evidence"]["narrow_energy_band_hz"] - frequency) <= 50
    assert described["generation_prompt"]["basis"] == "provisional_family_and_local_acoustics"
    assert described["generation_prompt"]["duration_seconds"] is None

    mixed = describe(clip, 0.15, "ui_sound", ["speech", "music"])
    assert mixed["acoustic_evidence"] == described["acoustic_evidence"]
    assert mixed["generation_prompt"]["basis"] == "provisional_family"
    assert describe(clip, 0.15, "unknown", ["speech"])["generation_prompt"] is None


@pytest.mark.parametrize("form", ["click_like", "beep"])
def test_ui_form_drives_prompt_only_when_the_clip_is_clear(tmp_path, form):
    rate = 16000
    time = np.arange(rate * 7 // 10) / rate
    if form == "beep":
        envelope = np.clip(np.minimum(time - 0.14, 0.26 - time) / 0.006, 0, 1)
        samples = 0.4 * np.sin(2 * np.pi * 1750 * time) * envelope
        wording = "UI beep"
    else:
        samples = np.zeros_like(time)
        samples[2400:2406] = [0.1, 0.9, -0.65, 0.4, -0.15, 0.05]
        wording = "UI click"
    clip = tmp_path / f"{form}.wav"
    with wave.open(str(clip), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(np.round(samples * 32767).astype("<i2").tobytes())
    clear = describe(clip, 0.15, "ui_sound", [])
    assert clear["ui_sound_form"] == form
    assert wording in clear["generation_prompt"]["text"]
    unchecked = describe(clip, 0.15, "ui_sound", None)
    assert unchecked["ui_sound_form"] == "unknown"
    assert wording not in unchecked["generation_prompt"]["text"]
    if form == "beep":
        mixed = describe(clip, 0.15, "ui_sound", ["speech", "music"])
        assert mixed["ui_sound_form"] == "unknown"


@pytest.mark.parametrize("form", ["click_like", "tap_like"])
def test_short_mixed_transient_drives_prompt_with_ui_context(tmp_path, form):
    rate = 32000
    time = np.arange(rate * 7 // 10) / rate
    rng = np.random.default_rng(16)
    bed = (
        0.045 * np.sin(2 * np.pi * 117 * time)
        + 0.018 * np.sin(2 * np.pi * 213 * time)
        + 0.005 * rng.normal(size=len(time))
    )
    if form == "click_like":
        bandpass = butter(3, [2500, 9500], btype="bandpass", fs=rate, output="sos")
        shape = sosfilt(bandpass, rng.normal(size=192))
        effect = 0.37 * shape / np.max(abs(shape)) * np.exp(-time[:192] / 0.0018)
        wording = "crisp, dry UI click"
    else:
        t = time[:320]
        effect = 0.35 * np.sin(2 * np.pi * 685 * t) * np.minimum(t / 0.001, 1) * np.exp(-t / 0.004)
        wording = "soft, muted UI tap"

    path = tmp_path / "mixed.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        start = round(0.15 * rate)
        bed[start : start + len(effect)] += effect
        audio.writeframes(np.round(bed * 32767).astype("<i2").tobytes())

    background = ["speech", "music"]
    mixed = describe(path, 0.15, "ui_sound", background)
    assert mixed["ui_sound_form"] == form
    assert wording in mixed["generation_prompt"]["text"]
    assert mixed["generation_prompt"]["basis"] == "provisional_ui_family_and_short_transient"
    draft = describe(path, 0.15, "unknown", background)
    assert draft["ui_sound_form"] == "unknown"
    assert draft["generation_prompt"]["basis"] == "observed_short_transient_candidate"
    assert wording in draft["generation_prompt"]["text"]
    rhythmic = describe(path, 0.15, "ui_sound", background, rhythmic_cue=lambda *_: True)
    assert rhythmic["ui_sound_form"] == "unknown"
    assert wording not in rhythmic["generation_prompt"]["text"]


def test_repeated_beat_is_rejected_but_single_cue_is_not(tmp_path):
    rate = 16000
    rng = np.random.default_rng(37)
    signal = 0.002 * rng.normal(size=rate * 4)
    tick = 0.13 * rng.normal(size=72) * np.exp(-np.arange(72) / 26)
    for start in (0.4, 0.9, 1.4, 1.9, 2.4, 3.17):
        first = round(start * rate)
        signal[first : first + len(tick)] += tick
    path = tmp_path / "rhythm.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(np.round(signal * 32767).astype("<i2").tobytes())
    assert repeats_on_beat(path, 0.75, 120, 0.151, "click_like")
    assert not repeats_on_beat(path, 3, 120, 0.171, "click_like")


def test_ringing_draft_keeps_mixed_event_unclassified(tmp_path):
    rate = 16000
    time = np.arange(rate * 7 // 10) / rate
    relative = np.maximum(time - 0.15, 0)
    envelope = np.clip(relative / 0.006, 0, 1) * np.exp(-relative / 0.065)
    envelope *= np.clip((0.23 - relative) / 0.01, 0, 1)
    audio = envelope * (
        0.36 * np.sin(2 * np.pi * 1250 * time) + 0.24 * np.sin(2 * np.pi * 1750 * time)
    )
    clip = tmp_path / "ring.wav"
    with wave.open(str(clip), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(np.round(audio * 32767).astype("<i2").tobytes())

    no_context = describe(clip, 0.15, "unknown", ["speech", "music"])
    assert no_context["generation_prompt"] is None
    observed = describe(clip, 0.15, "unknown", ["speech", "music"], ring_context=True)
    assert observed["ui_sound_form"] == "unknown"
    assert observed["ui_form_evidence"]["acoustic_form_candidate"] == "chime"
    assert observed["generation_prompt"]["basis"] == "observed_ringing_candidate"
    assert "resonance near" in observed["generation_prompt"]["text"]


@pytest.mark.parametrize(("direction", "frequencies"), [("up", (660, 1780)), ("down", (1780, 660))])
def test_unknown_glide_gets_a_prompt_describing_measured_pitch(tmp_path, direction, frequencies):
    rate = 16000
    rng = np.random.default_rng(45)
    values = rng.normal(0, 0.012, round(rate * 0.7))
    time = np.arange(round(rate * 0.185)) / rate
    envelope = np.minimum(np.minimum(time, 0.185 - time) / 0.018, 1)
    values[2400 : 2400 + len(time)] += (
        0.2 * chirp(time, frequencies[0], time[-1], frequencies[1], method="logarithmic") * envelope
    )
    path = tmp_path / "glide.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(np.round(values * 32767).astype("<i2").tobytes())

    described = describe(path, 0.15, "unknown", ["music", "speech"])
    assert described["ui_sound_form"] == "unknown"
    shape = described["motion_shape"]
    assert shape["direction"] == direction
    assert abs(shape["start_hz"] - frequencies[0]) < 200
    assert abs(shape["end_hz"] - frequencies[1]) < 200
    prompt = described["generation_prompt"]
    assert prompt["basis"] == "observed_pitch_glide"
    assert str(shape["start_hz"]) in prompt["text"]
    assert str(shape["end_hz"]) in prompt["text"]
