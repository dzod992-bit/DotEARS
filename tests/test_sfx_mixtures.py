from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dotears.audiosep import file_digest


def research(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalog(tmp_path, silent=False):
    sources, backgrounds = [], []
    for i, split in enumerate(("train", "calibration", "test")):
        times = np.arange(8 * 32000) / 32000
        rng = np.random.default_rng(81 + i)
        music = 0.02 * np.sin(2 * np.pi * (300 + i * 50) * times)
        music[::16000] += 1.0
        speech = np.r_[np.zeros(4 * 32000), rng.normal(0, 0.05, 4 * 32000)]
        for role, values in (("music", music), ("speech", speech)):
            path = tmp_path / (split + "-" + role + ".wav")
            sf.write(path, values * 0 if silent else values, 32000, subtype="FLOAT")
            backgrounds.append(
                {
                    "id": split + "-" + role,
                    "path": str(path),
                    "sha256": file_digest(path),
                    "role": role,
                    "split": split,
                }
            )
        if split == "calibration":
            continue
        audio = np.zeros(64000)
        audio[32000:41600] = rng.normal(0, 0.1, 9600) * np.hanning(9600)
        path = tmp_path / (split + "-source.wav")
        sf.write(path, audio, 32000, subtype="FLOAT")
        sources.append(
            {
                "id": split + "-source",
                "path": str(path),
                "sha256": file_digest(path),
                "split": split,
                "label": "ui_sound",
                "author_group": split + "-author",
                "label_origin": "synthetic_known_fixture",
            }
        )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"sources": sources, "backgrounds": backgrounds}))
    return path


def test_mixtures_keep_effects_on_active_speech_and_apply_background_headroom(tmp_path):
    source = catalog(tmp_path)
    root = tmp_path / "mixtures"
    research("build_sfx_mixtures").build(root, source)
    result = research("verify_sfx_mixtures").verify(root)
    assert result["passed"] and result["max_component_snr_error_db"] < 1e-6
    assert result["max_component_reconstruction_error"] < 1e-7
    assert result["peak_amplitude"] <= 0.900001
    rows = json.loads((root / "corpus.json").read_text())["rows"]
    assert all(r["effect_gain"] > 0 for r in rows if r["kind"] == "mixed")
    assert any(r["background_output_gain"] < 1 for r in rows if r["kind"] == "background")


def test_silent_background_is_rejected_before_constructing_positive_mixtures(tmp_path):
    source = catalog(tmp_path, silent=True)
    with pytest.raises(ValueError, match="silent"):
        research("build_sfx_mixtures").build(tmp_path / "mixtures", source)
