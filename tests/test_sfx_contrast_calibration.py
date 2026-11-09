import copy
import importlib
import json
from pathlib import Path

import pytest

from dotears.audiosep import file_digest
from dotears.sfx_contrast import FEATURE_NAMES, SCHEMA, VERSION
from dotears.sfx_taxonomy import UI_TAXONOMY, UI_TYPES


@pytest.mark.parametrize("coefficient", [1.0, 0.01])
def test_separate_calibration_can_fix_old_thresholds_then_test_only_disables(
    tmp_path, monkeypatch, coefficient
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    calibration = importlib.import_module("calibrate_sfx_contrast")
    rows = []
    for split, count in (("calibration", 3), ("test", 2)):
        for k, label in enumerate(UI_TYPES):
            for i in range(count):
                values = [0.0] * len(FEATURE_NAMES)
                values[k] = 3.0
                rows.append(
                    {
                        "id": f"{split}-{label}-{i}",
                        "split": split,
                        "label": label,
                        "snr_db": 0,
                        "family": f"author-{split}-{label}-{i}",
                        "author_group": f"author-{split}-{label}-{i}",
                        "source_sha256": f"source-{split}-{label}-{i}",
                        "evidence": {
                            "features": values,
                            "rms": 0.1,
                            "raw_target_scores": dict.fromkeys(UI_TYPES, 0.2),
                        },
                    }
                )
    false = copy.deepcopy(rows[-1])
    false.update(id="test-background", label="background")
    false["evidence"]["features"][2] = 4.0
    rows.append(false)
    path = tmp_path / "features.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    model = {"fixture": "model"}
    (tmp_path / "feature-run.json").write_text(
        json.dumps({"models": model, "features_sha256": file_digest(path)})
    )
    heads = {
        label: {
            "kind": "linear",
            "feature_count": 3,
            "feature_indices": [0, 1, 2],
            "mean": [0.0] * 3,
            "scale": [1.0] * 3,
            "coefficients": [float(j == k) * coefficient for j in range(3)],
            "intercept": 0.0,
            "enabled": False,
            "threshold": 100.0,
            "competition_center": 0.0,
            "competition_scale": 1.0,
        }
        for k, label in enumerate(UI_TYPES)
    }
    policy = {
        "schema_version": SCHEMA,
        "version": VERSION,
        "label_taxonomy": UI_TAXONOMY,
        "feature_names": FEATURE_NAMES,
        "models": model,
        "heads": heads,
    }
    (tmp_path / "evaluation.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"variant": "semantic", "C": 1.0, "learner": "linear", "policy": policy}
                ],
                "checks": "QUARANTINED_TEST_RESULTS",
            }
        )
    )
    (tmp_path / "posthoc-plan-before-test.json").write_text(
        json.dumps({"normalizations": ["original_oof", "calibration_medians"]})
    )
    output = tmp_path / "calibrated"
    calibration.select(tmp_path, output)
    selected = json.loads((output / "selected-before-test.json").read_text())
    assert all(
        h["enabled"] and h["threshold"] == pytest.approx(3.0 * coefficient)
        for h in selected["heads"].values()
    )
    assert not (output / "policy.json").exists()
    calibration.test(tmp_path, output)
    final = json.loads((output / "policy.json").read_text())
    assert final["heads"]["whoosh"]["enabled"] and final["heads"]["notification"]["enabled"]
    assert not final["heads"]["ui_sound"]["enabled"]
    for label in UI_TYPES:
        assert final["heads"][label]["threshold"] == selected["heads"][label]["threshold"]
