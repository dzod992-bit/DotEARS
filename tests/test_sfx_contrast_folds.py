import importlib
from pathlib import Path


def test_identical_background_views_stay_in_one_grouped_fitting_fold(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    evaluator = importlib.import_module("evaluate_sfx_contrast")
    rows = [
        {"id": f"source-{i}", "label": "whoosh", "author_group": f"author-{i}"} for i in range(9)
    ]
    rows += [
        {
            "id": f"background-{i}",
            "label": "background",
            "author_group": "background-music" if i % 2 else "background-speech",
        }
        for i in range(8)
    ]
    placements = []
    for fold, (_, queries) in enumerate(evaluator.fold_indices(rows)):
        placements += [fold for i in queries if rows[i]["label"] == "background"]
    assert len(placements) == 8 and len(set(placements)) == 1
