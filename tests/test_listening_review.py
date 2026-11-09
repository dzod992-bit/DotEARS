from __future__ import annotations

from copy import deepcopy

from scripts.build_listening_review import starter_ids


def attack(name: str, peak: float, score: float = 0.5) -> dict:
    return {
        "id": name,
        "peak_time": peak,
        "start": peak - 0.15,
        "end": peak + 0.55,
        "score": score,
    }


def test_starter_avoids_nearly_identical_excerpts_without_dropping_attacks():
    attacks = [attack("a", 1, 1), attack("b", 1.12, 0.9)]
    attacks += [attack(f"later_{i}", i + 2) for i in range(20)]
    original = deepcopy(attacks)
    selected = starter_ids(attacks, [{"event_id": "b"}])
    assert "a" in selected and "b" not in selected
    assert len(selected) == 12
    assert attacks == original
    chosen = [e for e in attacks if e["id"] in selected]
    for i, first in enumerate(chosen):
        for second in chosen[i + 1 :]:
            overlap = max(
                0, min(first["end"], second["end"]) - max(first["start"], second["start"])
            )
            assert overlap / 0.7 <= 0.8


def test_tiny_near_duplicate_pool_is_not_padded_with_repeat_windows():
    attacks = [attack(str(i), 1 + i * 0.01) for i in range(5)]
    assert len(starter_ids(attacks, [])) == 1
    assert starter_ids([], []) == []
