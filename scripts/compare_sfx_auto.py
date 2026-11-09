"""Compare a frozen automatic scan with partial human review, without refitting it."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dotears.audiosep import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scan = json.loads(args.scan.read_text("utf-8"))
    annotation = json.loads(args.annotations.read_text("utf-8"))
    matches = [
        r
        for r in annotation["references"]
        if r["analysis_id"] == scan["analysis_id"] and r["source_sha256"] == scan["source_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError("Annotations must uniquely match the source hash AND analysis ID.")
    reviewed = matches[0]
    lookup = {row["event_id"]: row for row in scan["events"]}
    expected = {row["id"]: row["peak_time"] for row in reviewed["reported_attacks"]}
    if set(expected) != set(lookup) or any(
        abs(lookup[k]["peak_time"] - v) > 1e-6 for k, v in expected.items()
    ):
        raise ValueError(
            "The scan does not match all manually reviewed report attack IDs and times."
        )
    policy_path = Path(scan["policy_path"])
    if file_digest(policy_path) != scan["policy_sha256"]:
        raise ValueError("Policy changed after inference.")
    policy = json.loads(policy_path.read_text("utf-8"))
    if datetime.fromisoformat(policy["created_at"]) >= datetime.fromisoformat(scan["created_at"]):
        raise ValueError("The policy must be frozen before the video evaluation.")
    if policy["video_used_for_fitting"] or scan["human_annotations_used_in_inference"]:
        raise ValueError("Comparison is invalid if video annotations entered fitting/inference.")
    # Only these three notes explicitly identify one of our narrow target types.
    # Pleasant UI, unspecified transition and two-sound notes are not click/whoosh truth.
    semantic_notes = {
        "attack_001": "notification",
        "attack_006": "notification",
        "attack_017": "whoosh",
    }
    comparisons = []
    for identity, rating in reviewed["events"].items():
        row = lookup[identity]
        named = row["label"] != "unknown"
        if rating["kind"] in {"nothing", "speech"}:
            assessment = (
                "potential_conflict_with_user_review" if named else "abstained_on_negative_review"
            )
        elif identity in semantic_notes:
            assessment = (
                "compatible_with_explicit_note"
                if row["label"] == semantic_notes[identity]
                else (
                    "effect_type_unresolved" if not named else "different_type_from_explicit_note"
                )
            )
        else:
            assessment = "type_not_precisely_annotated" if named else "effect_type_unresolved"
        comparisons.append(
            {
                "event_id": identity,
                "peak_time": row["peak_time"],
                "human": rating,
                "automatic_label": row["label"],
                "assessment": assessment,
                "explicit_note_type": semantic_notes.get(identity),
                "reason": row["reason"],
            }
        )
    markers = []
    for marker in reviewed["markers"]:
        nearest = min(scan["events"], key=lambda row: abs(row["peak_time"] - marker["time"]))
        covers = [
            row["event_id"]
            for row in scan["events"]
            if row["focus_start"] <= marker["time"] <= row["focus_end"]
        ]
        markers.append(
            {
                **marker,
                "nearest_event": nearest["event_id"],
                "nearest_peak_delta_seconds": nearest["peak_time"] - marker["time"],
                "nearest_automatic_label": nearest["label"],
                "covering_focus_windows": covers,
                "scope": "approximate human marker; window proximity is not semantic detection",
            }
        )
    counts = Counter(row["label"] for row in scan["events"])
    calibration = json.loads(args.calibration.read_text("utf-8"))
    result = {
        "schema_version": "dotears.sfx-auto-partial-comparison/1",
        "created_at": datetime.now(UTC).isoformat(),
        "scan_sha256": file_digest(args.scan),
        "annotations_sha256": file_digest(args.annotations),
        "policy_sha256": scan["policy_sha256"],
        "counts": dict(counts),
        "reviewed_event_count": len(comparisons),
        "marker_count": len(markers),
        "full_review_completed": reviewed["full_review_completed"],
        "comparisons": comparisons,
        "markers": markers,
        "whole_video_precision_recall_available": False,
        "reason_no_whole_video_metrics": (
            "Partial review, ambiguous type notes, multiple sounds per window."
        ),
        "protocol": "No video/annotation tuning; threshold policy frozen before scan.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "comparison.json"
    if path.exists():
        raise ValueError("Comparison exists; do not overwrite frozen evidence.")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    enabled = calibration["final_enabled_classes"]
    whoosh_a = calibration["calibration"]["whoosh"]["statistics"]
    whoosh_b = calibration["validation_rounds"][-1]["whoosh"]
    lines = [
        "# DotEARS: автоматическая проверка SFX на Apple Stocks",
        "",
        f"Проверено {len(scan['events'])} атак. "
        "Каждое окно проверено с тремя одинаковыми для всех окон запросами. "
        f"Итоговые метки: {dict(counts)}. Время прохода: {scan['seconds']:.1f} с.",
        "",
        "Исходный отчёт и звук сохранены. Метки нового прохода находятся в отдельном файле. "
        "Принятая моделью метка остаётся предварительной; "
        "числовые оценки не являются вероятностями.",
        "",
        "## Как зафиксировано решение",
        "",
        "AudioSep выделяет три предполагаемых компонента; CLAP оценивает исходное и выделенное "
        "окно. Отдельная небольшая линейная модель проверяет присутствие типа. Тихий остаток, "
        "преобладание сходства с фоном, близкие ответы и непрошедший проверку тип дают unknown.",
        "",
        "Настройка: 133 окна A, в том числе 30 исходных размеченных звуков, 90 смесей и 13 фоновых/"
        "пустых окон. При выборе порогов из обучения исключалось всё семейство проверяемого звука. "
        "Дополнительная проверка: 102 смеси/фоновых окна B с другой музыкой и речью. "
        "Исходные эффекты в A и B общие. Это ограниченная разработческая проверка, "
        "а не независимая оценка качества на новых эффектах.",
        "Фокус синтетических окон привязан к известному началу вставки; в ролике он привязан "
        "к обнаруженной атаке. Короткое окно может не содержать позднюю часть эффекта. "
        "Слышимость именно этих смесей и фрагментов отдельно человеком не оценивалась.",
        "",
        "Типы, прошедшие оба локальных ограничения: "
        f"{', '.join(enabled) if enabled else 'ни один'}. "
        "Остальные типы отключены для окончательного именования; их гипотезы и аудио сохраняются.",
        (
            f"У свушей при подборе порога A: {whoosh_a['true_positives']} верных названий "
            f"и {whoosh_a['false_positives']} ложное, "
            f"в том числе {whoosh_a['blade_false_accepts']} на звуке лезвия. "
            f"В B: {whoosh_b['true_positives']} верных из 36 смесей со свушем, "
            f"{whoosh_b['false_positives']} ложных. "
            "Пропуски велики; проверка B с уже знакомыми исходными эффектами "
            "не доказывает перенос на новые звуки."
        )
        if whoosh_a
        else "Порог для свушей подобрать не удалось.",
        "",
        f"SHA256 правил до прохода видео: `{scan['policy_sha256']}`. "
        "Разметка ролика не участвовала в настройке или выборе запроса.",
        "",
        "## Сравнение с сохранёнными заметками",
        "",
        "Есть 12 оценок атак и 5 приблизительных отметок. Полный просмотр не отмечен завершённым. "
        "Точные названия звуков в поле sound_name не заполнены. Приятный UI-звук не обязательно "
        "является щелчком; слово «переход» не задаёт тип whoosh. Поэтому общую точность и полноту "
        "распознавания по этой разметке посчитать нельзя.",
        "",
        "| Атака | Время, с | Твоя оценка / заметка | Ответ | Сопоставление |",
        "|---|---:|---|---|---|",
    ]
    explanations = {
        "potential_conflict_with_user_review": "Возможное ложное срабатывание",
        "abstained_on_negative_review": "Воздержалась от названия",
        "compatible_with_explicit_note": "Совпадает с описанием типа",
        "effect_type_unresolved": "Тип эффекта не установлен",
        "different_type_from_explicit_note": "Расходится с описанием типа",
        "type_not_precisely_annotated": "Узкий тип в заметке не задан",
    }
    for row in comparisons:
        human = row["human"]
        note = (human["kind"] + ": " + human["note"]).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {row['event_id']} | {row['peak_time']:.3f} | {note} | "
            f"{row['automatic_label']} | {explanations[row['assessment']]} |"
        )
    lines += [
        "",
        "## Приблизительные отметки",
        "",
        "Близость атаки к отметке показывает только покрытие по времени, а не распознанный звук.",
        "",
        "| Отметка, с | Заметка | Ближайшая атака | Разница, мс | Ответ |",
        "|---|---|---|---:|---|",
    ]
    for row in markers:
        note = row["note"].replace("|", "/")
        lines.append(
            f"| {row['time']:.3f} | {note} | {row['nearest_event']} | "
            f"{row['nearest_peak_delta_seconds'] * 1000:+.1f} | {row['nearest_automatic_label']} |"
        )
    lines += [
        "",
        "## Все окна и результаты выделения",
        "",
        "Все WAV содержат оценки компонентов смешанной дорожки. Уровень не нормализован: "
        "тихий/неудачный результат остаётся тихим. Компонент может содержать музыку или речь.",
        "",
        "| Атака | Пик, с | Ответ | Исходный фрагмент | Whoosh | Notification | UI-click |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in scan["events"]:

        def link(relative, title):
            return f"[{title}](<{(args.scan.parent / relative).resolve().as_posix()}>)"

        audio = [link(row["mixture_focus_path"], "звук")]
        audio += [
            link(row["feature_evidence"][label]["focus_audio_path"], label)
            for label in ("whoosh", "notification", "ui_click")
        ]
        lines.append(
            f"| {row['event_id']} | {row['peak_time']:.3f} | {row['label']} | "
            + " | ".join(audio)
            + " |"
        )
    lines += [
        "",
        "## Что это даёт дальше",
        "",
        "Готов воспроизводимый автоматический проход с отказом от сомнительного ответа. "
        "Для улучшения названий нужны независимые короткие примеры эффектов в монтажных "
        "миксах, особенно электронные свуши и разные UI-события, с отрицательными примерами "
        "музыки и речи. Увеличение числа похожих физических взмахов не заменяет такую проверку. "
        "Модель и пороги после просмотра этого ролика не подгонялись.",
        "",
    ]
    (args.output / "RESULTS.ru.md").write_text("\n".join(lines), "utf-8")
    print(
        json.dumps(
            {
                "counts": dict(counts),
                "comparisons": len(comparisons),
                "markers": len(markers),
                "report": str(args.output / "RESULTS.ru.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
