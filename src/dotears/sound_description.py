"""Describe observable acoustics and prepare one sound-effect generation brief."""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from dotears.motion_cues import pitch_glide
from dotears.sound_expansion import FAMILIES
from dotears.ui_acoustics import classify_form, short_ui_transient

NAMES = {key: value[0] for key, value in FAMILIES.items()}
NAMES.update(whoosh="Свуш", ui_sound="Короткий UI-сигнал", unknown="Неопределённый акцент")
NAMES.update(
    click_like="Клик",
    tap_like="Мягкий тап",
    click_candidate="Кандидат на клик",
    tap_candidate="Кандидат на тап",
    beep="Бип",
    tonal="Тональный сигнал",
    chime="Звон",
    ring_candidate="Эскиз короткого звона",
)
PROMPT_SOUNDS = {key: value[1] for key, value in FAMILIES.items()}
PROMPT_SOUNDS.update(whoosh="a short rushing sweep", ui_sound="a short electronic interface sound")
UI_FORMS = {
    "click_like": ("По форме похож на сухой щелчок.", "a single short dry UI click"),
    "tonal": ("По форме похож на короткий тон или бип.", "a short clear electronic UI beep"),
    "tap_like": ("По форме похож на мягкий тап.", "a soft UI tap with a muted midrange attack"),
    "beep": ("По форме похож на короткий бип.", "a short UI beep with a steady electronic pitch"),
    "chime": (
        "По форме похож на короткий звон.",
        "a short electronic UI chime with a clear onset and a fading tonal ring",
    ),
}
MIXED_TRANSIENT_DRAFTS = {
    "click_like": (
        "One crisp, dry UI click. A sharp bright tick with an immediate cutoff, "
        "no ringing or reverb. Single isolated hit, no music or speech."
    ),
    "tap_like": (
        "One soft, muted UI tap. A rounded midrange attack with a quick decay and "
        "very little treble. Single isolated hit, no reverb, music or speech."
    ),
}
BAND_RU = {
    "low": "больше всего энергии в низких частотах",
    "mid": "больше всего энергии в средних частотах",
    "high": "больше всего энергии в верхних частотах",
}
BAND_EN = {"low": "low-end weight", "mid": "a clear midrange", "high": "bright upper frequencies"}


def read_clip(path: Path, *, max_seconds: int = 21) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as audio:
        rate, channels = audio.getframerate(), audio.getnchannels()
        if audio.getsampwidth() != 2 or audio.getnframes() > max_seconds * rate:
            raise ValueError("Expected a short 16-bit PCM clip.")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return samples.reshape(-1, channels).astype(float) / 32768.0, rate


def measure_clip(values: np.ndarray, rate: int, relative_peak: float) -> dict | None:
    """Measure a local excess over surrounding frames in the actual mixed PCM."""
    size, hop = round(rate * 0.02), round(rate * 0.01)
    if len(values) < 5 * hop or not 0 <= relative_peak <= len(values) / rate:
        return None
    starts = np.arange(0, len(values) - size + 1, hop)
    frames = values[starts[:, None] + np.arange(size)]
    times = (starts + size / 2) / rate - relative_peak
    nearby = np.abs(times) <= 0.09
    reference = (times < -0.08) | (times > 0.14)
    if not nearby.any() or reference.sum() < 4:
        return None
    power = np.mean(
        np.abs(np.fft.rfft(frames * np.hanning(size)[None, :, None], axis=1)) ** 2, axis=2
    )
    floor = np.quantile(power[reference], 0.3, axis=0)
    excess = np.maximum(power - floor, 0)
    totals = excess.sum(axis=1)
    peak = int(np.argmax(np.where(nearby, totals, -1)))
    envelope = np.mean(frames**2, axis=(1, 2))
    contrast = float(
        10 * np.log10((envelope[peak] + 1e-10) / (np.median(envelope[reference]) + 1e-10))
    )
    result = {
        "scope": "mixed_clip_local_contrast",
        "peak_to_surround_db": round(contrast, 1),
        "strong_local_accent": bool(contrast >= 8 and envelope[peak] >= 1e-6),
    }
    if not result["strong_local_accent"]:
        return result
    # Stay in the contiguous bump around the strongest frame near the candidate.
    first, last = peak, peak
    threshold = 0.25 * totals[peak]
    while first > 0 and totals[first - 1] >= threshold:
        first -= 1
    while last + 1 < len(totals) and totals[last + 1] >= threshold:
        last += 1
    spectrum = excess[first : last + 1].sum(axis=0)
    frequencies = np.fft.rfftfreq(size, 1.0 / rate)
    bands = {
        "low": spectrum[frequencies < 300].sum(),
        "mid": spectrum[(frequencies >= 300) & (frequencies < 2500)].sum(),
        "high": spectrum[frequencies >= 2500].sum(),
    }
    total = float(spectrum.sum())
    fractions = {key: round(float(value) / max(total, 1e-12), 3) for key, value in bands.items()}
    largest = max(fractions, key=fractions.__getitem__)
    tone_bin = int(np.argmax(spectrum))
    tone_share = spectrum[max(0, tone_bin - 1) : tone_bin + 2].sum() / max(total, 1e-12)
    frequency = float(frequencies[tone_bin])
    result.update(
        {
            "dominant_band": largest if fractions[largest] >= 0.64 else None,
            "excess_band_fractions": fractions,
            "contrast_bump_ms": round(1000 * (size + (last - first) * hop) / rate),
            "bump_fully_in_window": first > 0 and last + 1 < len(totals),
            "narrow_energy_band_hz": round(frequency)
            if tone_share >= 0.68 and 120 <= frequency <= 8000
            else None,
            "contrast_rise_ms": round(1000 * (peak - first) * hop / rate),
            "contrast_fall_ms": round(1000 * (last - peak) * hop / rate),
        }
    )
    return result


def ui_form(measured: dict | None, overlap: list[str] | None) -> str:
    if overlap != [] or not measured or not measured["strong_local_accent"]:
        return "unknown"
    if not measured["bump_fully_in_window"]:
        return "unknown"
    resonance = measured["narrow_energy_band_hz"]
    width = measured["contrast_bump_ms"]
    if resonance and 600 <= resonance <= 6500 and 50 <= width <= 380:
        return "tonal"
    if resonance is None and width <= 40 and measured["dominant_band"] in {"mid", "high"}:
        return "click_like"
    return "unknown"


def generation_prompt(text: str, basis: str) -> dict:
    return {
        "target": "elevenlabs_sound_effects",
        "text": text,
        "language": "en",
        "duration_seconds": None,
        "loop": False,
        "basis": basis,
    }


def describe_transient(proposal: dict) -> tuple[str, dict]:
    """Describe a brief acoustic shape without assigning a source or an effect name."""
    shape = proposal["impulse"]
    width = shape["width_ms"]
    bright = proposal["form"] == "click_like"
    if bright:
        description = "Резкий яркий импульс с преобладанием верхних частот."
        text = "One very short, bright percussive tick with a sharp treble attack"
    else:
        description = "Короткий приглушённый импульс, основная энергия в средних частотах."
        text = "One soft, compact percussive pulse with a muted midrange body and little treble"
    description += f" Заметная часть занимает около {width:g} мс."
    text += f", a compact transient of roughly {width:g} milliseconds"
    if shape["fall_ms"] >= 2 * max(shape["rise_ms"], 1):
        description += " Энергия убывает медленнее, чем нарастает."
        text += ", with a quick attack and a slightly longer decay"
    text += ". Dry isolated one-shot, no music or speech."
    return description, generation_prompt(text, "observed_motion_transient")


def describe(
    path: Path,
    relative_peak: float,
    family: str,
    overlap: list[str] | None,
    *,
    ring_context: bool = False,
    rhythmic_cue: Callable[[float, str], bool] | None = None,
) -> dict:
    values, rate = read_clip(path)
    measured = measure_clip(values, rate, relative_peak)
    pieces = [
        "Неопределённый акцент."
        if family == "unknown"
        else f"Предварительная метка «{NAMES[family]}»."
    ]
    attributes = []
    if measured and measured["strong_local_accent"]:
        band = measured["dominant_band"]
        if measured["bump_fully_in_window"] and measured["contrast_bump_ms"] <= 90:
            pieces.append("Возле пика выделяется короткий всплеск.")
            attributes.append("a short transient")
        else:
            pieces.append("Возле пика заметно прибавляется энергия.")
        if band:
            pieces.append(f"У этого всплеска {BAND_RU[band]}.")
            attributes.append(BAND_EN[band])
        if resonance := measured["narrow_energy_band_hz"]:
            pieces.append(f"В спектре выделяется узкая полоса около {resonance} Гц.")
            attributes.append(f"a narrow resonance near {resonance} Hz")
        if measured["bump_fully_in_window"] and measured["contrast_fall_ms"] >= max(
            40, 2 * measured["contrast_rise_ms"]
        ):
            pieces.append("После пика энергия спадает медленнее, чем нарастает.")
            attributes.append("a quick rise and a fading tail")
    if overlap is None:
        pieces.append("Фон в этом окне не проверен.")
        attributes = []
    elif overlap:
        pieces.append("В том же окне есть признаки речи или музыки.")
        # The detected bump could come from one of the other sounds in the mix.
        attributes = []
    form = "unknown"
    shape_evidence = None
    ringing = None
    mixed_transient = None
    glide = None
    short_form, probes = (
        short_ui_transient(values, rate, relative_peak)
        if overlap and family in {"ui_sound", "unknown"}
        else ("unknown", {})
    )
    if family == "ui_sound" or (family == "unknown" and (ring_context or short_form != "unknown")):
        acoustic, shape_evidence = classify_form(values, rate, relative_peak)
        if family == "ui_sound" and overlap == []:
            form = acoustic if acoustic != "unknown" else ui_form(measured, overlap)
        if probes:
            shape_evidence["short_transient"] = {"candidate": short_form, "probes": probes}
        if short_form != "unknown" and acoustic not in {"beep", "chime"}:
            cue = probes[short_form]
            time = cue["relative_peak_seconds"]
            agrees = (
                acoustic == short_form
                and abs(shape_evidence["impulse"]["relative_peak_seconds"] - time) <= 0.016
            )
            if family == "ui_sound" or agrees:
                repeats = rhythmic_cue(time, short_form) if rhythmic_cue else None
                shape_evidence["short_transient"]["rhythmic_match"] = repeats
                if repeats is not True:
                    mixed_transient = short_form
                    if family == "ui_sound":
                        form = short_form
                    pieces.append(
                        f"{UI_FORMS[short_form][0]} Локальный импульс около {cue['width_ms']:g} мс."
                    )
        if ring_context and acoustic == "chime" and overlap != []:
            ringing = shape_evidence["lines"][0]
            pieces.append(
                f"В этом окне выделяется короткая затухающая составляющая около {ringing['hz']} Гц."
            )
        shape_evidence.update(acoustic_form_candidate=acoustic, ui_ring_context=ring_context)
    if family in {"unknown", "ui_sound", "whoosh"} and ringing is None and mixed_transient is None:
        glide = pitch_glide(values, rate, relative_peak)
    prompt = None
    if ringing is not None:
        prompt = generation_prompt(
            f"One short electronic UI ping with a resonance near {ringing['hz']} Hz "
            "and a quick fading tail. Dry isolated foreground sound, no music or speech.",
            "observed_ringing_candidate",
        )
    elif mixed_transient is not None:
        prompt = generation_prompt(
            MIXED_TRANSIENT_DRAFTS[mixed_transient],
            "provisional_ui_family_and_short_transient"
            if family == "ui_sound"
            else "observed_short_transient_candidate",
        )
    elif glide is not None:
        direction = "поднимается" if glide["direction"] == "up" else "опускается"
        sweep = "rises" if glide["direction"] == "up" else "falls"
        span = glide["span_ms"]
        start, end = glide["start_hz"], glide["end_hz"]
        pieces.append(
            f"Выделяется короткая тональная линия. Высота плавно {direction} "
            f"примерно с {start} до {end} Гц за {span} мс."
        )
        prompt = generation_prompt(
            f"One short smooth pitched glide. The pitch {sweep} from roughly "
            f"{start} to {end} Hz over about {span} milliseconds. "
            "Dry isolated one-shot, no music or speech.",
            "observed_pitch_glide",
        )
    elif family in PROMPT_SOUNDS:
        sound = PROMPT_SOUNDS[family]
        if family == "whoosh":
            sound = "a compact motion-graphics whoosh with one smooth sweep and a clean finish"
        if form in UI_FORMS:
            description, sound = UI_FORMS[form]
            pieces.append(description)
        text = f"Generate {sound} as an isolated one-shot."
        if attributes:
            text += f" Shape it with {' and '.join(attributes)}."
        text += " Quiet background, no music or speech."
        basis = "provisional_family_and_local_acoustics" if attributes else "provisional_family"
        prompt = generation_prompt(text, basis)
    elif not measured or not measured["strong_local_accent"]:
        pieces.append("Выделить характер акцента по этому окну трудно.")
    return {
        "description_ru": " ".join(pieces),
        "acoustic_evidence": measured,
        "ui_sound_form": form,
        "ui_form_evidence": shape_evidence,
        "motion_shape": {"kind": "pitch_glide", **glide} if glide else None,
        "generation_prompt": prompt,
    }
