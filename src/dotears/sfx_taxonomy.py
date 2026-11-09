"""User-approved UI scope; legacy acoustic labels and scores remain immutable."""

from __future__ import annotations

from dotears.errors import ConfigurationError

UI_TAXONOMY = "dotears.sfx-types/2"
UI_TYPES = ("whoosh", "notification", "ui_sound")
UI_DEFINITIONS = {
    "whoosh": "Sweeping or rushing transition sound.",
    "notification": "Message or alert notification, when supported by reviewed evidence.",
    "ui_sound": "Interface feedback including clicks, taps, beeps, blips and confirmation tones.",
    "ambiguity": "A beep or tone alone does not establish notification versus interface role; "
    "conflicting reference types remain unknown.",
}


def broad_ui_label(label: str) -> str:
    """Map narrow UI labels only; never reinterpret notification or other reviews."""
    if label in ("ui_click", "ui_tone", "ui_sound"):
        return "ui_sound"
    if label in ("whoosh", "notification", "other", "background", "unknown"):
        return label
    raise ConfigurationError(f"Unsupported source label for broad UI scope: {label}")
