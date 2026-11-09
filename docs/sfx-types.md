# SFX label scope, 2026-10-30

The user explicitly included short beeps and UI confirmation tones in the UI category.
Reference-matching policies retain taxonomy `dotears.sfx-types/2`:

| Label | Scope |
|---|---|
| `whoosh` | Sweeping/rushing transition sound. |
| `notification` | Message/alert notification supported by reviewed evidence. |
| `ui_sound` | Interface feedback: clicks, taps, beeps, blips and confirmation tones. |
| `unknown` | Insufficient evidence or ambiguous type/role. |

The newly fitted [mixture family profile](sfx-mixtures.md) uses
`dotears.sfx-types/3-hierarchical`: its acoustic outputs are `whoosh`, `ui_sound` or `unknown`.
Here `ui_sound` means the broad **short click/beep/chime family**, including notification-like
signals. `function_role` and `notification_subtype` remain `unknown`; a sound's role as an
alert, confirmation or interface action is not established from its acoustic family.
Original notification/UI source labels and all human reviews remain unchanged. Both labels
contribute to a newly trained parent-family detector, without relabeling the original data.
This differs from the earlier reference-bank scope migration, which did not retrain a model.
The label applies to the exported focus window; its DSP peak is not a verified SFX onset.

Notification and interface roles can share an acoustic form. A beep alone does not prove
either role. Competing reference types within the frozen margin remain unknown. A compound
UI/transition note does not imply one unique class. Previous notification and other reviews
are preserved; they are not automatically relabeled as UI sounds.

Reference policy version 2 requires an explicit taxonomy ID and uses `ui_sound` as the parent
label. Historical `ui_click` and `ui_tone` source labels map to this parent, preserving their
original review label. Version 1 remains readable with its narrow-click semantics. Old CLAP
prompts, neural policies, feature caches and result files stay frozen; the failed narrow neural
heads are not reinterpreted as trained broad UI classifiers.

`scripts/update_sfx_ui_scope.py POLICY --output NEW_FOLDER` creates a versioned reference
policy with the same bank, thresholds, disabled entries and competition rules. It records the
parent hash and explicitly states that this is a label-scope migration without retraining.
Future reviewed broad-UI banks can be prepared with `calibrate_sfx_match.py prepare --ui-scope
broad`; fitting maps both narrow UI source types to the parent before evaluating them.

Use the prepared migrated policy:

```powershell
.\.venv\Scripts\python.exe -m dotears sfx-match 'C:\ReferenceLibrary\ID\report.json' --policy validation-output\sfx-ui-scope-2026-10-30\policy\active-policy.json --output validation-output\new-broad-ui-check
```

The older bank still has six active whoosh, five notification and three UI examples. Broader label
membership adds no new validated reference samples and does not imply reliable recognition
of arbitrary tones. Its Apple Stocks run returns 60 unknowns. The earlier UI
note at attack_051 is now relevant to the broad UI comparison, but remains unresolved by the
model; video annotations did not enter inference or fitting.

Tests cover a known tonal confirmation mixed with noise, unchanged scores/disabled flags
across migration, explicit schema binding, and an identical beep with two competing roles.
The previously seen full-track scenario and Apple reference are also rerun as compatibility
checks. They are not new independent evidence of tone generalization.
