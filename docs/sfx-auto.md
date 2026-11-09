# Automatic three-query SFX check

This page describes the earlier AudioSep mode (`--audiosep`). The additional `--ced` mode
checks raw audio without separation; `sfx-match` checks known local waveform patterns.
See [the raw-audio and known-reference follow-up](sfx-presence.md) for current results.
The latest [varied-mixture family profile](sfx-mixtures.md) uses `--ced` with a new frozen
policy and recognizes broad whoosh/UI-signal window presence while keeping function unknown.
The old AudioSep artifacts and naming rejection below remain unchanged.

`sfx-auto` checks every existing DSP attack with the same three local AudioSep queries,
then compares raw and estimated focus windows using the prepared CLAP model. The caller
does not supply a target description or a sound label. The original analysis stays intact.

This is experimental. Separation conditioned on a description does not establish that the
sound occurred. A frozen, separately evaluated presence policy controls provisional names.
Unsupported classes, weak estimates, background-dominated estimates, low evidence and
ambiguous candidates remain `unknown`. Scores are not probabilities; confidence stays null.
An abstention does not mean silence or absence of an effect.

## Run

Use the isolated AudioSep environment with both prepared local models. No network or
additional model download is used for inference. See [model preparation](separation.md).
The policy must be produced by a completed calibration run with the same model artifacts.

```powershell
.\.cache\audiosep-venv\Scripts\python.exe -m dotears sfx-auto 'C:\ReferenceLibrary\ID\report.json' --audiosep models\audiosep --clap models\clap-htsat-unfused --policy validation-output\sfx-auto-2026-10-29\active-policy.json --output validation-output\new-auto-sfx --cache .cache\sfx-auto-cache
```

Use a new output directory and a cache outside it. Default: CPU, four threads, 1,800 s budget;
`--threads` accepts 1–16, `--timeout` accepts 30–7,200 s. Up to 200 existing attacks are checked.
Progress goes to stderr; stdout contains the JSON completion summary. If interrupted, the
original report remains intact and completed model-cache entries can be reused. There is
no publication, media upload, remote API, telemetry or automatic model download.

Outputs:

- `sfx-auto.json`: each original event ID/time, provisional type or unknown, rejection
  reasons, raw features, all three query results, policy/model/source hashes and timings.
- `audio/`: each original focus excerpt plus three unnormalized float-WAV estimates.

Contexts contain up to three seconds around each peak. The scoring/listening focus is
150 ms before to 550 ms after the peak, clipped at file boundaries. All estimates retain
their actual levels. A quiet residue is not normalized upwards. An estimate may contain
music, speech or several effects. A pleasant UI tone is not automatically a UI click.

Only existing DSP attacks are scanned. Neighboring attacks can refer to the same sound;
counts are attack-window counts, not counts of independently identified effects. New
sidecar results do not rewrite baseline `audio_attack` labels or broaden normal MCP tools.
An agent can call this optional CLI after ordinary analysis and read its separate report.

## Frozen development protocol

The protocol under gitignored `validation-output/sfx-auto-2026-10-29/calibration/` uses
the 30 eligible human-reviewed clean sources: 12 whoosh, five notification, five click,
eight other, spanning ten source families. The four blade-like sounds stay negative controls.

Set A has 133 windows: 30 full effects, 90 synthetic mixtures at component SNR 0 dB, 12
background windows and silence. All three queries run on every window, including incorrect
queries for known positives. CLAP represents the raw/estimated focus and their difference;
energy and embedding similarity complete a 40-feature vector. Three small binary logistic
heads use weighted standardization, C=1 and equal source/family weighting. Foundation-model
weights remain fixed. Threshold selection uses A predictions whose entire source family,
source hashes and identical audio are excluded from that fold's training and standardization.

A threshold needs at least 95% precision, three true accepted windows and two positive source
families in that development check. The fixed threshold is then tested on 102 B mixture/
background windows with different music and speech. B shares A's foreground sources, so it
tests background transfer rather than independent foreground generalization. Acceptance
needs at least 90% precision, three true windows from two families, no background false
names and no blade false names. Failed classes are disabled monotonically. Validation
repeats after disabling, because removing a competitor can unmask a previously ambiguous
false answer. Thresholds are never refit to B or the user's video.

The three globally fixed queries use the preceding separation study's exploratory results:
whoosh wording 3, notification wording 1 and click wording 1 from the original motion-sfx-v1
descriptions. This selection is not a new independent prompt benchmark. There is no
per-window choice from ground truth. The policy freezes before the Apple Stocks scan.

These are small convenience datasets with repeated source packs and synthetic mixtures.
They do not establish general precision, recall or calibrated confidence on motion design.
Calibration focuses use the known insertion time, not a detector-selected acoustic peak;
the short focus can omit later parts of a full source effect. Human mixture/focus audibility
was not separately rated. The user's Apple review contains only
12 attack ratings and five approximate marks; full-video precision/recall cannot be derived.

Research scripts: `calibrate_sfx_auto.py` (prepare/features/fit) and `compare_sfx_auto.py`.
The calibration script imports only hash-verified matching estimates from the completed
AudioSep study, blocks inference sockets and saves resumable feature evidence. Identical
PCM/prompt/model combinations reuse estimates; different candidate peaks still receive
their own focus features. Keep each cache to one writer at a time.

## Executed result, 2026-10-29

Calibration completed all 266 windows × three fixed queries: 798 estimates, including
554 fresh calls and 244 verified cache hits. Notification and UI-click thresholds did not
meet the family-excluded calibration constraints. Whoosh A threshold selection accepted
20 true windows and one false blade-like mixture; B accepted 11 of the 36 whoosh mixtures
with no false names. B uses already-seen foreground sources, and the low coverage and A
blade error prevent any claim of general motion-design accuracy.

The unchanged policy then ran on all 60 Apple Stocks attacks, with every query attempted.
It took 323.515 s on this CPU/four threads, exit code 0, sampled peak RSS 2,465,665,024 bytes.
There were 59 unknowns and one provisional whoosh at attack_001. The user described that
window as notification/bell-like; their explicit whoosh at attack_017 remained unknown.
All three user-reviewed negative/speech windows stayed unknown. Twelve ratings and five
approximate markers were joined by exact source hash, analysis ID, event IDs and peak times.
No whole-video precision/recall is reported from that incomplete review.

The real-video comparison does not support promotion of this whoosh profile. The original
experiment and policy remain immutable. `active-policy.json` disables naming for all three
classes; weights and thresholds were not refit to the video. `sfx-auto-final.json` applies
that naming rejection to the same verified features: 60 unknowns, retaining every query
estimate and the original provisional answer for audit. This is deterministic rescoring,
not another neural pass or a new independent accuracy test. Future calls using the active
policy still automatically perform all three queries and abstain from unsupported names.
An actual CLI call with that active policy also completed: 60 unknowns, exit 0, 26.395 s,
all 180 estimates and audio embeddings reused from verified cache. Every query's features
matched the experimental pass exactly. This checks the active configuration, not accuracy.

The local pipeline is implemented; reliable automatic names are **not achieved**. Further
work needs independent, varied editing-SFX mixtures and source-family-separated evaluation.
Repeating narrow physical swish/weapon packs or fitting to the Apple notes would not supply
that evidence. No new annotation worksheet was created for this check.
