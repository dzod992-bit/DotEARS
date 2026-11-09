# Raw SFX presence and known-reference matching

Latest, 2026-10-30: the [varied-mixture family profile](sfx-mixtures.md) supersedes the
all-unknown result below for the new optional `sfx-auto --ced` profile. On all 114 Apple
attacks it proposes 5 whoosh and 5 broad UI-signal windows; precise notification roles remain
unknown. This document's measurements describe the preserved earlier, narrower experiments.

UI scope update, 2026-10-30: the current category includes clicks, taps, beeps and confirmation
tones. New reference policies emit `ui_sound`. See [the definitions and current policy](sfx-types.md).
The measurements below preserve the historical narrow-click protocol; changing scope does
not turn the old neural heads or existing test results into a validated broad UI classifier.

The 2026-10-29/30 follow-up has two explicit local commands. `sfx-auto --ced` tests learned
raw-audio presence heads. `sfx-match` searches a reviewed library for a familiar waveform.
Both scan every attack in an existing report, preserve the report/audio, and write a sidecar.
Neither changes ordinary analysis or automatically loads models through MCP.

General recognition of arbitrary new whooshes, notifications and clicks in music/speech is
**not achieved**. All three learned raw heads failed the fresh-mixture gate and remain
disabled in the active policy. The narrower reference matcher supports 14 reviewed patterns:
six whooshes, five notifications and three UI clicks. Those counts are library entries,
not independently validated classes or counts of effects in a video.

## Use the prepared known-pattern bank

This needs the ordinary project environment, its scientific audio dependencies, and the
explicitly calibrated local bank. It does not need Torch, CLAP, CED, AudioSep or a download.
Use a new output folder:

```powershell
.\.venv\Scripts\python.exe -m dotears sfx-match 'C:\ReferenceLibrary\ID\report.json' --policy validation-output\sfx-ui-scope-2026-10-30\policy\active-policy.json --output validation-output\new-reference-match
```

Read `sfx-match.json`. `known_reference_pattern` is a provisional match to a library pattern;
it does not establish that an arbitrary unseen sound belongs to that class. The sidecar
includes all template scores, the best reference ID, original attack times, a matched-peak
offset, rejection reasons, policy/bank/source hashes, and null confidence. Scores are
normalized correlations, not probabilities. A video occurrence is not human-confirmed.

The matcher uses 32kHz mono audio, a 300Hz highpass, reviewed 0.5–99.5% energy crops, and
absolute normalized waveform correlation within 120ms of an attack. All templates, including
disabled/negative ones, compete. A different-label margin of 0.05 and per-template frozen
thresholds prevent falling back to another label when the strongest pattern is disabled.
Quiet matches are rejected. Filtering creates numerical matching features; it does not
export or claim an isolated effect. Pitch/speed changes, masking and codecs can break a match.
Only up to 200 existing attacks are checked; missed/truncated attacks remain a separate limit.

## Executed reference validation

The bank uses the 30 earlier human-reviewed sources: 12 whoosh, five notification, five
click and eight other. Blade-like sounds remain negative. Foreground waveforms intentionally
overlap between preparation, calibration and testing. This experiment tests recognition of
known sources on different backgrounds, not transfer to novel foregrounds.

Thresholds are max(0.65, maximum wrong-type calibration correlation + 0.03), capped at 1.01.
Enabling requires two correct 0dB mixtures from two background kinds. Calibration uses one
music recording and LibriSpeech voice; testing uses Battle music and VOiCES voice. Public
non-bank sounds and background-only/silent windows are additional negative controls.
Some background controls recur across the combined manifests. Matching decoded PCM, rather
than WAV container hashes, identifies these repetitions. They are not independent observations.

Before the full-track check, 15 patterns passed. The known-source test accepted 21 whoosh,
24 notification and 15 click windows without wrong names; 0dB recovery was 10/36, 14/15 and
8/15 respectively. A generated 64s integration track then exposed a false swish-4 match
away from any inserted effect. That pattern was disabled, with thresholds and bank unchanged.
This rejection was based on generated ground truth, not the user's video or its annotations.

After rejection there are 14 active patterns. Deterministic rescoring of the same integration
track recovers 11/15 inserted sources without extra named windows. This is a regression repair,
not a new independent accuracy result. Multiple attack windows can describe the same source.
Three sources were missed before rejection; four total remain after rejecting the additional
pattern. No broader reliability claim follows from this small test.

## Raw-audio research mode

The isolated CPU environment additionally needs matching Torch/TorchAudio, Transformers and
Safetensors (`.[sfx-presence]`). Model downloads are explicit; inference uses local paths only.
The float [CED-Mini model](https://huggingface.co/mispeech/ced-mini) and its inspected
Apache-2.0 code are pinned to revision `26c3ebcae85d4330f4fc26763f029539a3afcda0`.
The unrelated GPL training repository is not imported. Preparation does not execute downloaded
code; inference executes only hash-verified local inference files with strict weight loading.
The upstream label-download path is replaced by its pinned local label map.

```powershell
.\.cache\sfx-venv\Scripts\python.exe scripts\prepare_ced.py --output models\ced-mini
.\.cache\sfx-venv\Scripts\python.exe -m dotears sfx-auto 'C:\ReferenceLibrary\ID\report.json' --ced models\ced-mini --clap models\clap-htsat-unfused --policy validation-output\sfx-v2-2026-10-29\final-presence\policy.json --output validation-output\new-raw-sfx --cache .cache\sfx-presence-cache
```

The existing CLAP preparation route is described in [SFX hypotheses](sfx.md). Never substitute
model files or an ONNX export under a policy trained on another representation. The quantized
CED export failed parity (max score error 0.23234); it was rejected. The selected float adapter
matched 16 independently loaded upstream reference forwards exactly. Broad CED tags alone
are not named-SFX evidence.

Three time scales, CLAP similarities and acoustic shape features form 85 features. Research
variants add 512 short/context latent dimensions or 60 features describing 8ms/1ms attack
shape. Source-balanced linear and RBF heads were compared across three regularization values:
24 global candidates. Every head competes, including disabled types; disabling cannot expose
another name. Learned scores and text cosine similarities are not probabilities.

The final catalog has 72 sources from 23 author groups and 693 windows: 513 train, 76
calibration, 55 fresh test and 49 known regression windows. Foreground authors, source hashes
and identical source PCM are separated across these splits. All Kenney packs count as one
author group. Public author captions supplement the 30 human labels and are weaker evidence;
mixture audibility was not separately human-rated. Model choice uses calibration only.

The selected semantic linear candidate C=0.1 found some clean fresh examples, but 0dB
recovery was only 1/6 whoosh, 0/6 notification and 0/6 click windows. All failed the frozen
20% recovery gate. No thresholds or weights were changed after that failure. A serialization
bug in the research evaluator's mutable selected-candidate export was fixed; the final choice
was recovered from its immutable pre-test snapshot, with a repair receipt saved.

On Apple Stocks, the rejected research policy returned one unreviewed whoosh at attack_036;
the active raw policy returned 60 unknowns. The initial reference policy proposed whoosh at
attack_054; its template was independently rejected by the generated integration check.
An actual CLI run with the final reference policy also returned 60 unknowns. The three explicit
user type notes (notification at 001/006, whoosh at 017) remain unresolved. Twelve ratings and
five approximate marks were bound to the exact source, analysis, event IDs and peak times.
Partial review cannot supply whole-video precision or recall.

## Reproduction and evidence

Scripts: `prepare_sfx_presence.py`, `evaluate_sfx_presence.py`,
`evaluate_sfx_presence_deep.py`, `evaluate_sfx_presence_rbf.py`,
`evaluate_sfx_presence_micro.py`, and `calibrate_sfx_match.py`.
The last script builds a bank from an explicitly reviewed source catalog and fits thresholds
using frozen calibration/test corpus manifests. It blocks inference sockets and refuses to
overwrite outputs. Its resulting bank and thresholds reproduced the original experiment.
Research fitting requires scikit-learn; normal matching does not.

Local artifacts are gitignored under `validation-output/sfx-v2-2026-10-29/`: public-source
receipts/licenses, frozen protocols, all feature evidence and model candidates, policies,
test decisions, generated integration media, video sidecars, comparison, verification and
`RESULTS.ru.md`. Private media stay local. Regression: 105 tests passed, two standalone
model tests deselected; real CED/CLAP forwards were verified separately. Ruff passed.
