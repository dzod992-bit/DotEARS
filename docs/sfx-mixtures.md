# Varied-mixture acoustic families — 2026-10-30

An experimental local profile now recognizes two acoustic families in mixed tracks:
`whoosh` and broad `ui_sound` (short clicks, beeps and chimes, including notification-like
signals). It preserves `function_role=unknown` and `notification_subtype=unknown`.
Notification versus confirmation/interface intent has not been established.

The actual Apple Stocks CLI pass checks all **114** DSP attacks rather than the original
60 returned clips. It proposes **5 whoosh windows, 5 UI-signal windows and 104 unknowns**.
These are windows, not ten distinct or human-confirmed effects. The 7.343311s whoosh agrees
with the user's explicit whoosh note. The 1.536508s signal is compatible with the notification
note at the parent-family level. The 19.495692s whoosh differs from the user's broad UI note
and remains disputed. The original first-bell attack itself remains unknown, although nearby
signal windows overlap that region. Negative reviewed attacks at 6.685s, 8.700s and 9.907s
remain unknown. Partial notes cannot establish whole-video precision or recall.

## Prepared sources and mixtures

The local catalog contains **139 foreground sources from 88 authors**, plus **15 background
recordings**. The frozen final manifest has 1,084 windows: 139 clean, 834 mixed and 111
background controls; 821 train, 128 calibration and 135 development-test windows.
Foreground authors, identical source PCM and exact background recordings are disjoint
between splits. All prior foregrounds belong to training; new calibration/test authors were
selected by metadata before fitting. Four old Kenney controls are ambiguous under the newly
broadened UI definition: their 28 training views were quarantined without relabeling reviews,
leaving 793 effective training windows.

Sources combine the prior human-reviewed material with individually licensed public previews
and [FSD50K](https://zenodo.org/records/4060432) source labels. FSD50K human labels describe
acoustic source events; some notification/UI roles additionally rely on author descriptions.
Generated mixture labels record known inserted components; **the mixtures were not separately
human-rated for audibility**. Repeated views of one source are not independent observations.
License records, authors, pages, URLs, archived metadata and content hashes are retained.

The mixtures cover music, speech and simultaneous music/speech at 0 and ±6dB component SNR,
with peak-centered 3s excerpts and ±60ms candidate jitter. Long-source cropping and retained
energy are recorded. Component reconstruction passed with maximum sample error 2.98e-8 and
SNR error 4.45e-15dB. Peak amplitude stays below 0.9. Two preliminary manifests were superseded
before the final fitting; use `experiment-final`, not the invalid first corpus.

## Selected models and evaluation limits

One pinned CED/CLAP feature pass provides the raw waveform and a fixed spectral-contrast view.
The latter suppresses some stationary energy; it is **not an isolated SFX** and does not use
a target word. The full representation has 1,075 features. The whoosh branch uses an RBF
head on the full representation; the signal branch uses an RBF head on 51 semantic/acoustic
features. Both were selected using calibration, without private video or annotations.

The three-way notification/UI competition failed. A new binary parent-family detector was
therefore fitted with original notification and UI members as broad signal positives, while
preserving original labels. The final profile composes the two frozen detectors: one accepted
family returns its name; competing families return unknown. Scores are not probabilities.

On all 135 source-held development-test windows, deterministic composed-profile regression gives:

| Family | Correct named windows | Wrong named windows | Recovery in all 0dB mixture views |
|---|---:|---:|---:|
| Whoosh | 14 | 1 zipper | 5/15 |
| Broad click/beep/chime | 24 | 0 | 14/21 |

The broad signal selection check used a restricted common comparison set: 8/8 accepted
windows were correct; 6/7 simultaneous music/speech 0dB windows were recovered. The wider
135-window rescore above adds other backgrounds and SNRs. It is not another independent test.
The development sources were reused across several research variants, including a failed
component-separation experiment. These small results do not establish population accuracy.

A complete generated 60s track using the same held-out sources recovers **7/12 inserted
target occurrences**, with **three premature UI assignments** under the pre-existing strict
peak-within-active-interval ±100ms rule. All three focus windows overlap the subsequent
inserted signal, but the DSP peak precedes its start. Thus classification is explicitly
`presence_in_mixture_focus_window`, and `effect_onset_seconds` remains null. Exact localization
is unresolved; overlapping windows can repeat a sound. Of 274 detected attacks, the fixed
200-attack cap permits only 200 to be classified. This is an integration regression, not a
fresh source test, and no thresholds were adjusted after it.

The optional query-free [FUSS baseline](https://zenodo.org/records/4012661) was also tested.
It improved source/interference ratios but lost most effect amplitude in mixed tracks;
all three full-effect retention gates failed. Its separate component-recognition candidates
also failed the calibration gate. FUSS is not required or used by the selected profile.

## Run the prepared family profile

Use the existing isolated CPU environment and pinned local models. Use a new output folder:

```powershell
.\.cache\sfx-venv\Scripts\python.exe -m dotears sfx-auto 'C:\ReferenceLibrary\ID\report.json' --ced models\ced-mini --clap models\clap-htsat-unfused --policy validation-output\sfx-mixtures-2026-10-30\hierarchy-policy\policy.json --output validation-output\new-family-check --cache .cache\sfx-contrast-video-cache
```

Read `sfx-auto.json`. Each event has the original candidate peak, focus interval, mixed WAV,
family decision, both underlying decisions and rejection reasons. Notification role and effect
onset remain unknown. Original reports/audio are preserved. This explicit command is separate
from normal `analyze` and MCP; a normal baseline report does not acquire these names automatically.

The profile is `dotears.sfx-presence-policy/5`, taxonomy `dotears.sfx-types/3-hierarchical`.
It pins both child-policy hashes and a common feature/model identity. Composition does not
refit weights or thresholds. `scripts/prepare_sfx_hierarchy.py` reproduces composition from
the whoosh and signal policies in a new folder.

## Local evidence and next work

Artifacts are in gitignored `validation-output/sfx-mixtures-2026-10-30/`:
`catalog.json`, `experiment-final/`, `mixture-verification-final.json`, `calibration-final/`,
`signal-family/`, `hierarchy-policy/`, `heldout-sequence/hierarchy-regression.json`,
`video-hierarchy-final/`, `reproduction/` and `RESULTS.ru.md`.
The originals and all 114 feature vectors/mixed PCM excerpts were verified unchanged.
Three fresh offline feature forwards reproduced cached features exactly (3 CED, 6 CLAP,
zero cache hits). The cached full-video run is not a cold-inference speed benchmark.

Regression: 128 tests passed, two separate model tests deselected; actual CED/CLAP forwards
were checked separately. Ruff passed. No private media left the machine.
Next work is tighter temporal localization and a fresh untouched source/track holdout,
especially musical transients, zipper/blade negatives and quiet overlapping tones. Specific
notification/confirmation roles require evidence beyond membership in the acoustic family.
No additional user listening worksheet is needed for this completed experiment.
