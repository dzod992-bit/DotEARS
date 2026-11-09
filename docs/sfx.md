# Experimental SFX hypotheses

The optional `sfx` command compares existing attack windows with fixed sound descriptions
using the official [LAION CLAP checkpoint](https://huggingface.co/laion/clap-htsat-unfused).
It runs on CPU with local weights. It produces ranked hypotheses and retains `label: unknown`;
there is no calibrated threshold or validated automatic motion-SFX classifier yet.

## Isolated CPU setup

The first pilot used a separate environment to avoid dependency changes to the working
YAMNet/Whisper environment. For a new environment, using installed `uv`:

```powershell
uv venv --cache-dir .cache/uv --python .venv/Scripts/python.exe .cache/sfx-venv
uv pip install --cache-dir .cache/uv --python .cache/sfx-venv/Scripts/python.exe --torch-backend cpu -e '.[sfx]'
.\.cache\sfx-venv\Scripts\python.exe -m dotears prepare-model clap --output models/clap-htsat-unfused
```

Preparation downloads roughly 618 MB of official weights/configuration/tokenizer data. It pins
all files to the full repository commit returned by the public model API, checks exact sizes
and the publisher's weight SHA-256, and saves a local manifest. It downloads no Python code
and needs no account. The optional weights/runtime are not bundled in DotEARS distributions.

The official checkpoint currently contains `pytorch_model.bin`; loading uses PyTorch's
`weights_only=True` with PyTorch >=2.6 and `local_files_only=True`. The tested runtime is
PyTorch **2.9.1+cpu**, Transformers **4.57.6**. No CUDA or GPU is required.

## Run on an existing report

```powershell
.\.cache\sfx-venv\Scripts\python.exe -m dotears sfx 'C:\ReferenceLibrary\ID\report.json' --model models/clap-htsat-unfused --output validation-output/sfx-hypotheses.json --event-id attack_017
```

Omit `--event-id` to process all reported attacks, or repeat it to select several. `--window
excerpt` uses the actual exported excerpt boundaries in native PCM. `--window context
--context-seconds 3` uses up to three seconds around each attack. Both use mono audio resampled
to 48 kHz. `--padding repeatpad` follows the prepared model's default; `--padding pad` is a
separate preprocessing condition, not an automatically chosen winner.

The output is separate from the source report. `top_sfx_hypotheses` contains up to three
ranked effect descriptions; `best_match` also considers music, speech, speech with music and
noise. `proposed_label` is null when a control description ranks first. `label` remains
`unknown` in every case. Cosine similarities and margins are uncalibrated, not probabilities.
The JSON records model/prompt/report hashes, the actual windows and runtime versions.

The frozen v1 prompt set covers whoosh, notification, UI click, UI tone, pop, impact, riser
and glitch, with three descriptions per class. These categories overlap: a notification can
be a UI tone or a musical chime. User notes are not included in model prompts. DSP peaks are
not new confirmed sound events. The module neither separates sources nor discovers smooth
effects missing from the original attack set. MCP `analyze_video` does not enable this
experimental pass automatically; an authorized local task can invoke the separate CLI.

## First pilot, 2026-10-27

Protocol was written before inference: the 12 user-reviewed Apple Stocks events, fixed
descriptions, two windows (existing excerpt and three-second context), model-default
`repeatpad`, CPU with four threads. Inference workers blocked socket connections and set
offline flags. User notes were joined to results only afterwards. No prompt/threshold fitting
or training was performed.

In **both conditions all 12 best overall matches were music or music with speech**. The
effect-only rankings sometimes included notification but also offered effects for excerpts
the user described as speech or nothing noticeable. This does not support deploying automatic
SFX names. The two process runs took 39.584 s / 18.350 s and sampled peak process-tree RSS was
1,106,325,504 / 1,000,488,960 bytes. These are individual runs with different warm-cache states,
not a window-performance benchmark.

For diagnosis, the public [Kenney Interface Sounds](https://kenney.nl/assets/interface-sounds)
CC0 pack was downloaded from its official link (834,536-byte archive). Its included license
was read and retained. Twelve fixed filename-based controls were tested with repeat padding
and zero padding. For example, `click_001` ranked UI click under repeat padding, while both
`confirmation` controls ranked UI tone under both conditions. Several very short controls
changed their leading class when padding changed. Filenames are provisional descriptions;
the agent did not independently listen, so this is not a semantic accuracy benchmark.
In particular, a file named `scroll` is not assumed to be a whoosh.

Private pilot artifacts, inputs, protocols and JSON comparisons are under gitignored
`validation-output/sfx-pilot-2026-10-27/`. The controls are stored there with their source,
license and SHA-256 manifest. No third-party audio was added to the source distribution.

## Next evaluation gate

A 30-example offline labeling page is prepared in gitignored
`validation-output/sfx-labeling-2026-10-27/index.html`, with no prefilled human labels. It mixes
CC0 swish/air sounds from three authors with Kenney tonal/UI candidates. See
[manual validation](manual-validation.md#clean-sfx-labeling-worksheet) for the generator,
export format, provenance and limitations. The user completed all 30 ratings on 2026-10-28;
the labeled baseline is recorded below.

1. Curate independently listened clean examples for whoosh, notification-like chime and UI
   click, plus music/speech/no-effect negatives. Keep provenance, reuse licenses and ambiguous
   or multilabel descriptions; do not infer semantic truth solely from a filename.
2. Freeze development and held-out splits before choosing descriptions or decision thresholds.
3. Test the same effects alone and at known levels over music/voice. Measure per-family
   precision/recall, false positives and abstentions, with preprocessing fixed in advance.
4. Promote an SFX label only if the held-out results support it. Otherwise retain hypotheses
   or evaluate another local method. A larger context or source separation is not an assumed fix.

## First completed labels and frozen baseline, 2026-10-28

The export matched the dataset ID, all 30 source/preview hashes and family bindings. A second
downloaded snapshot had identical sample annotations. The user labeled eight whooshes, five
notifications, six UI clicks, five other sounds and six uncertain sounds. Confidence was
high for 19, medium for three and low for eight. No labels were inferred from filenames or
changed to match model output.

The primary descriptive comparison used the 22 high/medium, non-compound ratings: eight
whooshes, five notifications, five UI clicks and four other controls. The six uncertain
answers and two additional low-confidence answers were retained separately, not scored as
positive or negative semantic ground truth. A curated reference catalog preserves provenance
and family IDs; no development/held-out split or model training has been performed.

All 30 exact playback previews were passed through the original checkpoint and unchanged
motion-sfx-v1 prompts, mono at 48 kHz with repeat padding and four CPU threads. Human labels
were joined after inference. Socket connections were blocked. First-name agreement on primary
targets: whoosh 8/8, UI click 5/5, notification 0/5 (all five ranked the broader `ui_tone` first,
with `notification` in the overall top three). The four primary other controls ranked `ui_tone`
three times and `ui_click` once. Relabeling every UI tone as a notification would therefore
introduce three false target names on those controls; raw argmax also produces one false click.
All confirmed SFX output labels remain unknown. This is a comparison of ranked hypotheses,
not a deployed classifier's accuracy or a held-out generalization result.

The run took 47.566 s with sampled peak process RSS of 906,924,032 bytes. Artifacts are in
gitignored `validation-output/sfx-labeled-eval-2026-10-28/`, including `audit.json`, the untouched
submitted export, curated references, ambiguous cases, frozen protocol, baseline results and
`RESULTS.ru.md`. All eight whoosh labels share one source family; random file splits would
overstate independent evidence. Next compare audio-reference similarity as a separate method
for distinguishing notification-like tones from other UI tones, with whole-family validation
and controlled background mixtures before promoting automatic names.

## Audio-reference and background experiments, 2026-10-28

`src/dotears/sfx_references.py` adds a separate experimental audio-to-audio ranking method.
It normalizes CLAP embeddings, averages within each source family, then gives families equal
weight in each class prototype. Whole query families and identical source hashes are excluded
before constructing prototypes. Scores remain uncalibrated; confirmed labels stay unknown.
This method is used by the evaluation script, not automatically by `analyze`, `sfx` or MCP.

The frozen first experiment evaluated 274 inputs, including 198 controlled mixtures. On the
original clean previews, first-name agreement was notification 4/5, UI click 5/5 and other 1/4.
All eight whooshes were unscorable for audio-reference generalization: excluding their only
family removes every whoosh reference. These are descriptive family cross-validation results;
Kenney subfamilies still share a source pack, and this is not an independent final test.
All 12 no-insertion background windows nevertheless ranked a target first, so apparent success
on mixed notifications was not evidence of detecting the inserted effect.

A separately frozen exploratory follow-up used 0.7-second windows and 12 explicit background
references. Reference backgrounds used cinameng music and a local English Zira TTS voice;
evaluation backgrounds used drakzlin music and the Russian Irina TTS voice. On 12 new
no-insertion windows, background ranked first 11 times and notification once. However,
UI click ranked first in 0/45 mixtures with a known inserted UI click, and notification in
13/45 corresponding mixtures. Short-window labels inherit the original source annotation;
the shortened/combined audio was not independently heard. Oracle-known insertion positions
make this a favorable test, not end-to-end attack-detector validation. No automatic rule was
promoted. Two conditions changed together, so this is not a causal ablation.

Artifacts, protocols, source receipts, full rankings, audio hashes, component SNR/clipping
checks and Russian assessment are in gitignored
`validation-output/sfx-reference-eval-2026-10-28/`. The follow-up's `protocol-notes.json`
clarifies one stale inherited metadata flag; original frozen protocol bytes remain intact.
Both CPU runs blocked sockets, used the existing checkpoint and took 97.536 / 64.106 seconds;
sampled peak process RSS was 919,248,896 / 918,417,408 bytes.

To rerun a **new prepared** local corpus with an existing `protocol.json` and `corpus.json`
(the script refuses to overwrite results), use the isolated SFX environment with psutil:

```powershell
.\.cache\sfx-venv\Scripts\python.exe scripts\evaluate_sfx_references.py validation-output\NEW-RUN --model models\clap-htsat-unfused
```

An independent eight-sound review is prepared alongside the first experiment as `index.html`.
It contains fixed selections from qubodup and StarNinjas CC0 packs, with no labels inferred
from filenames or model rankings. Its dataset ID is `231825dd8df68bf1c877ba38`; export is
`dotears-sfx-round2-labels.json`. Existing 30 annotations are preserved. The next modeling
experiment needs to address short foreground events in mixtures while preserving both
positive recovery and background rejection; more clean labels alone do not fix this failure.

## Completed second review and selection correction, 2026-10-28

The user submitted all eight ratings. Source/preview hashes, dataset/group IDs and completed
count were verified; an untouched copy and audit are retained in
`validation-output/sfx-reference-eval-2026-10-28/round2-review/`. The four qubodup examples
were labeled whoosh/high; the four StarNinjas examples were other (three medium, one high),
each described by the user as sounding like a knife blade. Existing model predictions
preceded submission: both text and audio-reference methods ranked whoosh first on all eight.
Thus the four other examples are four false first-ranked whoosh guesses, not successful
whoosh recognition. Confirmed outputs still remain unknown. No rerun or threshold tuning
was needed to establish this; no reviewed holdout was inserted into the reference bank.

The candidate selection itself was too narrow: four bamboo-stick variants and four sword
variants do not represent a varied motion-design/UI review. The second pack supplies negative
examples, not another positive whoosh family. Preserve these confusing negatives for
regression. Future candidate selection must cover acoustically varied editing transitions,
notifications and UI clicks; source names containing whoosh/woosh are insufficient evidence
of task suitability. No replacement annotation page was requested from the user in this step.

## Short-event transfer heads and foreground masking, 2026-10-29

`src/dotears/sfx_events.py` adds research helpers for a prefix-estimated spectral mask,
whole-family/source/PCM exclusion, source-balanced linear-head fitting, and JSON coefficient
export/prediction. `scripts/evaluate_sfx_events.py` uses the prepared local YAMNet's 1024-D
embeddings for a short-event experiment. This follows the official
[YAMNet transfer-learning workflow](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio),
but does not imply suitability for editing SFX. Base weights remain frozen; only small linear
classifiers are fitted. New helpers are not automatically called by normal analysis or MCP.

Thirty distinct high/medium labeled source files (12 whoosh, five notification, five UI click,
eight other, including the four blade-like negatives) form ten families. Training contains
30 clean crops, 270 mixtures over music/voice and 12 no-insertion controls. Query audio uses
different music/voice: 30 clean crops, 270 mixtures, 12 background windows and 12 real excerpts.
Before each fit, all derivatives of the query family/source and identical PCM are excluded.
The scaler is fitted only to allowed training rows. Class/family/source weighting prevents
near variants from dominating the relative training weights. This is exploratory family
cross-validation on known data, not a new independent final test. Cropped/mixed labels inherit
the original full-preview labels; perceptual audibility has not been separately reviewed.

Four fixed candidates were measured: raw YAMNet, masked YAMNet, CLAP linear and concatenated
CLAP/YAMNet linear. The latter matched 12/12 whoosh, 4/5 notification and 4/5 UI click clean
names, but also assigned target names to 7/8 clean other controls. At 0 dB it matched only
3/36 whooshes, 3/15 notifications and 2/15 clicks. No candidate passed all preregistered
research checks for positive recovery, background rejection and blade rejection. The simple
spectral mask was rejected too; removing blade-like guesses while losing desired sounds is
insufficient. All predictions retain unknown/null confirmed label/confidence.

Artifacts are in gitignored `validation-output/sfx-events-2026-10-29/`: frozen protocols,
corpus, features, 44 JSON heads, full results, original-note comparisons, rejection decision,
verification and `RESULTS.ru.md`. Every one of 1,296 predictions was recomputed from exported
parameters; all normalizer fits, exclusions and 540 component SNR/no-clipping checks passed.
Regression: **66 passed, two standalone model tests deselected**; actual YAMNet/CLAP runs were
performed separately. Ruff passed. Both backbones ran on CPU with sockets blocked; no new
weights, packages or user annotations were requested. The mask is not source separation.

## Explicit query-conditioned extraction

The later [AudioSep study and command](separation.md) added a distinct optional `separate`
stage. It provides estimated waveforms for a supplied description; it does not promote the
description into a recognized class. Conditional extraction helped some notification and
click mixtures, but preservation/background checks failed jointly and whooshes depended
strongly on wording. All confirmed labels stay unknown. Normal analyze/MCP behavior is unchanged.
