# Executed validation

Date: **2026-10-24**. Host: Windows 10 build 19045, AMD Ryzen 7 1700 (8 cores / 16 threads),
Python 3.11.15, FFmpeg 8.1.1. CPU execution throughout; no CUDA/ROCm or GPU model execution.
Exact installed Python versions are in `requirements-validation.windows-py311.txt`.

## Varied-mixture family profile — 2026-10-30

The [new study](sfx-mixtures.md) prepared 139 sources from 88 authors, 15 backgrounds and
1,084 windows including 834 mixtures. Component reconstruction, SNR and split leakage audits
passed. Four scope-ambiguous old controls were quarantined without relabeling human reviews.
The newly fitted whoosh and broad click/beep/chime detectors compose in one explicit local
CLI pass; precise notification/confirmation roles remain unknown.

All 114 Apple Stocks attacks were checked: 5 whoosh windows, 5 UI-signal windows, 104 unknowns.
The whoosh at 7.343311s agrees with the user's note; the signal at 1.536508s agrees at the
broad family level. The 19.495692s whoosh differs from a broad UI note and is disputed.
All three reviewed negative peaks remain unknown. Counts refer to windows, not distinct
effects. Full-video accuracy cannot be measured from partial annotations.

Composed-profile development regression on all 135 source-held windows gives 14 correct
whoosh names and one zipper false name, plus 24 correct broad signal names and no wrong
signal names. The repeatedly used development sources are not a fresh independent holdout.
A full generated track recovers 7/12 target occurrences and exposes three premature UI
assignments. Their windows contain a later inserted signal, while candidate peaks fall
outside its active interval. The profile therefore reports window presence and null effect
onset. No thresholds were retuned after that check. FUSS extraction/component experiments
failed their gates and are not used by the selected profile.

128 regression tests passed, two standalone model tests deselected (46.04s); Ruff passed.
Three fresh offline windows reproduced all 1,075 cached features exactly, using 3 CED and
6 CLAP forwards with zero cache hits. Original video/report/annotation hashes, all 114 raw
feature vectors and decoded mixed-clip PCM were preserved. FLOAT WAV container hashes can
differ because their PEAK chunks contain write timestamps. Local evidence and the Russian
result are in `validation-output/sfx-mixtures-2026-10-30/`.

## Broad UI label scope — 2026-10-30

The user includes clicks/taps, beeps and confirmation tones in the UI category. Reference
policy/report v2 now bind `ui_sound` to explicit taxonomy `dotears.sfx-types/2`. Existing
notification/other reviews, bank audio, scores, thresholds and disabled templates are preserved.
Legacy neural models/prompts remain frozen; this is not retraining or new tone validation.
110 regression tests passed, two standalone model tests deselected (41.90s); Ruff passed.
Actual migrated-policy runs returned 60 unknowns on Apple and the same 11/15 known-source
integration recoveries with no extra names. Files are in `validation-output/sfx-ui-scope-2026-10-30/`.
See [label scope and migration](sfx-types.md).

## Raw SFX and known-pattern follow-up — 2026-10-29/30

The raw CED/CLAP experiment now covers 72 sources, 23 disjoint foreground author groups and
693 windows, with 24 calibration-selected candidate combinations. The selected model's fresh
0dB recovery was 1/6 whoosh, 0/6 notification and 0/6 click. All general naming heads failed
and remain disabled. Float CED parity against 16 upstream reference forwards was exact;
the quantized export failed and was not selected. Fresh CED/CLAP feature verification and
source/split/fold/clip audits are recorded in the local verification receipt.

A new `sfx-match` command checks known human-reviewed waveforms. Fifteen initial patterns
passed alternate-background controls; a 64s generated integration track exposed one false
swish match. Its pattern was disabled without threshold changes. Fourteen active patterns
remain (6/5/3 whoosh/notification/click); rescoring the same track recovers 11/15 sources with
no extra names. This is a bounded known-pattern capability and a repaired regression check,
not a new independent test of general recognition.

Actual raw and reference CLI passes with their active policies each returned 60 unknowns on
Apple Stocks. The explicit notification and whoosh notes remain unresolved. Twelve ratings,
five markers and all 60 event bindings are checked; whole-video accuracy is unavailable.
105 regression tests passed, two standalone model tests deselected (42.72s); Ruff passed.
Local source/model inference stayed offline. Artifacts and the Russian result are in
`validation-output/sfx-v2-2026-10-29/`. See [method and commands](sfx-presence.md).

## Automatic three-query check and partial video comparison — 2026-10-29

The new optional `sfx-auto` CLI checks all three fixed queries on every existing attack and
uses raw/estimated CLAP features with source-family-excluded presence thresholds. A completed
266-window calibration ran 798 query estimates; notification/click thresholds failed, while
whoosh passed the alternate-background check with low coverage. Its A threshold selection
still accepted one blade-like false example. The same foreground sources appear in A/B.

The frozen policy then ran all 60 attacks of the original Apple Stocks report: 323.515 s,
exit 0, one provisional whoosh and 59 unknowns. That sole accepted name at attack_001 differs
from the user's notification/bell note; their whoosh at attack_017 remained unknown. All
12 ratings and five approximate marks matched the original analysis/source/event bindings.
The partial review cannot establish whole-video precision or recall.

Naming promotion is withheld for all three classes. The original experiment is preserved;
the active policy changes only enabled flags, retaining thresholds/weights. The final sidecar
deterministically applies that rejection to the same features and returns 60 unknowns.
This is rejection after an unsuccessful transfer check, not a successful recognition claim
or a new test after fitting to the video. Normal analysis/MCP remains unchanged.
The actual CLI also ran with the active policy: 60 unknowns, exit 0, 26.395 s with all
180 estimates/audio embeddings cached and feature-identical to the original pass.

Regression: 87 passed, two standalone model tests deselected, 53.03 s; Ruff passed. Actual
AudioSep/CLAP runs used the prepared isolated environment, with inference sockets blocked.
Artifacts, source/audio/model hashes, frozen policies, comparisons, all excerpts and the
Russian report are in gitignored `validation-output/sfx-auto-2026-10-29/`. See
[automatic check](sfx-auto.md) for the method and limitations.

## Local AudioSep separation and explicit CLI — 2026-10-29

The official checkpoint was pinned and SHA-256 verified. The adapter strictly loaded 332
separator, 199 text and four projection keys from the same checkpoint; upstream text-forward
parity was exact, including singleton batching. All 204 paired text/projection tensors matched
the publisher's companion file bitwise. That redundant 2.35 GB file was removed; the retained
1,264,844,076-byte checkpoint supplies both separator and text encoder. Source/tokenizer data
and the isolated CPU environment remain local. No existing unfused CLAP weights were substituted.

The frozen primary study ran 295 query/window cases: 198 known mixtures, 22 clean cases,
36 background/query controls, 24 other-sound/query controls, three silence queries and 12 real
contexts. Correct descriptions were provided from known labels: this is conditional separation,
not sound recognition. At 0 dB nominal component level, overlap-support SI-SDR improvements
were notification +9.26 dB, UI click +6.40 dB, whoosh -12.67 dB. Only 16/36 background/query
controls attenuated sufficiently, so the joint research gate failed. A separately frozen
236-case wording follow-up used the unchanged earlier prompt set: one whoosh formulation
reached +3.99 dB, but retention and background rejection remained insufficient. No per-window
best-prompt selection was used, and no automatic labels were enabled.

All 531 outputs and 712 reference metrics were independently checked. Known components sum
to each mixture within 5.42e-7; model-domain SNR differs from nominal by at most 0.028 dB.
The explicit `separate` CLI ran on Apple Stocks attack_001 in the normal sandbox. Its input
and output matched the corresponding benchmark exactly; original report/native hashes remain
unchanged. The command preserves raw/estimated WAVs and unknown/null labels. Normal analysis
and MCP do not enable it. See [separation](separation.md) for setup and limits.

CPU/four-thread timings: 423.730 / 342.802 s, peak sampled process RSS 1,985,921,024 /
1,946,025,984 bytes. Sockets were blocked during benchmark inference. Final code checks:
**77 passed, two standalone model tests deselected**, 40.46 s; actual model execution occurred
separately. Ruff passed. Artifacts, protocols, listening examples, reproduction and Russian
report are under gitignored `validation-output/audiosep-2026-10-29/`.

## Short-event and mixed-training experiment — 2026-10-29

Four research candidates were executed on prepared local models: YAMNet linear head,
prefix-mask YAMNet head, CLAP head and CLAP/YAMNet concatenation. Base weights stayed frozen.
Thirty distinct labeled sources from ten families generated 312 training and 324 query rows;
whole query families, original hashes and identical input WAVs were excluded before fitting
each scaler/head. The query music/voice differ from training. Synthetic views are correlated;
these are exploratory family checks on a known convenience set, not an independent benchmark.

CLAP/YAMNet clean first-name matches were 12/12 whoosh, 4/5 notification, 4/5 UI click, but
7/8 other clean examples also received target names. On 0 dB mixtures it matched only 3/36
whoosh, 3/15 notification and 2/15 UI click. All four candidates failed the frozen research
gate. No automatic labels were promoted; normal analyze and MCP behavior are unchanged.

Artifacts in gitignored `validation-output/sfx-events-2026-10-29/` contain two protocols,
44 exported heads, 1,296 recomputed predictions, all training-only normalizer/exclusion checks,
540 component SNR/no-PCM-clipping checks, audio diagnostics, a rejection decision and Russian
report. CPU runs took 85.110 / 150.684 seconds with sampled peak process RSS 622,690,304 /
592,089,088 bytes; cache states differ. Model-inference/training sockets were blocked. No new
weights or packages were installed. Regression **66 passed, two model tests deselected**, in
42.64 s; actual backbones ran separately above. Ruff passed for source, tests and scripts.

## Second SFX labels and candidate-selection failure — 2026-10-28

All eight annotations from `dotears-sfx-round2-labels.json` matched dataset
`231825dd8df68bf1c877ba38`, source/preview hashes and family bindings. Four qubodup sounds
were labeled whoosh/high; four StarNinjas sounds were other with blade-like notes (three
medium, one high). Both pre-submission model methods ranked whoosh first for all eight,
including all four negative examples. This exposes false first-ranked guesses on confusing
metallic sounds; it does not establish general editing-SFX quality. Confirmed labels remain
unknown. The four-plus-four weapon/bamboo selection was too narrow for the intended task.

An exact submitted copy (SHA-256
`a467fc24ff277735088b87b56604fc41812f2ceacb1e84efb9c96c410c8df0ef`), audit, comparison and
Russian assessment are saved in gitignored
`validation-output/sfx-reference-eval-2026-10-28/round2-review/`. Human labels were joined to
already saved predictions without rerunning the model, modifying the reference bank or
tuning thresholds. This step changed evidence/documentation only; the 61-test code validation
above the earlier artifacts remains applicable.

## Audio-reference ranking and controlled mixtures — 2026-10-28

The first frozen family-exclusion experiment processed 274 inputs, including 198 mixtures:
22 user-labeled sounds × three backgrounds × +6/0/−6 dB component SNR. Clean first-name
agreement was notification 4/5, click 5/5, other 1/4; whoosh lacked an independent reference
family. All 12 no-insertion background windows ranked a target first. A frozen exploratory
follow-up used short windows and explicit background references, evaluated on different music
and a different TTS voice. It processed 244 inputs including another 198 mixtures. Background
ranked first 11/12 times, but known inserted clicks ranked first 0/45 and notifications 13/45.
No automatic labels were enabled. These are small correlated experiments, not final accuracy.

Both runs used local CLAP on four CPU threads with sockets blocked (97.536 s / 64.106 s;
sampled peak process RSS 919,248,896 / 918,417,408 bytes). Every input hash, family exclusion,
all 396 component SNRs and absence of PCM clipping were verified. Original video reports and
human annotations remain unchanged. Protocols, a metadata clarification for the exploratory
follow-up, results and `RESULTS.ru.md` are retained under gitignored
`validation-output/sfx-reference-eval-2026-10-28/`. Reproduction scripts are retained there.

Eight candidates from two new CC0 source packs are reserved for human validation in that
folder's `index.html`, dataset `231825dd8df68bf1c877ba38`. Their embedded audio matches independently
decoded mono PCM with added silence exactly. Data/interaction/export/import/storage checks
passed in a Node VM; browser visual rendering was not inspected. See `docs/sfx.md`.

## Completed SFX labels and frozen-model comparison — 2026-10-28

All 30 exported ratings matched dataset `15c2ab44fcadcf5bd87cc79f`, source/preview hashes and
family bindings. The later duplicate download contains the same sample annotations. User
labels: eight whoosh, five notification, six UI click, five other and six uncertain. Confidence:
19 high, three medium and eight low. An untouched export, full audit and curated reference
catalog are under gitignored `validation-output/sfx-labeled-eval-2026-10-28/`.

The primary comparison policy was fixed before inference: use high/medium, non-compound
decisive labels (18 targets plus four other controls); retain six uncertain and two additional
low-confidence cases without treating them as positive or negative ground truth. All 30 exact
playback previews were scored, with the unchanged prior CLAP model and prompt hashes, 48 kHz
mono, repeat padding and four CPU threads. Human labels were joined only after prediction.
No training, threshold fitting, source separation or automatic label promotion occurred.

First ranked name agreed on eight of eight primary whoosh and five of five UI click examples.
All five notifications ranked `ui_tone` first (`notification` was in the overall top three).
The four primary other examples ranked UI tone three times and UI click once; converting all
UI tones into notifications would create three false target names. Confirmed output labels
remain unknown. This is a small descriptive comparison, not classifier precision/recall on
held-out data. All eight whoosh labels share one source family.

The local run completed in 47.566 s, with sampled peak process RSS 906,924,032 bytes, PyTorch
2.9.1+cpu / Transformers 4.57.6. Socket connections were blocked during inference. Results,
frozen protocol and `RESULTS.ru.md` are retained beside the audit. No runtime product code was
changed in this step; the previous regression suite remains the applicable code validation.

## Experimental CLAP SFX pilot — 2026-10-27

An optional, separate `sfx` CLI command now ranks fixed descriptions using the official
`laion/clap-htsat-unfused` checkpoint. Preparation pinned commit
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`, verified the publisher's weight hash, and retained
617,895,823 bytes of model/config/tokenizer files. CPU PyTorch 2.9.1 / Transformers 4.57.6 were
installed into a separate experimental environment. No media uploads or paid services were used.

The protocol was written before inference: the 12 user-reviewed Apple Stocks events, three
fixed descriptions per class, speech/music/noise controls, original excerpt vs three-second
context, default repeat padding, four CPU threads. Socket connections were blocked during
inference. In both conditions every best overall match was music or music with speech.
Effect-only rankings also proposed effects for the user's speech/no-noticeable-effect cases;
this does not justify automatic SFX labels. Original reports remain unchanged and all confirmed
labels remain unknown. Process timings were 39.584 / 18.350 s, sampled peak tree RSS
1,106,325,504 / 1,000,488,960 bytes; warm-cache differences prevent a performance comparison.

The official Kenney Interface Sounds CC0 archive supplied 100 clean control files. Twelve fixed
filename-based examples were tested with repeat and zero padding. Some UI click/tone results
were plausible; short-file rankings changed with padding. No independent listening labels
were assigned from filenames, and the controls are not a held-out semantic accuracy benchmark.

The regression run passed **53 tests in 44.16 s**, excluding two standalone YAMNet/Whisper
real-weight tests. Actual CLAP ran separately on the 24 mixed windows and 24 clean-control
conditions. Private protocols, comparisons, hashes and `RESULTS.ru.md` are under
`validation-output/sfx-pilot-2026-10-27/`. See [SFX setup and evaluation gate](sfx.md).

## First user listening sample and 0.1.3 — 2026-10-27

The user exported a listening worksheet with 12 Apple Stocks candidate ratings and five
approximate video-position marks. Source hashes, engine versions and all 120 reported attack
IDs/times matched the original two reports. The export was preserved byte-for-byte under
gitignored `validation-output/listening-review-2026-10-27/evaluation/`, with an `evaluation.json`
audit and `RESULTS.ru.md`. Nine excerpts were described as effects, two as nothing noticeable
and one as speech. Six were rated useful, four not useful and two uncertain. TransformAr
remains unrated; neither source has its full-review checkbox set. These selected observations
are not population precision/recall or automatic SFX recognition results.

The user described one excerpt as repeating the effect in the preceding starter item.
Those windows overlap by 0.580271 s (about 83%), with peaks 119.728 ms apart. Starter selection
now avoids over-80% overlap of the shorter excerpt, without removing any detector event.
Three additional marks are within 100 ms of a returned peak. All five fall inside existing
excerpt windows; the marks are approximate, so none is automatically counted as a detector miss.

DSP rechecking reproduced all 60 returned peaks and the original total of 114 candidates.
Version 0.1.3 exposes `attack_detection` counters in JSON/Markdown and warns when the limit
truncates the set. A fresh normal-permission local run with prepared YAMNet/Whisper completed
in 32.438 s and cached repeat in 0.946 s: 78 observations, 60 clips, 114 candidate attacks,
60 returned, 54 omitted by limit, zero removed by timeline filtering. All 60 attack IDs/times
are unchanged from the old report. New and historical reports pass the compatible schema.
The regression suite passed **45 tests in 42.83 s**, excluding the two standalone real-weight
tests; both models completed in the separate real-reference run just described.

## URL networking update — 2026-10-26

Dodik's two candidate URL attempts failed before media acquisition through a process-only
proxy at `127.0.0.1:9`. Read-only comparison found no such proxy variables in persistent user
or machine environment. An approved network operation inherited the existing system proxy at
`127.0.0.1:10809` and reached Vimeo, producing separate HTTP 401 / login-client errors.
No proxy or security settings were changed, no credentials were supplied, and no URL-to-report
success is claimed. Details and timings are in [network access](network-access.md).

DotEARS 0.1.2 now reports effective proxy endpoints without credentials and returns
`proxy_connection_failed` for this class of failure. Proxy settings remain inherited and no
automatic direct retry is attempted. The full regression run passed **44 tests in 78.28 s**.

## First successful discovered URL — 2026-10-26

The user subsequently relayed Dodik's Wistia success after standard host network approval.
The successful source was `https://berre.wistia.com/medias/4jcqq1cnwl`, titled
`TransformAr-teaser - With SRT-subtitles` in the acquired metadata. The preceding candidates
in that run failed separately (Vimeo requested login; YouTube lacked the requested muxed format).

The retained downloaded MP4 was independently probed: 34.005011 s, 1,890,012 bytes, 1280×720,
25 fps, stereo AAC at 44.1 kHz. SHA-256
`305c31478dfd9d56800260d3df86cf636a980849fc8b50d8e40810affef6bc1c` matches the report.
Version 0.1.2 completed both models, returned 79 observations / 60 clips and passed the report
schema. All 60 clips were compared byte-for-byte with native PCM at their recorded intervals.
The successful CLI call took 32.819828 s, exit code 0, cache miss; engine stage time was 31.173014 s.

The reported attack at 17.410431 s and visual change at 17.480 s differ by -0.069569 s.
The saved six-frame image sequence was inspected. It shows the orange scene changing to a
gray overhead view of a manhole that then opens. Editing intent and sound identity remain
hypotheses. YAMNet returned only Speech/Music labels; independent listening was not performed.

Artifacts, discovery summary and independent verification remain in gitignored
`validation-output/url-live-012-20261026T000502Z-9f6b44/`. A separate reference review records
the candidate relation, tags, caveats and a Remotion experiment. Original report data was preserved.
This validates one assigned agent-directed URL cycle; unattended recurrence remains untested.

## Real reference update — 2026-10-25

A user-supplied 23.125 s, 1280×720 MP4 with 48 kHz stereo AAC was processed on the same host.
Full CPU analysis took 42.576 s, baseline DSP took 11.108 s. Sampled peak process-tree RSS was
754,671,616 and 312,184,832 bytes respectively. These are individual observed runs.

Schema/source hashes, all 60 native stereo clips at exact sample boundaries and all 542 remuxed
AAC packet payload hashes were checked. Baseline/full DSP attack timestamps agreed. A real
MCP client initiated a fresh model-backed analysis in 27.507 s, then received a cached repeat
in 1.460 s. No external media service or paid API was used.

YAMNet returned 18 Speech/Music intervals; it did not identify specific editing SFX. Whisper
returned English text with suspected errors, including Stocks rendered as Stokes relative to
the visible end-card text. DSP found 114 attack candidates, of which the default cap retained 60.
That historical report lacks a truncation count (added for new reports in 0.1.3). At that stage,
no listening ground truth, named-SFX precision/recall or speech-onset error was measured.

Five raw visual observations included a three-frame cluster at 3.60–3.70 s. The two reported
audio/image coincidences were at 5.77/5.89 s (about 120 ms apart) and 20.90 s (about 2.5 ms apart).
Sampled image sequences were inspected; temporal coincidence does not identify the sound type.

Private outputs and verification evidence remain under gitignored
`validation-output/user-reference-2026-10-25/`. The user confirmed that Dot has access to this
computer. The subsequent Dodik test and permission fix are recorded below. The remainder
also retains the earlier synthetic baseline so the different evidence sources remain explicit.

## Dodik result and Windows permission fix — 2026-10-25

The user relayed Dodik's successful 0.1.0 assigned local-file run. Its saved `agent-check.json`
and report were independently read and checked: fresh call 39.162 s, cached repeat 1.474 s,
78 events, 60 clips, source SHA-256 matching the supplied video and a valid report schema.
Dodik's original tool trace was not independently inspected; caller identity comes from the
human-relayed response. This confirms the local-task level-4 scenario, not autonomous research.

The reported `WinError 5` was reproduced: a Windows `mkdir(mode=0o700)` directory was
inaccessible to the restricted runner, while inherited parent access worked. Python documents
this special Windows mode behavior in [os.mkdir](https://docs.python.org/3.11/library/os.html#os.mkdir).
Version 0.1.1 now uses randomly named, exclusively created directories inheriting the parent's
authorized access on Windows. POSIX still uses standard private temporary directories.
The fix applies to analysis staging, downloads and model preparation. Existing model copies
were recreated with unchanged content hashes; no existing ACL or system setting was edited.

Without escalation, `agent_check.py` on the real video then passed: fresh call 24.072 s,
cached repeat 1.459 s, the same 78 events / 60 clips. The new reference ID is expected because
the engine version changed to 0.1.1. Source/media/model contents were preserved.
Regression tests passed **39 tests in 49.22 s**, including three new temporary-workspace tests.
The pytest harness needed normal unrestricted temporary-directory access; the separate real
CLI run above used the restricted profile with no escalation.

Dodik's subsequent normal-permission 0.1.1 run also passed, per the user's relayed result and
independently checked artifacts: fresh 26.479 s, cached 1.753 s, 78 events and 60 clips both
times. The report version, schema, source hash and common analysis ID were checked without
escalation. The user reported no permission retry or environment/code change in this confirming
run. Evidence is in `validation-output/dodik-live/run-20261025T152346Z-93ffc8/`.

## What was actually run

- Installed the project in its own `.venv`, including optional download/MCP/ASR/YAMNet backends.
  `uv pip check` reported that all 101 installed packages were compatible.
- Generated original synthetic PCM and AAC-in-MP4 media. No third-party test clips are bundled.
- Ran pytest over extraction, invalid/missing/no-audio media, missing FFmpeg, duration and size
  limits, exact clipping, silence, known attacks, JSON Schema, cache/options, failure cleanup,
  URL errors/timeouts, archive traversal, CLI UTF-8, local neural models and actual stdio MCP.
  The final complete run passed **36 tests in 51.20 s**, including the delayed-audio timeline
  test and both prepared local neural models. No tests were skipped in that run.
- Ran real YAMNet and faster-whisper CPU inference using prepared local weights. The separate
  two-test real-weight run passed in 16.49 s with `HF_HUB_OFFLINE=1`.
- Used a real MCP client to initialize the local server, enumerate/call tools, receive an analysis
  report, and verify rejection of an out-of-root file and disabled URL input.
- Built both a source archive and installable wheel, then inspected their contents. No model,
  `.venv`, caches, user media or generated reference library was included.
- Tried the public yt-dlp YouTube test ID `BaW_jenozKc`; the platform returned unavailable.
  DotEARS returned a structured failure with local-file fallback. Successful live platform
  acquisition is **not** claimed for YouTube Shorts, TikTok or Instagram Reels.

The Windows restricted runner initially denied access to Python temporary directories. The
meaningful integration runs used an authorized process with normal temporary-file access.
An early lint invocation accidentally included the project's dependency cache; that environment
and cache were discarded and freshly installed before the reported successful tests.

## Synthetic timing evidence

`scripts/make_demo.py` created an 8 s video with 15 decaying percussive pulses at
0.5, 1.0, …, 7.5 s and a black-to-white frame change at 3.5 s. The baseline found 15 attacks,
all matched to the known pulse times, with maximum timestamp error **8.844 ms** in this sample.
Its tempo candidate was **120.273 BPM** against the 120 BPM pulse grid. It found the frame
change and the corresponding timestamp coincidence. This is a controlled clean-signal result,
not evidence of real-world motion-SFX classification precision.

YAMNet labeled portions of the pulse train Music, Drum machine, Percussion and Drum kit.
These are broad model hypotheses; the code did not rename them impact/whoosh/click.
Whisper returned no speech segments on the pulse-only video.

## Speech and mixed-audio smoke tests

Locally synthesized Windows voices generated English and Russian speech. Whisper recognized
the two English sentences correctly in this sample. In Russian, the target ending
“вместе со сменой кадра” was transcribed as “вместе со смены кадра”. No WER or speech-onset
accuracy claim is made from two generated samples.

A second 10 s synthetic MP4 combined the generated Russian voice (delayed 0.5 s), the pulse
track and the frame change. YAMNet detected Speech and Music above the selected threshold.
Whisper returned one Russian segment from 0.4–9.4 s; this illustrates coarse segment timing,
not a precisely measured speech onset. The same Russian grammatical error remained.
It produced 36 total observations and 31 mixed-track clips. Raw results stay in gitignored
`validation-output/` and `private-media/`; the public JSON example is sanitized.

## Measured resource use

Each row is one fresh worker process. Filesystem/JIT caches may already be warm; some runs
overlapped tests. These are actual observations, not controlled repeated benchmarks or guarantees.
Model loading and process startup are included in wall time. RAM is sampled every 50 ms as
the sum of process-tree RSS: shared pages may count twice and the exact peak may be missed.

| Input / mode | Wall s | Extract s | DSP s | Classifier s | ASR s | Sampled peak RSS MiB | All artifacts bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 s pulse MP4, baseline | 11.886 | 1.211 | 8.454 | — | — | 243 | 3,678,946 |
| 8 s pulse MP4, YAMNet + tiny Whisper | 32.816 | 1.239 | 6.692 | 20.201 | 1.822 | 771 | 4,425,775 |
| 10 s Russian mixed MP4, same models | 23.650 | 1.190 | 6.555 | 10.543 | 2.478 | 715 | 6,211,099 |

Prepared YAMNet bundle files totaled **18,275,381 bytes**. Whisper tiny's downloaded local
directory files totaled **78,205,120 bytes**, including configuration/tokenizer and small download
metadata. File-by-file SHA-256 manifests are stored with the local models. The TensorFlow runtime
is substantially larger than YAMNet weights (the downloaded Windows wheel was about 335 MiB),
which is why it remains optional.

Reproduction commands (PowerShell; use new output folders to avoid a cache hit):

```powershell
$env:DOTEARS_TEST_YAMNET = (Resolve-Path models/yamnet).Path
$env:DOTEARS_TEST_ASR = (Resolve-Path models/whisper-tiny).Path
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\python.exe scripts/benchmark.py private-media/demo.mp4 --output validation-output/new-baseline
.\.venv\Scripts\python.exe scripts/benchmark.py private-media/demo.mp4 --output validation-output/new-models --yamnet-model models/yamnet --asr-model models/whisper-tiny
uv build --out-dir dist
```

## Readiness against the requested levels

| Level | Evidence and remaining work |
| --- | --- |
| 1 — Audio Extraction | Implemented and exercised end to end on generated valid/no-audio MP4; errors tested. |
| 2 — Audio Understanding | DSP and local models work on synthetic and two real references. Manual SFX quality and specific motion-effect recognition remain unvalidated. |
| 3 — Agent Tool | Python/CLI and actual stdio MCP calls exercised. Core reports/clips were read and interpreted during this Codex work. |
| 4 — Dot Integration | **Assigned local-file and permitted URL scenarios passed.** Caller identity/host mode come from human-relayed responses; source/report/schema/hash/cache/clip artifacts were independently checked. |
| 5 — Autonomous Reference Research | **One assigned discovered-URL cycle exercised**, including interpretation and saved reference evidence. Full unattended recurring operation and reliable named-SFX quality are not established. |

Next decisive work: expand independent editing-SFX mixture evidence after the unsuccessful
three-query transfer check, improve candidate selection, then test authorized unattended
recurrence. See [manual validation](manual-validation.md)
and [Dot integration](dot-integration.md). Nothing was published to GitHub or configured by
this repository in Dodik's account, and no recurring task was created.

