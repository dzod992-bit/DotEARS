# Experimental local AudioSep extraction

The explicit `separate` command attempts to retain a described sound from one existing attack
context. It preserves the mixed window and saves an estimated component. The query is an
instruction to the separator, not evidence that the sound exists. Confirmed `label` stays
`unknown`, `confidence` stays null. Normal `analyze` and MCP do not invoke this pass.

## Prepared CPU environment

The working Windows project already has `.cache/audiosep-venv` and `models/audiosep` prepared.
For a **new** isolated environment, with installed uv:

```powershell
uv venv --cache-dir .cache/uv --python .venv/Scripts/python.exe .cache/audiosep-venv
uv pip install --cache-dir .cache/uv --python .cache/audiosep-venv/Scripts/python.exe --torch-backend cpu -e '.[separation]' 'torch==2.9.1' 'transformers==4.57.6' 'torchlibrosa==0.1.0'
.\.cache\audiosep-venv\Scripts\python.exe scripts\prepare_audiosep.py --output models\audiosep
```

Preparation explicitly downloads the 1,264,844,076-byte official checkpoint, pinned source
files and about 1.36 MB of tokenizer/configuration data. It checks SHA-256 and resumes partial
downloads; it executes no downloaded code. The runtime uses the separator and its paired
text encoder from that same checkpoint. No additional learned RoBERTa weights are needed.
The first validation also downloaded the publisher's 2.35 GB companion CLAP checkpoint,
verified all 204 text/projection tensors bitwise against the embedded weights, and removed
that redundant file afterwards. The default preparation script does not download it.

Validate an existing preparation without networking:

```powershell
.\.cache\audiosep-venv\Scripts\python.exe scripts\prepare_audiosep.py --output models\audiosep --check-only
```

The inspected AudioSep source revision is `944583f18b84589dc965de3ad77525c945334252`;
checkpoint revision `0807a713283035d8b0d0628f1b8e3d6580459f7b`. The adapter loads only
hash-pinned separator modules under private names. Weights use `weights_only=True`, CPU
mapping and mmap. Required parameter names/shapes must match strictly. Text-forward math
was checked against the inspected upstream functions, including singleton batching.

## One described context

```powershell
.\.cache\audiosep-venv\Scripts\python.exe -m dotears separate 'C:\ReferenceLibrary\ID\report.json' --event-id attack_001 --prompt 'The sound of a notification chime.' --model models\audiosep --output validation-output\notification-estimate
```

Use an existing `audio_attack` ID and a new output directory. The command takes up to three
seconds around the attack, downmixes to mono, resamples to 32 kHz and runs on CPU/four threads
by default. `--threads` accepts 1–16. English descriptions were tested. It writes:

- `mixture.wav`: the original context used by the model, mono 32 kHz float.
- `estimated.wav`: the model output, same duration/rate, without amplitude normalization.
- `separation.json`: original report/native hashes, source timeline, query, model evidence,
  output hashes and explicit uncertainty. Relative output energy is not a quality score.

Existing output directories are refused. The original analysis is unchanged. New artifacts
are staged in the output parent and atomically moved into place on completion. This is an
explicit optional local-task route, not a validated automatic named-SFX pass.

## Executed study, 2026-10-29

The frozen primary protocol covered 198 known mixtures (22 human-labeled sources × three
backgrounds × three component levels), 22 clean preservation cases, 36 no-insertion
background/query combinations, 24 other-sound/query controls, three silence queries and
12 real-video query contexts: **295 calls**. Full original effects were retained in the
three-second inputs. Correct class descriptions were supplied from ground truth, making
this a conditional-separation test rather than a semantic detection test.

At nominal 0 dB, median zero-lag SI-SDR improvement measured **on the interval where effect
and background overlap** was notification **+9.26 dB**, UI click **+6.40 dB**, whoosh
**−12.67 dB** for the primary descriptions. Target gain stayed within ±6 dB with positive
projection in 13/15 notification cases, 9/15 click cases and 9/36 whoosh cases. Only 16/36
no-insertion background/query outputs attenuated by at least 15 dB. The research gate failed.

A separately frozen exploratory follow-up ran 236 calls using the two remaining descriptions
from the pre-existing motion-sfx-v1 prompt set, at 0 dB for whoosh/click. A whoosh formulation
reached median +3.99 dB, but target retention was only 22/36 and background attenuation 6/12.
No per-clip oracle choice of the best wording or threshold fitting occurred. This demonstrates
prompt sensitivity, not a validated universal whoosh profile. UI alternatives also failed
the joint preservation/background requirements.

Artifacts under gitignored `validation-output/audiosep-2026-10-29/` contain both frozen
protocols, full results, all output WAVs, input construction checks, a rejection decision,
listening examples, reproduction scripts and `RESULTS.ru.md`. All 531 evaluation outputs
and 712 reference measurements were independently checked. A real `separate` CLI call on
the Apple Stocks report matched the benchmark input/output byte-for-byte numerically and
left the original report unchanged. A known correct query is still required; real-video
separation quality was not established by a clean reference or independent listening.

Measured CPU/four-thread runs: 423.730 s and 342.802 s; sampled peak process RSS
1,985,921,024 / 1,946,025,984 bytes. The primary median warm call took 1.3106 s for a
three-second input. These are observations on this host, not a hardware-independent SLA.
Inference sockets were blocked; private media was not uploaded. Regression: 77 passed,
two standalone model tests deselected; actual model runs were performed separately.

The current evidence supports an optional inspection aid, most promising for notification-like
sounds. It does not establish reliable automatic discovery, background rejection, source
separation quality on arbitrary videos, or reuse rights for extracted components.

## Sources

- [AudioSep code and MIT license](https://github.com/Audio-AGI/AudioSep).
- [Official checkpoints](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint).
- [TorchLibrosa](https://github.com/qiuqiangkong/torchlibrosa).
