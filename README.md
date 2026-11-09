# DotEARS

DotEARS is a free, MIT-licensed alpha for analyzing the sound of video references. It runs on the
CPU and saves its analysis locally. Give it a short video to get timestamps for audio attacks,
abrupt image changes, and places where the two line up. The report includes WAV excerpts so you
can listen to the candidates yourself or pass them to an agent.

## Try the demo

Install Python 3.11 and FFmpeg. Make sure both `ffmpeg` and `ffprobe` are on your PATH. From the
project folder on Windows, run these commands in PowerShell.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\dotears.exe doctor
.\.venv\Scripts\dotears.exe analyze examples/demo.synthetic.mp4 --profile motion --output validation-output/demo-library
```

On Linux or macOS, create the environment with `python3.11 -m venv .venv`. Use `.venv/bin/python`
and `.venv/bin/dotears` in the commands above. The first public demo was verified on Windows.

The [eight-second demo](examples/demo.synthetic.mp4) contains synthetic percussion and a picture
cut at 3.5 seconds. The [example JSON report](examples/report.synthetic.json) shows what the
analyzer found. Open the `report_path` returned by the command, or open `report.md` and the WAV
clips in the same folder. The demo is generated from code in this project.

## What works in this alpha

- **Local analysis.** Extract audio, find attacks and energy changes, estimate rhythm, compare
  candidate peaks with image cuts, and export mixed audio excerpts. Repeated analysis can reuse
  the saved result when the input and options match.
- **Optional speech and audio tags.** Prepare Whisper for timestamped transcripts and YAMNet for
  broad audio labels such as speech and music. See the [local model setup](#optional-local-models).
- **Agent access.** Call `dotears analyze` from a shell task, use the Python API, or install the
  `mcp` extra for the local stdio server. The server accepts files under a configured media folder.
  URL input is opt-in. See [agent setup](docs/agents.md).
- **Optional URL input.** The `download` extra enables supported public video links. Compatibility
  depends on the site, available formats, and host network access. Local files are the easiest
  way to get started. See [network and download limits](docs/network-access.md).

Audio attacks are candidate timestamps. An excerpt may contain music, speech, and effects at the
same time. Closely spaced attack markers can describe the same sound.

## Experimental sound effects

The separate Windows review used by Dodik checks for quiet impulses and can draft English prompts
for tools such as ElevenLabs based on the sound's shape. Some drafts describe a sound with an
unknown effect name. These are text drafts for recreating a similar sound. The full local setup uses
separately prepared model files, a recognition policy, and an SFX environment outside this public
checkout. A fresh checkout can produce a partial review. Specific effect names, roles, and isolation
quality are still experimental, especially for quiet sounds under music and speech.

See the [experimental review guide](docs/agents.md#experimental-windows-review-for-dodik),
[effect research](docs/sfx-mixtures.md), and [known limits](docs/limitations.md).

## Optional local models

Install the extras and explicitly download the models once. Local analysis then loads them from
those directories.

```powershell
.\.venv\Scripts\python.exe -m pip install '.[asr,yamnet]'
.\.venv\Scripts\dotears.exe prepare-model whisper --size tiny --output models/whisper-tiny
.\.venv\Scripts\dotears.exe prepare-model yamnet --output models/yamnet
.\.venv\Scripts\dotears.exe analyze examples/demo.synthetic.mp4 --asr-model models/whisper-tiny --yamnet-model models/yamnet
```

For the Python API and report structure, see the [architecture guide](docs/architecture.md) and
[JSON Schema](schemas/report-0.1.json). To regenerate the demo, use an FFmpeg build with
`libx264` and run `python scripts/make_demo.py --output validation-output/demo-copy.mp4` from
the installed environment.

## Local files and license

The default analysis library lives outside the checkout. The quick start writes into the ignored
`validation-output/` folder so its files are easy to find. Keep personal media and their reports
local. Reference clips retain the reuse rights of their sources.

DotEARS source is available under the [MIT License](LICENSE). The included demo uses only
project-generated material. FFmpeg and optional dependencies have their own
[licenses and notices](THIRD_PARTY_NOTICES.md).
