# Third-party components

Reviewed 2026-10-24. DotEARS code is MIT. This source repository does not include upstream
implementation code, executable binaries, model weights, or third-party media. Installed packages
retain their own notices/licenses. A redistributed executable/model bundle needs a fresh audit.

| Component | Upstream license / distribution note |
| --- | --- |
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | BSD-3-Clause; binary dependencies carry additional notices. |
| [Librosa](https://github.com/librosa/librosa/blob/main/LICENSE.md) | ISC. Its installed dependencies include SciPy (BSD-3-Clause), Numba (BSD-2-Clause), llvmlite (BSD-style/LLVM notices), scikit-learn (BSD-3-Clause), SoundFile (BSD-3-Clause), libsndfile (LGPL), audioread (MIT), soxr (LGPL-2.1-or-later), pooch (BSD-3-Clause), and others. No such binaries are bundled here. |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md) | LGPL-2.1-or-later base; GPL and version-3 configuration options alter a particular binary's license. External program installed separately. Validation host uses a GPL/version3 build. |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp/blob/master/LICENSE) | Unlicense for source; bundled upstream executables can include differently licensed dependencies. Optional Python package here. |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) | MIT; CTranslate2 MIT, PyAV BSD, FFmpeg-linked binaries retain their own terms. Optional. |
| [Whisper model](https://github.com/openai/whisper/blob/main/LICENSE) / [converted tiny](https://huggingface.co/Systran/faster-whisper-tiny) | MIT; downloads come from the named conversion repository. No weights are bundled in DotEARS. |
| [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE) | Apache-2.0; optional runtime, retain third-party notices when redistributing it. |
| [YAMNet code](https://github.com/tensorflow/models/blob/master/LICENSE) / [model](https://www.kaggle.com/models/google/yamnet/tensorFlow2/yamnet/1) | Apache-2.0 for official code/model distribution. Model URL, file hashes and measured bytes are recorded by explicit preparation. |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE) | MIT; optional local stdio adapter. |
| [CLAP checkpoint](https://huggingface.co/laion/clap-htsat-unfused) | Apache-2.0 model card; optional explicit local download, full commit/weight hashes retained. Upstream LAION repository code is CC0-1.0 but is not vendored or used as a runtime dependency here. |
| [CED-Mini checkpoint and HF implementation](https://huggingface.co/mispeech/ced-mini) | Apache-2.0, copyright Xiaomi and Hugging Face. Optional explicit local download pinned to revision `26c3ebcae85d4330f4fc26763f029539a3afcda0`; safetensors and hash-inspected implementation stay in gitignored models. Its GPL training repository is not imported or bundled. The quantized ONNX export was inspected diagnostically and not selected for the presence pipeline. |
| [TorchAudio](https://github.com/pytorch/audio/blob/main/LICENSE) | Optional CED feature-extraction runtime; installed separately with matching CPU PyTorch. Preserve its and binary dependencies' upstream notices if redistributed. |
| [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE) / [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Apache-2.0 / BSD-style; optional SFX runtime. Installed binary dependencies retain their notices; CPU wheels are separate downloads. |
| [pytest](https://github.com/pytest-dev/pytest/blob/main/LICENSE), [Ruff](https://github.com/astral-sh/ruff/blob/main/LICENSE) | MIT; development tools. psutil is BSD-3-Clause, jsonschema and build are MIT. |

The dependency lock captures the exact validation environment, not a claim that all optional
dependencies are necessary. No model training data is distributed. Model outputs and media clips
do not acquire reuse rights from the software license. Demucs, CoNeTTE and WavCaps remain
research references rather than installed project dependencies; see `docs/research.md`.
The 2026-10-27 diagnostic downloaded Kenney Interface Sounds under its included CC0 license
into gitignored validation output. Its audio is not part of the source distribution.
The 2026-10-28 reference experiments additionally use CC0 assets by
[qubodup](https://opengameart.org/content/swish-bamboo-stick-weapon-swhoshes),
[StarNinjas](https://opengameart.org/content/20-sword-sound-effects-attacks-and-clashes),
[cinameng / James Gargette](https://opengameart.org/content/can-be-so-beautiful) and
[drakzlin](https://opengameart.org/content/music-loops-0). Source pages, hashes and receipts are
retained in gitignored output; no audio is bundled in the distribution. Speech backgrounds
were synthesized locally using installed Windows voices. The temporary asset-extraction
environment (py7zr and its dependencies) is not a DotEARS runtime dependency or distribution.

The optional [AudioSep](https://github.com/Audio-AGI/AudioSep) adapter uses the inspected
MIT-licensed source/checkpoint release, downloaded explicitly into gitignored `models/`.
Its upstream license remains there; source and weights are not bundled in DotEARS packages.
[TorchLibrosa](https://github.com/qiuqiangkong/torchlibrosa/blob/master/LICENSE.md) is MIT.
RoBERTa tokenizer/configuration assets come from the official FacebookAI repository;
learned text weights are loaded from the paired AudioSep checkpoint. Existing Transformers
and PyTorch notices apply. Checkpoint/source/tokenizer revisions and hashes are retained.

The raw-presence study additionally retains source pages, CC0 dedications, public-preview
URLs and content hashes for SFXMint, StumpyStrust, LokiF, Fupicat, deadrobotmusic, Ranner,
crackles04 and JW_Audio. New author-caption labels are weaker evidence than the user's
listening annotations and remain marked separately. No assets or trained source catalogs
are included in DotEARS packages. Two speech controls come from public TorchAudio tutorial
assets: VOiCES and LibriSpeech (CC BY 4.0); their source documentation/receipts remain with
the local research artifacts. No private media was sent to these sources.

The 2026-10-30 mixture follow-up uses labels and metadata from
[FSD50K (Fonseca et al.)](https://zenodo.org/records/4060432), with the published metadata
archive checksums retained. Its human labels describe source-level acoustic events;
notification/UI roles additionally use explicitly recorded author descriptions. Selected
public Freesound HQ previews have individually checked CC0 or CC-BY licenses. Current and
archived license URLs, uploader names, titles, source pages, file hashes and retrieval receipts
are retained in `validation-output/sfx-mixtures-2026-10-30/`; no audio or metadata archive is
included in DotEARS packages. Synthetic mixtures retain their component provenance and
do not gain reuse rights from DotEARS's software license.

The 2026-11-04 sound-family expansion reuses the same published FSD50K annotations and
checks CC0 or CC BY status for each additional public Freesound preview. Source pages,
uploader names, licenses and hashes are recorded with the local research data. Preview
audio and synthesized test mixtures remain outside the source distribution.

The same follow-up explicitly downloaded the optional
[FUSS dry baseline model](https://zenodo.org/records/4012661).
The upstream model record specifies CC BY 4.0;
its inference example is Apache-2.0. Record metadata, attribution, archive checksum and
three inference-file hashes remain with the local pilot. The small adapter uses TensorFlow
and does not vendor upstream inference code. FUSS failed full-effect retention in the pilot
and was not selected for the deployed experimental family profile. Weights, generated streams
and any upstream files remain outside the source distribution.

