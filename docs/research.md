# Component research and decisions

Reviewed 2026-10-24. References below are the original projects or official documentation.
No third-party project was copied wholesale and no third-party implementation code is vendored.

| Component | Useful existing capability | Decision for 0.1 |
| --- | --- | --- |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg) | Stream probing, decode/resample, packet remux, scene score | Required external program. CPU execution, no bundled binary. |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | URL extractors, duration/size filters, downloads | Optional worker with time/disk budget; no auth or DRM bypass. Platform success is not guaranteed. |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Local ASR, CPU support and quantized models | Good future lightweight executable backend; not implemented alongside a duplicate ASR backend now. |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 local ASR, CPU int8, timed segments | Optional implementation, CPU/int8 forced, local model directory required. Multilingual tiny/base/small preparation. |
| [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) | 521 broad AudioSet classes; 16 kHz mono; overlapping patches | Optional local SavedModel backend. Keep original vocabulary, uncalibrated scores and coarse temporal bounds. TensorFlow is too large for the minimal install. |
| [CLAP](https://github.com/LAION-AI/CLAP) | Audio/text embeddings for candidate-description similarity | Experimental CPU pass tested 2026-10-27. All 12 mixed excerpts ranked music/speech above effect descriptions; clean controls showed some plausible UI matches and padding sensitivity. Keep hypotheses and unknown labels; see `sfx.md`. |
| [Librosa](https://github.com/librosa/librosa) | Existing onset strength/detection, RMS and beat tracking | Required DSP implementation. The project does not invent a new onset detector or label DSP attacks as named SFX. |
| [Demucs](https://github.com/facebookresearch/demucs) | Separates music stems including vocals, bass and drums | Deferred: CPU/memory cost; general stems do not isolate arbitrary overlapping editing effects. Upstream notes limited maintenance. |
| [CoNeTTE](https://github.com/Labbeti/conette-audio-captioning) | Local audio captioning research implementation | Deferred: larger stack, caption hallucination risk, separate checkpoint/dataset license review required. No captioner claim in 0.1. |
| [WavCaps](https://github.com/XinhaoMei/WavCaps) | Audio-language dataset and downstream research | Research reference only. Source audio licenses differ; not a redistributable pool of cleared SFX. |
| [mcp-music-analysis](https://github.com/hugohow/mcp-music-analysis) | Demonstrates Librosa/Whisper/URL analysis exposed to agents | Reviewed as an integration example; no code imported. Its music focus does not validate motion SFX recognition. |
| [mcp-audio-server](https://github.com/git-scarrow/mcp-audio-server) | Chord, key and tempo service; repository points to a moved project | Research example, not a dependency. DotEARS uses the official MCP SDK with stdio and explicit path boundaries. |

The result is an orchestration and evidence toolkit. FFmpeg already provides extraction and
scene observations; Librosa supplies DSP; optional local models supply semantic observations.
Dedicated whoosh/pop/riser/glitch classification remains a dataset and evaluation problem.
Cutting a time interval from a video creates a mixed excerpt, not an extracted isolated SFX.

## Why this architecture

Start with a reproducible audio pipeline that runs without weights. Optional local backends
are imported only when requested. Model setup is explicit and separate from analysis. This
keeps offline local-media behavior predictable and lets modest CPU machines skip heavy models.
The agent receives measured observations and provenance, then performs the editing interpretation.

## License selection

DotEARS source is MIT. Required Python libraries use permissive licenses. Optional model code,
weights and FFmpeg remain separate distributions under their upstream licenses. The local FFmpeg
build used for validation enables GPL and version 3; DotEARS invokes it as an external process.
We do not bundle it or relicense it. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Event-focused follow-up, 2026-10-29

The official [YAMNet transfer-learning tutorial](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
documents 1024-D event embeddings. Local experiments used those features for small linear
heads, compared prefix spectral masking and then CLAP/concatenated features. All failed the
research gate for mixed-track semantics; full evidence is in `docs/sfx.md`. These are tested
local failures rather than untested promises of improved quality.

[AudioSep](https://github.com/Audio-AGI/AudioSep) was inspected as a separate, text-conditioned
foreground-separation candidate. Its official inference code supports CPU and operates at
32 kHz. The linked [official checkpoint directory](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint)
lists a 1.26 GB separator checkpoint and a separate 2.35 GB CLAP checkpoint (about 3.62 GB
as displayed by the publisher). Its encoder expects a different HTSAT-base checkpoint, so
the existing unfused model is not interchangeable. The subsequent executed pilot found that
the separator checkpoint already contains its paired text weights. Strict loading and a
bitwise companion-file comparison allowed retaining only that checkpoint for inference.
The optional [local AudioSep adapter](separation.md) is now installed/tested on CPU. It improved
some notification/click mixtures but failed the joint preservation/background criteria;
whoosh results were strongly query-dependent. The explicit tool remains experimental.

