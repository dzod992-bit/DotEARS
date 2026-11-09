# Known limitations

- User-relayed Dodik local-task runs were verified against their saved artifacts on 2026-10-25.
  They covered an assigned local file and cached repeat: 0.1.0 needed a permission retry;
  the confirming 0.1.1 run passed with normal permissions and no retry.
  On 2026-10-26, one agent-directed discovery/URL acquisition/analysis cycle succeeded for
  Wistia with host-approved networking. Cloud installation and unattended recurrence remain unvalidated.
- Version 0.1.1 inherits existing Windows parent-directory access for temporary workspaces.
  A normal restricted local run passed after the fix. The host must still authorize the parent
  folders; this does not grant access to other directories or establish every Dot host's settings.
- One real user MP4 was processed on 2026-10-25. YAMNet returned Speech/Music only; specific
  editing effects remain unidentified. On 2026-10-27 the user reviewed 12 selected Apple Stocks
  excerpts and added five approximate video-position marks. Six excerpts were rated useful;
  this selected sample does not establish real-world SFX precision, recall or onset accuracy.
  TransformAr has no listening ratings yet, and neither source has a confirmed complete review.
- On Apple Stocks, DSP found 114 attacks and the default cap returned 60. From 0.1.3,
  `attack_detection` exposes total/selected/returned counts, omissions by limit and timeline
  filtering. JSON/Markdown warn when the limit truncates candidates. Historical reports lack
  these counters. The cap still selects by onset strength, not semantic editing usefulness.
- Baseline DSP is language independent and detects attacks/energy/rhythm, not SFX semantics.
  Strong musical notes and speech attacks are expected confounders. Smooth risers can be missed.
- The experimental `sfx` CLAP pass returns ranked hypotheses, with confirmed labels kept
  unknown. In its first two-window pilot, all 12 reviewed mixed-track excerpts ranked
  music/music-with-speech first. Clean controls showed some plausible UI matches and strong
  preprocessing sensitivity. This is not a validated whoosh/notification/UI recognizer;
  it is not automatically enabled in normal CLI analysis or MCP. See `docs/sfx.md`.
- A 30-example clean SFX set was fully rated by the user on 2026-10-28. On high/medium
  examples, frozen CLAP-v1 first names matched 8/8 whoosh and 5/5 UI clicks, while all five
  notifications ranked the broader UI tone first. Three of four other controls also ranked
  UI tone first and one ranked UI click. This does not validate automatic target decisions;
  all whoosh positives share one source family. The later audio-reference family evaluation
  matched 4/5 clean notifications and 5/5 clicks but falsely ranked targets on 12/12 background
  windows. A short-window/background-reference follow-up rejected most new background windows
  while losing all 45 known inserted click cases. Neither method supports automatic names.
  The follow-up review labeled four new sounds as whoosh and four blade-like sounds as other;
  both methods falsely ranked those four other sounds as whoosh. The narrow weapon/bamboo
  selection does not represent the intended motion-design/UI domain. Names are not labels.
- On 2026-10-29 four source-balanced event heads were tested with family-excluded training
  on synthetic mixtures. Even concatenated CLAP/YAMNet features, which matched most clean
  target names, failed on mixtures and other/background controls. Prefix spectral masking
  also failed. These heads remain research artifacts; exported decision scores are not
  calibrated probabilities. A 0.7 s crop can omit part of the original human-rated sound,
  and known insertion positions make mixture tests easier than full video event discovery.
- YAMNet is optional, broad and coarse in time. It does not reliably cover all motion-design
  effects. Scores are uncalibrated. Low-score events may be missed and false positives are possible.
- The optional `sfx-auto` sidecar checks all three fixed AudioSep queries and applies a small
  locally fitted presence/abstention policy. Its source-family checks and alternate-background
  validation use 30 convenience sources; alternate backgrounds share the same foreground
  sources. Synthetic audibility is not separately human-rated. Accepted types remain
  provisional, disabled/ambiguous types remain unknown, and scores are not probabilities.
  This does not establish general named-SFX accuracy or continuous-event recall. See
  [automatic SFX check](sfx-auto.md). No extra models are enabled in normal analysis/MCP.
- Tiny Whisper is optional and especially limited on difficult speech/music mixtures. It may
  hallucinate or omit speech. Language probabilities and segment scores are model outputs.
- Tempo estimates have half/double ambiguity; rhythmic SFX and voice can look musical. Beat
  candidates do not establish music presence. There is no learned music segmentation without YAMNet.
- Abrupt frame changes can be flashes or motion, and subtle typography motion may be missed.
  There is no optical flow, semantic visual action recognition, or automatic editing-role analysis.
- Normal WAV clips are native-rate/channel mixed excerpts. The optional explicit AudioSep
  `separate` command saves an unconfirmed query-conditioned component, not a certified isolated
  SFX. It may suppress the desired sound or retain background; the current joint research
  gate failed. Query wording is not proof of event presence. No output grants reuse rights.
  Do not publish third-party excerpts from this repository.
- Defaults: 120 s, 200 MiB input; configurable up to 600 s / 1 GiB. Native decoded audio is capped
  at an estimated 512 MiB. Native clip export currently supports mono/stereo up to 192 kHz.
  Up to 200 attacks and 200 classifier intervals may be reported; exported clips have a shared cap.
- Downloads require a finite declared duration, reject live streams, and use a single muxed
  format to avoid uncontrolled external merge processes. Some platform formats will be unavailable.
  No browser cookies, authentication, DRM bypass, remote JS solving or plugin extensions are enabled.
- Local analysis passing does not imply URL networking is permitted by the host. On the tested
  restricted host, downloads failed through process-only proxy `127.0.0.1:9`; approved network
  operations reached Vimeo but encountered separate HTTP 401/login-client failures. A subsequent
  Wistia URL completed the full pipeline. Vimeo and YouTube compatibility is not established
  by that success. See `docs/network-access.md` for the evidence.
- The download worker is terminated on timeout/disk budget. The disk monitor samples every
  100 ms, so a download may briefly overshoot the configured byte budget before termination.
- Per-operation timeouts cover download and FFmpeg. Python DSP/ML has duration/size/thread bounds
  but no hard whole-job timeout or OS RAM quota. This is not a hostile public upload service.
- No in-memory neural model cache, database, automatic garbage collection, or resume after a
  machine crash. Completed local results deduplicate; URL input must first acquire media again.
- Verification is on this Windows host. Linux/macOS portability is designed but not yet exercised.
- The project never transmits local media for analysis. The external agent may send text reports
  to its own model service as part of its normal operation; that is outside DotEARS processing.

