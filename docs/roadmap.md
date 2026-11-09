# Next steps toward autonomous reference research

Latest completed step, 2026-10-30: [139-source mixture study](sfx-mixtures.md), 88 authors,
834 mixtures. A new optional whoosh + broad signal family profile now proposes 5 whoosh and
5 UI-signal windows on all 114 Apple Stocks attacks. The explicit 7.343s whoosh and parent
signal at the noted 1.537s notification are recovered; specific notification roles stay unknown.
The 19.496s proposal differs from the user's UI note. Whole-track regression also exposes
three premature UI assignments, so labels describe focus-window presence, not exact onset.
Next improve localization and evaluate an untouched track/source holdout; no further small
annotation worksheet is needed now. The older results below remain historical evidence.

UI scope was clarified on 2026-10-30: clicks/taps, beeps and confirmation tones belong to
`ui_sound`. The reference policy is versioned accordingly; notification/other reviews and
all rejected patterns remain preserved. Future general UI training/evaluation must use this
broader membership. [Definitions](sfx-types.md) distinguish a scope change from new accuracy.

2026-10-29/30 update: the varied-source raw CED/CLAP study is complete (72 sources,
23 author groups, 693 windows, 24 candidates). All general heads failed fresh mixed-source
coverage and remain disabled. `sfx-match` now supports a narrower bank of six whooshes,
five notifications and three clicks from the existing human review. A full generated track
exposed one false pattern and disabled it; the repaired integration check recovers 11/15
known sources. Apple Stocks still yields 60 unknowns with both active modes. See
[current evidence](sfx-presence.md). Next generalization work requires varied, genuinely
reviewed editing mixtures and a fresh held-out set; preserve Apple as a regression reference.
No additional listening worksheet is required by the current implementation work.

1. Improve semantic SFX quality and attack selection using manual annotations of the two real
   references. The local-file workflow, permission fix and first agent-directed Wistia URL cycle
   are verified. Repeating those same technical acceptance tests is not the next priority.
2. Extend the first listening sample before changing models. On 2026-10-27 the user rated
   12 Apple Stocks excerpts and added five approximate marks: six useful, four not useful,
   two uncertain. TransformAr is still unrated and full recall cannot be estimated.
   Version 0.1.3 exposes candidate truncation (114 detected / 60 returned / 54 omitted on
   Apple Stocks); the listening starter selection avoids nearly identical excerpt windows.
   Do not suppress lower-strength or speech-overlapping candidates solely from this sample:
   several useful effects would be lost. See `docs/validation.md` for evidence.
3. Build on the completed 30-example SFX labels received 2026-10-28. The frozen CLAP-v1
   baseline matched 8/8 whoosh and 5/5 UI-click primary labels; all five notifications ranked
   the broader UI tone first. Three of four other controls also ranked UI tone first and one
   ranked UI click, so a broad tone-to-notification rename is unsupported. Compare an
   audio-reference method using the 18 high/medium target references and four other controls.
   This comparison and two controlled-mixture experiments are now completed: clean notification
   ranking improved, but background false names or lost short clicks prevent automatic labels.
   Preserve the frozen positive/negative corpus when evaluating event-focused representations
   or foreground extraction. The eight-sound follow-up is now reviewed: four whooshes and four
   blade-like other sounds; both methods incorrectly ranked all four blade-like controls as
   whoosh. That narrow weapon/bamboo selection did not cover the intended editing/UI domain.
   Preserve it as a regression check. Future review candidates must cover acoustically varied
   editing transitions, notifications and UI clicks rather than near variants from weapon packs.
   On 2026-10-29 four short-event/mixed-training heads were tested using frozen YAMNet/CLAP
   features, including a prefix spectral mask. All failed the preregistered research gate:
   clean-name gains did not survive mixed tracks and negative controls. Do not enable those
   candidates or keep retuning them against this same small corpus. Retain the reproducible
   benchmark for foreground-separation evaluation. AudioSep has now completed 295 primary and
   236 wording-robustness cases, with useful conditional notification/click extraction but
   failed overall preservation/background checks. The explicit `separate` CLI is available;
   normal analysis still leaves named SFX unconfirmed. The fixed three-query presence pass
   is now implemented and tested: 798 calibration estimates and all 60 Apple Stocks attacks.
   Only whoosh passed the local alternate-background gate; its sole real-video name differed
   from the user's notification/bell note, while the noted whoosh stayed unknown. Promotion
   is withheld: the active policy keeps all three types unknown, preserving the experimental
   answers and estimates. Thresholds/weights were not refit to the video. Next build a varied,
   independent editing-SFX mixture set with separate source families; retain Apple as a
   regression reference instead of fitting to its notes. See `docs/sfx-auto.md`.
   No additional annotation worksheet is currently requested. See `docs/separation.md`.
4. Test an explicitly assigned recurring Dot research workflow with per-run source/count/resource
   limits and notifications only on useful findings, failure or required user action.
5. Improve model reuse, crash recovery, library annotations and cross-platform tests. Consider
   SQLite only when local JSON management becomes a real constraint.
6. Evaluate source separation and richer visual measurements on real failures. Keep them optional.

