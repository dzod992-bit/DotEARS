# Dot integration feasibility

Initially assessed **2026-10-24** against official OpenAI documentation. On **2026-10-25**
the user relayed Dodik's successful local-task result, and the referenced artifacts were checked
independently. This verifies one assigned local-file workflow. Dodik's originating tool trace,
cloud computer and saved schedules were not inspected directly by this Codex chat.

## Ten requested questions

| Question | Finding | Evidence / verification still needed |
| --- | --- | --- |
| Python in Dot's cloud computer? | Software work and running software are documented. A guaranteed Python executable/version is not established. | Inspect that computer and run `python --version` before selecting variant A. |
| Install FFmpeg and models there? | Not established by the reviewed Dot documentation. | Verify package installation permissions, network, executable policy and available storage there. |
| Download, executable, RAM, filesystem limits? | No concrete Dot quotas were found in these pages. | Measure on the actual Dot host. Do not reuse Codex Cloud quotas as Dot quotas. Platform access may fail independently. |
| User skills? | Dot supports enabled plugins; local skills require a connected computer. | Install/enable the intended skill through the supported user flow and check it is visible. |
| Third-party MCP? | Plugins can supply tools. Arbitrary local stdio access directly in Dot's cloud is not established. | Public desktop-only local-MCP submissions currently require approved-partner eligibility; a Codex MCP configuration alone does not configure Dot. |
| Through Codex? | A local-task invocation/report-return workflow was exercised by Dodik, per the user's relayed response and matching saved artifacts. | One local-file analysis and repeat succeeded; other task types remain untested. |
| Windows computer? | A personal computer connection can expose local files, tools and apps through tasks. | User must allow access in Dot's profile; keep that computer online and the desktop app open. |
| Recurring tasks? | Assigned recurring work can perform authorized actions; local work still needs the computer. | Save and test an actual recurring task after the manual integration succeeds. No schedule was created here. |
| Proactive vs assigned? | Proactive research is read-only and cannot control browser/computer. Assigned recurring work has separate permissions. | Local audio processing belongs in expressly assigned research work. |
| Autonomous per-video processing? | Dodik exercised an assigned local-file analysis and cached repeat. | Autonomous discovery/URL acquisition and recurring research remain untested. |

Sources: [computers and apps](https://learn.chatgpt.com/docs/dots/computers-and-apps),
[tasks and memory](https://learn.chatgpt.com/docs/dots/tasks-and-memory),
[desktop-only plugin submission](https://developers.openai.com/plugins/deploy/desktop-only),
[scheduled tasks](https://learn.chatgpt.com/docs/automations).

## A — Run in Dot's cloud computer (desired, conditional)

First ask Dot to inspect its own permitted runtime, without changing security settings:

```text
Check whether your cloud computer can run Python 3.11–3.13 and FFmpeg/ffprobe.
Report installation permissions, available RAM/disk and network access. Do not upload my
media to other services. Do not claim this check passed without executing it.
```

If the runtime is suitable, transfer the source through a supported file/repository workflow,
install in an isolated environment, run the synthetic fixture, then run one authorized video.
Prepared models must fit the actual host. No hypothetical Dot-specific API is used.
Do not install or transfer private material until that environment and permission are known.

## B — Connected Windows computer (primary practical candidate)

1. Install DotEARS locally and run `dotears doctor` and the demo.
2. In Dot's profile, connect this computer using the documented **Computers → Your computer →
   Allow access** flow. This is a user action; this repository does not grant access.
3. Give Dot an assigned task to create a local Work/Codex task on that computer with the
   absolute source/media/library paths and the command in [agents.md](agents.md).
4. Have that task return the report JSON, warnings and file location. Dot should combine
   those observations with its own visual review. It must preserve uncertain labels.
5. Test a second video in the same assignment, then a repeated local file to verify deduplication.
6. Only after that, save a recurring research task with a time zone, allowed sources, video count,
   local resource limits and a clear notification policy. Inspect its first runs.

The computer must remain available. Expired service logins or platform blocks can require the
user. DotEARS never reads browser cookies or signs in. A local file is the fallback for blocked URLs.
The user confirmed the computer connection and relayed the test assignment/result. This
repository changed no account setting and created no schedule, public server or GitHub publication.

## Executed local-task evidence — 2026-10-25

The user relayed Dodik's response: a fresh analysis completed in 39.162 s, the repeat in
1.474 s; both returned 78 events, 60 clips and the same reference ID. Codex independently
read `agent-check.json` and `report.json`, validated the schema, verified the reference ID,
source hash, cache flags and presence of all 60 clips. Dodik's reply also interpreted the
20.90 s visual/audio coincidence while keeping the sound label unknown.

This is evidence for **level 4 in the assigned local-file scenario**. Caller identity relies
on the human-relayed Dot response; the helper does not attest it. It does not establish
autonomous URL discovery/acquisition, independent listening quality or recurring execution.

The 0.1.0 attempt initially hit `WinError 5` in a temporary folder and succeeded after a
permission retry. The cause was reproduced in this Windows restricted runner. Version 0.1.1
uses inherited parent-directory access for temporary workspaces, preserving private POSIX
behavior. Legacy model files were recreated byte-for-byte in accessible project directories;
all file hashes matched. Existing ACLs and system settings were not altered.

A fresh 0.1.1 local analysis then passed **without escalation** in 24.072 s, with a cached
repeat in 1.459 s. Dodik's confirming repeat subsequently passed under normal host permissions:
26.479 s for fresh analysis and 1.753 s for the cached call, 78 events and 60 clips each.
The new report's 0.1.1 version, schema, source hash, cache flags and clip presence were checked
independently without escalation. The user relayed that no access failures or permission changes
occurred. This closes the permission-fix repeat for the tested Dot/Windows local-file workflow.

The confirming artifacts are under
`validation-output/dodik-live/run-20261025T152346Z-93ffc8/`. They remain private and gitignored.
No autonomous-discovery or recurring-execution completion is implied by this local-file result.

## URL acquisition boundary — 2026-10-26

Dodik found two video references, but both downloads stopped at a process-only proxy pointing
to `127.0.0.1:9`. A read-only restricted/approved comparison identified a host runtime network
restriction, distinct from the user's persistent Windows proxy configuration. Approved control
requests reached Vimeo but produced separate HTTP 401 and yt-dlp login-client errors. No video,
report, transcript or clips were generated by those URL attempts. Level 5 remains unachieved.

Version 0.1.2 provides redacted proxy diagnostics and a distinct `proxy_connection_failed`
error, preserving network settings. A subsequent permitted Wistia URL cycle succeeded as
recorded below. See [network access](network-access.md) for both failures and the success.

## First completed discovered-URL workflow — 2026-10-26

The user relayed Dodik's successful third-candidate run on the public Wistia source
`https://berre.wistia.com/medias/4jcqq1cnwl`. The retained discovery summary records a new
source search, host-approved networking, failed Vimeo/YouTube candidates and a successful
Wistia acquisition. No proxy, security or code changes were reported in that run.

Codex independently checked the saved MP4 (34.005011 s, 1280×720, 25 fps, stereo AAC), its
SHA-256 against the report, the JSON schema, both model statuses and all 60 sample-exact clips.
The report contains 79 observations. The successful CLI call took 32.820 s, exit 0, cache miss.
A retained image sequence confirms the visual change around 17.480 s; the attack candidate
at 17.410431 s precedes it by about 70 ms. Its sound identity remains unknown.

This verifies one assigned discovery → URL acquisition → local analysis → clips/report →
agent interpretation cycle. Caller identity and discovery/approval history rely on the
human-relayed Dot result plus saved summary; original Dot tool history was not opened here.
A separate review sidecar records the candidate editing relation and a Remotion experiment
without modifying the analyzer report or establishing reuse rights.

Private evidence is under `validation-output/url-live-012-20261026T000502Z-9f6b44/`.
This is the first exercised cycle toward level 5, not proof of unattended recurring research,
standing network permission, universal platform access or reliable named-SFX classification.

## C — Standalone CLI / stdio MCP (implemented)

Python, CLI and MCP use the same core. The MCP server has an explicit media root and library,
and URL processing is opt-in. Codex can launch the local executable as described in
[official MCP guidance](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
Other agents need a compatible local stdio MCP client or a permitted shell.

## Acceptance evidence required for levels 4 and 5

Keep an execution record showing: Dot found a permitted video; Dot initiated the actual local
task; Dot received the generated report; the file library contains its clips; Dot explained an
audio/visual relation with uncertainty; and a second analysis succeeded within the assigned work.
Then test the saved recurrence. A Codex CLI call, a model unit test, or a documentation citation
does not satisfy those Dot-specific acceptance criteria.

