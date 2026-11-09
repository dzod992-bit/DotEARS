# Windows installation

1. Install a 64-bit Python 3.11, 3.12 or 3.13 from an official distribution. Python 3.14 is
   intentionally excluded until the selected ML dependencies have been evaluated there.
2. Install FFmpeg from a distribution you trust. Its `bin` directory must expose both
   `ffmpeg.exe` and `ffprobe.exe` on PATH. Restart the terminal after changing PATH.
   DotEARS does not download or redistribute that binary.
3. Open PowerShell in the repository. Run the commands in the [README](../README.md) or
   `./scripts/install.ps1`.
   An existing environment can also use `uv pip install --python .venv/Scripts/python.exe -e '.[download,mcp,dev]'`.
4. Run `./.venv/Scripts/dotears.exe doctor`. The two executable paths must be non-null.
5. Analyze the included `examples/demo.synthetic.mp4`. Results are CPU based and run without
   GPU acceleration.

The install script does not activate the environment, so PowerShell activation-policy changes
are unnecessary. It leaves system FFmpeg installation to the user. If script execution itself
is blocked by organization policy, use the individual README commands without changing policy.

Optional `.[asr,yamnet]` installs substantially larger packages. TensorFlow is a separate opt-in
and is not required for extraction/DSP. On Windows it runs on CPU in this project. Obtain model
weights with the explicit `prepare-model` commands, then pass their local directories to analysis.
Do not install into an unrelated application's Python environment.

Troubleshooting:

- `missing_dependency`: check PATH and the selected virtual environment.
- Unsupported/deleted/blocked video URL: supply an authorized local file. Login and DRM bypass
  are not provided. Some sites require capabilities the minimal downloader does not enable.
- Download fails behind a proxy: use your network's supported configuration; do not disable TLS.
- `proxy_connection_failed`: run `dotears doctor` to inspect redacted effective proxy endpoints.
  In the tested restricted host, process-only proxies pointed to `127.0.0.1:9`; an approved
  network operation used the existing system proxy instead. Request network access through
  the host's supported approval mechanism. Do not clear proxy variables or change Windows
  proxy settings to work around the boundary. See [network access](network-access.md).
- First Librosa run slower: numerical routines may compile/cache locally. Compare cold and warm runs.
- `WinError 5` on a temporary directory or report created by 0.1.0: upgrade to 0.1.1. New
  workspaces inherit existing parent access instead of an owner-only temporary ACL. Older
  downloaded model copies may need to be recreated from hash-verified contents in an authorized
  folder. This changes no Windows security setting; the host must already permit that folder.
- Reference already locked: confirm another analysis is not running. After a verified process
  crash, remove only that stale reference's lock file, then retry.
- Files in the report point to another machine: return the JSON through the local task and keep
  its media local, or deliberately export the permitted artifacts through an approved file workflow.

