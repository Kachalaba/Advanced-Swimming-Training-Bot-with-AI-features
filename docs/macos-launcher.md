# 🖥️ Sprint AI — desktop launcher (macOS)

A one-time installer that puts a double-clickable **Sprint AI** icon on your
Desktop. Double-clicking it starts the local Streamlit app and opens it in your
browser at <http://localhost:8501> — no Terminal commands needed afterwards.

> This wraps the existing local app; it does **not** bundle Python or the ML
> dependencies. Set those up once (see [`SETUP.md`](../SETUP.md)), ideally in a
> virtualenv, before installing the launcher.

## Install

From anywhere inside the repo:

```bash
bash scripts/install-macos-launcher.sh
```

This creates `~/Desktop/Sprint AI.app`. To install into Applications instead:

```bash
bash scripts/install-macos-launcher.sh /Applications
```

The repo path is baked into the shortcut, so you can move or rename the `.app`
freely — just re-run the installer if you later move the **repo** itself.

## First launch

Because the app is built locally (not downloaded), macOS Gatekeeper is usually
happy. If you see *"cannot be opened"* on the very first run:

- **Right-click** the icon → **Open** → **Open** (only needed once), or
- System Settings → Privacy & Security → **Open Anyway**.

A Terminal window opens alongside the app — it shows logs and lets you **stop**
the server with `Ctrl+C` or by closing the window.

## How it works

| File | Role |
|------|------|
| `scripts/sprint-ai-run.sh` | The actual runner: picks a venv/Python, ensures Streamlit, opens the browser, runs `streamlit run app.py`. Works on Linux too. |
| `scripts/install-macos-launcher.sh` | Builds the `.app` bundle, bakes in the repo path, and converts `assets/sprint-ai-icon.png` → `.icns` with the built-in `sips` + `iconutil`. |
| `assets/sprint-ai-icon.png` | Source icon (1024×1024). Replace it and re-run the installer to use your own artwork. |

## Custom icon

Drop your own 1024×1024 PNG at `assets/sprint-ai-icon.png` and re-run the
installer. If the new icon doesn't refresh in Finder, restart Finder:

```bash
killall Finder
```

## Run without the icon

The launcher is optional — the underlying command still works directly:

```bash
bash scripts/sprint-ai-run.sh        # or: python3 -m streamlit run app.py
```

## Note on a "real" packaged app

This is the lightweight, personal-use launcher. Turning Sprint AI into a
distributable, signed `.dmg` (Tauri/Electron wrapper + bundled Python sidecar,
code signing, notarization) is a separate, larger effort — see the discussion in
the project notes before going down that path.
