# CLAUDE.md

Desktop app that runs local speaker diarization (pyannote 3.1) on an MP3/WAV and exports clean single-speaker audio (one file per speaker, or per-clip files) for voice-model training. Target: macOS Apple Silicon (MPS); should work on Windows/Linux (CPU/CUDA). UI is a dark neumorphic ("soft UI") CustomTkinter window.

## Layout

```
Speaker Diarization/
├── CLAUDE.md          # this file
├── main.py            # entry point + App window (UI only)
├── core.py            # processing core: settings, token, pyannote, slicing, export (NO Tk)
├── neu.py             # neumorphic widgets (Pillow-rendered shadows)
├── check_layout.py    # dev tool: geometry check of the live window (run after any UI change)
├── make_app.py        # builds + installs the macOS app (~/Applications/Speaker Separator.app)
├── requirements.txt
├── .env.example       # template; user copies to .env and adds HF_TOKEN
└── .gitignore         # ignores .env, .venv, caches
```

Runtime-only files (not in repo): `.env` next to `main.py` (holds `HF_TOKEN`), and `~/.speaker_separator/config.json` (last output / audio dir only; chmod 600).

## Commands

Dev environment on this machine: conda env `speaker-separator` (Python 3.12, requirements installed; the system Pythons are 3.13/3.14, too new for pyannote 3.x).

```bash
conda activate speaker-separator
python main.py
# fresh machine instead:
#   python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
#   brew install ffmpeg        # system dep, required for MP3
python -m py_compile main.py core.py neu.py   # quick syntax check (no model/token needed)
python check_layout.py                        # must print '0 problem(s)' for every state
```

## Architecture

- **core.py**: UI-agnostic. `separate(audio, out_dir, token, emit)` reports through `emit(kind, payload)`; kinds are documented in its docstring (`log`, `status`, `progress`, `stage`, `device`, `result`). Importing it sets the torch-related env vars, so `main.py` imports `core` before anything else.
- **main.py / `App(ctk.CTk)`**: UI only. Starts a daemon `threading.Thread` per job.
- **Thread rule (critical):** the worker thread must NEVER touch Tk widgets. It puts `(kind, payload)` tuples on `App.events`; `App._tick` (every 40 ms, UI thread) drains them via `_handle`, eases the progress bar and animates the highlight. Any new feedback from the worker goes through `emit` plus a branch in `_handle`.
- `torch` / `pyannote` / `pydub` are imported lazily inside the worker so the window opens instantly.
- The pipeline is cached in `core` module globals so repeat runs skip the load.

## Design system (neu.py)

- Tk has no shadows: every surface is a Pillow image. Raised = dark shadow bottom-right + light shadow top-left; inset = the same shadows flipped inward. Body colour == window `BG`, so content widgets with transparent backgrounds blend in.
- Palette constants at the top of `neu.py` (`BG`, `LIGHT`, `DARK`, `ACCENT`, ...). Change colours there only.
- Pure neumorphism has weak contrast, so state is carried by the accent colour (primary button, active step, progress fill), text uses high-contrast colours, and every button has normal/hover/pressed/disabled looks. Keep that when adding widgets.
- Widgets: `NeuPanel` (card or `well=True`), `NeuButton` (`primary=True` for the accent gradient), `ProgressBar`, `Stepper`. A panel's content widgets must be created AFTER the panel so they stack above its background image.
- `Segmented` (pill selector) cuts its picture into one slice per option and uses a separate label per slice so Tk draws text over the image; a separate text label would paint an opaque rectangle over it. Same reason `NeuButton` uses a single label with image + text.
- Fixed-size window (900x760) with pixel layout: surfaces are fixed-size images. If resizing is ever wanted, the images must be regenerated on resize.
- CustomTkinter gotchas hit already: `width`/`height` go in the widget constructor, never in `.place()`; do not name instance attributes `_w` (Tk's widget path) or other names Tk/CTk use.

## Purpose and design rule

The user trains an AI voice from the output. They need **clean single-voice samples at original quality**; they explicitly do NOT care how much audio is thrown away. So: purity over coverage, and never resample/denoise/"enhance". When tuning, any trade-off between leaving other-voice audio in and dropping more audio resolves toward dropping.

## Processing flow (core.separate)

1. Check `ffmpeg` is on PATH. Load `pyannote/speaker-diarization-3.1` (cached in `core._cache`) on `mps`/`cuda`/`cpu`.
2. Decode with pydub to 16-bit numpy `(n, channels)`; downmix to mono 16 kHz in memory for pyannote (avoids torchaudio file decoding). Optional `num_speakers` hint. One retry on CPU if a GPU op fails.
3. `clean_timelines()`: per speaker keep only stretches at least `guard` seconds away from ANY other speaker's speech (covers true overlaps, detector boundary error, reverb tails), then drop clips shorter than `min_clip`. Pauses < 0.3 s inside one speaker's speech are bridged first.
4. `identity_filter()`: embed each clip with the pipeline's own embedding model and drop clips closer to another speaker's centroid than to their own (leave-one-out centroid), or below an absolute similarity floor. Catches diarizer mislabels. Failure here is logged and skipped, never fatal.
5. `build_tracks()` (pure numpy): 10 ms fades per clip; layouts: `joined` (one file per speaker, 0.25 s silence between clips so splitters can find them), `clips` (a folder of numbered clips per speaker). Names `Speaker_1..N` by first remaining clip. Export MP3 (192k) or WAV.

Purity presets (`core.PURITY`): `strict` = 0.5 s guard, 2 s min clip (default); `balanced` = 0.2 s guard, 1 s min clip. History: earlier versions had natural/clean/AI modes; natural kept bleed, AI (PixIT `pyannote/speech-separation-ami-1.0`, needs a second gated model `pyannote/separation-ami-1.0`) was dropped because separated sources are 16 kHz with artifacts, which is wrong for voice training.

## Measured results (synthetic, `say` voices with known stems; SIR = target vs other voices)

- 2 speakers, ~8 s true overlap: old behaviour SIR 14.5 / 7.1 dB; balanced and strict ~58 dB (other-voice gain 0.00); strict keeps ~12 s of ~19 s per speaker.
- 3 speakers, interjections ("mm-hmm") inside others' sentences, -34 dB noise: old behaviour left one track with ~50% other-voice amplitude (SIR 4.1 dB); balanced and strict ~50-57 dB (gain 0.001-0.002); strict keeps 64% of all speech.
- Identity filter: a clip deliberately moved to the wrong speaker was removed; all legitimate clips kept.
- Untested: real human recordings with room reverb, 4+ speakers, hour-long files (memory use is whole-file numpy), speakers with very similar voices.
- Evaluation recipe: generate stems with `say`, place with deliberate overlaps, sum + noise, run `core.run_diarization` -> `clean_timelines` -> `identity_filter`, then paste each speaker's kept clips onto a silent full-length array (the removed `timeline` layout did this; write a 5-line helper in the test) and regress it on the true stems (`numpy.linalg.lstsq`) for SIR.

## UI wording and layout rules (learned the hard way)

- User-facing words are plain: STRICTNESS (Very strict / Normal), NUMBER OF SPEAKERS (Auto-detect / 2 / 3 / 4), SAVE EACH SPEAKER AS (One file / Many clips), FILE TYPE (MP3 / WAV). Every option has a one-line explanation under it. Internal names (`purity`, `joined`/`clips`, `fmt`) stay in code/config only. Don't reintroduce jargon like "purity", "layout", "timeline".
- **`CTkLabel` defaults to `height=28` whatever the font size.** Always pass `height=` explicitly, or labels silently overlap what is below them. This caused the first overlap bug.
- Every element owns its own rectangle: cards are caption row (y 24-40), well + button side by side (y 42-98); options are a 2x2 grid of caption(16) / selector(44) / hint(16). `python check_layout.py` verifies sibling overlaps, children outside parents and clipped text in idle, long-names, busy, done and error states.
- Disabled/busy state must stay readable: selected option = white text on a dimmed pill (not grey on bright blue). Text/background pairs are all >= 4.5:1 (checked numerically); keep it that way when changing the palette.
- Console text selection uses accent bg + dark text (`selectbackground`/`selectforeground`); the Tk default can hide the text.
- Start button shows "Working..." while busy.
- The user's real screenshot is the ground truth for visuals; the agent cannot screen-capture, so ask for a screenshot after visual changes.

## Packaging (macOS app)

`python make_app.py` (run with the conda env's python, e.g. `/opt/anaconda3/envs/speaker-separator/bin/python`) builds `~/Applications/Speaker Separator.app`. **Re-run it after every code change**: the bundle holds a COPY of main.py/core.py/neu.py, not a link to the project.

- Deliberately NOT a PyInstaller/py2app freeze (torch makes that ~3 GB and fragile). The bundle is source + icon + launcher, running on the Python env it was built with.
- `Contents/MacOS/Speaker Separator` is a **symlink to that env's python**, and `Contents/MacOS/launcher` (the CFBundleExecutable) `exec`s it. Because the running executable lives inside the bundle, macOS shows the app's own name/icon (not "Python"). Verified with `lsappinfo` and a CGWindowList query (owner name "Speaker Separator", window 900x788).
- The launcher logs to `~/Library/Logs/SpeakerSeparator.log` (overwritten each launch) since Finder launches have no terminal. If the env is moved/deleted the launcher shows a dialog telling the user to re-run make_app.py.
- Token for the packaged app: `~/.speaker_separator/.env` (make_app.py copies the project's `.env` there once, mode 600, never overwriting, never printing). `core.load_token` order: `HF_TOKEN` env var -> `.env` beside the code -> `~/.speaker_separator/.env`.
- Info.plist sets `NSAppSleepDisabled` (no App Nap throttling of long jobs); `main.py` also runs `caffeinate -i -w <pid>` only while a job is running so the Mac doesn't idle-sleep mid-file.
- First launch: macOS asks permission when the app first writes to Desktop/Documents/Downloads (the output folder). Normal; the user clicks Allow. Locally built apps carry no quarantine flag, so Gatekeeper doesn't block them; they are unsigned.
- Verified headless: the bundled python with a Finder-like env (`env -i`, no conda/Homebrew in PATH) ran the full pipeline (MPS, ffmpeg found via core's PATH patch, token from the per-user .env).
- Icon is generated by `make_icon_png()` in make_app.py (dark squircle + two-tone waveform in the UI accent colours). No design files to maintain.

## Token handling

- The UI has NO token field. `core.load_token()` reads `HF_TOKEN` from the environment, else `.env` in `core.APP_DIR` (next to the code), else `~/.speaker_separator/.env` (packaged app). A pill in the header shows "Token ready" / "Token missing".
- Never hardcode the token in source. Never print it, never put it in `config.json`, never ask the user to paste it into chat, never read `.env` contents.
- If the app is later bundled (PyInstaller/py2app), ship `.env` beside the app, not inside it. Anything embedded in a bundle can be extracted.

## Gotchas / why the pins exist

- `PYTORCH_ENABLE_MPS_FALLBACK=1` and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` are set at the top of `core.py` **before** torch is imported. Don't move them below imports.
- pyannote.audio must stay `<4` (4.x uses torchcodec and a different auth arg). torch/torchaudio `<2.9`, `huggingface_hub <1.0`, `numpy <2` keep pyannote 3.x importable. If bumping any, test end-to-end.
- Prefer Python 3.12 for the venv: the machine's default is 3.13, whose wheel coverage for pyannote's dependency tree is less certain.
- The HF token alone is not enough: the user must accept terms on the Hub for BOTH `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`. A load failure is almost always this.
- Overlapping speech: segments from different speakers can overlap, so the outputs can total more than the input. Expected.
- macOS system Python has a broken Tk 8.5; use python.org Python or `brew install python-tk@3.12`.
- GUI apps started from Finder don't see Homebrew in PATH; `core.py` prepends `/opt/homebrew/bin` and `/usr/local/bin` on macOS.

## Verification notes

- `screencapture` is blocked in the agent sandbox, so the live window can't be screenshotted. Visuals were checked by rendering `neu.py` images to PNG; behaviour was checked by driving `App` with a stubbed `core.separate` (success + failure runs) in a scratch venv with only `customtkinter` + `Pillow`.
- End-to-end verified once (2026-11-12) in the conda env: a 27 s clip made with macOS `say` (Samantha + Daniel alternating) -> pipeline loaded on `mps` in ~13 s (weights now cached), diarization ~6 s, 2 speakers detected, two MP3s of ~13 s each. Token and gated-model access confirmed working. Recreate with `say -v Samantha/Daniel -o x.aiff "..."`, ffmpeg concat, then call `core.separate` directly.
- A real two-person recording (overlap, noise, 3+ speakers) is still untested.

## Conventions

- Never log or echo the HF token. Never commit `.env` or `config.json`.
- Worker errors are caught in `_worker_main`, logged with traceback, and surfaced via `error` events. The UI must always be re-enabled on `done`/`error`.
