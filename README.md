# Speaker Separator

A Mac desktop app that splits a recording into **clean, single-speaker audio**, one speaker at a time, for training an AI voice. Everything runs locally; your audio never leaves your computer.

It deliberately favors purity over coverage: anything near another speaker (overlaps, interjections, uncertain edges) is cut, and each remaining clip is checked to sound like its assigned speaker. The audio is never denoised or resampled.

## How it works

1. **Diarize:** [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) finds who speaks when (on Apple's GPU via MPS, falling back to CPU).
2. **Keep only clean stretches:** per speaker, drop everything within 0.5 s of another voice and any clip shorter than 2 s ("Very strict"; "Normal" uses 0.2 s and 1 s).
3. **Voice check:** clips that sound closer to a different speaker are removed (catches mislabels).
4. **Export:** one file per speaker, or a folder of clips per speaker, as MP3 or WAV.

On synthetic test clips with known overlaps, this left the other voice at about -50 dB or lower, where the plain approach left it clearly audible. Real recordings (room echo, 4+ speakers, similar voices) are not yet tested.

## Requirements

- macOS on Apple Silicon (other platforms: the core runs on CPU/CUDA, but the packaging script is Mac-only)
- Python 3.10-3.12 (3.12 recommended; pyannote 3.x does not support 3.13+ cleanly)
- [ffmpeg](https://ffmpeg.org) (`brew install ffmpeg`)
- A free Hugging Face account

## Setup

```bash
conda create -n speaker-separator python=3.12 -y
conda activate speaker-separator
pip install -r requirements.txt
brew install ffmpeg
```

### Hugging Face token (required)

The pyannote models are gated. Once:

1. Accept the terms on both model pages while logged in:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
2. Create a **Read** token at https://huggingface.co/settings/tokens.
3. Put it in a `.env` file next to `main.py`:
   ```bash
   cp .env.example .env     # then edit .env and paste your token
   ```

`.env` is git-ignored. The token is never stored in the source or shown in the UI.

## Run

```bash
python main.py
```

Choose an audio file (MP3/WAV) and an output folder, set the options, and press **Start Separation**.

| Option | Choices |
|---|---|
| Strictness | **Very strict** (default, best for training) or Normal (keeps more audio) |
| Number of speakers | Auto-detect, or 2 / 3 / 4 if you know it (more accurate) |
| Save each speaker as | One file, or Many clips |
| File type | MP3 (small) or WAV (full quality, best for training) |

## Build the Mac app

```bash
python make_app.py        # installs ~/Applications/Speaker Separator.app
```

This builds a small app (a copy of the source, an icon and a launcher) that runs on the Python environment you built it with, not a multi-gigabyte frozen bundle. Re-run the script after changing the code. For the packaged app, the token is read from `~/.speaker_separator/.env` (the script copies your project `.env` there once).

## Development

- `core.py` processing (no GUI), `main.py` window, `neu.py` neumorphic widgets, `make_app.py` packaging.
- `python check_layout.py` checks the live window for overlapping or clipped widgets (needs only `customtkinter` and `Pillow`).
- `CLAUDE.md` holds the design notes, measured results and gotchas.

## Limitations

- Tested on synthetic clips and short samples; judge real recordings by listening.
- Short interjections and anything near another voice are removed by design.
- Whole files are processed in memory; very long recordings need a lot of RAM.
- No license has been chosen yet.
