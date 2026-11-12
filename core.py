"""Speaker Separator - processing core (no GUI code in here).

Goal: per-speaker audio that is safe to train a voice model on, i.e. every
sample contains ONE voice, at original quality. Coverage is deliberately
sacrificed for purity: anything near another speaker is cut.

Pipeline: diarize (pyannote 3.1) -> per speaker keep only stretches that are
at least `guard` seconds away from every other speaker's speech -> drop clips
that are too short -> drop clips whose voice sounds closer to another speaker
(mislabelled by the diarizer) -> export.

Everything the worker thread does lives here. It talks to the outside world
only through the `emit(kind, payload)` callback passed to `separate()`:
  log      str                      line for the activity console
  status   str                      short current-step text
  progress float 0..1               overall progress
  stage    int 0..2                 0 load model, 1 analyze, 2 export
  device   str                      "mps" | "cuda" | "cpu"
  result   dict                     {"duration", "overlap_seconds", "speakers": [{name, path, seconds, segments}]}
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Must be set before torch is imported (it is imported lazily in the worker).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")       # CPU fallback for ops MPS lacks
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # pyannote 3.x checkpoints vs torch>=2.6

# GUI apps launched from Finder don't inherit the shell PATH; make Homebrew's ffmpeg findable.
if platform.system() == "Darwin":
    os.environ["PATH"] = os.pathsep.join(
        ["/opt/homebrew/bin", "/usr/local/bin", os.environ.get("PATH", "")]
    )

APP_NAME = "Speaker Separator"
PIPELINE_ID = "pyannote/speaker-diarization-3.1"
SAMPLE_RATE = 16_000          # what the pyannote models expect
MP3_BITRATE = "192k"
RAMP_MS = 10                  # fade at every cut so joins never click
JOIN_GAP_S = 0.25             # silence between clips in the "joined" layout
MERGE_GAP_S = 0.30            # bridge pauses shorter than this inside one speaker's speech
MIN_SIMILARITY = 0.25         # absolute floor for the voice-identity check
CONFIG_PATH = Path.home() / ".speaker_separator" / "config.json"
# Folder holding the app (next to main.py, or next to the executable when frozen).
APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent

# purity -> (guard seconds kept clear of every other speaker, minimum clip seconds)
PURITY = {"balanced": (0.20, 1.0), "strict": (0.50, 2.0)}
LAYOUTS = ("joined", "clips")
FORMATS = ("mp3", "wav")

# Rough share of the progress bar owned by each pyannote pipeline stage.
STAGE_WEIGHTS = {
    "segmentation": (0.12, 0.35),
    "speaker_counting": (0.47, 0.02),
    "embeddings": (0.49, 0.38),
}
DIARIZATION_DONE = 0.90       # progress after diarization
FILTER_DONE = 0.93            # progress after the voice-identity check


@dataclass
class Options:
    purity: str = "strict"           # key of PURITY
    num_speakers: int | None = None  # None = detect automatically
    layout: str = "joined"           # "joined": one file per speaker | "clips": a folder of clips per speaker
    fmt: str = "mp3"                 # one of FORMATS


@dataclass
class Track:
    name: str
    clips: list = field(default_factory=list)   # int16 arrays (n, channels), one per clean clip
    samples: object = None                      # int16 (n, channels), clips joined (layout "joined")
    seconds: float = 0.0                        # speech in this track


# --------------------------------------------------------------------------- #
# Settings + Hugging Face token
# --------------------------------------------------------------------------- #
def load_token() -> str:
    """HF token from the HF_TOKEN env var, else from a `.env` file: first next to
    the code (development), then in ~/.speaker_separator/ (the packaged app).

    The token is deliberately NOT hardcoded in source, so the code can be
    shared or committed without leaking it.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    for env_file in (APP_DIR / ".env", CONFIG_PATH.parent / ".env"):
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip().removeprefix("export ").strip() == "HF_TOKEN":
                    value = value.strip().strip("\"'")
                    if value:
                        return value
        except OSError:
            continue
    return ""


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def expected_device_label() -> str:
    """Cheap guess for the UI before torch is loaded."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    return "auto"


# --------------------------------------------------------------------------- #
# Pipeline (cached; loading takes seconds)
# --------------------------------------------------------------------------- #
_cache: dict = {}     # {"pipeline", "token", "device"}


def pick_device(torch):
    """mps on Apple Silicon, else cuda if present, else cpu."""
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple_silicon and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_pipeline(token: str, emit):
    import torch
    from pyannote.audio import Pipeline

    if _cache and _cache["token"] == token:
        emit("device", _cache["device"].type)
        return _cache["pipeline"]

    emit("status", "Loading model...")
    emit("log", f"Loading {PIPELINE_ID} (the first run downloads the weights)...")
    help_text = (
        "Check the Hugging Face token and make sure you accepted the terms at\n"
        "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
        "  https://huggingface.co/pyannote/segmentation-3.0"
    )
    try:
        pipeline = Pipeline.from_pretrained(PIPELINE_ID, use_auth_token=token)
    except Exception as exc:  # bad token / gated repo -> the hub raises varied errors
        raise RuntimeError(f"Could not load {PIPELINE_ID}: {exc}\n{help_text}") from exc
    if pipeline is None:  # pyannote returns None (instead of raising) when the model is gated for you
        raise RuntimeError(f"Access not granted for {PIPELINE_ID}.\n{help_text}")

    device = pick_device(torch)
    try:
        pipeline.to(device)
    except Exception as exc:
        emit("log", f"Could not move pipeline to {device} ({exc}); using CPU.")
        device = torch.device("cpu")
        pipeline.to(device)
    emit("log", f"Pipeline ready on device: {device.type}")
    emit("device", device.type)

    _cache.update(pipeline=pipeline, token=token, device=device)
    return pipeline


def to_pyannote_input(samples, sample_rate: int):
    """int16 (n, ch) array -> in-memory mono 16 kHz waveform dict for pyannote.

    Passing a tensor sidesteps pyannote's own file decoding (the usual source
    of MP3/torchaudio backend problems).
    """
    import torch
    from scipy.signal import resample_poly

    mono = samples.astype("float32").mean(axis=1) / 32768.0
    if sample_rate != SAMPLE_RATE:
        g = math.gcd(sample_rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, sample_rate // g).astype("float32")
    return {"waveform": torch.from_numpy(mono).unsqueeze(0), "sample_rate": SAMPLE_RATE}


def run_diarization(pipeline, audio_input, num_speakers, emit):
    """Run the pipeline with progress; one retry on CPU if a GPU op fails."""
    import torch

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        base, span = STAGE_WEIGHTS.get(step_name, (None, None))
        if total and completed is not None:
            if base is not None:
                emit("progress", base + span * (completed / total))
            emit("status", f"Analyzing audio... ({step_name} {completed}/{total})")
        elif step_name in STAGE_WEIGHTS:
            emit("progress", base + span)
            emit("log", f"  finished stage: {step_name}")

    kwargs = {} if not num_speakers else {"num_speakers": int(num_speakers)}
    try:
        return pipeline(audio_input, hook=hook, **kwargs)
    except Exception as exc:
        # An op that is unsupported on MPS can still blow up mid-inference.
        if _cache and _cache["device"].type != "cpu":
            emit("log", f"Inference failed on {_cache['device'].type} ({exc}). Retrying on CPU...")
            _cache["device"] = torch.device("cpu")
            pipeline.to(_cache["device"])
            emit("device", "cpu")
            return pipeline(audio_input, hook=hook, **kwargs)
        raise


# --------------------------------------------------------------------------- #
# Selecting clean, single-voice stretches (pure logic: testable without models)
# --------------------------------------------------------------------------- #
def clean_timelines(diarization, guard: float, min_clip: float) -> dict:
    """{label: Timeline} of stretches where ONLY that speaker is talking.

    Anything within `guard` seconds of another speaker's speech is removed (this
    covers true overlaps, detector boundary error and the other voice's reverb
    tail), then clips shorter than `min_clip` are dropped.
    """
    from pyannote.core import Segment, Timeline

    labels = diarization.labels()
    own = {l: diarization.label_timeline(l).support(collar=MERGE_GAP_S) for l in labels}
    out = {}
    for label in labels:
        others = Timeline([s for l in labels if l != label for s in diarization.label_timeline(l)])
        blocked = Timeline(
            [Segment(max(0.0, s.start - guard), s.end + guard) for s in others]
        ).support()
        kept = own[label].extrude(blocked)
        out[label] = Timeline([s for s in kept if s.duration >= min_clip])
    return out


def identity_filter(pipeline, waveform, timelines: dict, emit) -> dict:
    """Drop clips whose voice is closer to ANOTHER speaker's voice than to their
    own (diarizer mislabels), or implausibly far from their own.

    Embeddings use the pipeline's own speaker-embedding model. Each clip is
    compared with its speaker's centroid computed WITHOUT that clip, so a
    speaker with few clips can't vouch for itself.
    """
    import numpy as np
    from pyannote.core import Timeline

    model = getattr(pipeline, "_embedding", None)
    if model is None:
        emit("log", "Voice check skipped (embedding model not available).")
        return timelines

    sr = SAMPLE_RATE
    clips = {}   # label -> [(segment, unit vector, duration)]
    for label, tl in timelines.items():
        items = []
        for seg in tl:
            a, b = int(seg.start * sr), int(seg.end * sr)
            if b - a > 12 * sr:                       # cap long clips to a central 12 s window
                mid = (a + b) // 2
                a, b = mid - 6 * sr, mid + 6 * sr
            vec = np.asarray(model(waveform[:, a:b].unsqueeze(0)))[0].astype(np.float64)
            norm = np.linalg.norm(vec)
            items.append((seg, vec / norm if norm > 0 and np.isfinite(norm) else None, seg.duration))
        clips[label] = items

    sums = {l: sum(d * v for _, v, d in items if v is not None) if items else 0 for l, items in clips.items()}
    out, dropped = {}, 0
    for label, items in clips.items():
        keep = []
        for seg, vec, dur in items:
            if vec is None:
                dropped += 1
                continue
            rest = sums[label] - dur * vec
            sim_own = float(vec @ rest / np.linalg.norm(rest)) if np.linalg.norm(rest) > 1e-9 else None
            sim_other = max(
                (float(vec @ s / np.linalg.norm(s)) for l, s in sums.items()
                 if l != label and np.linalg.norm(s) > 1e-9),
                default=None,
            )
            bad = (sim_own is not None and sim_own < MIN_SIMILARITY) or (
                sim_own is not None and sim_other is not None and sim_own < sim_other)
            if bad:
                dropped += 1
            else:
                keep.append(seg)
        out[label] = Timeline(keep)
    if dropped:
        emit("log", f"  voice check removed {dropped} clip(s) that did not sound like their speaker.")
    return out


# --------------------------------------------------------------------------- #
# Building and exporting tracks
# --------------------------------------------------------------------------- #
def ordered_labels(diarization, timelines: dict) -> list:
    """Raw labels that still have clips, ordered by their first remaining clip."""
    firsts = {l: min(s.start for s in tl) for l, tl in timelines.items() if len(list(tl))}
    return sorted(firsts, key=firsts.get)


def _faded(x, ramp: int):
    import numpy as np

    m = len(x)
    r = min(ramp, m // 2)
    w = np.ones(m, dtype=np.float32)
    if r > 0:
        w[:r] = np.linspace(0.0, 1.0, r, endpoint=False)
        w[m - r:] = np.linspace(1.0, 0.0, r, endpoint=False)
    return (x.astype(np.float32) * w[:, None]).astype(np.int16)


def build_tracks(samples, sr: int, named_timelines: list, layout: str = "joined") -> list[Track]:
    """named_timelines: [(name, Timeline)]. Returns one Track per speaker."""
    import numpy as np

    n, ch = samples.shape
    ramp = int(sr * RAMP_MS / 1000)
    gap = np.zeros((int(JOIN_GAP_S * sr), ch), dtype=np.int16)
    tracks = []
    for name, tl in named_timelines:
        spans = [(max(0, int(s.start * sr)), min(n, int(s.end * sr))) for s in tl]
        clips = [_faded(samples[a:b], ramp) for a, b in spans if b > a]
        track = Track(name=name, clips=clips, seconds=sum(len(c) for c in clips) / sr)
        if layout == "joined" and clips:
            parts = []
            for c in clips:
                parts += [c, gap]
            track.samples = np.concatenate(parts[:-1])
        tracks.append(track)
    return tracks


def _write(samples, sr: int, ch: int, path: Path, fmt: str) -> None:
    from pydub import AudioSegment

    seg = AudioSegment(data=samples.tobytes(), sample_width=2, frame_rate=sr, channels=ch)
    if fmt == "mp3":
        seg.export(path, format="mp3", bitrate=MP3_BITRATE)
    else:
        seg.export(path, format="wav")


def export_tracks(tracks, sr: int, ch: int, out_dir: Path, opt: Options, emit) -> list[dict]:
    results = []
    for i, tr in enumerate(tracks):
        emit("status", f"Exporting {tr.name}...")
        emit("log", f"  {tr.name}: {len(tr.clips)} clips, {tr.seconds:.1f}s of clean speech")
        if opt.layout == "clips":
            folder = out_dir / tr.name
            folder.mkdir(parents=True, exist_ok=True)
            for k, clip in enumerate(tr.clips, start=1):
                _write(clip, sr, ch, folder / f"{tr.name}_{k:03d}.{opt.fmt}", opt.fmt)
            path = folder
        else:
            path = out_dir / f"{tr.name}.{opt.fmt}"
            _write(tr.samples, sr, ch, path, opt.fmt)
        results.append({"name": tr.name, "path": str(path), "seconds": tr.seconds, "segments": len(tr.clips)})
        emit("progress", FILTER_DONE + (1 - FILTER_DONE) * (i + 1) / len(tracks))
    return results


# --------------------------------------------------------------------------- #
# The job
# --------------------------------------------------------------------------- #
def separate(audio_path: str, out_dir: str, token: str, emit, options: Options | None = None) -> None:
    """Full job. Raises on failure; UI-agnostic, reports through `emit`."""
    import numpy as np

    opt = options or Options()
    if opt.purity not in PURITY or opt.layout not in LAYOUTS or opt.fmt not in FORMATS:
        raise ValueError(f"Bad options: {opt}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH. Install it (macOS: `brew install ffmpeg`).")
    guard, min_clip = PURITY[opt.purity]

    emit("stage", 0)
    emit("status", "Starting up...")
    emit("progress", 0.01)
    from pydub import AudioSegment

    pipeline = get_pipeline(token, emit)
    emit("progress", 0.06)

    emit("stage", 1)
    emit("status", "Reading audio...")
    emit("log", f"Reading {Path(audio_path).name}")
    audio = AudioSegment.from_file(audio_path).set_sample_width(2)
    sr, ch = audio.frame_rate, audio.channels
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).reshape(-1, ch)
    emit("log", f"  duration {len(audio) / 1000:.1f}s, {sr} Hz, {ch} ch")
    hint = f", {opt.num_speakers} speakers" if opt.num_speakers else ""
    emit("log", f"  strictness: {opt.purity} (keep {guard:.2f}s clear of other voices, clips >= {min_clip:.1f}s){hint}")
    audio_input = to_pyannote_input(samples, sr)
    emit("progress", 0.11)

    emit("status", "Analyzing audio...")
    diarization = run_diarization(pipeline, audio_input, opt.num_speakers, emit)
    emit("progress", DIARIZATION_DONE)

    speech_s = diarization.get_timeline().support().duration()
    overlap_s = diarization.get_overlap().duration()
    emit("log", f"Found {len(diarization.labels())} speaker(s); {speech_s:.1f}s of speech, "
                f"{overlap_s:.1f}s of it overlapping.")

    emit("stage", 2)
    emit("status", "Selecting clean speech...")
    timelines = clean_timelines(diarization, guard, min_clip)
    emit("status", "Checking voices...")
    try:
        timelines = identity_filter(pipeline, audio_input["waveform"], timelines, emit)
    except Exception as exc:  # the check is a safety net; never fail the job because of it
        emit("log", f"Voice check skipped: {exc}")
    emit("progress", FILTER_DONE)

    labels = ordered_labels(diarization, timelines)
    if not labels:
        emit("log", "No clean single-speaker speech found. Try purity 'Balanced', or a recording with less overlap.")
        emit("result", {"duration": len(audio) / 1000, "overlap_seconds": overlap_s, "speakers": []})
        return
    named = [(f"Speaker_{i}", timelines[l]) for i, l in enumerate(labels, start=1)]
    tracks = build_tracks(samples, sr, named, opt.layout)
    kept = sum(t.seconds for t in tracks)
    emit("log", f"Keeping {kept:.1f}s of clean speech ({100 * kept / max(speech_s, 1e-9):.0f}% of all speech).")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = export_tracks(tracks, sr, ch, out, opt, emit)
    emit("log", "Saved:\n" + "\n".join(f"  {r['path']}" for r in results))
    emit("result", {"duration": len(audio) / 1000, "overlap_seconds": overlap_s, "speakers": results})
