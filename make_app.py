#!/usr/bin/env python3
"""Build "Speaker Separator.app" for macOS (run with the project's Python env).

    conda activate speaker-separator
    python make_app.py              # installs into ~/Applications
    python make_app.py --dest ./dist

What it makes: a small self-contained bundle, NOT a frozen PyTorch bundle (that
would be ~3 GB and fragile). The bundle holds a copy of the source, an icon and
a launcher; it runs with THIS Python environment (a symlink to the interpreter
that ran this script, placed inside the bundle so macOS shows the app's own
name and icon instead of "Python"). Re-run this script after changing the code.
If the Python environment is moved or deleted, the app shows a clear dialog.

The Hugging Face token is read from ~/.speaker_separator/.env. If the project
has a `.env` and that file does not exist yet, it is copied there (never
overwritten, never printed).
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
APP_NAME = "Speaker Separator"
BUNDLE_ID = "local.speaker-separator"
SOURCE_FILES = ["main.py", "core.py", "neu.py"]
USER_DIR = Path.home() / ".speaker_separator"

LAUNCHER = """#!/bin/bash
# Launches Speaker Separator with the Python environment it was built against.
APP="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$APP/Contents/MacOS/{app_name}"
LOG="$HOME/Library/Logs/SpeakerSeparator.log"
mkdir -p "$HOME/Library/Logs"
if [ ! -x "$PY" ]; then
  osascript -e 'display dialog "Speaker Separator cannot find its Python environment ({python}). Re-run make_app.py from the project folder to rebuild the app." buttons {{"OK"}} default button "OK" with icon stop with title "Speaker Separator"' >/dev/null 2>&1
  exit 1
fi
cd "$HOME"
exec "$PY" "$APP/Contents/Resources/app/main.py" >"$LOG" 2>&1
"""


# --------------------------------------------------------------------------- #
# Icon
# --------------------------------------------------------------------------- #
def make_icon_png(size: int = 1024):
    """Dark soft-UI squircle with a two-tone waveform, in the app's accent colours."""
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    k = 2                                   # draw at 2x, then downsample for smooth edges
    S = size * k
    body = int(S * 0.805)                   # macOS icon grid: ~80% of the canvas
    x0 = (S - body) // 2
    radius = int(body * 0.225)

    def rr_mask(inset=0, dy=0):
        m = Image.new("L", (S, S), 0)
        ImageDraw.Draw(m).rounded_rectangle(
            (x0 + inset, x0 + inset + dy, x0 + body - inset, x0 + body - inset + dy), radius=radius - inset, fill=255)
        return m

    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow = rr_mask(dy=int(S * 0.012)).filter(ImageFilter.GaussianBlur(S * 0.02))
    canvas.paste(Image.new("RGBA", (S, S), (0, 0, 0, 120)), (0, 0), shadow)

    top, bottom = (0x3b, 0x3f, 0x50), (0x1d, 0x1f, 0x27)
    grad = Image.linear_gradient("L").resize((S, S))
    face = Image.composite(Image.new("RGB", (S, S), bottom), Image.new("RGB", (S, S), top), grad)
    canvas.paste(face, (0, 0), rr_mask())
    edge = ImageChops.subtract(rr_mask(), rr_mask(inset=int(S * 0.004)))
    canvas.paste(Image.new("RGBA", (S, S), (255, 255, 255, 40)), (0, 0), edge)

    heights = [0.26, 0.46, 0.70, 0.50, 0.86, 0.50, 0.70, 0.46, 0.26]
    n, bar_w, gap = len(heights), int(S * 0.050), int(S * 0.032)
    total_w = n * bar_w + (n - 1) * gap
    left, max_h, cy = (S - total_w) // 2, int(S * 0.46), S // 2
    c1, c2 = (0x7c, 0x9c, 0xff), (0xb1, 0x96, 0xff)
    bars = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(bars)
    for i, h in enumerate(heights):
        t = i / (n - 1)
        col = tuple(int(c1[j] + (c2[j] - c1[j]) * t) for j in range(3)) + (255,)
        x = left + i * (bar_w + gap)
        hh = int(max_h * h)
        d.rounded_rectangle((x, cy - hh // 2, x + bar_w, cy + hh // 2), radius=bar_w // 2, fill=col)
    glow = bars.filter(ImageFilter.GaussianBlur(S * 0.018))
    glow.putalpha(glow.getchannel("A").point(lambda v: int(v * 0.55)))
    canvas = Image.alpha_composite(canvas, glow)
    canvas = Image.alpha_composite(canvas, bars)
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def make_icns(dest: Path) -> None:
    from PIL import Image

    lanczos = Image.Resampling.LANCZOS
    master = make_icon_png(1024)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            master.resize((base, base), lanczos).save(iconset / f"icon_{base}x{base}.png")
            master.resize((base * 2, base * 2), lanczos).save(iconset / f"icon_{base}x{base}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dest)], check=True)


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
def build(dest_dir: Path) -> Path:
    app = dest_dir / f"{APP_NAME}.app"
    plist_path = app / "Contents" / "Info.plist"
    if app.exists():
        try:
            ours = plistlib.loads(plist_path.read_bytes()).get("CFBundleIdentifier") == BUNDLE_ID
        except (OSError, ValueError):
            ours = False
        if not ours:
            sys.exit(f"{app} exists and was not built by this script; refusing to replace it.")
        shutil.rmtree(app)

    macos, res = app / "Contents" / "MacOS", app / "Contents" / "Resources"
    (res / "app").mkdir(parents=True)
    macos.mkdir(parents=True, exist_ok=True)

    for name in SOURCE_FILES:
        shutil.copy2(SRC / name, res / "app" / name)

    make_icns(res / "AppIcon.icns")

    python = Path(sys.executable)
    (macos / APP_NAME).symlink_to(python)
    launcher = macos / "launcher"
    launcher.write_text(LAUNCHER.format(app_name=APP_NAME, python=python))
    launcher.chmod(0o755)

    plist_path.write_bytes(plistlib.dumps({
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": "launcher",
        "CFBundleIconFile": "AppIcon",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.utilities",
        "NSHighResolutionCapable": True,
        "NSAppSleepDisabled": True,          # long jobs must not be throttled when the window is in the background
        "NSSupportsAutomaticGraphicsSwitching": True,
    }))
    subprocess.run(["touch", str(app)], check=False)   # nudge Finder/Launchpad to refresh the icon
    return app


def install_token() -> str:
    env, target = SRC / ".env", USER_DIR / ".env"
    if target.exists():
        return f"token: using existing {target}"
    if not env.exists():
        return f"token: none found. Create {target} containing HF_TOKEN=hf_..."
    USER_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(env, target)
    os.chmod(target, 0o600)
    return f"token: copied the project's .env to {target} (permissions 600)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path.home() / "Applications", help="folder to install into")
    args = ap.parse_args()
    if sys.platform != "darwin":
        sys.exit("make_app.py builds a macOS app bundle; run it on a Mac.")
    for mod in ("customtkinter", "PIL", "pyannote.audio", "torch"):
        try:
            __import__(mod)
        except ImportError:
            sys.exit(f"Run this from the project's Python environment (missing module: {mod}).")
    args.dest.mkdir(parents=True, exist_ok=True)
    app = build(args.dest.expanduser().resolve())
    print(f"built:  {app}")
    print(f"python: {sys.executable}")
    print(install_token())


if __name__ == "__main__":
    main()
