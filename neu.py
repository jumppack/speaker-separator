"""Neumorphic ("soft UI") building blocks for CustomTkinter.

Tk has no shadows, so every surface is a Pillow-rendered image: a dark shadow
falls to the bottom-right and a light one to the top-left of a body that has
the same colour as the window. Pressed / inset elements flip this inward.

Design notes (what keeps this usable rather than just pretty):
  * Pure neumorphism has weak contrast, so interactive state is carried by an
    accent colour, not shadow alone: primary button, active step, progress fill.
  * Text uses high-contrast colours; "muted" is only for secondary labels.
  * Every control has distinct normal / hover / pressed / disabled looks.
"""

from __future__ import annotations

import sys
from functools import lru_cache

import customtkinter as ctk
from PIL import Image, ImageChops, ImageDraw, ImageFilter

# --- palette -------------------------------------------------------------- #
BG = "#2b2e38"
LIGHT = "#3b3f4e"      # highlight shadow (top-left)
DARK = "#1b1d24"       # drop shadow (bottom-right)
TEXT = "#eceff6"
MUTED = "#9aa1b4"
ACCENT = "#7c9cff"
ACCENT2 = "#b196ff"
ON_ACCENT = "#0f1220"  # text on accent surfaces (7:1 contrast)
GOOD = "#4cd7a0"
WARN = "#f5b84c"
BAD = "#ff6b6b"

SS = 3                 # supersampling factor for smooth edges
HAND_CURSOR = "pointinghand" if sys.platform == "darwin" else "hand2"


# --- low-level drawing ---------------------------------------------------- #
def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb)


def mix(c1: str, c2: str, t: float) -> str:
    a, b = _rgb(c1), _rgb(c2)
    return _hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def _canvas(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w * SS, h * SS), _rgb(BG))


def _mask(size, box, radius) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle(box, radius=radius, fill=255)
    return m


def _shift(img: Image.Image, dx: int, dy: int, fill: int = 0) -> Image.Image:
    out = Image.new(img.mode, img.size, fill)
    out.paste(img, (dx, dy))
    return out


def _paint(img: Image.Image, color: str, mask: Image.Image) -> None:
    img.paste(Image.new("RGB", img.size, _rgb(color)), (0, 0), mask)


def _gradient(size, c1: str, c2: str, horizontal: bool = False) -> Image.Image:
    if c1 == c2:
        return Image.new("RGB", size, _rgb(c1))
    g = Image.linear_gradient("L")  # black at top -> white at bottom
    if horizontal:
        g = g.transpose(Image.Transpose.ROTATE_90)  # black left -> white right
    g = g.resize(size)
    return Image.composite(Image.new("RGB", size, _rgb(c2)), Image.new("RGB", size, _rgb(c1)), g)


def _finish(img: Image.Image, w: int, h: int) -> ctk.CTkImage:
    """Downsample the supersampled drawing to 2x (retina) and wrap it."""
    small = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=small, dark_image=small, size=(w, h))


def shadow_pad(depth: int) -> int:
    """Margin a raised surface needs so its shadow is not clipped."""
    return int(depth * 2.8)


@lru_cache(maxsize=None)
def _raised_pil(w, h, radius, depth, top, bottom, horizontal) -> Image.Image:
    img = _canvas(w, h)
    pad = shadow_pad(depth) * SS
    W, H = img.size
    m = _mask(img.size, (pad, pad, W - pad - 1, H - pad - 1), radius * SS)
    d = depth * SS
    blur = ImageFilter.GaussianBlur(d * 0.85)
    _paint(img, DARK, _shift(m, d, d).filter(blur))
    _paint(img, LIGHT, _shift(m, -d, -d).filter(blur))
    img.paste(_gradient(img.size, top, bottom, horizontal), (0, 0), m)
    return img


@lru_cache(maxsize=None)
def _inset_pil(w, h, radius, depth, fill) -> Image.Image:
    img = _canvas(w, h)
    pad = max(2, depth) * SS
    W, H = img.size
    m = _mask(img.size, (pad, pad, W - pad - 1, H - pad - 1), radius * SS)
    outside = ImageChops.invert(m)
    d = depth * SS
    blur = ImageFilter.GaussianBlur(d * 0.8)
    dark = ImageChops.multiply(_shift(outside, d, d, 255).filter(blur), m)
    light = ImageChops.multiply(_shift(outside, -d, -d, 255).filter(blur), m)
    _paint(img, fill, m)
    _paint(img, DARK, dark)
    _paint(img, LIGHT, light)
    return img


def raised(w, h, radius, depth=4, top=BG, bottom=BG, horizontal=False) -> ctk.CTkImage:
    return _finish(_raised_pil(w, h, radius, depth, top, bottom, horizontal), w, h)


def inset(w, h, radius, depth=4, fill=BG) -> ctk.CTkImage:
    return _finish(_inset_pil(w, h, radius, depth, fill), w, h)


# --- progress bar rendering ------------------------------------------------ #
def progress_image(w: int, h: int, frac: float, phase: float = 0.0) -> ctk.CTkImage:
    """Inset track with a gradient fill and (optionally) a moving highlight."""
    img = _inset_pil(w, h, h // 2, 3, BG).copy()
    frac = max(0.0, min(1.0, frac))
    margin = 3 * SS + 3 * SS
    W, H = img.size
    x0, y0, x1_max, y1 = margin, margin, W - margin, H - margin
    fill_w = int((x1_max - x0) * frac)
    if fill_w >= (y1 - y0):  # smaller than a full cap looks broken; wait until it fits
        box = (x0, y0, x0 + fill_w, y1)
        m = _mask(img.size, box, (y1 - y0) // 2)
        img.paste(_gradient(img.size, ACCENT, ACCENT2, horizontal=True), (0, 0), m)
        if phase:  # soft highlight sweeping across the filled part
            band = Image.new("L", img.size, 0)
            cx = x0 + int(((phase % 1.0) * 1.4 - 0.2) * fill_w)
            half = max(8, fill_w // 8)
            ImageDraw.Draw(band).rectangle((cx - half, y0, cx + half, y1), fill=110)
            band = ImageChops.multiply(band.filter(ImageFilter.GaussianBlur(half * 0.6)), m)
            _paint(img, "#ffffff", band)
    return _finish(img, w, h)


# --- widgets ---------------------------------------------------------------- #
class NeuPanel(ctk.CTkFrame):
    """A raised card or inset well. Create content widgets AFTER the panel
    (they stack above the background image) and position them inside it."""

    def __init__(self, master, width, height, *, well=False, radius=24, depth=4):
        super().__init__(master, width=width, height=height, fg_color=BG, corner_radius=0)
        self.grid_propagate(False)
        self.pack_propagate(False)
        image = inset(width, height, radius, depth) if well else raised(width, height, radius, depth)
        self._surface = ctk.CTkLabel(self, text="", image=image, fg_color=BG, width=width, height=height)
        self._surface.place(x=0, y=0)


class NeuButton(ctk.CTkLabel):
    """Raised button that sinks when pressed. `primary` uses the accent gradient."""

    def __init__(self, master, text, command=None, *, width=190, height=60, primary=False, font=None):
        self._cmd = command
        self._primary = primary
        self._enabled = True
        self._down = False
        self._hover = False
        r = (height - 2 * shadow_pad(3)) // 2 + 2  # pill shape
        if primary:
            up = raised(width, height, r, 3, ACCENT, ACCENT2, True)
            hov = raised(width, height, r, 3, mix(ACCENT, "#ffffff", 0.16), mix(ACCENT2, "#ffffff", 0.16), True)
            down = inset(width, height, r, 3, mix(ACCENT, ACCENT2, 0.5))
            self._text_colors = {"normal": ON_ACCENT, "hover": ON_ACCENT, "pressed": ON_ACCENT, "disabled": MUTED}
        else:
            up = raised(width, height, r, 3, mix(BG, LIGHT, 0.55), mix(BG, DARK, 0.30))
            hov = raised(width, height, r, 3, mix(BG, LIGHT, 1.0), mix(BG, DARK, 0.15))
            down = inset(width, height, r, 3)
            self._text_colors = {"normal": TEXT, "hover": "#ffffff", "pressed": ACCENT, "disabled": MUTED}
        off = inset(width, height, r, 2)
        self._looks = {"normal": up, "hover": hov, "pressed": down, "disabled": off}

        super().__init__(
            master, text=text, image=up, compound="center", width=width, height=height,
            fg_color="transparent", text_color=self._text_colors["normal"],
            font=font or ctk.CTkFont(size=15, weight="bold"),
        )
        for seq, fn in (("<Enter>", self._on_enter), ("<Leave>", self._on_leave),
                        ("<ButtonPress-1>", self._on_press), ("<ButtonRelease-1>", self._on_release)):
            self.bind(seq, fn, add=True)
        self._set_cursor(HAND_CURSOR)

    # -- public --
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._down = False
        self._set_cursor(HAND_CURSOR if enabled else "")
        self._refresh()

    def set_text(self, text: str) -> None:
        self.configure(text=text)

    # -- internals --
    def _set_cursor(self, cursor: str) -> None:
        try:
            self._label.configure(cursor=cursor)
        except Exception:
            pass

    def _look(self) -> str:
        if not self._enabled:
            return "disabled"
        if self._down:
            return "pressed"
        return "hover" if self._hover else "normal"

    def _refresh(self) -> None:
        state = self._look()
        self.configure(image=self._looks[state], text_color=self._text_colors[state])

    def _pointer_inside(self) -> bool:
        px, py = self.winfo_pointerxy()
        x, y = self.winfo_rootx(), self.winfo_rooty()
        return x <= px < x + self.winfo_width() and y <= py < y + self.winfo_height()

    def _on_enter(self, _e) -> None:
        self._hover = True
        self._refresh()

    def _on_leave(self, _e) -> None:
        # Enter/Leave also fire between the canvas and label inside this widget.
        self.after(15, self._check_hover)

    def _check_hover(self) -> None:
        self._hover = self._pointer_inside()
        if not self._hover:
            self._down = False
        self._refresh()

    def _on_press(self, _e) -> None:
        if self._enabled:
            self._down = True
            self._refresh()

    def _on_release(self, _e) -> None:
        was_down = self._down
        self._down = False
        self._refresh()
        if was_down and self._enabled and self._pointer_inside() and self._cmd:
            self._cmd()


class ProgressBar(ctk.CTkLabel):
    def __init__(self, master, width=844, height=34):
        self._bar_w, self._bar_h = width, height
        self.value = 0.0
        super().__init__(master, text="", image=progress_image(width, height, 0), width=width,
                         height=height, fg_color="transparent")
        self._drawn = (-1.0, -1.0)

    def render(self, value: float, phase: float, animate: bool) -> None:
        key = (round(value, 3), round(phase, 2) if animate else 0.0)
        if key == self._drawn:
            return
        self._drawn = key
        self.configure(image=progress_image(self._bar_w, self._bar_h, value, phase if animate else 0.0))


class Stepper(ctk.CTkFrame):
    """Compact row of numbered steps joined by progress lines.
    States: pending / active / done / error."""

    def __init__(self, master, labels, width=844, dot=48, caption_w=112, line_h=22):
        super().__init__(master, fg_color="transparent", width=width, height=dot)
        n = len(labels)
        line_w = (width - n * (dot + caption_w)) // (n - 1)
        self._images = {
            "pending": raised(dot, dot, dot // 2, 3),
            "active": inset(dot, dot, dot // 2, 3),
            "done": raised(dot, dot, dot // 2, 3, ACCENT, ACCENT2, True),
            "error": raised(dot, dot, dot // 2, 3, BAD, mix(BAD, "#ffffff", 0.25), True),
        }
        self._line_off = progress_image(line_w, line_h, 0)
        self._line_on = progress_image(line_w, line_h, 1)
        self._dots, self._lines, self._captions = [], [], []
        col = 0
        for i, text in enumerate(labels):
            d = ctk.CTkLabel(self, text=str(i + 1), image=self._images["pending"], compound="center",
                             width=dot, height=dot, fg_color="transparent",
                             font=ctk.CTkFont(size=15, weight="bold"), text_color=MUTED)
            d.grid(row=0, column=col)
            c = ctk.CTkLabel(self, text=text, fg_color="transparent", text_color=MUTED, anchor="w",
                             width=caption_w, font=ctk.CTkFont(size=13, weight="bold"))
            c.grid(row=0, column=col + 1)
            self._dots.append(d)
            self._captions.append(c)
            col += 2
            if i < n - 1:
                ln = ctk.CTkLabel(self, text="", image=self._line_off, width=line_w,
                                  height=line_h, fg_color="transparent")
                ln.grid(row=0, column=col)
                self._lines.append(ln)
                col += 1

    def set_stage(self, stage: int, error: bool = False, idle: bool = False) -> None:
        """stage = index of the active step; stage == len(steps) means all done;
        idle=True shows every step as pending (nothing running)."""
        for i, (d, c) in enumerate(zip(self._dots, self._captions)):
            if idle:
                kind, num, fg, cap = "pending", str(i + 1), MUTED, MUTED
            elif i < stage:
                kind, num, fg, cap = "done", "✓", ON_ACCENT, TEXT
            elif i == stage and error:
                kind, num, fg, cap = "error", "!", ON_ACCENT, BAD
            elif i == stage:
                kind, num, fg, cap = "active", str(i + 1), ACCENT, TEXT
            else:
                kind, num, fg, cap = "pending", str(i + 1), MUTED, MUTED
            d.configure(image=self._images[kind], text=num, text_color=fg)
            c.configure(text_color=cap)
        for i, ln in enumerate(self._lines):
            ln.configure(image=self._line_on if (not idle and i < stage) else self._line_off)


def segmented_slices(w: int, h: int, n: int, weights=None, disabled: bool = False):
    """images[selected][segment]: an inset track with a raised accent pill under
    `selected`, cut into n vertical slices. Each slice becomes its own label so
    Tk draws the text over the picture (a separate text label would paint an
    opaque rectangle over it). `disabled` draws a dimmed pill (still readable)."""
    weights = weights or [1] * n
    total = float(sum(weights))
    bounds = [0]
    for wt in weights:
        bounds.append(bounds[-1] + wt / total * w)
    bounds = [round(b) for b in bounds]
    c1, c2 = (mix(ACCENT, BG, 0.55), mix(ACCENT2, BG, 0.55)) if disabled else (ACCENT, ACCENT2)
    out = []
    for sel in range(n):
        img = _inset_pil(w, h, h // 2, 3, BG).copy()
        W, H = img.size
        margin = 6 * SS
        left, right = margin + (bounds[sel] / w) * (W - 2 * margin), margin + (bounds[sel + 1] / w) * (W - 2 * margin)
        box = (int(left), margin, int(right) - 1, H - margin - 1)
        pill = _mask(img.size, box, (H - 2 * margin) // 2)
        shadow = _shift(pill, 2 * SS, 2 * SS).filter(ImageFilter.GaussianBlur(2 * SS)).point(lambda v: int(v * 0.8))
        _paint(img, DARK, shadow)
        img.paste(_gradient(img.size, c1, c2, True), (0, 0), pill)
        small = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        row = []
        for i in range(n):
            part = small.crop((bounds[i] * 2, 0, bounds[i + 1] * 2, h * 2))
            row.append(ctk.CTkImage(light_image=part, dark_image=part, size=(bounds[i + 1] - bounds[i], h)))
        out.append(row)
    return out


class Segmented(ctk.CTkFrame):
    """Pill-shaped single-choice selector: options = [(label, value), ...].
    `weights` sets relative segment widths (e.g. a longer first label)."""

    def __init__(self, master, options, width, height=44, value=None, command=None, weights=None):
        super().__init__(master, fg_color="transparent", width=width, height=height)
        self._opts, self._cmd, self._enabled = options, command, True
        n = len(options)
        self._images = {False: segmented_slices(width, height, n, weights),
                        True: segmented_slices(width, height, n, weights, disabled=True)}
        self._labels = []
        for i, (text, _) in enumerate(options):
            slice_w = self._images[False][0][i]._size[0]
            lb = ctk.CTkLabel(self, text=text, compound="center", width=slice_w, height=height,
                              fg_color="transparent", font=ctk.CTkFont(size=13, weight="bold"))
            lb.grid(row=0, column=i)
            lb.bind("<ButtonRelease-1>", lambda _e, i=i: self._click(i), add=True)
            try:
                lb._label.configure(cursor=HAND_CURSOR)
            except Exception:
                pass
            self._labels.append(lb)
        values = [v for _, v in options]
        self._index = values.index(value) if value in values else 0
        self._render()

    @property
    def value(self):
        return self._opts[self._index][1]

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._render()

    def _click(self, i: int) -> None:
        if not self._enabled or i == self._index:
            return
        self._index = i
        self._render()
        if self._cmd:
            self._cmd(self.value)

    def _render(self) -> None:
        dim = not self._enabled
        for i, lb in enumerate(self._labels):
            selected = i == self._index
            # Selected text is always high-contrast against its pill, enabled or not.
            color = (TEXT if dim else ON_ACCENT) if selected else (MUTED if dim else TEXT)
            lb.configure(image=self._images[dim][self._index][i], text_color=color)


def shorten_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = limit - 1
    head = keep // 3
    return text[:head] + "…" + text[-(keep - head):]
