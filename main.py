"""Speaker Separator - entry point and UI.

The heavy lifting is in core.py and runs on a worker thread. The worker never
touches Tk widgets: it posts events to a queue.Queue that the UI thread drains
inside `_tick()` (~25 fps), which also drives the progress animation.

Layout rule: every element owns its own rectangle. Inside a card, children are
placed at coordinates that cannot intersect (checked by the geometry test in
CLAUDE.md), never "roughly below" something else.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import core  # first: sets torch-related env vars before anything can import torch
import customtkinter as ctk

import neu
from neu import ACCENT, BAD, BG, GOOD, MUTED, TEXT, WARN

WIN_W, WIN_H = 900, 760
CARD_W, CARD_H = 422, 118
FULL_W = 2 * CARD_W          # 844: width of every full-row element
OPTS_H = 220
CONSOLE_H = 110
TICK_MS = 40
STEPS = ["Load model", "Analyze audio", "Export files"]
DEVICE_LABELS = {
    "mps": ("Metal GPU (MPS)", GOOD),
    "cuda": ("CUDA GPU", GOOD),
    "cpu": ("CPU", WARN),
    "auto": ("Device: auto", MUTED),
}

# --- options, in plain language -------------------------------------------- #
STRICT_OPTIONS = [("Very strict", "strict"), ("Normal", "balanced")]
STRICT_HINTS = {
    "strict": "Only long, clearly isolated clips. Best for training.",
    "balanced": "Keeps shorter clips too: more audio, a little less safe.",
}
SPEAKER_OPTIONS = [("Auto-detect", None), ("2", 2), ("3", 3), ("4", 4)]
SPEAKER_HINT = "Pick the real number if you know it. It improves accuracy."
SAVE_OPTIONS = [("One file", "joined"), ("Many clips", "clips")]
SAVE_HINTS = {
    "joined": "One long file per speaker (Speaker_1, Speaker_2, ...).",
    "clips": "A folder of short clips for each speaker.",
}
TYPE_OPTIONS = [("MP3", "mp3"), ("WAV", "wav")]
TYPE_HINTS = {
    "mp3": "Smaller files, slight quality loss.",
    "wav": "Full quality. Best for training a voice.",
}
COL_X = (36, 436)            # x of the two option columns inside the options card
ROW_Y = (22, 118)            # y of the caption of each option row


def reveal(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", path])


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def neu_device(kind: str) -> tuple[str, str]:
    return DEVICE_LABELS.get(kind, DEVICE_LABELS["auto"])


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__(fg_color=BG)
        ctk.set_appearance_mode("dark")
        self.title(core.APP_NAME)
        self.resizable(False, False)
        x = max(0, (self.winfo_screenwidth() - WIN_W) // 2)
        y = max(0, (self.winfo_screenheight() - WIN_H) // 4)
        self.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

        cfg = self.cfg = core.load_config()
        self.token = core.load_token()
        self.audio_path = ""
        self.output_dir = cfg.get("output_dir", "")
        self.opt_purity = cfg.get("purity") if cfg.get("purity") in core.PURITY else "strict"
        self.opt_speakers = cfg.get("speakers") if cfg.get("speakers") in (2, 3, 4) else None
        self.opt_layout = cfg.get("layout") if cfg.get("layout") in core.LAYOUTS else "joined"
        self.opt_fmt = cfg.get("fmt") if cfg.get("fmt") in core.FORMATS else "mp3"
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.busy = False
        self._awake: subprocess.Popen | None = None   # `caffeinate` while a job runs (macOS)
        self.stage = 0
        self.target = 0.0      # progress reported by the worker
        self.shown = 0.0       # eased value actually drawn
        self.phase = 0.0       # highlight sweep position
        self.last_result: dict | None = None

        self._build_ui()
        self._refresh_ready()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        mod = "Command" if sys.platform == "darwin" else "Control"
        self.bind(f"<{mod}-o>", lambda _e: self._pick_audio())
        self.after(TICK_MS, self._tick)

    # ------------------------------------------------------------------ UI -- #
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        pad = {"padx": 16}

        # Row 0 - header: title + status pills
        header = ctk.CTkFrame(self, fg_color="transparent", height=64)
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(20, 0))
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text=core.APP_NAME, text_color=TEXT, anchor="w",
                     font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="Clean single-voice audio for training. Everything runs on this computer.",
                     text_color=MUTED, anchor="w", font=ctk.CTkFont(size=13)).grid(row=1, column=0, sticky="w")
        pills = ctk.CTkFrame(header, fg_color="transparent")
        pills.grid(row=0, column=1, rowspan=2, sticky="e")
        self.pill_token = self._pill(pills, 0)
        self.pill_device = self._pill(pills, 1)
        self._set_pill(self.pill_token, *(("Token ready", GOOD) if self.token else ("Token missing", BAD)))
        self._set_pill(self.pill_device, *neu_device(core.expected_device_label()))

        # Row 1 - input cards
        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.grid(row=1, column=0, **pad, pady=(6, 0))
        self.title_audio, self.lbl_audio, self.btn_audio = self._card(
            cards, 0, "AUDIO FILE", "No file chosen", "Choose file", self._pick_audio)
        self.title_out, self.lbl_out, self.btn_out = self._card(
            cards, 1, "SAVE RESULTS IN", "No folder chosen", "Choose folder", self._pick_output)
        if self.output_dir:
            self.lbl_out.configure(text=neu.shorten_middle(self.output_dir, 24), text_color=TEXT)

        # Row 2 - options (2 x 2 grid; every control has a caption above and a one-line explanation below)
        opts = neu.NeuPanel(self, FULL_W, OPTS_H, radius=26, depth=5)
        opts.grid(row=2, column=0, **pad, pady=(4, 0))
        self.seg_purity = self._option(opts, 0, 0, "STRICTNESS", STRICT_OPTIONS, self.opt_purity,
                                       STRICT_HINTS, width=300)
        self.seg_speakers = self._option(opts, 1, 0, "NUMBER OF SPEAKERS", SPEAKER_OPTIONS, self.opt_speakers,
                                         SPEAKER_HINT, width=370, weights=[2.6, 1, 1, 1])
        self.seg_layout = self._option(opts, 0, 1, "SAVE EACH SPEAKER AS", SAVE_OPTIONS, self.opt_layout,
                                       SAVE_HINTS, width=300)
        self.seg_fmt = self._option(opts, 1, 1, "FILE TYPE", TYPE_OPTIONS, self.opt_fmt,
                                    TYPE_HINTS, width=300)

        # Row 3 - actions
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=3, column=0, **pad, pady=(4, 0))
        self.btn_start = neu.NeuButton(actions, "Start Separation", self._start,
                                       width=FULL_W - 300, height=64, primary=True,
                                       font=ctk.CTkFont(size=17, weight="bold"))
        self.btn_start.grid(row=0, column=0)
        self.btn_open = neu.NeuButton(actions, "Open Folder", self._open_output, width=300, height=64)
        self.btn_open.grid(row=0, column=1)

        # Row 4 - steps
        self.stepper = neu.Stepper(self, STEPS, width=FULL_W - 40)
        self.stepper.grid(row=4, column=0, **pad, pady=(10, 0))
        self.stepper.set_stage(0, idle=True)

        # Row 5/6 - progress bar + status line
        self.bar = neu.ProgressBar(self, FULL_W, 34)
        self.bar.grid(row=5, column=0, **pad, pady=(6, 0))
        status = ctk.CTkFrame(self, fg_color="transparent", width=FULL_W - 40, height=28)
        status.grid(row=6, column=0, **pad)
        status.grid_propagate(False)
        status.grid_columnconfigure(0, weight=1)
        self.lbl_status = ctk.CTkLabel(status, text="", anchor="w", text_color=TEXT, height=24, width=700,
                                       font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_status.grid(row=0, column=0, sticky="w")
        self.lbl_pct = ctk.CTkLabel(status, text="", anchor="e", text_color=MUTED, height=24,
                                    font=ctk.CTkFont(size=13))
        self.lbl_pct.grid(row=0, column=1, sticky="e")

        # Row 7 - activity log (inset well)
        well = neu.NeuPanel(self, FULL_W, CONSOLE_H, well=True, radius=20, depth=4)
        well.grid(row=7, column=0, **pad, pady=(2, 14))
        self.console = ctk.CTkTextbox(
            well, fg_color=BG, text_color=TEXT, border_width=0, wrap="word",
            font=ctk.CTkFont(family="Menlo" if sys.platform == "darwin" else "Consolas", size=12),
            scrollbar_button_color=neu.LIGHT, scrollbar_button_hover_color=MUTED,
            width=FULL_W - 56, height=CONSOLE_H - 30,
        )
        self.console.place(x=24, y=14)
        # Readable text selection (the default highlight can hide the text).
        self.console._textbox.configure(selectbackground=ACCENT, selectforeground=neu.ON_ACCENT,
                                        inactiveselectbackground=neu.mix(ACCENT, BG, 0.5), padx=8, pady=4)
        self.console.tag_config("err", foreground=BAD)
        self.console.tag_config("ok", foreground=GOOD)
        self.console.tag_config("dim", foreground=MUTED)
        self.console.configure(state="disabled")

    def _pill(self, parent, col: int):
        panel = neu.NeuPanel(parent, 196, 46, well=True, radius=23, depth=3)
        panel.grid(row=0, column=col)
        dot = ctk.CTkLabel(panel, text="●", width=14, fg_color="transparent", font=ctk.CTkFont(size=13))
        dot.place(x=22, rely=0.5, anchor="w")
        txt = ctk.CTkLabel(panel, text="", anchor="w", fg_color="transparent", text_color=TEXT,
                           width=130, font=ctk.CTkFont(size=13, weight="bold"))
        txt.place(x=42, rely=0.5, anchor="w")
        return dot, txt

    @staticmethod
    def _set_pill(pill, text: str, color: str) -> None:
        dot, txt = pill
        dot.configure(text_color=color)
        txt.configure(text=text)

    def _option(self, parent, col, row, title, options, value, hints, width, weights=None):
        """Caption (above), selector, one-line explanation (below). hints: str or {value: str}."""
        # NOTE: CTkLabel defaults to height=28 whatever the font; always pass height explicitly.
        x, y = COL_X[col], ROW_Y[row]
        ctk.CTkLabel(parent, text=title, text_color=MUTED, fg_color="transparent", anchor="w",
                     width=width, height=16, font=ctk.CTkFont(size=12, weight="bold")).place(x=x, y=y)
        hint = ctk.CTkLabel(parent, text="", text_color=MUTED, fg_color="transparent", anchor="w",
                            width=380, height=16, font=ctk.CTkFont(size=12))
        hint.place(x=x, y=y + 66)

        def show(v):
            hint.configure(text=hints if isinstance(hints, str) else hints.get(v, ""))

        seg = neu.Segmented(parent, options, width=width, height=44, value=value, command=show, weights=weights)
        seg.place(x=x - 3, y=y + 20)
        show(seg.value)
        return seg

    def _card(self, parent, col, title, placeholder, button_text, command):
        """Card = caption, a read-only well showing the choice, and a button to its right."""
        card = neu.NeuPanel(parent, CARD_W, CARD_H, radius=26, depth=5)
        card.grid(row=0, column=col)
        title_lbl = ctk.CTkLabel(card, text=title, text_color=MUTED, fg_color="transparent", anchor="w",
                                 width=340, height=16, font=ctk.CTkFont(size=12, weight="bold"))
        title_lbl.place(x=36, y=24)
        well = neu.NeuPanel(card, 222, 48, well=True, radius=14, depth=3)
        well.place(x=30, y=46)
        label = ctk.CTkLabel(well, text=placeholder, anchor="w", text_color=MUTED, fg_color="transparent",
                             width=186, height=20, font=ctk.CTkFont(size=14))
        label.place(x=18, rely=0.5, anchor="w")
        button = neu.NeuButton(card, button_text, command, width=150, height=56)
        button.place(x=258, y=42)
        return title_lbl, label, button

    # --------------------------------------------------------- state helpers -- #
    def _log(self, message: str, tag: str | None = None) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.console.configure(state="normal")
        self.console.insert("end", f"{stamp}  ", "dim")
        self.console.insert("end", message + "\n", tag or ())
        self.console.see("end")
        self.console.configure(state="disabled")

    def _set_status(self, text: str, color: str = TEXT) -> None:
        self.lbl_status.configure(text=text, text_color=color)

    def _hint(self) -> str:
        if not self.token:
            return f"Add HF_TOKEN=... to {core.CONFIG_PATH.parent / '.env'} and restart the app."
        if not self.audio_path:
            return "Choose an audio file to begin."
        if not self.output_dir:
            return "Choose where to save the results."
        return "Ready. Press Start Separation."

    def _ready(self) -> bool:
        return bool(self.token and self.audio_path and self.output_dir)

    def _refresh_ready(self) -> None:
        if self.busy:
            return
        self.btn_start.set_enabled(self._ready())
        self.btn_open.set_enabled(bool(self.output_dir and os.path.isdir(self.output_dir)))
        if self.last_result is None:
            self._set_status(self._hint(), TEXT if self._ready() else MUTED)

    def _keep_awake(self, on: bool) -> None:
        """Stop the Mac from idle-sleeping mid-job (a 20-minute file would otherwise stall)."""
        if sys.platform != "darwin":
            return
        if on and self._awake is None:
            try:
                # -i: no idle sleep; -w: exits by itself when this process exits
                self._awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            except OSError:
                self._awake = None
        elif not on and self._awake is not None:
            self._awake.terminate()
            self._awake = None

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self._keep_awake(busy)
        for b in (self.btn_audio, self.btn_out, self.btn_start,
                  self.seg_purity, self.seg_speakers, self.seg_layout, self.seg_fmt):
            b.set_enabled(not busy)
        self.btn_start.set_text("Working..." if busy else "Start Separation")
        if not busy:
            self._refresh_ready()

    # --------------------------------------------------------------- actions -- #
    def _pick_audio(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Select audio file",
            initialdir=self.cfg.get("audio_dir") or str(Path.home()),
            filetypes=[("Audio files", "*.mp3 *.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        self.audio_path = path
        self.cfg["audio_dir"] = str(Path(path).parent)
        p = Path(path)
        self.title_audio.configure(text=f"AUDIO FILE  ·  {human_size(p.stat().st_size)}")
        self.lbl_audio.configure(text=neu.shorten_middle(p.name, 24), text_color=TEXT)
        self._reset_run_state()
        self._refresh_ready()

    def _pick_output(self) -> None:
        if self.busy:
            return
        path = filedialog.askdirectory(title="Select output folder", initialdir=self.output_dir or str(Path.home()))
        if not path:
            return
        self.output_dir = path
        self.lbl_out.configure(text=neu.shorten_middle(path, 24), text_color=TEXT)
        self._reset_run_state()
        self._refresh_ready()

    def _open_output(self) -> None:
        if self.output_dir and os.path.isdir(self.output_dir):
            reveal(self.output_dir)

    def _reset_run_state(self) -> None:
        self.last_result = None
        self.target = self.shown = 0.0
        self.lbl_pct.configure(text="")
        self.stepper.set_stage(0, idle=True)

    def _start(self) -> None:
        if self.busy or not self._ready():
            return
        options = core.Options(purity=self.seg_purity.value, num_speakers=self.seg_speakers.value,
                               layout=self.seg_layout.value, fmt=self.seg_fmt.value)
        for stale in ("hf_token", "mode", "keep_timing"):   # settings from earlier versions
            self.cfg.pop(stale, None)
        self.cfg.update(output_dir=self.output_dir, purity=options.purity, speakers=options.num_speakers,
                        layout=options.layout, fmt=options.fmt)
        try:
            core.save_config(self.cfg)
        except OSError as exc:
            self._log(f"Could not save settings: {exc}", "dim")

        self._reset_run_state()
        self._set_busy(True)
        self.stage = 0
        self.stepper.set_stage(0)
        self._set_status("Starting...")
        self._log(f"Starting: {Path(self.audio_path).name}", "dim")

        self.worker = threading.Thread(
            target=self._worker_main,
            args=(self.audio_path, self.output_dir, self.token, options),
            name="diarization-worker",
            daemon=True,  # don't keep the process alive if the window is closed mid-run
        )
        self.worker.start()

    # -------------------------------------------------- worker thread (NO Tk) -- #
    def _worker_main(self, audio_path: str, out_dir: str, token: str, options: core.Options) -> None:
        def emit(kind, payload=None):
            self.events.put((kind, payload))

        try:
            core.separate(audio_path, out_dir, token, emit, options)
            emit("done")
        except Exception as exc:
            emit("log_err", traceback.format_exc())
            emit("error", str(exc))

    # ------------------------------------------------ UI thread: event pump -- #
    def _tick(self) -> None:
        try:
            while True:
                self._handle(*self.events.get_nowait())
        except queue.Empty:
            pass

        # Ease the displayed progress toward the reported one; sweep a highlight while busy.
        self.shown += (self.target - self.shown) * 0.18
        if abs(self.target - self.shown) < 0.002:
            self.shown = self.target
        if self.busy:
            self.phase = (self.phase + 0.025) % 1.0
        self.bar.render(self.shown, self.phase, animate=self.busy)
        self.after(TICK_MS, self._tick)

    def _handle(self, kind: str, payload) -> None:
        if kind == "log":
            self._log(payload)
        elif kind == "log_err":
            self._log(payload.rstrip(), "dim")
        elif kind == "status":
            self._set_status(payload)
        elif kind == "progress":
            self.target = max(self.target, min(1.0, payload))
            self.lbl_pct.configure(text=f"{int(self.target * 100)}%")
        elif kind == "stage":
            self.stage = payload
            self.stepper.set_stage(payload)
        elif kind == "device":
            self._set_pill(self.pill_device, *neu_device(payload))
        elif kind == "result":
            self.last_result = payload
        elif kind == "done":
            self.target = 1.0
            self.lbl_pct.configure(text="100%")
            self.stepper.set_stage(len(STEPS))
            self._set_busy(False)
            speakers = (self.last_result or {}).get("speakers", [])
            if speakers:
                n = len(speakers)
                total = sum(sp["seconds"] for sp in speakers)
                self._set_status(f"Done. {n} speaker{'s' if n != 1 else ''}, "
                                 f"{clock(total)} of clean speech saved.", GOOD)
                self._log(f"Finished: {n} speaker{'s' if n != 1 else ''} saved.", "ok")
            else:
                self._set_status("No clean single-speaker speech found. Try 'Normal' strictness.", WARN)
        elif kind == "error":
            self.stepper.set_stage(self.stage, error=True)
            self._set_busy(False)
            first_line = payload.strip().splitlines()[0] if payload.strip() else "Something went wrong."
            self._set_status(f"Failed: {neu.shorten_middle(first_line, 70)}", BAD)
            self._log("Failed. Details above.", "err")
            self.last_result = {}  # keep the error visible instead of the idle hint

    def _on_close(self) -> None:
        if self.busy and not messagebox.askyesno(
                core.APP_NAME, "A separation is still running. Quit anyway?"):
            return
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
