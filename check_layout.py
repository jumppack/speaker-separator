"""Layout checker for the live Tk window: sibling overlaps, children outside parents, clipped text.

Run after ANY UI change:  python check_layout.py   (needs only customtkinter + Pillow; no model/token)
Expect "0 problem(s)" in every state. Screen capture is unavailable to the agent, so this is the layout test."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main, core, neu
import customtkinter as ctk
from customtkinter.windows.widgets.core_widget_classes import CTkBaseClass

core.load_token = lambda: "hf_fake"; core.save_config = lambda c: None
app = main.App(); app.update()

def name(w):
    t = ""
    try: t = w.cget("text")
    except Exception: pass
    return f"{type(w).__name__}({t[:24]!r})" if t else type(w).__name__

def box(w): return (w.winfo_rootx(), w.winfo_rooty(), w.winfo_rootx() + w.winfo_width(), w.winfo_rooty() + w.winfo_height())
def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 1 and h > 1 else 0

def kids(w): return [c for c in w.winfo_children() if isinstance(c, CTkBaseClass) and c.winfo_ismapped() and getattr(c, "_is_surface", False) is False]

problems = []
def walk(w):
    ks = [c for c in kids(w) if c is not getattr(w, "_surface", None)]
    pb = box(w)
    for c in ks:
        b = box(c)
        if b[0] < pb[0] - 1 or b[1] < pb[1] - 1 or b[2] > pb[2] + 1 or b[3] > pb[3] + 1:
            problems.append(f"OUTSIDE  {name(c)} {b} not inside {name(w)} {pb}")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a = inter(box(ks[i]), box(ks[j]))
            if a: problems.append(f"OVERLAP  {name(ks[i])} {box(ks[i])}  x  {name(ks[j])} {box(ks[j])}  ({a}px2)")
    for c in ks:
        if isinstance(c, ctk.CTkLabel) and c.cget("text") and not c.cget("image"):
            need = c._label.winfo_reqwidth(); have = c.winfo_width()
            if need > have + 2: problems.append(f"CLIPPED  {name(c)} needs {need}px, has {have}px")
        walk(c)

def check(label):
    app.update(); problems.clear(); walk(app)
    h = app.winfo_height(); sh = app.winfo_screenheight()
    print(f"[{label}] window {app.winfo_width()}x{h} (screen {app.winfo_screenwidth()}x{sh}) -> {len(problems)} problem(s)")
    for p in problems[:25]: print("   ", p)

check("idle")
# worst-case content
app.title_audio.configure(text="AUDIO FILE  ·  1023.9 MB")
app.lbl_audio.configure(text=neu.shorten_middle("karan_raj_20_final_interview_take_two.MP3", 24), text_color=main.TEXT)
app.lbl_out.configure(text=neu.shorten_middle("/Users/someone/projects/voice-training/speaker-samples", 24), text_color=main.TEXT)
app.audio_path = "x"; app.output_dir = os.path.expanduser("~"); app._refresh_ready()
for seg, v in ((app.seg_purity, 1), (app.seg_layout, 1), (app.seg_fmt, 1), (app.seg_speakers, 3)): seg._click(v)
check("long names + all options changed")
app._set_busy(True); app._set_status("Analyzing audio... (embeddings 85/112)"); app.lbl_pct.configure(text="77%")
check("busy")
app._set_busy(False); app._set_status("Done. 3 speakers, 12:34 of clean speech saved.", main.GOOD)
check("done")
for s in ("Failed: Could not load pyannote/speaker-diarization-3.1: 401 Client Error", "No clean single-speaker speech found. Try 'Normal' strictness."):
    app._set_status(s); check("status: " + s[:30])
