"""OCT Stack: one window that turns a raw spectral cube into a TIFF stack.

Choose a raw file. The middle frame is shown, auto-tuned for dispersion. Move
the C2 and C3 sliders and watch the preview sharpen or blur; move the frame
slider to check other frames; click the preview to look at it pixel for pixel.
"Write TIFF stack" then writes every frame with the values shown, and the
preview follows the frames as they are written.

The reconstruction is recon.py; this file is only the window. Everything that
computes runs on one background thread, so the window stays responsive and the
GPU is never used from two places at once.
"""
from __future__ import annotations

import json
import sys
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:                                    # the tuner prints H₂ and →
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import numpy as np
from PIL import Image, ImageTk

import recon as R

HERE = Path(__file__).resolve().parent
PREVIEW_EVERY = 5           # while writing, show every fifth frame


def to_photo(img, levels, size, zoom_at=None):
    """A log10 image as a Tk photo `size` = (w, h): the whole frame stretched
    to fit, or with zoom_at = (row, column) the pixels around that point 1:1."""
    w, h = size
    if zoom_at is not None:
        z0 = int(np.clip(zoom_at[0] - h // 2, 0, max(img.shape[0] - h, 0)))
        x0 = int(np.clip(zoom_at[1] - w // 2, 0, max(img.shape[1] - w, 0)))
        img = img[z0:z0 + h, x0:x0 + w]
    lo, hi = levels
    grey = np.clip((img - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    picture = Image.fromarray(grey)
    if zoom_at is None:
        picture = picture.resize((w, h), Image.BILINEAR)
    return ImageTk.PhotoImage(picture)


class App:
    def __init__(self, root):
        self.root = root
        root.title("OCT Stack")
        root.geometry("%dx%d" % (min(1400, root.winfo_screenwidth() - 60),
                                 min(900, root.winfo_screenheight() - 110)))
        root.minsize(900, 600)
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="compute")
        self.xp = None                  # the backend, chosen on the compute thread
        self.recon = None
        self.cube, self.n = None, 0
        self.prepared, self.prepared_key = None, None
        self.img, self.levels = None, None
        self.zoom_at = None
        self.rendering = self.dirty = False
        self.writing = False
        self.stop = threading.Event()
        self.progress = None            # (frames done, latest image) from the writer
        self._photo = None

        self.raw = tk.StringVar()
        self.cal = tk.StringVar(value=str(HERE / R.LAMBDA_CAL_FILE))
        self.out = tk.StringVar()
        self.frame = tk.IntVar(value=1)          # 1-based, as shown and as TIFF pages are counted
        self.c2 = tk.DoubleVar(value=0.0)
        self.c3 = tk.DoubleVar(value=0.0)
        self.background = tk.BooleanVar(value=R.SUBTRACT_BACKGROUND)
        self.status = tk.StringVar(value="Choose a raw file.")
        self.caption = tk.StringVar(value="")
        self.backend = tk.StringVar(value="")
        self._build()
        for var in (self.frame, self.c2, self.c3, self.background):
            var.trace_add("write", lambda *_: self.request_preview())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.run(lambda: R.select_backend(True), self.backend_ready)

    # ── the window ──────────────────────────────────────────────────────
    def _build(self):
        root = self.root
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        files = ttk.Frame(root, padding=(10, 10, 10, 4))
        files.grid(row=0, column=0, columnspan=2, sticky="ew")
        files.columnconfigure(1, weight=1)
        rows = (("Raw file", self.raw, self.browse_raw, self.open_raw),
                ("Calibration", self.cal, self.browse_cal, self.open_raw),
                ("Output TIFF", self.out, self.browse_out, None))
        for i, (label, var, browse, on_enter) in enumerate(rows):
            ttk.Label(files, text=label).grid(row=i, column=0, sticky="w", pady=2)
            entry = ttk.Entry(files, textvariable=var)
            entry.grid(row=i, column=1, sticky="ew", padx=8, pady=2)
            if on_enter:
                entry.bind("<Return>", lambda e, f=on_enter: f())
            ttk.Button(files, text="Browse…", command=browse).grid(row=i, column=2, pady=2)

        side = ttk.Frame(root, padding=(10, 4, 4, 4))
        side.grid(row=1, column=0, sticky="ns")
        self.sliders = []
        for label, var, lo, hi, step in (("Frame", self.frame, 1, 1, 1),
                                         ("Dispersion C2", self.c2, *R.C2_RANGE, 0.1),
                                         ("Dispersion C3", self.c3, *R.C3_RANGE, 0.1)):
            head = ttk.Frame(side)
            head.pack(fill="x", pady=(8, 0))
            ttk.Label(head, text=label).pack(side="left")
            box = ttk.Spinbox(head, textvariable=var, from_=lo, to=hi, increment=step, width=8)
            box.pack(side="right")
            scale = tk.Scale(side, variable=var, from_=lo, to=hi, resolution=step, orient="horizontal",
                             showvalue=False, length=260)
            scale.pack(fill="x")
            self.sliders.append((scale, box))
        ttk.Label(side, text="Arrow keys on a slider move it one step.", foreground="#777").pack(
            anchor="w", pady=(2, 0))
        self.tune_btn = ttk.Button(side, text="Auto-tune dispersion", command=self.auto_tune)
        self.tune_btn.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(side, text="Subtract background", variable=self.background).pack(
            anchor="w", pady=(10, 0))
        ttk.Separator(side).pack(fill="x", pady=14)
        self.write_btn = ttk.Button(side, text="Write TIFF stack", command=self.write)
        self.write_btn.pack(fill="x")
        self.stop_btn = ttk.Button(side, text="Stop", command=self.stop.set, state="disabled")
        self.stop_btn.pack(fill="x", pady=(4, 0))
        self.bar = ttk.Progressbar(side, mode="determinate")
        self.bar.pack(fill="x", pady=(10, 0))
        ttk.Label(side, textvariable=self.backend, foreground="#777").pack(anchor="w", side="bottom")

        view = ttk.Frame(root, padding=(4, 4, 10, 4))
        view.grid(row=1, column=1, sticky="nsew")
        view.rowconfigure(0, weight=1)
        view.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(view, background="black", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self.show())
        self.canvas.bind("<Button-1>", self.toggle_zoom)
        ttk.Label(view, textvariable=self.caption, foreground="#444").grid(row=1, column=0, sticky="w",
                                                                           pady=(4, 0))
        ttk.Label(root, textvariable=self.status, padding=(10, 2, 10, 8)).grid(
            row=2, column=0, columnspan=2, sticky="ew")

    def set_controls(self, on: bool):
        state = "normal" if on else "disabled"
        for scale, box in self.sliders:
            scale.configure(state=state)
            box.configure(state=state)
        self.tune_btn.configure(state=state)
        self.write_btn.configure(state=state)

    # ── the compute thread ──────────────────────────────────────────────
    def run(self, work, done):
        """work() on the compute thread, then done(result) back in the window."""
        future = self.pool.submit(work)

        def poll():
            if not future.done():
                self.root.after(25, poll)
                return
            try:
                result = future.result()
            except Exception as e:          # noqa: BLE001 - shown to the user
                self.failed(e)
                return
            done(result)
        poll()

    def failed(self, e):
        self.rendering = self.dirty = False
        if self.writing:
            self.write_ended()
        self.set_controls(True)
        self.status.set("Stopped: %s" % e)
        messagebox.showerror("OCT Stack", str(e) if isinstance(e, R.UnusableFile)
                             else "%s: %s" % (type(e).__name__, e))

    def backend_ready(self, xp):
        self.xp = xp
        self.backend.set("Computing on the %s" % ("GPU" if xp is not np else "CPU (no usable GPU)"))
        if self.raw.get().strip():
            self.open_raw()

    # ── files ───────────────────────────────────────────────────────────
    def start_folder(self) -> str:
        raw = self.raw.get().strip()
        return str(Path(raw).parent) if raw and Path(raw).parent.is_dir() else str(Path.home())

    def browse_raw(self):
        path = filedialog.askopenfilename(title="Raw file", initialdir=self.start_folder(),
                                          filetypes=[("Raw OCT data", "*.raw"), ("All files", "*.*")])
        if path:
            self.raw.set(path)
            self.open_raw()

    def browse_cal(self):
        path = filedialog.askopenfilename(title="Wavelength calibration", initialdir=str(HERE),
                                          filetypes=[("MATLAB file", "*.mat"), ("All files", "*.*")])
        if path:
            self.cal.set(path)
            self.open_raw()

    def browse_out(self):
        path = filedialog.asksaveasfilename(title="Output TIFF", initialdir=self.start_folder(),
                                            defaultextension=".tiff",
                                            filetypes=[("TIFF stack", "*.tiff *.tif")])
        if path:
            self.out.set(path)

    def open_raw(self):
        """Open the raw file with the calibration, then auto-tune on its middle frame."""
        if self.writing or self.xp is None:
            return
        raw, cal = Path(self.raw.get().strip()), Path(self.cal.get().strip())
        if not raw.is_file():
            self.status.set("The raw file was not found.")
            return
        if not cal.is_file():
            self.status.set("The calibration file was not found.")
            return
        self.status.set("Opening %s…" % raw.name)

        def work():
            cube, n = R.open_cube(raw, R.NUM_K, R.NUM_A)
            return cube, n, R.Reconstructor(R.load_wavelength(cal), self.xp)

        def done(result):
            self.cube, self.n, self.recon = result
            self.prepared_key = self.levels = self.zoom_at = None
            self.sliders[0][0].configure(to=self.n)
            self.sliders[0][1].configure(to=self.n)
            self.out.set(str(raw.parent / "oct_output" / (raw.stem + ".tiff")))
            self.frame.set(self.n // 2 + 1)
            self.auto_tune()
        self.run(work, done)

    # ── the preview ─────────────────────────────────────────────────────
    def values(self):
        """(frame index from 0, c2, c3, background) as set, or None while a box holds half-typed text."""
        try:
            frame = int(np.clip(self.frame.get() - 1, 0, max(self.n - 1, 0)))
            return frame, float(self.c2.get()), float(self.c3.get()), bool(self.background.get())
        except (tk.TclError, ValueError):
            return None

    def request_preview(self):
        """Redraw with the current values: at once, or as soon as the draw in hand ends."""
        if self.recon is None or self.writing:
            return
        self.dirty = True
        if not self.rendering:
            self.render()

    def render(self):
        values = self.values()
        self.dirty = False
        if values is None:
            return
        frame, c2, c3, background = values
        self.rendering = True
        cube, recon = self.cube, self.recon

        def work():
            key = (id(cube), frame, background)
            if key != self.prepared_key:        # a new frame: the slow half, once
                recon.subtract_background = background
                self.prepared = recon.prepare(R.load_frame(cube, frame))
                self.prepared_key = key
                self.levels = None
            return recon.image(self.prepared, c2, c3)

        def done(img):
            self.rendering = False
            self.img = img
            # Levels are fitted when a frame is first shown and then HELD while
            # the sliders move: re-levelling every draw would partly hide the
            # sharpness differences the sliders are there to show.
            if self.levels is None:
                self.levels = R.display_levels(img)
            self.caption.set("Frame %d of %d   ·   C2 %+.1f   C3 %+.1f   ·   click the image to %s"
                             % (frame + 1, self.n, c2, c3,
                                "see the whole frame" if self.zoom_at else "see it pixel for pixel"))
            self.show()
            if self.status.get().endswith("…"):
                self.status.set("Ready. Adjust C2 and C3 until the image is sharpest, then write the stack.")
            if self.dirty:
                self.render()
        self.run(work, done)

    def show(self):
        if self.img is None or self.levels is None:
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 10 or h < 10:
            return
        self._photo = to_photo(self.img, self.levels, (w, h), self.zoom_at)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def toggle_zoom(self, event):
        if self.img is None:
            return
        if self.zoom_at is None:
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            self.zoom_at = (int(event.y / h * self.img.shape[0]), int(event.x / w * self.img.shape[1]))
        else:
            self.zoom_at = None
        self.show()
        self.request_preview()                  # refreshes the caption

    def auto_tune(self):
        values = self.values()
        if self.recon is None or self.writing or values is None:
            return
        frame, _, _, background = values
        cube, recon = self.cube, self.recon
        self.status.set("Auto-tuning dispersion on frame %d…" % (frame + 1))
        self.set_controls(False)

        def work():
            recon.subtract_background = background
            return R.auto_tune_dispersion(recon, R.load_frame(cube, frame))

        def done(found):
            self.set_controls(True)
            self.c2.set(round(found[0], 1))
            self.c3.set(round(found[1], 1))
            self.status.set("Auto-tuned: C2 %+.1f, C3 %+.1f. Check the image and adjust if needed, then write the stack." % found)
            self.request_preview()
        self.run(work, done)

    # ── writing the stack ───────────────────────────────────────────────
    def write(self):
        values = self.values()
        if self.recon is None or self.writing or values is None:
            return
        _, c2, c3, background = values
        out = Path(self.out.get().strip())
        if out.suffix.lower() not in (".tif", ".tiff"):
            messagebox.showerror("OCT Stack", "The output must end in .tif or .tiff.")
            return
        if out.exists() and not messagebox.askyesno("OCT Stack", "%s exists. Replace it?" % out.name):
            return
        cube, n, recon = self.cube, self.n, self.recon
        # Read here, not on the compute thread: Tk's variables belong to the window's thread.
        settings = {"raw_file": self.raw.get().strip(), "calibration": self.cal.get().strip(),
                    "dispersion": [c2, c3], "subtract_background": background, "frames": n}
        self.writing, self.progress, self.zoom_at = True, None, None
        self.stop.clear()
        self.set_controls(False)
        self.stop_btn.configure(state="normal")
        self.bar.configure(maximum=n, value=0)
        self.status.set("Writing %d frames…" % n)

        def note(idx, image):
            self.progress = (idx + 1, image if idx % PREVIEW_EVERY == 0 else None)

        def work():
            recon.subtract_background = background
            self.prepared_key = None            # the preview's frame is no longer the one prepared
            written, seconds = R.write_stack(cube, n, recon, c2, c3, out, note, self.stop.is_set)
            if written < n:
                out.unlink()                    # a stopped run leaves no half stack behind
            else:
                out.with_name(out.stem + "_settings.json").write_text(json.dumps(settings, indent=2))
            return written, seconds

        def done(result):
            written, seconds = result
            self.write_ended()
            if written < n:
                self.status.set("Stopped after %d of %d frames. Nothing was kept." % (written, n))
            else:
                self.bar.configure(value=n)
                self.status.set("Done: %d frames in %.1f s (%.0f ms per frame). Saved %s"
                                % (n, seconds, 1e3 * seconds / n, out))
            self.request_preview()
        self.run(work, done)
        self.follow()

    def follow(self):
        """While writing: the bar, and every fifth frame in the preview as it is written."""
        if not self.writing:
            return
        if self.progress is not None:
            count, image = self.progress
            self.bar.configure(value=count)
            if image is not None and self.levels is not None:
                self.img = image
                self.caption.set("Writing: frame %d of %d" % (count, self.n))
                self.show()
        self.root.after(100, self.follow)

    def write_ended(self):
        self.writing = False
        self.set_controls(True)
        self.stop_btn.configure(state="disabled")

    def close(self):
        if self.writing and not messagebox.askyesno("OCT Stack", "A stack is being written. Stop and quit?"):
            return
        self.stop.set()
        self.pool.shutdown(wait=False)
        self.root.destroy()


def main():
    if sys.platform.startswith("win"):          # sharp text on a scaled Windows display
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    app = App(root)
    if len(sys.argv) > 1:                       # python app.py scan.raw
        app.raw.set(sys.argv[1])
    root.mainloop()


if __name__ == "__main__":
    main()
