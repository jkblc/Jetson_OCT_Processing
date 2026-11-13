"""OCT reconstruction: a raw spectral cube to a log10 B-scan stack.

The chain, per frame: Butterworth low-pass DC removal, PCHIP resampling onto a
uniform optical-frequency grid, background subtraction, dispersion kernel
(C2, C3), Gaussian window, FFT. `Reconstructor` does it on the GPU (CuPy) when
one works and on the CPU (NumPy) otherwise, with the same code in the same
float64 precision, so the TIFF does not depend on which machine made it.

`write_stack` streams a whole acquisition to a multi-page TIFF. app.py is the
window over this module; nothing here needs a display.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import tifffile
from scipy.fft import fft as scipy_fft
import scipy.io
from scipy.signal import butter, filtfilt, windows

from dispersion_tuning import C2_RANGE, C3_RANGE, renyi2_entropy, two_stage_search


# The tuner prints H₂ and →, which a legacy console code page cannot encode.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


class UnusableFile(Exception):
    """A file that cannot be used, said in plain words."""


# ──────────────────────────────────────────────────────────────────────────────
# The system and the reconstruction
# ──────────────────────────────────────────────────────────────────────────────
NUM_A             : int            = 2048
NUM_K             : int            = 2048
LAMBDA_CAL_FILE   : Path           = Path("Lin_loc.mat")

# CPU threads for the per-frame compute (unused on the GPU). The PCHIP
# resampling and the per-A-line transform are independent across A-lines, so a
# frame is split into column chunks run in parallel; the two whole-frame means
# (DC and background) are still taken over the full frame, so the result is
# bit-identical to one thread. None uses every core.
CPU_THREADS       : int | None     = None

N_FFT             : int            = NUM_K
N_DEPTH           : int            = N_FFT // 2    # positive-delay rows kept

# Automatic dispersion tuning scores a coarse grid then polishes with
# Nelder-Mead (~100-160 evaluations), on a subset of the preview frame's
# A-lines: dispersion is a property of the system, identical on every A-line,
# so scoring every Nth one finds the same minimum for 1/N of the cost.
TUNE_A_LINE_STRIDE   : int  = 4
# Depth rows skipped at zero delay, by the tuner and by auto levels. The DC
# residual there does not depend on dispersion and is the brightest thing in
# the image: it would dominate the entropy and flatten the minimum, and it
# would set the white point.
DC_CROP_ROWS         : int  = 8

# Subtract the across-A-line mean of the resampled spectrum — the part common
# to every A-line: the DC-removal residual and any fixed-pattern ripple. It
# removes the haze band at the top of the image that the full-length FFT
# exposed (~10 dB on RawBuffer5) and the deep horizontal stripes, for ~0.7 dB
# on the retina. The one real thing it also dims is a flat layer uniform across
# the whole B-scan. False reproduces the images made before this was added.
# The app's checkbox overrides it per run.
SUBTRACT_BACKGROUND  : bool = True

# Auto levels for the preview windows. On, the black point sits
# AUTO_LEVELS_BLACK_SIGMA noise σ above the noise floor — ~88% of pure-noise
# pixels render black, while weak structure just above the floor stays visible
# — and the white point at the AUTO_LEVELS_WHITE_PCT percentile, so only the
# brightest 0.1% saturate. The noise floor is read from the darkest
# AUTO_LEVELS_NOISE_ROWS of the rows wherever they are, so a sample that fills
# the bottom of the frame does not get mistaken for noise. Off, LOG_RANGE is
# used. DISPLAY ONLY: the TIFF stores the unclipped log10 image either way.
AUTO_LEVELS             : bool  = True
AUTO_LEVELS_BLACK_SIGMA : float = 1.0
AUTO_LEVELS_WHITE_PCT   : float = 99.9
AUTO_LEVELS_NOISE_ROWS  : float = 0.10
GAUSS_STD_ALPHA   : float          = 1.6
BUTTER_N, BUTTER_CUTOFF = 3, 0.05
LOG_RANGE         : Tuple[float, float] = (1, 4)


def load_wavelength(path: str | Path, column: int = 1) -> np.ndarray:
    mat = scipy.io.loadmat(path, squeeze_me=True,
                           struct_as_record=False, simplify_cells=True)
    def _v(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, np.float64).ravel() * 1e-9

    for key in ("b", "y"):
        if key in mat and isinstance(mat[key], np.ndarray) and mat[key].size > 100:
            return _v(mat[key])

    if "L" in mat:
        L = mat["L"]
        if isinstance(L, dict):
            cols = list(L.keys())
            if len(cols) > column:
                return _v(L[cols[column]])
        arr = np.asarray(L)
        if arr.dtype.names and len(arr.dtype.names) > column:
            return _v(arr[arr.dtype.names[column]])
        if arr.ndim == 2 and arr.shape[1] > column:
            return _v(arr[:, column])

    for val in mat.values():
        arr = np.asarray(val)
        if arr.ndim == 2 and arr.shape[1] > column and arr.size > 100:
            return _v(arr[:, column])

    raise UnusableFile("Wavelength vector not found in Lin_loc.mat")

def gaussian_window(m: int, alpha: float) -> np.ndarray:
    std = (m - 1) / (2.0 * alpha)
    return windows.gaussian(m, std, sym=False)

def dispersion_kernel(n: int, c2: float, c3: float, fc: float = 0.0, xp=np):
    # The grid is built by NumPy on either backend, so it is identical on both.
    fdd = xp.asarray(np.linspace(0.0, 1.0, n, dtype=np.float64))
    return xp.exp(-1j * (c2 * (fdd - fc) ** 2 + c3 * (fdd - fc) ** 3))

def linear_freq_grid(wl: np.ndarray) -> np.ndarray:
    c0 = 3.0e8
    f = c0 / wl
    if f[0] > f[-1]:
        f = f[::-1]
    return np.linspace(f[0], f[-1], len(f))

def open_cube(path: str | Path, k: int, a: int) -> Tuple[np.ndarray, int]:
    """Memory-map the raw cube: frames are read from disk as they are used.

    Replaces reading the whole file up front (2 GB for 250 frames), so the
    tuner starts as soon as the one frame it needs is in, and memory holds a
    frame or two rather than the whole acquisition. Frame t of an F-ordered
    (k, a, n) cube is one contiguous block of the file, so each read is a
    single sequential ~8 MB read.
    """
    n_values = Path(path).stat().st_size // np.dtype(np.uint16).itemsize
    per = k * a
    if n_values % per:
        raise UnusableFile(f"{path}: its size is not a whole number of {k} × {a} frames")
    n = n_values // per
    return np.memmap(path, dtype=np.uint16, mode="r", shape=(k, a, n), order="F"), n

def load_frame(cube: np.ndarray, idx: int) -> np.ndarray:
    """Read one frame off the memory map into memory, in row (C) order.

    The file holds a frame A-line by A-line; the reconstruction works along k
    on whole rows of A-lines, which is several times faster on a row-ordered
    array. The reordering is done here, by whichever thread reads the frame.
    """
    return np.array(cube[:, :, idx], order="C")

# ──────────────────────────────────────────────────────────────────────────────
# Backend
# ──────────────────────────────────────────────────────────────────────────────
def select_backend(use_gpu: bool = True):
    """The array module to compute with: numpy, or cupy when asked and usable."""
    if not use_gpu:
        print("Backend: CPU (use_gpu is False)")
        return np
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device")
        # Touch a kernel and cuFFT now, so a machine with CuPy installed but no
        # working GPU fails here, with a message, rather than mid-stack.
        cp.fft.fft(cp.arange(8, dtype=cp.complex128)).sum().item()
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        name = name.decode() if isinstance(name, bytes) else name
        print(f"Backend: GPU ({name})")
        return cp
    except Exception as e:
        print(f"Backend: CPU — use_gpu is True but no GPU is usable "
              f"({type(e).__name__}: {e})")
        return np

# ──────────────────────────────────────────────────────────────────────────────
# Frame processing
# ──────────────────────────────────────────────────────────────────────────────
def _pchip_edge(h0, h1, m0, m1, xp):
    """One-sided three-point end slope, shape-preserving (scipy's rule)."""
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    wrong_sign = xp.sign(d) != xp.sign(m0)
    overshoot = (xp.sign(m0) != xp.sign(m1)) & (xp.abs(d) > 3.0 * xp.abs(m0))
    return xp.where(wrong_sign, 0.0, xp.where(overshoot, 3.0 * m0, d))

class Reconstructor:
    """Raw spectral frame → OCT image, on NumPy or CuPy.

    The original chain: Butterworth low-pass DC removal, PCHIP resampling onto
    a uniform optical-frequency grid, optional background subtraction,
    dispersion kernel, Gaussian window, FFT. One implementation for both
    backends, so a CPU run and a GPU run cannot drift apart.

    PCHIP is implemented here rather than taken from scipy, because there is
    no CuPy equivalent to rely on. It is scipy.interpolate.PchipInterpolator's
    interpolant (Fritsch-Carlson interior slopes, the same edge rule), and
    agrees with it to rounding (1e-15 of full scale; test_recon.py).
    Both grids are fixed by the calibration, so everything that depends only
    on them — the interval search, the slope weights, the cubic's four basis
    weights at each target point — is worked out once here, and a frame costs
    the slopes and four weighted row gathers.

    Split at the one seam that matters: `prepare` (DC + resampling) does not
    depend on C2/C3, so the tuner and the sliders prepare a frame once and
    re-run only `transform`. A prepared frame is real (float64); it becomes
    complex when the dispersion kernel is applied.

    On the CPU the per-A-line work runs on column chunks in a thread pool
    (NumPy releases the GIL inside its array operations); see CPU_THREADS.
    """

    def __init__(self, wl: np.ndarray, xp=np,
                 subtract_background: bool = SUBTRACT_BACKGROUND,
                 threads: int | None = CPU_THREADS):
        self.xp = xp
        self.subtract_background = subtract_background
        self.on_gpu = xp is not np
        self.to_host = getattr(xp, "asnumpy", np.asarray)
        self.threads = 1 if self.on_gpu else max(1, threads or os.cpu_count() or 1)
        self._pool = ThreadPoolExecutor(self.threads) if self.threads > 1 else None
        if self.on_gpu:
            self._fft = lambda a: xp.fft.fft(a, N_FFT, axis=0)
        else:
            # One FFT thread per chunk when the pool already spreads the chunks
            # over the cores; all cores when there is no pool.
            workers = 1 if self._pool else -1
            self._fft = lambda a: scipy_fft(a, N_FFT, axis=0, workers=workers)

        f = 3.0e8 / np.asarray(wl, dtype=np.float64)
        self.flip = bool(f[0] > f[-1])
        if self.flip:
            f = f[::-1]
        flin = linear_freq_grid(wl)
        # Interval holding each target point, found the way scipy's PPoly does.
        idx = np.clip(np.searchsorted(f, flin, side="right") - 1, 0, f.size - 2)
        h = np.diff(f)
        col = lambda v: xp.asarray(v)[:, None]               # (n, 1): one value per row
        self._h = col(h)                                     # (K-1, 1), for the end slopes
        self._inv_h = col(1.0 / h)
        # Interior slopes: the weighted harmonic mean of the two secants,
        # m0·m1 / (a·m1 + b·m0), with a + b = 1 fixed by the spacing.
        w1 = 2 * h[1:] + h[:-1]
        w2 = h[1:] + 2 * h[:-1]
        self._wa, self._wb = col(w1 / (w1 + w2)), col(w2 / (w1 + w2))
        # The cubic on interval i at t = (x - x_i)/h_i, in Hermite form:
        #   y_i·H00 + y_i+1·H01 + d_i·H10 + d_i+1·H11   (h_i folded into H10, H11)
        t = (flin - f[idx]) / h[idx]
        self._idx, self._idx1 = xp.asarray(idx), xp.asarray(idx + 1)
        self._H00 = col(2 * t**3 - 3 * t**2 + 1)
        self._H01 = col(-2 * t**3 + 3 * t**2)
        self._H10 = col((t**3 - 2 * t**2 + t) * h[idx])
        self._H11 = col((t**3 - t**2) * h[idx])
        self._window = xp.asarray(gaussian_window(NUM_K, GAUSS_STD_ALPHA))[:, None]
        self._butter = butter(BUTTER_N, BUTTER_CUTOFF, btype="low")

    def _by_columns(self, fn, a, rows: int, dtype):
        """fn(a) — computed on column chunks in parallel when there is a pool.

        Only for work that is independent across A-lines: every column's
        result is then exactly what the unsplit call gives it. Each worker
        writes its chunk straight into one preallocated (rows, A) output, so
        reassembly costs nothing serial.
        """
        if self._pool is None:
            return fn(a)
        n = a.shape[1]
        out = np.empty((rows, n), dtype)
        bounds = np.linspace(0, n, min(self.threads, n) + 1).astype(int)

        def work(lo, hi):
            out[:, lo:hi] = fn(a[:, lo:hi])
        list(self._pool.map(work, bounds[:-1], bounds[1:]))
        return out

    def _pchip(self, y):
        """PCHIP-resample `y` (K, A), ascending in f, onto the uniform grid."""
        xp, h = self.xp, self._h
        m = y[1:] - y[:-1]
        m *= self._inv_h                                     # secant slopes
        m0, m1 = m[:-1], m[1:]

        # Zero slope at an extremum or beside a flat segment (the secants'
        # product is not positive); their weighted harmonic mean elsewhere.
        # The substitution only avoids a 0-division in entries the second
        # `where` then discards.
        d = xp.empty_like(y)
        p = m0 * m1
        rising = p > 0
        den = self._wa * m1
        den += self._wb * m0
        xp.divide(p, xp.where(rising, den, 1.0), out=p)
        d[1:-1] = xp.where(rising, p, 0.0)
        d[0] = _pchip_edge(h[0], h[1], m[0], m[1], xp)
        d[-1] = _pchip_edge(h[-1], h[-2], m[-1], m[-2], xp)

        out = xp.take(y, self._idx, axis=0)
        out *= self._H00
        for src, index, weight in ((y, self._idx1, self._H01), (d, self._idx, self._H10),
                                   (d, self._idx1, self._H11)):
            part = xp.take(src, index, axis=0)
            part *= weight
            out += part
        return out

    def prepare(self, fringe: np.ndarray):
        """DC removal + PCHIP resampling (+ background) → float64.

        The c2/c3-independent half of `process` — and the expensive half.
        `fringe` must be a WHOLE frame: the background is a mean over its
        A-lines, and a subset would get a different one.
        """
        xp = self.xp
        fr = xp.asarray(fringe)
        # The two whole-frame means are the only steps that span A-lines; each
        # is taken over the full frame, and everything between and after them
        # runs per column chunk. 2048-point DC filter: on the host either way.
        dc = xp.asarray(filtfilt(*self._butter, self.to_host(fr.mean(1)), axis=0))[:, None]

        def resample(cols):
            # Row order first: the resampling slices and gathers whole rows.
            y = xp.ascontiguousarray(cols) - dc
            return self._pchip(y[::-1, :] if self.flip else y)
        r = self._by_columns(resample, fr, NUM_K, np.float64)

        if self.subtract_background:
            # AFTER resampling, not before: PCHIP is nonlinear, so the mean of
            # the resampled A-lines is not the resampled mean. Everything
            # downstream is linear and identical on every A-line, so removing
            # the common part here removes it from the image exactly.
            r -= r.mean(axis=1, keepdims=True)
        return r

    def _shaping(self, c2: float, c3: float, window=None):
        """Dispersion kernel × apodization window, as one (K, 1) complex column."""
        kernel = dispersion_kernel(NUM_K, c2, c3, xp=self.xp)[:, None]
        if window is None:
            return kernel * self._window
        return kernel * self.xp.asarray(window, dtype=self.xp.float64)[:, None]

    def _field(self, fr, shaping):
        # One pass: real prepared frame × (kernel × window) → complex, out of
        # place (`fr` is a cached prepared frame, reused per call).
        # Bins 0..N_DEPTH-1 are the positive delays: the same rows the old
        # fftshift(...)[N/2:] selected, without shifting the whole array.
        return self._fft(fr * shaping)[:N_DEPTH]

    def _magnitude(self, fr, shaping):
        return self.xp.abs(self._field(fr, shaping))

    def _log_image(self, fr, shaping):
        xp = self.xp
        mag = self._magnitude(fr, shaping)
        xp.maximum(mag, 1e-12, out=mag)
        xp.log10(mag, out=mag)
        return mag.astype(xp.float32)

    def transform(self, fr, c2: float, c3: float, window=None):
        """Dispersion + window + FFT on a prepared frame → linear magnitude.

        `window` (length NUM_K) replaces the Gaussian apodization.
        """
        shaping = self._shaping(c2, c3, window)
        return self._by_columns(lambda c: self._magnitude(c, shaping), fr, N_DEPTH, np.float64)

    def field(self, fr, c2: float, c3: float):
        """Dispersion + window + FFT on a prepared frame → the complex field.

        What `transform` takes the magnitude of, phase included. Its phase
        grows with depth.
        """
        shaping = self._shaping(c2, c3)
        return self._by_columns(lambda c: self._field(c, shaping), fr, N_DEPTH, np.complex128)

    def image(self, fr, c2: float, c3: float) -> np.ndarray:
        """Prepared frame → log10 image on the host, as written to the TIFF."""
        shaping = self._shaping(c2, c3)
        return self.to_host(self._by_columns(lambda c: self._log_image(c, shaping), fr,
                                             N_DEPTH, np.float32))

    def process(self, fringe: np.ndarray, c2: float, c3: float) -> np.ndarray:
        return self.image(self.prepare(fringe), c2, c3)

def noise_floor(img: np.ndarray, dc_crop: int = DC_CROP_ROWS) -> Tuple[float, float]:
    """(median, robust σ) of a log10 image's noise, read from its darkest rows.

    The darkest AUTO_LEVELS_NOISE_ROWS of the rows wherever they are, so a
    sample that fills the bottom of the frame is not mistaken for noise.
    """
    body = img[dc_crop:]
    n_rows = max(1, int(AUTO_LEVELS_NOISE_ROWS * body.shape[0]))
    noise = body[np.argsort(np.median(body, axis=1))[:n_rows]]
    floor = float(np.median(noise))
    return floor, 1.4826 * float(np.median(np.abs(noise - floor)))   # robust σ (MAD)

def auto_levels(img: np.ndarray, dc_crop: int = DC_CROP_ROWS) -> Tuple[float, float]:
    """Display (black, white) for a log10 OCT image. See AUTO_LEVELS."""
    floor, sigma = noise_floor(img, dc_crop)
    black = floor + AUTO_LEVELS_BLACK_SIGMA * sigma
    white = float(np.percentile(img[dc_crop:], AUTO_LEVELS_WHITE_PCT))
    if not white > black:           # a blank or saturated frame: nothing to fit
        return LOG_RANGE
    return black, white

def display_levels(img: np.ndarray) -> Tuple[float, float]:
    return auto_levels(img) if AUTO_LEVELS else LOG_RANGE

def auto_tune_dispersion(recon: Reconstructor, frame: np.ndarray,
                         a_line_stride: int = TUNE_A_LINE_STRIDE,
                         dc_crop: int = DC_CROP_ROWS) -> Tuple[float, float]:
    """(C2A, C3A) minimising the Rényi-2 entropy of this frame's image.

    The frame is prepared WHOLE and subsampled afterwards, so the DC estimate
    — a mean over A-lines — is the one the TIFF will actually use, and the
    tuner scores a subset of exactly the A-lines it will write. The entropy is
    summed on the host on either backend, so both find the same minimum.
    """
    xp = recon.xp
    prepared = xp.ascontiguousarray(recon.prepare(frame)[:, ::a_line_stride])

    def cost(c2: float, c3: float) -> float:
        mag = recon.transform(prepared, c2, c3)[dc_crop:]
        return renyi2_entropy(recon.to_host(mag ** 2))

    return two_stage_search(cost, C2_RANGE, C3_RANGE)

# ──────────────────────────────────────────────────────────────────────────────
# The whole stack
# ──────────────────────────────────────────────────────────────────────────────
def write_stack(cube, n: int, recon: Reconstructor, c2: float, c3: float, out_tiff: Path,
                progress: Optional[Callable[[int, np.ndarray], None]] = None,
                stop: Optional[Callable[[], bool]] = None) -> Tuple[int, float]:
    """Write every frame of `cube` as a log10 float32 page of `out_tiff`.

    A streaming pipeline: while frame i is computed, frame i+1 is read off the
    memory map and page i-1 is written, with one TiffWriter open for the whole
    stack. BigTIFF only when a classic TIFF's 4 GB limit would be hit (~500
    frames). `progress(i, image)` is called after each frame; `stop()` true
    ends the run early. Returns (frames written, seconds).
    """
    out_tiff = Path(out_tiff)
    out_tiff.parent.mkdir(parents=True, exist_ok=True)
    bigtiff = n * N_DEPTH * NUM_A * 4 > 2**32 - 2**26
    t_start = time.perf_counter()
    written = 0
    # Exit runs bottom-up: the reader stops, the writer drains its last page,
    # and only then is the TIFF closed — also when something fails midway.
    with tifffile.TiffWriter(out_tiff, bigtiff=bigtiff) as tiff,          ThreadPoolExecutor(1, thread_name_prefix="tiff") as writer,          ThreadPoolExecutor(1, thread_name_prefix="read") as reader:
        next_frame = reader.submit(load_frame, cube, 0)
        pending = None
        for idx in range(n):
            if stop is not None and stop():
                break
            fringe = next_frame.result()
            if idx + 1 < n:
                next_frame = reader.submit(load_frame, cube, idx + 1)
            image = recon.process(fringe, c2, c3)
            if pending is not None:
                pending.result()     # at most one page in flight; re-raises a write error
            pending = writer.submit(tiff.write, image, photometric="minisblack", metadata=None)
            written += 1
            if progress is not None:
                progress(idx, image)
        if pending is not None:
            pending.result()
    return written, time.perf_counter() - t_start
