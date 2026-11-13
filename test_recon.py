"""Checks of the reconstruction and the stack writer on synthetic data. CPU only, a few seconds.

    python test_recon.py
"""
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from scipy.interpolate import PchipInterpolator

import recon as R

passed = 0


def check(name, cond, detail=""):
    global passed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")
    assert cond, name
    passed += 1


wl = R.load_wavelength(Path(__file__).with_name(str(R.LAMBDA_CAL_FILE)))
rec = R.Reconstructor(wl, np, threads=1)
rng = np.random.default_rng(1)

f = 3.0e8 / wl
f = f[::-1] if rec.flip else f
y = rng.integers(-3, 4, (R.NUM_K, 48)).astype(float)      # steps, sign changes, flat runs, zeros
y[100:140] = 0
y[500:520] = 7
with np.errstate(all="raise"):
    got = rec._pchip(y)
want = PchipInterpolator(f, y, axis=0)(R.linear_freq_grid(wl))
err = float(np.abs(got - want).max() / np.abs(want).max())
check("the resampling is scipy's PCHIP, to rounding", err < 1e-12, f"{err:.1e} of full scale")

# A reflector at a known depth: a fringe that is a pure tone on the uniform
# frequency grid, sampled where the spectrometer samples it.
depth = 300
flin = R.linear_freq_grid(wl)
phase = 2 * np.pi * depth * (f - flin[0]) / (flin[-1] - flin[0]) * (R.NUM_K - 1) / R.NUM_K
tone = 2000 + 800 * np.cos(phase)
fringe = np.repeat((tone[::-1] if rec.flip else tone)[:, None], R.NUM_A, axis=1)
fringe = (fringe + rng.normal(0, 5, fringe.shape)).astype(np.uint16)
flat = R.Reconstructor(wl, np, subtract_background=False)   # a reflector on every A-line IS background
img = flat.process(fringe, 0.0, 0.0)
peak = int(np.argmax(img[R.DC_CROP_ROWS:].mean(axis=1))) + R.DC_CROP_ROWS
check("a reflector comes out at its depth", peak == depth, f"row {peak}, expected {depth}")
check("the image is the stack's size and type", img.shape == (R.N_DEPTH, R.NUM_A) and img.dtype == np.float32)
blurred = flat.process(fringe, 60.0, 0.0)
check("a wrong dispersion blurs it (the peak falls)", blurred[depth].mean() < img[depth].mean() - 0.3,
      f"log10 peak {img[depth].mean():.2f} -> {blurred[depth].mean():.2f}")
c2, c3 = R.auto_tune_dispersion(flat, fringe)
check("auto-tune finds the dispersion that was put in (none)", abs(c2) < 3 and abs(c3) < 8, f"C2 {c2:+.2f}, C3 {c3:+.2f}")

noise = rng.integers(0, 4096, (R.NUM_K, R.NUM_A)).astype(np.uint16)
one = rec.process(noise, 40.0, -5.0)
check("every CPU thread count gives the same image",
      np.array_equal(one, R.Reconstructor(wl, np, threads=3).process(noise, 40.0, -5.0)))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    frames = rng.integers(0, 4096, (R.NUM_K, R.NUM_A, 3)).astype(np.uint16)
    (tmp / "tiny.raw").write_bytes(frames.tobytes(order="F"))
    cube, n = R.open_cube(tmp / "tiny.raw", R.NUM_K, R.NUM_A)
    seen = []
    written, _ = R.write_stack(cube, n, rec, 12.5, -3.0, tmp / "out" / "tiny.tiff",
                               progress=lambda i, im: seen.append(i))
    stack = tifffile.imread(tmp / "out" / "tiny.tiff")
    check("the stack has a page per frame", n == 3 and written == 3 and stack.shape == (3, R.N_DEPTH, R.NUM_A)
          and seen == [0, 1, 2], f"{stack.shape}")
    check("each page is the reconstruction of its frame",
          np.array_equal(stack[1], rec.process(np.ascontiguousarray(frames[:, :, 1]), 12.5, -3.0)))
    written, _ = R.write_stack(cube, n, rec, 12.5, -3.0, tmp / "out" / "stopped.tiff", stop=lambda: True)
    check("a stop ends the run before the next frame", written == 0)
    (tmp / "bad.raw").write_bytes(b"\0" * 1000)
    try:
        R.open_cube(tmp / "bad.raw", R.NUM_K, R.NUM_A); refused = False
    except R.UnusableFile:
        refused = True
    check("a file that is not whole frames is refused in plain words", refused)
    del cube

print(f"\n{passed}/{passed} passed")
