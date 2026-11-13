# OCT Stack

One window that turns a raw spectral-domain OCT acquisition (`.raw`) into a
multi-page TIFF of log10 B-scans, with a live preview for setting the
dispersion correction.

```bash
python app.py                 # or: python app.py path/to/scan.raw
```

## Using it

1. **Raw file → Browse…** The middle frame appears, with the dispersion
   auto-tuned (the sharpest image, by Rényi-2 entropy).
2. **Check the preview.** Move **C2** and **C3** and watch reflectors sharpen
   or blur; the arrow keys move a slider one step (0.1). Move **Frame** to
   check other frames. Click the image to see it pixel for pixel around that
   point; click again for the whole frame. The grey levels are fixed when a
   frame is first shown and held while the sliders move, so brightness changes
   are real.
3. **Write TIFF stack.** Every frame is written with the values shown. The
   preview follows every fifth frame as it is written. **Stop** ends the run
   and keeps nothing.

Output, by default beside the raw file:

| File | What it is |
|---|---|
| `oct_output/<name>.tiff` | float32 log10 magnitude, one page per frame, 1024 × 2048 (depth × A-lines) |
| `oct_output/<name>_settings.json` | the raw file, calibration, C2/C3 and background setting used |

**Subtract background** removes the part of the spectrum common to every
A-line (fixed-pattern stripes and the haze at the top of the image). Turn it
off for a sample that is one flat layer across the whole B-scan.

**Calibration** is the spectrometer's wavelength file. `Lin_loc.mat` in this
folder is the default; browse to another to replace it.

## Installing on a Jetson

```bash
sudo apt install git curl
curl -LsSf https://astral.sh/uv/install.sh | sh     # then open a new terminal
git clone <this repository> oct-stack
bash oct-stack/jetson/install.sh
```

The installer creates a Python 3.12 environment in `~/octstack-env` (JetPack
5's own Python 3.8 is too old), installs NumPy, SciPy, tifffile and Pillow,
tries the CuPy package that matches the board's CUDA (11 on JetPack 5, 12 on
JetPack 6), and adds **OCT Stack** to the applications menu and the desktop.
Run it again after moving the folder. To use an environment that already
exists: `OCTSTACK_ENV=~/ambientoce-env bash jetson/install.sh`.

The app uses the GPU when CuPy works and the CPU otherwise; the bottom-left
corner of the window says which. Both give the same TIFF. A launch from the
desktop writes its output to `~/.local/state/octstack/app.log`.

The installer has been syntax-checked but not yet run on a Jetson.

## On any other machine

Python 3.10 or newer:

```bash
pip install -r requirements.txt
python app.py
```

Add `cupy-cuda12x` (or the package for your CUDA) to use an NVIDIA GPU.

## Files

| File | |
|---|---|
| `app.py` | the window |
| `recon.py` | the reconstruction and the stack writer; no display needed |
| `dispersion_tuning.py` | the auto-tune search |
| `Lin_loc.mat` | wavelength calibration |
| `test_recon.py` | checks on synthetic data: `python test_recon.py` |
| `jetson/` | `jetson/install.sh` and the launcher it installs |

The acquisition format is fixed in `recon.py`: 2048 spectral samples × 2048
A-lines per frame, `uint16`, frames one after another. The frame count comes
from the file's size.
