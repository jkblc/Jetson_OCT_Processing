#!/usr/bin/env bash
# Installs OCT Stack on a Jetson (or any Linux desktop):
#   1. a Python environment with what the app needs (created if missing),
#   2. CuPy for the GPU, when a matching package installs (the app runs on the
#      CPU without it),
#   3. "OCT Stack" in the applications menu and on the desktop.
#
#   bash jetson/install.sh
#
#   OCTSTACK_ENV   the Python environment to use or create (default ~/octstack-env).
#                  An existing one is reused, e.g. OCTSTACK_ENV=~/ambientoce-env.
#
# Safe to run again: after moving the repository, or to retry CuPy.
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
ENV_DIR="${OCTSTACK_ENV:-$HOME/octstack-env}"
PY="$ENV_DIR/bin/python"

# ── 1. the environment ───────────────────────────────────────────────────────
if [ ! -x "$PY" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 "$ENV_DIR"
    else
        # JetPack 5's own Python is 3.8, too old for current NumPy and SciPy.
        python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
            echo "The system Python is older than 3.10. Install uv, open a new terminal, and run this again:" >&2
            echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
            exit 1
        }
        python3 -m venv "$ENV_DIR"
    fi
fi

install() {     # install packages into the environment, with uv or pip
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PY" "$@"
    else
        "$PY" -m pip install "$@"
    fi
}

install -r "$REPO/requirements.txt"
"$PY" -c "import tkinter" 2>/dev/null || {
    echo "This Python has no Tk. Install it (sudo apt install python3-tk), or use uv's Python:" >&2
    echo "    rm -rf \"$ENV_DIR\" && curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}

# ── 2. CuPy, for the GPU ─────────────────────────────────────────────────────
gpu_works() {
    "$PY" -c "import cupy as cp; cp.fft.fft(cp.arange(8, dtype=cp.complex128)).sum().item()" >/dev/null 2>&1
}
if gpu_works; then
    echo "CuPy already works in $ENV_DIR."
else
    # The CUDA version JetPack installed: 11.x on JetPack 5, 12.x on JetPack 6.
    CUDA="$(readlink -f /usr/local/cuda 2>/dev/null | sed -n 's/.*cuda-\([0-9]*\).*/\1/p')"
    case "$CUDA" in
        11) PACKAGE="cupy-cuda11x<14" ;;
        12) PACKAGE="cupy-cuda12x" ;;
        *)  PACKAGE="" ;;
    esac
    if [ -z "$PACKAGE" ]; then
        echo "No CUDA 11 or 12 found under /usr/local/cuda: CuPy not installed. The app will run on the CPU."
    elif install "$PACKAGE" && gpu_works; then
        echo "CuPy installed: the app will run on the GPU."
    else
        echo "CuPy ($PACKAGE) did not install or does not run. The app will run on the CPU." >&2
        echo "To build it for this board instead (slow), with the environment active:" >&2
        echo "    CUPY_NVCC_GENERATE_CODE=\"arch=compute_87,code=sm_87\" PATH=/usr/local/cuda/bin:\$PATH pip install \"cupy<14\"" >&2
    fi
fi

# ── 3. the launcher ──────────────────────────────────────────────────────────
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
APPS="$DATA/applications"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
NAME=octstack.desktop
chmod +x "$HERE/octstack.sh"
mkdir -p "$APPS"
cat >"$APPS/$NAME" <<ENTRY
[Desktop Entry]
Type=Application
Name=OCT Stack
Comment=Raw OCT spectra to a TIFF stack, with a dispersion preview
Exec=env "OCTSTACK_ENV=$ENV_DIR" /bin/bash "$HERE/octstack.sh"
Path=$REPO
Icon=applications-science
Terminal=false
Categories=Science;
StartupNotify=true
ENTRY
chmod +x "$APPS/$NAME"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true
echo "Added to the applications menu: $APPS/$NAME"
if [ -d "$DESKTOP" ]; then
    cp "$APPS/$NAME" "$DESKTOP/$NAME"
    chmod +x "$DESKTOP/$NAME"
    # GNOME launches a desktop file only once it is marked trusted.
    command -v gio >/dev/null 2>&1 && gio set "$DESKTOP/$NAME" metadata::trusted true 2>/dev/null || true
    echo "Added to the desktop: $DESKTOP/$NAME"
    echo "If the icon shows as a text file, right-click it and choose \"Allow Launching\"."
fi
echo "Done. Start it from the menu, or: bash \"$HERE/octstack.sh\""
