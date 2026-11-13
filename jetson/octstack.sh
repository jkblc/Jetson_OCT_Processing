#!/usr/bin/env bash
# Starts OCT Stack the way the desktop launcher does: the environment's Python,
# from the repository folder. A desktop launch has no terminal, so the output
# goes to a log file and a failure is shown in a dialog.
#
#   OCTSTACK_ENV   the Python environment (default ~/octstack-env)
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${OCTSTACK_ENV:-$HOME/octstack-env}"
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/octstack"
LOG="$LOG_DIR/app.log"
PY="$ENV_DIR/bin/python"

fail() {
    echo "$1" >&2
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --no-markup --title="OCT Stack" --width=480 --text="$1"
    elif command -v notify-send >/dev/null 2>&1; then
        notify-send "OCT Stack" "$1"
    fi
    exit 1
}

[ -x "$PY" ] || fail "There is no Python environment at $ENV_DIR. Run: bash jetson/install.sh"
cd "$REPO" || fail "Cannot open $REPO."

if [ -t 1 ]; then
    exec "$PY" -u app.py "$@"
fi
mkdir -p "$LOG_DIR"
"$PY" -u app.py "$@" >"$LOG" 2>&1
code=$?
[ "$code" -eq 0 ] || fail "OCT Stack stopped with an error (exit status $code).

The end of $LOG:

$(tail -n 15 "$LOG")"
