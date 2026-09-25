#!/usr/bin/env bash
# Pop up a terminal showing live left/right joystick stick commands.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/joystick_stick_monitor.py"
export DISPLAY="${DISPLAY:-:1}"

if ! ls /dev/input/js* >/dev/null 2>&1; then
  echo "No joystick found under /dev/input/js*. Plug it in and retry."
  exit 1
fi

PYTHON=python3
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  if conda activate unitree_rl_mjlab 2>/dev/null; then
    PYTHON="$(command -v python)"
  fi
fi

# Pass through extra args to the Python monitor (e.g. --rx 2 --ry 3).
RUN_CMD=$(printf '%q ' "$PYTHON" "$SCRIPT" "$@")

if command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal --title="Joystick L/R sticks" --geometry=90x28 -- bash -lc \
    "cd $(printf %q "$ROOT"); ${RUN_CMD}; echo; read -r -p 'Press Enter to close...'"
elif command -v xterm >/dev/null 2>&1; then
  xterm -T "Joystick L/R sticks" -geometry 90x28 -e bash -lc \
    "cd $(printf %q "$ROOT"); ${RUN_CMD}; echo; read -r -p 'Press Enter to close...'"
else
  echo "No gnome-terminal/xterm; running in current terminal."
  cd "$ROOT"
  exec "$PYTHON" "$SCRIPT" "$@"
fi
