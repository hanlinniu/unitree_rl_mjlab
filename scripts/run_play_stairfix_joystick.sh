#!/usr/bin/env bash
# Play latest R1-Rough-stairfix-from3000 policy with joystick twist control.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DISPLAY="${DISPLAY:-:1}"
export MUJOCO_GL=glfw

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab
cd "$ROOT"

exec python scripts/play_stairfix_joystick.py "$@"
