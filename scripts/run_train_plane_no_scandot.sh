#!/usr/bin/env bash
# Plane terrain, NO height-scan (scandot) input.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

TASK="${1:-Custom-R1-Flat}"
RUN_NAME="${2:-${TASK}-plane-no-scandot-$(date '+%Y%m%d-%H%M%S')}"

python - <<PY
import src.tasks  # noqa
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("$TASK", play=False)
assert cfg.scene.terrain is not None
assert cfg.scene.terrain.terrain_type == "plane", cfg.scene.terrain.terrain_type
assert "height_scan" not in cfg.observations["actor"].terms
assert not any(s.name == "terrain_scan" for s in (cfg.scene.sensors or ()))
print("OK: plane terrain, no scandot/height_scan")
PY

echo "Training $TASK (plane, no scandot) as $RUN_NAME"
stdbuf -oL -eL python scripts/train.py "$TASK" \
  --env.scene.num-envs=4096 \
  --agent.logger=tensorboard \
  --agent.run-name="$RUN_NAME" \
  --agent.resume=False
