#!/usr/bin/env bash
# Rough terrain WITH height-scan (scandot) input.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

TASK="${1:-Custom-R1-Rough}"
RUN_NAME="${2:-${TASK}-rough-with-scandot-$(date '+%Y%m%d-%H%M%S')}"

python - <<PY
import src.tasks  # noqa
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("$TASK", play=False)
assert cfg.scene.terrain is not None
assert cfg.scene.terrain.terrain_generator is not None, "expected rough terrain generator"
assert "height_scan" in cfg.observations["actor"].terms
assert any(s.name == "terrain_scan" for s in (cfg.scene.sensors or ()))
print("OK: rough terrain, scandot/height_scan present")
print("Terrains:", list(cfg.scene.terrain.terrain_generator.sub_terrains.keys()))
PY

echo "Training $TASK (rough, with scandot) as $RUN_NAME"
stdbuf -oL -eL python scripts/train.py "$TASK" \
  --env.scene.num-envs=4096 \
  --agent.logger=tensorboard \
  --agent.run-name="$RUN_NAME" \
  --agent.resume=False
