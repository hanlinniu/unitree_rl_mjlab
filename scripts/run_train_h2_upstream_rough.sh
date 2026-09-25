#!/usr/bin/env bash
# Train Unitree-H2-Rough exactly as in unitreerobotics/unitree_rl_mjlab.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

RUN_NAME="${1:-H2-Rough-upstream-$(date '+%Y%m%d-%H%M%S')}"
echo "$RUN_NAME" > logs/h2_rough_run_name.txt
mkdir -p logs

python - <<'PY'
import src.tasks  # noqa
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Unitree-H2-Rough", play=False)
assert cfg.scene.terrain is not None
assert cfg.scene.terrain.terrain_generator is not None
# Upstream: full rough curriculum terrains (not stairs-only).
subs = set(cfg.scene.terrain.terrain_generator.sub_terrains)
print("OK Unitree-H2-Rough terrains:", sorted(subs))
print("track_lin_w", cfg.rewards["track_linear_velocity"].weight)
print("foot_clearance_w", cfg.rewards["foot_clearance"].weight)
print("stand_still_w", cfg.rewards["stand_still"].weight)
PY

echo "Training Unitree-H2-Rough (upstream method) as $RUN_NAME"
stdbuf -oL -eL python scripts/train.py Unitree-H2-Rough \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=humanoid \
  --agent.run-name="$RUN_NAME" \
  --agent.resume=False
