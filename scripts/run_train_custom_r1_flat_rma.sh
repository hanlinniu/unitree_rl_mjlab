#!/usr/bin/env bash
# Train Custom-R1-Flat-RMA with H2-matched standstill recipe (plane, no scandot, RMA).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export WANDB_USERNAME="${WANDB_USERNAME:-niuhanlin-rl}"
export WANDB_ENTITY="${WANDB_ENTITY:-niuhanlin-rl}"
unset WANDB_MODE WANDB_DISABLED 2>/dev/null || true

BRANCH=$(git branch --show-current 2>/dev/null || echo unknown)
TS=$(date '+%Y%m%d-%H%M%S')
RUN_NAME="${1:-R1-Flat-RMA-standstill-${BRANCH}-${TS}}"
mkdir -p logs
echo "$RUN_NAME" > logs/custom_r1_flat_rma_run_name.txt

python - <<'PY'
import src.tasks  # noqa
from mjlab.tasks.registry import load_env_cfg, load_runner_cls
from src.tasks.velocity.rma import RMAOnPolicyRunner

cfg = load_env_cfg("Custom-R1-Flat-RMA", play=False)
assert cfg.scene.terrain is not None
assert cfg.scene.terrain.terrain_type == "plane", cfg.scene.terrain.terrain_type
assert "height_scan" not in cfg.observations["actor"].terms
assert not any(s.name == "terrain_scan" for s in (cfg.scene.sensors or ()))
assert "init_rma_buffers" in cfg.events
assert "randomize_rma_priv_latent" in cfg.events
assert cfg.rewards["stand_still"].weight == -1.0
assert cfg.commands["twist"].rel_standing_envs >= 0.10
assert cfg.commands["twist"].heading_command is True
assert cfg.commands["twist"].rel_heading_envs == 0.90
assert cfg.rewards["track_angular_velocity"].weight == 2.0
assert load_runner_cls("Custom-R1-Flat-RMA") is RMAOnPolicyRunner
print("OK: Custom-R1-Flat-RMA plane, standstill, 10% direct-wz, RMA runner")
PY

echo "=============================================="
echo " Task:    Custom-R1-Flat-RMA"
echo " Branch:  $BRANCH"
echo " Run:     $RUN_NAME"
echo " Envs:    4096"
echo " Logger:  wandb / humanoid"
echo "=============================================="

stdbuf -oL -eL python scripts/train.py Custom-R1-Flat-RMA \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=humanoid \
  --agent.run-name="$RUN_NAME" \
  --agent.resume=False
