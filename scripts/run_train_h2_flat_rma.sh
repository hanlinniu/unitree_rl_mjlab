#!/usr/bin/env bash
# Train Unitree-H2-Flat-RMA (plane, no scandot, RMA). Run name includes git branch.
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
RUN_NAME="${1:-H2-Flat-RMA-${BRANCH}-${TS}}"
mkdir -p logs
echo "$RUN_NAME" > logs/h2_flat_rma_run_name.txt

python - <<'PY'
import src.tasks  # noqa
from mjlab.tasks.registry import load_env_cfg, load_runner_cls
from src.tasks.velocity.rma import RMAOnPolicyRunner

cfg = load_env_cfg("Unitree-H2-Flat-RMA", play=False)
assert cfg.scene.terrain is not None
assert cfg.scene.terrain.terrain_type == "plane", cfg.scene.terrain.terrain_type
assert "height_scan" not in cfg.observations["actor"].terms
assert not any(s.name == "terrain_scan" for s in (cfg.scene.sensors or ()))
assert "init_rma_buffers" in cfg.events
assert "randomize_rma_priv_latent" in cfg.events
assert load_runner_cls("Unitree-H2-Flat-RMA") is RMAOnPolicyRunner
print("OK: Unitree-H2-Flat-RMA plane, no scandot, RMA events + runner")
PY

echo "=============================================="
echo " Task:    Unitree-H2-Flat-RMA"
echo " Branch:  $BRANCH"
echo " Run:     $RUN_NAME"
echo " Envs:    4096"
echo " Logger:  wandb / humanoid"
echo "=============================================="

stdbuf -oL -eL python scripts/train.py Unitree-H2-Flat-RMA \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=humanoid \
  --agent.run-name="$RUN_NAME" \
  --agent.resume=False
