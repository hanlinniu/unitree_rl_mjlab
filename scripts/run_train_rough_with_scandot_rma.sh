#!/usr/bin/env bash
# Rough terrain + scandot + RMA (stairs / uneven / slope).
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

TASK="${1:-Custom-R1-Rough-RMA}"
BRANCH=$(git branch --show-current 2>/dev/null || echo unknown)
TS=$(date '+%Y%m%d-%H%M%S')
RUN_NAME="${2:-${TASK}-rough-scandot-RMA-${BRANCH}-${TS}}"
mkdir -p logs
echo "$RUN_NAME" > logs/rough_scandot_rma_run_name.txt

python - <<PY
import src.tasks  # noqa
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.velocity.rma import RMAOnPolicyRunner

cfg = load_env_cfg("${TASK}", play=False)
assert cfg.scene.terrain is not None
assert cfg.scene.terrain.terrain_type != "plane"
assert "height_scan" in cfg.observations["actor"].terms
assert any(s.name == "terrain_scan" for s in (cfg.scene.sensors or ()))
keys = set(cfg.scene.terrain.terrain_generator.sub_terrains.keys())
assert "flat" not in keys
assert "pyramid_stairs" in keys and "hf_pyramid_slope" in keys
assert "random_rough" in keys
assert "init_rma_buffers" in cfg.events
rl = load_rl_cfg("${TASK}")
assert int(rl.rma["num_scan"]) > 0
assert rl.rma_policy.get("scan_encoder_dims") == [128, 64, 32]
assert load_runner_cls("${TASK}") is RMAOnPolicyRunner
print("OK:", "${TASK}", "num_scan=", rl.rma["num_scan"], "terrains=", sorted(keys))
PY

echo "=============================================="
echo " Task:    $TASK"
echo " Branch:  $BRANCH"
echo " Run:     $RUN_NAME"
echo " Envs:    4096"
echo "=============================================="

stdbuf -oL -eL python scripts/train.py "$TASK" \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=humanoid \
  --agent.run-name="$RUN_NAME" \
  --agent.resume=False
