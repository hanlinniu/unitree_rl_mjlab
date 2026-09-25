#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab

export DISPLAY="${DISPLAY:-:1}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export WANDB_USERNAME=niuhanlin-rl
export WANDB_ENTITY=niuhanlin-rl

RUN_NAME="${1:-H2-Rough-stairs-uneven-standstillL2-$(date '+%Y%m%d-%H%M%S')}"
echo "$RUN_NAME" > logs/h2_rough_run_name.txt
mkdir -p logs

echo "=============================================="
echo " Task:       Unitree-H2-Rough"
echo " Run:        $RUN_NAME"
echo " Terrains:   stairs + uneven (no slopes/flat)"
echo " Rewards:    stand_still L2 @ -1.0"
echo " Envs:       4096 (local)"
echo "=============================================="

python - <<'PY'
import inspect
import src.tasks  # noqa
from src.tasks.velocity.config.h2.env_cfgs import unitree_h2_rough_env_cfg

cfg = unitree_h2_rough_env_cfg(play=False)
sub = cfg.scene.terrain.terrain_generator.sub_terrains
print("Terrain mix:")
for k, v in sub.items():
  print(f"  {k}: proportion={v.proportion}")
assert "hf_pyramid_slope" not in sub
assert "hf_pyramid_slope_inv" not in sub
assert "flat" not in sub
assert "pyramid_stairs" in sub and "random_rough" in sub
assert "wave_terrain" in sub
ss = cfg.rewards["stand_still"]
src = inspect.getsource(ss.func)
assert "torch.square" in src and "torch.abs(diff_angle)" not in src
assert ss.weight == -1.0
print(f"stand_still: weight={ss.weight} L2=OK module={ss.func.__module__}")
print("OK: stairs + uneven + L2 stand_still")
PY

LOG=logs/h2_rough_stairs_uneven_train_local.log
stdbuf -oL -eL python scripts/train.py Unitree-H2-Rough \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=humanoid \
  --agent.run-name="$RUN_NAME" \
  --agent.resume=False \
  2>&1 | tee "$LOG"

echo
read -r -p 'Training finished/stopped. Press Enter to close...'
