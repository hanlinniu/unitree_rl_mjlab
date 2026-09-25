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

CKPT_RUN=2026-09-24_15-53-37_R1-Rough-scanfix-20260924-155331
RUN_NAME="${1:-R1-Rough-stairs-uneven-from3000-$(date '+%Y%m%d-%H%M%S')}"
echo "$RUN_NAME" > logs/custom_r1_rough_run_name.txt
mkdir -p logs

echo "=============================================="
echo " Run:        $RUN_NAME"
echo " Resume:     $CKPT_RUN / model_3000.pt"
echo " Terrains:   stairs + uneven (no slopes/flat)"
echo "=============================================="

# Verify terrain mix before training
python - <<'PY'
import src.tasks  # noqa
from src.tasks.velocity.config.custom_r1.env_cfgs import custom_r1_rough_env_cfg
cfg = custom_r1_rough_env_cfg(play=False)
sub = cfg.scene.terrain.terrain_generator.sub_terrains
print("Terrain mix:")
for k, v in sub.items():
  print(f"  {k}: proportion={v.proportion}")
assert "hf_pyramid_slope" not in sub
assert "hf_pyramid_slope_inv" not in sub
assert "flat" not in sub
assert "pyramid_stairs" in sub and "random_rough" in sub
print("OK: stairs + uneven only")
PY

stdbuf -oL -eL python scripts/train.py Custom-R1-Rough \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=humanoid \
  --agent.run-name="$RUN_NAME" \
  --agent.resume=True \
  --agent.load-run="$CKPT_RUN" \
  --agent.load-checkpoint=model_3000.pt \
  2>&1 | tee logs/custom_r1_rough_train_local.log

echo
read -r -p 'Training finished/stopped. Press Enter to close...'
