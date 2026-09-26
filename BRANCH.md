# Branch: `rough-with-scandot-RMA`

Rough terrain **with height-scan (scandot)** + **RMA**, following
[extreme-parkour](https://github.com/chengxuxin/extreme-parkour) main:

- Actor obs: `[proprio | scan | priv_explicit(9) | priv_latent | history]`
- `scan_encoder_dims = [128, 64, 32]`
- Terrains: **stairs** + **uneven** (`random_rough` / `wave`) + **slope**
  (`hf_pyramid_slope` / inv); no flat

Matched Rough-RMA recipe (R1 & H2): heading_command=True with rel_heading_envs=0.90 (10% direct wz), vx/vy/wz=(-1,2)/(-1,1)/(-1,1), 5% stand, track lin/ang=3.0, fall=-200, upright package, foot_clearance=-1, stairs max riser 0.18 m, action scale 0.25.

## Tasks
- `Custom-R1-Rough-RMA`
- `Unitree-H2-Rough-RMA`

(Plane / no-scandot RMA tasks `*-Flat-RMA` still work with `num_scan=0`.)

## Train
```bash
bash scripts/run_train_rough_with_scandot_rma.sh
# or:
python scripts/train.py Custom-R1-Rough-RMA --env.scene.num-envs=4096 --agent.logger=wandb
python scripts/train.py Unitree-H2-Rough-RMA --env.scene.num-envs=4096 --agent.logger=wandb
```

## Recover from W&B
```bash
python scripts/download_wandb_run.py \
  --run ENTITY/PROJECT/RUN_ID \
  --out logs/rsl_rl/custom_r1_rough_rma/restored_from_wandb

python scripts/train.py Custom-R1-Rough-RMA \
  --agent.resume=True \
  --agent.load-run=restored_from_wandb \
  --agent.load-checkpoint=model_XXXX.pt \
  --agent.logger=wandb
```
