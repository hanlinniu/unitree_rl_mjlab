# Branch: `rough-with-scandot`

Rough terrain training **with** height-scan (scandot) observations.

## What this means
- Uses `*-Rough` tasks (e.g. `Custom-R1-Rough`, `Unitree-H2-Rough`)
- Actor/critic include `height_scan` from `terrain_scan` ray sensor
- Stairs + uneven terrains (Custom-R1 / H2 configs)

## Train
```bash
bash scripts/run_train_rough_with_scandot.sh
# or:
python scripts/train.py Custom-R1-Rough --env.scene.num-envs=4096
python scripts/train.py Unitree-H2-Rough --env.scene.num-envs=4096
```

## Recover training from W&B (instance died)
See [docs/WANDB_RECOVERY.md](docs/WANDB_RECOVERY.md). Train with `--agent.logger=wandb` so configs + checkpoints upload as durable Artifacts.
