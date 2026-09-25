# Branch: `plane-no-scandot-RMA`

Plane / flat terrain training **without** height-scan (scandot), with **RMA**
(Rapid Motor Adaptation) from [extreme-parkour](https://github.com/chengxuxin/extreme-parkour):

- `priv_encoder` — encodes privileged latent (mass, friction, motor strength)
- `history_encoder` — estimates the same latent from proprio history (`hist_latent`)
- Velocity / `priv_explicit` estimator (base linear velocity, padded to 9-D)
- PPO updates with `priv_reg` (teacher) + dagger (student history encoder)

## Observation layout (actor)
```
[proprio | priv_explicit(9) | priv_latent | history(history_len * proprio)]
```
`num_scan = 0` (no scandot). Critic keeps the standard privileged Flat obs.

## Train
```bash
bash scripts/run_train_plane_no_scandot_rma.sh
# or:
python scripts/train.py Custom-R1-Flat-RMA --env.scene.num-envs=4096 --agent.logger=wandb
```

## Recover training from W&B (instance died)
Training uploads durable **Artifacts** (configs + each `model_*.pt` + `policy.onnx`)
whenever `--agent.logger=wandb`.

```bash
# On a new instance or your PC:
python scripts/download_wandb_run.py \
  --run ENTITY/PROJECT/RUN_ID \
  --out logs/rsl_rl/custom_r1_flat_rma/restored_from_wandb

# Resume (copy/rename folder to match load-run if needed):
python scripts/train.py Custom-R1-Flat-RMA \
  --agent.resume=True \
  --agent.load-run=restored_from_wandb \
  --agent.load-checkpoint=model_XXXX.pt \
  --agent.logger=wandb
```

Pack a local run before leaving an instance:
```bash
python scripts/download_wandb_run.py pack \
  --log-dir logs/rsl_rl/<experiment>/<run_dir> \
  --out /tmp/run_backup.tar.gz
```
