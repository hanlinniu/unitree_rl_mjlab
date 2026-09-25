# Recover training runs from Weights & Biases

When you train with `--agent.logger=wandb`, every checkpoint save uploads a
**durable Artifact** containing:

- `model_XXXX.pt` (weights + optimizer)
- `params/env.yaml`, `params/agent.yaml`, `params/run_meta.json`
- `policy.onnx` (when exported)

If the Vast / cloud instance dies, restore on another machine:

```bash
python scripts/download_wandb_run.py \
  --run ENTITY/PROJECT/RUN_ID \
  --out logs/rsl_rl/<experiment_name>/restored_from_wandb
```

Find `ENTITY/PROJECT/RUN_ID` in the W&B run URL:
`https://wandb.ai/<entity>/<project>/runs/<run_id>`.

Resume:

```bash
python scripts/train.py <TASK> \
  --agent.resume=True \
  --agent.load-run=restored_from_wandb \
  --agent.load-checkpoint=model_XXXX.pt \
  --agent.logger=wandb
```

## Copy a run off the instance before it dies

```bash
python scripts/download_wandb_run.py pack \
  --log-dir logs/rsl_rl/<experiment>/<run_dir> \
  --out /tmp/run_backup.tar.gz
# scp /tmp/run_backup.tar.gz to your PC, then:
tar xzf run_backup.tar.gz -C logs/rsl_rl/<experiment>/restored_local
```
