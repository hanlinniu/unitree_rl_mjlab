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
python scripts/train.py Custom-R1-Flat-RMA --env.scene.num-envs=4096 --agent.logger=tensorboard
```
