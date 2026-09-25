# Branch: `plane-no-scandot`

Plane / flat terrain training **without** height-scan (scandot) observations.

## What this means
- Uses `*-Flat` tasks (e.g. `Custom-R1-Flat`, `Unitree-H2-Flat`)
- No `terrain_scan` sensor and no `height_scan` in actor/critic obs
- Terrain is a flat plane

## Train
```bash
bash scripts/run_train_plane_no_scandot.sh
# or:
python scripts/train.py Custom-R1-Flat --env.scene.num-envs=4096
```
