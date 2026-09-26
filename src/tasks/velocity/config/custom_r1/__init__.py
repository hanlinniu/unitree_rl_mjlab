from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from src.tasks.velocity.rma import RMAOnPolicyRunner
from src.tasks.velocity.rma.scan_utils import terrain_scan_num_rays_from_cfg

from .env_cfgs import (
  custom_r1_flat_env_cfg,
  custom_r1_flat_rma_env_cfg,
  custom_r1_rough_env_cfg,
  custom_r1_rough_rma_env_cfg,
)
from .rl_cfg import (
  custom_r1_ppo_runner_cfg,
  custom_r1_rma_ppo_runner_cfg,
  custom_r1_rough_rma_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Custom-R1-Rough",
  env_cfg=custom_r1_rough_env_cfg(),
  play_env_cfg=custom_r1_rough_env_cfg(play=True),
  rl_cfg=custom_r1_ppo_runner_cfg(experiment_name="custom_r1_rough_velocity"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Custom-R1-Flat",
  env_cfg=custom_r1_flat_env_cfg(),
  play_env_cfg=custom_r1_flat_env_cfg(play=True),
  rl_cfg=custom_r1_ppo_runner_cfg(experiment_name="custom_r1_flat_velocity"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Custom-R1-Flat-RMA",
  env_cfg=custom_r1_flat_rma_env_cfg(),
  play_env_cfg=custom_r1_flat_rma_env_cfg(play=True),
  rl_cfg=custom_r1_rma_ppo_runner_cfg(experiment_name="custom_r1_flat_rma"),
  runner_cls=RMAOnPolicyRunner,
)

_rough_rma_env = custom_r1_rough_rma_env_cfg()
_rough_rma_num_scan = terrain_scan_num_rays_from_cfg(_rough_rma_env)
register_mjlab_task(
  task_id="Custom-R1-Rough-RMA",
  env_cfg=_rough_rma_env,
  play_env_cfg=custom_r1_rough_rma_env_cfg(play=True),
  rl_cfg=custom_r1_rough_rma_ppo_runner_cfg(
    experiment_name="custom_r1_rough_rma",
    num_scan=_rough_rma_num_scan,
  ),
  runner_cls=RMAOnPolicyRunner,
)
