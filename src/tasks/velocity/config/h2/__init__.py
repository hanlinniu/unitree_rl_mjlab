from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from src.tasks.velocity.rma import RMAOnPolicyRunner

from .env_cfgs import (
  unitree_h2_flat_env_cfg,
  unitree_h2_flat_rma_env_cfg,
  unitree_h2_rough_env_cfg,
)
from .rl_cfg import unitree_h2_ppo_runner_cfg, unitree_h2_rma_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-H2-Rough",
  env_cfg=unitree_h2_rough_env_cfg(),
  play_env_cfg=unitree_h2_rough_env_cfg(play=True),
  rl_cfg=unitree_h2_ppo_runner_cfg(experiment_name="h2_velocity"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Flat",
  env_cfg=unitree_h2_flat_env_cfg(),
  play_env_cfg=unitree_h2_flat_env_cfg(play=True),
  rl_cfg=unitree_h2_ppo_runner_cfg(experiment_name="h2_velocity"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Flat-RMA",
  env_cfg=unitree_h2_flat_rma_env_cfg(),
  play_env_cfg=unitree_h2_flat_rma_env_cfg(play=True),
  rl_cfg=unitree_h2_rma_ppo_runner_cfg(experiment_name="h2_flat_rma"),
  runner_cls=RMAOnPolicyRunner,
)
