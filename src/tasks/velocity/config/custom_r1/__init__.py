from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  custom_r1_flat_env_cfg,
  custom_r1_rough_env_cfg,
)
from .rl_cfg import custom_r1_ppo_runner_cfg

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
