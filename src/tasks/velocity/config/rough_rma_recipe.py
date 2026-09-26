"""Shared Rough-RMA training recipe (matched R1 / H2 A/B)."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

# Max stair riser (m) for Rough-RMA curriculum.
ROUGH_RMA_STAIR_STEP_HEIGHT_RANGE = (0.0, 0.18)


def apply_matched_rough_rma_recipe(
  cfg: ManagerBasedRlEnvCfg, *, play: bool
) -> None:
  """Align command / reward / action scale for fair R1 vs H2 Rough-RMA."""
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.25

  cfg.rewards.pop("waist_foot_distance", None)
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25)
  cfg.rewards["is_terminated"].weight = -200.0
  cfg.rewards["body_orientation_l2"].weight = -1.0
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.025
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_gait"].weight = 0.5

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # heading_command=True with 10% direct-wz envs (rel_heading_envs=0.90).
  twist_cmd.heading_command = True
  twist_cmd.ranges.heading = (-math.pi, math.pi)
  twist_cmd.rel_heading_envs = 0.90
  twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  if not play:
    twist_cmd.rel_standing_envs = 0.05
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.5, 1.0),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-1.0, 1.0),
      },
      {
        "step": 5000 * 24,
        "lin_vel_x": (-1.0, 2.0),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_z": (-1.0, 1.0),
      },
    ]


def assert_matched_rough_rma_recipe(cfg: ManagerBasedRlEnvCfg) -> None:
  twist = cfg.commands["twist"]
  assert twist.heading_command is True
  assert twist.rel_heading_envs == 0.90
  assert twist.rel_standing_envs == 0.05
  assert twist.ranges.lin_vel_x == (-1.0, 2.0)
  assert twist.ranges.lin_vel_y == (-1.0, 1.0)
  assert twist.ranges.ang_vel_z == (-1.0, 1.0)
  assert cfg.rewards["track_linear_velocity"].weight == 3.0
  assert cfg.rewards["track_angular_velocity"].weight == 3.0
  assert cfg.rewards["is_terminated"].weight == -200.0
  assert cfg.rewards["body_orientation_l2"].weight == -1.0
  assert cfg.rewards["body_ang_vel"].weight == -0.05
  assert cfg.rewards["angular_momentum"].weight == -0.025
  assert cfg.rewards["foot_clearance"].weight == -1.0
  assert cfg.actions["joint_pos"].scale == 0.25
  assert "waist_foot_distance" not in cfg.rewards
