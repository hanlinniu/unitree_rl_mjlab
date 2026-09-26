"""Unitree H2 velocity environment configurations."""

import math
from dataclasses import replace

from src.assets.robots import (
  H2_ACTION_SCALE,
  get_h2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from src.tasks.velocity import mdp as local_mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_h2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree H2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_h2_robot_cfg()}

  # Set raycast sensor frame to H2 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_pitch_link|right_ankle_pitch_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_roll.*": 0.1,
    r".*ankle_pitch.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_roll.*": 0.1,
    r".*ankle_pitch.*": 0.25,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_h2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree H2 flat terrain velocity configuration."""
  cfg = unitree_h2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  # Mixed heading: 90% heading servo, 10% direct sampled ang_vel_z (incl. wz≈0
  # while walking) so play/joystick straight walking is practiced. Strengthen
  # yaw tracking for the direct-wz minority without dropping heading training.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = True
  twist_cmd.ranges.heading = (-math.pi, math.pi)
  twist_cmd.rel_heading_envs = 0.90  # → 10% direct-wz envs
  twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  if not play:
    twist_cmd.rel_standing_envs = 0.10
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.5, 1.0),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-0.3, 0.3),
      },
      {
        "step": 5000 * 24,
        "lin_vel_x": (-1.0, 2.0),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_z": (-0.5, 0.5),
      },
    ]

  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25)

  # Explicit L2 stand_still (world_model gym uses L1@-0.5; mjlab L2@-1.0).
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=local_mdp.stand_still,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    },
  )

  if not play:
    # Gym-matched obs noise (ang_vel±0.2, gravity±0.05, dof_pos±0.01, dof_vel±1.5).
    cfg.observations["actor"].enable_corruption = True
  else:
    cfg.observations["actor"].enable_corruption = False
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_h2_flat_rma_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Plane / no-scandot H2 Flat env with RMA priv_latent domain randomization."""
  from src.tasks.velocity.rma import events as rma_events

  cfg = unitree_h2_flat_env_cfg(play=play)

  # Unified RMA randomizer replaces separate friction/COM startup terms.
  cfg.events.pop("foot_friction", None)
  cfg.events.pop("base_com", None)

  foot_geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  def _rma_params() -> dict:
    return {
      "friction_range": (0.3, 1.6),
      "mass_scale_range": (0.8, 1.2),
      "com_offset_range": (-0.05, 0.05),
      "motor_strength_range": (0.8, 1.2),
      "foot_asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    }

  cfg.events["init_rma_buffers"] = EventTermCfg(
    mode="startup",
    func=rma_events.init_rma_buffers,
    params={"history_len": 10, "num_priv_explicit": 9},
  )
  cfg.events["randomize_rma_priv_latent"] = EventTermCfg(
    mode="reset",
    func=rma_events.randomize_rma_priv_latent,
    params=_rma_params(),
  )
  cfg.events["randomize_rma_priv_latent_startup"] = EventTermCfg(
    mode="startup",
    func=rma_events.randomize_rma_priv_latent,
    params=_rma_params(),
  )
  return cfg


def _apply_h2_stairs_uneven_slope_terrains(cfg: ManagerBasedRlEnvCfg) -> None:
  if cfg.scene.terrain is None or cfg.scene.terrain.terrain_generator is None:
    return
  gen = cfg.scene.terrain.terrain_generator
  sub = dict(ROUGH_TERRAINS_CFG.sub_terrains)
  for key, prop in (
    ("pyramid_stairs", 0.22),
    ("pyramid_stairs_inv", 0.22),
    ("hf_pyramid_slope", 0.14),
    ("hf_pyramid_slope_inv", 0.14),
    ("random_rough", 0.14),
    ("wave_terrain", 0.14),
  ):
    if key not in sub:
      continue
    if key.startswith("hf_pyramid_slope"):
      sub[key] = replace(sub[key], proportion=prop, slope_range=(0.0, 0.4))
    else:
      sub[key] = replace(sub[key], proportion=prop)
  sub.pop("flat", None)
  cfg.scene.terrain.terrain_generator = replace(
    gen, curriculum=True, sub_terrains=sub
  )


def unitree_h2_rough_rma_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough H2 with scandot + RMA (stairs, uneven, slope)."""
  from src.tasks.velocity.rma import events as rma_events

  cfg = unitree_h2_rough_env_cfg(play=play)
  _apply_h2_stairs_uneven_slope_terrains(cfg)

  # Match R1 Rough-RMA contact budget (avoid EPA OOM at multi-k envs).
  if not play:
    cfg.sim.nconmax = 90
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64

  cfg.events.pop("foot_friction", None)
  cfg.events.pop("base_com", None)
  foot_geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  def _rma_params() -> dict:
    return {
      "friction_range": (0.3, 1.6),
      "mass_scale_range": (0.8, 1.2),
      "com_offset_range": (-0.05, 0.05),
      "motor_strength_range": (0.8, 1.2),
      "foot_asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    }

  cfg.events["init_rma_buffers"] = EventTermCfg(
    mode="startup",
    func=rma_events.init_rma_buffers,
    params={"history_len": 10, "num_priv_explicit": 9},
  )
  cfg.events["randomize_rma_priv_latent"] = EventTermCfg(
    mode="reset",
    func=rma_events.randomize_rma_priv_latent,
    params=_rma_params(),
  )
  cfg.events["randomize_rma_priv_latent_startup"] = EventTermCfg(
    mode="startup",
    func=rma_events.randomize_rma_priv_latent,
    params=_rma_params(),
  )

  assert "height_scan" in cfg.observations["actor"].terms
  assert any(s.name == "terrain_scan" for s in (cfg.scene.sensors or ()))
  if not play:
    keys = set(cfg.scene.terrain.terrain_generator.sub_terrains.keys())
    assert "flat" not in keys
    assert {"pyramid_stairs", "hf_pyramid_slope", "random_rough"} <= keys
  return cfg
