"""Custom 29-DoF R1 velocity environment configurations."""

from dataclasses import replace
import importlib.util
import math
import os
from pathlib import Path

from src.assets.robots.custom_r1 import (
  CUSTOM_R1_ACTION_SCALE,
  get_custom_r1_robot_cfg,
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
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# Local reward helpers (avoid importing src.tasks.velocity.mdp package, which
# re-enters task registration and would circular-import this module).
_REWARDS_PATH = Path(__file__).resolve().parents[2] / "mdp" / "rewards.py"
_spec = importlib.util.spec_from_file_location(
  "_custom_r1_velocity_rewards", _REWARDS_PATH
)
assert _spec is not None and _spec.loader is not None
_local_rewards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_rewards)

# Pyramid stairs for Custom-R1-Rough only (does not patch global ROUGH_TERRAINS_CFG).
# Real residential risers are typically ~0.15–0.18 m; codes often max ~0.19–0.20 m.
# With curriculum=True, row difficulty interpolates step height from low → high.
# terrain_levels_vel promotes robots that walk far enough to harder (taller) stairs.
_CUSTOM_R1_STAIR_STEP_HEIGHT_RANGE = (0.0, 0.18)  # meters: easy → real-height
# Start all envs on the easiest rows; curriculum raises them when ready.
_CUSTOM_R1_MAX_INIT_TERRAIN_LEVEL = 0


def custom_r1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  # Train: keep nconmax modest (128 OOMs EPA buffers at 4096 envs on 24GB).
  # Play: raise so single-env heightfield contacts do not overflow.
  cfg.sim.nconmax = 128 if play else 48

  cfg.scene.entities = {"robot": get_custom_r1_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis_link"
      # Custom R1 visuals are MuJoCo group 2; default ray groups (0,1,2) therefore
      # hit the robot body. Terrain primitives/hfields are group 0 — scan those only.
      sensor.include_geom_groups = (0,)
      sensor.exclude_parent_body = True

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_Link|right_ankle_roll_Link)$",
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
    primary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
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
    gen = cfg.scene.terrain.terrain_generator
    # replace() on ROUGH_TERRAINS_CFG is shallow; rebuild sub_terrains so we do not
    # mutate site-packages defaults shared by other robots.
    sub = dict(gen.sub_terrains)
    if "pyramid_stairs" in sub:
      sub["pyramid_stairs"] = replace(
        sub["pyramid_stairs"],
        step_height_range=_CUSTOM_R1_STAIR_STEP_HEIGHT_RANGE,
        step_width=0.35,
      )
    if "pyramid_stairs_inv" in sub:
      sub["pyramid_stairs_inv"] = replace(
        sub["pyramid_stairs_inv"],
        step_height_range=_CUSTOM_R1_STAIR_STEP_HEIGHT_RANGE,
        step_width=0.35,
      )
    cfg.scene.terrain.terrain_generator = replace(
      gen, curriculum=True, sub_terrains=sub
    )
    cfg.scene.terrain.max_init_terrain_level = _CUSTOM_R1_MAX_INIT_TERRAIN_LEVEL
    # Keep terrain_levels curriculum (promotes/demotes by walked distance).
    assert "terrain_levels" in cfg.curriculum

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = CUSTOM_R1_ACTION_SCALE

  cfg.viewer.body_name = "torso_Link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_Link",)

  # 29-DoF pose tolerances (legs + waist_yaw + head + arms).
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_roll.*": 0.1,
    r".*ankle_pitch.*": 0.15,
    r"waist_yaw.*": 0.15,
    r"head_.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_roll.*": 0.1,
    r".*ankle_pitch.*": 0.25,
    r"waist_yaw.*": 0.25,
    r"head_.*": 0.15,
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_Link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_Link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  # Penalize crouch: cost = relu(stand_dist - max(waist↔L_foot, waist↔R_foot)).
  # Single-leg bend keeps max ≈ stand_dist; both knees bent shrinks max → cost.
  cfg.rewards["waist_foot_distance"] = RewardTermCfg(
    func=_local_rewards.waist_foot_distance,
    weight=-2.0,
    params={
      "target_distance": 1.047,  # Custom-R1 HOME pelvis↔foot distance
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("pelvis_link",),
        site_names=("left_foot", "right_foot"),
      ),
    },
  )
  # Hold default pose when twist cmd ≈ 0 (L2 joint error, weight -1.0; matches
  # stable model_3500 — do not switch to gym L1 at this weight on 29-DoF).
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=_local_rewards.stand_still,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    },
  )

  # --- Stair traversal tuning ---
  # Absolute world-Z foot clearance fights stairs (stance foot rises with the
  # step). Disable it; rely on gait + xy velocity tracking.
  cfg.rewards["foot_clearance"].weight = 0.0
  # Stronger push to keep commanded horizontal speed (discourage freezing).
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.5)
  # Hold heading when wz_cmd≈0 (stairs previously induced unwanted yaw).
  # Match linear tracking priority so forward progress does not override yaw hold.
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25)
  cfg.rewards["foot_gait"].weight = 0.75
  # Falling is still bad, but -200 made the policy freeze instead of risk a step.
  cfg.rewards["is_terminated"].weight = -80.0
  # Pitch/roll vary on stairs; keep upright but don't over-penalize.
  cfg.rewards["body_orientation_l2"].weight = -0.5
  cfg.rewards["body_ang_vel"].weight = -0.02
  cfg.rewards["angular_momentum"].weight = -0.01
  # Allow larger hip/knee flexion for high stepping (single leg).
  # Keep knee walking std moderate so crouch is not free via pose reward alone.
  cfg.rewards["pose"].params["std_walking"].update(
    {
      r".*hip_pitch.*": 0.85,
      r".*knee.*": 0.55,
      r".*ankle_pitch.*": 0.35,
    }
  )
  cfg.rewards["pose"].params["std_running"].update(
    {
      r".*hip_pitch.*": 1.0,
      r".*knee.*": 0.7,
      r".*ankle_pitch.*": 0.45,
    }
  )
  # Prefer walking commands over standing still during Rough training.
  # Direct twist commands (no heading servo) so the robot walks into stairs
  # instead of turning in place when progress slows. Keep yaw range modest so
  # most episodes practice straight walking (wz≈0) on stairs; joystick still
  # commands yaw at play time within this band.
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  # Match stairs-uneven model_3500 standing mix (standyaw 0.10 destabilized standing).
  twist_cmd.rel_standing_envs = 0.02
  twist_cmd.ranges.lin_vel_x = (0.0, 1.5)  # bias forward; less reverse into steps
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)  # less lateral drift coupling into yaw
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  # Keep command curriculum aligned (step-0 stage otherwise reverts ranges).
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.0, 1.5),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 5000 * 24,
        "lin_vel_x": (0.0, 2.0),
        "lin_vel_y": (-0.3, 0.3),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]

  # Stairs + uneven only (no slopes / flat).
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    gen = cfg.scene.terrain.terrain_generator
    sub = dict(gen.sub_terrains)
    for drop in (
      "flat",
      "hf_pyramid_slope",
      "hf_pyramid_slope_inv",
    ):
      sub.pop(drop, None)
    for name, prop in (
      ("pyramid_stairs", 0.30),
      ("pyramid_stairs_inv", 0.30),
      ("random_rough", 0.20),
      ("wave_terrain", 0.20),
    ):
      if name in sub:
        sub[name] = replace(sub[name], proportion=prop)
    cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=sub)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      gen = cfg.scene.terrain.terrain_generator
      gen.curriculum = False
      gen.num_cols = 5
      gen.num_rows = 5
      gen.border_width = 10.0
      # Optional play-only stair height (meters), e.g. CUSTOM_R1_PLAY_STAIR_HEIGHT=0.15
      # Forces stairs-only terrain at that fixed riser for local testing.
      play_stair_h = os.environ.get("CUSTOM_R1_PLAY_STAIR_HEIGHT")
      if play_stair_h:
        h = float(play_stair_h)
        sub = {
          "pyramid_stairs": replace(
            gen.sub_terrains["pyramid_stairs"],
            step_height_range=(h, h),
            proportion=0.5,
          ),
          "pyramid_stairs_inv": replace(
            gen.sub_terrains["pyramid_stairs_inv"],
            step_height_range=(h, h),
            proportion=0.5,
          ),
        }
        cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=sub)
        print(f"[INFO] Play stairs-only at step_height={h:.3f} m")

  return cfg


def custom_r1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = custom_r1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)

  # Undo Rough stair reward hacks — Flat matches H2 Flat / base velocity defaults.
  cfg.rewards.pop("waist_foot_distance", None)  # H2 Flat has no crouch term
  # Match Unitree-H2-Flat-RMA action scale (H2 Flat uses base scalar 0.25).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.25
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
  # Stronger yaw hold so forward (vx>0, wz_cmd≈0) stays straight (play uses wz).
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25)
  cfg.rewards["foot_gait"].weight = 0.5
  cfg.rewards["is_terminated"].weight = -200.0
  cfg.rewards["body_orientation_l2"].weight = -1.0
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.025
  # Restore non-stair pose walking/running stds (Rough loosens hip/knee for steps).
  cfg.rewards["pose"].params["std_walking"].update(
    {
      r".*hip_pitch.*": 0.5,
      r".*knee.*": 0.5,
      r".*ankle_pitch.*": 0.15,
    }
  )
  cfg.rewards["pose"].params["std_running"].update(
    {
      r".*hip_pitch.*": 0.5,
      r".*knee.*": 0.5,
      r".*ankle_pitch.*": 0.25,
    }
  )

  # Mixed heading: 90% envs use heading servo, 10% keep sampled ang_vel_z
  # (direct wz, including wz≈0 while walking) so play/joystick straight walking
  # is practiced without dropping heading training entirely.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = True
  twist_cmd.ranges.heading = (-math.pi, math.pi)
  twist_cmd.rel_heading_envs = 0.90  # → 10% direct-wz envs
  twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  # Narrow yaw for the direct-wz minority (and heading-servo clip bounds).
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

  # Explicit L2 stand_still (H2 Flat / world_model intent; mjlab L2@-1.0).
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=_local_rewards.stand_still,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    },
  )

  # Train with gym-matched obs noise; play disables corruption.
  if not play:
    cfg.observations["actor"].enable_corruption = True
  else:
    cfg.observations["actor"].enable_corruption = False
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def custom_r1_flat_rma_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Plane / no-scandot Flat env with RMA priv_latent domain randomization."""
  # Import locally to avoid circular imports through src.tasks.velocity.mdp.
  from src.tasks.velocity.rma import events as rma_events

  cfg = custom_r1_flat_env_cfg(play=play)

  # Replace separate friction/COM startup terms with the unified RMA randomizer
  # so priv_latent matches the applied dynamics.
  cfg.events.pop("foot_friction", None)
  cfg.events.pop("base_com", None)

  foot_geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  cfg.events["init_rma_buffers"] = EventTermCfg(
    mode="startup",
    func=rma_events.init_rma_buffers,
    params={
      "history_len": 10,
      "num_priv_explicit": 9,
    },
  )
  cfg.events["randomize_rma_priv_latent"] = EventTermCfg(
    mode="reset",
    func=rma_events.randomize_rma_priv_latent,
    params={
      "friction_range": (0.3, 1.6),
      "mass_scale_range": (0.8, 1.2),
      "com_offset_range": (-0.05, 0.05),
      "motor_strength_range": (0.8, 1.2),
      "foot_asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_Link",)),
    },
  )
  # Also run once at startup so the first observation is valid before reset.
  cfg.events["randomize_rma_priv_latent_startup"] = EventTermCfg(
    mode="startup",
    func=rma_events.randomize_rma_priv_latent,
    params={
      "friction_range": (0.3, 1.6),
      "mass_scale_range": (0.8, 1.2),
      "com_offset_range": (-0.05, 0.05),
      "motor_strength_range": (0.8, 1.2),
      "foot_asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_Link",)),
    },
  )

  # Flat-RMA inherits H2-matched Flat standstill recipe; keep corruption on for train.
  if not play:
    cfg.observations["actor"].enable_corruption = True
    assert "waist_foot_distance" not in cfg.rewards
    assert cfg.actions["joint_pos"].scale == 0.25
    assert cfg.rewards["stand_still"].weight == -1.0
    assert cfg.commands["twist"].rel_standing_envs >= 0.10
    assert cfg.commands["twist"].heading_command is True
    assert cfg.commands["twist"].rel_heading_envs == 0.90
    assert cfg.rewards["track_angular_velocity"].weight == 2.0
    assert cfg.rewards["track_linear_velocity"].weight == 1.0

  return cfg
