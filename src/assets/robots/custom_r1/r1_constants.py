"""Custom 29-DoF R1 constants (from user URDF)."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

R1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "custom_r1" / "xmls" / "r1_29dof.xml"
)


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, R1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  assert R1_XML.exists(), f"Missing custom R1 MJCF: {R1_XML}"
  spec = mujoco.MjSpec.from_file(str(R1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


# 29 actuated DOFs:
# 12 legs + waist_yaw + head_yaw/pitch + 14 arm (7/side)
R1_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch.*",
    ".*_hip_roll.*",
    ".*_hip_yaw.*",
    ".*_knee.*",
  ),
  stiffness=200.0,
  damping=4.0,
  effort_limit=200.0,
  armature=0.01,
)
R1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch.*",
    ".*_ankle_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=50.0,
  armature=0.01,
)
R1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw.*",),
  stiffness=150.0,
  damping=3.0,
  effort_limit=120.0,
  armature=0.01,
)
R1_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("head_.*",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=30.0,
  armature=0.01,
)
R1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch.*",
    ".*_shoulder_roll.*",
    ".*_shoulder_yaw.*",
    ".*_elbow.*",
    ".*_wrist_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=54.0,
  armature=0.01,
)
R1_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_pitch.*",
    ".*_wrist_yaw.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=25.0,
  armature=0.01,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.03),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

R1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    R1_ACTUATOR_LEG,
    R1_ACTUATOR_ANKLE,
    R1_ACTUATOR_WAIST,
    R1_ACTUATOR_HEAD,
    R1_ACTUATOR_ARM,
    R1_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_custom_r1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=R1_ARTICULATION,
  )


CUSTOM_R1_ACTION_SCALE: dict[str, float] = {}
for a in R1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    CUSTOM_R1_ACTION_SCALE[n] = 0.25 * e / s
