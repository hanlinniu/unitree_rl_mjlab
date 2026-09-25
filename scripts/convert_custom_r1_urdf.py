#!/usr/bin/env python3
"""Convert custom R1 URDF (29 rev joints) into an mjlab-ready MJCF.

Important: MjSpec drops mesh geoms if both contype and conaffinity are set to 0
before compile/to_xml. We keep contact flags until after to_xml, then patch the
XML so visuals are group=2 / non-colliding / density=0.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import mujoco


FOOT_CAPS = [
  ("foot1", "0.05 -0.025 -0.04", "0.10 -0.025 -0.04"),
  ("foot2", "-0.03 -0.02 -0.04", "0.115 -0.02 -0.04"),
  ("foot3", "-0.038 -0.01 -0.04", "0.12 -0.01 -0.04"),
  ("foot4", "-0.04 0 -0.04", "0.123 0 -0.04"),
  ("foot5", "-0.038 0.01 -0.04", "0.12 0.01 -0.04"),
  ("foot6", "-0.03 0.02 -0.04", "0.115 0.02 -0.04"),
  ("foot7", "0.05 0.025 -0.04", "0.10 0.025 -0.04"),
]


def _prepare_urdf(src_urdf: Path, dst_urdf: Path, mesh_dir_name: str = "meshes") -> None:
  text = src_urdf.read_text()
  text = text.replace("package://R1/meshes/", f"{mesh_dir_name}/")
  # Fix known zero axis on right wrist pitch.
  text = re.sub(
    r'(name="right_wrist_pitch_joint"\s+type="revolute">.*?)<axis\s+xyz="0 0 0"\s*/>',
    r'\1<axis xyz="0 1 0" />',
    text,
    flags=re.S,
  )
  # Drop collision meshes; we inject capsule collisions later for RL stability.
  text = re.sub(
    r"<collision>\s*<origin[^/]*/>\s*<geometry>\s*<mesh[^/]*/>\s*</geometry>\s*</collision>",
    "",
    text,
    flags=re.S,
  )
  dst_urdf.write_text(text)


def _find_body(spec: mujoco.MjSpec, name: str):
  for b in spec.bodies:
    if b.name == name:
      return b
  raise KeyError(f"body not found: {name}")


def _patch_mesh_geoms(xml: str) -> str:
  """Make visual mesh geoms non-colliding after to_xml (MjSpec-safe)."""

  def patch(match: re.Match[str]) -> str:
    tag = match.group(0)
    if 'mesh="' not in tag:
      return tag
    for attr, val in (
      ("contype", "0"),
      ("conaffinity", "0"),
      ("group", "2"),
      ("density", "0"),
    ):
      if re.search(rf'\b{attr}="', tag):
        tag = re.sub(rf'\b{attr}="[^"]*"', f'{attr}="{val}"', tag)
      elif tag.endswith("/>"):
        tag = tag[:-2] + f' {attr}="{val}"/>'
      else:
        tag = tag[:-1] + f' {attr}="{val}">'
    return tag

  return re.sub(r"<geom\b[^>]*/?>", patch, xml)


def convert(urdf_path: Path, out_xml: Path, assets_dir: Path) -> None:
  work = out_xml.parent
  work.mkdir(parents=True, exist_ok=True)
  mesh_dst = work / "assets"
  mesh_dst.mkdir(exist_ok=True)

  # Copy meshes into assets/ (mjlab convention).
  src_meshes = assets_dir
  for mesh in list(src_meshes.glob("*.STL")) + list(src_meshes.glob("*.stl")):
    shutil.copy2(mesh, mesh_dst / mesh.name)

  prepared = work / "r1_29dof_prepared.urdf"
  _prepare_urdf(urdf_path, prepared, mesh_dir_name="assets")

  # Absolute meshdir so MjSpec can resolve STLs during compile.
  spec = mujoco.MjSpec.from_file(str(prepared))
  spec.modelname = "r1_custom_29dof"
  spec.meshdir = str(mesh_dst)

  # Ensure floating base on pelvis.
  pelvis = _find_body(spec, "pelvis_link")
  has_free = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in pelvis.joints)
  if not has_free:
    pelvis.add_freejoint(name="floating_base_joint")

  # Style visuals only — do NOT set contype=conaffinity=0 here (drops meshes).
  for g in spec.geoms:
    if g.type == mujoco.mjtGeom.mjGEOM_MESH:
      g.group = 2
      g.rgba = [0.79, 0.82, 0.93, 1.0]

  pelvis.add_geom(
    name="pelvis_collision",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.07, 0, 0],
    pos=[0.0, 0.0, -0.05],
    contype=1,
    conaffinity=1,
    condim=1,
    group=3,
    priority=1,
  )

  torso = _find_body(spec, "torso_Link")
  torso.add_geom(
    name="torso_collision",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=[0.08, 0, 0],
    fromto=[-0.02, 0, 0.08, -0.02, 0, 0.22],
    contype=1,
    conaffinity=1,
    condim=1,
    group=3,
    priority=1,
  )

  for side in ("left", "right"):
    hip = _find_body(spec, f"{side}_hip_roll_Link")
    hip.add_geom(
      name=f"{side}_hip_collision",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=[0.04, 0, 0],
      fromto=[0, 0, 0.02, 0, 0, -0.05],
      contype=1,
      conaffinity=1,
      condim=1,
      group=3,
      priority=1,
    )
    thigh = _find_body(spec, f"{side}_hip_yaw_Link")
    thigh.add_geom(
      name=f"{side}_thigh_collision",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=[0.045, 0, 0],
      fromto=[0, 0, -0.02, 0, 0, -0.18],
      contype=1,
      conaffinity=1,
      condim=1,
      group=3,
      priority=1,
    )
    shin = _find_body(spec, f"{side}_knee_Link")
    shin.add_geom(
      name=f"{side}_shin_collision",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=[0.04, 0, 0],
      fromto=[0, 0, -0.03, 0, 0, -0.25],
      contype=1,
      conaffinity=1,
      condim=1,
      group=3,
      priority=1,
    )
    foot = _find_body(spec, f"{side}_ankle_roll_Link")
    for tag, a, b in FOOT_CAPS:
      ax, ay, az = [float(x) for x in a.split()]
      bx, by, bz = [float(x) for x in b.split()]
      foot.add_geom(
        name=f"{side}_{tag}_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.012, 0, 0],
        fromto=[ax, ay, az, bx, by, bz],
        contype=1,
        conaffinity=1,
        condim=3,
        friction=[0.6, 0.005, 0.0001],
        group=3,
        priority=1,
      )
    foot.add_site(name=f"{side}_foot", pos=[0.04, 0.0, -0.05], size=[0.01, 0, 0])

  pelvis.add_site(name="imu", pos=[0, 0, 0], size=[0.01, 0, 0])

  spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")

  model = spec.compile()
  if model.nmesh < 1:
    raise RuntimeError("No meshes compiled — check STL paths / meshdir")

  xml = spec.to_xml()
  xml = re.sub(r'meshdir="[^"]*"', 'meshdir="assets"', xml)
  xml = _patch_mesh_geoms(xml)

  if "subtreeangmom" not in xml:
    xml = xml.replace(
      "</sensor>",
      '    <subtreeangmom name="root_angmom" body="pelvis_link"/>\n  </sensor>',
    )
  if 'pos="0 0 1.03"' not in xml:
    xml = xml.replace(
      '<body name="pelvis_link"',
      '<body name="pelvis_link" pos="0 0 1.03"',
      1,
    )

  out_xml.write_text(xml)
  m = mujoco.MjModel.from_xml_path(str(out_xml))
  n_mesh_geoms = sum(1 for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH)
  print(f"Wrote {out_xml} (nmesh={m.nmesh}, mesh_geoms={n_mesh_geoms}, ngeom={m.ngeom})")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--urdf", type=Path, required=True)
  p.add_argument("--meshes", type=Path, required=True)
  p.add_argument("--out-xml", type=Path, required=True)
  args = p.parse_args()
  convert(args.urdf, args.out_xml, args.meshes)


if __name__ == "__main__":
  main()
