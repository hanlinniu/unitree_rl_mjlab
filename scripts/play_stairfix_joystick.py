#!/usr/bin/env python3
"""Play Custom-R1-Rough with physical joystick twist commands.

Mapping (this ShanWan/PC pad on Linux — sticks at 0, triggers rest at -1):
  Left stick X  (axis 0)           -> yaw rate  wz
  Right stick Y (axis 3) * -1      -> forward   vx  (stick up = forward)
  Right stick X (axis 2)           -> lateral   vy  (stick left/right)

  Do NOT bind axis 4/5: those are triggers (rest at -1.0).

Usage:
  python scripts/play_stairfix_joystick.py
  python scripts/play_stairfix_joystick.py --checkpoint-file path/to/model.pt
  python scripts/play_stairfix_joystick.py --stair-height 0.10
"""

from __future__ import annotations

import os
import select
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Hide pygame banner if something imports it transitively.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
os.environ.setdefault("MUJOCO_GL", "glfw")

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

_JS_EVENT = struct.Struct("IhBB")
_JS_AXIS = 0x02
_JS_INIT = 0x80

# Default: latest stairfix-from3000 run under logs/
_DEFAULT_CKPT_GLOB = (
  "logs/rsl_rl/custom_r1_rough_velocity/"
  "*R1-Rough-stairfix-from3000-20260924-192243*/model_*.pt"
)


def _find_latest_ckpt(explicit: str | None) -> Path:
  if explicit:
    p = Path(explicit).expanduser().resolve()
    if not p.exists():
      raise FileNotFoundError(p)
    return p
  root = Path(__file__).resolve().parents[1]
  matches = sorted(
    root.glob(_DEFAULT_CKPT_GLOB),
    key=lambda p: (p.stat().st_mtime, int(p.stem.split("_")[1])),
  )
  if not matches:
    raise FileNotFoundError(
      f"No checkpoint matching {_DEFAULT_CKPT_GLOB}. Pass --checkpoint-file."
    )
  return matches[-1]


def _find_js_device(explicit: str | None) -> str:
  if explicit:
    if not Path(explicit).exists():
      raise FileNotFoundError(explicit)
    return explicit
  devices = sorted(Path("/dev/input").glob("js*"))
  if not devices:
    raise FileNotFoundError("No /dev/input/js* joystick found.")
  return str(devices[0])


class LinuxJoystickReader:
  """Background reader for Linux joystick axes."""

  def __init__(self, device: str) -> None:
    self.device = device
    self.axes: dict[int, float] = {}
    self._stop = threading.Event()
    self._fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
    self._thread = threading.Thread(target=self._loop, name="js-reader", daemon=True)
    self._thread.start()

  def _loop(self) -> None:
    while not self._stop.is_set():
      r, _, _ = select.select([self._fd], [], [], 0.05)
      if not r:
        continue
      while True:
        try:
          data = os.read(self._fd, _JS_EVENT.size)
        except BlockingIOError:
          break
        if len(data) < _JS_EVENT.size:
          break
        _t, value, typ, number = _JS_EVENT.unpack(data)
        if (typ & ~_JS_INIT) == _JS_AXIS:
          self.axes[number] = value / 32767.0

  def axis(self, idx: int, deadzone: float = 0.08) -> float:
    v = self.axes.get(idx, 0.0)
    return 0.0 if abs(v) < deadzone else v

  def close(self) -> None:
    self._stop.set()
    try:
      os.close(self._fd)
    except OSError:
      pass


class JoystickTwistController:
  """Joystick -> twist command for velocity play.

  Left stick X  -> yaw (wz)
  Right stick Y (reversed) -> forward (vx)
  Right stick X -> lateral (vy)  // left / right
  """

  def __init__(
    self,
    env: Any,
    js: LinuxJoystickReader,
    *,
    lx_axis: int = 0,
    # Pad layout: 2=RX, 3=RY, 4/5=triggers (rest -1) — never use 4/5 as sticks.
    rx_axis: int = 2,
    ry_axis: int = 3,
    deadzone: float = 0.08,
    invert_yaw: bool = False,
    invert_forward: bool = True,  # stick up (Linux -Y) -> +vx
    invert_lateral: bool = True,  # reverse left/right
    vx_scale: float | None = None,
    vy_scale: float | None = None,
    wz_scale: float | None = None,
  ) -> None:
    self._env = env
    self._js = js
    self.lx_axis = lx_axis
    self.rx_axis = rx_axis
    self.ry_axis = ry_axis
    self.deadzone = deadzone
    self.invert_yaw = invert_yaw
    self.invert_forward = invert_forward
    self.invert_lateral = invert_lateral
    self.vx_scale = vx_scale
    self.vy_scale = vy_scale
    self.wz_scale = wz_scale
    self._term: Any | None = None
    self.enabled = False
    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0
    self._last_print = 0.0

  def attach(self) -> bool:
    command_manager = getattr(self._env.unwrapped, "command_manager", None)
    if command_manager is None or "twist" not in command_manager.active_terms:
      return False

    term = command_manager.get_term("twist")
    term.cfg.heading_command = False
    term.cfg.rel_standing_envs = 0.0
    term.cfg.resampling_time_range = (1e9, 1e9)
    if hasattr(term.cfg.ranges, "heading"):
      term.cfg.ranges.heading = None
    term.is_heading_env[:] = False
    term.is_standing_env[:] = False
    term.vel_command_b.zero_()

    # Default scales from command ranges.
    if self.vx_scale is None:
      self.vx_scale = float(term.cfg.ranges.lin_vel_x[1])
    if self.vy_scale is None:
      lo, hi = term.cfg.ranges.lin_vel_y
      self.vy_scale = float(max(abs(lo), abs(hi)))
    if self.wz_scale is None:
      lo, hi = term.cfg.ranges.ang_vel_z
      self.wz_scale = float(max(abs(lo), abs(hi)))

    self._term = term
    orig_compute = term.compute

    def compute(dt: float) -> None:
      orig_compute(dt)
      if not self.enabled or self._term is None:
        return
      self._poll()
      self._term.is_heading_env[:] = False
      self._term.is_standing_env[:] = False
      self._term.vel_command_b[:, 0] = self.vx
      self._term.vel_command_b[:, 1] = self.vy
      self._term.vel_command_b[:, 2] = self.wz

    term.compute = compute  # type: ignore[method-assign]
    self.enabled = True
    return True

  def _poll(self) -> None:
    assert (
      self._term is not None
      and self.vx_scale is not None
      and self.vy_scale is not None
      and self.wz_scale is not None
    )
    lx = self._js.axis(self.lx_axis, self.deadzone)
    rx = self._js.axis(self.rx_axis, self.deadzone)
    ry = self._js.axis(self.ry_axis, self.deadzone)

    # Left X -> yaw. Default: stick left (negative on Linux) = turn left (+wz).
    yaw_cmd = -lx if not self.invert_yaw else lx
    # Right Y -> forward (stick up = forward with invert).
    fwd_cmd = -ry if self.invert_forward else ry
    # Right X -> lateral left/right.
    lat_cmd = -rx if self.invert_lateral else rx

    ranges = self._term.cfg.ranges
    self.wz = max(
      float(ranges.ang_vel_z[0]),
      min(float(ranges.ang_vel_z[1]), yaw_cmd * self.wz_scale),
    )
    self.vx = max(
      float(ranges.lin_vel_x[0]),
      min(float(ranges.lin_vel_x[1]), fwd_cmd * self.vx_scale),
    )
    self.vy = max(
      float(ranges.lin_vel_y[0]),
      min(float(ranges.lin_vel_y[1]), lat_cmd * self.vy_scale),
    )

    now = time.time()
    if now - self._last_print > 0.25:
      self._last_print = now
      print(
        f"[joy] Lx={lx:+.2f} Rx={rx:+.2f} Ry={ry:+.2f}  ->  "
        f"vx={self.vx:+.2f} vy={self.vy:+.2f} m/s  wz={self.wz:+.2f} rad/s",
        flush=True,
      )


@dataclass(frozen=True)
class JoyPlayConfig:
  checkpoint_file: str | None = None
  device: str | None = None
  num_envs: int = 1
  stair_height: float | None = 0.10
  """If set, play on stairs-only terrain at this riser height (m)."""
  js_device: str | None = None
  lx_axis: int = 0
  # This pad: RX=2, RY=3; axes 4/5 are triggers (rest -1).
  rx_axis: int = 2
  ry_axis: int = 3
  deadzone: float = 0.08
  # Right Y: stick up -> forward.
  invert_forward: bool = True
  invert_yaw: bool = False
  # Right X: invert so stick left/right matches robot left/right.
  invert_lateral: bool = True
  vx_max: float | None = 1.0
  vy_max: float | None = 0.5
  wz_max: float | None = 0.8


def run(cfg: JoyPlayConfig) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  ckpt = _find_latest_ckpt(cfg.checkpoint_file)
  js_dev = _find_js_device(cfg.js_device)

  if cfg.stair_height is not None:
    os.environ["CUSTOM_R1_PLAY_STAIR_HEIGHT"] = str(cfg.stair_height)
  else:
    os.environ.pop("CUSTOM_R1_PLAY_STAIR_HEIGHT", None)

  task_id = "Custom-R1-Rough"
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs

  # Disable random twist resampling before env construction.
  if "twist" in env_cfg.commands:
    twist = env_cfg.commands["twist"]
    twist.heading_command = False
    twist.rel_standing_envs = 0.0
    twist.resampling_time_range = (1e9, 1e9)
    if hasattr(twist.ranges, "heading"):
      twist.ranges.heading = None
    if cfg.vx_max is not None:
      twist.ranges.lin_vel_x = (0.0, float(cfg.vx_max))
    if cfg.vy_max is not None:
      twist.ranges.lin_vel_y = (-float(cfg.vy_max), float(cfg.vy_max))
    if cfg.wz_max is not None:
      twist.ranges.ang_vel_z = (-float(cfg.wz_max), float(cfg.wz_max))

  print(f"[INFO] Checkpoint: {ckpt}")
  print(f"[INFO] Joystick:   {js_dev}")
  print(
    "[INFO] Mapping: Left X -> yaw (wz), "
    "Right Y -> forward (vx), "
    "Right X -> lateral (vy left/right)"
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  js = LinuxJoystickReader(js_dev)
  joy = JoystickTwistController(
    env,
    js,
    lx_axis=cfg.lx_axis,
    rx_axis=cfg.rx_axis,
    ry_axis=cfg.ry_axis,
    deadzone=cfg.deadzone,
    invert_yaw=cfg.invert_yaw,
    invert_forward=cfg.invert_forward,
    invert_lateral=cfg.invert_lateral,
    vx_scale=cfg.vx_max,
    vy_scale=cfg.vy_max,
    wz_scale=cfg.wz_max,
  )
  if not joy.attach():
    js.close()
    raise RuntimeError("Failed to attach joystick twist controller (no twist command).")

  try:
    NativeMujocoViewer(env, policy, key_callback=None).run()
  finally:
    js.close()
    env.close()


def main() -> None:
  cfg = tyro.cli(JoyPlayConfig)
  run(cfg)


if __name__ == "__main__":
  main()
