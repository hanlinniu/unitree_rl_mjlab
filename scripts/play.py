"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.keys import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_Z


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  keyboard_control: bool = True
  """Use arrow keys to set twist commands (native viewer)."""
  lin_vel_step: float = 0.1
  """Forward-speed change per Up/Down press (m/s)."""
  ang_vel_step: float = 0.1
  """Yaw-rate change per Left/Right press (rad/s)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


class KeyboardTwistController:
  """Arrow-key twist commands for velocity play (native viewer).

  Up/Down  : forward speed (lin_vel_x)
  Left/Right: turning rate (ang_vel_z)
  Z        : zero command (stand)
  """

  def __init__(
    self,
    env: Any,
    *,
    lin_step: float = 0.1,
    ang_step: float = 0.1,
  ):
    self._env = env
    self._lin_step = lin_step
    self._ang_step = ang_step
    self._term: Any | None = None
    self._orig_compute: Any | None = None
    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0
    self.enabled = False

  def attach(self) -> bool:
    command_manager = getattr(self._env.unwrapped, "command_manager", None)
    if command_manager is None or "twist" not in command_manager.active_terms:
      return False

    term = command_manager.get_term("twist")
    # Stop random resampling / heading / standing from fighting the keyboard.
    term.cfg.heading_command = False
    term.cfg.rel_standing_envs = 0.0
    term.cfg.resampling_time_range = (1e9, 1e9)
    term.is_heading_env[:] = False
    term.is_standing_env[:] = False
    term.vel_command_b.zero_()

    self._term = term
    self._orig_compute = term.compute

    def compute(dt: float) -> None:
      assert self._orig_compute is not None and self._term is not None
      self._orig_compute(dt)
      if not self.enabled:
        return
      self._term.is_heading_env[:] = False
      self._term.is_standing_env[:] = False
      self._term.vel_command_b[:, 0] = self.vx
      self._term.vel_command_b[:, 1] = self.vy
      self._term.vel_command_b[:, 2] = self.wz

    term.compute = compute  # type: ignore[method-assign]
    self.enabled = True
    return True

  def on_key(self, key: int) -> None:
    if not self.enabled or self._term is None:
      return
    ranges = self._term.cfg.ranges
    if key == KEY_UP:
      self.vx = min(self.vx + self._lin_step, float(ranges.lin_vel_x[1]))
    elif key == KEY_DOWN:
      self.vx = max(self.vx - self._lin_step, float(ranges.lin_vel_x[0]))
    elif key == KEY_LEFT:
      self.wz = min(self.wz + self._ang_step, float(ranges.ang_vel_z[1]))
    elif key == KEY_RIGHT:
      self.wz = max(self.wz - self._ang_step, float(ranges.ang_vel_z[0]))
    elif key == KEY_Z:
      self.vx = 0.0
      self.vy = 0.0
      self.wz = 0.0
    else:
      return
    print(f"[cmd] vx={self.vx:+.2f} m/s  wz={self.wz:+.2f} rad/s")

def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  # Prepare twist command for manual keyboard control during play.
  if cfg.keyboard_control and "twist" in env_cfg.commands:
    twist_cfg = env_cfg.commands["twist"]
    twist_cfg.heading_command = False
    twist_cfg.rel_standing_envs = 0.0
    twist_cfg.resampling_time_range = (1e9, 1e9)
    # Required when heading_command=False (command ctor validates this).
    if hasattr(twist_cfg.ranges, "heading"):
      twist_cfg.ranges.heading = None

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  keyboard: KeyboardTwistController | None = None
  if cfg.keyboard_control:
    keyboard = KeyboardTwistController(
      env, lin_step=cfg.lin_vel_step, ang_step=cfg.ang_vel_step
    )
    if keyboard.attach():
      print(
        "[INFO] Keyboard twist: Up/Down = forward speed, "
        "Left/Right = turn rate, Z = stop"
      )
    else:
      keyboard = None

  if resolved_viewer == "native":
    NativeMujocoViewer(
      env,
      policy,
      key_callback=keyboard.on_key if keyboard is not None else None,
    ).run()
  elif resolved_viewer == "viser":
    if keyboard is not None:
      print(
        "[INFO] Arrow-key twist is for the native viewer; "
        "use the Viser joystick panel instead."
      )
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
