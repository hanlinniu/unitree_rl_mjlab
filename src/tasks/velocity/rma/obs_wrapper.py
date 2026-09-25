"""Wrap flat proprio actor obs into RMA layout (no scandot).

Packed actor observation:
  [proprio | priv_explicit(9) | priv_latent | history(history_len * proprio)]
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper


class RMAObsWrapper:
  """Adds priv_explicit, priv_latent, and proprio history around a VecEnv."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    history_len: int = 10,
    num_priv_explicit: int = 9,
    num_scan: int = 0,
  ):
    assert num_scan == 0, "plane-no-scandot-RMA does not use scandot"
    self.env = env
    self.history_len = history_len
    self.num_priv_explicit = num_priv_explicit
    self.num_scan = num_scan
    self.device = env.device
    self.num_envs = env.num_envs
    self.max_episode_length = env.max_episode_length
    self.num_actions = env.num_actions
    self.cfg = env.cfg

    # Probe proprio / critic dims from the unwrapped flat task.
    raw = env.get_observations()
    assert "actor" in raw and "critic" in raw
    self.num_prop = int(raw["actor"].shape[-1])
    self.num_critic = int(raw["critic"].shape[-1])

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "rma_priv_latent"):
      raise RuntimeError(
        "RMA buffers missing; ensure init_rma_buffers / randomize_rma_priv_latent "
        "events are registered on the env."
      )
    self.num_priv_latent = int(unwrapped.rma_priv_latent.shape[-1])
    # History buffer may have been allocated with a placeholder num_prop; resize.
    hist = getattr(unwrapped, "obs_history_buf", None)
    if hist is None or hist.shape[-1] != self.num_prop:
      unwrapped.obs_history_buf = torch.zeros(
        self.num_envs,
        self.history_len,
        self.num_prop,
        device=self.device,
        dtype=torch.float,
      )
    unwrapped.rma_num_prop = self.num_prop
    unwrapped.rma_history_len = self.history_len

    self.num_obs = (
      self.num_prop
      + self.num_scan
      + self.num_priv_explicit
      + self.num_priv_latent
      + self.history_len * self.num_prop
    )

  @property
  def unwrapped(self):
    return self.env.unwrapped

  def get_observations(self) -> TensorDict:
    raw = self.env.get_observations()
    return self._pack(raw)

  def step(self, actions: torch.Tensor):
    obs, rew, dones, extras = self.env.step(actions)
    return self._pack(obs), rew, dones, extras

  def close(self) -> None:
    if hasattr(self.env, "close"):
      self.env.close()

  def __getattr__(self, name: str) -> Any:
    return getattr(self.env, name)

  def _pack(self, raw: TensorDict) -> TensorDict:
    proprio = raw["actor"]
    critic = raw["critic"]
    unwrapped = self.unwrapped

    # priv_explicit: base lin vel (3) + zero pad to 9 (extreme-parkour layout).
    base_lin_vel = self._base_lin_vel()
    zeros = torch.zeros_like(base_lin_vel)
    priv_explicit = torch.cat([base_lin_vel, zeros, zeros], dim=-1)

    priv_latent = unwrapped.rma_priv_latent
    history = unwrapped.obs_history_buf.reshape(self.num_envs, -1)

    actor = torch.cat([proprio, priv_explicit, priv_latent, history], dim=-1)
    assert actor.shape[-1] == self.num_obs

    # Update history AFTER packing (matches extreme-parkour compute_observations).
    ep = unwrapped.episode_length_buf
    hist = unwrapped.obs_history_buf
    unwrapped.obs_history_buf = torch.where(
      (ep <= 1)[:, None, None],
      proprio.unsqueeze(1).expand(-1, self.history_len, -1).clone(),
      torch.cat([hist[:, 1:], proprio.unsqueeze(1)], dim=1),
    )

    return TensorDict(
      {"actor": actor, "critic": critic},
      batch_size=[self.num_envs],
    )

  def _base_lin_vel(self) -> torch.Tensor:
    """Read base linear velocity in the body / IMU frame used by critic."""
    # Prefer the same sensor the critic uses.
    try:
      from mjlab.envs import mdp as envs_mdp

      return envs_mdp.builtin_sensor(
        self.unwrapped, sensor_name="robot/imu_lin_vel"
      )
    except Exception:
      robot = self.unwrapped.scene["robot"]
      return robot.data.root_link_lin_vel_b
