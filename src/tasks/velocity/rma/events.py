"""Domain-randomization buffers for RMA priv_latent (no scandot).

priv_latent mirrors extreme-parkour:
  [mass_params(4) | friction(1) | motor_kp_scale-1 | motor_kd_scale-1]
where motor scales are per-action (Custom-R1: 29 + 29).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _as_id_list(ids) -> list[int]:
  if ids is None:
    return []
  if isinstance(ids, slice):
    return []
  if torch.is_tensor(ids):
    return [int(x) for x in ids.tolist()]
  return [int(x) for x in ids]


def init_rma_buffers(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  history_len: int = 10,
  num_priv_explicit: int = 9,
) -> None:
  """Allocate RMA buffers once at startup."""
  del env_ids
  n = env.num_envs
  device = env.device
  na = env.action_manager.total_action_dim
  num_priv_latent = 4 + 1 + na + na  # mass(4) + friction(1) + kp + kd

  env.rma_history_len = history_len  # type: ignore[attr-defined]
  env.rma_num_priv_latent = num_priv_latent  # type: ignore[attr-defined]
  env.rma_num_priv_explicit = num_priv_explicit  # type: ignore[attr-defined]
  env.rma_priv_latent = torch.zeros(  # type: ignore[attr-defined]
    n, num_priv_latent, device=device, dtype=torch.float
  )
  env.rma_mass_params = torch.zeros(n, 4, device=device, dtype=torch.float)  # type: ignore[attr-defined]
  env.rma_friction = torch.zeros(n, 1, device=device, dtype=torch.float)  # type: ignore[attr-defined]
  env.rma_motor_kp = torch.ones(n, na, device=device, dtype=torch.float)  # type: ignore[attr-defined]
  env.rma_motor_kd = torch.ones(n, na, device=device, dtype=torch.float)  # type: ignore[attr-defined]
  # History is allocated by RMAObsWrapper once num_prop is known.
  env.obs_history_buf = None  # type: ignore[attr-defined]


def randomize_rma_priv_latent(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  friction_range: tuple[float, float] = (0.3, 1.6),
  mass_scale_range: tuple[float, float] = (0.8, 1.2),
  com_offset_range: tuple[float, float] = (-0.05, 0.05),
  motor_strength_range: tuple[float, float] = (0.8, 1.2),
  foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  torso_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Sample privileged factors, apply to the sim model, and store priv_latent."""
  if not hasattr(env, "rma_priv_latent"):
    init_rma_buffers(env, None)

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  if len(env_ids) == 0:
    return

  n = len(env_ids)
  device = env.device
  na = env.action_manager.total_action_dim

  mass_scale = _uniform(n, 1, mass_scale_range, device)
  com = _uniform(n, 3, com_offset_range, device)
  mass_params = torch.cat([mass_scale - 1.0, com], dim=-1)
  friction = _uniform(n, 1, friction_range, device)
  motor_kp = _uniform(n, na, motor_strength_range, device)
  motor_kd = _uniform(n, na, motor_strength_range, device)

  # Build fresh SceneEntityCfg so repeated resolve() does not hit name/id
  # consistency errors after the first call mutates geom_ids/body_ids.
  foot_cfg = SceneEntityCfg(
    foot_asset_cfg.name, geom_names=foot_asset_cfg.geom_names
  )
  torso_cfg = SceneEntityCfg(
    torso_asset_cfg.name, body_names=torso_asset_cfg.body_names
  )
  foot_cfg.resolve(env.scene)
  torso_cfg.resolve(env.scene)
  _apply_geom_friction(env, env_ids, foot_cfg, friction.squeeze(-1))
  _apply_body_com(env, env_ids, torso_cfg, com)
  _apply_body_mass_scale(env, env_ids, torso_cfg, mass_scale.squeeze(-1))
  _apply_pd_scale(env, env_ids, motor_kp, motor_kd)

  env.rma_mass_params[env_ids] = mass_params  # type: ignore[attr-defined]
  env.rma_friction[env_ids] = friction  # type: ignore[attr-defined]
  env.rma_motor_kp[env_ids] = motor_kp  # type: ignore[attr-defined]
  env.rma_motor_kd[env_ids] = motor_kd  # type: ignore[attr-defined]
  env.rma_priv_latent[env_ids] = torch.cat(  # type: ignore[attr-defined]
    [mass_params, friction, motor_kp - 1.0, motor_kd - 1.0], dim=-1
  )

  if getattr(env, "obs_history_buf", None) is not None:
    env.obs_history_buf[env_ids] = 0.0  # type: ignore[attr-defined]


def _uniform(
  n: int, d: int, r: tuple[float, float], device: torch.device
) -> torch.Tensor:
  return (r[1] - r[0]) * torch.rand(n, d, device=device) + r[0]


def _apply_geom_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  friction: torch.Tensor,
) -> None:
  geom_ids = _as_id_list(asset_cfg.geom_ids)
  if not geom_ids:
    return
  field = env.sim.model.geom_friction
  for g in geom_ids:
    field[env_ids, g, 0] = friction


def _apply_body_com(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  com: torch.Tensor,
) -> None:
  body_ids = _as_id_list(asset_cfg.body_ids)
  if not body_ids:
    return
  default = env.sim.get_default_field("body_ipos")
  field = env.sim.model.body_ipos
  for b in body_ids:
    base = default[b].to(device=env.device)
    field[env_ids, b] = base.unsqueeze(0) + com


def _apply_body_mass_scale(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  mass_scale: torch.Tensor,
) -> None:
  body_ids = _as_id_list(asset_cfg.body_ids)
  if not body_ids:
    return
  default = env.sim.get_default_field("body_mass")
  field = env.sim.model.body_mass
  for b in body_ids:
    base = default[b].to(device=env.device)
    field[env_ids, b] = base * mass_scale


def _apply_pd_scale(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  kp_scale: torch.Tensor,
  kd_scale: torch.Tensor,
) -> None:
  """Scale actuator kp/kd from defaults by per-env per-actuator factors."""
  # actuator_gainprm[:, :, 0] is typically kp; biasprm has -kp, -kd layout in
  # position actuators. Mirror mjlab dr.pd_gains scale behaviour when possible.
  if not hasattr(env.sim.model, "actuator_gainprm"):
    return
  gain = env.sim.model.actuator_gainprm
  bias = env.sim.model.actuator_biasprm
  default_gain = env.sim.get_default_field("actuator_gainprm")
  default_bias = env.sim.get_default_field("actuator_biasprm")
  # gainprm: (nenv, nu, …); use first ``na`` actuators.
  na = kp_scale.shape[-1]
  nu = min(na, gain.shape[1])
  for i in range(nu):
    g0 = default_gain[i].to(device=env.device)
    b0 = default_bias[i].to(device=env.device)
    # Position actuator: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd (common).
    kp = kp_scale[:, i]
    kd = kd_scale[:, i]
    gain[env_ids, i, 0] = g0[0] * kp
    if gain.shape[-1] > 1:
      gain[env_ids, i, 1] = g0[1]
    if bias.shape[-1] > 2:
      bias[env_ids, i, 1] = b0[1] * kp
      bias[env_ids, i, 2] = b0[2] * kd
