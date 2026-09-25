"""RMA actor-critic and velocity estimator (ported from extreme-parkour).

Observation layout (num_scan=0 for plane-no-scandot):
  [proprio | priv_explicit | priv_latent | history]
history encodes proprio over ``num_hist`` steps into the same latent space as
``priv_encoder(priv_latent)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name: str) -> nn.Module:
  if act_name == "elu":
    return nn.ELU()
  if act_name == "selu":
    return nn.SELU()
  if act_name == "relu":
    return nn.ReLU()
  if act_name == "lrelu":
    return nn.LeakyReLU()
  if act_name == "tanh":
    return nn.Tanh()
  if act_name == "sigmoid":
    return nn.Sigmoid()
  raise ValueError(f"invalid activation function: {act_name}")


class StateHistoryEncoder(nn.Module):
  def __init__(
    self,
    activation_fn: nn.Module,
    input_size: int,
    tsteps: int,
    output_size: int,
  ):
    super().__init__()
    self.activation_fn = activation_fn
    self.tsteps = tsteps
    channel_size = 10
    self.encoder = nn.Sequential(
      nn.Linear(input_size, 3 * channel_size),
      self.activation_fn,
    )
    if tsteps == 50:
      self.conv_layers = nn.Sequential(
        nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=8, stride=4),
        self.activation_fn,
        nn.Conv1d(2 * channel_size, channel_size, kernel_size=5, stride=1),
        self.activation_fn,
        nn.Conv1d(channel_size, channel_size, kernel_size=5, stride=1),
        self.activation_fn,
        nn.Flatten(),
      )
    elif tsteps == 10:
      self.conv_layers = nn.Sequential(
        nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=4, stride=2),
        self.activation_fn,
        nn.Conv1d(2 * channel_size, channel_size, kernel_size=2, stride=1),
        self.activation_fn,
        nn.Flatten(),
      )
    elif tsteps == 20:
      self.conv_layers = nn.Sequential(
        nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=6, stride=2),
        self.activation_fn,
        nn.Conv1d(2 * channel_size, channel_size, kernel_size=4, stride=2),
        self.activation_fn,
        nn.Flatten(),
      )
    else:
      raise ValueError("tsteps must be 10, 20 or 50")
    self.linear_output = nn.Sequential(
      nn.Linear(channel_size * 3, output_size),
      self.activation_fn,
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    # obs: (nd, T, n_proprio)
    nd = obs.shape[0]
    t = self.tsteps
    projection = self.encoder(obs.reshape(nd * t, -1))
    output = self.conv_layers(projection.reshape(nd, t, -1).permute(0, 2, 1))
    return self.linear_output(output)


class Actor(nn.Module):
  def __init__(
    self,
    num_prop: int,
    num_scan: int,
    num_actions: int,
    scan_encoder_dims: list[int] | None,
    actor_hidden_dims: list[int],
    priv_encoder_dims: list[int],
    num_priv_latent: int,
    num_priv_explicit: int,
    num_hist: int,
    activation: nn.Module,
    tanh_encoder_output: bool = False,
  ) -> None:
    super().__init__()
    self.num_prop = num_prop
    self.num_scan = num_scan
    self.num_hist = num_hist
    self.num_actions = num_actions
    self.num_priv_latent = num_priv_latent
    self.num_priv_explicit = num_priv_explicit
    self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

    if len(priv_encoder_dims) > 0:
      layers: list[nn.Module] = [
        nn.Linear(num_priv_latent, priv_encoder_dims[0]),
        activation,
      ]
      for i in range(len(priv_encoder_dims) - 1):
        layers.append(nn.Linear(priv_encoder_dims[i], priv_encoder_dims[i + 1]))
        layers.append(activation)
      self.priv_encoder = nn.Sequential(*layers)
      priv_encoder_output_dim = priv_encoder_dims[-1]
    else:
      self.priv_encoder = nn.Identity()
      priv_encoder_output_dim = num_priv_latent

    self.history_encoder = StateHistoryEncoder(
      activation, num_prop, num_hist, priv_encoder_output_dim
    )

    if self.if_scan_encode:
      assert scan_encoder_dims is not None
      scan_layers: list[nn.Module] = [
        nn.Linear(num_scan, scan_encoder_dims[0]),
        activation,
      ]
      for i in range(len(scan_encoder_dims) - 1):
        scan_layers.append(
          nn.Linear(scan_encoder_dims[i], scan_encoder_dims[i + 1])
        )
        scan_layers.append(
          nn.Tanh() if i == len(scan_encoder_dims) - 2 else activation
        )
      self.scan_encoder = nn.Sequential(*scan_layers)
      self.scan_encoder_output_dim = scan_encoder_dims[-1]
    else:
      self.scan_encoder = nn.Identity()
      self.scan_encoder_output_dim = num_scan

    backbone_in = (
      num_prop
      + self.scan_encoder_output_dim
      + num_priv_explicit
      + priv_encoder_output_dim
    )
    actor_layers: list[nn.Module] = [
      nn.Linear(backbone_in, actor_hidden_dims[0]),
      activation,
    ]
    for i in range(len(actor_hidden_dims)):
      if i == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
      else:
        actor_layers.append(
          nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1])
        )
        actor_layers.append(activation)
    if tanh_encoder_output:
      actor_layers.append(nn.Tanh())
    self.actor_backbone = nn.Sequential(*actor_layers)

  def forward(
    self,
    obs: torch.Tensor,
    hist_encoding: bool,
    scandots_latent: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if self.if_scan_encode:
      obs_scan = obs[:, self.num_prop : self.num_prop + self.num_scan]
      scan_latent = (
        scandots_latent
        if scandots_latent is not None
        else self.scan_encoder(obs_scan)
      )
      obs_prop_scan = torch.cat([obs[:, : self.num_prop], scan_latent], dim=1)
    else:
      obs_prop_scan = obs[:, : self.num_prop + self.num_scan]

    start = self.num_prop + self.num_scan
    obs_priv_explicit = obs[:, start : start + self.num_priv_explicit]
    latent = (
      self.infer_hist_latent(obs) if hist_encoding else self.infer_priv_latent(obs)
    )
    return self.actor_backbone(
      torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
    )

  def infer_priv_latent(self, obs: torch.Tensor) -> torch.Tensor:
    start = self.num_prop + self.num_scan + self.num_priv_explicit
    priv = obs[:, start : start + self.num_priv_latent]
    return self.priv_encoder(priv)

  def infer_hist_latent(self, obs: torch.Tensor) -> torch.Tensor:
    hist = obs[:, -self.num_hist * self.num_prop :]
    return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))


class ActorCriticRMA(nn.Module):
  is_recurrent = False

  def __init__(
    self,
    num_prop: int,
    num_scan: int,
    num_critic_obs: int,
    num_priv_latent: int,
    num_priv_explicit: int,
    num_hist: int,
    num_actions: int,
    scan_encoder_dims: list[int] | None = None,
    actor_hidden_dims: list[int] | None = None,
    critic_hidden_dims: list[int] | None = None,
    priv_encoder_dims: list[int] | None = None,
    activation: str = "elu",
    init_noise_std: float = 1.0,
    tanh_encoder_output: bool = False,
    **kwargs,
  ):
    super().__init__()
    if kwargs:
      print(
        "ActorCriticRMA: ignoring unexpected kwargs: "
        + str(list(kwargs.keys()))
      )
    actor_hidden_dims = actor_hidden_dims or [512, 256, 128]
    critic_hidden_dims = critic_hidden_dims or [512, 256, 128]
    priv_encoder_dims = priv_encoder_dims or [64, 20]
    act = get_activation(activation)

    self.actor = Actor(
      num_prop,
      num_scan,
      num_actions,
      scan_encoder_dims,
      actor_hidden_dims,
      priv_encoder_dims,
      num_priv_latent,
      num_priv_explicit,
      num_hist,
      act,
      tanh_encoder_output=tanh_encoder_output,
    )

    critic_layers: list[nn.Module] = [
      nn.Linear(num_critic_obs, critic_hidden_dims[0]),
      act,
    ]
    for i in range(len(critic_hidden_dims)):
      if i == len(critic_hidden_dims) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
      else:
        critic_layers.append(
          nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1])
        )
        critic_layers.append(act)
    self.critic = nn.Sequential(*critic_layers)

    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    self.distribution: Normal | None = None
    Normal.set_default_validate_args(False)

  def reset(self, dones=None) -> None:
    pass

  @property
  def action_mean(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def action_std(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.stddev

  @property
  def entropy(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.entropy().sum(dim=-1)

  def update_distribution(self, observations: torch.Tensor, hist_encoding: bool) -> None:
    mean = self.actor(observations, hist_encoding)
    self.distribution = Normal(mean, mean * 0.0 + self.std)

  def act(self, observations: torch.Tensor, hist_encoding: bool = False, **kwargs):
    self.update_distribution(observations, hist_encoding)
    assert self.distribution is not None
    return self.distribution.sample()

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(actions).sum(dim=-1)

  def act_inference(
    self,
    observations: torch.Tensor,
    hist_encoding: bool = False,
    scandots_latent: torch.Tensor | None = None,
  ) -> torch.Tensor:
    return self.actor(observations, hist_encoding, scandots_latent=scandots_latent)

  def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
    return self.critic(critic_observations)


class Estimator(nn.Module):
  """Velocity / privileged-explicit state estimator from proprioception."""

  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | None = None,
    activation: str = "elu",
    **kwargs,
  ):
    super().__init__()
    hidden_dims = hidden_dims or [128, 64]
    act = get_activation(activation)
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), act]
    for i in range(len(hidden_dims)):
      if i == len(hidden_dims) - 1:
        layers.append(nn.Linear(hidden_dims[i], output_dim))
      else:
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(act)
    self.estimator = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.estimator(x)

  def inference(self, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      return self.estimator(x)
