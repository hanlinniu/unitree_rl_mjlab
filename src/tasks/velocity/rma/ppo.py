"""RMA PPO with priv_reg, dagger history adaptation, and velocity estimator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from src.tasks.velocity.rma.modules import ActorCriticRMA, Estimator
from src.tasks.velocity.rma.storage import RolloutStorage


class PPORMA:
  """PPO + RMA adaptation (priv encoder ↔ history encoder) + velocity estimator."""

  actor_critic: ActorCriticRMA

  def __init__(
    self,
    actor_critic: ActorCriticRMA,
    estimator: Estimator,
    estimator_paras: dict,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    device: str = "cpu",
    dagger_update_freq: int = 20,
    priv_reg_coef_schedual: list[float] | tuple[float, ...] | None = None,
    **kwargs,
  ):
    self.device = device
    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate

    self.actor_critic = actor_critic.to(device)
    self.storage: RolloutStorage | None = None
    self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    self.transition = RolloutStorage.Transition()

    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    self.hist_encoder_optimizer = optim.Adam(
      self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate
    )
    self.priv_reg_coef_schedual = list(
      priv_reg_coef_schedual or [0.0, 0.1, 2000, 3000]
    )
    self.counter = 0
    self.dagger_update_freq = dagger_update_freq

    self.estimator = estimator.to(device)
    self.priv_states_dim = estimator_paras["priv_states_dim"]
    self.num_prop = estimator_paras["num_prop"]
    self.num_scan = estimator_paras["num_scan"]
    self.estimator_optimizer = optim.Adam(
      self.estimator.parameters(), lr=estimator_paras["learning_rate"]
    )
    self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

  def init_storage(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    actor_obs_shape: list[int],
    critic_obs_shape: list[int],
    action_shape: list[int],
  ) -> None:
    self.storage = RolloutStorage(
      num_envs,
      num_transitions_per_env,
      actor_obs_shape,
      critic_obs_shape,
      action_shape,
      self.device,
    )

  def train_mode(self) -> None:
    self.actor_critic.train()
    self.estimator.train()

  def eval_mode(self) -> None:
    self.actor_critic.eval()
    self.estimator.eval()

  def act(
    self,
    obs: torch.Tensor,
    critic_obs: torch.Tensor,
    hist_encoding: bool = False,
  ) -> torch.Tensor:
    if self.train_with_estimated_states:
      obs_est = obs.clone()
      priv_est = self.estimator(obs_est[:, : self.num_prop])
      start = self.num_prop + self.num_scan
      obs_est[:, start : start + self.priv_states_dim] = priv_est
      self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
    else:
      self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()

    self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
      self.transition.actions
    ).detach()
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    self.transition.observations = obs
    self.transition.critic_observations = critic_obs
    return self.transition.actions

  def process_env_step(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    infos: dict,
  ) -> None:
    assert self.storage is not None
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in infos:
      time_outs = infos["time_outs"].to(self.device)
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * time_outs.unsqueeze(1), 1
      )
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.actor_critic.reset(dones)

  def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
    assert self.storage is not None
    last_values = self.actor_critic.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)

  def update(self) -> dict[str, float]:
    assert self.storage is not None
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_estimator_loss = 0.0
    mean_priv_reg_loss = 0.0
    priv_reg_coef = 0.0

    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for (
      obs_batch,
      critic_obs_batch,
      actions_batch,
      target_values_batch,
      advantages_batch,
      returns_batch,
      old_actions_log_prob_batch,
      old_mu_batch,
      old_sigma_batch,
    ) in generator:
      # Teacher path (priv encoder) for PPO update.
      self.actor_critic.act(obs_batch, hist_encoding=False)
      actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
      value_batch = self.actor_critic.evaluate(critic_obs_batch)
      mu_batch = self.actor_critic.action_mean
      sigma_batch = self.actor_critic.action_std
      entropy_batch = self.actor_critic.entropy

      priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
      with torch.inference_mode():
        hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
      priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(
        p=2, dim=1
      ).mean()
      stage = min(
        max((self.counter - self.priv_reg_coef_schedual[2]), 0)
        / max(self.priv_reg_coef_schedual[3], 1e-8),
        1.0,
      )
      priv_reg_coef = stage * (
        self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]
      ) + self.priv_reg_coef_schedual[0]

      # Velocity / priv-explicit estimator.
      priv_pred = self.estimator(obs_batch[:, : self.num_prop])
      start = self.num_prop + self.num_scan
      priv_tgt = obs_batch[:, start : start + self.priv_states_dim]
      estimator_loss = (priv_pred - priv_tgt).pow(2).mean()
      self.estimator_optimizer.zero_grad()
      estimator_loss.backward()
      nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
      self.estimator_optimizer.step()

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (
              torch.square(old_sigma_batch)
              + torch.square(old_mu_batch - mu_batch)
            )
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(
        actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
      )
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (
          value_batch - target_values_batch
        ).clamp(-self.clip_param, self.clip_param)
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
        + priv_reg_coef * priv_reg_loss
      )
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_estimator_loss += estimator_loss.item()
      mean_priv_reg_loss += priv_reg_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    self.counter += 1
    return {
      "value_function": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "estimator": mean_estimator_loss / num_updates,
      "priv_reg": mean_priv_reg_loss / num_updates,
      "priv_reg_coef": priv_reg_coef,
    }

  def update_dagger(self) -> float:
    """Train history encoder to match priv encoder (student adaptation)."""
    assert self.storage is not None
    mean_hist_latent_loss = 0.0
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for (
      obs_batch,
      _critic_obs_batch,
      _actions_batch,
      _target_values_batch,
      _advantages_batch,
      _returns_batch,
      _old_actions_log_prob_batch,
      _old_mu_batch,
      _old_sigma_batch,
    ) in generator:
      with torch.inference_mode():
        self.actor_critic.act(obs_batch, hist_encoding=True)
        priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
      hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
      hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(
        p=2, dim=1
      ).mean()
      self.hist_encoder_optimizer.zero_grad()
      hist_latent_loss.backward()
      nn.utils.clip_grad_norm_(
        self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm
      )
      self.hist_encoder_optimizer.step()
      mean_hist_latent_loss += hist_latent_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    self.counter += 1
    return mean_hist_latent_loss / num_updates

  def save(self) -> dict:
    return {
      "actor_critic_state_dict": self.actor_critic.state_dict(),
      "estimator_state_dict": self.estimator.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "estimator_optimizer_state_dict": self.estimator_optimizer.state_dict(),
      "hist_encoder_optimizer_state_dict": self.hist_encoder_optimizer.state_dict(),
      "learning_rate": self.learning_rate,
      "counter": self.counter,
    }

  def load(self, loaded: dict, load_optimizer: bool = True) -> None:
    self.actor_critic.load_state_dict(loaded["actor_critic_state_dict"])
    self.estimator.load_state_dict(loaded["estimator_state_dict"])
    if load_optimizer:
      self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
      self.estimator_optimizer.load_state_dict(
        loaded["estimator_optimizer_state_dict"]
      )
      if "hist_encoder_optimizer_state_dict" in loaded:
        self.hist_encoder_optimizer.load_state_dict(
          loaded["hist_encoder_optimizer_state_dict"]
        )
    if "learning_rate" in loaded:
      self.learning_rate = loaded["learning_rate"]
    if "counter" in loaded:
      self.counter = loaded["counter"]
