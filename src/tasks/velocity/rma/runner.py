"""On-policy runner for RMA (priv encoder + history encoder + velocity estimator)."""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import torch

from src.tasks.velocity.rma.modules import ActorCriticRMA, Estimator
from src.tasks.velocity.rma.obs_wrapper import RMAObsWrapper
from src.tasks.velocity.rma.ppo import PPORMA


def _cfg_get(cfg: Mapping[str, Any] | Any, key: str, default=None):
  if isinstance(cfg, Mapping):
    return cfg.get(key, default)
  return getattr(cfg, key, default)


class RMAOnPolicyRunner:
  """Train ActorCriticRMA with priv_reg + dagger + velocity estimator (no scandot)."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs,
  ):
    del kwargs
    self.device = device
    self.cfg = train_cfg
    self.log_dir = log_dir
    self.current_learning_iteration = 0

    rma_cfg = train_cfg.get("rma", {})
    history_len = int(rma_cfg.get("history_len", 10))
    num_priv_explicit = int(rma_cfg.get("num_priv_explicit", 9))
    num_scan = int(rma_cfg.get("num_scan", 0))
    assert num_scan == 0

    self.env = RMAObsWrapper(
      env,
      history_len=history_len,
      num_priv_explicit=num_priv_explicit,
      num_scan=num_scan,
    )

    num_prop = self.env.num_prop
    num_priv_latent = self.env.num_priv_latent
    num_critic_obs = self.env.num_critic
    num_actions = self.env.num_actions
    num_obs = self.env.num_obs

    policy_cfg = train_cfg.get("rma_policy", {})
    actor_critic = ActorCriticRMA(
      num_prop=num_prop,
      num_scan=num_scan,
      num_critic_obs=num_critic_obs,
      num_priv_latent=num_priv_latent,
      num_priv_explicit=num_priv_explicit,
      num_hist=history_len,
      num_actions=num_actions,
      scan_encoder_dims=None,
      actor_hidden_dims=list(
        policy_cfg.get("actor_hidden_dims", [512, 256, 128])
      ),
      critic_hidden_dims=list(
        policy_cfg.get("critic_hidden_dims", [512, 256, 128])
      ),
      priv_encoder_dims=list(policy_cfg.get("priv_encoder_dims", [64, 20])),
      activation=policy_cfg.get("activation", "elu"),
      init_noise_std=float(policy_cfg.get("init_noise_std", 1.0)),
      tanh_encoder_output=bool(policy_cfg.get("tanh_encoder_output", False)),
    ).to(device)

    estimator_cfg = train_cfg.get("rma_estimator", {})
    estimator = Estimator(
      input_dim=num_prop,
      output_dim=num_priv_explicit,
      hidden_dims=list(estimator_cfg.get("hidden_dims", [128, 64])),
      activation=estimator_cfg.get("activation", "elu"),
    ).to(device)

    alg_cfg = dict(train_cfg.get("algorithm", {}))
    alg_cfg.pop("class_name", None)
    # Drop fields only used by stock rsl_rl PPO.
    for k in (
      "normalize_advantage_per_mini_batch",
      "optimizer",
      "share_cnn_encoders",
      "rnd_cfg",
      "symmetry_cfg",
    ):
      alg_cfg.pop(k, None)

    estimator_paras = {
      "priv_states_dim": num_priv_explicit,
      "num_prop": num_prop,
      "num_scan": num_scan,
      "learning_rate": float(estimator_cfg.get("learning_rate", 1.0e-4)),
      "train_with_estimated_states": bool(
        estimator_cfg.get("train_with_estimated_states", True)
      ),
    }
    self.dagger_update_freq = int(
      rma_cfg.get("dagger_update_freq", alg_cfg.pop("dagger_update_freq", 20))
    )
    priv_reg = rma_cfg.get(
      "priv_reg_coef_schedual",
      alg_cfg.pop("priv_reg_coef_schedual", [0.0, 0.1, 2000, 3000]),
    )

    self.alg = PPORMA(
      actor_critic,
      estimator,
      estimator_paras,
      device=device,
      dagger_update_freq=self.dagger_update_freq,
      priv_reg_coef_schedual=priv_reg,
      **alg_cfg,
    )
    self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
    self.save_interval = int(train_cfg.get("save_interval", 100))
    self.alg.init_storage(
      self.env.num_envs,
      self.num_steps_per_env,
      [num_obs],
      [num_critic_obs],
      [num_actions],
    )

    self.tot_timesteps = 0
    self.tot_time = 0.0
    self.writer = None
    self._init_writer(train_cfg)

    print(
      f"[RMA] prop={num_prop} scan={num_scan} priv_explicit={num_priv_explicit} "
      f"priv_latent={num_priv_latent} hist={history_len} "
      f"actor_obs={num_obs} critic_obs={num_critic_obs} actions={num_actions}"
    )

  def _init_writer(self, train_cfg: dict) -> None:
    if self.log_dir is None:
      return
    os.makedirs(self.log_dir, exist_ok=True)
    logger = train_cfg.get("logger", "tensorboard")
    if logger == "wandb":
      try:
        import wandb

        wandb.init(
          project=train_cfg.get("wandb_project", "humanoid"),
          name=train_cfg.get("run_name") or os.path.basename(self.log_dir),
          dir=self.log_dir,
          config=train_cfg,
        )
        self.writer = "wandb"
        return
      except Exception as exc:
        print(f"[WARN] wandb init failed ({exc}); falling back to tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

  def add_git_repo_to_log(self, path: str) -> None:
    del path

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    # Params are written by train.py before learn(); upload them now that W&B is live.
    if self.writer == "wandb" and self.log_dir is not None:
      try:
        from src.utils.training_backup import upload_config_artifact

        upload_config_artifact(self.log_dir)
      except Exception as exc:
        print(f"[WARN] W&B config artifact upload failed: {exc}")

    if init_at_random_ep_len:
      self.env.unwrapped.episode_length_buf = torch.randint_like(
        self.env.unwrapped.episode_length_buf,
        high=int(self.env.max_episode_length),
      )

    obs_td = self.env.get_observations().to(self.device)
    actor_obs = obs_td["actor"]
    critic_obs = obs_td["critic"]
    self.alg.train_mode()

    ep_infos: list[dict] = []
    rewbuffer: deque[float] = deque(maxlen=100)
    lenbuffer: deque[float] = deque(maxlen=100)
    cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
    cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      hist_encoding = it % self.dagger_update_freq == 0

      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          actions = self.alg.act(actor_obs, critic_obs, hist_encoding=hist_encoding)
          obs_td, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          obs_td = obs_td.to(self.device)
          rewards = rewards.to(self.device)
          dones = dones.to(self.device)
          actor_obs = obs_td["actor"]
          critic_obs = obs_td["critic"]
          self.alg.process_env_step(rewards, dones, extras)

          if self.log_dir is not None:
            if "log" in extras:
              ep_infos.append(extras["log"])
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(
              cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
            )
            lenbuffer.extend(
              cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
            )
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

        self.alg.compute_returns(critic_obs)

      if hist_encoding:
        hist_loss = self.alg.update_dagger()
        loss_dict = {"hist_latent": hist_loss}
      else:
        loss_dict = self.alg.update()

      stop = time.time()
      self.current_learning_iteration = it
      collection_size = self.num_steps_per_env * self.env.num_envs
      self.tot_timesteps += collection_size
      self.tot_time += stop - start

      if self.log_dir is not None:
        self._log(it, loss_dict, rewbuffer, lenbuffer, ep_infos, stop - start)
      ep_infos.clear()

      if self.log_dir is not None and it % self.save_interval == 0:
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

    if self.log_dir is not None:
      self.save(
        os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
      )

  def _log(
    self,
    it: int,
    loss_dict: dict[str, float],
    rewbuffer: deque,
    lenbuffer: deque,
    ep_infos: list,
    iter_time: float,
  ) -> None:
    locs = {
      "iteration": it,
      "loss": loss_dict,
      "mean_reward": statistics.mean(rewbuffer) if rewbuffer else 0.0,
      "mean_ep_len": statistics.mean(lenbuffer) if lenbuffer else 0.0,
      "iter_time": iter_time,
      "total_timesteps": self.tot_timesteps,
    }
    print(
      f"it={it} reward={locs['mean_reward']:.3f} "
      f"ep_len={locs['mean_ep_len']:.1f} "
      + " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
    )
    if self.writer == "wandb":
      import wandb

      payload = {f"Loss/{k}": v for k, v in loss_dict.items()}
      payload["Train/mean_reward"] = locs["mean_reward"]
      payload["Train/mean_episode_length"] = locs["mean_ep_len"]
      wandb.log(payload, step=it)
    elif self.writer is not None:
      for k, v in loss_dict.items():
        self.writer.add_scalar(f"Loss/{k}", v, it)
      self.writer.add_scalar("Train/mean_reward", locs["mean_reward"], it)
      self.writer.add_scalar("Train/mean_episode_length", locs["mean_ep_len"], it)
      std = self.alg.actor_critic.std.detach().mean().item()
      self.writer.add_scalar("Policy/mean_std", std, it)

  def save(self, path: str, infos=None) -> None:
    saved = self.alg.save()
    saved["iter"] = self.current_learning_iteration
    saved["infos"] = infos
    torch.save(saved, path)
    if self.writer == "wandb" and self.log_dir is not None:
      try:
        from src.utils.training_backup import backup_after_checkpoint

        backup_after_checkpoint(
          self.log_dir, path, iteration=self.current_learning_iteration
        )
      except Exception as exc:
        print(f"[WARN] W&B backup failed: {exc}")
        try:
          import wandb

          wandb.save(path, base_path=self.log_dir)
        except Exception as exc2:
          print(f"[WARN] wandb.save fallback failed: {exc2}")


  def load(self, path: str, load_optimizer: bool = True) -> None:
    loaded = torch.load(path, map_location=self.device, weights_only=False)
    self.alg.load(loaded, load_optimizer=load_optimizer)
    if "iter" in loaded:
      self.current_learning_iteration = loaded["iter"]

  def get_inference_policy(self, device=None):
    self.alg.eval_mode()
    if device is not None:
      self.alg.actor_critic.to(device)

    def policy(obs_td: Any, hist_encoding: bool = True):
      if hasattr(obs_td, "keys") and "actor" in obs_td.keys():
        obs = obs_td["actor"]
      else:
        obs = obs_td
      return self.alg.actor_critic.act_inference(obs, hist_encoding=hist_encoding)

    return policy
