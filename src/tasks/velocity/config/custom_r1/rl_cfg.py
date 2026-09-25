"""RL configuration for custom 29-DoF R1 velocity task."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def custom_r1_ppo_runner_cfg(
  experiment_name: str = "custom_r1_velocity",
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
    logger="wandb",
    wandb_project="humanoid",
  )


@dataclass
class RslRlRmaOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """PPO runner cfg extended with extreme-parkour RMA hyperparameters."""

  rma: dict[str, Any] = field(
    default_factory=lambda: {
      "history_len": 10,
      "num_priv_explicit": 9,
      "num_scan": 0,
      "dagger_update_freq": 20,
      "priv_reg_coef_schedual": [0.0, 0.1, 2000, 3000],
    }
  )
  rma_policy: dict[str, Any] = field(
    default_factory=lambda: {
      "actor_hidden_dims": [512, 256, 128],
      "critic_hidden_dims": [512, 256, 128],
      "priv_encoder_dims": [64, 20],
      "activation": "elu",
      "init_noise_std": 1.0,
      "tanh_encoder_output": False,
    }
  )
  rma_estimator: dict[str, Any] = field(
    default_factory=lambda: {
      "hidden_dims": [128, 64],
      "activation": "elu",
      "learning_rate": 1.0e-4,
      "train_with_estimated_states": True,
    }
  )


def custom_r1_rma_ppo_runner_cfg(
  experiment_name: str = "custom_r1_flat_rma",
) -> RslRlRmaOnPolicyRunnerCfg:
  base = custom_r1_ppo_runner_cfg(experiment_name=experiment_name)
  return RslRlRmaOnPolicyRunnerCfg(
    actor=base.actor,
    critic=base.critic,
    algorithm=base.algorithm,
    experiment_name=experiment_name,
    save_interval=base.save_interval,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=base.max_iterations,
    logger=base.logger,
    wandb_project=base.wandb_project,
  )
