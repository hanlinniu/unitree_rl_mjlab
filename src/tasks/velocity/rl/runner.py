import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from src.utils.training_backup import backup_after_checkpoint, upload_config_artifact


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper
  _config_artifact_uploaded: bool = False

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    """Start W&B, upload configs immediately, then run the normal training loop."""
    original_init = self.logger.init_logging_writer

    def _init_and_backup() -> None:
      original_init()
      self._upload_configs_once()

    self.logger.init_logging_writer = _init_and_backup  # type: ignore[method-assign]
    try:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      self.logger.init_logging_writer = original_init  # type: ignore[method-assign]

  def _upload_configs_once(self) -> None:
    if self._config_artifact_uploaded:
      return
    if getattr(self.logger, "logger_type", None) != "wandb":
      return
    log_dir = self.logger.log_dir
    if not log_dir:
      return
    try:
      upload_config_artifact(log_dir)
      self._config_artifact_uploaded = True
    except Exception as exc:
      print(f"[WARN] W&B config artifact upload failed: {exc}")

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      try:
        backup_after_checkpoint(
          policy_path,
          path,
          iteration=self.current_learning_iteration,
        )
        self._config_artifact_uploaded = True
      except Exception as exc:
        print(f"[WARN] W&B backup failed: {exc}")
        try:
          wandb.save(path, base_path=os.path.dirname(path))
          wandb.save(onnx_path, base_path=os.path.dirname(onnx_path))
        except Exception as exc2:
          print(f"[WARN] wandb.save fallback failed: {exc2}")
