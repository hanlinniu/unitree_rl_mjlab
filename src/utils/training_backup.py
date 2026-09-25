"""Backup / restore training runs via Weights & Biases (and local packs).

When training on a remote instance (e.g. Vast), checkpoints and configs are
uploaded to W&B as **Artifacts** (durable) and mirrored with ``wandb.save``
(Files tab). If the instance dies, restore everything onto another machine:

  python scripts/download_wandb_run.py --run ENTITY/PROJECT/RUN_ID --out ./restored

Then resume:

  python scripts/train.py TASK \\
    --agent.resume=True \\
    --agent.load-run=restored_dir_name \\
    --agent.load-checkpoint=model_XXXX.pt
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# Files required to resume or replay a run.
_CONFIG_GLOBS = (
  "params/env.yaml",
  "params/agent.yaml",
  "params/run_meta.json",
  "run_meta.json",
)
_CHECKPOINT_SUFFIXES = (".pt", ".onnx")


def _wandb_active() -> bool:
  try:
    import wandb

    return wandb.run is not None
  except Exception:
    return False


def write_run_meta(
  log_dir: str | Path,
  *,
  task_id: str,
  experiment_name: str | None = None,
  run_name: str | None = None,
  extra: dict[str, Any] | None = None,
) -> Path:
  """Write ``params/run_meta.json`` with enough info to resume elsewhere."""
  log_dir = Path(log_dir)
  params = log_dir / "params"
  params.mkdir(parents=True, exist_ok=True)

  git_sha = None
  git_branch = None
  git_dirty = None
  try:
    git_sha = (
      subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
      )
      .decode()
      .strip()
    )
    git_branch = (
      subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
      )
      .decode()
      .strip()
    )
    dirty = subprocess.call(
      ["git", "diff", "--quiet"],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
    git_dirty = bool(dirty)
  except Exception:
    pass

  meta = {
    "task_id": task_id,
    "experiment_name": experiment_name,
    "run_name": run_name,
    "log_dir": str(log_dir.resolve()),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_sha": git_sha,
    "git_branch": git_branch,
    "git_dirty": git_dirty,
    "hostname": os.uname().nodename if hasattr(os, "uname") else None,
    "cwd": str(Path.cwd()),
  }
  if extra:
    meta.update(extra)

  path = params / "run_meta.json"
  path.write_text(json.dumps(meta, indent=2) + "\n")
  # Convenience copy at run root.
  (log_dir / "run_meta.json").write_text(path.read_text())
  return path


def _iter_existing(log_dir: Path, rel_paths: Iterable[str]) -> list[Path]:
  out: list[Path] = []
  for rel in rel_paths:
    p = log_dir / rel
    if p.is_file():
      out.append(p)
  return out


def sync_files_to_wandb(log_dir: str | Path, paths: Iterable[str | Path]) -> int:
  """Mirror files into the active W&B run Files tab (``wandb.save``)."""
  if not _wandb_active():
    return 0
  import wandb

  log_dir = Path(log_dir).resolve()
  n = 0
  for p in paths:
    path = Path(p)
    if not path.is_file():
      continue
    try:
      # Keep run-relative layout under the W&B Files tab.
      wandb.save(str(path), base_path=str(log_dir), policy="now")
      n += 1
    except Exception as exc:
      print(f"[WARN] wandb.save failed for {path}: {exc}")
  return n


def upload_config_artifact(log_dir: str | Path, *, name_suffix: str = "config") -> None:
  """Upload params / run_meta as a durable W&B Artifact."""
  if not _wandb_active():
    return
  import wandb

  log_dir = Path(log_dir).resolve()
  files = _iter_existing(log_dir, _CONFIG_GLOBS)
  if not files:
    return

  run = wandb.run
  assert run is not None
  art_name = f"{_safe_name(run.name)}-{name_suffix}"
  art = wandb.Artifact(art_name, type="training-config")
  for f in files:
    art.add_file(str(f), name=str(f.relative_to(log_dir)))
  run.log_artifact(art)
  sync_files_to_wandb(log_dir, files)
  print(f"[INFO] Uploaded W&B config artifact '{art_name}' ({len(files)} files)")


def upload_checkpoint_artifact(
  log_dir: str | Path,
  model_path: str | Path,
  iteration: int | None = None,
) -> None:
  """Upload a checkpoint (+ sidecar configs/onnx) as a durable W&B Artifact."""
  if not _wandb_active():
    return
  import wandb

  log_dir = Path(log_dir).resolve()
  model_path = Path(model_path).resolve()
  if not model_path.is_file():
    return

  files = [model_path]
  # Always attach current configs so a single artifact is enough to resume.
  files.extend(_iter_existing(log_dir, _CONFIG_GLOBS))
  onnx = log_dir / "policy.onnx"
  if onnx.is_file():
    files.append(onnx)

  run = wandb.run
  assert run is not None
  it = iteration if iteration is not None else _parse_iter(model_path.name)
  art_name = f"{_safe_name(run.name)}-ckpt"
  art = wandb.Artifact(
    art_name,
    type="model",
    metadata={"iteration": it, "model_file": model_path.name},
  )
  for f in files:
    art.add_file(str(f), name=str(f.relative_to(log_dir)))
  aliases = ["latest"]
  if it is not None:
    aliases.append(f"iter-{it}")
  run.log_artifact(art, aliases=aliases)
  sync_files_to_wandb(log_dir, files)
  print(
    f"[INFO] Uploaded W&B checkpoint artifact '{art_name}' "
    f"({model_path.name}, aliases={aliases})"
  )


def backup_after_checkpoint(
  log_dir: str | Path,
  model_path: str | Path,
  iteration: int | None = None,
) -> None:
  """Entry point for runners: sync configs + upload checkpoint artifact."""
  if not _wandb_active():
    return
  log_dir = Path(log_dir)
  # Refresh config artifact occasionally is cheap; always attach configs to ckpt.
  upload_config_artifact(log_dir)
  upload_checkpoint_artifact(log_dir, model_path, iteration=iteration)


def download_wandb_run(
  run_path: str,
  out_dir: str | Path,
  *,
  download_artifacts: bool = True,
  download_files: bool = True,
) -> Path:
  """Download a W&B run's files + artifacts into ``out_dir``.

  ``run_path`` format: ``entity/project/run_id`` (or run name resolved by API).
  """
  import wandb

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  api = wandb.Api()
  run = api.run(run_path)

  meta = {
    "wandb_run_path": run_path,
    "wandb_url": run.url,
    "wandb_name": run.name,
    "wandb_id": run.id,
    "wandb_project": run.project,
    "wandb_entity": run.entity,
    "wandb_config": dict(run.config),
    "downloaded_at": datetime.now(timezone.utc).isoformat(),
  }
  (out_dir / "wandb_run_info.json").write_text(json.dumps(meta, indent=2) + "\n")

  if download_files:
    files_dir = out_dir / "files"
    files_dir.mkdir(exist_ok=True)
    for f in run.files():
      try:
        f.download(root=str(files_dir), replace=True)
      except Exception as exc:
        print(f"[WARN] file download failed ({f.name}): {exc}")

  if download_artifacts:
    arts_dir = out_dir / "artifacts"
    arts_dir.mkdir(exist_ok=True)
    for art in run.logged_artifacts():
      try:
        dest = arts_dir / f"{art.type}_{art.name.replace(':', '_')}"
        dest.mkdir(parents=True, exist_ok=True)
        art.download(root=str(dest))
        print(f"[INFO] artifact -> {dest}")
      except Exception as exc:
        print(f"[WARN] artifact download failed ({art.name}): {exc}")

  # Flatten latest checkpoint + params into out_dir root for easy resume.
  _flatten_for_resume(out_dir)
  print(f"[INFO] Restored run into {out_dir.resolve()}")
  return out_dir


def pack_local_run(log_dir: str | Path, out_archive: str | Path | None = None) -> Path:
  """Tar a local run directory (configs + checkpoints + onnx) for offline copy."""
  log_dir = Path(log_dir).resolve()
  if not log_dir.is_dir():
    raise FileNotFoundError(log_dir)
  if out_archive is None:
    out_archive = log_dir.parent / f"{log_dir.name}_backup.tar.gz"
  out_archive = Path(out_archive)

  # Only pack essential files (skip huge wandb/ tensorboard event noise if desired).
  include: list[str] = []
  for pattern in (
    "params/*",
    "run_meta.json",
    "model_*.pt",
    "policy.onnx",
    "git/*",
  ):
    include.extend(str(p.relative_to(log_dir)) for p in log_dir.glob(pattern) if p.is_file())

  # Fallback: pack whole directory if nothing matched.
  if not include:
    shutil.make_archive(str(out_archive).removesuffix(".tar.gz"), "gztar", root_dir=log_dir)
    return out_archive

  # Use tar with explicit file list.
  import tarfile

  with tarfile.open(out_archive, "w:gz") as tar:
    for rel in sorted(set(include)):
      tar.add(log_dir / rel, arcname=rel)
  print(f"[INFO] Packed {len(include)} files -> {out_archive}")
  return out_archive


def _flatten_for_resume(out_dir: Path) -> None:
  """Copy latest model + params to out_dir root so train.py resume paths work."""
  candidates: list[Path] = []
  for p in out_dir.rglob("model_*.pt"):
    candidates.append(p)
  if candidates:
    def _it(p: Path) -> int:
      return _parse_iter(p.name) or -1

    latest = max(candidates, key=_it)
    dest = out_dir / latest.name
    if latest.resolve() != dest.resolve():
      shutil.copy2(latest, dest)
    print(f"[INFO] latest checkpoint: {dest.name}")

  for name in ("env.yaml", "agent.yaml", "run_meta.json"):
    hits = list(out_dir.rglob(name))
    if not hits:
      continue
    params = out_dir / "params"
    params.mkdir(exist_ok=True)
    target = params / name if name != "run_meta.json" else out_dir / "params" / name
    if name == "run_meta.json":
      shutil.copy2(hits[0], out_dir / "run_meta.json")
      shutil.copy2(hits[0], params / "run_meta.json")
    else:
      shutil.copy2(hits[0], target)

  onnx_hits = list(out_dir.rglob("policy.onnx"))
  if onnx_hits:
    shutil.copy2(onnx_hits[0], out_dir / "policy.onnx")


def _parse_iter(name: str) -> int | None:
  # model_1234.pt
  stem = Path(name).stem
  if not stem.startswith("model_"):
    return None
  try:
    return int(stem.split("_", 1)[1])
  except ValueError:
    return None


def _safe_name(name: str) -> str:
  return "".join(c if c.isalnum() or c in "-_" else "-" for c in name)[:64] or "run"
