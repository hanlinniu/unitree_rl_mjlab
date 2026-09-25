#!/usr/bin/env python3
"""Download a W&B training run (configs + checkpoints) for local / new-instance resume.

Examples:
  python scripts/download_wandb_run.py \\
    --run ENTITY/PROJECT/RUN_ID \\
    --out ./restored_runs/my_run

  python scripts/download_wandb_run.py pack \\
    --log-dir logs/rsl_rl/custom_r1_flat_velocity/2026-..._run \\
    --out /tmp/run_backup.tar.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.training_backup import download_wandb_run, pack_local_run


def main(argv: list[str] | None = None) -> None:
  argv = list(sys.argv[1:] if argv is None else argv)
  parser = argparse.ArgumentParser(description=__doc__)

  if argv and argv[0] == "pack":
    pack_p = argparse.ArgumentParser(prog="download_wandb_run.py pack")
    pack_p.add_argument("--log-dir", required=True, type=Path)
    pack_p.add_argument("--out", type=Path, default=None)
    args = pack_p.parse_args(argv[1:])
    pack_local_run(args.log_dir, args.out)
    return

  parser.add_argument(
    "--run",
    required=True,
    help="W&B run path: entity/project/run_id (from the run URL)",
  )
  parser.add_argument("--out", required=True, type=Path, help="Output directory")
  parser.add_argument("--no-artifacts", action="store_true")
  parser.add_argument("--no-files", action="store_true")
  args = parser.parse_args(argv)
  download_wandb_run(
    args.run,
    args.out,
    download_artifacts=not args.no_artifacts,
    download_files=not args.no_files,
  )


if __name__ == "__main__":
  main()
