#!/usr/bin/env bash
# DEPRECATED: H2 training restored to match unitreerobotics/unitree_rl_mjlab.
# This wrapper now trains stock Unitree-H2-Rough (full rough curriculum).
# Prefer: scripts/run_train_h2_upstream_rough.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_train_h2_upstream_rough.sh" "$@"
