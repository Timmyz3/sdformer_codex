#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR="$PWD/cupy_cache"
export TORCH_HOME="$PWD/torch_cache"
PY312="$PWD/../../r0_execution_trials_20260913/data_and_quality/py312/bin/python"
"$PY312" capture_grid.py
"$PY312" prepare_controls.py
"$PY312" evaluate_controls.py --fusion-npz "$PWD/../selector/mixed_retirement25_q16.npz"
