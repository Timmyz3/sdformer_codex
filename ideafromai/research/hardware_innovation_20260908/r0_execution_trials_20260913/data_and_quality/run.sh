#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR="$PWD/cupy_cache"
export TORCH_HOME="$PWD/torch_cache"
# This owned interpreter is Python3.12.7; model_access asserts major/minor.
PY312="$PWD/py312/bin/python"
"$PY312" capture_native.py
"$PY312" prepare_masks.py
"$PY312" evaluate_quality.py --data-root "$PWD/data_mirror" --output-label diverse10
"$PY312" evaluate_nb0.py
