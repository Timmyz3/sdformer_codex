#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-configs/model_variants/variant_c.yaml}"
shift || true

python -m src.utils.profiler --config "${CONFIG_PATH}" "$@"

