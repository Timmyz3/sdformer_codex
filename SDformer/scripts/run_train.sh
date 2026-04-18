#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-configs/sdformer_baseline.yaml}"
shift || true

python -m src.trainers.train --config "${CONFIG_PATH}" "$@"

