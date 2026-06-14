#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-}"

CMD=(python3 "${ROOT}/scripts/nts07_perf_model.py")
if [[ -n "${CONFIG}" ]]; then
  CMD+=(--config "${CONFIG}")
fi

"${CMD[@]}"