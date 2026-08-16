#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /opt/conda/bin/python3.11 "${ROOT}/scripts/local5_erep_statistics_v4.py"
