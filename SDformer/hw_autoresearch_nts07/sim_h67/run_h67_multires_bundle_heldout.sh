#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_multires_bundle_heldout_20260809}"
cd "$ROOT"

python3 -m unittest \
  tests.test_profile_h67_zkqi_multisample_ordered \
  tests.test_profile_h67_multires_bundle_heldout -v
python3 scripts/profile_h67_multires_bundle_heldout.py --output-dir "$OUT"

echo "PASS Motion multi-resolution bundle held-out DSE"
