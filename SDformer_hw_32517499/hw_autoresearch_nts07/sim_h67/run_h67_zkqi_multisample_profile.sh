#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m unittest tests.test_profile_h67_zkqi_multisample_ordered -v
python3 scripts/profile_h67_zkqi_multisample_ordered.py "$@"

echo "PASS Motion ZKQI multi-sample ordered profile"
