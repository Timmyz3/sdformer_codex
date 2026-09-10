#!/usr/bin/env bash
# GROKBOT: include C2 ablation ladder in regress (stable counter TB)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
bash "$ROOT/flows/oss/ablation_c2_ladder.sh"
