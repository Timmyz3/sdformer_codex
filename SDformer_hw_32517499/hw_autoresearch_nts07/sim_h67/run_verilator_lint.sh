#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
verilator --lint-only --timing -Wall -Wno-fatal \
  -f rtl_h67/filelist.f \
  --top-module h67_attention_top
