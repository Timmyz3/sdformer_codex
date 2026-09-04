#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDPARAM \
  "$ROOT/rtl_dc/unibin_h60_core_dc.sv"
