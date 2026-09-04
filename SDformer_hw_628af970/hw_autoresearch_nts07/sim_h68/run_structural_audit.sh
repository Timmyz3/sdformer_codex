#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if rg -n "matrix_aux|castling_matrix_aux|N[[:space:]]*x[[:space:]]*N" rtl_h68 rtl_h68/filelist.f; then
  echo "FAIL: H68部署RTL中发现训练期矩阵辅助结构" >&2
  exit 1
fi

if ! rg -n "ENABLE_MOTION_XOR\(1'b0\)" rtl_h68/h68_castling_deploy_top.sv >/dev/null; then
  echo "FAIL: H68部署顶层没有在编译时关闭Motion-XOR" >&2
  exit 1
fi

echo "PASS: H68部署结构不含训练期矩阵辅助，且Motion-XOR已编译时关闭"
