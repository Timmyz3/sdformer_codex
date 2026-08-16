#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 dc_handoff/scripts/audit_sdc.py
./sim_h67/run_all_checks.sh
./sim_h68/run_all_checks.sh
for design in h67_attention_top h68_castling_deploy_top; do
  DESIGN_NAME="$design" dc_handoff/scripts/run_yosys_generic.sh
done

if [[ "${RUN_LEC:-0}" == "1" ]]; then
  for design in h67_attention_top h68_castling_deploy_top; do
    DESIGN_NAME="$design" dc_handoff/scripts/run_yosys_lec.sh
  done
else
  echo "提示：顺序LEC尚未关闭；设置RUN_LEC=1可显式运行该门槛。"
fi

echo "通过：H67/H68仿真、断言、网表回灌、静态检查和通用综合完成"
