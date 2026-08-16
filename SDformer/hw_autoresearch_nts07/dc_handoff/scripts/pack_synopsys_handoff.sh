#!/usr/bin/env bash
# Pack a copy-to-Synopsys-server tree. Does not run dc_shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PACK_DIR="$ROOT/dc_handoff/packs"
STAGE="$PACK_DIR/stage_$STAMP"
OUT="$PACK_DIR/date_dual_synopsys_handoff_${STAMP}.tar"
LIST="$PACK_DIR/date_dual_synopsys_handoff_${STAMP}.filelist.txt"
CHECKSUM="$OUT.sha256"

mkdir -p "$STAGE" "$PACK_DIR"

# Refresh both fail-closed identity gates before collecting any file.
python3 "$ROOT/dc_handoff/scripts/audit_date_dual_handoff.py" \
  --root "$ROOT" \
  --output "$ROOT/dc_handoff/runs/date_dual_handoff_audit_20260815_v11.json"
python3 "$ROOT/scripts/audit_three_line_predc_gate.py" \
  --root "$ROOT" \
  --output "$ROOT/results/grok_codex_collab/three_line_predc_gate_20260815.json"

need() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "缺少交接文件: $p" >&2
    exit 2
  fi
}

collect_filelist() {
  local fl="$1"
  need "$fl"
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | sed 's/[[:space:]]*$//')"
    [[ -z "$line" ]] && continue
    need "$line"
    echo "$line"
  done < "$fl"
}

{
  echo "dc_handoff/SERVER_RUN.md"
  echo "dc_handoff/PPA_REVIEW_CHECKLIST.md"
  echo "dc_handoff/README.md"
  echo "dc_handoff/run_dc.sh"
  echo "dc_handoff/run_formality.sh"
  echo "dc_handoff/run_ptsta.sh"
  echo "dc_handoff/run_ptpx.sh"
  echo "dc_handoff/run_motion_activity.sh"
  echo "dc_handoff/run_local5_activity.sh"
  echo "dc_handoff/run_local5_1rw_activity.sh"
  echo "dc_handoff/constraints/date_dual_core.sdc"
  echo "dc_handoff/config/date_dual_constraints.yaml"
  echo "dc_handoff/config/saif_manifest.example.json"
  echo "dc_handoff/filelists/date_motion_2s.f"
  echo "dc_handoff/filelists/date_local5_out2.f"
  echo "dc_handoff/filelists/date_local5_out2_1rw.f"
  echo "dc_handoff/scripts/run_dc.tcl"
  echo "dc_handoff/scripts/run_formality.tcl"
  echo "dc_handoff/scripts/run_ptsta.tcl"
  echo "dc_handoff/scripts/run_ptpx.tcl"
  echo "dc_handoff/scripts/audit_date_dual_handoff.py"
  echo "dc_handoff/scripts/audit_dc_artifacts.py"
  echo "dc_handoff/scripts/audit_expected_macro_refs.py"
  echo "dc_handoff/scripts/audit_saif_manifest.py"
  echo "dc_handoff/scripts/audit_synopsys_postrun.py"
  echo "dc_handoff/scripts/make_saif_manifest.py"
  echo "dc_handoff/scripts/write_synopsys_run_manifest.py"
  echo "dc_handoff/scripts/report_activity_vcd.py"
  echo "dc_handoff/scripts/compare_motion_activity_contracts.py"
  echo "dc_handoff/scripts/pack_synopsys_handoff.sh"
  echo "docs/359_DATE终局冻结_20260813.md"
  echo "docs/416_H81创新合同确认与preDC分门_20260815.md"
  echo "docs/417_三线硬件优先级与preDC分门收口_20260815.md"
  echo "docs/418_本机不跑新思_服务器交接包_20260815.md"
  echo "docs/419_Local5当前生产前端跨Head_OUT32闭合_20260815.md"
  echo "docs/420_服务器交接包独立攻击_20260815.md"
  echo "docs/421_三线硬件preDC最终交接与剩余门_20260815.md"
  echo "docs/422_用户澄清本机不跑新思_可拷走剩余项_20260815.md"
  echo "docs/423_ep44_dyadic独立核与claim边界_20260815.md"
  echo "dc_handoff/packs/COPY_THIS.md"
  echo "dc_handoff/scripts/server_run_four_tops.sh"
  echo "scripts/audit_three_line_predc_gate.py"
  echo "scripts/generate_local5_t450_fullchain_oracle.py"
  echo "scripts/report_local5_score_active_cross_head.py"
  echo "sim_qfit/run_local5_score_active_cross_head_checks.sh"
  echo "tests/test_audit_three_line_predc_gate.py"
  echo "tests/test_dc_handoff_admission.py"
  echo "tests/test_make_saif_manifest_portability.py"
  echo "tests/test_report_local5_score_active_cross_head.py"
  echo "results/grok_codex_collab/h81_identity_contract_20260815.json"
  echo "results/grok_codex_collab/three_line_predc_gate_20260815.json"
  echo "dc_handoff/runs/date_dual_handoff_audit_20260815_v11.json"
  echo "dc_handoff/runs/motion_fair_activity_pair_20260814.json"
  echo "results/local5_score_active_cross_head_20260815/iverilog_seed_17717.log"
  echo "results/local5_score_active_cross_head_20260815/verilator_seed_17717.log"
  echo "results/local5_score_active_cross_head_20260815/verilator_shell_prod_lint.log"
  echo "results/local5_score_active_cross_head_20260815/source_sha256.txt"
  echo "results/local5_score_active_cross_head_20260815/report.md"
  echo "results/local5_score_active_cross_head_20260815/report.json"
  collect_filelist dc_handoff/filelists/date_motion_2s.f
  collect_filelist dc_handoff/filelists/date_local5_out2.f
  collect_filelist dc_handoff/filelists/date_local5_out2_1rw.f
  echo "rtl_hitflow/gatestack_output_tile_scheduler.sv"
  echo "rtl_qfit/qfit_local5_tile.sv"
  echo "rtl_qfit/qfit_local5_projection_tile.sv"
  echo "rtl_qfit/qfit_local5_tagged_t450_job_engine.sv"
  echo "rtl_qfit/qfit_fakeram45_acc_memory_90x1024.sv"
  echo "rtl_qfit/qfit_local5_cross_head_tile_executor.sv"
  echo "rtl_qfit/qfit_local5_encoder_job_scheduler.sv"
  echo "rtl_qfit/qfit_local5_encoder_t450_numeric_shell.sv"
  echo "tb_qfit/tb_qfit_local5_cross_head_tile_executor.sv"
  echo "verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv"
  echo "verif_qfit/qfit_retirement_scheduler_assertions.sv"
  echo "verif_qfit/qfit_sync_bank_assertions.sv"
  echo "verif_qfit/qfit_dual_color_relation_frontier_sync_assertions.sv"
  echo "verif_qfit/qfit_source_multicast_assertions.sv"
  echo "verif_qfit/qfit_tcfm5_assertions.sv"
  echo "verif_qfit/qfit_tcfm5_acc_bank_assertions.sv"
  echo "verif_qfit/qfit_local5_score_active_projection_assertions.sv"
  echo "verif_qfit/qfit_local5_tagged_t450_job_engine_assertions.sv"
  echo "verif_qfit/qfit_local5_cross_head_tile_executor_assertions.sv"
  echo "tb_h67/vectors/h67_fullres_ep35_postconvergence_t450_20260805/h67_checkpoint_rows.txt"
  find tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813 \
    -type f -printf '%p\n' | sort
  echo "dc_handoff/runs/motion_fixed_dc_activity_population138_fair/activity_contract.json"
  echo "dc_handoff/runs/motion_fixed_dc_activity_population138_fair/h67_fixed2s_mssb5_dc_top.vcd"
  echo "dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/activity_contract.json"
  echo "dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/h67_rqtb2s_mssb5_dc_top.vcd"
  echo "dc_handoff/runs/local5_dc_activity_full_population100/activity_contract.json"
  echo "dc_handoff/runs/local5_dc_activity_full_population100/local5_unified_out2_dc_top.vcd"
  echo "dc_handoff/runs/local5_1rw_activity_population100_full/activity_contract.json"
  echo "dc_handoff/runs/local5_1rw_activity_population100_full/local5_unified_out2_1rw_dc_top.vcd"
} | awk 'NF && !seen[$0]++' > "$LIST"

missing=0
while IFS= read -r rel; do
  if [[ ! -e "$ROOT/$rel" ]]; then
    echo "缺少: $rel" >&2
    missing=1
  fi
done < "$LIST"
if [[ "$missing" -ne 0 ]]; then
  exit 3
fi

# Stage with the hw_autoresearch_nts07 prefix so the server unpacks cleanly.
mkdir -p "$STAGE/hw_autoresearch_nts07"
while IFS= read -r rel; do
  dest="$STAGE/hw_autoresearch_nts07/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -a "$ROOT/$rel" "$dest"
done < "$LIST"

mkdir -p "$STAGE/hw_autoresearch_nts07/dc_handoff/packs"
cp -a "$LIST" "$STAGE/hw_autoresearch_nts07/dc_handoff/packs/$(basename "$LIST")"

tar -C "$STAGE" -cf "$OUT" hw_autoresearch_nts07
rm -rf "$STAGE"
(
  cd "$PACK_DIR"
  sha256sum "$(basename "$OUT")" > "$(basename "$CHECKSUM")"
)

# Prove that the isolated archive, not merely the live workspace, is auditable.
VERIFY_DIR="$(mktemp -d "$PACK_DIR/verify_${STAMP}_XXXXXX")"
tar -C "$VERIFY_DIR" -xf "$OUT"
if tar -tf "$OUT" | grep -q '/obj/'; then
  echo "归档中禁止出现 obj/ 编译目录" >&2
  exit 4
fi

yosys_hierarchy_check() {
  local verify_root="$1"
  local filelist="$2"
  local top="$3"
  local log="$VERIFY_DIR/${top}_yosys.log"
  local sources
  sources="$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' \
    "$verify_root/$filelist" | tr '\n' ' ')"
  if ! (
    cd "$verify_root"
    yosys -q -l "$log" -p \
      "read_verilog -sv $sources; hierarchy -check -top $top; proc; opt; check -assert"
  ) >/dev/null 2>&1; then
    tail -n 80 "$log" >&2
    return 1
  fi
}

saif_identity_roundtrip() {
  local verify_root="$1"
  local design="$2"
  local contract="$3"
  local activity="$4"
  local strip_path="$5"
  local manifest="$VERIFY_DIR/${design}_identity_manifest.json"

  # The activity file is used only as an opaque identity payload here. Actual
  # VCD-to-SAIF conversion and SAIF syntax validation remain server-side.
  python3 "$verify_root/dc_handoff/scripts/make_saif_manifest.py" \
    --root "$verify_root" \
    --activity-contract "$verify_root/$contract" \
    --saif "$verify_root/$activity" \
    --output "$manifest" >/dev/null
  python3 "$verify_root/dc_handoff/scripts/audit_saif_manifest.py" \
    --design "$design" \
    --saif "$verify_root/$activity" \
    --strip-path "$strip_path" \
    --manifest "$manifest" \
    --require-paper-power-eligible >/dev/null
}

command -v yosys >/dev/null 2>&1 || {
  echo "本机打包前必须有 yosys 用于隔离层次检查" >&2
  exit 5
}
ARCHIVE_ROOT="$VERIFY_DIR/hw_autoresearch_nts07"
yosys_hierarchy_check "$ARCHIVE_ROOT" \
  dc_handoff/filelists/date_motion_2s.f h67_fixed2s_mssb5_dc_top
yosys_hierarchy_check "$ARCHIVE_ROOT" \
  dc_handoff/filelists/date_motion_2s.f h67_rqtb2s_mssb5_dc_top
yosys_hierarchy_check "$ARCHIVE_ROOT" \
  dc_handoff/filelists/date_local5_out2.f local5_unified_out2_dc_top
yosys_hierarchy_check "$ARCHIVE_ROOT" \
  dc_handoff/filelists/date_local5_out2_1rw.f local5_unified_out2_1rw_dc_top
python3 "$VERIFY_DIR/hw_autoresearch_nts07/dc_handoff/scripts/audit_date_dual_handoff.py" \
  --root "$VERIFY_DIR/hw_autoresearch_nts07" \
  --output "$VERIFY_DIR/archive_handoff_audit.json"
python3 "$VERIFY_DIR/hw_autoresearch_nts07/scripts/audit_three_line_predc_gate.py" \
  --root "$VERIFY_DIR/hw_autoresearch_nts07" \
  --output "$VERIFY_DIR/archive_three_line_gate.json"
saif_identity_roundtrip "$ARCHIVE_ROOT" \
  h67_fixed2s_mssb5_dc_top \
  dc_handoff/runs/motion_fixed_dc_activity_population138_fair/activity_contract.json \
  dc_handoff/runs/motion_fixed_dc_activity_population138_fair/h67_fixed2s_mssb5_dc_top.vcd \
  TOP/tb_h67_motion_dc_activity/g_fixed/dut
saif_identity_roundtrip "$ARCHIVE_ROOT" \
  h67_rqtb2s_mssb5_dc_top \
  dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/activity_contract.json \
  dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/h67_rqtb2s_mssb5_dc_top.vcd \
  TOP/tb_h67_motion_dc_activity/g_rqtb/dut
saif_identity_roundtrip "$ARCHIVE_ROOT" \
  local5_unified_out2_dc_top \
  dc_handoff/runs/local5_dc_activity_full_population100/activity_contract.json \
  dc_handoff/runs/local5_dc_activity_full_population100/local5_unified_out2_dc_top.vcd \
  TOP/tb_qfit_local5_score_projection_postg0/g_dc_wrapper/dut
saif_identity_roundtrip "$ARCHIVE_ROOT" \
  local5_unified_out2_1rw_dc_top \
  dc_handoff/runs/local5_1rw_activity_population100_full/activity_contract.json \
  dc_handoff/runs/local5_1rw_activity_population100_full/local5_unified_out2_1rw_dc_top.vcd \
  TOP/tb_qfit_local5_score_projection_postg0/g_dc_wrapper/dut
rm -rf "$VERIFY_DIR"

python3 - <<PY
from pathlib import Path
out = Path("$OUT")
lst = Path("$LIST")
n = sum(1 for _ in lst.read_text().splitlines() if _.strip())
print(f"packed {n} files -> {out} ({out.stat().st_size} bytes)")
print(f"sha256 -> {out}.sha256")
print("isolated archive audit/gate, four-top Yosys, and four activity identity round-trips: PASS")
print("this machine does not run dc_shell; copy the tar to the Synopsys server")
print("see dc_handoff/SERVER_RUN.md")
PY
