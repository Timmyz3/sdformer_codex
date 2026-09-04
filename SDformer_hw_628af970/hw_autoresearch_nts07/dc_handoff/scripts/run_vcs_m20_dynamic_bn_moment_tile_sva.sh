#!/usr/bin/env bash
set -euo pipefail

task_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_run_root="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
task_run_dir="${RUN_DIR:-${task_run_root}/m20_dynamic_bn_moment_tile_vcs_sva_r5_default_smoke_20260822}"

if [[ -e "${task_run_dir}/evidence.sha256" || -e "${task_run_dir}/simv" ]]; then
    echo "refusing to overwrite existing M20 VCS evidence: ${task_run_dir}" >&2
    exit 2
fi
mkdir -p "${task_run_dir}"
cd "${task_run_dir}"

vcs -full64 -lca -sverilog -assert svaext -debug_access+all -timescale=1ns/1ps \
    +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
    -top tb_qfit_dynamic_bn_moment_tile \
    "${task_hw_root}/rtl_m20/qfit_dynamic_bn_moment_tile.sv" \
    "${task_hw_root}/dc_handoff/rtl/date_m20_dynamic_bn_moment_tile_dc_top.sv" \
    "${task_hw_root}/verif_m20/qfit_dynamic_bn_moment_tile_assertions.sv" \
    "${task_hw_root}/tb_m20/tb_qfit_dynamic_bn_moment_tile.sv" \
    -o simv 2>&1 | tee compile.log
./simv -assert report="${task_run_dir}/assertion_report.txt" \
    +ntb_random_seed=20260822 2>&1 | tee simulation.log

grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "M20_RESULT transactions=12 legal_beats=1105 illegal=7" simulation.log
grep -q "M20_SVA_COVERAGE legal=1105 illegal=7 final=12" simulation.log
grep -q "PASS: Synopsys VCS M20 exact 16-lane dynamic-BN moment tile miter" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log assertion_report.txt

mkdir default_parameter_smoke
cd default_parameter_smoke
vcs -full64 -lca -sverilog -timescale=1ns/1ps \
    -top tb_qfit_dynamic_bn_moment_tile_default_smoke \
    "${task_hw_root}/rtl_m20/qfit_dynamic_bn_moment_tile.sv" \
    "${task_hw_root}/dc_handoff/rtl/date_m20_dynamic_bn_moment_tile_dc_top.sv" \
    "${task_hw_root}/tb_m20/tb_qfit_dynamic_bn_moment_tile_default_smoke.sv" \
    -o simv 2>&1 | tee compile.log
./simv 2>&1 | tee simulation.log
grep -q "M20_DEFAULT_SMOKE max_population=4194304 count_w=23 sum_w=54 sumsq_w=85 above_max=PASS midflight_reset=PASS result_reset=PASS early_ready=PASS" simulation.log
grep -q "PASS: Synopsys VCS M20 default-parameter elaboration and protocol smoke" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log
cd "${task_run_dir}"
sha256sum \
    "${task_hw_root}/rtl_m20/qfit_dynamic_bn_moment_tile.sv" \
    "${task_hw_root}/dc_handoff/rtl/date_m20_dynamic_bn_moment_tile_dc_top.sv" \
    "${task_hw_root}/verif_m20/qfit_dynamic_bn_moment_tile_assertions.sv" \
    "${task_hw_root}/tb_m20/tb_qfit_dynamic_bn_moment_tile.sv" \
    "${task_hw_root}/tb_m20/tb_qfit_dynamic_bn_moment_tile_default_smoke.sv" \
    "${task_hw_root}/dc_handoff/scripts/run_vcs_m20_dynamic_bn_moment_tile_sva.sh" \
    compile.log simulation.log assertion_report.txt simv \
    default_parameter_smoke/compile.log default_parameter_smoke/simulation.log \
    default_parameter_smoke/simv > evidence.sha256
echo "PASS Synopsys VCS/SVA M20 dynamic-BN moment tile"
