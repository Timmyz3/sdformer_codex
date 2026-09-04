#!/usr/bin/env bash
set -euo pipefail

task_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_run_root="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
task_run_dir="${RUN_DIR:-${task_run_root}/m21_banked_moment_scheduler_vcs_sva_r1_20260822}"

command -v vcs >/dev/null
if [[ -e "${task_run_dir}/evidence.sha256" || -e "${task_run_dir}/simv" ]]; then
    echo "refusing to overwrite existing M21 VCS evidence: ${task_run_dir}" >&2
    exit 2
fi
mkdir -p "${task_run_dir}"
cd "${task_run_dir}"

vcs -full64 -lca -sverilog -assert svaext -debug_access+all \
    -timescale=1ns/1ps +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
    -top tb_qfit_dynamic_bn_banked_moment_scheduler \
    "${task_hw_root}/rtl_m21/qfit_dynamic_bn_banked_moment_scheduler.sv" \
    "${task_hw_root}/verif_m21/qfit_dynamic_bn_banked_moment_scheduler_assertions.sv" \
    "${task_hw_root}/tb_m21/tb_qfit_dynamic_bn_banked_moment_scheduler.sv" \
    -o simv 2>&1 | tee compile.log
./simv -assert report="${task_run_dir}/assertion_report.txt" \
    +ntb_random_seed=20260822 2>&1 | tee simulation.log

grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "M21_RESULT legal_packets=52 illegal_packets=7 results=66 done=5" simulation.log
grep -q "directed_full_swaps=1 directed_illegal_full_cancels=1 directed_pending_result_cancels=1" simulation.log
grep -q "M21_SVA_COVERAGE legal_packets=52 illegal_packets=7 results=66" simulation.log
grep -q "pending_result_cancels=1" simulation.log
grep -q "PASS: Synopsys VCS M21 banked raw-moment scheduler reference miter" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log assertion_report.txt
task_disable_log="assertion_report.txt.disablelog"
if [[ ! -s "${task_disable_log}" ]]; then
    echo "M21 VCS missing assertion disable log" >&2
    exit 3
fi
mapfile -t task_disable_lines < <(sed '/^[[:space:]]*$/d' "${task_disable_log}")
if [[ "${#task_disable_lines[@]}" -ne 3 \
      || "${task_disable_lines[0]-}" != "Disabled Module Assertions (compiletime)" \
      || "${task_disable_lines[1]-}" != "Assertions disabled via '-assert hier' switch" \
      || "${task_disable_lines[2]-}" != "Dynamically disabled assertions at End-of-Simulation" ]]; then
    echo "M21 VCS disabled one or more concrete assertions" >&2
    sed -n '1,160p' "${task_disable_log}" >&2
    exit 4
fi

mkdir default_parameter_smoke
cd default_parameter_smoke
vcs -full64 -lca -sverilog -timescale=1ns/1ps \
    -top tb_qfit_dynamic_bn_banked_moment_scheduler_default_smoke \
    "${task_hw_root}/rtl_m21/qfit_dynamic_bn_banked_moment_scheduler.sv" \
    "${task_hw_root}/dc_handoff/rtl/date_m21_banked_moment_scheduler_dc_top.sv" \
    "${task_hw_root}/tb_m21/tb_qfit_dynamic_bn_banked_moment_scheduler_default_smoke.sv" \
    -o simv 2>&1 | tee compile.log
./simv 2>&1 | tee simulation.log
grep -q "M21_DEFAULT_SMOKE max_population=4194304 lane_tiles=16 fifo_depth=4 packet_lanes=96 slices=6 count_w=23 sum_w=54 sumsq_w=85 dynamic_population=1 dynamic_results=96 dynamic_done=1" simulation.log
grep -q "directed_packet_backpressure_cycles=3" simulation.log
grep -q "M21_DEFAULT_DYNAMIC .*tile_slice_results=96 arithmetic_lanes_checked=1536" simulation.log
grep -q "PASS: Synopsys VCS M21 default-parameter dynamic 16-tile smoke" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log

cd "${task_run_dir}"
sha256sum \
    "${task_hw_root}/rtl_m21/qfit_dynamic_bn_banked_moment_scheduler.sv" \
    "${task_hw_root}/verif_m21/qfit_dynamic_bn_banked_moment_scheduler_assertions.sv" \
    "${task_hw_root}/tb_m21/tb_qfit_dynamic_bn_banked_moment_scheduler.sv" \
    "${task_hw_root}/tb_m21/tb_qfit_dynamic_bn_banked_moment_scheduler_default_smoke.sv" \
    "${task_hw_root}/dc_handoff/rtl/date_m21_banked_moment_scheduler_dc_top.sv" \
    "${task_hw_root}/dc_handoff/scripts/run_vcs_m21_banked_moment_scheduler_sva.sh" \
    compile.log simulation.log assertion_report.txt \
    assertion_report.txt.disablelog simv \
    default_parameter_smoke/compile.log default_parameter_smoke/simulation.log \
    default_parameter_smoke/simv > evidence.sha256
echo "PASS Synopsys VCS/SVA M21 banked raw-moment scheduler"
