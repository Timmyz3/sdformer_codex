#!/usr/bin/env bash
set -euo pipefail

task_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_run_root="${RUN_ROOT:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs}"
task_run_dir="${RUN_DIR:-${task_run_root}/m27_atlif_rank3_exact96_vcs_sva_r4_explicit_add_width_20260822}"

command -v vcs >/dev/null
if [[ -e "${task_run_dir}/evidence.sha256" || -e "${task_run_dir}/simv" ]]; then
    echo "refusing to overwrite existing M27 VCS evidence: ${task_run_dir}" >&2
    exit 2
fi
mkdir -p "${task_run_dir}"
cd "${task_run_dir}"

vcs -full64 -lca -sverilog -assert svaext -debug_access+all \
    -timescale=1ns/1ps +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
    -top tb_qfit_atlif_rank3_exact96_core \
    "${task_hw_root}/rtl_m27/qfit_atlif_rank3_exact96_core.sv" \
    "${task_hw_root}/verif_m27/qfit_atlif_rank3_exact96_core_assertions.sv" \
    "${task_hw_root}/tb_m27/tb_qfit_atlif_rank3_exact96_core.sv" \
    -o simv 2>&1 | tee compile.log
./simv -assert report="${task_run_dir}/assertion_report.txt" \
    +ntb_random_seed=20260822 2>&1 | tee simulation.log

grep -q "SIMULATOR=Synopsys VCS" simulation.log
grep -q "ASSERTIONS=enabled" simulation.log
grep -q "M27_CYCLE_CONTRACT product_issue_cycles=10 accept_to_first_valid=6 accept_to_valid_beats=6,7,8,9,10 accept_to_done=11 arithmetic_cycles_per_unstalled_tile=10 stage1_cycles=5 stage2_cycles=5 transition_bubbles=0" simulation.log
grep -q "M27_RESULT tiles=28 random_tiles=24 beats=140" simulation.log
grep -q "M27_SCOPE multiplier_slots=96 multiplier_width=8x8 requant_shift=8 request_load_cycles=not_modeled result_dma_cycles=not_modeled system_speedup=not_claimed" simulation.log
grep -q "PASS: Synopsys VCS M27 rank-3 exact-96 ATLIF factor tile reference miter" simulation.log
! grep -Eq "Fatal:|^Error:|Assertion failed|failed at" simulation.log assertion_report.txt

task_disable_log="assertion_report.txt.disablelog"
if [[ ! -s "${task_disable_log}" ]]; then
    echo "M27 VCS missing assertion disable log" >&2
    exit 3
fi
mapfile -t task_disable_lines < <(sed '/^[[:space:]]*$/d' "${task_disable_log}")
if [[ "${#task_disable_lines[@]}" -ne 3 \
      || "${task_disable_lines[0]-}" != "Disabled Module Assertions (compiletime)" \
      || "${task_disable_lines[1]-}" != "Assertions disabled via '-assert hier' switch" \
      || "${task_disable_lines[2]-}" != "Dynamically disabled assertions at End-of-Simulation" ]]; then
    echo "M27 VCS disabled one or more concrete assertions" >&2
    sed -n '1,160p' "${task_disable_log}" >&2
    exit 4
fi

cp "${task_hw_root}/contracts/m27_atlif_rank3_exact96_vcs_contract_r1_20260822.json" \
   evidence_manifest.json
sha256sum \
    "${task_hw_root}/rtl_m27/qfit_atlif_rank3_exact96_core.sv" \
    "${task_hw_root}/verif_m27/qfit_atlif_rank3_exact96_core_assertions.sv" \
    "${task_hw_root}/tb_m27/tb_qfit_atlif_rank3_exact96_core.sv" \
    "${task_hw_root}/dc_handoff/filelists/date_m27_atlif_rank3_exact96_vcs.f" \
    "${task_hw_root}/dc_handoff/scripts/run_vcs_m27_atlif_rank3_exact96_sva.sh" \
    evidence_manifest.json compile.log simulation.log assertion_report.txt \
    assertion_report.txt.disablelog simv > evidence.sha256
echo "PASS Synopsys VCS/SVA M27 rank-3 exact-96 ATLIF factor tile"
