#!/usr/bin/env bash
set -euo pipefail

task_review="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_review/../.." && pwd)"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_dc="${DC_HOME:-/opt/synopsys/syn/V-2023.12-SP3}"
task_lib="${LIB_DB:-/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db}"
task_sealed="$task_review/sealed_vcs_replay"
task_hammer="$task_review/independent_vcs"
task_elab="$task_review/dc_elaboration"

if [[ -e "$task_sealed" || -e "$task_hammer" || -e "$task_elab" ]]; then
    echo "refusing to overwrite existing M131 independent evidence" >&2
    exit 2
fi
if pgrep -f 'common_shell_exec.*-shell dc_shell' >/dev/null; then
    echo "refusing to contend with an active dc_shell; retry after it exits" >&2
    exit 3
fi
mkdir "$task_sealed" "$task_hammer" "$task_elab"

cd "$task_hw_root"
declare -A task_expected=(
    ["contracts/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_contract_r1_20260824.json"]="0e657b5916e428fe09df82588479654185055ab734b74a2782fc9b1ec9bae8ba"
    ["contracts/m130_r1_dc_elaboration_failure_correction_r1_20260824.json"]="9164e6b79846cd6017b03592847d54453d3e2cbfa65549e2cbb9ce281b7fc2ef"
    ["contracts/m130_compact_canonical_k4_row_fold_vcs_contract_r1_20260824.json"]="0a67fb7c1466257edc7c6d2cad960565c050916d8456addb3e0330025b8b911b"
    ["rtl_m130/m130_compact_canonical_k4_row_fold.sv"]="ff6d10d2fa341a4ef855f8df196542b990fd71fca34b1b3b81b04c5cb7588e96"
    ["rtl_m131/m131_synthesis_safe_compact_canonical_k4_row_fold.sv"]="82987dd367892213c3f57f0b17b5df4e92603653be9d8a093c9d9b2229cda4ea"
    ["verif_m131/m131_synthesis_safe_compact_canonical_k4_row_fold_assertions.sv"]="17b6493046088f28c6f824e18b3563d703d7c89b4d8d90b6e760135523c79cd4"
    ["tb_m131/tb_m131_synthesis_safe_compact_canonical_k4_row_fold.sv"]="c81d0cd1a12a5860d1712a71bd04d31960008ce3e21a3914618a30c89488c434"
    ["dc_handoff/filelists/date_m131_synthesis_safe_compact_canonical_k4_row_fold_directed_vcs.f"]="f65d8f05819ade452b06a4e8442c47e79ff74a52331afd43c08e63e597fd7013"
    ["dc_handoff/filelists/date_m131_synthesis_safe_compact_canonical_k4_row_fold_logic_only_dc.f"]="6015b365af52a5469e6d4e48661f2916e21c540a3f775e4124d2b2ffec1dced0"
    ["dc_handoff/scripts/run_vcs_m131_synthesis_safe_compact_canonical_k4_row_fold.sh"]="86c51f9d5246ddf86572e231fd09284055823bc35c654868135fac58d99f9887"
    ["dc_handoff/runs/m130_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="3eec7d86d5129c752f812dd27a9192644349d459d92f4b493bd18ebe0c105135"
    ["dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="e30e273ff791475d7f015ae4fb580a8c5fa0b018a432adf666519ffd44184316"
    ["dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/input_sha256.txt"]="474bb4c2616953b3fe4a0e246ab76f582f2537a9020194cdbc231b4fe8be6761"
    ["dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/output_sha256.txt"]="62b2d75d5c2f825602e0c52706f6252d7f422f4c51555496d942a105713886c5"
    ["dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/preflight_sha_checks.txt"]="2f4858da0a0b0ae7c515d91fe46dd320bb73d6f6d2733d2747bb3bedf90fe667"
    ["dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/runner_sha256.txt"]="93ddf41719f98a585cf553a709259d362484fd8ee5e23be62c74a95e4f76c2d1"
    ["dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/compile.raw.log"]="db264de3f8a8148d1c621aaad2ddb2d2baa54357881e94513e9ce17efeeca622"
    ["dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/sim.raw.log"]="5e4f9f8016fd332ed0ba5ae22fdea2b0b7c06ed21b4138f0325a6d29bbaaafa2"
    ["dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/assert.report"]="f19d48a64635a1ef20df93f769648bb7ec5fe5bfb43827e9b8c6131db93de588"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    ["reviews/m131_synthesis_safe_compact_canonical_k4_row_fold_independent_hammer_r1_20260824/m131_production_replay.f"]="9baf31330f35ca1c8b691fd607c69af9e1b2697a6712a00a5095fb33452d8747"
    ["reviews/m131_synthesis_safe_compact_canonical_k4_row_fold_independent_hammer_r1_20260824/m131_independent.f"]="7005bccf629eaab9f82e4ccc84258a55f2a5ed7c14a72eacbb71f87b61dd97fb"
    ["reviews/m131_synthesis_safe_compact_canonical_k4_row_fold_independent_hammer_r1_20260824/tb_m131_independent_hammer.sv"]="87777cbcbea494c94b5f6df096e4c003d2b1b264c5b3477d2195217db6619a7a"
    ["reviews/m131_synthesis_safe_compact_canonical_k4_row_fold_independent_hammer_r1_20260824/m131_dc_elaboration_check.tcl"]="d8dd22d343bed65a9c868964d97d1351f188906e092126e6cbbf6957e7d6508e"
)

: > "$task_review/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_review/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]]
done

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
cd "$task_sealed"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir=csrc \
    -F "$task_review/m131_production_replay.f" \
    -top tb_m131_synthesis_safe_compact_canonical_k4_row_fold \
    -o simv > compile.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > compile.rc
[[ "$task_rc" -eq 0 && -x simv ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' compile.raw.log
set +e
./simv -no_save -assert report=assert.report \
    -cm line+cond+tgl+fsm+assert > sim.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > sim.rc
[[ "$task_rc" -eq 0 ]]
grep -Fqx 'PASS M131 compact canonical K4 row fold VCS groups=237 updates=237 sources=691 lanes=22752 done=193 done_overlap=190 stalls=60 long_stall=17 cross_row_updates=64 cross_row_ii1=63 plus512=1 protocol_attacks=4 reset_attacks=1 idle_payload=1 descriptor_bits=35 producer_implemented=false physical_speedup=false system_speedup=false headline=false' sim.raw.log
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    sim.raw.log assert.report

cd "$task_hammer"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir=csrc \
    -F "$task_review/m131_independent.f" \
    -top tb_m131_independent_hammer -o simv > compile.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > compile.rc
[[ "$task_rc" -eq 0 && -x simv ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' compile.raw.log
set +e
./simv -no_save -assert report=assert.report \
    -cm line+cond+tgl+fsm+assert > sim.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > sim.rc
[[ "$task_rc" -eq 0 ]]
grep -Fqx 'PASS M131 independent hammer descriptor_bits=35 groups=110 updates=109 reset_aborted_descriptors=1 sources=420 lanes=10464 k1=1 k2=7 k3=3 k4=99 cross_group_ii1_intervals=95 cross_update_ii1_intervals=95 done=104 done_tags=104 done_overlap_next_row=100 output_stall_cycles=73 max_output_stall=73 group_stall_cycles=73 long_stall_replace=1 plus512=7 minus512=7 idle_payload_ready_checks=16 open_row_idle_payload_ready_checks=1 within_duplicate_attacks=1 within_descending_attacks=1 cross_repeat_attacks=1 cross_backtrack_attacks=1 row_identity_attacks=1 dirty_source_attacks=1 dirty_negate_attacks=1 nonlast_source15_attacks=1 cache_miss_attacks=1 block_attacks=1 reset_checks=1 gapped_partition_descriptors_accepted=3 internal_ready_valid_loop_observed=false predecessor_negative_index_present=false complete_row_partition_losslessness=false descriptor_producer_implemented=false descriptor_payload_bits_only=true dc_frequency_improvement=false physical_speedup=false system_speedup=false headline=false' sim.raw.log
! grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
    sim.raw.log assert.report

cd "$task_elab"
export HW_ROOT="$task_hw_root" OUTPUT_DIR="$task_elab" LIB_DB="$task_lib"
set +e
"$task_dc/bin/dc_shell" -64bit -f "$task_review/m131_dc_elaboration_check.tcl" \
    > dc.raw.log 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > dc.rc
[[ "$task_rc" -eq 0 ]]
grep -Fq 'PASS M131 independent DC analyze_elaborate_check_design no_elab312=true negative_index=false compile_run=false physical_speedup=false' dc.raw.log
[[ -s reports/check_design.rpt && -s reports/resources.rpt \
   && -s netlist/m131_synthesis_safe_compact_canonical_k4_row_fold_elaborated.ddc ]]
! grep -Eiq 'ELAB-312|group_source\[-1\]|out[- ]of[- ]bounds|^Error:' \
    dc.raw.log reports/check_design.rpt reports/resources.rpt

cd "$task_review"
sha256sum sealed_vcs_replay/{compile.raw.log,sim.raw.log,assert.report} \
    independent_vcs/{compile.raw.log,sim.raw.log,assert.report} \
    > vcs_output.sha256
sha256sum dc_elaboration/dc.raw.log dc_elaboration/dc.rc \
    dc_elaboration/reports/{check_design.rpt,hierarchy.rpt,resources.rpt,references.rpt} \
    dc_elaboration/netlist/m131_synthesis_safe_compact_canonical_k4_row_fold_elaborated.ddc \
    > dc_output.sha256
sha256sum run_m131_independent_hammer.sh > review_runner_sha256.txt
{
    echo 'status=PASS_M131_INDEPENDENT_VCS_AND_DC_ELABORATION_HAMMER'
    echo 'exact_sha_production_vcs_replay=true'
    echo 'independent_vcs=true'
    echo 'independent_dc_analyze_elaborate_check_design=true'
    echo 'negative_predecessor_index_removed=true'
    echo 'elab312_absent=true'
    echo 'idle_ready_payload_independent=true'
    echo 'tagged_done_cross_row_ii1=true'
    echo 'descriptor_payload_bits=35'
    echo 'complete_row_partition_losslessness=false'
    echo 'descriptor_producer_implemented=false'
    echo 'full_dc_compile_performed_by_review=false'
    echo 'dc_frequency_improvement=false'
    echo 'physical_speedup=false'
    echo 'system_speedup=false'
    echo 'headline=false'
} > RUN_COMPLETE.txt
echo "PASS M131 independent hammer outputs at $task_review"
