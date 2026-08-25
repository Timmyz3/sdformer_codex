#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_design=guarded_wordpacked_pwp_stream
task_filelist="${task_dc_root}/filelists/date_m97_m85_logic_only_dc.f"
task_sdc="${task_dc_root}/constraints/date_m97_m85_logic_only_3ns.sdc"
task_contract="${task_hw_root}/contracts/m97_m85_logic_only_synopsys_contract_r1_20260824.json"
task_period="${CLOCK_PERIOD_NS:-3.000}"
task_period_tag="${task_period//./p}ns"
task_output="${OUTPUT_DIR:-${task_dc_root}/runs/m97_m85_logic_only_dc_${task_period_tag}_$(date -u +%Y%m%dT%H%M%SZ)}"
task_lock="${task_output}.launch_lock"

task_slow_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
task_fast_default=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
LIB_DB="${LIB_DB:-$task_slow_default}"
MIN_LIB_DB="${MIN_LIB_DB:-$task_fast_default}"

if ! command -v dc_shell >/dev/null 2>&1; then
    echo "M97 requires Synopsys dc_shell" >&2
    exit 2
fi
if [[ ! -f "$LIB_DB" || ! -f "$MIN_LIB_DB" ]]; then
    echo "M97 slow/fast library is missing" >&2
    exit 3
fi
if [[ "$task_period" != "3.000" \
        || "${OPERATING_CONDITION:-ssg0p9v125c}" != "ssg0p9v125c" ]]; then
    echo "M97 r1 contract forbids clock-period or operating-condition overrides" >&2
    exit 10
fi
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null \
        || pgrep -f '[c]ommon_shell_exec -shell dc_shell' >/dev/null; then
    echo "refusing M97 because another dc_shell is active" >&2
    exit 4
fi
if [[ -e "$task_output" ]]; then
    echo "refusing to overwrite M97 evidence: $task_output" >&2
    exit 5
fi
if ! mkdir "$task_lock"; then
    echo "refusing M97 because the output launch lock is held: $task_lock" >&2
    exit 8
fi
trap 'rmdir "$task_lock" 2>/dev/null || true' EXIT

declare -A task_expected=(
    ["$task_contract"]="9bcf76152fd9c291423310b6c24f543e1466b941e5c568f5bb2d16ec9c8ecb5c"
    ["$task_filelist"]="6e2c6c7f831eecadba604675447f8425c3427e6cf83a6c6310e7a20483789d00"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["${task_dc_root}/scripts/run_dc_m97_m85_logic_only.tcl"]="8d30dfd2a6b2480c538b751640aa17d52549162c35905de3bf384798ce3dfdde"
    ["${task_hw_root}/rtl_m82/zero_bubble_elastic_pwp_stream.sv"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["${task_hw_root}/rtl_m85/guarded_wordpacked_pwp_stream.sv"]="ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0"
    ["${task_hw_root}/contracts/m85_guarded_wordpacked_pwp_stream_vcs_contract_r1_20260823.json"]="2f1225acb79ceaf16df35bc477dcd05c54bf0d299675cec388bce66cb1e576af"
    ["${task_dc_root}/runs/m85_guarded_wordpacked_pwp_stream_vcs_r1_sealed_20260823/RUN_COMPLETE.txt"]="3577b00dc5b3f59d45ea0d8bd2f0a74f96672c4e6f5493307793ff4cd5d3051a"
    ["$LIB_DB"]="79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af"
    ["$MIN_LIB_DB"]="a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a"
)
for task_path in "${!task_expected[@]}"; do
    task_actual="$(sha256sum "$task_path" | awk '{print $1}')"
    if [[ "$task_actual" != "${task_expected[$task_path]}" ]]; then
        echo "M97 frozen input mismatch: $task_path" >&2
        exit 7
    fi
done
python3 -m json.tool "$task_contract" >/dev/null

mkdir -p "$task_output"
export DESIGN_NAME="$task_design"
export HW_ROOT="$task_hw_root"
export RTL_FILELIST="$task_filelist"
export SDC_FILE="$task_sdc"
export OUTPUT_DIR="$task_output"
export CLOCK_PERIOD_NS="$task_period"
export LIB_DB MIN_LIB_DB
export OPERATING_CONDITION="${OPERATING_CONDITION:-ssg0p9v125c}"

{
    echo "paper_ppa_ready=false"
    echo "full_pwp_frontend_ppa=false"
    echo "system_speedup_admitted=false"
    echo "headline_admitted=false"
    echo "logic_only=true"
    echo "pre_macro=true"
    echo "ideal_clock=true"
    echo "wireload=ZeroWireload"
    echo "clock_period_ns=$task_period"
    echo "operating_condition=$OPERATING_CONDITION"
    echo "setup_library=$LIB_DB"
    echo "hold_library=$MIN_LIB_DB"
    echo "macros=0"
    echo "identity=M97_M85_PWP_LOGIC_ISLAND_ONLY"
    sha256sum "$0"
    for task_path in "${!task_expected[@]}"; do sha256sum "$task_path"; done
} > "$task_output/admission.txt"

dc_shell -f "${task_dc_root}/scripts/run_dc_m97_m85_logic_only.tcl" \
    2>&1 | tee "$task_output/dc.log"

if grep -q '^Error:' "$task_output/dc.log"; then
    echo "M97 DC log contains a Tcl/DC error" >&2
    exit 9
fi
for task_report in \
    reports/qor.rpt reports/area.rpt \
    reports/timing_setup.rpt reports/timing_hold.rpt \
    netlist/${task_design}_mapped.v; do
    if [[ ! -s "$task_output/$task_report" ]]; then
        echo "M97 missing evidence: $task_report" >&2
        exit 6
    fi
done
touch "$task_output/RUN_COMPLETE.txt"
echo "M97 DC completed at $task_output"
