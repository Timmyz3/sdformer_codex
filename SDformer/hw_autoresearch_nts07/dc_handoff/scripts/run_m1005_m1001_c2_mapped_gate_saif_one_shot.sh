#!/usr/bin/env bash
set -euo pipefail

# Future-only M1005 runner for the additive M1001 rekey of frozen M979
# workload semantics. It is inert until M1002 -> M1003 -> M1004 is sealed.
dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hw_root="$(cd "${dc_root}/.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
contract="${hw_root}/contracts/m1001_m979_c2_mapped_gate_saif_rekey_source_contract_r1_20260829.json"
source_hammer="${hw_root}/reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"
release="${hw_root}/contracts/m1003_m1001_c2_mapped_gate_saif_launch_release_r1_20260829.json"
release_hammer="${hw_root}/reviews/m1004_m1003_m1001_c2_mapped_gate_saif_release_hammer_r1_20260829"
checker="${hw_root}/system_simulator/scripts/check_m1001_m979_c2_mapped_gate_saif_rekey_source.py"
tb="${hw_root}/dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
ucli="${hw_root}/dc_handoff/scripts/m979_c2_mapped_gate_per_case_saif.ucli.tcl"
memory_model="${hw_root}/tb_m349/m349_fc2_scalar_bank_memory_model.sv"
base="${hw_root}/dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
cell_model=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v
vcs=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
python=/opt/anaconda3/envs/pytorch310/bin/python3.10
result="${hw_root}/results/m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829"
attempt="${hw_root}/results/.m1005_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
work="${hw_root}/results/.m1005_m1001_c2_three_axis_mapped_gate_saif_work.$$"
failure="${hw_root}/results/m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829.failed_or_incomplete.$$.quarantine"
phase=SOURCE_PREFLIGHT
attempt_consumed=0
complete=0

sha() { sha256sum "$1" | awk '{print $1}'; }
fail() { printf 'M1005 M1001 gate failure: %s\n' "$*" >&2; exit 3; }
expect_sha() {
    [[ -f "$1" && ! -L "$1" && "$(sha "$1")" == "$2" ]] || fail "identity drift: $1"
}
verify_seal() {
    local dir=$1 expected_outer=$2
    [[ -d "${dir}" && ! -L "${dir}" && -f "${dir}/SHA256SUMS" \
       && -f "${dir}/SHA256SUMS.seal.sha256" ]] || fail "missing sealed directory: ${dir}"
    (cd "${dir}" && sha256sum -c SHA256SUMS >/dev/null && \
       sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || fail "seal failure: ${dir}"
    [[ "$(sha "${dir}/SHA256SUMS.seal.sha256")" == "${expected_outer}" ]] \
        || fail "outer seal mismatch: ${dir}"
}
collision_gate() {
    ! pgrep -x vcs1 >/dev/null && ! pgrep -x vlogan >/dev/null \
      && ! pgrep -x dc_shell >/dev/null && ! pgrep -x dc_shell-t >/dev/null \
      && ! pgrep -x fm_shell >/dev/null && ! pgrep -x pt_shell >/dev/null \
      || fail "VCS/DC/FM/PT collision"
}
seal_dir() {
    local dir=$1
    (cd "${dir}" && find . -type f ! -name SHA256SUMS \
      ! -name SHA256SUMS.seal.sha256 -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS \
      && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256)
}
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ "${complete}" -ne 1 && "${attempt_consumed}" -eq 1 && -d "${work}" ]]; then
        [[ "${rc}" -ne 0 ]] || rc=97
        printf '{"status":"FAILED_OR_INCOMPLETE","phase":"%s","return_code":%s}\n' \
            "${phase}" "${rc}" >"${work}/failure.json"
        seal_dir "${work}" || { printf 'M1005 failure seal failed; work retained at %s\n' "${work}" >&2; exit "${rc}"; }
        mv "${work}" "${failure}" || { printf 'M1005 failure move failed; work retained at %s\n' "${work}" >&2; exit "${rc}"; }
    fi
    exit "${rc}"
}

[[ -n "${M1005_EXPECTED_RUNNER_SHA256:-}" && "$(sha "${runner}")" == "${M1005_EXPECTED_RUNNER_SHA256}" ]] \
    || fail "caller must pin exact runner SHA"
[[ -n "${M1005_EXPECTED_M1002_OUTER_SHA256:-}" \
   && -n "${M1005_EXPECTED_M1004_OUTER_SHA256:-}" ]] \
    || fail "caller must pin M1002 and M1004 outer seals"
verify_seal "${source_hammer}" "${M1005_EXPECTED_M1002_OUTER_SHA256}"
[[ -f "${release}" && ! -L "${release}" ]] || fail "M1003 release absent"
verify_seal "${release_hammer}" "${M1005_EXPECTED_M1004_OUTER_SHA256}"
[[ "$(jq -r '.status' "${source_hammer}/review.json")" == PASS_M1002_M1001_SOURCE_HAMMER \
   && "$(jq -r '.status' "${release}")" == PASS_M1003_M1001_LAUNCH_RELEASE \
   && "$(jq -r '.launch_now' "${release}")" == true \
   && "$(jq -r '.status' "${release_hammer}/review.json")" == PASS_M1004_M1003_M1001_RELEASE_HAMMER \
   && "$(jq -r '.runner_sha256' "${release}")" == "$(sha "${runner}")" \
   && "$(jq -r '.source_contract_sha256' "${release}")" == "$(sha "${contract}")" ]] \
    || fail "release chain content mismatch"
"${python}" "${checker}" --contract "${contract}" >/dev/null
expect_sha "${vcs}" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
expect_sha "${cell_model}" 3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a
expect_sha "${memory_model}" 4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa
[[ ! -e "${result}" && ! -e "${attempt}" && ! -e "${work}" ]] \
    || fail "result/attempt/work collision"
collision_gate
phase=ATTEMPT_ATOMIC_CONSUME
mkdir "${attempt}" || fail "attempt already consumed or incomplete"
attempt_consumed=1
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
printf '{"status":"M1005_ATTEMPT_CONSUMED","runner_sha256":"%s","contract_sha256":"%s"}\n' \
    "$(sha "${runner}")" "$(sha "${contract}")" >"${attempt}/attempt.json"
seal_dir "${attempt}"
phase=WORK_CREATE
mkdir "${work}"

for axis in k1 k8 k1x8; do
    phase="COMPILE_${axis}"
    case "${axis}" in
        k1) define=M979_AXIS_K1 ;;
        k8) define=M979_AXIS_K8 ;;
        k1x8) define=M979_AXIS_K1X8 ;;
    esac
    axis_dir="${work}/${axis}"
    mkdir "${axis_dir}"
    netlist="${base}/${axis}/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
    "${vcs}" -full64 -sverilog +v2k +define+"${define}" -timescale=1ns/1ps \
        -Mdir="${axis_dir}/csrc" "${cell_model}" "${netlist}" "${memory_model}" "${tb}" \
        -top tb_m979_c2_three_axis_mapped_gate_case_saif -o "${axis_dir}/simv" \
        >"${axis_dir}/compile.log" 2>&1
    [[ -x "${axis_dir}/simv" ]] || fail "fresh compile failed: ${axis}"
    for case_id in 0 1 2 3 4; do
        phase="RUN_${axis}_CASE${case_id}"
        saif="${axis_dir}/case${case_id}.saif"
        log="${axis_dir}/case${case_id}.log"
        M979_SAIF_FILE="${saif}" "${axis_dir}/simv" +M979_UCLI_SAIF \
            +M979_CASE="${case_id}" -ucli -i "${ucli}" >"${log}" 2>&1
        pass=$(grep -E "^PASS M979 mapped replay axis=.* case=${case_id} .*cycles=[0-9]+ " "${log}" | tail -1)
        [[ -n "${pass}" ]] || fail "missing unique PASS: ${axis}/case${case_id}"
        cycles=$(sed -E 's/.* cycles=([0-9]+) .*/\1/' <<<"${pass}")
        "${python}" "${checker}" --saif "${saif}" --axis "${axis}" \
            --case "${case_id}" --cycles "${cycles}" \
            >"${axis_dir}/case${case_id}.saif_check.json"
    done
done

printf '%s\n' PASS_M1005_M1001_THREE_AXIS_FIFTEEN_CASE_MAPPED_GATE_SAIF >"${work}/RUN_COMPLETE.txt"
phase=SUCCESS_SEAL
seal_dir "${work}"
phase=SUCCESS_PUBLISH
mv "${work}" "${result}"
complete=1
trap - EXIT
printf '%s\n' PASS_M1005_M1001_THREE_AXIS_FIFTEEN_CASE_MAPPED_GATE_SAIF
