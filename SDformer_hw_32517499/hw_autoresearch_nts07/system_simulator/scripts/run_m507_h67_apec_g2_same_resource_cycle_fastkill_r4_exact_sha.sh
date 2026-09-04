#!/usr/bin/env bash
set -euo pipefail

m507_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${m507_root}"

m507_analyzer="system_simulator/scripts/analyze_m507_h67_apec_g2_same_resource_cycle_fastkill_r4.py"
m507_contract="contracts/m507_h67_apec_g2_same_resource_cycle_fastkill_contract_r4_20260827.json"
m507_review="reviews/m507_cycle_preflight_hammer_r4_20260827"
m507_out="results/m507_h67_apec_g2_same_resource_cycle_fastkill_r4_20260827"
m507_attempt="results/.m507_h67_apec_g2_same_resource_cycle_fastkill_r4_attempt_consumed"
m507_python="/opt/anaconda3/envs/pytorch310/bin/python"

m507_sha() { sha256sum "$1" | awk '{print $1}'; }
m507_expect() {
    local m507_path=$1 m507_expected=$2
    [[ -f "${m507_path}" ]] || {
        echo "M507 missing ${m507_path}" >&2
        exit 3
    }
    [[ "$(m507_sha "${m507_path}")" == "${m507_expected}" ]] || {
        echo "M507 SHA mismatch ${m507_path}" >&2
        exit 3
    }
}

[[ -x "${m507_python}" ]] || {
    echo "M507 missing pinned Python ${m507_python}" >&2
    exit 3
}
[[ ! -e "${m507_out}" && ! -e "${m507_attempt}" ]] || {
    echo "M507 one-shot already consumed or output exists" >&2
    exit 5
}

m507_identity_paths=(
    "${m507_analyzer}"
    "${m507_contract}"
    "${m507_review}/SHA256SUMS.seal.sha256"
    "docs/359_DATE终局冻结_20260813.md"
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json"
    "system_handoff/incoming/m73_h67_ep35_train_calibration_sources_s32_r1_20260823/m73_train_calibration_source_manifest.json"
    "results/m501_h67_exact_adjacent_overlap_fastkill_r1_20260827/m501_h67_exact_adjacent_overlap_fastkill_result_r1.json"
    "results/m501_h67_exact_adjacent_overlap_fastkill_r1_20260827/SHA256SUMS.seal.sha256"
    "reviews/m501_result_independent_hammer_r1_20260827/SHA256SUMS.seal.sha256"
    "reviews/m507_cycle_preflight_hammer_r1_20260827/SHA256SUMS.seal.sha256"
    "reviews/m507_cycle_preflight_hammer_r2_20260827/SHA256SUMS.seal.sha256"
    "reviews/m507_cycle_preflight_hammer_r3_20260827/SHA256SUMS.seal.sha256"
)

m507_verify_all_identities() {
    m507_expect "${m507_analyzer}" 13db92a7094ba6acce168be0f0c070318c76726edb28fb4bfa3db903302e4968
    m507_expect "${m507_contract}" 241ae6c8a5f2194e14a0573099a9d574003197c1c4a9c01626ecf1e81f2f3a5a
    m507_expect "${m507_review}/SHA256SUMS.seal.sha256" b884a74fbc71037a300bda74515db826c1e78b51dd093b78bca403ee86161934
    m507_expect "docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

    # Pin every producer independently of the analyzer's own contract walk.
    m507_expect "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json" e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3
    m507_expect "system_handoff/incoming/m73_h67_ep35_train_calibration_sources_s32_r1_20260823/m73_train_calibration_source_manifest.json" 3fb3468066fe1f7d61f5e39398cb2f8655643080f03e5b1deb58ef2911db17e2
    m507_expect "results/m501_h67_exact_adjacent_overlap_fastkill_r1_20260827/m501_h67_exact_adjacent_overlap_fastkill_result_r1.json" 37ce6d66a73c5dc3c19e887497ac85b473bc4789c0c241b4073d6af5d4c6cd18
    m507_expect "results/m501_h67_exact_adjacent_overlap_fastkill_r1_20260827/SHA256SUMS.seal.sha256" 1847520919b14c177087f40c0cb5457c44de5c7aef36097262996ae337edc7f6
    m507_expect "reviews/m501_result_independent_hammer_r1_20260827/SHA256SUMS.seal.sha256" 62573852f6154f25aaed4bca9bcb00fcac8395c7d055d17cdbbd9815060cb9eb
    m507_expect "reviews/m507_cycle_preflight_hammer_r1_20260827/SHA256SUMS.seal.sha256" 4f79d4baa826249ef65686c570428c0512cfce1156a8c900619196236113f538
    m507_expect "reviews/m507_cycle_preflight_hammer_r2_20260827/SHA256SUMS.seal.sha256" a6701d3aa36c2d328b2a03ffcbbdca4805c14e619599f481d17da80fc316d30d
    m507_expect "reviews/m507_cycle_preflight_hammer_r3_20260827/SHA256SUMS.seal.sha256" 83520159ce03e653063292868db5b6e5f94ec7791a8f59309d6043328aaf6329
    (cd "${m507_review}" && sha256sum -c SHA256SUMS >/dev/null && \
        sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

m507_verify_all_identities

m507_forbidden_process_gate() {
    local m507_hits=0
    for m507_name in dc_shell dc_shell-t fm_shell pt_shell vcs simv; do
        if pgrep -x "${m507_name}" >/dev/null; then
            echo "M507 resource gate: forbidden ${m507_name} process" >&2
            pgrep -a -x "${m507_name}" >&2 || true
            m507_hits=1
        fi
    done
    if ps -eo pid=,comm=,args= | awk '
        $2 ~ /^python(3([.][0-9]+)?)?$/ &&
        $0 ~ /(analyze|simulate|sweep|dse|independent|run)_m[0-9]/ {found=1}
        END {exit !found}
    '; then
        echo "M507 resource gate: forbidden project CPU DSE is active" >&2
        m507_hits=1
    fi
    [[ "${m507_hits}" -eq 0 ]]
}

m507_resource_snapshot() {
    local m507_limit m507_committed m507_available m507_swap m507_headroom
    local m507_failcnt m507_under_oom
    m507_limit=$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)
    m507_committed=$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)
    m507_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    m507_swap=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    m507_headroom=$((m507_limit - m507_committed))
    m507_failcnt=$(cat /sys/fs/cgroup/memory/user.slice/memory.failcnt)
    m507_under_oom=$(awk '/^under_oom / {print $2}' \
        /sys/fs/cgroup/memory/user.slice/memory.oom_control)
    printf 'commit_headroom_kib=%s mem_available_kib=%s swap_free_kib=%s cgroup_failcnt=%s cgroup_under_oom=%s\n' \
        "${m507_headroom}" "${m507_available}" "${m507_swap}" \
        "${m507_failcnt}" "${m507_under_oom}"
    [[ "${m507_headroom}" -ge 67108864 && \
       "${m507_available}" -ge 134217728 && \
       "${m507_swap}" -ge 33554432 && \
       "${m507_failcnt}" -eq 0 && "${m507_under_oom}" -eq 0 ]]
}

# Three independent samples prevent a transient gap between another process'
# exit and its memory-accounting cleanup from consuming the sole experiment.
m507_preflight_log=$(mktemp)
m507_preflight_ok=1
for m507_sample in 1 2 3; do
    printf 'timestamp=%s sample=%s\n' "$(date --iso-8601=seconds)" \
        "${m507_sample}" >>"${m507_preflight_log}"
    m507_resource_snapshot >>"${m507_preflight_log}" || m507_preflight_ok=0
    m507_forbidden_process_gate || m507_preflight_ok=0
    if [[ "${m507_sample}" -ne 3 ]]; then sleep 10; fi
done
if [[ "${m507_preflight_ok}" -ne 1 ]]; then
    cat "${m507_preflight_log}" >&2
    rm -f "${m507_preflight_log}"
    echo "M507 resource gate failed without consuming the one-shot" >&2
    exit 40
fi
m507_forbidden_process_gate || {
    rm -f "${m507_preflight_log}"
    exit 41
}
m507_verify_all_identities

mkdir "${m507_attempt}"
{
    echo "status=CONSUMED_IMMEDIATELY_BEFORE_ONE_SHOT_EXECUTION"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "output=${m507_out}"
    echo "analyzer_sha256=$(m507_sha "${m507_analyzer}")"
    echo "contract_sha256=$(m507_sha "${m507_contract}")"
    echo "preflight_review_seal_sha256=$(m507_sha "${m507_review}/SHA256SUMS.seal.sha256")"
} >"${m507_attempt}/ATTEMPT_CONSUMED.txt"
mv "${m507_preflight_log}" "${m507_attempt}/resource_preflight.log"
sha256sum "${BASH_SOURCE[0]}" "${m507_identity_paths[@]}" \
    >"${m507_attempt}/identity.sha256"

"${m507_python}" "${m507_analyzer}" \
    --contract "${m507_contract}" \
    --output-dir "${m507_out}"

[[ -f "${m507_out}/RUN_COMPLETE.txt" ]] || {
    echo "M507 missing RUN_COMPLETE marker" >&2
    exit 50
}
(cd "${m507_out}" && sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
m507_verify_all_identities
sha256sum -c "${m507_attempt}/identity.sha256" >/dev/null
printf 'status=PASS_POST_PUBLICATION_REHASH\ntimestamp=%s\n' \
    "$(date --iso-8601=seconds)" >"${m507_attempt}/POST_PUBLICATION_REHASH_PASS.txt"
(cd "${m507_attempt}" && sha256sum ATTEMPT_CONSUMED.txt \
    resource_preflight.log identity.sha256 POST_PUBLICATION_REHASH_PASS.txt \
    >SHA256SUMS && sha256sum SHA256SUMS >SHA256SUMS.seal.sha256 && \
    sha256sum -c SHA256SUMS >/dev/null && \
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
