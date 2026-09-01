#!/usr/bin/env bash
set -euo pipefail

# M1613 one-shot directed VCS runner source.  An independent exact-SHA launch
# admission is required before execution; authoring this file consumes no run.

[[ $# -eq 0 ]] || { printf 'M1613 takes no arguments\n' >&2; exit 2; }

dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hw_root="$(cd "${dc_root}/.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
vcs=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
python36=/usr/bin/python3.6
python312=/usr/bin/python3.12
filelist="${hw_root}/dc_handoff/filelists/date_m1613_c2_m1609_registered_fault_directed_vcs.f"
testbench="${hw_root}/dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv"
successor="${hw_root}/rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
contract="${hw_root}/contracts/m1613_c2_m1609_registered_fault_directed_source_contract_r1_20260901.json"
static_test="${hw_root}/system_simulator/tests/test_m1613_c2_m1609_registered_fault_directed_source.py"
m1611="${hw_root}/reviews/m1611_m1609_c2_registered_fault_successor_source_independent_review_r1_20260901"
hammer_dir="${hw_root}/reviews/m1617_m1613_c2_m1609_registered_fault_directed_source_hammer_r1_20260901"
hammer_review="${hammer_dir}/review.json"
release="${hw_root}/contracts/m1618_m1617_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json"
docs359="${hw_root}/docs/359_DATE终局冻结_20260813.md"
result="${hw_root}/results/m1613_c2_m1609_registered_fault_directed_vcs_r1_20260901"
attempt="${hw_root}/results/.m1613_c2_m1609_registered_fault_directed_vcs_attempt_consumed"
work="${hw_root}/results/.m1613_c2_m1609_registered_fault_directed_vcs_work.$$"
failure="${hw_root}/results/m1613_c2_m1609_registered_fault_directed_vcs_r1_20260901.failed.$$.quarantine"
top=tb_m1613_c2_m1609_registered_fault_directed
complete=0

sha() { sha256sum "$1" | awk '{print $1}'; }
fail() { printf 'M1613 gate failure: %s\n' "$*" >&2; exit 3; }
expect_sha() {
    local path=$1 expected=$2
    [[ -f "${path}" && ! -L "${path}" && "$(sha "${path}")" == "${expected}" ]] \
        || fail "missing/nonregular/SHA mismatch: ${path}"
}
verify_dir_seal() {
    local path=$1
    [[ -d "${path}" && ! -L "${path}" \
       && -f "${path}/SHA256SUMS" && ! -L "${path}/SHA256SUMS" \
       && -f "${path}/SHA256SUMS.seal.sha256" \
       && ! -L "${path}/SHA256SUMS.seal.sha256" ]] \
        || fail "sealed directory absent/nonregular: ${path}"
    (cd "${path}" && sha256sum -c SHA256SUMS >/dev/null \
        && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) \
        || fail "directory seal mismatch: ${path}"
}
verify_file_seal() {
    local path=$1 base parent
    base=$(basename "${path}")
    parent=$(dirname "${path}")
    [[ -f "${path}" && ! -L "${path}" \
       && -f "${path}.sha256" && ! -L "${path}.sha256" \
       && -f "${path}.sha256.seal.sha256" \
       && ! -L "${path}.sha256.seal.sha256" ]] \
        || fail "sealed file absent/nonregular: ${path}"
    (cd "${parent}" && sha256sum -c "${base}.sha256" >/dev/null \
        && sha256sum -c "${base}.sha256.seal.sha256" >/dev/null) \
        || fail "file seal mismatch: ${path}"
}
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ "${complete}" -ne 1 && -d "${work}" && ! -L "${work}" ]]; then
        mv -T "${work}" "${failure}" || true
    fi
    exit "${rc}"
}
signal_exit() { trap - INT TERM HUP; exit 130; }

[[ -n "${M1613_EXPECTED_RUNNER_SHA256:-}" \
   && "$(sha "${runner}")" == "${M1613_EXPECTED_RUNNER_SHA256}" ]] \
    || fail "caller must pin the independently reviewed runner SHA"
expect_sha "${vcs}" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
expect_sha "${python36}" 9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f
expect_sha "${python312}" 0876a8f712651a0c6a2e54aabd163fb85464b2a4ca8e96a15074f2826a1d8814
expect_sha "${successor}" 7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931
expect_sha "${testbench}" 096f32095a81abbedf4c0bda59a2a146df764d18bdc2136c31e0c7a7319a57a4
expect_sha "${filelist}" 071e37d731988a12c3b0adc6380e179de55260c34325fce944daabba6c58671d
expect_sha "${contract}" 248c9065d81608a8fc2aacdd8539a3287462653e411ee545a8f320a98a8a5f8d
expect_sha "${static_test}" 0f8ef678ef40ee1413939e894bd86fc658821847b6254437e7ecca08ea59b4ea
expect_sha "${m1611}/review.json" 6109dff51fb6b60463afbfa32f3756c6ceffae1b12dc085134a1c008cd2bf480
expect_sha "${m1611}/SHA256SUMS" 58f2e9701fab6450557d1bef44604997b4b501a18d799c4f9e91719a6494f0d5
expect_sha "${m1611}/SHA256SUMS.seal.sha256" 6e56d25c27c59fad37875d533e2dcc9e03abd0635d687d874fdbed41bbbf45fd
expect_sha "${docs359}" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

mapfile -t rows < <(sed '/^[[:space:]]*$/d' "${filelist}")
[[ "${#rows[@]}" -eq 2 \
   && "${rows[0]}" == rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv \
   && "${rows[1]}" == dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv ]] \
    || fail "filelist is not the exact successor-only pair"
! grep -Fq 'rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv' "${filelist}" \
    || fail "frozen predecessor present in M1613 filelist"

"${python36}" "${static_test}" >/dev/null
"${python312}" "${static_test}" >/dev/null

# These intentionally absent-at-authoring authorities are the hard launch
# stop.  A caller-supplied runner hash alone can never authorize execution.
verify_dir_seal "${hammer_dir}"
verify_file_seal "${release}"
"${python312}" -I - "${hammer_review}" "${release}" "${runner}" \
    "${contract}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
hammer, release, runner, contract = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
h = json.loads(hammer.read_text())
r = json.loads(release.read_text())
assert h["status"] == "PASS_M1617_M1613_C2_REGISTERED_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT"
assert h["score"] >= 95 and h["p0_count"] == 0 and h["p1_count"] == 0
assert r["status"] == "AUTHORIZE_ONE_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT"
assert r["authorization"] == {"vcs_compiles": 1, "simv_runs": 1,
                               "all_other_eda_runs": 0}
assert r["identity"]["runner_sha256"] == sha(runner)
assert r["identity"]["source_contract_sha256"] == sha(contract)
assert r["identity"]["hammer_review_sha256"] == sha(hammer)
PY
[[ -n "${M1613_EXPECTED_RELEASE_SHA256:-}" \
   && "$(sha "${release}")" == "${M1613_EXPECTED_RELEASE_SHA256}" ]] \
    || fail "caller must pin the independently reviewed release SHA"

[[ ! -e "${result}" && ! -L "${result}" \
   && ! -e "${attempt}" && ! -L "${attempt}" \
   && ! -e "${work}" && ! -L "${work}" \
   && ! -e "${failure}" && ! -L "${failure}" ]] \
    || fail "result/attempt/work namespace is not fresh"
"${python312}" -I - <<'PY'
import os
from pathlib import Path
blocked = {"vcs", "vcs1", "vlogan", "simv"}
ancestry = set()
pid = os.getpid()
while pid > 1 and pid not in ancestry:
    ancestry.add(pid)
    try:
        pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
    except Exception:
        break
hits = []
for p in Path("/proc").iterdir():
    if not p.name.isdigit() or int(p.name) in ancestry:
        continue
    try:
        if p.stat().st_uid != os.getuid():
            continue
        comm = (p / "comm").read_text().strip()
        argv = {Path(item.decode(errors="replace")).name
                for item in (p / "cmdline").read_bytes().split(b"\0") if item}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if comm in blocked or blocked & argv:
        hits.append((p.name, comm, sorted(argv)))
if hits:
    raise SystemExit("same-UID VCS collision: %r" % hits)
PY
[[ "${VCS_HOME:-}" == /opt/synopsys/vcs/V-2023.12-SP1 \
   && "${VCS_ARCH_OVERRIDE:-}" == linux \
   && "${SNPSLMD_LICENSE_FILE:-}" == 27030@ic.ismd-nemo ]] \
    || fail "VCS environment mismatch"

mkdir "${attempt}"
printf 'M1613_ATTEMPT_CONSUMED runner_sha256=%s automatic_retry=false\n' \
    "$(sha "${runner}")" >"${attempt}/attempt.txt"
mkdir "${work}"
trap cleanup EXIT
trap signal_exit INT TERM HUP
cd "${hw_root}"

printf 'VCS_COMPILE vcs_compiles=1\n' >"${work}/runner.log"
set +e
"${vcs}" -full64 -sverilog -assert svaext -timescale=1ns/1ps \
    -cm assert -Mdir="${work}/csrc" -f "${filelist}" -top "${top}" \
    -o "${work}/simv" >"${work}/compile.log" 2>&1
compile_rc=$?
set -e
printf '%s\n' "${compile_rc}" >"${work}/compile.rc"
[[ "${compile_rc}" -eq 0 && -x "${work}/simv" ]] \
    || fail "VCS compile failed"
! grep -Eiq 'Error-\[|^Error|^Fatal|Fatal:' "${work}/compile.log" \
    || fail "VCS compile log contains an error"

simv="${work}/simv"
printf 'SIMV_RUN simv_runs=1 seed=1613\n' >>"${work}/runner.log"
set +e
"${simv}" +ntb_random_seed=1613 -no_save -cm assert \
    -assert report="${work}/assert.report" >"${work}/sim.log" 2>&1
sim_rc=$?
set -e
printf '%s\n' "${sim_rc}" >"${work}/sim.rc"
[[ "${sim_rc}" -eq 0 ]] || fail "simv failed"
[[ -f "${work}/assert.report" && ! -L "${work}/assert.report" ]] \
    || fail "assertion report missing/nonregular"
! grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "${work}/sim.log" "${work}/assert.report" \
    || fail "simulation/assertion report contains a failure"
grep -Eq '^PASS M1613 M1609 registered-fault directed legal_terminal_no_false_pulse=1 legal_descriptor_accepts=1 illegal_header_latched=1 illegal_raw_latched=1 sticky_checks=3 source_only=false performance=false$' \
    "${work}/sim.log" || fail "exact M1613 PASS token absent"

printf '{"schema":"m1613_c2_m1609_registered_fault_directed_vcs_receipt_r1_v1","status":"PASS","vcs_compiles":1,"simv_runs":1,"seed":1613,"performance":false,"dc":false,"power":false,"runner_sha256":"%s","successor_sha256":"%s","testbench_sha256":"%s","filelist_sha256":"%s"}\n' \
    "$(sha "${runner}")" "$(sha "${successor}")" "$(sha "${testbench}")" \
    "$(sha "${filelist}")" >"${work}/receipt.json"
printf '%s\n' PASS_M1613_M1609_REGISTERED_FAULT_DIRECTED_VCS \
    >"${work}/RUN_COMPLETE.txt"
(
    cd "${work}"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum -c SHA256SUMS >/dev/null
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
)
[[ ! -e "${result}" && ! -L "${result}" ]] \
    || fail "canonical result appeared before publication"
mv -T -n "${work}" "${result}"
[[ -d "${result}" && ! -L "${result}" && ! -e "${work}" ]] \
    || fail "atomic no-clobber publication failed"
complete=1
trap - EXIT INT TERM HUP
printf 'M1613 directed VCS result published; independent receipt hammer required\n'
