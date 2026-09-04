#!/usr/bin/env bash
set -euo pipefail
umask 002

# One-shot functional VCS runner for additive M912.  This is foundry
# UNIT_DELAY functional evidence only; it cannot establish timing, cycles,
# speedup, PPA, energy, trace recurrence or a paper headline.

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
RTL="${HW_ROOT}/rtl_m912_c1_pipeline/m912_m528_metadata_pipelined_product_capture_island.sv"
MACRO_RTL="${HW_ROOT}/rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
SVA="${HW_ROOT}/verif_m912_c1_pipeline/m912_m528_metadata_pipeline_assertions.sv"
TB="${HW_ROOT}/verif_m912_c1_pipeline/tb_m912_metadata_pipeline_unit_delay_r1.sv"
STATIC_CHECK="${HW_ROOT}/verif_m912_c1_pipeline/static_check_m912_metadata_pipeline.py"
CONTRACT="${HW_ROOT}/contracts/m912_c1_metadata_pipeline_unit_delay_vcs_source_contract_r1_20260829.json"
FOUNDRY_V="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"
VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
RESULT="${HW_ROOT}/results/m912_c1_metadata_pipeline_unit_delay_vcs_r1_20260829"
ATTEMPT="${HW_ROOT}/results/.m912_c1_metadata_pipeline_unit_delay_vcs_r1_attempt_consumed"
WORK="${HW_ROOT}/results/.m912_c1_metadata_pipeline_unit_delay_vcs_r1_work.$$"
PASS_TOKEN="PASS_M912_C1_METADATA_PIPELINE_UNIT_DELAY_DIRECTED_RANDOM_AND_ATTACKS"
TOP="tb_m912_metadata_pipeline_unit_delay_r1"

sha_exact() {
  local expected="$1" path="$2" got
  [[ -f "${path}" && ! -L "${path}" ]] || {
    echo "ERROR: missing/nonregular ${path}" >&2; exit 3; }
  got="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${got}" == "${expected}" ]] || {
    echo "ERROR: SHA mismatch ${path}: ${got}" >&2; exit 3; }
}

seal_dir() {
  local dir="$1"
  (cd -- "${dir}" &&
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}

WORK_ACTIVE=0
on_exit() {
  local rc=$?
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE\nexit_code=%s\nfunctional_vcs_verified=false\n' \
      "${rc}" >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    local q="${RESULT}.failed_or_incomplete.$$.quarantine"
    [[ ! -e "${q}" ]] && mv -- "${WORK}" "${q}" || true
  fi
}
trap on_exit EXIT

# Exact source identity.  These literals are deliberately fail closed.
sha_exact eef2f8d3344620cfbf518bf4ac382a2f0be5b46084d56308a660e4c172c65e53 "${RTL}"
sha_exact 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783 "${MACRO_RTL}"
sha_exact 18a988e95f6beab57c9b6a37d48f1a6fc8973e176ff7eb16b09bc381568e9ce5 "${SVA}"
sha_exact de19e962c1ffb16d74f6505e425843f3fbe399ef47d746bf3329770d48daa78d "${TB}"
sha_exact 6eb874aacd6d06cbb5a5036f0b4a80a83f6abdf28b8a86d1a389e85b89475ce6 "${STATIC_CHECK}"
sha_exact 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d "${FOUNDRY_V}"
sha_exact 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "${VCS_BIN}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 \
  "${HW_ROOT}/docs/359_硬件论文口径与冻结数字_20260812.md"

[[ -f "${CONTRACT}.sha256" && -f "${CONTRACT}.sha256.seal.sha256" ]] || {
  echo "ERROR: contract seals absent" >&2; exit 3; }
(cd -- "$(dirname -- "${CONTRACT}")" &&
  sha256sum -c "$(basename -- "${CONTRACT}.sha256")" >/dev/null &&
  sha256sum -c "$(basename -- "${CONTRACT}.sha256.seal.sha256")" >/dev/null)

python3 -I "${STATIC_CHECK}"
python3 -I - "${CONTRACT}" "${RUNNER}" "${RTL}" "${SVA}" "${TB}" <<'PY'
import hashlib, json, sys
from pathlib import Path

contract, runner, rtl, sva, tb = map(Path, sys.argv[1:])
d = json.loads(contract.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert d["status"] == "READY_FOR_FRESH_INDEPENDENT_HAMMER_THEN_ONE_VCS_ATTEMPT"
assert d["authorization"] == {"vcs_compiles": 1, "simv_runs": 1,
                               "dc_runs": 0, "pt_runs": 0,
                               "formality_runs": 0}
i = d["identity"]
assert i["runner_sha256"] == sha(runner)
assert i["rtl_sha256"] == sha(rtl)
assert i["sva_sha256"] == sha(sva)
assert i["tb_sha256"] == sha(tb)
assert d["claim_boundary"]["functional_vcs_only"] is True
for key in ("timing_verified", "cycles_measured", "speedup",
            "ppa", "energy", "paper_citable"):
    assert d["claim_boundary"][key] is False
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]] || {
  echo "ERROR: result/attempt/work identity already exists" >&2; exit 4; }

# No concurrent commercial EDA process under this uid.
python3 -I - <<'PY'
import os
from pathlib import Path
blocked = {"vcs", "vcs1", "simv", "dc_shell", "pt_shell", "fm_shell", "icc2_shell"}
hits=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) in {os.getpid(), os.getppid()}:
        continue
    try:
        if p.stat().st_uid != os.getuid(): continue
        comm=(p/'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if comm in blocked: hits.append((p.name,comm))
if hits: raise SystemExit("EDA collision: %r" % hits)
PY

mem_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "${mem_kib}" =~ ^[0-9]+$ && "${mem_kib}" -ge 67108864 ]] || {
  echo "ERROR: MemAvailable below 64 GiB" >&2; exit 5; }

mkdir -- "${ATTEMPT}"
printf 'runner_sha256=%s\ncreated_utc=%s\n' \
  "$(sha256sum -- "${RUNNER}" | awk '{print $1}')" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${ATTEMPT}/identity.txt"
mkdir -- "${WORK}"
WORK_ACTIVE=1
cd -- "${WORK}"

export VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"
export VCS_ARCH_OVERRIDE="linux"
export SNPSLMD_LICENSE_FILE="27030@ic.ismd-nemo"
export LM_LICENSE_FILE="/opt/synopsys/Synopsys.dat"

"${VCS_BIN}" -full64 -sverilog -timescale=1ns/1ps -assert svaext \
  -debug_access+pp +define+UNIT_DELAY +vcs+lic+wait \
  "${FOUNDRY_V}" "${MACRO_RTL}" "${RTL}" "${SVA}" "${TB}" \
  -top "${TOP}" -o simv 2>&1 | tee compile.log
compile_rc=("${PIPESTATUS[@]}")
[[ "${compile_rc[0]}" -eq 0 && "${compile_rc[1]}" -eq 0 ]] || exit 20

/usr/bin/timeout --signal=TERM --kill-after=30s 600s ./simv -no_save \
  2>&1 | tee sim.log
sim_rc=("${PIPESTATUS[@]}")
[[ "${sim_rc[0]}" -eq 0 && "${sim_rc[1]}" -eq 0 ]] || exit 21

[[ "$(rg -c "^${PASS_TOKEN} " sim.log)" -eq 1 ]] || exit 22
[[ "$(rg -c '^COVERAGE_M912_C1_METADATA_PIPELINE ' sim.log)" -eq 1 ]] || exit 23
[[ "$(rg -c '^HELD_FINAL_RECOVERY_M533_M528_DW1RW_R10 ' sim.log)" -eq 1 ]] || exit 24
rg -q '^PASS_M912_C1_METADATA_PIPELINE_UNIT_DELAY_DIRECTED_RANDOM_AND_ATTACKS .*attacks=6 .*two_cycle_fill=[1-9][0-9]* .*timing_verified=false .*speedup=false .*ppa=false .*headline=false$' sim.log || exit 25

python3 -I - "${RUNNER}" "${CONTRACT}" "${RTL}" "${SVA}" "${TB}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
runner, contract, rtl, sva, tb = map(Path, sys.argv[1:])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
d={
  "schema":"m912_c1_metadata_pipeline_unit_delay_vcs_receipt_v1",
  "status":"PASS_FUNCTIONAL_VCS_ONLY",
  "created_utc":datetime.now(timezone.utc).isoformat(),
  "identity":{"runner_sha256":sha(runner),"contract_sha256":sha(contract),
              "rtl_sha256":sha(rtl),"sva_sha256":sha(sva),"tb_sha256":sha(tb)},
  "macro_model":"foundry_UNIT_DELAY_functional",
  "attack_count":6,
  "claim_boundary":{"functional_vcs_verified":True,"timing_verified":False,
                    "cycles_measured":False,"speedup":False,"ppa":False,
                    "energy":False,"paper_citable":False}}
Path("m912_c1_metadata_pipeline_unit_delay_vcs_receipt_r1.json").write_text(
    json.dumps(d,indent=2,sort_keys=True)+"\n")
PY
printf 'PASS_FUNCTIONAL_VCS_ONLY\n' >RUN_COMPLETE.txt
seal_dir "${WORK}"
mv -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
printf 'PASS M912 functional VCS result=%s\n' "${RESULT}"
