#!/usr/bin/env bash
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: M2110 accepts no arguments" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
TOP=m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend
TCL="${SCRIPT_DIR}/run_icc2_m2110_m2029_m2018_matched_macrofree_axis.tcl"
PARSER="${HW_ROOT}/system_simulator/scripts/parse_m2110_m2029_m2018_matched_macrofree_icc2_pnr.py"
CONTRACT="${HW_ROOT}/contracts/m2110_m2029_m2018_matched_macrofree_icc2_pnr_source_contract_r1_20260904.json"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
M2029="${HW_ROOT}/dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
ADDENDUM="${HW_ROOT}/reviews/tcasii2027_m2018_icc2_physical_tech_readonly_addendum_r1_20260904"
M2111="${HW_ROOT}/reviews/m2111_m2110_m2029_m2018_matched_macrofree_icc2_pnr_source_hammer_r1_20260904"
M2111_REVIEW="${M2111}/review.json"

ORDINARY_V="${M2029}/ordinary_lru4/netlist/${TOP}_mapped.v"
ORDINARY_SDC="${M2029}/ordinary_lru4/netlist/${TOP}_mapped.sdc"
TSBG_V="${M2029}/tsbg_b4/netlist/${TOP}_mapped.v"
TSBG_SDC="${M2029}/tsbg_b4/netlist/${TOP}_mapped.sdc"

TECH_BASE=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital
DB_BASE="${TECH_BASE}/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a"
TT_DB="${DB_BASE}/tcbn28hpcplusbwp35p140tt0p9v25c.db"
SS_DB="${DB_BASE}/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
FF_DB="${DB_BASE}/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
MW_REF="${TECH_BASE}/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140"
NXTGRD=/opt/tech/tsmc28/RC_Extraction/starRC/typical/crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical.nxtgrd
LAYER_MAP=/opt/tech/tsmc28/RC_Extraction/starRC/typical/Reference/MAP/star.map_icc_crn28hpc+_1p9m_6x1z1u_ut-alrdl
ITF=/opt/tech/tsmc28/RC_Extraction/starRC/typical/crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical.itf

ICC2=/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_shell
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_SERVER=27030@ic.ismd-nemo
LICENSE_FILE=/opt/synopsys/Synopsys.dat
RESULT="${HW_ROOT}/dc_handoff/runs/m2110_m2029_m2018_matched_macrofree_icc2_pnr_r1_20260904"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m2110_m2029_m2018_matched_macrofree_icc2_pnr_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m2110_m2029_m2018_matched_macrofree_icc2_pnr_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m2110_m2029_m2018_matched_macrofree_icc2_pnr_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: M2110 identity mismatch: ${path}" >&2
    exit 3
  }
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" ]] || return 1
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
seal_dir() {
  local dir="$1"
  (cd -- "${dir}" &&
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
on_exit() {
  local rc=$?
  set +e
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" && ! -L "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" \
      >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

# Exact frozen design, review, technology, and protected-document identities.
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact f5847f355329a52511ab044ef458284a19ae424ac778418a4bc4778bb2d3a2b0 "${ORDINARY_V}"
sha_exact 46b4bd73ace0cfb67f7794321f641ebfabfc0cabd542776ed586d65438970838 "${ORDINARY_SDC}"
sha_exact 739eb76dcb732ec0c66b75392c768cbe36027ecc5d458bd4b088f8488f67c9af "${TSBG_V}"
sha_exact c7b894cee479badcca22977b29d6ba69a20ca85d9b20e402c9c46ad92ed16d70 "${TSBG_SDC}"
sha_exact d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070 "${TT_DB}"
sha_exact 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af "${SS_DB}"
sha_exact a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a "${FF_DB}"
sha_exact 0dae685414322ce33a28b09c8213789e4dabc55e282a3000840af0831d70daa3 "${MW_REF}/lib"
sha_exact 399ffd29421f99384e5d172629b6bb147c4a0a44abf71f57e4bd9e038f0aede1 "${MW_REF}/lib_1"
sha_exact 48a65f71c25d477aedb927cf692e58ddde08abfe419c7d69a92d0b6783aca082 "${MW_REF}/lib_2"
sha_exact 424477b89c352173da2c3adc1d723764e8ff68425289ef688793be364646fd02 "${NXTGRD}"
sha_exact da6e70dae3b50cc8e7520e3576477f2f80c3ac55dbe2b61baad73eb36fe44ed3 "${LAYER_MAP}"
sha_exact d55fe511848cedc58f8d8e2c4a081487a57a723b3361cec075306dd42b55a1cb "${ITF}"
sha_exact d66521b4bb35e465171ff8a18d69de4e8ea49e092af11b5229ae5cec5f7ca97a "${ADDENDUM}/addendum.json"
sha_exact 11f3cca09e96401e59a88c55446e8b0bb633650d10711bd1dc5f44347dcb439e "${ADDENDUM}/mechanical_checks.json"
verify_dir_seal "${M2029}"
verify_dir_seal "${ADDENDUM}"
verify_dir_seal "${M2111}"

[[ -n "${M2110_EXPECTED_RUNNER_SHA256:-}" && "$(sha_file "${RUNNER}")" == "${M2110_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M2110_EXPECTED_SOURCE_REVIEW_SHA256:-}" && "$(sha_file "${M2111_REVIEW}")" == "${M2110_EXPECTED_SOURCE_REVIEW_SHA256}" ]] || exit 3
/usr/libexec/platform-python3.6 -I - "${M2111_REVIEW}" "${RUNNER}" "${TCL}" "${PARSER}" "${CONTRACT}" <<'PY'
from __future__ import print_function
import hashlib, json, sys
from pathlib import Path
review, runner, tcl, parser, contract = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads(review.read_text())
assert r['status'].startswith('PASS_M2111')
assert r['score_over_100'] >= 95
assert r['severity_counts'] == {'p0': 0, 'p1': 0, 'p2': 0}
for key, path in [('runner_sha256', runner), ('tcl_sha256', tcl),
                  ('parser_sha256', parser), ('contract_sha256', contract)]:
    assert r['identity'][key] == sha(path), (key, r['identity'].get(key), sha(path))
assert r['authorization'] == {'license_queries': 1, 'icc2_shell_runs': 2,
                              'all_other_eda_runs': 0, 'automatic_retry': False}
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 4
/usr/libexec/platform-python3.6 -I - <<'PY'
from __future__ import print_function
import os
from pathlib import Path
blocked = {'vcs', 'vcs1', 'vlogan', 'simv', 'dc_shell', 'dc_shell-t',
           'pt_shell', 'fm_shell', 'icc2_shell', 'icc2_lm_shell',
           'common_shell_exec', 'common_shell_exe'}
hits = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        if p.stat().st_uid != os.getuid():
            continue
        comm = (p / 'comm').read_text().strip()
        argv = {Path(x.decode(errors='replace')).name
                for x in (p / 'cmdline').read_bytes().split(b'\0') if x}
    except Exception:
        continue
    if comm in blocked or blocked & argv:
        hits.append((p.name, comm))
if hits:
    raise SystemExit('M2110 same-UID EDA collision: %r' % hits)
PY
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
[[ "${mem_available}" -ge 67108864 && $((commit_limit-committed)) -ge 50331648 ]] || exit 4

cd -- "${REPO_ROOT}"
mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M2110_ATTEMPT_CONSUMED\nlicense_queries=1\nicc2_shell_runs=2\naxes=ordinary_lru4,tsbg_b4\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f ICCompilerII >"${WORK}/license_preflight.log" 2>&1
grep -Fq 'Users of ICCompilerII:' "${WORK}/license_preflight.log"

# Normalize the two DC SDCs to one physical constraint identity: strip comments,
# the embedded operating condition, and the two-line ZeroWireload command only.
/usr/libexec/platform-python3.6 -I - "${ORDINARY_SDC}" "${TSBG_SDC}" "${WORK}" <<'PY'
from __future__ import print_function
import hashlib, sys
from pathlib import Path
ordinary, tsbg, out = map(Path, sys.argv[1:])
def normalize(path):
    lines = []
    skip_cont = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if skip_cont:
            skip_cont = stripped.endswith('\\')
            continue
        if not stripped or stripped.startswith('#') or stripped.startswith('set_operating_conditions'):
            continue
        if stripped.startswith('set_wire_load_model'):
            skip_cont = stripped.endswith('\\')
            continue
        lines.append(line.rstrip())
    text = '\n'.join(lines) + '\n'
    assert 'wire_load' not in text.lower()
    assert 'set_operating_conditions' not in text
    return text
a, b = normalize(ordinary), normalize(tsbg)
assert a == b, 'M2110 normalized physical constraints differ across axes'
p = out / 'physical_common.sdc'
p.write_text(a)
(out / 'physical_common.sdc.sha256').write_text(hashlib.sha256(a.encode()).hexdigest() + '  physical_common.sdc\n')
PY
PHYSICAL_SDC="${WORK}/physical_common.sdc"
PHYSICAL_SDC_SHA="$(sha_file "${PHYSICAL_SDC}")"
[[ "${PHYSICAL_SDC_SHA}" == e16f6ee72f22a2d48e1d600f6f3c0fd32c8aa5c29ccda6ff48d94396c0ef92bf ]] || exit 5

axis_names=(ordinary_lru4 tsbg_b4)
axis_netlists=("${ORDINARY_V}" "${TSBG_V}")
for index in 0 1; do
  axis="${axis_names[$index]}"
  netlist="${axis_netlists[$index]}"
  axis_dir="${WORK}/${axis}"
  mkdir -- "${axis_dir}"
  /usr/bin/timeout --signal=TERM --kill-after=300s 43200s \
    env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
      SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
      M2110_AXIS="${axis}" M2110_TOP="${TOP}" M2110_AXIS_DIR="${axis_dir}" \
      M2110_MAPPED_V="${netlist}" M2110_PHYSICAL_SDC="${PHYSICAL_SDC}" \
      M2110_PHYSICAL_SDC_SHA256="${PHYSICAL_SDC_SHA}" \
      M2110_DESIGN_LIB="${axis_dir}/m2110_${axis}.nlib" M2110_MW_REF_LIB="${MW_REF}" \
      M2110_TT_DB="${TT_DB}" M2110_SS_DB="${SS_DB}" M2110_FF_DB="${FF_DB}" \
      M2110_NXTGRD="${NXTGRD}" M2110_LAYER_MAP="${LAYER_MAP}" \
      "${ICC2}" -f "${TCL}" >"${axis_dir}.icc2.log" 2>&1
  printf '0\n' >"${axis_dir}.icc2.rc"
  [[ "$(grep -c '^PASS_M2110_MATCHED_MACROFREE_ICC2_AXIS axis=' "${axis_dir}.icc2.log")" -eq 1 ]]
  [[ "$(grep -c '^M2110_GATE1_IMPORT_AND_LIBRARY_CHECK_PASS axis=' "${axis_dir}.icc2.log")" -eq 1 ]]
  [[ "$(grep -c '^M2110_GATE2_NXTGRD_ADVANCED_SANITY_PASS axis=' "${axis_dir}.icc2.log")" -eq 1 ]]
  [[ -s "${axis_dir}/machine_facts.txt" && -s "${axis_dir}/RUN_COMPLETE.txt" ]]
  port_sha="$(sha_file "${axis_dir}/reports/ports_sorted.txt")"
  /usr/libexec/platform-python3.6 -I - "${axis_dir}/machine_facts.txt" "${port_sha}" <<'PY'
from __future__ import print_function
import sys
from pathlib import Path
p, value = Path(sys.argv[1]), sys.argv[2]
text = p.read_text()
token = 'port_inventory_sha256=POPULATED_BY_ONE_SHOT_RUNNER\n'
assert text.count(token) == 1
p.write_text(text.replace(token, 'port_inventory_sha256=' + value + '\n'))
PY
done

cmp -s -- "${WORK}/ordinary_lru4/reports/ports_sorted.txt" "${WORK}/tsbg_b4/reports/ports_sorted.txt"
"${PARSER}" --ordinary-dir "${WORK}/ordinary_lru4" --tsbg-dir "${WORK}/tsbg_b4" \
  --output "${WORK}/receipt.json" >"${WORK}/parser.log"
grep -Fxq 'PASS_RAW_M2110_MATCHED_MACROFREE_ICC2_PNR_PENDING_INDEPENDENT_RESULT_HAMMER' "${WORK}/parser.log"

printf 'RAW_PASS_M2110_MATCHED_MACROFREE_ICC2_PNR_PENDING_INDEPENDENT_RESULT_HAMMER\n' >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2110_MATCHED_MACROFREE_ICC2_PNR_PENDING_INDEPENDENT_RESULT_HAMMER\n'
