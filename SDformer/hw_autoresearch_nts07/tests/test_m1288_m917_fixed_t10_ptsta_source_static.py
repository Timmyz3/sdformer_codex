#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "dc_handoff/scripts/run_m1288_m917_fixed_t10_ptsta_inert_exact_sha.sh"
TCL = ROOT / "dc_handoff/scripts/run_ptsta_m1288_m917_fixed_t10_slowmax_fastmin_inert.tcl"
CONTRACT = ROOT / "contracts/m1288_c3_m917_fixed_t10_ptsta_source_contract_r1_20260830.json"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

runner = RUNNER.read_text()
tcl = TCL.read_text()
contract = json.loads(CONTRACT.read_text())
checks = {}
checks["source_only"] = (
    contract["status"] == "M1288_SOURCE_ONLY__NO_PT_EDA_AUTHORIZED"
    and contract["authorization"]["launch_now"] is False
    and contract["authorization"]["run_pt"] is False
)
checks["runner_sha"] = contract["identity"]["runner_sha256"] == sha(RUNNER)
checks["tcl_sha"] = contract["exact_files"][str(TCL.relative_to(ROOT))] == sha(TCL)
checks["admission_precedes_namespace"] = (
    runner.index("m1288_expect \"${m1288_admission}\"")
    < runner.index("mkdir \"${m1288_attempt}\"")
    < runner.index("\"${m1288_setsid}\" env -i")
)
checks["no_version_or_license_query"] = " -version" not in runner and "lmstat" not in runner
checks["one_shot_fresh"] = all(x in runner for x in (
    "! -e \"${m1288_canonical}\"", "! -e \"${m1288_work}\"",
    "! -e \"${m1288_attempt}\"", "mkdir \"${m1288_attempt}\""))
checks["private_home"] = all(x in runner for x in (
    "chmod 0700", "HOME=\"${m1288_work}/safe_home\"", "stat -c '%a'"))
checks["collision_and_drain"] = all(x in runner for x in (
    "m1288_collisions", "same-UID", "m1288_wait_job_empty",
    "m1288_terminate_job", "m1288_job_members"))
checks["docs359_guard"] = (
    contract["exact_files"]["docs/359_DATE终局冻结_20260813.md"]
    == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    and "m1288_sha docs/359_DATE终局冻结_20260813.md" in runner)
checks["dual_corner_pt"] = all(x in tcl for x in (
    "set_min_library $slow_db -min_version $fast_db",
    "-max $slow_opcond -max_library $slow_lib_name",
    "-min $fast_opcond -min_library $fast_lib_name",
    "report_timing -delay_type max", "report_timing -delay_type min"))
forbidden = ("fix_eco_timing", "set_fix_hold", "read_parasitics", "update_power", "report_power")
checks["no_hold_fix_or_power"] = not any(
    re.search(rf"(?m)^\s*{re.escape(token)}\b", tcl) for token in forbidden)
checks["diagnostic_stop"] = all(x in runner for x in (
    "DIAGNOSTIC_STOP_M1288_M917_FIXED_T10_HOLD_NEGATIVE",
    "new netlist-only hold-fix identity plus Formality and repeated PT",
    "'speedup':False", "'paper_ppa_ready':False"))
checks["upstream_seals"] = all(x in contract["exact_files"] for x in (
    "dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829/SHA256SUMS.seal.sha256",
    "reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829/SHA256SUMS.seal.sha256",
    "reviews/m1285_c3_m917_m928_pt_hold_saif_ptpx_readonly_audit_r1_20260830/SHA256SUMS.seal.sha256"))
failed = [name for name, value in checks.items() if not value]
print(json.dumps({"checks": checks, "failed": failed,
                  "status": "PASS" if not failed else "FAIL"}, indent=2, sort_keys=True))
raise SystemExit(bool(failed))
