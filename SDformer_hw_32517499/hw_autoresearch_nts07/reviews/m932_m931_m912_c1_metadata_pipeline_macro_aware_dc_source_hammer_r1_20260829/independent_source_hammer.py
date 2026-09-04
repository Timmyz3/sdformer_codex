#!/usr/bin/env python3
"""Static, no-EDA independent source hammer for the frozen M931 DC candidate."""

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import List


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/m931_m912_c1_metadata_pipeline_macro_aware_dc_source_contract_r1_20260829.json"
RUNNER = ROOT / "dc_handoff/scripts/run_dc_m931_m912_c1_metadata_pipeline_macro_aware_exact_sha_r1.sh"
FILELIST = ROOT / "dc_handoff/filelists/date_m931_m912_c1_metadata_pipeline_macro_aware_dc.f"
SDC = ROOT / "dc_handoff/constraints/date_m931_m912_c1_metadata_pipeline_macro_aware_3ns.sdc"
TCL = ROOT / "dc_handoff/scripts/run_dc_m931_m912_c1_metadata_pipeline_macro_aware_candidate.tcl"
ADAPTER = ROOT / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
RTL = ROOT / "rtl_m912_c1_pipeline/m912_m528_metadata_pipelined_product_capture_island.sv"
M929 = ROOT / "reviews/m929_m926_m923_m912_c1_metadata_pipeline_vcs_result_hammer_r1_20260829"
DOC359 = ROOT / "docs/359_DATE终局冻结_20260813.md"
MACRO_ROOT = pathlib.Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821")

EXPECTED_CONTRACT_SHA = "9e617a319d451a712282947dd5d246de93a7c2ffed9e8e5e604f78afb966eab7"
EXPECTED_RUNNER_SHA = "9c4a4354c5c784e895588d49f7d6f51bd6db51d94435398fbf4af4a3aed15421"
EXPECTED_CONTRACT_SIDECAR_SHA = "079f7f09f4c5bc00ff51a7b167e362122f5f79f86fd3e32cf2c10b1948eb74e6"
EXPECTED_DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = []  # type: List[str]


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    checks.append("PASS " + label)


def verify_manifest(directory: pathlib.Path, manifest_name: str = "SHA256SUMS") -> None:
    manifest = directory / manifest_name
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("* ")
        require(sha(directory / rel) == digest, f"manifest {directory.name}/{rel}")


def command_lines(text: str, command: str) -> List[str]:
    return [line for line in text.splitlines()
            if re.match(rf"^\s*{re.escape(command)}(?:\s|$)", line)]


require(sha(CONTRACT) == EXPECTED_CONTRACT_SHA, "frozen contract SHA")
require(sha(RUNNER) == EXPECTED_RUNNER_SHA, "frozen runner SHA")
require(sha(CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")) == EXPECTED_CONTRACT_SIDECAR_SHA,
        "contract inner sidecar SHA")
verify_manifest(CONTRACT.parent, CONTRACT.name + ".sha256")
verify_manifest(CONTRACT.parent, CONTRACT.name + ".sha256.seal.sha256")

contract = json.loads(CONTRACT.read_text())
require(contract["schema"] == "m931_m912_c1_metadata_pipeline_macro_aware_dc_source_contract_v1",
        "contract schema")
require(contract["status"] == "SOURCE_ONLY__INDEPENDENT_HAMMER_AND_RELEASE_REQUIRED__NO_EDA_AUTHORIZED",
        "source-only status")
require(not any(contract["authorization"].values()), "all author execution permissions false")

for rel, expected in contract["exact_files"].items():
    path = ROOT / rel
    require(path.is_file() and not path.is_symlink(), f"exact regular file {rel}")
    require(sha(path) == expected, f"exact SHA {rel}")

require(sha(DOC359) == EXPECTED_DOC359_SHA, "docs/359 frozen SHA")
require(contract["functional_authority"]["review_sha256"] == sha(M929 / "review.json"),
        "M929 functional authority SHA")
verify_manifest(M929)
verify_manifest(M929, "SHA256SUMS.seal.sha256")
m929 = json.loads((M929 / "review.json").read_text())
require(m929["score"] == 100 and m929["p0_count"] == 0 and m929["p1_count"] == 0,
        "M929 PASS100 functional VCS authority")
require(m929["claim_boundary"]["functional_vcs_verified"] is True,
        "M929 functional VCS verified")
require(m929["claim_boundary"]["timing_verified"] is False and
        m929["claim_boundary"]["speedup"] is False and
        m929["claim_boundary"]["paper_citable"] is False,
        "M929 functional-only claim boundary")

point = contract["physical_point"]
require(point["technology_nm"] == 28 and point["clock_period_ns"] == 3.0,
        "28nm 3.000ns point")
require(point["ideal_clock"] is True and point["wireload"] == "ZeroWireload",
        "ideal-clock ZeroWireload point")
require(point["macro_cell"] == "TS1N28HPCPHVTB128X128M4S" and
        point["expected_macro_count"] == 9, "nine exact 1RW macros")
require(point["logical_parent_capacity_bytes"] == 64 * 1152 // 8 == 9216,
        "logical parent capacity 9216 B")
require(point["bound_parent_macro_capacity_bytes"] == 9 * 128 * 128 // 8 == 18432,
        "bound parent macro capacity 18432 B")
require(point["total_capacity_obligation_bytes"] == 213376 and
        point["unbound_remaining_storage_obligation_bytes"] == 213376 - 18432 == 194944,
        "total versus unbound storage obligation")
require(point["all_240kib_storage_physically_bound_in_this_top"] is False,
        "full 240KiB storage not falsely bound")
require(point["compile_ultra_count"] == 1 and point["incremental_compile_count"] == 0,
        "one compile and zero incremental contract")
require(point["hold_diagnostic_only"] is True and point["debug_false_paths"] is False,
        "hold diagnostic and no debug false paths")

tools = {
    "dc_shell_sha256": pathlib.Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"),
    "lmutil_sha256": pathlib.Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil"),
    "license_file_sha256": pathlib.Path("/opt/synopsys/Synopsys.dat"),
}
for field, path in tools.items():
    require(path.is_file() and sha(path) == contract["tool_identity"][field],
            f"exact tool/license identity {field}")

views = {
    "std_slow_sha256": pathlib.Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"),
    "std_fast_sha256": pathlib.Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"),
    "macro_slow_sha256": MACRO_ROOT / "ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db",
    "macro_fast_sha256": MACRO_ROOT / "ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db",
}
for field, path in views.items():
    require(path.is_file() and not path.is_symlink() and sha(path) == contract["foundry_views"][field],
            f"exact foundry view {field}")
require(sha(MACRO_ROOT / "SHA256SUMS") == contract["foundry_views"]["macro_manifest_sha256"],
        "macro manifest exact SHA")
verify_manifest(MACRO_ROOT)

filelist_lines = [line.strip() for line in FILELIST.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
require(filelist_lines == [
    "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
    "rtl_m912_c1_pipeline/m912_m528_metadata_pipelined_product_capture_island.sv",
], "DC filelist exact two RTL sources")
require(all(not line.endswith(".v") for line in filelist_lines),
        "behavioral macro Verilog absent from DC filelist")
adapter = ADAPTER.read_text()
require(adapter.count("TS1N28HPCPHVTB128X128M4S u_parent_sram") == 1 and
        "slice < 9" in adapter and "{1'b0, address}" in adapter,
        "nine generated 128x128 macro binding with low 64 logical rows")
require("register-array fallback" in adapter and
        not re.search(r"logic\s+\[[^\]]+\]\s+\w+\s*\[\s*0\s*:\s*127\s*\]", adapter),
        "no synthesizable parent register-array fallback")
rtl = RTL.read_text()
require(rtl.count("m528_dw1rw_parent_scratch_9x128_macro u_parent_scratch") == 1,
        "top instantiates one nine-macro adapter")

sdc = SDC.read_text()
require("-period 3.000" in sdc and "0.000 1.500" in sdc,
        "3.000ns clock waveform")
require("set_clock_uncertainty -setup 0.200" in sdc and
        "set_clock_uncertainty -hold 0.050" in sdc and
        "set_input_delay 0.250" in sdc and "set_output_delay 0.250" in sdc,
        "frozen I/O and uncertainty constraints")
false_paths = command_lines(sdc, "set_false_path")
require(false_paths == ["set_false_path -from [get_ports reset_n]"],
        "only reset false path; no debug exception")

tcl = TCL.read_text()
compile_lines = command_lines(tcl, "compile_ultra")
require(compile_lines == ["compile_ultra -no_autoungroup"], "exactly one compile_ultra command")
require(not command_lines(tcl, "compile") and "-incremental" not in tcl,
        "zero plain/incremental compile command")
require("analyze -format sverilog -define SYNTHESIS $rtl_files" in tcl,
        "SYNTHESIS-bound SystemVerilog analyze")
require("set_min_library $std_slow_db -min_version $std_fast_db" in tcl and
        "set_min_library $macro_slow_db -min_version $macro_fast_db" in tcl,
        "standard-cell and macro slow/fast min pairs")
require("set_operating_conditions ssg0p9v125c" in tcl and
        "set_wire_load_model -name ZeroWireload [current_design]" in tcl,
        "slow operating condition and ZeroWireload")
require("macro_count_pre != $expected_macro_count" in tcl and
        "macro_count_post != $expected_macro_count" in tcl and
        "set_dont_touch $macro_cells_pre true" in tcl,
        "nine-macro pre/post count and preservation gates")
gate_pos = tcl.index("set pre_tim209")
compile_pos = tcl.index("compile_ultra -no_autoungroup")
require(gate_pos < compile_pos and "pre_tim209 != 0 || $pre_opt150 != 0" in tcl,
        "TIM-209/OPT-150 fail gate before compile")
require("read_verilog" not in tcl and "ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v" not in tcl,
        "behavioral macro Verilog not read by Tcl")
for artifact in ("_mapped.v", "_mapped.sdc", ".ddc", ".svf"):
    require(artifact in tcl, f"mapped artifact emission {artifact}")

runner = RUNNER.read_text()
require(subprocess.run(["bash", "-n", str(RUNNER)], check=False).returncode == 0,
        "runner Bash syntax")
require("m931_lock_owned=0" in runner and "m931_lock_owned=1" in runner and
        'if [[ "$m931_lock_owned" -eq 1 ]]' in runner,
        "lock ownership explicitly tracked")
require(runner.index('mkdir "$m931_lock"') < runner.index("m931_lock_owned=1") <
        runner.index('"$m931_lmutil" lmstat'), "lock acquired before license query and ownership set")
require(runner.index("m931_collision_present") < runner.index('mkdir "$m931_lock"'),
        "same-UID DC collision gate before lock")
require("dc_shell:*|dc_shell-t:*|common_shell_exec:*" in runner,
        "dc/common-shell collision identities covered")
require(runner.index('mkdir "$m931_attempt"') < runner.index('"$m931_dc" -no_gui'),
        "one-shot attempt consumed before DC")
require("max_attempts=1" in runner and "retry=false" in runner,
        "permanent one-shot marker semantics")
require("FAILED_OR_INCOMPLETE_DO_NOT_CITE" in runner and
        "failed_or_incomplete.$$.quarantine" in runner and
        "mv -T -- \"$m931_work\" \"$m931_quarantine\"" in runner,
        "post-attempt failures quarantined")
require("m931_verify_file_seal \"$m931_contract\"" in runner,
        "contract double-seal verified before resource admission")
for needle in ("m931_verify_linkable_file \"$m931_dc\"",
               "m931_verify_linkable_file \"$m931_lmutil\"",
               "m931_verify_file \"$m931_license\"",
               "m931_verify_file \"$m931_std_slow\"",
               "m931_verify_file \"$m931_std_fast\"",
               "m931_verify_file \"$m931_macro_slow\"",
               "m931_verify_file \"$m931_macro_fast\""):
    require(needle in runner, f"runner exact identity gate {needle.split()[1]}")
require("PIPESTATUS[0]" in runner and
        '[[ "$m931_rc" -eq 0 && -f "$m931_work/TCL_PASS_TERMINAL.txt" ]]' in runner,
        "DC pipeline status and terminal gates")
require("RAW_DC_COMPLETE__INDEPENDENT_RESULT_HAMMER_REQUIRED" in runner and
        '"independent_result_hammered": False' in runner and
        '"setup_admitted": False' in runner and '"paper_ppa_ready": False' in runner,
        "raw result cannot self-admit claims")
require("m931_seal_dir \"$m931_work\"" in runner and
        "mv -T -- \"$m931_work\" \"$m931_result\"" in runner,
        "sealed staging then atomic result promotion")

# Execute only the preflight bad-argument branch in an isolated temporary tree.
# It exits before any identity, license, tool, or EDA action and proves that a
# foreign lock is not removed by a non-owner cleanup trap.
with tempfile.TemporaryDirectory(prefix="m932_lock_owner_test.") as tmp_s:
    tmp = pathlib.Path(tmp_s)
    script_dir = tmp / "dc_handoff/scripts"
    lock = tmp / "dc_handoff/runs/.m931_m912_c1_metadata_pipeline_macro_aware_dc_launch_lock"
    script_dir.mkdir(parents=True)
    lock.mkdir(parents=True)
    copied = script_dir / RUNNER.name
    shutil.copy2(RUNNER, copied)
    proc = subprocess.run([str(copied), "unexpected_argument"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, check=False)
    require(proc.returncode == 2 and lock.is_dir(),
            "foreign lock survives non-owner failure (no EDA branch)")

result = ROOT / "dc_handoff/runs/m931_m912_c1_metadata_pipeline_macro_aware_dc_3p000ns_r1_20260829"
attempt = ROOT / "dc_handoff/runs/.m931_m912_c1_metadata_pipeline_macro_aware_dc_attempt_consumed"
lock = ROOT / "dc_handoff/runs/.m931_m912_c1_metadata_pipeline_macro_aware_dc_launch_lock"
require(not result.exists() and not attempt.exists() and not lock.exists(),
        "no production result/attempt/lock consumed by reviewer")

boundary = contract["claim_boundary"]
require(boundary["source_candidate_only"] is True and
        all(boundary[k] is False for k in (
            "macro_linked_dc_result", "full_storage_macro_integrated",
            "setup_admitted", "hold_signoff", "power", "energy", "ppa",
            "speedup", "system_speedup", "paper_ppa_ready", "headline")),
        "strict source-only claim boundary")
require(contract["fairness"]["zero_rtl_baseline_present"] is False and
        contract["fairness"]["bit_rtl_baseline_present"] is False and
        contract["fairness"]["fair_K_zero_bit"] is False,
        "no fair RTL speedup baseline claimed")

print("\n".join(checks))
print(f"PASS_TOTAL={len(checks)}")
