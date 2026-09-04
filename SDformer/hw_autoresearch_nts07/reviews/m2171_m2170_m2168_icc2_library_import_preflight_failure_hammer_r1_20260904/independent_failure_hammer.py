#!/usr/bin/python3.12
"""Read-only M2171 failure hammer; invokes no license client or EDA tool."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
Q = HW / "dc_handoff/runs/m2170_m2168_icc2_library_import_preflight_raw_r1_20260904.failed_or_incomplete.2923115.quarantine"
A = HW / "dc_handoff/runs/.m2170_m2168_icc2_library_import_preflight_attempt_consumed"
RUNNER = HW / "dc_handoff/scripts/run_m2168_m2167_icc2_library_import_preflight_one_shot.sh"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
LM_SHELL = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell")
LM_ALIAS = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_lm_shell")
LM_REAL = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec")
MW_EXEC = Path("/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway")
GEN_DOC = Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/generate_frame_from_mw.2")
MW_OPT_DOC = Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat3/lib.setting.milkyway_exec.3")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def seal(root: Path) -> dict[str, object]:
    need(root.is_dir() and not root.is_symlink(), f"sealed root: {root}")
    need(not any(path.is_symlink() for path in root.rglob("*")),
         f"symlink under sealed root: {root}")
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    members: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(maxsplit=1); rel = rel.lstrip("*")
        need(rel not in members and sha(root / rel) == digest, f"member {rel}")
        members[rel] = digest
    actual = sorted(str(path.relative_to(root)) for path in root.rglob("*")
                    if path.is_file() and path.name not in
                    {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(sorted(members) == actual, "nonexhaustive seal")
    return {"members": len(members), "manifest_sha256": sha(manifest),
            "outer_sha256": sha(outer)}


def main() -> int:
    a_seal, q_seal = seal(A), seal(Q)
    need(sha(DOCS359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs359 identity")
    need(len(list((HW / "dc_handoff/runs").glob(
         "m2170_m2168_icc2_library_import_preflight_raw_r1_20260904.failed_or_incomplete.*.quarantine"))) == 1,
         "M2170 quarantine multiplicity")
    need(not (HW / "dc_handoff/runs/m2170_m2168_icc2_library_import_preflight_raw_r1_20260904").exists(),
         "canonical M2170 result exists")
    attempt = dict(line.split("=", 1) for line in
                   (A / "ATTEMPT_CONSUMED.txt").read_text().splitlines())
    need(attempt == {"status": "M2170_ATTEMPT_CONSUMED", "license_queries": "1",
                     "top_level_icc2_shell_runs": "1", "pnr_runs": "0",
                     "retry": "false"}, "attempt marker")
    failure = dict(line.split("=", 1) for line in
                   (Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text().splitlines())
    need(failure == {"status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                     "exit_code": "1", "retry": "false"}, "failure marker")
    need((Q / "icc2_preflight.rc").read_text().strip() == "42", "ICC2 rc")
    execution = json.loads((Q / "execution_contract.json").read_text())
    need(execution["license_queries"] == 1 and
         execution["top_level_icc2_shell_runs"] == 1 and
         execution["pnr_runs"] == 0 and
         execution["automatic_retry"] is False and
         execution["scope"] == "library_import_only", "execution contract")
    runner = RUNNER.read_text()
    need(runner.count('"${LMUTIL}" lmstat') == 1 and
         runner.count('"${ICC2}" -no_init -f "${TCL}"') == 1,
         "unique external sites")

    license_log = (Q / "license_preflight.log").read_text(errors="replace")
    need(len(re.findall(r"^Users of ICCompilerII:", license_log,
                       re.MULTILINE)) == 1, "license response")
    process = json.loads((Q / "process_tree.json").read_text())
    need(process["root_seen"] is True and
         process["icc2_actual_exec_observation_count"] == 1 and
         process["tool_spawned_conversion_exec_observation_count"] == 0,
         "process execution census")
    actuals = process["icc2_actual_exec_observations"]
    need(len(actuals) == 1 and actuals[0]["exe_path"] ==
         "/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/dgcom_exec" and
         "-no_init" in actuals[0]["cmdline"], "exact ICC2 actual exec")
    need("PASS_M2153_PROCESS_TREE_CENSUS" in
         (Q / "process_monitor.log").read_text(), "process monitor completion")

    log = (Q / "icc2_preflight.log").read_text(errors="replace")
    need(len(re.findall(r"^M2153_GATE1_OPTION_ROUND_TRIP_PASS ", log,
                       re.MULTILINE)) == 1, "gate1 execution")
    need(len(re.findall(r"^Error: unknown command 'generate_frame_from_mw' "
                       r"\(CMD-005\)$", log, re.MULTILINE)) == 1,
         "exact CMD-005 failure")
    need(len(re.findall(r"^M2153_GATE[2-6]_", log, re.MULTILINE)) == 0,
         "post-failure gate unexpectedly executed")
    need(len(re.findall(r"^M2153_FATAL_FAIL_CLOSED:", log,
                       re.MULTILINE)) == 1, "fatal boundary")
    need(not any((Q / "isolated_cwd" / rel).exists() for rel in (
         "m2153_disposable_design.nlib",
         "frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm",
         "reports/machine_facts.txt", "reports/master_coverage.tsv")),
         "library or post-conversion artifact exists")
    need(not any((Q / "isolated_cwd/reports").iterdir()),
         "reports produced before failure")

    # Read-only local installation/manual audit; no tool is launched.
    need(LM_SHELL.is_file() and not LM_SHELL.is_symlink() and
         os.access(LM_SHELL, os.X_OK) and
         sha(LM_SHELL) == "1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942",
         "lm_shell wrapper identity")
    need(LM_ALIAS.is_symlink() and os.readlink(LM_ALIAS) == "lm_shell",
         "icc2_lm_shell alias")
    need(LM_REAL.is_file() and sha(LM_REAL) ==
         "3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab",
         "lm_shell real executable")
    need(MW_EXEC.is_file() and os.access(MW_EXEC, os.X_OK) and sha(MW_EXEC) ==
         "09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec",
         "Milkyway executable identity")
    need(sha(GEN_DOC) ==
         "f9424346c44d9d48cbae5a3839f26cadad46b4d85e405deb19354356cd232952" and
         "generate_frame_from_mw" in GEN_DOC.read_text(errors="replace"),
         "LM command manual")
    mw_doc = MW_OPT_DOC.read_text(errors="replace")
    need(sha(MW_OPT_DOC) ==
         "b497b940eaf9c1f044362d701ec2eea5710391f4c5995370cee74d511916a1e9" and
         "must be specified before running" in mw_doc,
         "Milkyway option manual")
    lm_commands = ("set_app_options", "get_app_option_value",
                   "generate_frame_from_mw", "create_lib", "current_lib",
                   "get_libs", "get_lib_cells", "get_site_defs", "get_layers",
                   "get_techs", "read_parasitic_tech", "get_parasitic_techs",
                   "report_ref_libs", "report_design", "save_lib")
    for command in lm_commands:
        need((Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2") /
              f"{command}.2").is_file(), f"LM command manual absent: {command}")

    print("PASS_M2171_FAILURE_HAMMER seals=2 license=1 icc2=1 pnr=0 "
          "cause=ICC2_CMD005_UNKNOWN_generate_frame_from_mw next=LM_SHELL_SOURCE_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
