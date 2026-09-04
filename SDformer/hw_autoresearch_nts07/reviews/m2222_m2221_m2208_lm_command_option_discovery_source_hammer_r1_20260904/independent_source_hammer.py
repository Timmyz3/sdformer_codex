#!/usr/bin/python3.12
"""Independent CPU-only M2222 hammer; never invokes LM/license/EDA/GPU/Git."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m2221_m2208_lm_command_option_discovery_source_contract_r1_20260904.json"
TCL = HW / "dc_handoff/scripts/run_lm_m2221_command_option_discovery.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2223_m2222_m2221_lm_command_option_discovery_one_shot.sh"
CENSUS = HW / "dc_handoff/scripts/census_m2205_same_uid_tools.py"
INVENTORY = HW / "dc_handoff/scripts/inventory_m2153_repo_root.py"
CHECKER = HW / "system_simulator/scripts/check_m2223_lm_command_option_discovery.py"
TEST = HW / "tests/test_m2221_lm_command_option_discovery_source.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2208 = HW / "reviews/m2208_m2207_m2205_lm_library_conversion_preflight_failure_hammer_r1_20260904"
AUTHOR = HW / "reviews/m2221_m2208_lm_command_option_discovery_source_author_receipt_r1_20260904"
RUNS = HW / "dc_handoff/runs"
MW_MANIFEST = HW / "dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
MW_REF = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140")
LM = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell")
LM_EXEC = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec")
MW = Path("/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")

EXPECTED_SOURCE = {
    TCL: "70ad2401b9d798dc60dc9f9b9d6235a1351189cb2a66bce913b1e76e16fe61d6",
    RUNNER: "7a44233fb6e763752902da61f38bd92a5576fec26993339476d6a1343ba02518",
    CENSUS: "ec452719e68c5caa88039ec7e37512647e2c737d54842eb2adf55e66639160bf",
    INVENTORY: "351db733e16f15895c7f1658b21c16901ff907ed5613cb89c2f4a85ce8928f94",
    CHECKER: "537ebe18cba63de9df81a5078a2b633a53ef384b0c43c194e7fd6a7673bcfe0f",
    TEST: "3e511b28fc2b15176c56d0197ed19e9e76b128245b41eda9133f2f6d297d5723",
    CONTRACT: "e1b3091c14a06e0727b013a5693e7011601049f857d96545ee7abd405ede6c08",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_TOOLS = {
    LM: "1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942",
    LM_EXEC: "3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab",
    MW: "09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
}
EXPECTED_DOCS = {
    Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/generate_frame_from_mw.2"): "f9424346c44d9d48cbae5a3839f26cadad46b4d85e405deb19354356cd232952",
    Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/set_app_options.2"): "ae28a2f50dc5ed7457adad00428a0c0e7fa57cc4555866015d4ab4563e4ec0da",
    Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/get_app_option_value.2"): "f0d7b2b4334d00f90432c7fcdb319fe80668578633dfbda0bcdc644302e4e47a",
    Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/report_app_options.2"): "6be35b3549beaa7ac73886f88cdaf80d40bfd985fc0dd4c96efd3587df89c3ba",
    Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat2/get_app_options.2"): "a9ef5c15a2022c38b0da1140638c1ff23d1806caf939f8e9f0d94ef1eb8b8135",
    Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat3/lib.setting.milkyway_exec.3"): "b497b940eaf9c1f044362d701ec2eea5710391f4c5995370cee74d511916a1e9",
    Path("/opt/synopsys/icc2/V-2023.12-SP3/doc/LM/man/cat3/lib.configuration.local_output_dir.3"): "5354ec5b5964e454395a8f8d8cfecd489470d5c6555ec78242213d5925c6d9ea",
}
M2223_NAMES = (
    "m2223_m2221_lm_command_option_discovery_raw_r1_20260904",
    ".m2223_m2221_lm_command_option_discovery_attempt_consumed",
    ".m2223_m2221_lm_command_option_discovery_launch_lock",
    ".m2223_m2221_lm_command_option_discovery_work",
)


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_seal(directory: Path) -> int:
    need(directory.is_dir() and not directory.is_symlink(), f"invalid seal {directory}")
    need(not any(path.is_symlink() for path in directory.rglob("*")), "symlink in seal")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed: set[str] = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe seal member")
        need((directory / rel).is_file() and sha(directory / rel) == digest,
             f"sealed member drift {rel}")
        listed.add(rel.as_posix())
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "non-exhaustive seal")
    return len(listed)


def check_regular(path: Path, digest: str, executable: bool) -> None:
    mode = path.lstat().st_mode
    need(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == digest,
         f"identity/shape drift {path}")
    if executable:
        need(os.access(path, os.X_OK), f"tool not executable {path}")


def validate_static(runner: str, tcl: str) -> None:
    top = '"${LM_SHELL}" -no_init -f "${TCL}"'
    need(runner.count(top) == 1, "not exactly one lm_shell -no_init -f")
    need(runner.count('"${LMUTIL}" lmstat ') == 1, "not exactly one future license query")
    need("exec env -i" in runner, "environment not cleared")
    for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "M2221_ISOLATED_CWD",
                 "M2221_LIBRARY_CACHE", "M2221_FRAME_DIR", "M2221_MILKYWAY_EXEC"):
        need(runner.count(name + "=") >= 1, f"missing isolated environment {name}")
    for suffix in ("home", "tmp", "cache/xdg", "cache/library", "frame_output",
                   "frame_logs", "reports"):
        need(suffix in runner, f"missing isolated directory {suffix}")
    need("M2221_EXPECTED_SOURCE_REVIEW_SHA256" in runner and
         "PASS_M2222_M2221_SOURCE_HAMMER__M2223_ONE_SHOT_AUTHORIZED" in runner,
         "M2222 gate absent")
    need("score_over_100']>=95" in runner and
         "{'p0':0,'p1':0,'p2':0}" in runner, "score/severity gate absent")
    for value in ("generate_frame_commands':0", "create_lib_commands':0",
                  "milkyway_process_runs':0", "pnr_runs':0", "automatic_retry':False"):
        need(value in runner, f"authorization gate drift {value}")
    for name in M2223_NAMES[:3]:
        need(name in runner, f"fresh namespace missing {name}")
    need('[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]]' in runner,
         "virgin result/attempt/work/lock gate absent")

    need("info commands $name" in tcl and
         tcl.count("[list get_app_option_value -name $name]") == 1,
         "command/option observation surface drift")
    command_list = "[list generate_frame_from_mw set_app_options get_app_option_value report_app_options]"
    need(tcl.count(command_list) == 1, "command query list drift")
    need(tcl.count("m2221_query_option $local_name") == 1 and
         tcl.count("m2221_query_option $mw_name") == 1, "option query count drift")
    gate = ("if {$available(set_app_options) && $available(get_app_option_value) &&\n"
            "            [lindex $mw_query 2] == 1}")
    need(tcl.count(gate) == 1, "milkyway option visibility gate drift")
    need(tcl.count("[list set_app_options -name $mw_name -value $milkyway_exec]") == 1 and
         tcl.count("[list get_app_option_value -name $mw_name]") == 1,
         "session-local set/readback drift")
    need("setup file contamination" in tcl and
         all(token in tcl for token in (".synopsys*", "*.setup", ".tclshrc")),
         "setup rejection absent")
    need("[file normalize [pwd]] ne $work" in tcl and
         "[string match \"${work}/*\" $path]" in tcl, "Tcl path containment absent")
    need("frame_files=0 ndm_files=0 nlib_files=0" in tcl, "output absence marker drift")
    executable_forbidden = re.compile(
        r"(?m)^\s*(?:generate_frame_from_mw|create_lib|open_lib|save_lib|read_verilog|"
        r"read_sverilog|place_opt|clock_opt|route_opt|compile_fusion|exec|source|eval|"
        r"uplevel|load|package\s+require)\b")
    need(not executable_forbidden.search(tcl), "executable conversion/child/P&R path")
    need(not re.search(r"(?m)^\s*open\s+\|", tcl), "executable pipe path")


def rejected_static(runner: str, tcl: str) -> None:
    try:
        validate_static(runner, tcl)
    except RuntimeError:
        return
    raise RuntimeError("static mutation accepted")


def refresh_manifest(work: Path) -> None:
    path = work / "execution_output_manifest.json"
    value = json.loads(path.read_text())
    mapping = {
        "execution_contract_sha256": "execution_contract.json",
        "lm_log_sha256": "lm_discovery.log",
        "same_uid_census_before_sha256": "same_uid_census_before.json",
        "same_uid_census_after_sha256": "same_uid_census_after.json",
        "repo_root_before_sha256": "repo_root_before.json",
        "repo_root_after_sha256": "repo_root_after.json",
    }
    for key, name in mapping.items():
        value[key] = sha(work / name)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rejected_receipt(checker, work: Path) -> None:
    try:
        checker.validate(work, work / "receipt.json")
    except checker.Failure:
        return
    raise RuntimeError("receipt mutation accepted")


def replace_log(work: Path, old: str, new: str) -> None:
    path = work / "lm_discovery.log"
    text = path.read_text()
    need(old in text, f"fixture token absent {old}")
    path.write_text(text.replace(old, new, 1))
    refresh_manifest(work)


def main() -> int:
    for path, digest in EXPECTED_SOURCE.items():
        check_regular(path, digest, executable=False)
    for path, digest in EXPECTED_TOOLS.items():
        check_regular(path, digest, executable=True)
    for path, digest in EXPECTED_DOCS.items():
        check_regular(path, digest, executable=False)

    sidecar = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
    outer = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name], "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name], "contract outer seal")
    seals = {"m2208_members": verify_seal(M2208), "m2221_author_members": verify_seal(AUTHOR)}
    predecessor = json.loads((M2208 / "review.json").read_text())
    need(sha(M2208 / "review.json") ==
         "8faf4234f4577cbfa751c5efc5c2ea01baa1f1a59fd4261b3dbf0513ccffa6ed",
         "M2208 review identity")
    need(predecessor["status"] ==
         "PASS_M2208_M2207_FAILURE_HAMMER__CONSUMED_NO_RETRY_NOT_CITABLE__NEW_LM_DISCOVERY_SOURCE_REQUIRED" and
         predecessor["failure"]["first_native_failure"] ==
         "Invalid option name 'lib.configuration.local_output_dir'" and
         predecessor["authorization"]["m2207_retry"] is False,
         "M2208 failure lineage")
    author = json.loads((AUTHOR / "author_receipt.json").read_text())
    need(author["status"] ==
         "PASS_M2221_SOURCE_AUTHOR_RECEIPT__M2222_INDEPENDENT_REVIEW_REQUIRED__M2223_NOT_AUTHORIZED",
         "M2221 author status")

    need(sha(MW_MANIFEST) ==
         "7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3",
         "Milkyway manifest identity")
    members = MW_MANIFEST.read_text().splitlines()
    need(len(members) == 1051, "Milkyway manifest member count")
    for line in members:
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe Milkyway member")
        member = MW_REF / rel
        need(member.is_file() and not member.is_symlink() and sha(member) == digest,
             f"Milkyway member drift {rel}")
    need(sum(path.is_file() for path in MW_REF.rglob("*")) == 1051 and
         not any(path.is_symlink() for path in MW_REF.rglob("*")), "Milkyway inventory shape")

    runner_text = RUNNER.read_text()
    tcl_text = TCL.read_text()
    validate_static(runner_text, tcl_text)
    static_mutations = [
        (runner_text.replace('"${LM_SHELL}" -no_init -f "${TCL}"',
                             '"${LM_SHELL}" -f "${TCL}"', 1), tcl_text),
        (runner_text.replace("exec env -i", "exec env", 1), tcl_text),
        (runner_text + '\n"${LM_SHELL}" -no_init -f "${TCL}"\n', tcl_text),
        (runner_text.replace('[[ ! -e "${RESULT}"', '[[ -e "${RESULT}"', 1), tcl_text),
        (runner_text, tcl_text + "\ngenerate_frame_from_mw forbidden\n"),
        (runner_text, tcl_text + "\nexec $milkyway_exec\n"),
        (runner_text, tcl_text.replace("setup file contamination", "setup contamination", 1)),
        (runner_text, tcl_text.replace("[lindex $mw_query 2] == 1", "1", 1)),
    ]
    for mutated_runner, mutated_tcl in static_mutations:
        rejected_static(mutated_runner, mutated_tcl)

    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    for path in (CENSUS, INVENTORY, CHECKER, TEST):
        compile(path.read_text(), str(path), "exec")
    official = subprocess.run([sys.executable, "-B", "-I", str(TEST)], check=True,
                              text=True, capture_output=True)
    need("PASS_M2221_CPU_ONLY_SOURCE_AND_MUTATION_TESTS cases=8" in official.stdout,
         "official source suite")

    checker = load(CHECKER, "m2223_checker_m2222_independent")
    fixtures = load(TEST, "m2221_fixtures_m2222_independent")
    controls = 0
    receipt_mutations = 0
    with tempfile.TemporaryDirectory(prefix="m2222_independent_") as temporary:
        base = Path(temporary)
        work = fixtures.make_full_work(checker, base / "control_docs_present_runtime_absent")
        result = checker.validate(work, work / "receipt.json")
        need(result["commands"]["generate_frame_from_mw"] == 1 and
             result["options"]["lib.configuration.local_output_dir"]["registered"] == 0 and
             EXPECTED_DOCS[next(path for path in EXPECTED_DOCS if path.name ==
                                "lib.configuration.local_output_dir.3")],
             "documentation/runtime distinction control")
        controls += 1

        cases: list[tuple[str, str]] = [
            ("duplicate_command", "append_duplicate"),
            ("missing_command", "remove_command"),
            ("invalid_command_bit", "invalid_command"),
            ("option_rc_inconsistent", "option_rc"),
            ("set_outside_gate", "set_outside_gate"),
            ("readback_exact_inconsistent", "readback_exact"),
            ("startup_setup_nonzero", "startup_setup"),
            ("side_effect_counter", "side_counter"),
        ]
        for name, action in cases:
            work = fixtures.make_full_work(checker, base / name)
            log = work / "lm_discovery.log"
            text = log.read_text()
            if action == "append_duplicate":
                text += "M2221_COMMAND name=generate_frame_from_mw available=1\n"
            elif action == "remove_command":
                text = text.replace("M2221_COMMAND name=report_app_options available=1\n", "", 1)
            elif action == "invalid_command":
                text = text.replace("name=report_app_options available=1", "name=report_app_options available=2", 1)
            elif action == "option_rc":
                text = text.replace("query_rc=1 registered=0", "query_rc=0 registered=0", 1)
            elif action == "set_outside_gate":
                text = text.replace("name=set_app_options available=1", "name=set_app_options available=0", 1)
            elif action == "readback_exact":
                text = text.replace("exact=1 value_hex=", "exact=0 value_hex=", 1)
            elif action == "startup_setup":
                text = text.replace("setup_files=0", "setup_files=1", 1)
            elif action == "side_counter":
                text = text.replace("generate_calls=0", "generate_calls=1", 1)
            log.write_text(text)
            refresh_manifest(work)
            rejected_receipt(checker, work)
            receipt_mutations += 1

        for name, relative in (("frame", "isolated_cwd/frame_output/forbidden"),
                               ("ndm", "isolated_cwd/forbidden.ndm"),
                               ("nlib", "isolated_cwd/forbidden.nlib")):
            work = fixtures.make_full_work(checker, base / name)
            (work / relative).write_text("forbidden")
            rejected_receipt(checker, work)
            receipt_mutations += 1

        work = fixtures.make_full_work(checker, base / "execution")
        execution_path = work / "execution_contract.json"
        execution = json.loads(execution_path.read_text())
        execution["milkyway_process_runs"] = 1
        execution_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
        refresh_manifest(work)
        rejected_receipt(checker, work)
        receipt_mutations += 1

        work = fixtures.make_full_work(checker, base / "census")
        census_path = work / "same_uid_census_after.json"
        census_value = json.loads(census_path.read_text())
        census_value["matching_process_count"] = 1
        census_path.write_text(json.dumps(census_value, indent=2, sort_keys=True) + "\n")
        refresh_manifest(work)
        rejected_receipt(checker, work)
        receipt_mutations += 1

        work = fixtures.make_full_work(checker, base / "inventory")
        repo_path = work / "repo_root_after.json"
        repo_value = json.loads(repo_path.read_text())
        repo_value["node_count"] = 1
        repo_path.write_text(json.dumps(repo_value, indent=2, sort_keys=True) + "\n")
        refresh_manifest(work)
        rejected_receipt(checker, work)
        receipt_mutations += 1

        work = fixtures.make_full_work(checker, base / "manifest")
        manifest_path = work / "execution_output_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["automatic_retry"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        rejected_receipt(checker, work)
        receipt_mutations += 1

    virgin = sorted(path.name for path in RUNS.iterdir()
                    if any(path.name == name or path.name.startswith(name)
                           for name in M2223_NAMES))
    need(not virgin, f"M2223 namespace not virgin: {virgin}")

    result = {
        "schema": "m2222_independent_mechanical_checks_r1_v1",
        "status": "PASS_M2222_INDEPENDENT_MECHANICAL_CHECKS",
        "source_hashes_exact": len(EXPECTED_SOURCE),
        "tool_hashes_exact_read_only": len(EXPECTED_TOOLS),
        "documentation_hashes_exact_read_only": len(EXPECTED_DOCS),
        "milkyway_inventory_members_verified_read_only": 1051,
        "predecessor_and_author_seals": seals,
        "single_lm_no_init_f_invocation": True,
        "single_future_license_query": True,
        "isolated_env_and_setup_rejection": True,
        "executable_conversion_create_lib_milkyway_pnr_paths": 0,
        "conditional_session_local_milkyway_option_set_readback": True,
        "documentation_present_runtime_option_absent_control": controls,
        "static_mutations_rejected": len(static_mutations),
        "receipt_mutations_rejected": receipt_mutations,
        "m2223_namespace_members": virgin,
        "lm_runs": 0,
        "license_queries": 0,
        "eda_runs": 0,
        "gpu_runs": 0,
        "git_mutation": False,
        "docs359_sha256": sha(DOC359),
        "note": "Zero Milkyway execution is guaranteed by the closed Tcl command surface; before/after census is containment evidence, not an exhaustive micro-short-process observation claim.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
