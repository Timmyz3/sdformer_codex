#!/usr/bin/python3.12
"""Independent read-only M2190 hammer; invokes no LM/EDA/license/GPU."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent
CONTRACT = HW / "contracts/m2189_m2181_lm_library_conversion_preflight_source_contract_r1_20260904.json"
TCL = HW / "dc_handoff/scripts/run_lm_m2189_library_conversion_preflight.tcl"
OLD_TCL = HW / "dc_handoff/scripts/run_lm_m2180_library_conversion_preflight.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2189_m2181_lm_library_conversion_preflight_one_shot.sh"
MONITOR = HW / "dc_handoff/scripts/monitor_m2189_lm_conversion_process_tree.py"
CHECKER = HW / "system_simulator/scripts/check_m2189_lm_library_conversion_preflight.py"
TEST = HW / "tests/test_m2189_lm_library_conversion_preflight_source.py"
OLD_CHECKER = HW / "system_simulator/scripts/check_m2180_lm_library_conversion_preflight.py"
OLD_TEST = HW / "tests/test_m2180_lm_library_conversion_preflight_source.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2171 = HW / "reviews/m2171_m2170_m2168_icc2_library_import_preflight_failure_hammer_r1_20260904"
M2181 = HW / "reviews/m2181_m2180_m2171_lm_library_conversion_preflight_source_hammer_r1_20260904"
AUTHOR = HW / "reviews/m2189_m2181_lm_library_conversion_preflight_source_author_receipt_r1_20260904"
RUNS = HW / "dc_handoff/runs"
MW_MANIFEST = HW / "dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
MW_REF = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140")
LM = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell")
LM_EXEC = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec")
MW = Path("/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
EXPECTED_SOURCE = {
    TCL: "aaba6057fbcc0723f35fccd8a990bfa111ca6b98cea479fb11ee168df2c98e80",
    MONITOR: "c37f393c31e8558042a8db9247f6782d2546d4feae2c202a5c208c8f3752dd60",
    CHECKER: "3c64cddfa1c44b14ba2a7b8cb4cc81ee5923ff51374d7d0bfb0e4733d2f5341c",
    RUNNER: "74c3459fdf434a74b1b325955e4ecb2c82d9cf883ff5d0d70c9c483ce255aece",
    TEST: "367b4ad347daf1995023c1240347c60b706accce25928096f1182f6b25e9df88",
    CONTRACT: "6810302d2ec525eed23e99a29e50c730fad068a998e0a28f4566fc447e395f1d",
    OLD_TCL: "20228c790145a814ab4ef7c638c465318b6960c58561bcf99a31ceffa2c9b6d2",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_TOOLS = {
    LM: "1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942",
    LM_EXEC: "3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab",
    MW: "09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    need(isinstance(value, dict), "JSON object required: " + str(path))
    return value


def verify_seal(directory: Path) -> int:
    need(directory.is_dir() and not directory.is_symlink(), "sealed dir invalid")
    need(not any(path.is_symlink() for path in directory.rglob("*")), "symlink in seal")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed: set[str] = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe seal path")
        need(sha(directory / rel) == digest, "member hash")
        listed.add(rel.as_posix())
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "non-exhaustive seal")
    return len(listed)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "module load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_tool(path: Path, digest: str) -> dict:
    mode = path.lstat().st_mode
    need(stat.S_ISREG(mode) and not path.is_symlink() and os.access(path, os.X_OK),
         "tool is not regular executable")
    need(path.resolve(strict=True) == path and sha(path) == digest, "tool identity drift")
    return {"path": str(path), "sha256": digest, "regular": True,
            "nonsymlink": True, "executable": True}


def census() -> dict:
    patterns = (
        "m2182_m2180_lm_library_conversion_preflight_raw_r1_20260904",
        ".m2182_m2180_lm_library_conversion_preflight",
        "m2191_m2189_lm_library_conversion_preflight_raw_r1_20260904",
        ".m2191_m2189_lm_library_conversion_preflight",
    )
    paths = sorted(path.name for path in RUNS.iterdir()
                   if any(path.name.startswith(prefix) for prefix in patterns))
    blocked = {"vcs", "simv", "dc_shell", "pt_shell", "fm_shell", "icc2_shell",
               "icc2_exec", "dgcom_exec", "lm_shell", "lm_shell_exec", "Milkyway",
               "lmutil", "lmstat"}
    processes = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / "comm").read_text().strip()
            exe = Path(os.readlink(proc / "exe")).name
            argv = {Path(item.decode(errors="replace")).name for item in
                    (proc / "cmdline").read_bytes().split(b"\0") if item}
        except (OSError, ValueError):
            continue
        if comm in blocked or exe in blocked or blocked & argv:
            processes.append({"pid": int(proc.name), "comm": comm, "exe": exe})
    need(not paths and not processes, "attempt/run/tool census not empty")
    return {"matching_run_paths": paths, "running_blocked_processes": processes, "empty": True}


def payload_with_extra_sleep(tests, isolated: Path, parent_pid: int,
                             parent_ticks: int, duration: str) -> dict:
    tree = tests.good_tree(isolated)
    extra = tests.obs("/usr/bin/sleep", ["/usr/bin/sleep", duration], isolated)
    tree["all_observed_processes"].append({
        "pid": 103, "starttime_ticks": 1003, "first_ppid": parent_pid,
        "parent_links": [{"ppid": parent_pid, "parent_starttime_ticks": parent_ticks}],
        "exec_observations": [extra]})
    tree["unique_process_identity_count"] += 1
    tree["exec_observation_count"] += 1
    return tree


def rejected(checker, payload: dict, isolated: Path) -> tuple[bool, str]:
    try:
        value = checker.validate_process_tree(payload, isolated)
    except checker.Failure as exc:
        return True, str(exc)
    return False, repr(value)


def collapsed_control(checker, tests, isolated: Path) -> dict:
    base = tests.good_tree(isolated)
    root = copy.deepcopy(base["all_observed_processes"][0])
    actual = copy.deepcopy(base["all_observed_processes"][1]["exec_observations"][0])
    mw = copy.deepcopy(base["all_observed_processes"][2])
    bootstrap = {"comm": "bash", "exe_path": "/usr/bin/bash",
                 "cmdline": ["/usr/bin/bash", str(RUNNER)], "selected_environment": {}}
    root["exec_observations"] = [bootstrap, root["exec_observations"][0], actual]
    mw["first_ppid"] = 100
    mw["parent_links"] = [{"ppid": 100, "parent_starttime_ticks": 1000}]
    flat = ([{"pid": 100, "starttime_ticks": 1000, **obs}
             for obs in root["exec_observations"]] +
            [{"pid": 102, "starttime_ticks": 1002, **obs}
             for obs in mw["exec_observations"]])
    tree = {"schema": "m2189_lm_conversion_process_tree_r1_v1", "root_pid": 100,
            "root_seen": True, "sample_count": 9, "unique_process_identity_count": 2,
            "exec_observation_count": len(flat),
            "lm_wrapper_observations": [x for x in flat if str(LM) in x["cmdline"]],
            "lm_actual_exec_observations": [x for x in flat if x["exe_path"] == str(LM_EXEC)],
            "milkyway_child_observations": [x for x in flat if x["exe_path"] == str(MW)],
            "unexpected_process_identities": [], "unexpected_process_observations": [],
            "all_observed_processes": [root, mw]}
    return checker.validate_process_tree(tree, isolated)


def main() -> int:
    for path, digest in EXPECTED_SOURCE.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "source identity drift: " + str(path))
    sidecar = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
    outer = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name], "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name], "contract outer seal")
    seals = {"m2171_members": verify_seal(M2171),
             "m2181_members": verify_seal(M2181),
             "m2189_author_members": verify_seal(AUTHOR)}
    m2181 = read_json(M2181 / "review.json")
    need(sha(M2181 / "review.json") ==
         "2364f4323cd7c2beea3468201d5c6e3e0eaca7b8c25516c6a5a30ec530bc575c",
         "M2181 review identity")
    need(m2181["status"] ==
         "FAIL_M2181_M2180_SOURCE_HAMMER__M2182_NOT_AUTHORIZED__SOURCE_REPAIR_REQUIRED" and
         m2181["finding"]["name"] == "process allowlist is not exhaustive",
         "M2181 failure lineage")
    author = read_json(AUTHOR / "author_receipt.json")
    need(author["status"] ==
         "PASS_M2189_AUTHOR_SOURCE_ONLY__M2190_INDEPENDENT_HAMMER_REQUIRED__NO_EXECUTION",
         "M2189 author status")
    tools = {path.name: exact_tool(path, digest) for path, digest in EXPECTED_TOOLS.items()}
    need(sha(MW_MANIFEST) ==
         "7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3",
         "Milkyway manifest identity")
    manifest_lines = MW_MANIFEST.read_text().splitlines()
    need(len(manifest_lines) == 1051 and not any(path.is_symlink() for path in MW_REF.rglob("*")),
         "Milkyway inventory shape")
    manifest_check = subprocess.run(["sha256sum", "-c", str(MW_MANIFEST)], cwd=MW_REF,
                                    check=True, text=True, capture_output=True)
    need(manifest_check.stdout.count(": OK") == 1051, "Milkyway inventory members")

    tcl = TCL.read_text()
    normalized = (tcl.replace("M2189", "M2180").replace("m2189", "m2180")
                  .replace("M2191", "M2182").replace("M2192", "M2183"))
    need(normalized == OLD_TCL.read_text(), "TCL semantic normalization")
    ordered = [
        "set_app_options -name lib.setting.milkyway_exec -value $milkyway_exec",
        "get_app_option_value -name lib.setting.milkyway_exec",
        "M2189_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS",
        "generate_frame_from_mw $frame_name",
    ]
    positions = [tcl.index(token) for token in ordered]
    need(positions == sorted(positions) and tcl.count("generate_frame_from_mw $frame_name") == 1,
         "option/readback/generate ordering")
    forbidden = ("create_lib", "read_verilog", "read_sverilog", "place_opt",
                 "clock_opt", "route_opt", "report_timing", "report_area", "report_power")
    need(not any(re.search(rf"(?m)^\s*{name}(?:\s|$)", tcl) for name in forbidden),
         "design/P&R command")
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    for path in (MONITOR, CHECKER, TEST):
        compile(path.read_text(), str(path), "exec")
    official = subprocess.run([sys.executable, str(TEST)], check=True, text=True,
                              capture_output=True,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    need("native_controls=1" in official.stdout and "process_mutations=14" in official.stdout and
         "eda_runs=0" in official.stdout and "license_queries=0" in official.stdout,
         "official source suite")

    new_checker = load(CHECKER, "m2189_checker_independent")
    new_tests = load(TEST, "m2189_tests_fixture_only")
    old_checker = load(OLD_CHECKER, "m2180_checker_independent")
    old_tests = load(OLD_TEST, "m2180_tests_fixture_only")
    isolated = Path("/tmp/m2190_synthetic_no_io/isolated")
    old_sleep = payload_with_extra_sleep(old_tests, isolated, 101, 1001, "5")
    old_rejected, old_detail = rejected(old_checker, old_sleep, isolated)
    need(not old_rejected, "M2181 failure no longer reproduced")
    new_sleep = payload_with_extra_sleep(new_tests, isolated, 101, 1001, "5")
    new_rejected, new_detail = rejected(new_checker, new_sleep, isolated)
    need(new_rejected, "M2189 did not repair M2181 extra child")
    collapse = collapsed_control(new_checker, new_tests, isolated)
    need(collapse["identities"] == 2 and collapse["unexpected_identities"] == 0,
         "root/actual collapse control")

    runner_text = RUNNER.read_text()
    need('while [[ ! -e "${LAUNCH_GATE}" ]]; do /usr/bin/sleep 0.01; done' in runner_text,
         "runner bootstrap sleep source")
    bootstrap_sleep = payload_with_extra_sleep(new_tests, isolated, 100, 1000, "0.01")
    bootstrap_rejected, bootstrap_detail = rejected(new_checker, bootstrap_sleep, isolated)
    need(bootstrap_rejected, "bootstrap sleep should conflict with three-identity allowlist")
    wrapper = LM.read_text()
    wrapper_evidence = {
        "dirname": "dir_path=`dirname $0`",
        "uname": "case \"`uname`\" in",
        "cat_grep_cut": "cat /etc/*-release | grep VERSION_ID | cut",
        "basename": "bname=`basename $dir`",
    }
    need(all(snippet in wrapper for snippet in wrapper_evidence.values()),
         "regular wrapper helper evidence")
    need("time.sleep(0.005)" in MONITOR.read_text(), "monitor polling interval")
    process_finding = {
        "severity": "P1",
        "name": "the promised exhaustive three-identity census conflicts with the required regular wrapper path",
        "evidence": {
            "runner_connected_bootstrap_child": "/usr/bin/sleep 0.01",
            "bootstrap_child_checker_result": "REJECT",
            "bootstrap_child_detail": bootstrap_detail,
            "regular_lm_shell_external_helpers": sorted(wrapper_evidence),
            "monitor_method": "5 ms /proc polling",
        },
        "cause": "The monitor begins before the launch gate and classifies every sampled root descendant, while the runner and pinned POSIX lm_shell wrapper themselves spawn identities outside set(root, actual, Milkyway). Sampling may reject them when observed or miss them and overclaim exhaustiveness.",
        "risk": "The unique M2191 attempt is nondeterministic at the evidence gate and cannot prove other_connected_descendants=0 or other_process_observations=0.",
        "minimum_repair": "Create a new source identity that defines and verifies an exact bootstrap/wrapper-helper phase separately from the actual-LM subtree, removes the polled sleep launch gate, and uses event-complete child accounting (or narrows the claim to observed samples). Re-hammer before any license or LM action.",
    }
    result = {
        "schema": "m2190_m2189_m2181_lm_library_conversion_preflight_source_mechanical_checks_r1_v1",
        "status": "FAIL_M2190_EXHAUSTIVE_PROCESS_CONTRACT_NOT_EXECUTABLE",
        "identity": {"contract_sha256": sha(CONTRACT), "runner_sha256": sha(RUNNER),
                     "tcl_sha256": sha(TCL), "monitor_sha256": sha(MONITOR),
                     "checker_sha256": sha(CHECKER), "test_sha256": sha(TEST),
                     "docs359_sha256": sha(DOC359),
                     "m2181_review_sha256": sha(M2181 / "review.json")},
        "seals": seals,
        "tools": tools,
        "semantic_preservation": {"normalized_m2189_equals_frozen_m2180": True,
                                  "milkyway_option_set_readback_before_generate": True,
                                  "generate_frame_calls": 1,
                                  "design_import_or_pnr_commands": 0},
        "lineage": {"m2181_extra_sleep_accepted": not old_rejected,
                    "m2181_detail": old_detail,
                    "m2189_extra_sleep_rejected": new_rejected,
                    "m2189_detail": new_detail,
                    "root_actual_collapse_control_accepted": True,
                    "root_actual_collapse_result": collapse},
        "source_suite": {"return_code": official.returncode,
                         "stdout": official.stdout.strip(),
                         "native_controls": 1, "native_mutations": 1,
                         "process_controls": 1, "process_mutations": 14},
        "input_inventory": {"manifest_sha256": sha(MW_MANIFEST), "regular_files": 1051,
                            "verified_members": 1051, "symbolic_links": 0},
        "freshness": {"census": census()},
        "finding": process_finding,
        "severity_counts": {"p0": 0, "p1": 1, "p2": 0},
        "execution": {"m2182_runs": 0, "m2191_runs": 0, "lm_runs": 0,
                      "eda_runs": 0, "license_queries": 0, "gpu_runs": 0,
                      "pnr_runs": 0, "git_mutation": False,
                      "m2189_source_modified": False, "docs359_modified": False},
        "authorization": {"m2191": False, "license_queries": 0,
                          "top_level_lm_shell_runs": 0, "pnr_runs": 0,
                          "automatic_retry": False},
    }
    output = HERE / "mechanical_checks.json"
    need(not output.exists(), "fresh mechanical output required")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
