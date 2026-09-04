#!/usr/bin/python3.12
"""Independent, read-only M2181 source hammer; invokes no LM/EDA/license/GPU."""
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
REVIEW_DIR = Path(__file__).resolve().parent
CONTRACT = HW / "contracts/m2180_m2171_lm_library_conversion_preflight_source_contract_r1_20260904.json"
TCL = HW / "dc_handoff/scripts/run_lm_m2180_library_conversion_preflight.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2180_m2171_lm_library_conversion_preflight_one_shot.sh"
MONITOR = HW / "dc_handoff/scripts/monitor_m2180_lm_conversion_process_tree.py"
CHECKER = HW / "system_simulator/scripts/check_m2180_lm_library_conversion_preflight.py"
TEST = HW / "tests/test_m2180_lm_library_conversion_preflight_source.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2171 = HW / "reviews/m2171_m2170_m2168_icc2_library_import_preflight_failure_hammer_r1_20260904"
AUTHOR = HW / "reviews/m2180_m2171_lm_library_conversion_preflight_source_author_receipt_r1_20260904"
RUNS = HW / "dc_handoff/runs"
LM = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell")
LM_EXEC = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec")
MW = Path("/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway")
EXPECTED_SOURCE = {
    TCL: "20228c790145a814ab4ef7c638c465318b6960c58561bcf99a31ceffa2c9b6d2",
    MONITOR: "e473b6d34b49131f6faeadaad28b9d71fafd8ff06d03464769ea6ad922afb22e",
    CHECKER: "5dfd3cf7f06be4604a2c1c0dde60c1b02d3763de7f28c570d756fe6b6a31b8ee",
    RUNNER: "eed756c2c4a167f662c9467137131b8437005b58c134692e4c2854d427bd5320",
    TEST: "f3bc8cbde5cecb161f721edee908bdd6fee15f561a822aacf2405f80ab87a0e2",
    CONTRACT: "3500f0416deb616f7c6476422b403c773bb042311e784c94df3f33fdb297fcac",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_TOOLS = {
    LM: "1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942",
    LM_EXEC: "3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab",
    MW: "09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec",
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
    need(not any(p.is_symlink() for p in directory.rglob("*")), "symlink in sealed dir")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe seal path")
        need(sha(directory / rel) == digest, "member hash")
        listed.add(rel.as_posix())
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
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
         "tool is not regular executable: " + str(path))
    need(path.resolve(strict=True) == path and sha(path) == digest, "tool identity drift")
    return {"path": str(path), "sha256": digest, "regular": True,
            "nonsymlink": True, "executable": True}


def m2182_census() -> dict:
    names = sorted(p.name for p in RUNS.iterdir()
                   if p.name.startswith("m2182_m2180_lm_library_conversion_preflight") or
                   p.name.startswith(".m2182_m2180_lm_library_conversion_preflight"))
    tool_hits = []
    blocked = {"lm_shell", "lm_shell_exec", "Milkyway"}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / "comm").read_text().strip()
            exe = Path(os.readlink(proc / "exe")).name
        except (OSError, ValueError):
            continue
        if comm in blocked or exe in blocked:
            tool_hits.append({"pid": int(proc.name), "comm": comm, "exe": exe})
    need(not names and not tool_hits, "M2182 census is not empty")
    return {"matching_run_paths": names, "running_lm_processes": tool_hits,
            "empty": True}


def extra_child_mutation() -> dict:
    checker = load(CHECKER, "m2180_checker_independent")
    tests = load(TEST, "m2180_tests_fixture_only")
    isolated = Path("/tmp/m2181_synthetic_no_io")
    tree = tests.good_tree(isolated)
    extra = tests.obs("/usr/bin/sleep", ["/usr/bin/sleep", "5"], isolated)
    tree["all_observed_processes"].append({
        "pid": 103, "starttime_ticks": 1003, "first_ppid": 101,
        "parent_links": [{"ppid": 101, "parent_starttime_ticks": 1001}],
        "exec_observations": [extra]})
    tree["unique_process_identity_count"] = 4
    tree["exec_observation_count"] = 4
    accepted = False
    detail = ""
    try:
        result = checker.validate_process_tree(tree, isolated)
        accepted = True
        detail = repr(result)
    except checker.Failure as exc:
        detail = str(exc)
    return {"mutation": "extra connected non-tool child /usr/bin/sleep",
            "expected": "REJECT", "observed": "ACCEPT" if accepted else "REJECT",
            "fail_closed": not accepted, "detail": detail}


def main() -> int:
    for path, digest in EXPECTED_SOURCE.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "source identity drift: " + str(path))
    contract_side = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
    contract_outer = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
    need(contract_side.read_text().split() == [sha(CONTRACT), CONTRACT.name], "contract sidecar")
    need(contract_outer.read_text().split() == [sha(contract_side), contract_side.name],
         "contract outer")
    seals = {"m2171_members": verify_seal(M2171),
             "m2180_author_members": verify_seal(AUTHOR)}
    prior = read_json(M2171 / "review.json")
    need(sha(M2171 / "review.json") ==
         "c42ffe2cea367f6a0bb43c73279ec1c340fd20f37fac990a5876c8193b52ccb9",
         "M2171 review identity")
    need(prior["status"] ==
         "PASS_M2171_M2170_FAILURE_HAMMER__CONSUMED_NO_RETRY__LM_SHELL_SOURCE_ONLY_NEXT" and
         prior["execution_census"]["license_queries"] == 1 and
         prior["execution_census"]["top_level_icc2_shell_runs"] == 1 and
         prior["execution_census"]["pnr_runs"] == 0 and
         prior["execution_census"]["automatic_retry"] is False and
         prior["failure"]["first_failure"] ==
         "Error: unknown command 'generate_frame_from_mw' (CMD-005)" and
         prior["failure"]["gate2_through_gate6_executed"] is False,
         "M2171 root-cause fingerprint")
    tools = {path.name: exact_tool(path, digest) for path, digest in EXPECTED_TOOLS.items()}

    tcl = TCL.read_text()
    ordered = [
        "set_app_options -name lib.setting.milkyway_exec -value $milkyway_exec",
        "get_app_option_value -name lib.setting.milkyway_exec",
        "M2180_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS",
        "generate_frame_from_mw $frame_name",
    ]
    positions = [tcl.index(token) for token in ordered]
    need(positions == sorted(positions), "Milkyway option order")
    need(tcl.count("generate_frame_from_mw $frame_name") == 1 and "-overwrite" not in tcl,
         "single non-overwriting conversion")
    forbidden = ("create_lib", "read_verilog", "read_sverilog", "place_opt",
                 "create_placement", "clock_opt", "route_opt", "route_auto",
                 "report_timing", "report_power", "report_area")
    active_forbidden = [command for command in forbidden if
                        re.search(rf"(?m)^\s*{re.escape(command)}(?:\s|$)", tcl)]
    need(not active_forbidden, "forbidden LM command")
    runner = RUNNER.read_text()
    seven = ("home", "tmp", "cache/xdg", "cache/library", "frame_output",
             "frame_logs", "reports")
    need(all(f'"${{ISOLATED}}/{item}"' in runner for item in seven), "seven isolated dirs")
    need(all(token in runner for token in ('[[ ! -e "${RESULT}"', '! -e "${ATTEMPT}"',
                                            '! -e "${WORK}"', '! -e "${LOCK}"')),
         "fresh result/attempt/work/lock gate")
    need(runner.count('"${LM_SHELL}" -no_init -f "${TCL}"') == 1 and
         runner.count('"${LMUTIL}" lmstat ') == 1 and
         "'pnr_runs':0" in runner and "'automatic_retry':False" in runner,
         "one-shot budget")
    census = m2182_census()

    official = subprocess.run([sys.executable, str(TEST)], check=True,
                              text=True, capture_output=True,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    need("PASS_M2180_SOURCE_TESTS" in official.stdout and "eda_runs=0" in official.stdout,
         "official source tests")
    mutation = extra_child_mutation()
    severity = {"p0": 0, "p1": 0 if mutation["fail_closed"] else 1, "p2": 0}
    passed = severity == {"p0": 0, "p1": 0, "p2": 0}
    result = {
        "schema": "m2181_m2180_m2171_lm_library_conversion_preflight_source_mechanical_checks_r1_v1",
        "status": ("PASS_M2181_SOURCE_MECHANICAL_CHECKS" if passed else
                   "FAIL_M2181_PROCESS_CENSUS_MUTATION_NOT_FAIL_CLOSED"),
        "identity": {"contract_sha256": sha(CONTRACT), "runner_sha256": sha(RUNNER),
                     "tcl_sha256": sha(TCL), "monitor_sha256": sha(MONITOR),
                     "checker_sha256": sha(CHECKER), "test_sha256": sha(TEST),
                     "docs359_sha256": sha(DOC359)},
        "m2171_root_cause": {"fingerprint_match": True,
            "first_failure": prior["failure"]["first_failure"],
            "m2170_consumed": True, "retry": False, "pnr_runs": 0},
        "seals": seals, "tools": tools,
        "lm_tcl": {"set_readback_before_conversion": True,
                   "generate_frame_calls": 1, "forbidden_active_commands": [],
                   "frame_only": True},
        "freshness": {"seven_isolated_directories": list(seven),
                      "result_attempt_work_lock_gated": True, "m2182_census": census},
        "source_suite": {"stdout": official.stdout.strip(), "return_code": official.returncode},
        "independent_mutation": mutation,
        "severity_counts": severity,
        "execution": {"lm_runs": 0, "eda_runs": 0, "license_queries": 0,
                      "gpu_runs": 0, "pnr_runs": 0, "m2182_runs": 0,
                      "m2180_source_modified": False, "git_mutation": False},
        "authorization": {"m2182": passed, "license_queries": 1 if passed else 0,
                          "top_level_lm_shell_runs": 1 if passed else 0,
                          "pnr_runs": 0, "automatic_retry": False},
    }
    out = REVIEW_DIR / "mechanical_checks.json"
    need(not out.exists(), "fresh mechanical output required")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
