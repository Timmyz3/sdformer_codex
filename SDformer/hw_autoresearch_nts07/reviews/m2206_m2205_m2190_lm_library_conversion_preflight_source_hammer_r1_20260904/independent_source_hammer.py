#!/usr/bin/python3.12
"""Independent CPU-only M2206 source hammer; never invokes LM/EDA/license/GPU/Git."""
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
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent
CONTRACT = HW / "contracts/m2205_m2190_lm_library_conversion_preflight_source_contract_r1_20260904.json"
TCL = HW / "dc_handoff/scripts/run_lm_m2205_library_conversion_preflight.tcl"
MONITOR = HW / "dc_handoff/scripts/monitor_m2205_lm_conversion_sampled_processes.py"
CENSUS = HW / "dc_handoff/scripts/census_m2205_same_uid_tools.py"
CHECKER = HW / "system_simulator/scripts/check_m2205_lm_library_conversion_preflight.py"
RUNNER = HW / "dc_handoff/scripts/run_m2205_m2190_lm_library_conversion_preflight_one_shot.sh"
TEST = HW / "tests/test_m2205_lm_library_conversion_preflight_source.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2190 = HW / "reviews/m2190_m2189_m2181_lm_library_conversion_preflight_source_hammer_r1_20260904"
AUTHOR = HW / "reviews/m2205_m2190_lm_library_conversion_preflight_source_author_receipt_r1_20260904"
RUNS = HW / "dc_handoff/runs"
MW_MANIFEST = HW / "dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
MW_REF = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140")
LM = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell")
LM_EXEC = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec")
MW = Path("/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
EXPECTED_SOURCE = {
    TCL: "c9ecf9eda32bd8d79f65e108d84c2851dc8a392f5ca8019fda3bf4a035dc6505",
    MONITOR: "4dd651bf0c55afe95d05c589ebf12f242144393f1d66468affc373920a576394",
    CENSUS: "ec452719e68c5caa88039ec7e37512647e2c737d54842eb2adf55e66639160bf",
    CHECKER: "74b7c82cf4c39ce7648ad0f35ada34a9e239019aae8e7100879350dca143564b",
    RUNNER: "ae4d01346c948bf8de6be37c135fca2ea79473b83151ff0e2a62870a880f8867",
    TEST: "624bdd3373203bde36300434543147bf31b99a555bc1ca6d89730f5812787cff",
    CONTRACT: "65ae56329a89088f9f329cce3be51fbd4d98378fbdd8741585262ed7164d6deb",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_TOOLS = {
    LM: "1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942",
    LM_EXEC: "3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab",
    MW: "09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
}
BLOCKED = {"vcs", "simv", "dc_shell", "pt_shell", "fm_shell", "icc2_shell",
           "icc2_exec", "dgcom_exec", "lm_shell", "lm_shell_exec", "Milkyway",
           "lmutil", "lmstat"}


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
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_seal(directory: Path) -> int:
    need(directory.is_dir() and not directory.is_symlink(), f"invalid seal dir {directory}")
    need(not any(path.is_symlink() for path in directory.rglob("*")), "symlink in seal")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed: set[str] = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe member")
        need(sha(directory / rel) == digest, f"member drift {rel}")
        listed.add(rel.as_posix())
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "non-exhaustive seal")
    return len(listed)


def tool_identity(path: Path, digest: str) -> dict[str, object]:
    mode = path.lstat().st_mode
    need(stat.S_ISREG(mode) and not path.is_symlink() and os.access(path, os.X_OK),
         f"tool shape drift {path}")
    need(path.resolve(strict=True) == path and sha(path) == digest, f"tool identity drift {path}")
    return {"path": str(path), "sha256": digest, "regular": True,
            "nonsymlink": True, "executable": True, "executed": False}


def census() -> dict[str, object]:
    prefixes = (
        "m2182_m2180_lm_library_conversion_preflight_raw_r1_20260904",
        ".m2182_m2180_lm_library_conversion_preflight",
        "m2191_m2189_lm_library_conversion_preflight_raw_r1_20260904",
        ".m2191_m2189_lm_library_conversion_preflight",
        "m2207_m2205_lm_library_conversion_preflight_raw_r1_20260904",
        ".m2207_m2205_lm_library_conversion_preflight",
    )
    run_paths = sorted(path.name for path in RUNS.iterdir()
                       if any(path.name.startswith(prefix) for prefix in prefixes))
    processes = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / "comm").read_text().strip()
            exe = Path(os.readlink(proc / "exe")).name
            argv_names = {Path(item.decode(errors="replace")).name for item in
                          (proc / "cmdline").read_bytes().split(b"\0") if item}
        except (OSError, ValueError):
            continue
        if comm in BLOCKED or exe in BLOCKED or BLOCKED & argv_names:
            processes.append({"pid": int(proc.name), "comm": comm, "exe": exe,
                              "argv_names": sorted(argv_names)})
    need(not run_paths, f"forbidden attempt/run paths: {run_paths}")
    need(not processes, f"same-UID blocked processes: {processes}")
    return {"matching_run_paths": run_paths, "matching_processes": processes,
            "matching_process_count": 0, "status": "PASS_EMPTY"}


def reject_process(checker, payload: dict, isolated: Path) -> str:
    try:
        checker.validate_sampled_process(payload, isolated)
    except checker.Failure as exc:
        return str(exc)
    raise RuntimeError("process mutation accepted")


def reject_full(checker, work: Path, output: Path) -> str:
    try:
        checker.validate(work, output)
    except checker.Failure as exc:
        return str(exc)
    raise RuntimeError("full receipt mutation accepted")


def mutate_path(base: dict, path: tuple[str, ...], value: object) -> dict:
    item = copy.deepcopy(base)
    target = item
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return item


def main() -> int:
    before = census()
    for path, digest in EXPECTED_SOURCE.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             f"source identity drift {path}")
    sidecar = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
    outer = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name], "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name], "contract outer seal")
    seals = {"m2190_members": verify_seal(M2190),
             "m2205_author_members": verify_seal(AUTHOR)}
    m2190 = read_json(M2190 / "review.json")
    need(sha(M2190 / "review.json") ==
         "e0f49c4c61428a49d1bfeb4fafc3c2abea9fd50a1a90692f79a34de3c8707929",
         "M2190 review identity")
    need(m2190.get("status") ==
         "FAIL_M2190_M2189_SOURCE_HAMMER__M2191_NOT_AUTHORIZED__SOURCE_REPAIR_REQUIRED" and
         m2190.get("severity_counts") == {"p0": 0, "p1": 1, "p2": 0},
         "M2190 failure lineage")
    need("exhaustive process contract" in m2190["finding"]["name"], "M2190 root cause")
    author = read_json(AUTHOR / "author_receipt.json")
    need(author.get("status") ==
         "PASS_M2205_AUTHOR_SOURCE_ONLY__M2206_INDEPENDENT_HAMMER_REQUIRED__NO_EXECUTION",
         "M2205 author status")
    tools = {path.name: tool_identity(path, digest) for path, digest in EXPECTED_TOOLS.items()}

    need(sha(MW_MANIFEST) ==
         "7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3",
         "Milkyway manifest identity")
    lines = MW_MANIFEST.read_text().splitlines()
    need(len(lines) == 1051, "Milkyway manifest count")
    verified = 0
    for line in lines:
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe reference member")
        member = MW_REF / rel
        need(member.is_file() and not member.is_symlink() and sha(member) == digest,
             f"Milkyway member drift {rel}")
        verified += 1
    need(sum(1 for path in MW_REF.rglob("*") if path.is_file()) == 1051 and
         not any(path.is_symlink() for path in MW_REF.rglob("*")), "reference inventory shape")

    monitor_text = MONITOR.read_text()
    runner_text = RUNNER.read_text()
    tcl_text = TCL.read_text()
    need("time.sleep(0.005)" in monitor_text and
         not re.search(r"\b(subprocess|Popen|os\.system)\b", monitor_text), "monitor polling")
    need("/usr/bin/sleep" not in monitor_text and "/usr/bin/sleep" not in runner_text,
         "child-sleep regression")
    need('"${LM_SHELL}" -no_init -f "${TCL}"' in runner_text and
         runner_text.count('"${LM_SHELL}" -no_init -f "${TCL}"') == 1,
         "top-level LM command count")
    need(runner_text.count('"${LMUTIL}" lmstat ') == 1 and
         runner_text.count('"${CENSUS}" --phase before') == 1 and
         runner_text.count('"${CENSUS}" --phase after') == 1, "future count/census closure")
    wait_at = tcl_text.index("M2205_GATE0_TCL_WAITING")
    release_at = tcl_text.index("M2205_GATE0_TCL_RELEASED")
    option_at = tcl_text.index("set_app_options -name lib.setting.milkyway_exec")
    generate_at = tcl_text.index("generate_frame_from_mw $frame_name")
    need(wait_at < release_at < option_at < generate_at and
         tcl_text.count("generate_frame_from_mw $frame_name") == 1, "Tcl gate/order")
    prefix = tcl_text[:release_at]
    need(not any(token in prefix for token in
                 ("set_app_var", "set_app_options", "generate_frame_from_mw", " open ", " w]")),
         "conversion-side effect before gate release")
    for command in ("create_lib", "read_verilog", "read_sverilog", "place_opt",
                    "clock_opt", "route_opt", "report_timing", "report_area", "report_power"):
        need(not re.search(rf"(?m)^\s*{re.escape(command)}(?:\s|$)", tcl_text),
             f"forbidden design/P&R command {command}")
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    for path in (MONITOR, CENSUS, CHECKER, TEST):
        compile(path.read_text(), str(path), "exec")
    official = subprocess.run([sys.executable, "-B", "-I", str(TEST)], check=True,
                              text=True, capture_output=True)
    need("native_controls=1 native_mutations=1" in official.stdout and
         "process_controls=1 process_mutations=25" in official.stdout and
         "full_receipt_controls=1 full_receipt_mutations=5" in official.stdout and
         "lm_runs=0 eda_runs=0 license_queries=0 gpu_runs=0" in official.stdout,
         "official source suite")

    checker = load(CHECKER, "m2205_checker_m2206_independent")
    fixtures = load(TEST, "m2205_fixtures_m2206_independent")
    with tempfile.TemporaryDirectory(prefix="m2206_independent_") as temporary:
        base = Path(temporary)
        native = base / "control.ndm"
        native.write_bytes(checker.NATIVE_HEADER + b"M2206_INDEPENDENT_CONTROL")
        native_control = checker.validate_native_frame(native)
        bad_native = base / "bad.ndm"
        bad_native.write_bytes(b"not-native")
        try:
            checker.validate_native_frame(bad_native)
        except checker.Failure as exc:
            native_rejection = str(exc)
        else:
            raise RuntimeError("native mutation accepted")

        isolated = base / "isolated"
        good = fixtures.good_process(isolated)
        process_control = checker.validate_sampled_process(good, isolated)
        scalar_cases = [
            ("schema", ("schema",), "wrong"),
            ("status", ("status",), "FAIL"),
            ("sampled_claim", ("claim_scope", "sampled_live_processes_only"), False),
            ("exhaustive_claim", ("claim_scope", "exhaustive_short_lived_processes"), True),
            ("gate_released", ("gate", "released"), False),
            ("gate_creator", ("gate", "created_by_monitor"), False),
            ("gate_token", ("gate", "token"), "bad"),
            ("wait_marker", ("gate", "tcl_wait_marker_seen"), False),
            ("stability", ("gate", "actual_stable_samples_observed"), 2),
            ("frame_preexists", ("gate", "frame_absent_before_release"), False),
            ("release_time", ("gate", "release_monotonic_ns"), 0),
            ("violation", ("violation",), "extra child"),
            ("post_samples", ("post_gate_sample_count",), 0),
            ("actual_count", ("sampled_actual_identity_count",), 2),
            ("milkyway_count", ("sampled_milkyway_identity_count",), 2),
            ("actual_identity_path", ("actual_identity", "exe_path"), "/tmp/fake_lm"),
        ]
        process_results = []
        for name, path, value in scalar_cases:
            process_results.append({"name": name,
                                    "rejection": reject_process(checker, mutate_path(good, path, value), isolated)})
        item = copy.deepcopy(good)
        item["pre_gate_milkyway_observations"] = [{"exe_path": str(MW)}]
        process_results.append({"name": "pre_gate_milkyway",
                                "rejection": reject_process(checker, item, isolated)})
        item = copy.deepcopy(good)
        item["unexpected_sampled_post_gate_descendants"] = [{"exe_path": "/usr/bin/dirname"}]
        process_results.append({"name": "unexpected_descendant_field",
                                "rejection": reject_process(checker, item, isolated)})
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][0]["exec_observations"][0]["exe_path"] = "/tmp/fake_lm"
        process_results.append({"name": "actual_exec_drift",
                                "rejection": reject_process(checker, item, isolated)})
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][1]["exec_observations"][0]["exe_path"] = "/tmp/fake_mw"
        process_results.append({"name": "milkyway_exec_drift",
                                "rejection": reject_process(checker, item, isolated)})
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][0]["exec_observations"][0]["selected_environment"]["HOME"] = "/tmp/wrong"
        process_results.append({"name": "actual_environment_drift",
                                "rejection": reject_process(checker, item, isolated)})
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][1]["exec_observations"][0]["selected_environment"]["HOME"] = "/tmp/wrong"
        process_results.append({"name": "milkyway_environment_drift",
                                "rejection": reject_process(checker, item, isolated)})
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][1]["parent_links"] = [
            {"ppid": 100, "parent_starttime_ticks": 1000}]
        process_results.append({"name": "milkyway_reparent",
                                "rejection": reject_process(checker, item, isolated)})
        item = copy.deepcopy(good)
        item["post_gate_actual_subtree_processes"][0]["exec_observations"][0]["cmdline"] = [str(LM_EXEC), "-no_init"]
        process_results.append({"name": "actual_command_drift",
                                "rejection": reject_process(checker, item, isolated)})
        item = copy.deepcopy(good)
        extra = fixtures.identity(104, 1004, 101, 1002, [fixtures.observation(
            "/usr/bin/dirname", ["dirname", str(LM)], isolated, "post_gate", env=False)])
        item["post_gate_actual_subtree_processes"].append(extra)
        process_results.append({"name": "extra_post_gate_descendant",
                                "rejection": reject_process(checker, item, isolated)})
        need(len(process_results) == 25, "independent process mutation count")

        work, _ = fixtures.make_full_work(checker, base / "full")
        receipt = work / "independent_control_receipt.json"
        full_control = checker.validate(work, receipt)
        full_results = []
        log = work / "lm_preflight.log"
        original = log.read_bytes()
        log.write_bytes(original + b"unexpected\n")
        full_results.append({"name": "log_drift",
                             "rejection": reject_full(checker, work, work / "reject_log.json")})
        log.write_bytes(original)
        census_before = work / "same_uid_census_before.json"
        original = census_before.read_bytes()
        changed = json.loads(original)
        changed["matching_process_count"] = 1
        fixtures.write_json(census_before, changed)
        full_results.append({"name": "before_census_nonempty",
                             "rejection": reject_full(checker, work, work / "reject_census.json")})
        census_before.write_bytes(original)
        frame = work / "isolated_cwd/frame_output/m2205_tcbn28hpcplusbwp35p140_frame.ndm"
        original = frame.read_bytes()
        frame.write_bytes(original + b"drift")
        full_results.append({"name": "frame_drift",
                             "rejection": reject_full(checker, work, work / "reject_frame.json")})
        frame.write_bytes(original)
        process_path = work / "sampled_processes.json"
        original = process_path.read_bytes()
        changed = json.loads(original)
        changed["claim_scope"]["exhaustive_short_lived_processes"] = True
        fixtures.write_json(process_path, changed)
        full_results.append({"name": "claim_widening",
                             "rejection": reject_full(checker, work, work / "reject_claim.json")})
        process_path.write_bytes(original)
        execution = work / "execution_contract.json"
        original = execution.read_bytes()
        changed = json.loads(original)
        changed["top_level_lm_shell_runs"] = 2
        fixtures.write_json(execution, changed)
        full_results.append({"name": "execution_count_drift",
                             "rejection": reject_full(checker, work, work / "reject_execution.json")})
        execution.write_bytes(original)
        need(len(full_results) == 5, "independent full mutation count")

    after = census()
    result = {
        "schema": "m2206_m2205_m2190_lm_library_conversion_preflight_source_mechanical_checks_r1_v1",
        "status": "PASS_M2206_M2205_SOURCE_MECHANICAL_CHECKS",
        "identity": {"contract_sha256": sha(CONTRACT), "runner_sha256": sha(RUNNER),
                     "tcl_sha256": sha(TCL), "monitor_sha256": sha(MONITOR),
                     "census_sha256": sha(CENSUS), "checker_sha256": sha(CHECKER),
                     "test_sha256": sha(TEST), "m2190_review_sha256": sha(M2190 / "review.json"),
                     "m2190_manifest_sha256": sha(M2190 / "SHA256SUMS"),
                     "docs359_sha256": sha(DOC359)},
        "seals": seals,
        "tool_identities_read_only": tools,
        "input_inventory": {"manifest_sha256": sha(MW_MANIFEST), "members": len(lines),
                            "verified_members": verified, "symbolic_links": 0},
        "m2190_lineage": {"failure_reproduced_from_sealed_review": True,
                          "failed_claim": "exhaustive whole-wrapper process census",
                          "repair": "sampled-live gate with disclosed pre-gate helpers and strict post-gate actual subtree"},
        "static_source": {"monitor_uses_python_sleep": True, "monitor_child_sleep": False,
                          "runner_child_sleep": False, "tcl_gate_before_conversion": True,
                          "generate_frame_calls": 1, "design_or_pnr_commands": 0,
                          "future_lm_shell_commands": 1, "future_license_queries": 1,
                          "same_uid_census_before_after": True,
                          "claim_is_sampled_not_microprocess_exhaustive": True},
        "official_reproduction": {"return_code": official.returncode,
                                  "stdout": official.stdout.strip()},
        "independent_reproduction": {
            "native_controls": 1, "native_control": native_control,
            "native_mutations": 1, "native_rejection": native_rejection,
            "process_controls": 1, "process_control": process_control,
            "process_mutations": len(process_results), "process_rejections": process_results,
            "full_receipt_controls": 1, "full_receipt_status": full_control["status"],
            "full_receipt_mutations": len(full_results), "full_receipt_rejections": full_results,
        },
        "freshness": {"before": before, "after": after, "m2207_paths": 0,
                      "m2182_paths": 0, "m2191_paths": 0},
        "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
        "execution": {"m2182_runs": 0, "m2191_runs": 0, "m2207_runs": 0,
                      "lm_runs": 0, "eda_runs": 0, "license_queries": 0,
                      "gpu_runs": 0, "pnr_runs": 0, "git_mutation": False,
                      "m2205_source_modified": False, "docs359_modified": False},
        "claim_boundary": {"source_hammer": True, "sampled_live_processes_only": True,
                           "exhaustive_short_lived_processes": False,
                           "library_conversion": False, "library_compatibility": False,
                           "design_library": False, "pnr": False, "timing": False,
                           "area": False, "power": False, "paper_ppa_ready": False},
    }
    output = HERE / "mechanical_checks.json"
    need(not output.exists(), "fresh mechanical output required")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
