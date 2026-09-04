#!/usr/bin/python3.12
"""Independent, no-EDA M2169 hammer for committed M2168 source."""
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

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HW = REPO / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_m2168_m2167_icc2_library_import_preflight_one_shot.sh"
CONTRACT = HW / "contracts/m2168_m2167_icc2_library_import_preflight_source_contract_r1_20260904.json"
AUTHOR = HW / "reviews/m2168_m2167_icc2_library_import_preflight_source_author_receipt_r1_20260904"
M2167 = HW / "reviews/m2167_m2166_m2164_icc2_library_preflight_startup_failure_hammer_r1_20260904"
SELFCHECK = AUTHOR / "selfcheck.py"
TESTS = AUTHOR / "tests.py"
TCL = HW / "dc_handoff/scripts/run_icc2_m2153_library_import_preflight.tcl"
MONITOR = HW / "dc_handoff/scripts/monitor_m2153_icc2_process_tree.py"
INVENTORY = HW / "dc_handoff/scripts/inventory_m2153_repo_root.py"
CHECKER = HW / "system_simulator/scripts/check_m2164_icc2_library_import_preflight.py"
MASTER_LIST = HW / "dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "dc_handoff/runs/m2170_m2168_icc2_library_import_preflight_raw_r1_20260904"
ATTEMPT = HW / "dc_handoff/runs/.m2170_m2168_icc2_library_import_preflight_attempt_consumed"
LOCK = HW / "dc_handoff/runs/.m2170_m2168_icc2_library_import_preflight_launch_lock"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def verify_seal(root: Path) -> int:
    need(root.is_dir() and not root.is_symlink(), f"sealed root: {root}")
    need(not any(path.is_symlink() for path in root.rglob("*")),
         f"symlink in sealed root: {root}")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and outer.is_file(), f"seal files: {root}")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         f"outer seal mismatch: {root}")
    members: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, f"manifest syntax: {line}")
        digest, rel = fields
        rel = rel.lstrip("*")
        path = root / rel
        need(path.is_file() and not path.is_symlink(), f"manifest member: {rel}")
        need(sha(path) == digest, f"manifest digest: {rel}")
        need(rel not in members, f"duplicate manifest member: {rel}")
        members[rel] = digest
    actual = sorted(str(path.relative_to(root)) for path in root.rglob("*")
                    if path.is_file() and path.name not in
                    {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(sorted(members) == actual, f"nonexhaustive sealed root: {root}")
    return len(members)


def load_selfcheck():
    spec = importlib.util.spec_from_file_location("m2168_author_selfcheck", SELFCHECK)
    need(spec is not None and spec.loader is not None, "selfcheck import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_reject(fn, label: str) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"independent mutation survived: {label}")


def layout_attack_census(selfcheck) -> int:
    attacks = 0
    with tempfile.TemporaryDirectory(prefix="m2169_layout.") as raw:
        root = Path(raw)
        isolated = root / "isolated_cwd"
        listed = [isolated / "home", isolated / "tmp",
                  isolated / "cache/xdg", isolated / "cache/library",
                  isolated / "frame_output", isolated / "frame_logs",
                  isolated / "reports"]
        for path in listed:
            path.mkdir(parents=True, exist_ok=True)
        design = isolated / "m2153_disposable_design.nlib"
        frame = isolated / "frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm"
        selfcheck.validate_isolated_layout_for_test(isolated, listed, design, frame)

    def endpoint_symlink() -> None:
        with tempfile.TemporaryDirectory(prefix="m2169_endpoint.") as raw:
            root = Path(raw); isolated = root / "isolated_cwd"
            listed = [isolated / "home", isolated / "tmp",
                      isolated / "cache/xdg", isolated / "cache/library",
                      isolated / "frame_output", isolated / "frame_logs",
                      isolated / "reports"]
            for path in listed: path.mkdir(parents=True, exist_ok=True)
            outside = root / "outside"; outside.mkdir()
            listed[3].rmdir(); os.symlink(outside, listed[3])
            selfcheck.validate_isolated_layout_for_test(
                isolated, listed, isolated / "m2153_disposable_design.nlib",
                isolated / "frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm")

    def intermediate_symlink() -> None:
        with tempfile.TemporaryDirectory(prefix="m2169_intermediate.") as raw:
            root = Path(raw); isolated = root / "isolated_cwd"; isolated.mkdir()
            outside = root / "outside"; (outside / "xdg").mkdir(parents=True)
            (outside / "library").mkdir(); os.symlink(outside, isolated / "cache")
            for name in ("home", "tmp", "frame_output", "frame_logs", "reports"):
                (isolated / name).mkdir()
            listed = [isolated / "home", isolated / "tmp",
                      isolated / "cache/xdg", isolated / "cache/library",
                      isolated / "frame_output", isolated / "frame_logs",
                      isolated / "reports"]
            selfcheck.validate_isolated_layout_for_test(
                isolated, listed, isolated / "m2153_disposable_design.nlib",
                isolated / "frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm")

    for label, function in (("endpoint_symlink", endpoint_symlink),
                            ("intermediate_symlink", intermediate_symlink)):
        expect_reject(function, label); attacks += 1
    return attacks


def main() -> int:
    need(sha(DOCS359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "protected docs359 identity")
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "M2170 result/attempt/lock already exists")
    need(not list((HW / "dc_handoff/runs").glob("*m2170_m2168*")),
         "M2170 work/result residue exists")
    author_members = verify_seal(AUTHOR)
    predecessor_members = verify_seal(M2167)

    contract_sidecar = Path(str(CONTRACT) + ".sha256")
    contract_outer = Path(str(contract_sidecar) + ".seal.sha256")
    need(contract_sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(contract_outer.read_text().split() == [sha(contract_sidecar),
                                                 contract_sidecar.name],
         "contract outer seal")
    contract = json.loads(CONTRACT.read_text())
    need(contract["status"] == "SOURCE_ONLY_PENDING_M2169_INDEPENDENT_HAMMER",
         "source-only contract status")
    need(contract["predecessor"]["review_sha256"] == sha(M2167 / "review.json"),
         "M2167 contract identity")
    predecessor = json.loads((M2167 / "review.json").read_text())
    need(predecessor["status"] ==
         "PASS_M2167_M2166_STARTUP_FAILURE_DIAGNOSIS__M2166_PERMANENTLY_NONCITABLE__M2168_SOURCE_ONLY",
         "M2167 status")
    need(predecessor["authorization"]["m2170_license_queries"] == 0 and
         predecessor["authorization"]["m2170_top_level_icc2_shell_runs"] == 0,
         "M2167 direct execution authority")

    receipt = json.loads((AUTHOR / "source_receipt.json").read_text())
    identity = receipt["source_identity"]
    expected_identity = {
        "contract_sha256": sha(CONTRACT),
        "contract_sha256_file_sha256": sha(contract_sidecar),
        "runner_sha256": sha(RUNNER),
        "reused_checker_sha256": sha(CHECKER),
        "reused_tcl_sha256": sha(TCL),
        "reused_monitor_sha256": sha(MONITOR),
        "reused_inventory_sha256": sha(INVENTORY),
        "selfcheck_py_sha256": sha(SELFCHECK),
        "selfcheck_txt_sha256": sha(AUTHOR / "selfcheck.txt"),
        "tests_py_sha256": sha(TESTS),
        "tests_txt_sha256": sha(AUTHOR / "tests.txt"),
        "m2167_review_sha256": sha(M2167 / "review.json"),
        "docs359_sha256": sha(DOCS359),
    }
    need(identity == expected_identity, "author receipt identity map")

    selfcheck_run = subprocess.run([sys.executable, "-B", str(SELFCHECK)],
                                   capture_output=True, text=True, check=False)
    need(selfcheck_run.returncode == 0 and
         "PASS_M2168_AUTHOR_SOURCE_SELFCHECK" in selfcheck_run.stdout and
         "eda_runs=0" in selfcheck_run.stdout and
         "license_queries=0" in selfcheck_run.stdout,
         f"author selfcheck failed: {selfcheck_run.stdout} {selfcheck_run.stderr}")
    tests_run = subprocess.run([sys.executable, "-B", str(TESTS)],
                               capture_output=True, text=True, check=False)
    need(tests_run.returncode == 0 and
         "PASS_M2168_MUTATION_TESTS tests=14 eda_runs=0 license_queries=0" in
         tests_run.stdout, f"author mutation suite failed: {tests_run.stdout} {tests_run.stderr}")

    source = RUNNER.read_text()
    selfcheck = load_selfcheck()
    selfcheck.validate_runner_source(source)
    subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], check=True)
    independent_mutations = 0
    license_site = '"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f ICCompilerII >"${WORK}/license_preflight.log" 2>&1'
    # Move rather than duplicate, proving the order check is causal.
    moved_license = source.replace(license_site + "\n", "", 1).replace(
        "printf 'M2168_OUTPUT_ABSENCE_GATE_PASS", license_site +
        "\nprintf 'M2168_OUTPUT_ABSENCE_GATE_PASS", 1)
    expect_reject(lambda: selfcheck.validate_runner_source(moved_license),
                  "moved_license_before_contract"); independent_mutations += 1
    release = ': >"${LAUNCH_GATE}"'
    moved_release = source.replace(release + "\n", "", 1).replace(
        '[[ -e "${MONITOR_READY}" ]] || exit 5', release +
        '\n[[ -e "${MONITOR_READY}" ]] || exit 5', 1)
    expect_reject(lambda: selfcheck.validate_runner_source(moved_release),
                  "moved_release_before_ready"); independent_mutations += 1
    expect_reject(lambda: selfcheck.validate_runner_source(
        source.replace("mkdir -p -- \\\n", "mkdir -- \\\n", 1)),
        "plain_mkdir"); independent_mutations += 1
    independent_mutations += layout_attack_census(selfcheck)

    need(source.count("mkdir -p -- \\\n") == 1, "single mkdir-p site")
    creation = source[source.index("mkdir -p -- \\\n"):
                      source.index("/usr/libexec/platform-python3.6 -I - \"${ISOLATED}\"")]
    exact_dirs = ["home", "tmp", "cache/xdg", "cache/library", "frame_output",
                  "frame_logs", "reports"]
    need(all(creation.count(f'"${{ISOLATED}}/{name}"') == 1 for name in exact_dirs),
         "seven exact creation paths")
    need(source.count('"${LMUTIL}" lmstat') == 1, "exact one LMUTIL site")
    need(source.count('"${ICC2}" -no_init -f "${TCL}"') == 1,
         "exact one ICC2 site")
    need("env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C" in source,
         "isolated env invocation")
    need(source.count('[[ ! -e "${DESIGN_LIB}" && ! -L "${DESIGN_LIB}" ]]') == 2,
         "design output absence before execution")
    need(source.count('[[ ! -e "${FRAME_NDM}" && ! -L "${FRAME_NDM}" ]]') == 2,
         "frame output absence before execution")
    anchors = [source.index("M2168_LAYOUT_GATE_PASS"),
               source.index("M2168_OUTPUT_ABSENCE_GATE_PASS"),
               source.index("M2168_EXECUTION_CONTRACT_WRITE_PASS"),
               source.index("M2168_EXECUTION_CONTRACT_REREAD_PASS"),
               source.index('"${LMUTIL}" lmstat'),
               source.index('while [[ ! -e "${LAUNCH_GATE}" ]]'),
               source.index('"${MONITOR}" --root-pid'),
               source.index('[[ -e "${MONITOR_READY}" ]] || exit 5'),
               source.index(': >"${LAUNCH_GATE}"')]
    need(anchors == sorted(anchors), "gated execution order")

    # Immutable library-only Tcl contains no design/RTL/P&R/timing/power stage.
    tcl = TCL.read_text()
    forbidden = [r"\bread_verilog\b", r"\bread_ddc\b", r"\blink_block\b",
                 r"\bcompile_fusion\b", r"\binitialize_floorplan\b",
                 r"\bplace_opt\b", r"\bclock_opt\b", r"\broute_opt\b",
                 r"\broute_auto\b", r"\bextract_rc\b", r"\breport_timing\b",
                 r"\breport_power\b"]
    need(not any(re.search(pattern, tcl) for pattern in forbidden),
         "P&R/design operation in immutable Tcl")
    need(tcl.count("generate_frame_from_mw") == 2,
         "one command plus one diagnostic mention expected")
    need(tcl.count("create_lib -ref_libs") == 1, "single disposable create_lib")
    need(contract["exact_runtime_budget_after_m2169_pass"] == {
        "license_queries": 1, "top_level_icc2_shell_runs": 1,
        "pnr_runs": 0, "automatic_retry": False,
        "tool_spawned_children": "observed and counted, not additional top-level launches"},
        "runtime budget")
    need(contract["author_authorization"] == {
        "m2169_independent_hammer": True, "m2170": False,
        "license_queries": 0, "top_level_icc2_shell_runs": 0,
        "pnr_runs": 0, "automatic_retry": False}, "author authority")

    payload = {
        "schema": "m2169_m2168_m2167_icc2_library_import_preflight_source_mechanical_checks_r1_v1",
        "status": "PASS_M2169_MECHANICAL_CHECKS__NO_EDA",
        "author_members": author_members,
        "predecessor_members": predecessor_members,
        "author_mutation_tests_rerun": 14,
        "independent_additional_mutations": independent_mutations,
        "single_mkdir_p_site": 1,
        "exact_isolation_leaf_paths": 7,
        "license_sites": 1,
        "top_level_icc2_sites": 1,
        "pnr_sites": 0,
        "m2170_attempt_result_lock_absent": True,
        "docs359_sha256": sha(DOCS359),
        "execution": {"license_queries": 0, "icc2_runs": 0,
                      "eda_runs": 0, "gpu_runs": 0},
    }
    (HERE / "mechanical_checks.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PASS_M2169_INDEPENDENT_SOURCE_HAMMER_CHECKS author_tests=14 "
          f"independent_mutations={independent_mutations} eda_runs=0 license_queries=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
