#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent source-only blind hammer for M1433; never invokes EDA."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1433_m1337r15_m1162_c1_real_m935_runtime_witness_unit_delay_runtime_split_exact.py"
CHECKER = HW / "verif_m1433_c1_r16_vcs_runtime_split/check_m1433_c1_r16_vcs_runtime_split_source.py"
SOURCE_TESTS = HW / "verif_m1433_c1_r16_vcs_runtime_split/test_m1433_c1_r16_vcs_runtime_split_source.py"
RUNTIME_TESTS = HW / "verif_m1433_c1_r16_vcs_runtime_split/test_m1433_c1_r16_vcs_runtime_present.py"
CONTRACT = HW / "contracts/m1433_c1_r16_real_m935_runtime_witness_vcs_runtime_split_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1433_c1_r16_real_m935_runtime_witness_vcs_runtime_split_source_author_r1_20260831"
DESTINATION = HW / "reviews/m1441_m1433_c1_r16_runtime_split_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1442_m1441_m1433_c1_r16_runtime_split_vcs_launch_release_r1_20260831.json"
FINAL = HW / "reviews/m1443_m1442_m1433_c1_r16_runtime_split_vcs_final_launch_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
CLAIMS = {"source_only": True, "functional_vcs": False, "timing_verified": False,
          "cycles_measured": False, "speedup": False, "ppa": False,
          "power": False, "energy": False, "system_speedup": False,
          "headline": False}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1441_m1433_checker", CHECKER)
R = load("m1441_m1433_runtime", RUNTIME_TESTS)


def verify_file_sidecar(path: Path) -> dict[str, str]:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    for item in (path, sidecar, outer):
        mode = item.lstat().st_mode
        if not stat.S_ISREG(mode) or item.is_symlink():
            raise AssertionError("sidecar member not regular")
    if sidecar.read_text().split() != [sha(path), path.name]:
        raise AssertionError("inner sidecar mismatch")
    if outer.read_text().split() != [sha(sidecar), sidecar.name]:
        raise AssertionError("outer sidecar mismatch")
    return {"payload_sha256": sha(path), "sidecar_sha256": sha(sidecar),
            "outer_file_sha256": sha(outer)}


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1441_MUTATED"
    if type(value) is dict:
        result = dict(value); result["m1441_extra"] = True; return result
    raise TypeError(type(value))


def attacks():
    base = M.expected_contract()
    cases = [
        ("contract_extra_top_level", lambda d: d.__setitem__("m1441_extra", True)),
        ("contract_date_changed", lambda d: d.__setitem__("date", "2099-01-01")),
        ("contract_future_execution_removed", lambda d: d.pop("future_execution")),
        ("contract_future_execution_extra", lambda d:
         d["future_execution"].__setitem__("m1441_extra", True)),
    ]
    cases.extend(("future_execution_" + key, lambda d, key=key:
                  d["future_execution"].__setitem__(key, changed(d["future_execution"][key])))
                 for key in base["future_execution"])
    cases.extend([
        ("author_execution_extra", lambda d:
         d["author_execution"].__setitem__("m1441_extra", False)),
        ("claim_boundary_extra", lambda d:
         d["claim_boundary"].__setitem__("m1441_extra", False)),
    ])
    if len(cases) != 16:
        raise AssertionError("attack count")
    return cases


def split_states() -> dict[str, bool]:
    source = M.validate_future("source_absent")
    saved = (M.FUTURE_HAMMER, M.FUTURE_RELEASE, M.FUTURE_FINAL,
             M.ATTEMPT, M.RESULT, M.QUARANTINE)
    with tempfile.TemporaryDirectory(prefix="m1441_split_") as temporary:
        root = Path(temporary)
        hammer, release, final = root / "m1441", root / "m1442.json", root / "m1443"
        hammer.mkdir(); release.write_text("{}\n"); final.mkdir()
        M.FUTURE_HAMMER, M.FUTURE_RELEASE, M.FUTURE_FINAL = hammer, release, final
        M.ATTEMPT, M.RESULT, M.QUARANTINE = root / "attempt", root / "result", root / "quarantine"
        runtime = M.validate_future("runtime_present")
        absent_rejected = False
        try:
            M.validate_future("source_absent")
        except AssertionError:
            absent_rejected = True
    (M.FUTURE_HAMMER, M.FUTURE_RELEASE, M.FUTURE_FINAL,
     M.ATTEMPT, M.RESULT, M.QUARANTINE) = saved
    return {"canonical_source_absent": source["future_absent"],
            "synthetic_runtime_present": runtime["future_present"],
            "source_absent_rejects_runtime_present": absent_rejected}


def seal(root: Path) -> tuple[str, str, str]:
    rows = []
    for path in root.rglob("*"):
        if path.name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise AssertionError("unsealable member")
        rows.append((path.relative_to(root).as_posix(), sha(path)))
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(rows)))
    outer = root / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        if name in listed or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise AssertionError("manifest row")
        listed[name] = digest
        if sha(root / name) != digest:
            raise AssertionError("manifest drift")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if actual != set(listed) or outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise AssertionError("recursive seal drift")
    return sha(root / "review.json"), sha(manifest), sha(outer)


def main() -> int:
    if any(os.path.lexists(path) for path in (DESTINATION, RELEASE, FINAL)):
        raise RuntimeError("future namespace not fresh")
    common = M.validate_common(skip_author=False)
    contract_seal = verify_file_sidecar(CONTRACT)
    author_review = M.verify_dir(AUTHOR)
    author_seals = {"review_sha256": sha(AUTHOR / "review.json"),
                    "manifest_sha256": sha(AUTHOR / "SHA256SUMS"),
                    "outer_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256")}
    if M.strict_json(CONTRACT) != M.expected_contract():
        raise AssertionError("contract exact-set drift")
    expected_author_bindings = {
        "runner_sha256": sha(RUNNER), "source_checker_sha256": sha(CHECKER),
        "source_tests_sha256": sha(SOURCE_TESTS),
        "runtime_tests_sha256": sha(RUNTIME_TESTS),
        "source_contract_sha256": sha(CONTRACT),
        "m1364_review_sha256": M.SEALS[M.M1364_FAIL][0]}
    if author_review.get("bindings") != expected_author_bindings or \
            author_review.get("claim_boundary") != CLAIMS:
        raise AssertionError("author binding/seal drift")

    completed = subprocess.run([str(PYTHON), "-I", str(SOURCE_TESTS)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=120, check=False)
    source_output = completed.stdout
    if completed.returncode or "Ran 23 tests" not in source_output or \
            re.search(r"^OK$", source_output, re.MULTILINE) is None:
        raise AssertionError("23 tests failed")

    outcomes = []
    for name, mutate in attacks():
        candidate = copy.deepcopy(M.expected_contract()); mutate(candidate)
        rejected = False
        try:
            M.check_contract_dict(candidate)
        except AssertionError:
            rejected = True
        outcomes.append({"attack": name, "rejected": rejected})
    if not all(row["rejected"] for row in outcomes) or R.validate_contract_regressions() != {
            "attacks": 16, "rejected": 16, "false_negatives": 0}:
        raise AssertionError("historical false negative")

    runner_text = RUNNER.read_text(); source_text = SOURCE_TESTS.read_text()
    runtime_text = RUNTIME_TESTS.read_text(); main_text = runner_text[runner_text.index("def main()") :]
    future_paths = {"m1441": M.FUTURE_HAMMER.relative_to(HW).as_posix(),
                    "m1442": M.FUTURE_RELEASE.relative_to(HW).as_posix(),
                    "m1443": M.FUTURE_FINAL.relative_to(HW).as_posix()}
    separation = {
        "source_suite_requires_absent": 'validate_future("source_absent")' in source_text,
        "source_suite_not_invoked_by_runner": "run_python_gate(SOURCE_TESTS" not in main_text,
        "runtime_suite_requires_present": 'validate_future("runtime_present")' in runtime_text,
        "runtime_suite_never_requires_absent": 'validate_future("source_absent")' not in runtime_text,
        "runtime_suite_invoked_once_by_runner": main_text.count(
            'run_python_gate(RUNTIME_TESTS, "runtime_present")') == 1,
        "checker_runtime_present_invoked_once": main_text.count(
            'run_python_gate(SOURCE_CHECKER, "runtime_present")') == 1,
        "future_paths_exact": future_paths == {
            "m1441": "reviews/m1441_m1433_c1_r16_runtime_split_source_blind_hammer_r1_20260831",
            "m1442": "contracts/m1442_m1441_m1433_c1_r16_runtime_split_vcs_launch_release_r1_20260831.json",
            "m1443": "reviews/m1443_m1442_m1433_c1_r16_runtime_split_vcs_final_launch_hammer_r1_20260831"},
        **split_states(),
    }
    static_runner = M.audit_runner(runner_text)
    if not all(separation.values()) or static_runner != {
            "one_compile": True, "one_sim": True, "attempt_before_tool": True,
            "collision_gates": 2, "failure_quarantine_recursive_seal": True,
            "runtime_suite_only": True}:
        raise AssertionError("runtime split/static protocol drift")
    if sha(DOCS359) != "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4":
        raise AssertionError("docs359 drift")

    checks = {"common": common, "contract_sidecar": contract_seal,
              "author_seals": author_seals, "source_tests_run": 23,
              "source_tests_passed": 23, "historical_attacks": outcomes,
              "attacks_rejected": 16, "false_negatives": 0,
              "separation": separation, "static_runner": static_runner,
              "future_paths": future_paths}
    output = {"schema": "m1441_m1433_c1_r16_runtime_split_source_blind_hammer_output_r1_v1",
              "status": "PASS", "p0": [], "checks": checks,
              "authorization": {"launch_release": False, "license_queries": 0,
                                "vcs_compiles": 0, "simv_runs": 0,
                                "all_other_eda_runs": 0, "automatic_retry": False}}
    review = {
        "schema": "m1441_m1433_c1_r16_runtime_split_source_blind_hammer_r1_v1",
        "status": "PASS_M1441_M1433_C1_R16_RUNTIME_SPLIT_SOURCE__RELEASE_NOT_AUTHORED",
        "score": 100, "date": "2026-08-31", "p0_count": 0, "p1_count": 0,
        "verdict": "M1433 closes the M1364 source-absent/runtime-present deadlock with exact-pinned disjoint suites. Independent replay passes 23/23 and rejects all 16 historical exploit families. The runner remains one compile, one simulation, attempt-before-tool, double-collision-gated, recursively quarantined, and no-retry. This hammer does not authorize launch.",
        "bindings": {"runner_sha256": sha(RUNNER),
                     "source_checker_sha256": sha(CHECKER),
                     "source_tests_sha256": sha(SOURCE_TESTS),
                     "runtime_tests_sha256": sha(RUNTIME_TESTS),
                     "source_contract_sha256": sha(CONTRACT),
                     "author_review_sha256": author_seals["review_sha256"],
                     "author_manifest_sha256": author_seals["manifest_sha256"],
                     "author_outer_file_sha256": author_seals["outer_file_sha256"]},
        "validation": checks,
        "authorization": {"launch_release": False, "license_query": False,
                          "vcs": False, "simv": False, "dc": False,
                          "pt": False, "ptpx": False, "eda": False,
                          "gpu": False, "remote": False, "automatic_retry": False},
        "claim_boundary": CLAIMS, "docs359_sha256": sha(DOCS359)}

    DESTINATION.mkdir()
    shutil.copy2(__file__, DESTINATION / "independent_hammer.py")
    (DESTINATION / "hammer_output.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    (DESTINATION / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    (DESTINATION / "review.md").write_text(
        "# M1441 independent blind hammer\n\n**PASS 100/100.** M1433 separates the 23-test "
        "future-absent author gate from the exact-pinned runtime-present launch gate. "
        "All 16 historical exploit families fail closed. No release or EDA run is authorized.\n")
    (DESTINATION / "source_test_output.txt").write_text(source_output)
    (DESTINATION / "mechanical_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    (DESTINATION / "NO_LICENSE_NO_VCS_NO_SIMV_NO_EDA.txt").write_text(
        "Source-only Python checks. No license query, VCS, simv, DC, PT, PTPX, GPU, or remote command.\n")
    (DESTINATION / "RUN_COMPLETE.txt").write_text("PASS_M1441_M1433_C1_R16_RUNTIME_SPLIT_SOURCE\n")
    pins = seal(DESTINATION)
    print(json.dumps({"status": review["status"], "score": 100,
                      "review_sha256": pins[0], "manifest_sha256": pins[1],
                      "outer_file_sha256": pins[2]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
