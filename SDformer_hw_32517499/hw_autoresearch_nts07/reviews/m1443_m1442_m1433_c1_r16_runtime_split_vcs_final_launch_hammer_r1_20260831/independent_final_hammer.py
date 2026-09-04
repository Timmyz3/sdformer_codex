#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1443 final launch hammer for the M1433 C1 VCS runner.

This is a source-only hammer.  It never queries a license and never invokes
VCS, simv, or any other EDA program.  The only child is the exact-pinned
Python runtime-present regression suite.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Callable


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1433_m1337r15_m1162_c1_real_m935_runtime_witness_unit_delay_runtime_split_exact.py"
CHECKER = HW / "verif_m1433_c1_r16_vcs_runtime_split/check_m1433_c1_r16_vcs_runtime_split_source.py"
SOURCE_TESTS = HW / "verif_m1433_c1_r16_vcs_runtime_split/test_m1433_c1_r16_vcs_runtime_split_source.py"
RUNTIME_TESTS = HW / "verif_m1433_c1_r16_vcs_runtime_split/test_m1433_c1_r16_vcs_runtime_present.py"
SOURCE_CONTRACT = HW / "contracts/m1433_c1_r16_real_m935_runtime_witness_vcs_runtime_split_source_contract_r1_20260831.json"
SOURCE_AUTHOR = HW / "reviews/m1433_c1_r16_real_m935_runtime_witness_vcs_runtime_split_source_author_r1_20260831"
SOURCE_HAMMER = HW / "reviews/m1441_m1433_c1_r16_runtime_split_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1442_m1441_m1433_c1_r16_runtime_split_vcs_launch_release_r1_20260831.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
ATTEMPT = HW / "results/.m1433_c1_r16_runtime_split_vcs_attempt_consumed"
RESULT = HW / "results/m1433_c1_r16_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
QUARANTINE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")

EXPECTED = {
    "runner": "443ef3f2a2bc777095a5574da6b91aa2c97786505f86bff607fbc537adbae07a",
    "checker": "0e7976f11a01588c00f55af83c224e148296f5e5fe6d8c85371e64dc1dfff1d5",
    "source_tests": "9a2bd010aa5b0b97cd8923848940faac7dc1ce74caa9c7e1174bbf8257970e85",
    "runtime_tests": "b3b9d130749eb4a8a79148072350b76aeeb59520f85718e0663df62f40731ad4",
    "source_contract": "eacc909123b18f9e2314cdb01bf4d2c5a98865a9754329c75a15568ae91c0379",
    "source_author_review": "b2207075e229e3b3a92135d5e950c51373225e2fe78ce26915165dff17ebc8fd",
    "source_author_manifest": "bfebacc92719ab2c42338dd2bfe254f9e7ae076eeb5e6697f98091dd6de168ee",
    "source_author_outer": "fa842bb5b43740e663f3c998a51c54f2295afab51698eed53b2fac4a891cfa1e",
    "source_hammer_review": "d5f5672c13f3dd3f6ce8927871d3959a538a74d9ad9d021e6026186d931a8716",
    "source_hammer_manifest": "6f8e2ce105a51d595bf2caa77b02964b44f447a6cfe710a0d8c19ea99fe67f4a",
    "source_hammer_outer": "bfef3978b151c1fd898b31341fc90914c3200e2e06b93226d0251c2ff207a256",
    "release": "84bb5c0c6f1b808008c7fbc4adb637a183a759b348c9f08f2432aa5d8ac41f1a",
    "release_side": "5967bcc2af5ec8a1ace456d72d475c2432000bc3b5ac123ace7d94ad6b731265",
    "release_outer": "6ace9edfac76553e8939037e221ef55baf88b41eb826c5f2dc939451cf7e118a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
}
CLAIMS = {"source_only": True, "functional_vcs": False, "timing_verified": False,
          "cycles_measured": False, "speedup": False, "ppa": False,
          "power": False, "energy": False, "system_speedup": False,
          "headline": False}
AUTHORIZATION = {"vcs_compiles": 1, "simv_runs": 1,
                 "all_other_eda_runs": 0, "automatic_retry": False}
FINAL_STATUS = "PASS_M1443_AUTHORIZE_ONE_M1433_C1_R16_RUNTIME_SPLIT_VCS_LAUNCH"
checks = 0
attacks: list[dict[str, Any]] = []


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == digest,
            "identity drift: " + str(path))


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           HammerFailure("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root is not object")
    return value


def verify_sidecar(path: Path, payload_digest: str, side_digest: str,
                   outer_digest: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(side) + ".seal.sha256")
    regular(path, payload_digest); regular(side, side_digest); regular(outer, outer_digest)
    require(side.read_text(encoding="utf-8").split() == [payload_digest, path.name],
            "release sidecar content")
    require(outer.read_text(encoding="utf-8").split() == [side_digest, side.name],
            "release outer content")


def verify_tree(root: Path, review_digest: str, manifest_digest: str,
                outer_digest: str) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    manifest = root / "SHA256SUMS"; outer = root / "SHA256SUMS.seal.sha256"
    regular(root / "review.json", review_digest)
    regular(manifest, manifest_digest); regular(outer, outer_digest)
    require(outer.read_text(encoding="utf-8").split() == [manifest_digest, "SHA256SUMS"],
            "tree outer content")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and name not in listed and
                not rel.is_absolute() and ".." not in rel.parts, "unsafe manifest row")
        listed[name] = digest
    actual = set()
    for member in root.rglob("*"):
        rel = member.relative_to(root).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        require(not member.is_symlink(), "sealed symlink")
        if member.is_file(): actual.add(rel)
        else: require(member.is_dir(), "sealed special member")
    require(actual == set(listed), "sealed exact member-set drift")
    for name, digest in listed.items(): regular(root / name, digest)
    return strict_json(root / "review.json")


def validate_release(candidate: dict[str, Any], canonical: dict[str, Any]) -> None:
    require(candidate == canonical, "release exact-set/value drift")
    identity = candidate.get("identity", {})
    require(candidate.get("status") ==
            "AUTHORIZE_ONE_M1433_C1_R16_RUNTIME_SPLIT_UNIT_DELAY_VCS_ATTEMPT",
            "release status")
    require(candidate.get("launch_now") is False and candidate.get("inert_until_m1443") is True,
            "release inertness")
    require(identity.get("runner_sha256") == EXPECTED["runner"] and
            identity.get("source_checker_sha256") == EXPECTED["checker"] and
            identity.get("source_tests_sha256") == EXPECTED["source_tests"] and
            identity.get("runtime_tests_sha256") == EXPECTED["runtime_tests"] and
            identity.get("source_contract_sha256") == EXPECTED["source_contract"] and
            identity.get("source_hammer_review_sha256") == EXPECTED["source_hammer_review"] and
            identity.get("source_hammer_manifest_sha256") == EXPECTED["source_hammer_manifest"] and
            identity.get("source_hammer_outer_file_sha256") == EXPECTED["source_hammer_outer"],
            "release binding")
    require(candidate.get("authorization") == AUTHORIZATION, "release authorization")
    bounds = candidate.get("execution_bounds", {})
    require(bounds.get("attempt_consumed_before_license_or_tool") is True and
            bounds.get("same_uid_collision_gates_before_attempt") == 2 and
            bounds.get("failure_quarantine_recursive_manifest_and_outer_seal") is True and
            bounds.get("canonical_success_recursive_manifest_and_outer_seal") is True and
            bounds.get("automatic_retry") is False, "release execution bounds")
    split = candidate.get("runtime_split", {})
    require(split.get("source_tests_invoked_by_runner") is False and
            split.get("runtime_present_tests_invoked_by_runner") is True and
            split.get("runtime_present_tests_require_future_absent") is False,
            "release runtime split")
    require(candidate.get("claim_boundary") == CLAIMS, "release claim boundary")


def expected_final() -> dict[str, Any]:
    return {
        "schema": "m1443_m1442_m1433_c1_r16_runtime_split_vcs_final_launch_hammer_r1_v1",
        "status": FINAL_STATUS,
        "date": "2026-08-31",
        "bindings": {
            "runner_sha256": EXPECTED["runner"],
            "source_checker_sha256": EXPECTED["checker"],
            "source_tests_sha256": EXPECTED["source_tests"],
            "runtime_tests_sha256": EXPECTED["runtime_tests"],
            "source_contract_sha256": EXPECTED["source_contract"],
            "source_hammer_review_sha256": EXPECTED["source_hammer_review"],
            "source_hammer_manifest_sha256": EXPECTED["source_hammer_manifest"],
            "source_hammer_outer_file_sha256": EXPECTED["source_hammer_outer"],
            "launch_release_sha256": EXPECTED["release"],
            "launch_release_sidecar_sha256": EXPECTED["release_side"],
            "launch_release_outer_file_sha256": EXPECTED["release_outer"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "score": 100, "p0_count": 0, "p1_count": 0,
        "authorization": dict(AUTHORIZATION),
        "validation": {},
        "hammer_execution": {"license_queries": 0, "vcs": 0, "simv": 0,
                             "dc": 0, "pt": 0, "ptpx": 0, "eda": 0,
                             "attempt_consumed": False, "result_created": False},
        "claim_boundary": dict(CLAIMS),
        "verdict": "Exact-byte M1433/M1441/M1442 chain passes the different-author final source hammer. One future foundry-UNIT_DELAY VCS compile and one simv run are authorized with no retry; no launch occurred in this hammer.",
    }


def validate_final(candidate: dict[str, Any], canonical: dict[str, Any]) -> None:
    require(candidate == canonical, "final authority exact-set/value drift")
    require(candidate.get("status") == FINAL_STATUS and
            candidate.get("authorization") == AUTHORIZATION and
            candidate.get("bindings", {}).get("launch_release_sha256") == EXPECTED["release"] and
            candidate.get("claim_boundary") == CLAIMS,
            "final authority core fields")


def reject(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except BaseException as error:
        attacks.append({"attack": label, "rejected": True,
                        "exception": type(error).__name__ + ": " + str(error)})
        return
    raise HammerFailure("false negative: " + label)


def changed(value: Any) -> Any:
    if type(value) is bool: return not value
    if type(value) is int: return value + 1
    if type(value) is str: return "M1443_MUTATED"
    if type(value) is dict:
        result = dict(value); result["m1443_extra"] = True; return result
    raise TypeError(type(value))


def mutate_path(value: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    result = copy.deepcopy(value); node = result
    for key in path[:-1]: node = node[key]
    node[path[-1]] = changed(node[path[-1]])
    return result


def audit_runner() -> dict[str, Any]:
    text = RUNNER.read_text(encoding="utf-8"); tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    main = ast.unparse(functions["main"])
    require(text.count("COMPILE_COMMAND = [") == 1 and text.count("SIM_COMMAND = [") == 1,
            "command declarations")
    require(text.count("run_tool(COMPILE_COMMAND") == 1 and
            text.count("run_tool(SIM_COMMAND") == 1, "one compile/one simulation")
    require("run_python_gate(RUNTIME_TESTS, 'runtime_present')" in main and
            "run_python_gate(SOURCE_TESTS" not in main, "runtime-present suite reachability")
    resource = main.index("phase = 'RESOURCE_PREFLIGHT'")
    attempt = main.index("phase = 'ATTEMPT_CONSUME'")
    consume = main.index("publish_no_replace(ATTEMPT_STAGE, ATTEMPT)")
    license_at = main.index("SNPSLMD_LICENSE_FILE")
    compile_at = main.index("phase = 'COMPILE'")
    require(main.count("collision_gate()", resource, attempt) == 2 and
            main.index("namespace_gate()") < resource < attempt < consume < license_at < compile_at,
            "double collision/attempt-before-license-tool ordering")
    require("seal_dir(FAILURE_STAGE)" in main and
            "publish_no_replace(FAILURE_STAGE, QUARANTINE)" in main and
            "seal_dir(WORK)" in main and "publish_no_replace(WORK, RESULT)" in main,
            "recursive result/failure isolation")
    require("automatic_retry=True" not in text and "automatic_retry=true" not in text and
            "shutil.rmtree" not in text and "renameat2" in text,
            "no-retry/noreplace invariant")
    required_pins = (
        "M1433_EXPECTED_RUNNER_SHA256", "M1433_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
        "M1433_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256", "M1433_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
        "M1433_EXPECTED_LAUNCH_RELEASE_SHA256", "M1433_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
        "M1433_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256", "M1433_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256")
    require(all(main.count(name) >= 1 for name in required_pins), "external pins unreachable")
    require("PASS_M1443_AUTHORIZE_ONE_M1433_C1_R16_RUNTIME_SPLIT_VCS_LAUNCH" in main and
            "launch_release_sha256" in main and "authorization" in main,
            "final authority checks unreachable")
    return {"one_compile": True, "one_sim": True, "source_suite_unreachable": True,
            "runtime_present_suite_reachable": True, "collision_gates_before_attempt": 2,
            "attempt_before_license_and_tool": True, "recursive_failure_isolation": True,
            "recursive_success_publish": True, "atomic_noreplace": True,
            "automatic_retry": False, "external_exact_pins": len(required_pins)}


def namespace_audit() -> dict[str, Any]:
    targets = (ATTEMPT, RESULT, QUARANTINE)
    require(all(not os.path.lexists(str(path)) for path in targets),
            "canonical one-shot namespace already consumed")
    patterns = (".m1433_c1_r16_runtime_split_vcs_work.*",
                ".m1433_c1_r16_runtime_split_vcs_attempt_stage.*",
                ".m1433_c1_r16_runtime_split_vcs_failure_stage.*")
    residues = [str(path) for pattern in patterns for path in (HW / "results").glob(pattern)]
    require(not residues, "temporary namespace residue")
    return {"attempt_absent": True, "result_absent": True, "quarantine_absent": True,
            "temporary_prefix_residue_count": 0}


def run_runtime_suite() -> dict[str, Any]:
    completed = subprocess.run([str(PYTHON), "-I", str(RUNTIME_TESTS),
                                "--mode", "runtime_present"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, timeout=180,
                               check=False, env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                                                 "PATH": "/usr/bin:/bin",
                                                 "PYTHONDONTWRITEBYTECODE": "1"})
    require(completed.returncode == 0, "runtime suite failed: " + completed.stderr)
    value = json.loads(completed.stdout)
    require(value.get("status") == "PASS_M1433_RUNTIME_PRESENT_LAUNCH_TESTS" and
            value.get("checks", {}).get("regressions") ==
            {"attacks": 16, "rejected": 16, "false_negatives": 0} and
            all(value.get(name) == 0 for name in
                ("license_queries", "vcs_runs", "simv_runs", "eda_runs")),
            "runtime suite receipt drift")
    (HERE / "runtime_present_output.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def seal() -> None:
    rows = []
    for path in HERE.rglob("*"):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(HERE).as_posix(), sha(path)))
    rows.sort()
    manifest = HERE / "SHA256SUMS"
    manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in rows), encoding="utf-8")
    (HERE / "SHA256SUMS.seal.sha256").write_text(
        f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")


def main() -> int:
    verify_sidecar(RELEASE, EXPECTED["release"], EXPECTED["release_side"],
                   EXPECTED["release_outer"])
    for path, digest in ((RUNNER, EXPECTED["runner"]), (CHECKER, EXPECTED["checker"]),
                         (SOURCE_TESTS, EXPECTED["source_tests"]),
                         (RUNTIME_TESTS, EXPECTED["runtime_tests"]),
                         (SOURCE_CONTRACT, EXPECTED["source_contract"]),
                         (DOCS359, EXPECTED["docs359"]), (PYTHON, EXPECTED["python"])):
        regular(path, digest)
    author = verify_tree(SOURCE_AUTHOR, EXPECTED["source_author_review"],
                         EXPECTED["source_author_manifest"], EXPECTED["source_author_outer"])
    hammer = verify_tree(SOURCE_HAMMER, EXPECTED["source_hammer_review"],
                         EXPECTED["source_hammer_manifest"], EXPECTED["source_hammer_outer"])
    require(author.get("authorization", {}).get("release") is False and
            hammer.get("status") ==
            "PASS_M1441_M1433_C1_R16_RUNTIME_SPLIT_SOURCE__RELEASE_NOT_AUTHORED" and
            hammer.get("score") == 100 and hammer.get("p0_count") == 0 and
            hammer.get("p1_count") == 0, "predecessor authority state")
    canonical_release = strict_json(RELEASE); validate_release(canonical_release, canonical_release)
    runner_findings = audit_runner(); namespaces = namespace_audit()

    final = expected_final()
    release_paths = [
        ("status",), ("launch_now",), ("inert_until_m1443",),
        ("identity", "runner_sha256"), ("identity", "source_checker_sha256"),
        ("identity", "source_tests_sha256"), ("identity", "runtime_tests_sha256"),
        ("identity", "source_contract_sha256"),
        ("identity", "source_hammer_review_sha256"),
        ("identity", "source_hammer_manifest_sha256"),
        ("identity", "source_hammer_outer_file_sha256"),
        ("authorization", "vcs_compiles"), ("authorization", "simv_runs"),
        ("authorization", "all_other_eda_runs"), ("authorization", "automatic_retry"),
        ("execution_bounds", "attempt_consumed_before_license_or_tool"),
        ("execution_bounds", "same_uid_collision_gates_before_attempt"),
        ("execution_bounds", "failure_quarantine_recursive_manifest_and_outer_seal"),
        ("execution_bounds", "canonical_success_recursive_manifest_and_outer_seal"),
        ("runtime_split", "source_tests_invoked_by_runner"),
        ("runtime_split", "runtime_present_tests_invoked_by_runner"),
        ("runtime_split", "runtime_present_tests_require_future_absent"),
        ("one_shot_namespaces", "attempt"), ("one_shot_namespaces", "result"),
        ("final_hammer_gate", "required_status"), ("claim_boundary", "headline"),
    ]
    for path in release_paths:
        reject("release_" + "__".join(path),
               lambda path=path: validate_release(mutate_path(canonical_release, path),
                                                  canonical_release))
    validate_final(final, final)
    final_paths = [("status",), ("bindings", "runner_sha256"),
                   ("bindings", "launch_release_sha256"),
                   ("authorization", "vcs_compiles"), ("authorization", "simv_runs"),
                   ("authorization", "all_other_eda_runs"),
                   ("authorization", "automatic_retry"), ("claim_boundary", "headline")]
    for path in final_paths:
        reject("final_" + "__".join(path),
               lambda path=path: validate_final(mutate_path(final, path), final))
    expected_pins = {
        "M1433_EXPECTED_RUNNER_SHA256": EXPECTED["runner"],
        "M1433_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256": EXPECTED["source_hammer_review"],
        "M1433_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256": EXPECTED["source_hammer_manifest"],
        "M1433_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256": EXPECTED["source_hammer_outer"],
        "M1433_EXPECTED_LAUNCH_RELEASE_SHA256": EXPECTED["release"],
        "M1433_EXPECTED_FINAL_HAMMER_REVIEW_SHA256": "0" * 64,
        "M1433_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256": "1" * 64,
        "M1433_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256": "2" * 64,
    }
    def pin_gate(candidate):
        require(candidate == expected_pins and all(re.fullmatch(r"[0-9a-f]{64}", value)
                for value in candidate.values()), "external pin drift")
    pin_gate(expected_pins)
    for name in expected_pins:
        reject("external_pin_" + name,
               lambda name=name: pin_gate({**expected_pins, name: "f" * 63}))

    runtime = run_runtime_suite()
    false_negatives = sum(not item["rejected"] for item in attacks)
    require(false_negatives == 0 and len(attacks) == len(release_paths) + len(final_paths) + 8,
            "adversarial accounting")
    validation = {
        "exact_live_chain": True,
        "runner": runner_findings,
        "namespaces": namespaces,
        "runtime_present_suite_status": runtime["status"],
        "runtime_present_regressions": runtime["checks"]["regressions"],
        "adversarial_mutations": len(attacks),
        "adversarial_false_negatives": false_negatives,
        "release_status_exact": True,
        "release_external_self_sha_placeholder_is_inert_and_bound_by_m1443": True,
        "source_hammer_recursive_seal_exact": True,
        "final_review_exact_binds_release": True,
        "fresh_attempt_and_result_namespaces": True,
    }
    final["validation"] = validation
    validate_final(final, final)
    (HERE / "adversarial_mutations.json").write_text(
        json.dumps({"attacks": attacks, "false_negatives": false_negatives},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "mechanical_checks.json").write_text(
        json.dumps({"checks": checks, "validation": validation,
                    "license_queries": 0, "vcs_runs": 0, "simv_runs": 0,
                    "eda_runs": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "review.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    (HERE / "review.md").write_text(
        "# M1443 independent final launch hammer\n\n"
        "**PASS 100/100.** Exact M1433/M1441/M1442 authorities, the runtime-present "
        "suite, one-compile/one-simulation bounds, attempt-before-license/tool ordering, "
        "two pre-attempt same-UID collision gates, recursive quarantine, and fresh namespaces "
        "all pass. All adversarial mutations fail closed. This authorizes exactly one future "
        "foundry-UNIT_DELAY VCS attempt; this hammer launched no EDA tool.\n",
        encoding="utf-8")
    (HERE / "NO_LICENSE_NO_VCS_NO_SIMV_NO_EDA.txt").write_text(
        "M1443_SOURCE_ONLY_HAMMER__NO_LICENSE_QUERY_NO_VCS_NO_SIMV_NO_EDA\n",
        encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1443_AUTHORIZE_ONE_M1433_C1_R16_RUNTIME_SPLIT_VCS_LAUNCH\n",
        encoding="utf-8")
    seal()
    print(json.dumps({"status": FINAL_STATUS, "score": 100,
                      "checks": checks, "attacks": len(attacks),
                      "false_negatives": false_negatives,
                      "review_sha256": sha(HERE / "review.json"),
                      "manifest_sha256": sha(HERE / "SHA256SUMS"),
                      "outer_file_sha256": sha(HERE / "SHA256SUMS.seal.sha256")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
