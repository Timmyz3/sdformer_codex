#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author M1466 final launch authority for the inert M1459 runner.

This hammer is source-only.  It may execute exact-pinned Python regressions,
but it never queries a license and never invokes VCS, simv, or any EDA tool.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable
import unittest


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1459_m1433_c1_runtime_split_generic_seal_successor.py"
CHECKER = HW / "verif_m1459_c1_generic_seal_successor/check_m1459_c1_generic_seal_successor_source.py"
SOURCE_TESTS = HW / "verif_m1459_c1_generic_seal_successor/test_m1459_c1_generic_seal_successor_source.py"
RUNTIME_TESTS = HW / "verif_m1433_c1_r16_vcs_runtime_split/test_m1433_c1_r16_vcs_runtime_present.py"
SOURCE_CONTRACT = HW / "contracts/m1459_m1433_c1_generic_seal_successor_source_contract_r1_20260831.json"
SOURCE_AUTHOR = HW / "reviews/m1459_m1433_c1_generic_seal_successor_source_author_r1_20260831"
SOURCE_HAMMER = HW / "reviews/m1464_m1459_c1_generic_seal_successor_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1465_m1464_m1459_c1_generic_seal_successor_vcs_launch_release_r1_20260831.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
ATTEMPT = HW / "results/.m1459_c1_generic_seal_vcs_attempt_consumed"
RESULT = HW / "results/m1459_c1_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
QUARANTINE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")

EXPECTED = {
    "runner": "3c0028c41fbbd8f6d1ede4b284aece877dd926a2b82a67de26d71f5322a9e891",
    "checker": "efdf56d8b22ef6205c9f7059648bbb62c6c0cbc81606571b3473864cb613bbd9",
    "source_tests": "d47c951e3e8dc75be733438e2504fa6b920d893704df890cd6c2761f553bdbb4",
    "runtime_tests": "b3b9d130749eb4a8a79148072350b76aeeb59520f85718e0663df62f40731ad4",
    "source_contract": "cd4e2d6075a644f365f1c6c7b097afbae0e287101e563d92c9b52241c60fb910",
    "source_author_review": "af6c40d7ffdfe5299501d50b39f693f7a4f4e818402f07f7b3a0cc4617563758",
    "source_author_manifest": "2eb4a6da4cf67a5473f6cc1bd84ec19ca8d0574ad06453ae40ff0ab24e312106",
    "source_author_outer": "2f7150cec7421a2a226cbb7118d35357eb117535d6b5fbd4c8e2f9302fb46721",
    "source_hammer_review": "20158101501768f10ad450b73b7a37b60b3cf9284c7aea13592a58e04afab3fe",
    "source_hammer_manifest": "63df66fbd14c42bce268fc20cf0871051e35030eeae2f36f60de72e93e153767",
    "source_hammer_outer": "3069be217e81888a99dd535c3db340902ce5a0a45480824b27e436bb1d73fb8b",
    "release": "77810abab9730fcf1bf23e5cb827fe8b4b9e44119b696f0acfa21dbce90b91c3",
    "release_side": "6e4d8cd8da5550799a02da1f8154d90c3c5e2c3912ed48da6b865ff5d17b7aad",
    "release_outer": "1f2c6348affe5804fca55e6b1fc7a56d33dc37f6d4f340dbab2dea4a1a34d470",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
}
CLAIMS = {"source_only": True, "functional_vcs": False,
          "timing_verified": False, "cycles_measured": False,
          "speedup": False, "ppa": False, "power": False,
          "energy": False, "system_speedup": False, "headline": False}
AUTHORIZATION = {"vcs_compiles": 1, "simv_runs": 1,
                 "all_other_eda_runs": 0, "automatic_retry": False}
FINAL_STATUS = "PASS_M1466_AUTHORIZE_ONE_M1459_C1_GENERIC_SEAL_VCS_LAUNCH"
checks = 0
attacks: list[dict[str, Any]] = []


class HammerFailure(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        raise HammerFailure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    need(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == digest,
         "identity drift: " + str(path))


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    need(path.is_file() and not path.is_symlink(), "JSON not regular")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           HammerFailure("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root is not object")
    return value


def verify_sidecar(path: Path, payload_digest: str, side_digest: str,
                   outer_digest: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(side) + ".seal.sha256")
    regular(path, payload_digest)
    regular(side, side_digest)
    regular(outer, outer_digest)
    need(side.read_text(encoding="utf-8").split() == [payload_digest, path.name],
         "release sidecar content")
    need(outer.read_text(encoding="utf-8").split() == [side_digest, side.name],
         "release outer content")


def verify_tree(root: Path, review_digest: str, manifest_digest: str,
                outer_digest: str) -> dict[str, Any]:
    need(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular(root / "review.json", review_digest)
    regular(manifest, manifest_digest)
    regular(outer, outer_digest)
    need(outer.read_text().split() == [manifest_digest, "SHA256SUMS"],
         "authority outer seal")
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, "manifest row")
        digest, name = fields
        name = name.lstrip("*")
        rel = Path(name)
        need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
             name not in listed and not rel.is_absolute() and ".." not in rel.parts,
             "unsafe manifest member")
        listed[name] = digest
    actual = set()
    for member in root.rglob("*"):
        rel = member.relative_to(root).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        need(not member.is_symlink(), "sealed symlink")
        if member.is_file():
            actual.add(rel)
        else:
            need(member.is_dir(), "sealed special member")
    need(actual == set(listed), "sealed membership drift")
    for name, digest in listed.items():
        regular(root / name, digest)
    return strict_json(root / "review.json")


def changed(value: Any) -> Any:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1466_MUTATED"
    if type(value) is list:
        return list(value) + ["M1466_MUTATED"]
    if type(value) is dict:
        return {**value, "m1466_extra": True}
    raise TypeError(type(value))


def mutate(value: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    node = result
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = changed(node[path[-1]])
    return result


def reject(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except BaseException as error:
        attacks.append({"attack": label, "rejected": True,
                        "exception": type(error).__name__ + ": " + str(error)})
        return
    raise HammerFailure("false negative: " + label)


def validate_release(candidate: dict[str, Any], canonical: dict[str, Any]) -> None:
    need(candidate == canonical, "release exact-set/value drift")
    identity = candidate.get("identity", {})
    need(candidate.get("status") ==
         "AUTHORIZE_ONE_M1459_C1_GENERIC_SEAL_UNIT_DELAY_VCS_ATTEMPT",
         "release status")
    need(candidate.get("launch_now") is False and
         candidate.get("inert_until_m1466") is True, "release inertness")
    need(identity.get("runner_sha256") == EXPECTED["runner"] and
         identity.get("source_checker_sha256") == EXPECTED["checker"] and
         identity.get("source_tests_sha256") == EXPECTED["source_tests"] and
         identity.get("runtime_tests_sha256") == EXPECTED["runtime_tests"] and
         identity.get("source_contract_sha256") == EXPECTED["source_contract"] and
         identity.get("source_hammer_review_sha256") == EXPECTED["source_hammer_review"] and
         identity.get("source_hammer_manifest_sha256") == EXPECTED["source_hammer_manifest"] and
         identity.get("source_hammer_outer_file_sha256") == EXPECTED["source_hammer_outer"],
         "release identity binding")
    need(candidate.get("authorization") == AUTHORIZATION, "release authorization")
    bounds = candidate.get("execution_bounds", {})
    need(bounds.get("attempt_consumed_before_license_or_tool") is True and
         bounds.get("same_uid_collision_gates_before_attempt") == 2 and
         bounds.get("failure_quarantine_recursive_manifest_and_outer_seal") is True and
         bounds.get("canonical_success_recursive_manifest_and_outer_seal") is True and
         bounds.get("namespace_reuse_restore_delete_alias_or_substitution_forbidden") is True and
         bounds.get("automatic_retry") is False, "release bounds")
    split = candidate.get("runtime_split", {})
    need(split.get("source_tests_invoked_by_runner") is False and
         split.get("runtime_present_tests_invoked_by_runner") is True and
         split.get("runtime_present_tests_require_future_absent") is False and
         split.get("runtime_present_suite_sha256") == EXPECTED["runtime_tests"],
         "release runtime split")
    repair = candidate.get("repair_boundary", {})
    need(repair.get("generic_recursive_verifier_requires_review_json") is False and
         repair.get("authority_verifier_requires_review_json") is True and
         repair.get("attempt_stage_uses_generic_verifier") is True and
         repair.get("failure_stage_uses_generic_verifier") is True and
         repair.get("success_stage_uses_generic_verifier") is True and
         repair.get("authority_chain_uses_authority_verifier") is True,
         "release seal split")
    need(candidate.get("claim_boundary") == CLAIMS, "release claims")


def expected_final() -> dict[str, Any]:
    return {
        "schema": "m1466_m1465_m1459_c1_generic_seal_successor_final_launch_hammer_r1_v1",
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
        "score": 100,
        "p0_count": 0,
        "p1_count": 0,
        "authorization": dict(AUTHORIZATION),
        "validation": {},
        "hammer_execution": {"license_queries": 0, "vcs": 0, "simv": 0,
                             "dc": 0, "pt": 0, "ptpx": 0, "eda": 0,
                             "attempt_consumed": False, "result_created": False},
        "claim_boundary": dict(CLAIMS),
        "verdict": (
            "Exact-byte M1459/M1464/M1465 chain passes the different-author final "
            "source hammer. Exactly one future foundry-UNIT_DELAY VCS compile and "
            "one simv run are authorized with no retry; this hammer launched no tool."
        ),
    }


def validate_final(candidate: dict[str, Any], canonical: dict[str, Any]) -> None:
    need(candidate == canonical, "final authority exact-set/value drift")
    need(candidate.get("status") == FINAL_STATUS and
         candidate.get("authorization") == AUTHORIZATION and
         candidate.get("bindings", {}).get("launch_release_sha256") == EXPECTED["release"] and
         candidate.get("claim_boundary") == CLAIMS, "final authority core fields")


def audit_runner() -> dict[str, Any]:
    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    main = ast.unparse(functions["main"])
    generic = text[text.index("def verify_recursive_seal_generic("):
                   text.index("def verify_authority(")]
    authority = text[text.index("def verify_authority("):
                     text.index("def seal_dir_generic(")]
    need("review.json" not in generic, "generic verifier requires review")
    need("verify_recursive_seal_generic(root" in authority and
         'review = root / "review.json"' in authority and
         "strict_json(review)" in authority, "authority verifier is not generic-plus-review")
    need(text.count("run_tool(COMPILE_COMMAND") == 1 and
         text.count("run_tool(SIM_COMMAND") == 1 and
         text.count("compile_count = 1") == 1 and
         text.count("sim_count = 1") == 1, "one tool count")
    resource = main.index("phase = 'RESOURCE_PREFLIGHT'")
    attempt = main.index("phase = 'ATTEMPT_CONSUME'")
    consume = main.index("publish_no_replace(ATTEMPT_STAGE, ATTEMPT)")
    license_at = main.index("SNPSLMD_LICENSE_FILE")
    compile_at = main.index("phase = 'COMPILE'")
    need(main.count("BASE.collision_gate()", resource, attempt) == 2 and
         main.index("namespace_gate()") < resource < attempt < consume < license_at < compile_at,
         "collision/attempt/tool ordering")
    need('run_python_gate(BASE.SOURCE_CHECKER, "runtime_present")' in text and
         'run_python_gate(BASE.RUNTIME_TESTS, "runtime_present")' in text and
         "BASE.SOURCE_TESTS" not in text, "runtime suite reachability")
    need("seal_dir_generic(ATTEMPT_STAGE)" in text and
         "seal_dir_generic(FAILURE_STAGE)" in text and
         "seal_dir_generic(WORK)" in text and
         "publish_no_replace(FAILURE_STAGE, QUARANTINE)" in text and
         "publish_no_replace(WORK, RESULT)" in text, "generic stage sealing")
    need("automatic_retry=True" not in text and
         "automatic_retry=true" not in text and
         "shutil.rmtree" not in text and "renameat2" in text, "no-retry/noreplace")
    need(all(name in text for name in (
        "M1459_EXPECTED_RUNNER_SHA256", "M1459_EXPECTED_SOURCE_CONTRACT_SHA256",
        "M1459_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
        "M1459_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256",
        "M1459_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
        "M1459_EXPECTED_LAUNCH_RELEASE_SHA256",
        "M1459_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
        "M1459_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256",
        "M1459_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256")), "external pin reachability")
    need(FINAL_STATUS in text and "launch_release_sha256" in text and
         "authorization" in text, "final authority reachability")
    return {"one_compile": True, "one_sim": True,
            "collision_gates_before_attempt": 2,
            "attempt_before_license_and_tool": True,
            "source_suite_unreachable_at_launch": True,
            "runtime_suite_reachable": True,
            "generic_stage_seal": True, "authority_review_required": True,
            "atomic_noreplace": True, "automatic_retry": False}


def namespace_audit() -> dict[str, Any]:
    need(all(not os.path.lexists(path) for path in (ATTEMPT, RESULT, QUARANTINE)),
         "canonical one-shot namespace consumed")
    patterns = (".m1459_c1_generic_seal_vcs_work.*",
                ".m1459_c1_generic_seal_vcs_attempt_stage.*",
                ".m1459_c1_generic_seal_vcs_failure_stage.*")
    residues = [str(path) for pattern in patterns for path in
                (HW / "results").glob(pattern)]
    need(not residues, "M1459 staging residue")
    return {"attempt_absent": True, "result_absent": True,
            "quarantine_absent": True, "stage_residue_count": 0}


def run_source_suite_with_futures_virtualized() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("m1466_source_suite", SOURCE_TESTS)
    need(spec is not None and spec.loader is not None, "source suite import")
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.argv)
    try:
        sys.argv = [str(SOURCE_TESTS)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    with tempfile.TemporaryDirectory() as temp_name:
        absent = Path(temp_name)
        module.C.M1464 = absent / "m1464_absent"
        module.C.M1465 = absent / "m1465_absent"
        module.C.M1466 = absent / "m1466_absent"
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    need(result.testsRun == 18 and not result.failures and not result.errors,
         "source suite failed: " + output)
    (HERE / "source_suite_output.txt").write_text(output, encoding="utf-8")
    return {"tests": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "future_paths_virtualized_absent": True}


def run_runtime_suite() -> dict[str, Any]:
    completed = subprocess.run(
        [str(PYTHON), "-I", str(RUNTIME_TESTS), "--mode", "runtime_present"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=180, check=False,
        env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
             "PYTHONDONTWRITEBYTECODE": "1"})
    need(completed.returncode == 0, "runtime suite failed: " + completed.stderr)
    value = json.loads(completed.stdout)
    need(value.get("status") == "PASS_M1433_RUNTIME_PRESENT_LAUNCH_TESTS" and
         value.get("checks", {}).get("regressions") ==
         {"attacks": 16, "rejected": 16, "false_negatives": 0} and
         all(value.get(name) == 0 for name in
             ("license_queries", "vcs_runs", "simv_runs", "eda_runs")),
         "runtime suite receipt")
    (HERE / "runtime_present_output.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def sidecar_attacks() -> None:
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        payload = temp / RELEASE.name
        side = Path(str(payload) + ".sha256")
        outer = Path(str(side) + ".seal.sha256")
        shutil.copy2(RELEASE, payload)
        shutil.copy2(Path(str(RELEASE) + ".sha256"), side)
        shutil.copy2(Path(str(RELEASE) + ".sha256.seal.sha256"), outer)
        reject("release_payload_sha_mutation",
               lambda: (payload.write_text(payload.read_text() + " \n"),
                        verify_sidecar(payload, EXPECTED["release"], EXPECTED["release_side"],
                                       EXPECTED["release_outer"])))
        shutil.copy2(RELEASE, payload)
        side.write_text("0" * 64 + "  " + payload.name + "\n")
        reject("release_sidecar_content_mutation",
               lambda: verify_sidecar(payload, EXPECTED["release"], EXPECTED["release_side"],
                                      EXPECTED["release_outer"]))
        shutil.copy2(Path(str(RELEASE) + ".sha256"), side)
        outer.write_text("0" * 64 + "  " + side.name + "\n")
        reject("release_outer_content_mutation",
               lambda: verify_sidecar(payload, EXPECTED["release"], EXPECTED["release_side"],
                                      EXPECTED["release_outer"]))


def seal() -> None:
    rows = []
    for path in HERE.rglob("*"):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            need(not path.is_symlink(), "review symlink")
            rows.append((path.relative_to(HERE).as_posix(), sha(path)))
    rows.sort()
    manifest = HERE / "SHA256SUMS"
    manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in rows),
                        encoding="utf-8")
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
    need(author.get("authorization", {}).get("launch_release_authoring") is False and
         author.get("authorization", {}).get("vcs_compiles") == 0 and
         author.get("authorization", {}).get("simv_runs") == 0 and
         author.get("authorization", {}).get("all_other_eda_runs") == 0 and
         hammer.get("status") ==
         "PASS_M1464_M1459_C1_GENERIC_SEAL_SUCCESSOR_SOURCE__RELEASE_NOT_AUTHORED" and
         hammer.get("score") == 100 and hammer.get("p0_count") == 0 and
         hammer.get("p1_count") == 0, "predecessor authority state")

    canonical_release = strict_json(RELEASE)
    validate_release(canonical_release, canonical_release)
    runner_findings = audit_runner()
    namespaces = namespace_audit()
    source_suite = run_source_suite_with_futures_virtualized()
    runtime = run_runtime_suite()

    release_paths = [
        ("status",), ("launch_now",), ("inert_until_m1466",),
        ("identity", "runner_sha256"), ("identity", "source_checker_sha256"),
        ("identity", "source_tests_sha256"), ("identity", "runtime_tests_sha256"),
        ("identity", "source_contract_sha256"),
        ("identity", "source_hammer_review_sha256"),
        ("identity", "source_hammer_manifest_sha256"),
        ("identity", "source_hammer_outer_file_sha256"),
        ("authorization", "vcs_compiles"), ("authorization", "simv_runs"),
        ("authorization", "all_other_eda_runs"),
        ("authorization", "automatic_retry"),
        ("execution_bounds", "attempt_consumed_before_license_or_tool"),
        ("execution_bounds", "same_uid_collision_gates_before_attempt"),
        ("execution_bounds", "failure_quarantine_recursive_manifest_and_outer_seal"),
        ("execution_bounds", "canonical_success_recursive_manifest_and_outer_seal"),
        ("execution_bounds", "namespace_reuse_restore_delete_alias_or_substitution_forbidden"),
        ("execution_bounds", "automatic_retry"),
        ("runtime_split", "source_tests_invoked_by_runner"),
        ("runtime_split", "runtime_present_tests_invoked_by_runner"),
        ("runtime_split", "runtime_present_tests_require_future_absent"),
        ("runtime_split", "runtime_present_suite_sha256"),
        ("repair_boundary", "generic_recursive_verifier_requires_review_json"),
        ("repair_boundary", "authority_verifier_requires_review_json"),
        ("repair_boundary", "attempt_stage_uses_generic_verifier"),
        ("repair_boundary", "failure_stage_uses_generic_verifier"),
        ("repair_boundary", "success_stage_uses_generic_verifier"),
        ("repair_boundary", "authority_chain_uses_authority_verifier"),
        ("one_shot_namespaces", "attempt"), ("one_shot_namespaces", "result"),
        ("one_shot_namespaces", "failure_quarantine"),
        ("one_shot_namespaces", "work_prefix"),
        ("one_shot_namespaces", "attempt_stage_prefix"),
        ("one_shot_namespaces", "failure_stage_prefix"),
        ("final_hammer_gate", "required_status"),
    ] + [("claim_boundary", key) for key in CLAIMS]
    for path in release_paths:
        reject("release_" + "__".join(path),
               lambda path=path: validate_release(mutate(canonical_release, path),
                                                  canonical_release))
    sidecar_attacks()

    final = expected_final()
    validate_final(final, final)
    final_paths = [
        ("status",), ("bindings", "runner_sha256"),
        ("bindings", "runtime_tests_sha256"),
        ("bindings", "launch_release_sha256"),
        ("bindings", "launch_release_sidecar_sha256"),
        ("bindings", "launch_release_outer_file_sha256"),
        ("authorization", "vcs_compiles"), ("authorization", "simv_runs"),
        ("authorization", "all_other_eda_runs"),
        ("authorization", "automatic_retry"),
        ("claim_boundary", "functional_vcs"), ("claim_boundary", "headline"),
        ("hammer_execution", "license_queries"), ("hammer_execution", "vcs"),
        ("hammer_execution", "simv"), ("hammer_execution", "eda"),
        ("hammer_execution", "attempt_consumed"),
    ]
    for path in final_paths:
        reject("final_" + "__".join(path),
               lambda path=path: validate_final(mutate(final, path), final))

    false_negatives = sum(not item["rejected"] for item in attacks)
    need(false_negatives == 0 and len(attacks) == len(release_paths) + 3 + len(final_paths),
         "adversarial accounting")
    validation = {
        "exact_live_chain": True,
        "runner": runner_findings,
        "namespaces": namespaces,
        "source_suite": source_suite,
        "runtime_present_suite_status": runtime["status"],
        "runtime_present_regressions": runtime["checks"]["regressions"],
        "adversarial_mutations": len(attacks),
        "adversarial_false_negatives": false_negatives,
        "release_status_and_exact_set": True,
        "release_payload_sidecar_outer_exact": True,
        "generic_stage_vs_authority_seal_split": True,
        "fresh_attempt_result_quarantine_and_stage_namespaces": True,
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
    (HERE / "review.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "review.md").write_text(
        "# M1466 independent final launch hammer\n\n"
        "**PASS 100/100; P0=0, P1=0.** The exact M1459/M1464/M1465 "
        "chain, virtualized-absence source suite, runtime-present suite, release "
        "sidecars, generic-versus-authority seal split, two collision gates, "
        "one-shot namespaces, no-retry bound, and adversarial mutations all pass "
        "with zero false negatives. Exactly one future UNIT_DELAY VCS compile and "
        "one simv run are authorized. This hammer ran no EDA tool.\n",
        encoding="utf-8")
    (HERE / "NO_LICENSE_NO_VCS_NO_SIMV_NO_EDA.txt").write_text(
        "M1466_SOURCE_ONLY_HAMMER__NO_LICENSE_QUERY_NO_VCS_NO_SIMV_NO_EDA\n",
        encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(FINAL_STATUS + "\n", encoding="utf-8")
    seal()
    print(json.dumps({"status": FINAL_STATUS, "score": 100, "p0_count": 0,
                      "p1_count": 0, "checks": checks, "attacks": len(attacks),
                      "false_negatives": false_negatives,
                      "review_sha256": sha(HERE / "review.json"),
                      "manifest_sha256": sha(HERE / "SHA256SUMS"),
                      "outer_file_sha256": sha(HERE / "SHA256SUMS.seal.sha256")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
