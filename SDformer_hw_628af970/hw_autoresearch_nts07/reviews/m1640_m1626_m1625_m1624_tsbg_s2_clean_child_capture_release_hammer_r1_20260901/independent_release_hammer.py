#!/usr/bin/env python3
"""M1640 read-only hammer for the M1626 clean-child capture release."""
from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_"
    "successor_r1.py")
TEST = HW / "tests/test_m1624_motion_ep34_s2_tsbg_clean_child_source.py"
CONTRACT = HW / (
    "contracts/m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_"
    "source_contract_r1_20260901.json")
REVIEW = HW / (
    "reviews/m1625_m1624_motion_ep34_s2_tsbg_clean_child_source_hammer_"
    "r1_20260901")
RELEASE = HW / (
    "contracts/m1626_m1625_m1624_motion_ep34_s2_tsbg_clean_child_capture_"
    "release_r1_20260901.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_IDENTITY = {
    "source_sha256": "ad36ab02b598f28458ed226f816b47281b7d388fddfe80bc7ea15155709ba76f",
    "test_sha256": "5b44434df85b2832435ded94258a9a9f038f902ed6e77de1f4b7d690c497891b",
    "source_contract_sha256": "2ba3445c2c40c437124c62f49881db1b8443344aa19afc504f4f45aa1c1eacd9",
    "m1434_source_sha256": "b28c8507f077b754048fc54afd9fe04900dac854b273df2ba1981fa5f892b6ed",
    "m1558_source_sha256": "e6686564064ae3acda2bfcfc8c2d75061eb9cb591bc739d090bc03911469b089",
    "m1458_manifest_sha256": "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d",
    "m1512_review_sha256": "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    "m1598_review_sha256": "e887266475d28f7c2cfba3f69cbbbd103eed9db08905eebe042528f2baea1065",
    "checkpoint_sha256": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "docs359_sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "review_sha256": "fdb024b00ca337e90cca2b482b7db92aa40abdd72a50122482d54c56b6ab37ac",
    "review_manifest_sha256": "3b4ae63cc2244f4508dc862718306e13a134815d67d9db3fe00f59d96f8b6cf7",
    "review_outer_file_sha256": "21ea95d2a0eec39bdbec80b6c584be1aec85f92d5202338192d4412d916c13b1",
}
EXPECTED_REVIEW_IDENTITY = dict(
    (key, value) for key, value in EXPECTED_IDENTITY.items()
    if key not in ("review_sha256", "review_manifest_sha256",
                   "review_outer_file_sha256"))
EXPECTED_INTERPRETER = {
    "path": "/opt/conda/envs/sdformerflow/bin/python3.10",
    "sha256": "89520a3f2bc6e4f670921bd7a71a66eb0073775e685f6cbefda0dcda7bc42aa0",
}
EXPECTED_AUTH = {
    "parent_calls": 1,
    "clean_child_processes": 1,
    "gpu_runs": 1,
    "production_captures": 1,
    "automatic_retry": False,
    "all_other_runs": 0,
}
EXPECTED_NAMESPACES = {
    "result": "hw_autoresearch_nts07/results/m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_r1_20260901",
    "attempt": "hw_autoresearch_nts07/results/.m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_r1_20260901.attempt_consumed",
    "work": "hw_autoresearch_nts07/results/.m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_r1_20260901.work",
    "failure": "hw_autoresearch_nts07/results/m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_r1_20260901.failed_no_retry",
}
EXPECTED_CLAIMS = {
    "tsbg_dse": False,
    "aee": False,
    "rtl": False,
    "eda": False,
    "performance": False,
    "paper_result": False,
}
EXPECTED_SHA = {
    SOURCE: EXPECTED_IDENTITY["source_sha256"],
    TEST: EXPECTED_IDENTITY["test_sha256"],
    CONTRACT: EXPECTED_IDENTITY["source_contract_sha256"],
    REVIEW / "review.json": EXPECTED_IDENTITY["review_sha256"],
    REVIEW / "SHA256SUMS": EXPECTED_IDENTITY["review_manifest_sha256"],
    REVIEW / "SHA256SUMS.seal.sha256": EXPECTED_IDENTITY["review_outer_file_sha256"],
    RELEASE: "ce15529bcfceda5be92084bdb411330b0c56c8fe47c7024dd9b35a1a0490e273",
    Path(str(RELEASE) + ".sha256"): "c296bf381aa281ce16088ce2bf1f4d56f440087cd4bed6a80969b01ef0897381",
    Path(str(RELEASE) + ".sha256.seal.sha256"): "6c8baae067616e3dac38b2a7caf87227fd89fb71c0877c3b2bd79e7d7fd015ef",
    DOCS359: EXPECTED_IDENTITY["docs359_sha256"],
}


class Failure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load_text(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("non-finite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def strict_json(path):
    return strict_load_text(Path(path).read_text(encoding="utf-8"))


def verify_regular(path, digest):
    require(path.is_file() and not path.is_symlink() and
            stat.S_ISREG(path.lstat().st_mode), "nonregular: " + str(path))
    require(sha(path) == digest, "identity drift: " + str(path))


def verify_file_seal(path):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sidecar.read_text(encoding="ascii") ==
            sha(path) + "  " + path.name + "\n", "release inner seal mismatch")
    require(outer.read_text(encoding="ascii") ==
            sha(sidecar) + "  " + sidecar.name + "\n", "release outer seal mismatch")


def verify_dir_seal(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(root.is_dir() and not root.is_symlink(), "M1625 review dir absent")
    require(outer.read_text(encoding="ascii") ==
            sha(manifest) + "  SHA256SUMS\n", "M1625 outer seal mismatch")
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        require(re.match(r"^[0-9a-f]{64}  (?:\./)?[^/\n][^\n]*$", row) is not None,
                "malformed M1625 manifest row")
        digest, raw_name = row.split("  ", 1)
        name = raw_name[2:] if raw_name.startswith("./") else raw_name
        require(name not in listed and not Path(name).is_absolute() and
                all(part not in ("", ".", "..") for part in Path(name).parts),
                "unsafe/duplicate M1625 member")
        listed[name] = digest
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        for name in list(dirs) + list(files):
            path = Path(base) / name
            require(not path.is_symlink(), "M1625 seal contains symlink")
            rel = path.relative_to(root).as_posix()
            if path.is_file() and rel not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                actual.add(rel)
    require(actual == set(listed), "M1625 sealed topology drift")
    for name, digest in listed.items():
        verify_regular(root / name, digest)


def validate_release(release, review):
    require(set(release) == {
        "schema", "milestone", "date_cst", "status", "scope", "identity",
        "child_interpreter", "authorization", "namespaces",
        "execution_environment", "claim_boundary", "post_run_gate",
        "author_execution_receipt"}, "release top-level schema drift")
    require(release.get("schema") ==
            "m1626_m1625_m1624_clean_child_capture_release_r1_v1" and
            release.get("milestone") == "M1626" and
            release.get("date_cst") == "2026-09-01" and
            release.get("status") ==
            "AUTHORIZE_ONE_M1624_EP34_S2_TSBG_REDUCED_BINARY_CLEAN_CHILD_CAPTURE",
            "release schema/status drift")
    require(release.get("identity") == EXPECTED_IDENTITY,
            "release identity is not exactly the M1624 expected dictionary")
    require(review.get("identity") == EXPECTED_REVIEW_IDENTITY and
            review.get("status") ==
            "PASS_M1625_M1624_CLEAN_CHILD_SOURCE__AUTHORIZE_RELEASE_AUTHORING__NO_CAPTURE" and
            review.get("score") >= 95 and review.get("p0_count") == 0 and
            review.get("p1_count") == 0 and
            review.get("authorization") == {
                "release_authoring": True, "capture": False,
                "gpu": False, "automatic_retry": False},
            "M1625 authority drift")
    require(release.get("child_interpreter") == EXPECTED_INTERPRETER,
            "remote fixed interpreter path/SHA drift")
    require(release.get("authorization") == EXPECTED_AUTH,
            "single parent/child/GPU/capture/no-retry budget drift")
    require(release.get("namespaces") == EXPECTED_NAMESPACES and
            len(set(release["namespaces"].values())) == 4,
            "four exact distinct namespaces drift")
    require(release.get("claim_boundary") == EXPECTED_CLAIMS and
            all(value is False for value in release["claim_boundary"].values()),
            "release claim boundary is not all false")
    require(release.get("execution_environment") == {
        "host": "ssh.sd5ai.scnet.cn:10037",
        "worktree_policy": "fresh hardware-branch worktree; checkpoint/config read from fixed /root/private_data active training tree",
        "checkpoint_read_only": True,
        "shared_gpu_lease_required": True,
    }, "execution environment drift")
    require(release.get("post_run_gate") == {
        "different_author_result_hammer_required": True,
        "tsbg_bundles": [2, 4, 8],
        "s2_requires_paired_40_sample_aee": True,
        "no_performance_before_result_hammer": True,
    }, "post-run gate drift")
    require(release.get("author_execution_receipt") == {
        "remote_write": False, "checkpoint_load": False, "gpu_runs": 0,
        "capture_runs": 0, "dse_runs": 0, "eda_runs": 0,
    }, "release author execution receipt drift")
    require("one fixed clean-child capture" in release.get("scope", "") and
            "no retry" in release.get("scope", "") and
            "performance claim" in release.get("scope", ""),
            "release scope boundary drift")


def function_block(text, name, next_name):
    begin = text.index("def " + name + "(")
    end = text.index("def " + next_name + "(", begin)
    return text[begin:end]


def audit_source_path(text):
    tree = ast.parse(text)
    functions = dict((node.name, node) for node in tree.body
                     if isinstance(node, ast.FunctionDef))
    require("validate_future_authorities" in functions and
            "launch_parent" in functions and "fixed_clean_child" in functions,
            "future/runtime functions absent")
    authority = function_block(text, "validate_future_authorities",
                               "require_fresh_namespaces")
    parent = function_block(text, "launch_parent", "source_self_check")
    child = function_block(text, "fixed_clean_child", "launch_parent")
    require("expected_release_identity = dict(expected_identity" in authority and
            'release.get("identity") == expected_release_identity' in authority,
            "source does not exact-compare the release identity dictionary")
    require('interpreter.get("path") == str(CHILD_PYTHON)' in authority and
            'regular_exact(CHILD_PYTHON, interpreter["sha256"]' in authority,
            "source does not terminate at exact remote interpreter dependency")
    require('release.get("authorization") == {' in authority and
            '"parent_calls": 1, "clean_child_processes": 1' in authority and
            '"gpu_runs": 1, "production_captures": 1' in authority and
            '"automatic_retry": False, "all_other_runs": 0' in authority,
            "source release budget gate drift")
    require('release.get("namespaces") == {' in authority and
            'release.get("claim_boundary") == {' in authority,
            "source namespace/claim exact gates absent")
    require(parent.count("subprocess.run(") == 1 and
            'command = [str(CHILD_PYTHON), "-I", str(SOURCE), "--fixed-clean-child"]' in parent and
            "while " not in parent and "for " not in parent and
            "fixed clean child failed; no retry" in parent,
            "parent single-child/no-retry path drift")
    ordered = ("verify_fixed_metadata(expect_future_absent=False)",
               "validate_future_authorities()", "require_fresh_namespaces()",
               "subprocess.run(")
    require([parent.index(token) for token in ordered] ==
            sorted(parent.index(token) for token in ordered),
            "parent bypasses future authority/runtime preconditions")
    child_order = ("verify_fixed_metadata(expect_future_absent=False)",
                   "release = validate_future_authorities()",
                   "require_fresh_namespaces()",
                   "Path(sys.executable).resolve() == CHILD_PYTHON.resolve()",
                   "with substrate.exclusive_gpu_lease",
                   "consume_attempt(release)",
                   "model = profile.build_model",
                   "for row in samples:", "WORK.rename(RESULT)")
    require([child.index(token) for token in child_order] ==
            sorted(child.index(token) for token in child_order),
            "child authority/interpreter/attempt/runtime order drift")


def run_remote_dependency_probe():
    before = {name: os.path.lexists(str(ROOT / relative))
              for name, relative in EXPECTED_NAMESPACES.items()}
    require(not any(before.values()), "namespace present before dependency probe")
    module_name = "m1640_read_only_m1624_probe"
    before_modules = set(sys.modules)
    spec = importlib.util.spec_from_file_location(module_name, str(SOURCE))
    require(spec is not None and spec.loader is not None, "cannot import source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    added = set(sys.modules) - before_modules
    require("torch" not in added and "numpy" not in added,
            "inert import loaded GPU/payload libraries")
    launches = []

    def forbidden_subprocess(*args, **kwargs):
        launches.append((args, kwargs))
        raise Failure("subprocess launch reached")

    module.subprocess.run = forbidden_subprocess
    try:
        module.launch_parent()
    except module.M1624Error as error:
        require(str(error) == "missing fixed child interpreter",
                "source stopped before/after expected remote interpreter dependency: " + str(error))
    else:
        raise Failure("source bypassed missing remote interpreter dependency")
    require(not launches, "source launched child/remote process during dependency probe")
    after = {name: os.path.lexists(str(ROOT / relative))
             for name, relative in EXPECTED_NAMESPACES.items()}
    require(before == after and not any(after.values()),
            "dependency probe mutated a production namespace")
    return "PASS_FUTURE_AUTHORITY_TO_MISSING_REMOTE_INTERPRETER__NO_SUBPROCESS_NO_WRITE"


def mutated(value, path, replacement):
    candidate = copy.deepcopy(value)
    cursor = candidate
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    return candidate


def run_mutations(release, review, source_text):
    labels = []

    def reject_release(label, candidate):
        try:
            validate_release(candidate, review)
        except Failure:
            labels.append(label)
            return
        raise Failure("release mutation escaped: " + label)

    def reject_source(label, candidate):
        try:
            audit_source_path(candidate)
        except (Failure, SyntaxError, ValueError):
            labels.append(label)
            return
        raise Failure("source mutation escaped: " + label)

    for key in sorted(EXPECTED_IDENTITY):
        reject_release("identity_" + key,
                       mutated(release, ["identity", key], "0" * 64))
    extra = copy.deepcopy(release)
    extra["identity"]["extra_sha256"] = "0" * 64
    reject_release("identity_extra_key", extra)
    missing = copy.deepcopy(release)
    del missing["identity"]["checkpoint_sha256"]
    reject_release("identity_missing_key", missing)
    reject_release("interpreter_path", mutated(
        release, ["child_interpreter", "path"], "/usr/bin/python3"))
    reject_release("interpreter_sha", mutated(
        release, ["child_interpreter", "sha256"], "0" * 64))
    for key in ("parent_calls", "clean_child_processes", "gpu_runs",
                "production_captures"):
        reject_release("budget_" + key, mutated(
            release, ["authorization", key], 2))
    reject_release("automatic_retry", mutated(
        release, ["authorization", "automatic_retry"], True))
    reject_release("all_other_runs", mutated(
        release, ["authorization", "all_other_runs"], 1))
    for key in sorted(EXPECTED_NAMESPACES):
        reject_release("namespace_" + key, mutated(
            release, ["namespaces", key], EXPECTED_NAMESPACES["result"] + ".alias"))
    aliased = copy.deepcopy(release)
    aliased["namespaces"]["failure"] = aliased["namespaces"]["result"]
    reject_release("namespace_alias", aliased)
    for key in sorted(EXPECTED_CLAIMS):
        reject_release("claim_" + key, mutated(
            release, ["claim_boundary", key], True))
    reject_release("status", mutated(release, ["status"], "AUTHORIZE_RETRY"))
    reject_release("schema", mutated(release, ["schema"], "m1626_fake"))
    reject_release("remote_write_receipt", mutated(
        release, ["author_execution_receipt", "remote_write"], True))
    reject_release("capture_receipt", mutated(
        release, ["author_execution_receipt", "capture_runs"], 1))
    reject_release("post_result_hammer", mutated(
        release, ["post_run_gate", "different_author_result_hammer_required"], False))
    reject_release("performance_gate", mutated(
        release, ["post_run_gate", "no_performance_before_result_hammer"], False))

    reject_source("identity_subset_compare", source_text.replace(
        'release.get("identity") == expected_release_identity',
        'set(expected_release_identity).issubset(release.get("identity", {}))', 1))
    reject_source("interpreter_disk_check_removed", source_text.replace(
        'regular_exact(CHILD_PYTHON, interpreter["sha256"],\n                  "fixed child interpreter")',
        'pass', 1))
    reject_source("authority_after_subprocess", source_text.replace(
        "    validate_future_authorities()\n    require_fresh_namespaces()",
        "    require_fresh_namespaces()", 1))
    reject_source("second_child", source_text.replace(
        "    completed = subprocess.run(command", "    subprocess.run(command)\n    completed = subprocess.run(command", 1))
    reject_source("child_without_isolation", source_text.replace(
        'str(CHILD_PYTHON), "-I", str(SOURCE)',
        'str(CHILD_PYTHON), str(SOURCE)', 1))
    reject_source("retry_loop", source_text.replace(
        "    completed = subprocess.run(command", "    while True:\n        completed = subprocess.run(command", 1))
    reject_source("child_interpreter_check_after_gpu", source_text.replace(
        "    require(Path(sys.executable).resolve() == CHILD_PYTHON.resolve(),\n            \"child did not run under fixed interpreter\")",
        "    pass", 1))
    require(len(labels) == 47, "mutation population drift: " + str(len(labels)))
    return labels


def main():
    for path, digest in EXPECTED_SHA.items():
        verify_regular(path, digest)
    verify_file_seal(RELEASE)
    verify_dir_seal(REVIEW)
    release = strict_json(RELEASE)
    review = strict_json(REVIEW / "review.json")
    validate_release(release, review)
    contract = strict_json(CONTRACT)
    require(contract.get("future_budget_after_valid_release") == EXPECTED_AUTH and
            contract.get("new_namespaces", {}).get("result") == EXPECTED_NAMESPACES["result"] and
            contract.get("new_namespaces", {}).get("attempt") == EXPECTED_NAMESPACES["attempt"] and
            contract.get("new_namespaces", {}).get("work") == EXPECTED_NAMESPACES["work"] and
            contract.get("new_namespaces", {}).get("failure") == EXPECTED_NAMESPACES["failure"],
            "M1624 source contract future budget/namespaces drift")
    source_text = SOURCE.read_text(encoding="utf-8")
    audit_source_path(source_text)
    dependency = run_remote_dependency_probe()
    labels = run_mutations(release, review, source_text)
    print(json.dumps({
        "schema": "m1640_release_hammer_stdout_v1",
        "status": "PASS",
        "static_checks": 18,
        "mutations_rejected": len(labels),
        "mutation_labels": labels,
        "remote_runtime_dependency": dependency,
        "remote_connections": 0,
        "subprocess_launches": 0,
        "gpu_runs": 0,
        "capture_runs": 0,
        "namespace_writes": 0,
        "release_sha256": sha(RELEASE),
        "review_sha256": sha(REVIEW / "review.json"),
        "source_sha256": sha(SOURCE),
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Failure as error:
        raise SystemExit("FAIL_CLOSED_M1640: " + str(error))
