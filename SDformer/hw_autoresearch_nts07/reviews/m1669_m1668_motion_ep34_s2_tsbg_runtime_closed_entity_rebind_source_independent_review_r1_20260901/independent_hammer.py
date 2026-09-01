#!/usr/bin/env python3
"""M1669 read-only independent hammer for the M1668 TSBG source.

No remote connection, capture, GPU, checkpoint load, attempt creation, or EDA
is permitted here.  The source-only unit tests and pure pre-budget seams are
exercised under CPython 3.6 and 3.12.
"""
from __future__ import print_function

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_successor_r1.py"
TEST = HW / "tests/test_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_source.py"
SELECTION = HW / "contracts/m1668_motion_ep34_s2_tsbg_current_selection_entity_r1_20260901.json"
CONTRACT = HW / "contracts/m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_source_author_receipt_r1_20260901"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
FUTURE_REVIEW = HW / "reviews/m1669_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_source_independent_review_r1_20260901"
FUTURE_RELEASE = HW / "contracts/m1670_m1669_m1668_motion_ep34_s2_tsbg_runtime_closed_entity_rebind_capture_release_r1_20260901.json"

EXPECTED = {
    "source": "7e728162de630da2086dee5a39536fc9c4141d24dcde4f4840549c9aabc77d8b",
    "test": "ef36f416df749fc646fc901b662dc1fac7de4d9872989e29f5ba21e34c202fee",
    "selection": "e6b3dd82d5d1eb54e605595369bfc8228fd616ab707d58b2e4afd95c159f87c7",
    "contract": "723e8797889d231e36dca343281abff7eccb4c3080f4231e2746c4a083100165",
    "runtime_tar": "0524a94ccb36adc7ebc17603dedc322810141d8b14dc743923c5b942a5c6c36f",
    "m1647_source": "3e16c6f4b740a7a9454ad243de3c128185d3135f7a26ccbdfb7e94ae5505682a",
    "m1648_review": "c8292001df4481f78e09018a65ce86d1b512983634daffcd7ed1e1f034b9de7c",
    "m1649_release": "64cb869004753d1e9c8aeda3f6533657dd53334f4b3639ee85b3bd64981e555a",
    "checkpoint": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "config": "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39",
    "profile": "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
REMOTE = {"host": "ssh.sd5ai.scnet.cn", "port": 10037, "user": "root",
          "repository_root": "/root/private_data/work/sdformer_codex/SDformer"}


class HammerError(Exception):
    pass


def need(value, message):
    if not value:
        raise HammerError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not stat.S_ISLNK(mode)


def verify_file_seal(path, expected):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(regular(path) and sha(path) == expected, "payload identity drift " + str(path))
    need(regular(sidecar) and regular(outer), "double seal absent/nonregular " + str(path))
    need(sidecar.read_text() == expected + "  " + path.name + "\n", "inner seal drift")
    need(outer.read_text() == sha(sidecar) + "  " + sidecar.name + "\n", "outer seal drift")
    return {"payload_sha256": expected, "sidecar_sha256": sha(sidecar),
            "outer_seal_file_sha256": sha(outer)}


def verify_tree(root, label):
    need(root.is_dir() and not root.is_symlink(), label + " tree absent/symlink")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(regular(manifest) and regular(outer), label + " seal absent")
    need(outer.read_text() == sha(manifest) + "  SHA256SUMS\n", label + " outer drift")
    listed = {}
    for line in manifest.read_text().splitlines():
        fields = line.split("  ", 1)
        need(len(fields) == 2 and fields[1] not in listed and
             not Path(fields[1]).is_absolute() and ".." not in Path(fields[1]).parts,
             label + " malformed/unsafe member")
        listed[fields[1]] = fields[0]
        member = root / fields[1]
        need(regular(member) and sha(member) == fields[0], label + " member drift")
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        base_path = Path(base)
        for name in dirs:
            point = base_path / name
            need(not point.is_symlink() and any(point.iterdir()), label + " symlink/empty dir")
        for name in files:
            point = base_path / name
            need(not point.is_symlink(), label + " symlink file")
            if regular(point):
                actual.add(point.relative_to(root).as_posix())
    expected_population = set(listed) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    need(actual == expected_population, label + " recursive population drift")
    return {"manifest_entries": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def strict_json(path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            need(key not in output, "duplicate JSON key " + key)
            output[key] = value
        return output
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerError("nonfinite JSON " + token)))


def load_test_module():
    spec = importlib.util.spec_from_file_location("m1669_reviewed_m1668_test", str(TEST))
    need(spec is not None and spec.loader is not None, "cannot load M1668 tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_reject(label, function):
    try:
        function()
    except Exception as error:
        need(type(error).__name__ in ("M1668Error", "HammerError", "FileNotFoundError",
                                     "AssertionError", "ValueError", "KeyError", "TypeError"),
             "unexpected mutation exception %s: %s" % (label, type(error).__name__))
        return label
    raise HammerError("mutation not rejected: " + label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    errors = []
    try:
        need(regular(SOURCE) and sha(SOURCE) == EXPECTED["source"], "source SHA drift")
        need(regular(TEST) and sha(TEST) == EXPECTED["test"], "test SHA drift")
        need(regular(DOC359) and sha(DOC359) == EXPECTED["docs359"], "docs359 drift")
        selection_seal = verify_file_seal(SELECTION, EXPECTED["selection"])
        contract_seal = verify_file_seal(CONTRACT, EXPECTED["contract"])
        author_seal = verify_tree(AUTHOR, "M1668 author receipt")
        author = strict_json(AUTHOR / "review.json")
        remote_evidence = strict_json(AUTHOR / "remote_read_only_build_runtime.json")
        need(author.get("status") == "PASS_AUTHOR_DUAL_RUNTIME_CLOSED_ENTITY_REBIND_SOURCE__DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_CAPTURE" and
             author.get("p0_count") == 0 and author.get("p1_count") == 0,
             "author receipt status drift")
        need(remote_evidence == {
            "attempt_writes": 0, "capture": {"attention_windows_per_call": 100},
            "checkpoint_path": REMOTE["repository_root"] + "/neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth",
            "checkpoint_sha256": EXPECTED["checkpoint"],
            "config_path": REMOTE["repository_root"] + "/neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.yml",
            "config_sha256": EXPECTED["config"], "gpu_runs": 0,
            "runtime_keys": ["capture", "cohort", "contract_path", "output"],
            "status": "PASS_READ_ONLY_M1434_BUILD_RUNTIME"},
             "sealed remote build_runtime evidence drift")

        contract = strict_json(CONTRACT)
        selection = strict_json(SELECTION)
        need(contract["source"]["sha256"] == EXPECTED["source"] and
             contract["test"]["sha256"] == EXPECTED["test"] and
             contract["selection_identity"]["sha256"] == EXPECTED["selection"] and
             contract["runtime_data_handoff"]["archive_sha256"] == EXPECTED["runtime_tar"],
             "contract identity drift")
        need(contract["future_release_shape"] == {"parent_calls": 1,
             "clean_child_processes": 1, "gpu_runs": 1,
             "production_captures": 1, "automatic_retry": False,
             "all_other_runs": 0}, "one-shot contract drift")
        observation = selection["remote_read_only_observation"]
        need(observation["host"] == REMOTE["host"] + ":" + str(REMOTE["port"]) and
             observation["repository_root"] == REMOTE["repository_root"] and
             observation["observation_is_launch_authority"] is False and
             observation["future_launch_must_repeat_gpu_and_entity_preflight"] is True,
             "remote observation target/boundary drift")
        need(selection["checkpoint"]["sha256"] == EXPECTED["checkpoint"] and
             selection["configuration_current_capture_entity"]["sha256"] == EXPECTED["config"] and
             selection["profile"]["sha256"] == EXPECTED["profile"],
             "selection content identity drift")

        test_module = load_test_module()
        source = test_module.M
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
        result = unittest.TestResult()
        suite.run(result)
        need(result.testsRun == 18 and not result.failures and not result.errors,
             "M1668 source regression failed %r %r" % (result.failures, result.errors))
        source_check = source.source_self_check()
        need(source_check["status"] == "PASS_M1668_SOURCE_SELF_CHECK__RUNTIME_HANDOFF_CLOSED__NO_CAPTURE" and
             source_check["runtime_handoff_files"] == 9 and
             source_check["runtime_canonical_files"] == 7 and
             source_check["attempt_writes"] == 0 and source_check["gpu_runs"] == 0,
             "source self-check boundary drift")

        mutations = []
        old_tar = source.RUNTIME_TAR
        source.RUNTIME_TAR = old_tar.with_name("m1669_absent_runtime.tar")
        try:
            mutations.append(expect_reject("missing_m1257_runtime_tar",
                                           source.verify_runtime_handoff_source))
        finally:
            source.RUNTIME_TAR = old_tar

        with tempfile.TemporaryDirectory(prefix="m1669_entity_") as temp_name:
            entity = Path(temp_name) / "config.yml"
            entity.write_bytes(b"bound-config\n")
            st = entity.lstat()
            exact = {"absolute_path": str(entity), "device": st.st_dev,
                     "inode": st.st_ino, "mode": st.st_mode,
                     "mtime_ns": st.st_mtime_ns, "sha256": source.sha256(entity),
                     "size_bytes": st.st_size}
            source.verify_entity(entity, exact, "synthetic config")
            mutations.append(expect_reject("config_inode_drift", lambda:
                source.verify_entity(entity, dict(exact, inode=st.st_ino + 1), "synthetic config")))
            mutations.append(expect_reject("config_sha_drift", lambda:
                source.verify_entity(entity, dict(exact, sha256="0" * 64), "synthetic config")))

        events = []
        originals = (source.verify_predecessors, source.preflight_runtime_binding,
                     source.P.launch_parent)
        source.verify_predecessors = lambda: events.append("predecessors")
        source.preflight_runtime_binding = lambda: events.append("build_runtime")
        source.P.launch_parent = lambda: events.append("parent_budget") or 0
        try:
            need(source.launch_parent() == 0 and events ==
                 ["predecessors", "build_runtime", "parent_budget"],
                 "parent pre-budget order drift")
        finally:
            (source.verify_predecessors, source.preflight_runtime_binding,
             source.P.launch_parent) = originals

        events = []
        originals = (source.verify_predecessors, source.preflight_runtime_binding,
                     source.P.fixed_clean_child)
        source.verify_predecessors = lambda: events.append("predecessors")
        source.preflight_runtime_binding = lambda: events.append("repeat_identity_build_runtime")
        source.P.fixed_clean_child = lambda: events.append("gpu_attempt_delegate") or 0
        try:
            need(source.fixed_clean_child() == 0 and events ==
                 ["predecessors", "repeat_identity_build_runtime", "gpu_attempt_delegate"],
                 "child pre-GPU/attempt order drift")
        finally:
            (source.verify_predecessors, source.preflight_runtime_binding,
             source.P.fixed_clean_child) = originals

        reached = []
        originals = (source.verify_predecessors, source.preflight_runtime_binding,
                     source.P.launch_parent)
        source.verify_predecessors = lambda: None
        source.preflight_runtime_binding = lambda: (_ for _ in ()).throw(
            source.M1668Error("injected build_runtime failure"))
        source.P.launch_parent = lambda: reached.append("parent")
        try:
            mutations.append(expect_reject("failed_build_runtime_blocks_parent", source.launch_parent))
            need(not reached, "failed parent preflight reached budget")
        finally:
            (source.verify_predecessors, source.preflight_runtime_binding,
             source.P.launch_parent) = originals

        lower = source.P.P
        lower_text = source.P.M1624_SOURCE.read_text()
        lower_child = lower_text[lower_text.index("def fixed_clean_child():"):]
        need("os.O_EXCL" in lower_text and
             lower_child.index("m1434.build_runtime()") < lower_child.index("exclusive_gpu_lease") <
             lower_child.index("consume_attempt(release)") < lower_child.index("profile.load_config(CONFIG)"),
             "lower clean-child build/GPU/attempt/checkpoint order drift")
        old_loader = source.P.P.load_m1434
        with source._bound_exact_m1647():
            need(source.P.P.load_m1434 is source.load_m1434_rebound,
                 "child did not inherit rebound runtime loader")
        need(source.P.P.load_m1434 is old_loader, "rebound loader not restored")

        for label, mutate in (
            ("retry_true", lambda x: x["future_release_shape"].update(automatic_retry=True)),
            ("two_parent_calls", lambda x: x["future_release_shape"].update(parent_calls=2)),
            ("two_gpu_runs", lambda x: x["future_release_shape"].update(gpu_runs=2)),
            ("selection_checkpoint_reselected", lambda x:
                x["selection_semantics"].update(checkpoint_reselected=True)),
            ("selection_config_semantics_changed", lambda x:
                x["selection_semantics"].update(configuration_semantics_changed=True)),
            ("remote_host_drift", lambda x:
                x["remote_read_only_observation"].update(host="wrong:1")),
            ("remote_root_drift", lambda x:
                x["remote_read_only_observation"].update(repository_root="/wrong")),
        ):
            base = copy.deepcopy(contract if label in
                ("retry_true", "two_parent_calls", "two_gpu_runs") else selection)
            mutate(base)
            encoded = (json.dumps(base, sort_keys=True, separators=(",", ":")) + "\n").encode()
            expected_digest = EXPECTED["contract"] if base is contract else None
            # The exact whole-file identity is the first mutation gate.
            mutations.append(expect_reject(label, lambda data=encoded, expected=(
                EXPECTED["contract"] if label in ("retry_true", "two_parent_calls", "two_gpu_runs")
                else EXPECTED["selection"]): need(hashlib.sha256(data).hexdigest() == expected,
                                                   "whole-file identity mutation")))

        need(not os.path.lexists(str(FUTURE_RELEASE)), "M1670 unexpectedly exists")
        need(all(not os.path.lexists(str(path)) for path in
                 (source.RESULT, source.ATTEMPT, source.WORK, source.FAILURE)),
             "M1668 runtime namespace not fresh")
        status = source.REVIEW_STATUS
        score = 98
        p0 = []
        p1 = []
        p2 = ["The sealed remote build_runtime observation is transient, not launch authority. M1670 and its launcher hammer must pin ssh.sd5ai.scnet.cn:10037/root and /root/private_data/work/sdformer_codex/SDformer, install M1257 before source preflight, and repeat GPU/entity/runtime gates."]
    except Exception as error:
        errors.append("%s: %s" % (type(error).__name__, error))
        selection_seal = locals().get("selection_seal", {})
        contract_seal = locals().get("contract_seal", {})
        author_seal = locals().get("author_seal", {})
        source_check = locals().get("source_check", {})
        mutations = locals().get("mutations", [])
        status = "FAIL_M1669_M1668_TSBG_SOURCE_NOT_ADMITTED__NO_M1670_RELEASE"
        score = 0
        p0 = errors
        p1 = []
        p2 = []

    identity = {
        "source_sha256": EXPECTED["source"], "test_sha256": EXPECTED["test"],
        "source_contract_sha256": EXPECTED["contract"],
        "selection_identity_sha256": EXPECTED["selection"],
        "runtime_tar_sha256": EXPECTED["runtime_tar"],
        "m1647_source_sha256": EXPECTED["m1647_source"],
        "m1648_review_sha256": EXPECTED["m1648_review"],
        "m1649_release_sha256": EXPECTED["m1649_release"],
        "checkpoint_sha256": EXPECTED["checkpoint"],
        "config_sha256": EXPECTED["config"], "profile_sha256": EXPECTED["profile"],
        "docs359_sha256": EXPECTED["docs359"]}
    review = {
        "schema": "m1669_m1668_runtime_closed_entity_rebind_source_independent_review_r1_v1",
        "milestone": "M1669", "date_cst": "2026-09-01", "status": status,
        "verdict": "PASS" if not errors else "FAIL", "score": score,
        "p0": p0, "p0_count": len(p0), "p1": p1, "p1_count": len(p1),
        "p2": p2, "p2_count": len(p2), "identity": identity,
        "sealed_inputs": {"selection": selection_seal, "contract": contract_seal,
                          "author_receipt": author_seal},
        "failure_closure": {"m1257_runtime_handoff_archive_files": 9,
            "m1257_canonical_result_files": 7,
            "config_current_entity_bound": not errors,
            "build_runtime_before_parent_budget": not errors,
            "build_runtime_repeated_before_child_gpu_attempt": not errors,
            "lower_child_order": "build_runtime -> GPU lease -> O_EXCL attempt -> checkpoint/model",
            "one_parent_one_child_one_gpu_one_capture": not errors,
            "automatic_retry": False},
        "remote_read_only_evidence": {"status": remote_evidence.get("status") if 'remote_evidence' in locals() else None,
            "gpu_runs": 0, "attempt_writes": 0, "remote_writes_by_reviewer": 0,
            "host": REMOTE["host"], "port": REMOTE["port"], "user": REMOTE["user"],
            "repository_root": REMOTE["repository_root"],
            "transient_observation_is_launch_authority": False},
        "regression": {"python_runtime": "%d.%d.%d" % sys.version_info[:3],
            "source_tests_passed": 18 if not errors else 0,
            "source_self_check_status": source_check.get("status"),
            "dynamic_and_static_mutations_rejected": len(mutations),
            "mutation_labels": mutations},
        "authorization": {"release_authoring": not errors, "capture": False,
                          "gpu": False, "automatic_retry": False},
        "m1670_release_requirements": {"exact_remote_target": REMOTE,
            "install_m1257_before_source_preflight": True,
            "repeat_entity_and_build_runtime_parent_and_child": True,
            "one_attempt_no_retry": True,
            "different_author_release_hammer_before_remote_launch": True},
        "claim_boundary": {"source_only": True, "remote_deployment_complete": False,
            "capture": False, "gpu": False, "aee": False, "cycles": False,
            "traffic": False, "energy": False, "speedup": False,
            "system_speedup": False, "rtl": False, "eda": False,
            "paper_result": False},
        "review_execution": {"remote_connections": 0, "remote_writes": 0,
            "capture_runs": 0, "gpu_runs": 0, "attempt_writes": 0,
            "checkpoint_loads": 0, "eda_runs": 0, "git_commit": False,
            "git_push": False}}
    Path(args.output).write_text(json.dumps(review, ensure_ascii=False, indent=2,
                                           sort_keys=True, allow_nan=False) + "\n")
    print(status)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
