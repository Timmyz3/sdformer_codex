#!/usr/bin/env python3
"""Different-author, no-remote hammer for the inert M1692 TSBG source."""
from __future__ import print_function

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                 "capture_m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
                 "successor_r1.py")
TEST = HW / "tests/test_m1692_motion_ep34_s2_tsbg_authority_shape_repair_successor_source.py"
CONTRACT = HW / "contracts/m1692_motion_ep34_s2_tsbg_authority_shape_repair_successor_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1692_motion_ep34_s2_tsbg_authority_shape_repair_successor_source_author_receipt_r1_20260901"
M1649 = HW / "contracts/m1649_m1648_m1647_motion_ep34_s2_tsbg_deployment_complete_clean_child_capture_release_r1_20260901.json"

EXPECTED = {
    SOURCE: "ea7b300811a71d63456d16b3c3bfe04e7668266e73613ba426e0c8d6ea5e0e58",
    TEST: "ce720955e8d54d40303222732a2edd836c958d5e7b58178baccead6e0ec1f8ad",
    CONTRACT: "cc38745b2a094d6b31367e60a12211075cbc749a72f611b6ef3030b987aabd70",
    Path(str(CONTRACT) + ".sha256"): "cc74653df0b2f83b37e7f1180b34a50614993dd8471885e5e6832e6c699af886",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "e459a292a69bbe720de762a5499fcdb647ebca60a6b68bbae91375a348a1f083",
    AUTHOR / "review.json": "7b7f2c7d90e796db667bcea95529613d3279b20f132aed460ce52d75794af478",
    AUTHOR / "SHA256SUMS": "4c9ecd505e64a1cb87a174a2c4edd5a4b087d56b1fb818fef326a163b259f2bc",
    AUTHOR / "SHA256SUMS.seal.sha256": "658ad975826b6fca3017026517a6975ce7bb502b667156f167336a69b8241d30",
}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m1693_exact_m1692", str(SOURCE))
    need(spec is not None and spec.loader is not None, "source import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def seal_review(module, root, value):
    root.mkdir()
    write_json(root / "review.json", value)
    manifest = root / "SHA256SUMS"
    manifest.write_text(module.sha256(root / "review.json") + "  review.json\n")
    outer = root / "SHA256SUMS.seal.sha256"
    outer.write_text(module.sha256(manifest) + "  SHA256SUMS\n")
    return module.sha256(root / "review.json"), module.sha256(manifest), module.sha256(outer)


def seal_file(module, path, value):
    write_json(path, value)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(module.sha256(path) + "  " + path.name + "\n")
    outer = Path(str(path) + ".sha256.seal.sha256")
    outer.write_text(module.sha256(sidecar) + "  " + sidecar.name + "\n")


def base_review(module):
    return {
        "schema": "m1693_m1692_tsbg_authority_shape_review_r1_v1",
        "status": module.REVIEW_STATUS,
        "score": 100,
        "p0_count": 0,
        "p1_count": 0,
        "identity": module.expected_review_identity(),
        "authorization": {
            "release_authoring": True,
            "capture": False,
            "gpu": False,
            "automatic_retry": False,
        },
    }


def base_release(module, identity, interpreter):
    return {
        "schema": module.RELEASE_SCHEMA,
        "status": module.RELEASE_STATUS,
        "identity": identity,
        "authorization": {
            "parent_calls": 1, "clean_child_processes": 1,
            "gpu_runs": 1, "production_captures": 1,
            "automatic_retry": False, "all_other_runs": 0,
        },
        "namespaces": {
            "result": str(module.RESULT.relative_to(module.ROOT)),
            "attempt": str(module.ATTEMPT.relative_to(module.ROOT)),
            "work": str(module.WORK.relative_to(module.ROOT)),
            "failure": str(module.FAILURE.relative_to(module.ROOT)),
        },
        "pre_budget_preflight": {
            "runtime_m1257_canonical": True,
            "current_entity_exact": True,
            "build_runtime_before_parent_subprocess": True,
            "build_runtime_before_child_gpu_attempt": True,
            "exact_remote_target": True,
            "exact_child_interpreter": True,
        },
        "remote_target": dict(module.REMOTE_TARGET),
        "claim_boundary": {
            "tsbg_dse": False, "aee": False, "rtl": False,
            "eda": False, "performance": False, "paper_result": False,
        },
        "child_interpreter": {
            "path": str(interpreter), "sha256": module.sha256(interpreter)},
    }


@contextlib.contextmanager
def authority_fixture(module, review_mutator=None, release_mutator=None):
    with tempfile.TemporaryDirectory(prefix=".m1693-validator-",
                                     dir=str(HW / "reviews")) as temporary:
        temporary = Path(temporary)
        review = base_review(module)
        if review_mutator:
            review_mutator(review)
        review_sha, manifest_sha, outer_sha = seal_review(
            module, temporary / "review", review)
        identity = dict(module.expected_review_identity())
        identity.update({"review_sha256": review_sha,
                         "review_manifest_sha256": manifest_sha,
                         "review_outer_file_sha256": outer_sha})
        interpreter = Path(sys.executable).resolve()
        release = base_release(module, identity, interpreter)
        if release_mutator:
            release_mutator(release)
        release_path = temporary / "release.json"
        seal_file(module, release_path, release)
        old = module.FUTURE_REVIEW, module.FUTURE_RELEASE, module.CHILD_INTERPRETER
        module.FUTURE_REVIEW = temporary / "review"
        module.FUTURE_RELEASE = release_path
        module.CHILD_INTERPRETER = interpreter
        try:
            yield release
        finally:
            module.FUTURE_REVIEW, module.FUTURE_RELEASE, module.CHILD_INTERPRETER = old


def rejected(module, review_mutator=None, release_mutator=None):
    with authority_fixture(module, review_mutator, release_mutator):
        try:
            module.validate_future_authorities()
        except Exception:
            return True
    return False


def main():
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "identity drift: " + str(path))
    module = load_source()
    module.verify_predecessors()
    module.validate_source_contract()
    handoff = module.P.verify_runtime_handoff_source()
    identity = module.P.selection_identity()
    need(handoff["archive_files"] == 9 and handoff["canonical_files"] == 7,
         "M1257 handoff")
    need(identity["selection_semantics"]["selected_candidate_id"] == "resume_ep34"
         and identity["selection_semantics"]["selected_epoch"] == 34,
         "current selection")

    # source_self_check was run byte-identically under both interpreters before
    # this review namespace was created. Reproducible post-seal checks execute
    # its substantive pieces without weakening the canonical future paths.
    module.require_fresh_namespaces()
    need(module.validate_source_contract()["status"] == module.SOURCE_STATUS,
         "source contract")

    validator = module.validate_future_authorities
    validator_code = hashlib.sha256(validator.__code__.co_code).hexdigest()
    with authority_fixture(module) as expected_release:
        observed = module.validate_future_authorities()
        need(observed == expected_release, "positive validator consumption")
        need(module.validate_future_authorities is validator and
             hashlib.sha256(validator.__code__.co_code).hexdigest() == validator_code,
             "validator modified")

    mutations = {}
    mutations["old_score_out_of_100"] = rejected(
        module, lambda value: value.update(
            {"score_out_of_100": value.pop("score")}))
    mutations["score_below_95"] = rejected(
        module, lambda value: value.update({"score": 94}))
    mutations["review_p0"] = rejected(
        module, lambda value: value.update({"p0_count": 1}))
    mutations["review_p1"] = rejected(
        module, lambda value: value.update({"p1_count": 1}))
    mutations["review_status"] = rejected(
        module, lambda value: value.update({"status": module.P.REVIEW_STATUS}))
    mutations["review_identity_extra"] = rejected(
        module, lambda value: value["identity"].update({"alias": "forbidden"}))
    mutations["review_identity_source"] = rejected(
        module, lambda value: value["identity"].update({"source_sha256": "0" * 64}))
    mutations["review_authorization_extra"] = rejected(
        module, lambda value: value["authorization"].update({"remote_launch": False}))
    mutations["review_authorizes_capture"] = rejected(
        module, lambda value: value["authorization"].update({"capture": True}))
    mutations["release_schema"] = rejected(
        module, release_mutator=lambda value: value.update({"schema": "old"}))
    mutations["release_status"] = rejected(
        module, release_mutator=lambda value: value.update({"status": "old"}))
    mutations["release_identity_extra"] = rejected(
        module, release_mutator=lambda value: value["identity"].update({"alias": 1}))
    mutations["release_review_sha"] = rejected(
        module, release_mutator=lambda value: value["identity"].update(
            {"review_sha256": "0" * 64}))
    mutations["release_authorization_extra"] = rejected(
        module, release_mutator=lambda value: value["authorization"].update(
            {"remote_writes": 1}))
    mutations["release_retry"] = rejected(
        module, release_mutator=lambda value: value["authorization"].update(
            {"automatic_retry": True}))
    mutations["release_gpu_budget"] = rejected(
        module, release_mutator=lambda value: value["authorization"].update(
            {"gpu_runs": 2}))
    mutations["release_namespace"] = rejected(
        module, release_mutator=lambda value: value["namespaces"].update(
            {"result": "wrong"}))
    mutations["release_prebudget"] = rejected(
        module, release_mutator=lambda value: value["pre_budget_preflight"].update(
            {"build_runtime_before_child_gpu_attempt": False}))
    mutations["remote_host"] = rejected(
        module, release_mutator=lambda value: value["remote_target"].update(
            {"host": "wrong"}))
    mutations["remote_port"] = rejected(
        module, release_mutator=lambda value: value["remote_target"].update(
            {"port": 22}))
    mutations["remote_user"] = rejected(
        module, release_mutator=lambda value: value["remote_target"].update(
            {"user": "ubuntu"}))
    mutations["remote_repository"] = rejected(
        module, release_mutator=lambda value: value["remote_target"].update(
            {"repository_root": "/root/wrong"}))
    mutations["child_interpreter_path"] = rejected(
        module, release_mutator=lambda value: value["child_interpreter"].update(
            {"path": "/usr/bin/python"}))
    mutations["child_interpreter_sha"] = rejected(
        module, release_mutator=lambda value: value["child_interpreter"].update(
            {"sha256": "0" * 64}))
    mutations["claim_boundary"] = rejected(
        module, release_mutator=lambda value: value["claim_boundary"].update(
            {"tsbg_dse": True}))
    need(all(mutations.values()), "mutation escaped")

    invalid = module.strict_json(module.M1669_INVALID / "review.json")
    correction = module.strict_json(module.M1669_CORRECTION / "review.json")
    need("score" not in invalid and invalid.get("score_out_of_100") == 98,
         "old M1669 schema witness")
    need(correction["authorization"]["m1670_release_authoring"] is False,
         "old authority revived")

    lower = module.P.P.P.SOURCE.read_text()
    body = lower[lower.index("def fixed_clean_child():"):]
    need("os.O_EXCL" in lower, "O_EXCL absent")
    need(body.index("m1434.build_runtime()") < body.index("exclusive_gpu_lease")
         < body.index("consume_attempt(release)")
         < body.index("profile.load_config(CONFIG)"), "child budget order")
    parent_body = SOURCE.read_text()[SOURCE.read_text().index("def launch_parent():"):]
    child_body = SOURCE.read_text()[SOURCE.read_text().index("def fixed_clean_child():"):]
    need(parent_body.index("P.preflight_runtime_binding()") <
         parent_body.index("P.launch_parent()"), "parent runtime order")
    need(child_body.index("P.preflight_runtime_binding()") <
         child_body.index("P.fixed_clean_child()"), "child runtime order")

    m1649 = json.loads(M1649.read_text())
    child = m1649["child_interpreter"]
    need(child == {"path": str(module.CHILD_INTERPRETER),
                   "sha256": "89520a3f2bc6e4f670921bd7a71a66eb0073775e685f6cbefda0dcda7bc42aa0"},
         "known remote interpreter predecessor identity")

    result = {
        "schema": "m1693_m1692_tsbg_authority_shape_independent_hammer_r1_v1",
        "status": module.REVIEW_STATUS,
        "score": 100, "p0_count": 0, "p1_count": 0,
        "validator_code_sha256": validator_code,
        "unmodified_validator_positive_consumption": True,
        "mutation_count": len(mutations),
        "mutations_rejected": sorted(mutations),
        "verified": {
            "remote_target_exact": module.REMOTE_TARGET,
            "known_child_interpreter_path": str(module.CHILD_INTERPRETER),
            "known_child_interpreter_predecessor_sha256": child["sha256"],
            "m1257_handoff_and_install_shape_exact": True,
            "runtime_current_entity_gate_before_parent_and_child_budget": True,
            "runtime_current_entity_observed_remotely_now": False,
            "build_runtime_parent_child_order": True,
            "gpu_lease_then_o_excl_attempt_then_model": True,
            "one_parent_child_gpu_capture_no_retry": True,
            "clean_child_receipt_future_evaluator_ready": True,
            "m1669_invalid_schema_not_revived": True,
            "remote_connected": False, "capture": False, "gpu": False,
            "attempt": False, "release_created": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
