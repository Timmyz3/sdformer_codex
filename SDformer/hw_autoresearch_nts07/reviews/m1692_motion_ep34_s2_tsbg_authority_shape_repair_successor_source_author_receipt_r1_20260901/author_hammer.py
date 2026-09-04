#!/usr/bin/env python3
"""Read-only author hammer for M1692 additive TSBG source."""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "successor_r1.py")
TEST = HW / (
    "tests/test_m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "successor_source.py")
CONTRACT = HW / (
    "contracts/m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "successor_source_contract_r1_20260901.json")


class HammerError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise HammerError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def no_duplicates(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=no_duplicates)
    require(type(value) is dict, "JSON root must be object")
    return value


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            "cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_contract_seal():
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    require(CONTRACT.is_file() and not CONTRACT.is_symlink(),
            "contract absent/symlink")
    require(sidecar.is_file() and not sidecar.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "contract double seal absent/symlink")
    require(sidecar.read_text(encoding="ascii") ==
            sha256(CONTRACT) + "  " + CONTRACT.name + "\n",
            "contract sidecar mismatch")
    require(outer.read_text(encoding="ascii") ==
            sha256(sidecar) + "  " + sidecar.name + "\n",
            "contract outer mismatch")
    return {"payload_sha256": sha256(CONTRACT),
            "sidecar_sha256": sha256(sidecar),
            "outer_file_sha256": sha256(outer)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = load(SOURCE, "m1692_author_source")
    test = load(TEST, "m1692_author_test")
    contract_seal = verify_contract_seal()
    contract = strict_json(CONTRACT)
    require(contract["source"]["sha256"] == sha256(SOURCE) and
            contract["test"]["sha256"] == sha256(TEST),
            "source/test contract hash drift")
    require(contract["remote_target"] == source.REMOTE_TARGET and
            contract["child_interpreter_path"] ==
                str(source.CHILD_INTERPRETER),
            "remote/interpreter contract drift")
    require(not os.path.lexists(str(source.FUTURE_REVIEW)) and
            not os.path.lexists(str(source.FUTURE_RELEASE)) and
            not os.path.lexists(str(Path(str(source.FUTURE_RELEASE) +
                                         ".sha256"))) and
            not os.path.lexists(str(Path(str(source.FUTURE_RELEASE) +
                                         ".sha256.seal.sha256"))),
            "future M1693/M1694 authority is not absent")

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    require(result.testsRun == 21 and result.wasSuccessful(),
            "M1692 source regression failed: " + stream.getvalue())
    self_check = source.source_self_check()
    require(self_check["status"] ==
            "PASS_M1692_SOURCE_SELF_CHECK__AUTHORITY_SHAPE_REPAIRED__NO_CAPTURE" and
            self_check["gpu_runs"] == 0 and
            self_check["capture_runs"] == 0 and
            self_check["attempt_writes"] == 0 and
            self_check["remote_connected"] is False,
            "M1692 self check crossed source-only boundary")

    output = {
        "schema": "m1692_tsbg_authority_shape_repair_source_author_receipt_r1_v1",
        "date_cst": "2026-09-01",
        "status": (
            "SOURCE_ONLY_PASS_M1692_TSBG_AUTHORITY_SHAPE_REPAIR__"
            "DIFFERENT_AUTHOR_M1693_REQUIRED__NO_CAPTURE"),
        "score": 100,
        "p0_count": 0,
        "p1_count": 0,
        "identity": {
            "source_sha256": sha256(SOURCE),
            "test_sha256": sha256(TEST),
            "source_contract_sha256": sha256(CONTRACT),
            "source_contract_sidecar_sha256":
                contract_seal["sidecar_sha256"],
            "source_contract_outer_file_sha256":
                contract_seal["outer_file_sha256"],
            "m1668_source_sha256": source.M1668_SOURCE_SHA256,
            "m1669_invalid_review_sha256":
                source.M1669_INVALID_REVIEW_SHA256,
            "m1669_correction_review_sha256":
                source.M1669_CORRECTION_REVIEW_SHA256,
            "selection_identity_sha256": source.SELECTION_IDENTITY_SHA256,
            "runtime_tar_sha256": source.RUNTIME_TAR_SHA256,
            "checkpoint_sha256": source.CHECKPOINT_SHA256,
            "config_sha256": source.CONFIG_SHA256,
            "profile_sha256": source.PROFILE_SHA256,
            "docs359_sha256": source.DOCS359_SHA256,
        },
        "validator_in_loop": {
            "tests_passed": result.testsRun,
            "positive_exact_review_release": True,
            "review_score_identity_authorization_negative": 3,
            "remote_host_port_user_repository_negative": 4,
            "child_interpreter_path_sha_negative": 2,
            "release_authorization_and_prebudget_negative": 2,
            "parent_child_runtime_order_tests": 2,
            "lower_gpu_attempt_order_test": 1,
            "capture_receipt_evaluator_identity_test": 1,
        },
        "preserved_chain": {
            "m1668_runtime_handoff": True,
            "current_checkpoint_config_profile_entities": True,
            "build_runtime_before_parent": True,
            "build_runtime_before_child_gpu_attempt": True,
            "exclusive_gpu_lease": True,
            "o_excl_attempt_before_checkpoint_model": True,
            "automatic_retry": False,
        },
        "future_authority": {
            "review": str(source.FUTURE_REVIEW.relative_to(source.ROOT)),
            "release": str(source.FUTURE_RELEASE.relative_to(source.ROOT)),
            "present": False,
        },
        "capture_consumer": self_check["capture_consumer"],
        "remote_target": dict(source.REMOTE_TARGET),
        "child_interpreter_path": str(source.CHILD_INTERPRETER),
        "authorization": {
            "different_author_m1693_review": True,
            "m1694_release_authoring": False,
            "remote_launch": False,
            "capture": False,
            "gpu": False,
            "attempt_write": False,
            "automatic_retry": False,
        },
        "claim_boundary": {
            "source_only": True,
            "aee": False,
            "cycles": False,
            "traffic": False,
            "energy": False,
            "speedup": False,
            "rtl": False,
            "eda": False,
            "paper_result": False,
        },
        "execution": {
            "remote_connections": 0,
            "remote_writes": 0,
            "checkpoint_loads": 0,
            "parent_processes": 0,
            "child_processes": 0,
            "capture_runs": 0,
            "gpu_runs": 0,
            "attempt_writes": 0,
            "eda_runs": 0,
            "git_commit": False,
            "git_push": False,
        },
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output["status"])


if __name__ == "__main__":
    main()
