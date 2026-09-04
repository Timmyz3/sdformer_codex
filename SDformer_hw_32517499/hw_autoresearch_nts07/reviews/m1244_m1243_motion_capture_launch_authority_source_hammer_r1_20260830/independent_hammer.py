#!/opt/anaconda3/envs/python310/bin/python3.10
"""Independent local-only hammer for the M1243 production-capture launch gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1243_motion_final_checkpoint_unified_hardware_launch_authority_r3.py")
TEST = HW / "tests/test_m1243_motion_capture_launch_authority_successor_source.py"
CONTRACT = HW / (
    "contracts/m1243_motion_capture_launch_authority_successor_source_contract_r1_20260830.json")
AUTHOR = HW / "reviews/m1243_motion_capture_launch_authority_successor_author_r1_20260830"
OLD_TEST = HW / (
    "tests/test_m1233_motion_final_checkpoint_unified_capture_selection_interface_source.py")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_recursive(root: Path) -> dict:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    expected, name = outer.read_text(encoding="utf-8").split()
    assert name == "SHA256SUMS" and expected == sha(manifest)
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        member = root / name
        assert member.is_file() and not member.is_symlink()
        assert name not in rows and sha(member) == digest
        rows[name] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")
    }
    assert actual == set(rows)
    return {
        "manifest_sha256": sha(manifest),
        "outer_file_sha256": sha(outer),
        "members": rows,
    }


M = load("m1244_m1243_source", SOURCE)
T = load("m1244_m1233_fixture", OLD_TEST)


def source_authority() -> dict:
    return {
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha(SOURCE),
        "contract_path": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha(CONTRACT),
        "test_path": str(TEST.relative_to(ROOT)),
        "test_sha256": sha(TEST),
    }


def hammer_review() -> dict:
    return {
        "schema": "m1244_m1243_motion_capture_launch_authority_source_hammer_r1_v1",
        "status": (
            "PASS_M1244_M1243_CAPTURE_LAUNCH_AUTHORITY__"
            "PRODUCTION_CAPTURE_RELEASE_AUTHORING_ALLOWED"),
        "source_authority": source_authority(),
        "independence": {"different_author": True},
        "authorization": {"production_capture": True},
    }


class Fixture:
    def __init__(self, review=None):
        self.tmp = tempfile.TemporaryDirectory(prefix=".m1244_hammer_", dir=HW / "reviews")
        self.root = Path(self.tmp.name)
        self.review = hammer_review() if review is None else review
        (self.root / "review.json").write_text(
            json.dumps(self.review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        M.R1.write_double_seal(self.root)
        self.refresh()

    def refresh(self):
        self.entry = {
            "path": str(self.root.relative_to(ROOT)),
            "manifest_sha256": sha(self.root / "SHA256SUMS"),
            "outer_file_sha256": sha(self.root / "SHA256SUMS.seal.sha256"),
            "review_sha256": sha(self.root / "review.json"),
        }

    def cleanup(self):
        self.tmp.cleanup()


def selection_fixture():
    case = T.M1233SelectionInterfaceTest(
        "test_03_exact_m1234_shape_passes_and_keyerror_is_regressed")
    case.setUp()
    return case


def expect_rejected(label: str, operation) -> dict:
    try:
        operation()
    except Exception as exc:
        return {"attack": label, "status": "REJECTED", "exception": type(exc).__name__}
    return {"attack": label, "status": "ACCEPTED_UNEXPECTEDLY"}


def verify_mutated_review(label: str, mutate) -> dict:
    value = hammer_review()
    mutate(value)
    fixture = Fixture(value)
    try:
        return expect_rejected(label, lambda: M.verify_source_hammer(fixture.entry))
    finally:
        fixture.cleanup()


def launch(case, hammer_entry=None, include_hammer=True, mutate=None):
    with tempfile.TemporaryDirectory(prefix=".m1244_launch_", dir=HW / "contracts") as name:
        path = Path(name) / "launch.json"
        inputs = {
            "launcher": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)},
            "source_contract": {
                "path": str(CONTRACT.relative_to(ROOT)), "sha256": sha(CONTRACT)},
            "final_selection_result": case.selection_entry,
            "final_selection_result_hammer": case.hammer_entry,
        }
        if include_hammer:
            inputs["source_hammer"] = hammer_entry
        contract = {
            "schema": M.LAUNCH_SCHEMA,
            "status": M.LAUNCH_STATUS,
            "contract_path": str(path.relative_to(ROOT)),
            "inputs": inputs,
            "cohort": {"samples": []},
            "one_shot": {"attempt_marker": "attempt"},
            "output": {"path": "result"},
            "production_log": {"path": "log"},
        }
        if mutate is not None:
            mutate(contract)
        path.write_text(json.dumps(contract) + "\n", encoding="utf-8")

        def mapped(value, missing_leaf=False):
            assert missing_leaf
            return {
                "attempt": M.CANONICAL_ATTEMPT,
                "result": M.CANONICAL_RESULT,
                "log": M.CANONICAL_LOG,
            }[value]

        with mock.patch.object(M.R1, "validate_m1224", return_value={}), \
             mock.patch.object(M.R1, "validate_cohort", return_value=[]), \
             mock.patch.object(M.R1, "safe_repo_path", side_effect=mapped):
            return M.validate_launch_contract(contract, path)


def run() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    author_seal = verify_recursive(AUTHOR)
    identities = {
        "source_sha256": sha(SOURCE),
        "contract_sha256": sha(CONTRACT),
        "test_sha256": sha(TEST),
        "docs359_sha256": sha(DOCS359),
    }
    assert identities["source_sha256"] == contract["source"]["sha256"]
    assert identities["test_sha256"] == contract["test"]["sha256"]
    assert identities["docs359_sha256"] == contract["docs359_sha256"]
    assert author_seal["members"]["review.json"] == sha(AUTHOR / "review.json")

    fixture = Fixture()
    case = selection_fixture()
    try:
        positive = M.verify_source_hammer(fixture.entry)
        binding = launch(case, fixture.entry)
        assert positive["production_capture"] is True
        assert binding["identity"]["source_hammer"] == positive
        assert binding["identity"]["candidate_id"] == "resume_ep32"

        attacks = []
        attacks.append(expect_rejected(
            "missing_source_hammer", lambda: launch(case, include_hammer=False)))
        attacks.append(expect_rejected(
            "source_hammer_entry_extra", lambda: M.verify_source_hammer(
                dict(fixture.entry, extra=True))))
        missing_entry = dict(fixture.entry)
        del missing_entry["review_sha256"]
        attacks.append(expect_rejected(
            "source_hammer_entry_missing", lambda: M.verify_source_hammer(missing_entry)))
        for key in ("manifest_sha256", "outer_file_sha256", "review_sha256"):
            attacks.append(expect_rejected(
                key + "_drift", lambda key=key: M.verify_source_hammer(
                    dict(fixture.entry, **{key: "0" * 64}))))

        attacks.extend([
            verify_mutated_review("schema_drift", lambda row: row.__setitem__("schema", "wrong")),
            verify_mutated_review("status_drift", lambda row: row.__setitem__("status", "wrong")),
        ])
        for key in sorted(source_authority()):
            attacks.append(verify_mutated_review(
                "cross_sha_or_path_splice_" + key,
                lambda row, key=key: row["source_authority"].__setitem__(
                    key, str(row["source_authority"][key]) + ".splice")))
        attacks.extend([
            verify_mutated_review(
                "same_author_mutation",
                lambda row: row.__setitem__("independence", {"different_author": False})),
            verify_mutated_review(
                "author_shape_extra",
                lambda row: row.__setitem__(
                    "independence", {"different_author": True, "same_author": False})),
            verify_mutated_review(
                "production_capture_false",
                lambda row: row.__setitem__("authorization", {"production_capture": False})),
            verify_mutated_review(
                "authority_extra",
                lambda row: row.__setitem__(
                    "authorization", {"production_capture": True, "release": True})),
            expect_rejected(
                "launcher_source_sha_splice",
                lambda: launch(case, fixture.entry, mutate=lambda row: row["inputs"][
                    "launcher"].__setitem__("sha256", "0" * 64))),
            expect_rejected(
                "source_contract_sha_splice",
                lambda: launch(case, fixture.entry, mutate=lambda row: row["inputs"][
                    "source_contract"].__setitem__("sha256", "0" * 64))),
        ])

        deletion = Fixture()
        try:
            (deletion.root / "review.json").unlink()
            attacks.append(expect_rejected(
                "sealed_member_deletion", lambda: M.verify_source_hammer(deletion.entry)))
        finally:
            deletion.cleanup()
        decoy = Fixture()
        try:
            (decoy.root / "unsealed_decoy.json").write_text("{}\n", encoding="utf-8")
            attacks.append(expect_rejected(
                "unsealed_decoy_member", lambda: M.verify_source_hammer(decoy.entry)))
        finally:
            decoy.cleanup()
        author_splice = Fixture(
            json.loads((AUTHOR / "review.json").read_text(encoding="utf-8")))
        try:
            attacks.append(expect_rejected(
                "author_review_spliced_as_hammer",
                lambda: M.verify_source_hammer(author_splice.entry)))
        finally:
            author_splice.cleanup()

        assert all(row["status"] == "REJECTED" for row in attacks), attacks
    finally:
        case.tearDown()
        fixture.cleanup()

    source_only = expect_rejected(
        "source_only_contract_as_launch",
        lambda: M.validate_launch_contract(contract, CONTRACT))
    assert source_only["status"] == "REJECTED"

    aliases = {
        name: getattr(M, name) is getattr(M.P, name)
        for name in (
            "validate_final_selection", "EXPECTED_STATIC_COUNTS", "EXPECTED_LIVE_COUNTS",
            "DEAD_SN_V", "audit_call_matrix", "audit_attention_population",
            "validate_payload_population", "atomic_sample_snapshot", "final_validate_and_seal")
    }
    assert all(aliases.values())
    assert sum(M.EXPECTED_STATIC_COUNTS.values()) == 259
    assert sum(M.EXPECTED_LIVE_COUNTS.values()) == 247
    assert len(M.DEAD_SN_V) == 12
    assert M.ALLOWED_SELECTION_SCHEMA == contract["frozen_interfaces"]["selection_schema"]
    assert M.ALLOWED_SELECTION_STATUS == contract["frozen_interfaces"]["selection_status"]
    assert M.SELECTION_RESULT_HAMMER_SCHEMA == contract["frozen_interfaces"][
        "selection_result_hammer_schema"]
    assert M.SELECTION_RESULT_HAMMER_STATUS == contract["frozen_interfaces"][
        "selection_result_hammer_status"]
    assert M.CANONICAL_RESULT != M.P.CANONICAL_RESULT
    assert not M.CANONICAL_RESULT.exists()
    assert not M.CANONICAL_ATTEMPT.exists()
    assert not M.CANONICAL_LOG.exists()

    return {
        "schema": "m1244_m1243_motion_capture_launch_authority_source_hammer_r1_v1",
        "status": (
            "PASS_M1244_M1243_CAPTURE_LAUNCH_AUTHORITY__"
            "PRODUCTION_CAPTURE_RELEASE_AUTHORING_ALLOWED"),
        "identities": identities,
        "author_seal": {
            "manifest_sha256": author_seal["manifest_sha256"],
            "outer_file_sha256": author_seal["outer_file_sha256"],
            "members": len(author_seal["members"]),
        },
        "M1240_P0_closure": {
            "validate_launch_consumes_hammer": True,
            "positive_hammer_consumed_into_binding": True,
            "recursive_double_seal": True,
            "source_contract_test_cross_SHA": True,
            "different_author": True,
            "production_capture_authority_exact_true": True,
        },
        "frozen_aliases": aliases,
        "frozen_population": {
            "static": 259, "live": 247, "dead_sn_v": 12,
            "ordered": 9880, "attention": 480, "payload": 640,
            "atomic_snapshot": True,
        },
        "selection_binding": {
            "schema": M.ALLOWED_SELECTION_SCHEMA,
            "status": M.ALLOWED_SELECTION_STATUS,
            "result_hammer_schema": M.SELECTION_RESULT_HAMMER_SCHEMA,
            "result_hammer_status": M.SELECTION_RESULT_HAMMER_STATUS,
            "synthetic_candidate": "resume_ep32",
        },
        "fresh_namespace": {
            "result": str(M.CANONICAL_RESULT.relative_to(ROOT)),
            "attempt": str(M.CANONICAL_ATTEMPT.relative_to(ROOT)),
            "log": str(M.CANONICAL_LOG.relative_to(ROOT)),
            "all_absent": True,
        },
        "independent_mutations": {
            "total": len(attacks), "rejected": len(attacks), "details": attacks,
        },
        "authorization": {
            "source_hammer_pass": True,
            "production_capture_release_authoring": True,
            "production_capture_execution": False,
            "separate_one_shot_release_required": True,
        },
        "execution": {
            "remote": False, "gpu": False, "checkpoint": False, "capture": False,
            "release": False, "eda": False, "production_paths": False,
        },
        "claim_boundary": {
            "source_hammer": True, "capture_complete": False, "paper_result": False,
            "cycles": False, "speedup": False, "energy": False, "ppa": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
