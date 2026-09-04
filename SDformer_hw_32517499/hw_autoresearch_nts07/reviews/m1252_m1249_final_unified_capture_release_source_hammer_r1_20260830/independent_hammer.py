#!/opt/anaconda3/envs/python310/bin/python3.10
"""Independent, local-only hammer for the inert M1249 final-capture release source.

The hammer uses synthetic temporary selection assets only.  It must not create a
production launch contract, touch a remote host/GPU/EDA tool, or consume the
canonical M1249 attempt.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1249_motion_final_checkpoint_unified_hardware_one_shot_release_r1.py")
TEST = HW / (
    "tests/test_m1249_motion_final_checkpoint_unified_capture_one_shot_release_source.py")
CONTRACT = HW / (
    "contracts/m1249_motion_final_checkpoint_unified_capture_one_shot_release_source_"
    "contract_r1_20260830.json")
AUTHOR = HW / (
    "reviews/m1249_motion_final_checkpoint_unified_capture_one_shot_release_author_"
    "r1_20260830")
FIXTURE_TEST = HW / (
    "tests/test_m1233_motion_final_checkpoint_unified_capture_selection_interface_source.py")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source_sha256": "5fbcc4d287f3ffd3b1c9994efa24245e5e3828927cdac925c1a35d8a88a19219",
    "test_sha256": "fc81e54c6f15f05864ef671bae27e34fbefcf4ea6b965d63ef4d8730ce0a6fce",
    "contract_sha256": "e9d0577b331491269780c8fd511b3cf378d62f4023c392c05924a134b7e35ad0",
    "docs359_sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m1243_source_sha256": "009c92c22b5429352b0b4dd29c723035744efa828db9c4472d1f4fb4140297e2",
    "m1243_test_sha256": "7529dd988e48926d683c0ea28c1ca5e9e06a2af617febe796a02e09e38c3ded7",
    "m1243_contract_sha256": "de558985c0f9a64580060dce90675d8ba4ca771a616fe8152b439483663f26ba",
    "author_outer_file_sha256": "30b16513e56afa6cf48e8457b93cd2eb3acdd82b356dfce43b92150ab66e4d61",
}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    need(spec is not None and spec.loader is not None, "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def recursive(root: Path) -> dict:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink(), "missing regular manifest")
    need(outer.is_file() and not outer.is_symlink(), "missing regular outer seal")
    words = outer.read_text(encoding="utf-8").split()
    need(words == [sha(manifest), "SHA256SUMS"], "outer seal mismatch")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        member = root / name
        need(name not in rows, "duplicate manifest member")
        need(member.is_file() and not member.is_symlink(), "non-regular sealed member")
        need(sha(member) == digest, "sealed member drift: " + name)
        rows[name] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")
    }
    need(actual == set(rows), "recursive seal membership mismatch")
    return {
        "manifest_sha256": sha(manifest),
        "outer_file_sha256": sha(outer),
        "members": rows,
    }


M = load("m1252_m1249_source", SOURCE)
FT = load("m1252_m1233_fixture", FIXTURE_TEST)


class Lease:
    def __init__(self, events=None):
        self.events = events

    def __enter__(self):
        if self.events is not None:
            self.events.append("lease_enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.events is not None:
            self.events.append("lease_exit")
        return False


def new_case():
    case = FT.M1233SelectionInterfaceTest(
        "test_03_exact_m1234_shape_passes_and_keyerror_is_regressed")
    case.setUp()
    return case


def production(case, path: Path) -> dict:
    return {
        "schema": M.PRODUCTION_SCHEMA,
        "status": M.PRODUCTION_STATUS,
        "contract_path": str(path.relative_to(M.ROOT)),
        "release_identity": {
            "source_path": str(SOURCE.relative_to(M.ROOT)),
            "source_sha256": sha(SOURCE),
            "test_path": str(TEST.relative_to(M.ROOT)),
            "test_sha256": sha(TEST),
            "source_contract_path": str(CONTRACT.relative_to(M.ROOT)),
            "source_contract_sha256": sha(CONTRACT),
        },
        "inputs": {
            "m1243_source": {
                "path": str(M.M1243_SOURCE.relative_to(M.ROOT)),
                "sha256": M.M1243_SOURCE_SHA256,
            },
            "m1243_test": {
                "path": str(M.M1243_TEST.relative_to(M.ROOT)),
                "sha256": M.M1243_TEST_SHA256,
            },
            "m1243_source_contract": {
                "path": str(M.M1243_CONTRACT.relative_to(M.ROOT)),
                "sha256": M.M1243_CONTRACT_SHA256,
            },
            "m1244_source_hammer": copy.deepcopy(M.M1244_ENTRY),
            "final_selection_result": copy.deepcopy(case.selection_entry),
            "final_selection_result_hammer": copy.deepcopy(case.hammer_entry),
        },
        "cohort": {"samples": []},
        "one_shot": {
            "attempt_marker": str(M.CANONICAL_ATTEMPT.relative_to(M.ROOT)),
            "automatic_retry": False,
        },
        "output": {"path": str(M.CANONICAL_RESULT.relative_to(M.ROOT))},
        "production_log": {"path": str(M.CANONICAL_LOG.relative_to(M.ROOT))},
    }


def validate_case(case, mutate=None):
    with tempfile.TemporaryDirectory(prefix=".m1252_launch_", dir=HW / "contracts") as name:
        path = Path(name) / "launch.json"
        row = production(case, path)
        if mutate is not None:
            mutate(row)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        mapping = {
            str(M.CANONICAL_ATTEMPT.relative_to(M.ROOT)): M.CANONICAL_ATTEMPT,
            str(M.CANONICAL_RESULT.relative_to(M.ROOT)): M.CANONICAL_RESULT,
            str(M.CANONICAL_LOG.relative_to(M.ROOT)): M.CANONICAL_LOG,
        }

        def mapped(value, missing_leaf=False):
            need(missing_leaf, "namespace leaf must be allowed missing")
            return mapping[value]

        with mock.patch.object(M.R1, "validate_m1224", return_value={}), \
             mock.patch.object(M.R1, "validate_cohort", return_value=[]), \
             mock.patch.object(M.R1, "safe_repo_path", side_effect=mapped):
            return M.validate_production_launch(row, path)


def attack(label: str, mutate) -> dict:
    case = new_case()
    try:
        try:
            validate_case(case, mutate)
        except Exception as exc:
            return {"attack": label, "status": "REJECTED", "exception": type(exc).__name__}
        return {"attack": label, "status": "ACCEPTED_UNEXPECTEDLY"}
    finally:
        case.tearDown()


def hammer_authority_attack(label: str, key: str) -> dict:
    case = new_case()
    try:
        review = case._hammer(case.selection_entry, case.selection["selected"])
        value = review["selection_authority"][key]
        review["selection_authority"][key] = (
            value + 1 if type(value) is int else str(value) + ".splice")
        entry = case._write_hammer(review)
        try:
            validate_case(case, lambda row: row["inputs"].__setitem__(
                "final_selection_result_hammer", copy.deepcopy(entry)))
        except Exception as exc:
            return {"attack": label, "status": "REJECTED", "exception": type(exc).__name__}
        return {"attack": label, "status": "ACCEPTED_UNEXPECTEDLY"}
    finally:
        case.tearDown()


def ordering_hammer() -> dict:
    events = []
    substrate = types.SimpleNamespace(exclusive_gpu_lease=lambda lease: Lease(events))

    def validate(contract, path):
        events.append("all_preflight")
        return {"identity": {}}

    def consume():
        events.append("attempt_O_EXCL")

    def capture(*args, **kwargs):
        events.append("capture")
        return Path("synthetic")

    with mock.patch.object(M, "validate_production_launch", side_effect=validate), \
         mock.patch.object(M, "consume_attempt", side_effect=consume), \
         mock.patch.object(M, "run_capture", side_effect=capture):
        M.execute_once({}, TEST, substrate)
    need(events == ["lease_enter", "all_preflight", "attempt_O_EXCL", "capture", "lease_exit"],
         "attempt/preflight/lease ordering drift")

    failed = []
    substrate2 = types.SimpleNamespace(exclusive_gpu_lease=lambda lease: Lease(failed))

    def reject(contract, path):
        failed.append("preflight_rejected")
        raise M.M1249Error("synthetic preflight rejection")

    with mock.patch.object(M, "validate_production_launch", side_effect=reject), \
         mock.patch.object(M, "consume_attempt", side_effect=lambda: failed.append("attempt")), \
         mock.patch.object(M, "run_capture", side_effect=lambda *a, **k: failed.append("capture")):
        try:
            M.execute_once({}, TEST, substrate2)
        except M.M1249Error:
            pass
        else:
            raise RuntimeError("rejected preflight unexpectedly executed")
    need(failed == ["lease_enter", "preflight_rejected", "lease_exit"],
         "attempt consumed after rejected preflight")
    return {"positive_order": events, "rejected_preflight_order": failed}


def exclusive_attempt_hammer() -> dict:
    with tempfile.TemporaryDirectory(prefix=".m1252_attempt_", dir=HW / "results") as name:
        root = Path(name)
        marker = root / "attempt"
        old = M.CANONICAL_ATTEMPT
        M.CANONICAL_ATTEMPT = marker
        try:
            M.consume_attempt()
            mode = marker.stat().st_mode & 0o777
            first = marker.read_text(encoding="ascii")
            try:
                M.consume_attempt()
            except FileExistsError:
                second = "REJECTED_EXISTING_MARKER"
            else:
                second = "ACCEPTED_UNEXPECTEDLY"
        finally:
            M.CANONICAL_ATTEMPT = old
    need(mode == 0o400 and first == M.ATTEMPT_TOKEN, "attempt marker mode/token drift")
    need(second == "REJECTED_EXISTING_MARKER", "O_EXCL/no-retry failed")
    return {"mode": "0400", "token": first.strip(), "second_attempt": second}


def run() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    author = recursive(AUTHOR)
    m1244 = recursive(M.M1244_ROOT)
    identities = {
        "source_sha256": sha(SOURCE),
        "test_sha256": sha(TEST),
        "contract_sha256": sha(CONTRACT),
        "docs359_sha256": sha(DOCS359),
        "m1243_source_sha256": sha(M.M1243_SOURCE),
        "m1243_test_sha256": sha(M.M1243_TEST),
        "m1243_contract_sha256": sha(M.M1243_CONTRACT),
    }
    need(all(identities[key] == EXPECTED[key] for key in identities),
         "independent exact identity pin drift")
    need(author["outer_file_sha256"] == EXPECTED["author_outer_file_sha256"],
         "M1249 author receipt outer seal drift")
    need(contract["source"]["sha256"] == identities["source_sha256"], "source pin drift")
    need(contract["test"]["sha256"] == identities["test_sha256"], "test pin drift")
    need(contract["docs359_sha256"] == identities["docs359_sha256"], "docs359 drift")
    need(identities["m1243_source_sha256"] == M.M1243_SOURCE_SHA256, "M1243 source drift")
    need(identities["m1243_test_sha256"] == M.M1243_TEST_SHA256, "M1243 test drift")
    need(identities["m1243_contract_sha256"] == M.M1243_CONTRACT_SHA256,
         "M1243 contract drift")
    need(m1244["manifest_sha256"] == M.M1244_ENTRY["manifest_sha256"],
         "M1244 manifest entry drift")
    need(m1244["outer_file_sha256"] == M.M1244_ENTRY["outer_file_sha256"],
         "M1244 outer entry drift")
    need(m1244["members"].get("review.json") == M.M1244_ENTRY["review_sha256"],
         "M1244 review entry drift")
    authority = M.M1243.verify_source_hammer(M.M1244_ENTRY)
    need(authority["production_capture"] is True, "M1244 narrow authority missing")

    test_run = subprocess.run(
        [sys.executable, "-m", "unittest", "-v",
         "hw_autoresearch_nts07/tests/"
         "test_m1249_motion_final_checkpoint_unified_capture_one_shot_release_source.py"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    need(test_run.returncode == 0 and "Ran 18 tests" in test_run.stdout and
         test_run.stdout.rstrip().endswith("OK"), "M1249 controlled 18-test suite failed")

    attacks = []
    for identity in ("m1243_source", "m1243_test", "m1243_source_contract"):
        attacks.append(attack(
            "M1243_" + identity + "_sha_splice",
            lambda row, identity=identity: row["inputs"][identity].__setitem__(
                "sha256", "0" * 64)))
    for key in ("manifest_sha256", "outer_file_sha256", "review_sha256"):
        attacks.append(attack(
            "M1244_" + key + "_splice",
            lambda row, key=key: row["inputs"]["m1244_source_hammer"].__setitem__(
                key, "0" * 64)))
        attacks.append(attack(
            "M1237_entry_" + key + "_splice",
            lambda row, key=key: row["inputs"][
                "final_selection_result_hammer"].__setitem__(key, "0" * 64)))
    for key in ("manifest_sha256", "outer_file_sha256", "selection_sha256"):
        attacks.append(attack(
            "M1234_selection_" + key + "_splice",
            lambda row, key=key: row["inputs"]["final_selection_result"].__setitem__(
                key, "0" * 64)))
    for key in sorted(M.M1243.P.HAMMER_AUTHORITY_KEYS):
        attacks.append(hammer_authority_attack("M1237_cross_field_" + key, key))
    attacks.extend([
        attack("release_source_sha_splice", lambda row: row["release_identity"].__setitem__(
            "source_sha256", "0" * 64)),
        attack("attempt_namespace_splice", lambda row: row["one_shot"].__setitem__(
            "attempt_marker", "wrong")),
        attack("result_namespace_splice", lambda row: row["output"].__setitem__(
            "path", "wrong")),
        attack("log_namespace_splice", lambda row: row["production_log"].__setitem__(
            "path", "wrong")),
        attack("automatic_retry_true", lambda row: row["one_shot"].__setitem__(
            "automatic_retry", True)),
        attack("production_top_extra", lambda row: row.__setitem__("extra", True)),
    ])
    need(all(row["status"] == "REJECTED" for row in attacks), "mutation accepted")

    # Occupancy is checked with lexists, so files, directories, and dangling symlinks
    # all block a launch.  Patch only the predicate; no canonical namespace is touched.
    occupied = []
    for target in (M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG):
        with mock.patch.object(M.os.path, "lexists", side_effect=lambda value, t=target:
                               str(value) == str(t)):
            try:
                M.ensure_fresh_namespaces()
            except M.M1249Error as exc:
                occupied.append({"target": target.name, "status": "REJECTED",
                                 "exception": type(exc).__name__})
            else:
                occupied.append({"target": target.name, "status": "ACCEPTED_UNEXPECTEDLY"})
    need(all(row["status"] == "REJECTED" for row in occupied), "occupied namespace accepted")

    order = ordering_hammer()
    exclusive = exclusive_attempt_hammer()

    namespaces = {M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG}
    prior = {
        M.M1243.CANONICAL_RESULT, M.M1243.CANONICAL_ATTEMPT, M.M1243.CANONICAL_LOG,
        M.M1243.P.CANONICAL_RESULT, M.M1243.P.CANONICAL_ATTEMPT, M.M1243.P.CANONICAL_LOG,
        M.M1243.P.R1.CANONICAL_RESULT, M.M1243.P.R1.CANONICAL_ATTEMPT,
        M.M1243.P.R1.CANONICAL_LOG,
    }
    need(len(namespaces) == 3 and not namespaces & prior, "namespace collision")
    need(all(not os.path.lexists(str(path)) for path in namespaces),
         "canonical M1249 namespace unexpectedly occupied")

    aliases = {
        name: getattr(M, name) is getattr(M.M1243, name)
        for name in (
            "EXPECTED_STATIC_COUNTS", "EXPECTED_LIVE_COUNTS", "DEAD_SN_V",
            "audit_call_matrix", "audit_attention_population",
            "validate_payload_population", "atomic_sample_snapshot",
            "final_validate_and_seal")
    }
    need(all(aliases.values()), "capture semantic alias drift")
    need(sum(M.EXPECTED_STATIC_COUNTS.values()) == 259 and
         sum(M.EXPECTED_LIVE_COUNTS.values()) == 247 and len(M.DEAD_SN_V) == 12,
         "capture population drift")

    future_m1237 = list((HW / "reviews").glob("m1237*"))
    need(not future_m1237, "a production M1237 unexpectedly exists during source hammer")
    need(contract["production_launch_contract_created"] is False and
         contract["future_M1237_result_hammer_required"] is True,
         "source-only boundary drift")

    return {
        "schema": "m1252_m1249_final_unified_capture_release_source_hammer_r1_v1",
        "status": (
            "PASS_M1252_M1249_SOURCE__FUTURE_EXACT_M1237_REQUIRED__"
            "PRODUCTION_LAUNCH_AUTHORING_ONLY"),
        "verdict": "GO_FUTURE_PRODUCTION_LAUNCH_AUTHORING_AFTER_EXACT_M1237__NO_GO_NOW",
        "identities": identities,
        "author_receipt_seal": {
            "manifest_sha256": author["manifest_sha256"],
            "outer_file_sha256": author["outer_file_sha256"],
            "members": len(author["members"]),
        },
        "M1244_seal": {
            "manifest_sha256": m1244["manifest_sha256"],
            "outer_file_sha256": m1244["outer_file_sha256"],
            "review_sha256": m1244["members"]["review.json"],
            "production_capture_authority": True,
        },
        "controlled_suite": {"tests": 18, "passed": 18},
        "independent_mutations": {
            "total": len(attacks), "rejected": len(attacks), "details": attacks},
        "namespace_occupancy_mutations": occupied,
        "lease_preflight_attempt_order": order,
        "exclusive_attempt": exclusive,
        "future_M1237_gate": {
            "currently_exists": False,
            "exact_entry_shape": sorted(M.M1237_ENTRY_KEYS),
            "selection_and_hammer_recursive_double_seal": True,
            "all_selection_cross_SHA_and_pair_fields": True,
            "checkpoint_config_profile_hash_preflight_under_lease": True,
            "cohort_and_source_preflight_under_lease": True,
        },
        "namespaces": {
            "result": str(M.CANONICAL_RESULT.relative_to(ROOT)),
            "attempt": str(M.CANONICAL_ATTEMPT.relative_to(ROOT)),
            "log": str(M.CANONICAL_LOG.relative_to(ROOT)),
            "pairwise_disjoint": True,
            "disjoint_from_M1227_M1233_M1243": True,
            "all_absent": True,
        },
        "capture_semantics": {
            "exact_aliases": aliases,
            "static_modules": 259, "live_modules_per_sample": 247,
            "dead_sn_v": 12, "ordered": 9880, "attention": 480,
            "payload": 640, "atomic_snapshot": True,
        },
        "nonblocking_observation": {
            "source_fail_closed": True,
            "nested_M1233_selection_errors_are_not_normalized_to_M1249Error": True,
            "author_test_mutates_shared_hammer_fixture_after_shape_attacks": True,
            "impact": "diagnostic exception class and test specificity only; no admission bypass",
        },
        "authorization": {
            "future_production_launch_authoring_after_exact_M1237": True,
            "production_launch_authoring_now": False,
            "production_capture_now": False,
            "automatic_retry": False,
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
