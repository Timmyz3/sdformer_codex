#!/usr/bin/env python3
"""Independent local-only M1314 hammer over the frozen M1313 launch author package."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json"
CHECKER = HW / "system_simulator/scripts/check_m1313_motion_ep34_final_unified_capture_production_launch.py"
TEST = HW / "tests/test_m1313_motion_ep34_final_unified_capture_production_launch.py"
AUTHOR = HW / "reviews/m1313_motion_ep34_final_unified_capture_production_launch_author_r1_20260831"
M1182 = HW / "contracts/m1182_m1180_motion_ep29_unified_capture_launch_release_r1_20260830.json"
M1210 = HW / "contracts/m1210_m1208_motion_ep29_unified_capture_launch_release_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "contract": "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda",
    "checker": "b1241b66d589281e29dcedc5a535a2c3cee4bc2f0dee18fb1774e8b5efcd35f4",
    "test": "908166d9cd214bff220881b3a14f602e4990247484a81a3af63f030946413abe",
    "author_manifest": "6ba2bbb32d611f9a04995315289549798cb8d26342330e56a1bf1e72c9d827c8",
    "author_outer_file": "2d0f27bc06b42a972e1035dcfe346509bea04406d7690bd081a0e727f3410073",
    "author_receipt": "dca15e9a207d831dfd5c4d45ebabf25815bfddb70a382a9c0c57ae61bfeefd3e",
    "cohort_compact": "e9e6443c25a2f3d7ee6994b8c708eaecec7845f70dd920a132adc9276744745f",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    mode = path.lstat().st_mode
    need(stat.S_ISREG(mode) and not path.is_symlink(), label + " must be regular")


def strict(path: Path) -> dict:
    def reject(value: str):
        raise RuntimeError("non-finite JSON: " + value)

    def pairs(rows):
        out = {}
        for key, value in rows:
            need(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=reject)
    need(isinstance(value, dict), "JSON root is not object")
    return value


def compact_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    need(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_author_double_seal() -> dict:
    manifest = AUTHOR / "SHA256SUMS"
    outer = AUTHOR / "SHA256SUMS.seal.sha256"
    regular(manifest, "M1313 author manifest")
    regular(outer, "M1313 author outer")
    need(sha(manifest) == EXPECTED["author_manifest"], "author manifest SHA drift")
    need(sha(outer) == EXPECTED["author_outer_file"], "author outer-file SHA drift")
    need(outer.read_text(encoding="ascii").split() ==
         [EXPECTED["author_manifest"], "SHA256SUMS"], "author outer content drift")
    rows = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        need(name not in rows and "/" not in name, "unsafe/duplicate author member")
        member = AUTHOR / name
        regular(member, "author member " + name)
        need(sha(member) == digest, "author member SHA drift: " + name)
        rows[name] = digest
    actual = {path.name for path in AUTHOR.iterdir() if path.is_file() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == set(rows), "author sealed population mismatch")
    need(rows.get("author_receipt.json") == EXPECTED["author_receipt"],
         "author receipt member mismatch")
    return rows


def run_author_tests() -> tuple[int, str]:
    module = load("m1314_m1313_author_tests", TEST)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    need(result.wasSuccessful(), "M1313 author tests failed\n" + stream.getvalue())
    need(result.testsRun == 15, "M1313 author test count is not 15")
    return result.testsRun, stream.getvalue()


def rejected(checker, contract: dict, label: str, mutate=None, occupied=None) -> dict:
    value = copy.deepcopy(contract)
    if mutate is not None:
        mutate(value)
    exists = (lambda path: path == occupied) if occupied is not None else (lambda _path: False)
    try:
        checker.validate_contract(value, check_sample_bytes=False, namespace_exists=exists)
    except Exception as exc:
        return {"attack": label, "status": "REJECTED", "exception": type(exc).__name__}
    raise RuntimeError("attack accepted: " + label)


def run() -> dict:
    for path, expected, label in (
        (CONTRACT, EXPECTED["contract"], "contract"),
        (CHECKER, EXPECTED["checker"], "checker"),
        (TEST, EXPECTED["test"], "test"),
        (DOCS359, EXPECTED["docs359"], "docs359"),
    ):
        regular(path, label)
        need(sha(path) == expected, label + " SHA drift")
    author_rows = verify_author_double_seal()
    receipt = strict(AUTHOR / "author_receipt.json")
    need(receipt["production_launch"]["sha256"] == EXPECTED["contract"],
         "receipt contract identity drift")
    need(receipt["author_checker"]["sha256"] == EXPECTED["checker"],
         "receipt checker identity drift")
    need(receipt["author_tests"]["sha256"] == EXPECTED["test"],
         "receipt test identity drift")
    need(receipt["authorization"]["production_capture_executed"] is False and
         receipt["authorization"]["remote_gpu_used"] is False and
         receipt["authorization"]["eda_used"] is False,
         "author execution boundary drift")

    tests, test_log = run_author_tests()
    checker = load("m1314_m1313_checker", CHECKER)
    contract = strict(CONTRACT)
    positive = checker.validate_contract(contract, check_sample_bytes=True,
                                         namespace_exists=lambda _path: False)
    samples = contract["cohort"]["samples"]
    need(samples == strict(M1182)["cohort"]["samples"] ==
         strict(M1210)["cohort"]["samples"], "independent cohort authority mismatch")
    need(compact_sha(samples) == EXPECTED["cohort_compact"], "cohort compact SHA drift")
    byte_total = 0
    row_digest = hashlib.sha256()
    for row in samples:
        path = ROOT / row["path"]
        regular(path, "cohort sample")
        actual = sha(path)
        need(path.stat().st_size == row["bytes"] == 12288128, "cohort size drift")
        need(actual == row["sha256"], "cohort sample SHA drift")
        byte_total += row["bytes"]
        row_digest.update((row["path"] + "\0" + str(row["bytes"]) + "\0" + actual + "\n").encode())
    need(len(samples) == 40 and byte_total == 491525120, "cohort population/bytes drift")

    staged = (
        "hw_autoresearch_nts07/system_handoff/incoming/m1306_remote_selection_result_20260830/"
        "hw_autoresearch_nts07/results/m1257_motion_cross_run_final_checkpoint_selection_r5_20260830")
    attacks = [
        rejected(checker, contract, "canonical_path_to_staged", lambda row:
                 row["inputs"]["final_selection_result"].__setitem__("result_path", staged)),
        rejected(checker, contract, "M1312_path_splice", lambda row:
                 row["inputs"]["final_selection_result_hammer"].__setitem__("path", "wrong")),
        rejected(checker, contract, "M1312_manifest_splice", lambda row:
                 row["inputs"]["final_selection_result_hammer"].__setitem__("manifest_sha256", "0" * 64)),
        rejected(checker, contract, "M1312_outer_splice", lambda row:
                 row["inputs"]["final_selection_result_hammer"].__setitem__("outer_file_sha256", "0" * 64)),
        rejected(checker, contract, "M1312_review_splice", lambda row:
                 row["inputs"]["final_selection_result_hammer"].__setitem__("review_sha256", "0" * 64)),
        rejected(checker, contract, "cohort_sha_splice", lambda row:
                 row["cohort"]["samples"][0].__setitem__("sha256", "0" * 64)),
        rejected(checker, contract, "cohort_order_swap", lambda row:
                 row["cohort"]["samples"].__setitem__(slice(0, 2),
                 list(reversed(row["cohort"]["samples"][0:2])))),
        rejected(checker, contract, "cohort_population_drop", lambda row:
                 row["cohort"]["samples"].pop()),
        rejected(checker, contract, "attempt_namespace_splice", lambda row:
                 row["one_shot"].__setitem__("attempt_marker", "wrong")),
        rejected(checker, contract, "result_namespace_splice", lambda row:
                 row["output"].__setitem__("path", "wrong")),
        rejected(checker, contract, "log_namespace_splice", lambda row:
                 row["production_log"].__setitem__("path", "wrong")),
        rejected(checker, contract, "automatic_retry_true", lambda row:
                 row["one_shot"].__setitem__("automatic_retry", True)),
        rejected(checker, contract, "occupied_result", occupied=str(checker.RESULT)),
        rejected(checker, contract, "occupied_attempt", occupied=str(checker.ATTEMPT)),
        rejected(checker, contract, "occupied_log", occupied=str(checker.LOG)),
    ]
    need(len(attacks) == 15 and all(row["status"] == "REJECTED" for row in attacks),
         "mutation population failure")

    checker_text = CHECKER.read_text(encoding="utf-8")
    for token in ("import subprocess", "import torch", "paramiko", "ssh ", "dc_shell",
                  "vcs -full64", "execute_once(", "run_capture(", "write_text", "write_bytes"):
        need(token not in checker_text, "checker contains execution/write token: " + token)
    need(positive["remote_gpu_capture_executed"] is False, "checker promoted execution")
    return {
        "schema": "m1314_m1313_motion_ep34_final_unified_capture_production_launch_blind_hammer_r1_v1",
        "status": "PASS_M1314_M1313_BLIND_HAMMER__ROOT_AGENT_SINGLE_REMOTE_CAPTURE_ONLY__NO_RETRY",
        "verdict": "GO_ROOT_AGENT_ONE_REMOTE_M1249_CAPTURE_ONLY",
        "identity": {
            "contract_sha256": EXPECTED["contract"],
            "checker_sha256": EXPECTED["checker"],
            "test_sha256": EXPECTED["test"],
            "author_manifest_sha256": EXPECTED["author_manifest"],
            "author_outer_file_sha256": EXPECTED["author_outer_file"],
            "author_receipt_sha256": EXPECTED["author_receipt"],
            "author_members": len(author_rows),
        },
        "verification": {
            "author_tests_passed": tests,
            "author_tests_failed": 0,
            "full_NPY_SHA_files": len(samples),
            "full_NPY_bytes": byte_total,
            "cohort_compact_sha256": EXPECTED["cohort_compact"],
            "cohort_path_size_sha_aggregate": row_digest.hexdigest(),
            "mutation_attacks_rejected": len(attacks),
            "mutation_attacks_total": len(attacks),
            "mutation_attacks": attacks,
            "namespaces_fresh_local": True,
            "automatic_retry": False,
        },
        "authorization": {
            "authorized_actor": "root_agent",
            "remote_capture_runs": 1,
            "automatic_retry": False,
            "hammer_author_executed_remote": False,
            "hammer_author_used_gpu": False,
            "hammer_author_used_eda": False,
        },
        "claim_boundary": {
            "capture_complete": False,
            "paper_metric": False,
            "hardware_speedup": False,
            "system_speedup": False,
            "energy": False,
            "ppa": False,
        },
        "test_log_tail": test_log.strip().splitlines()[-4:],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
