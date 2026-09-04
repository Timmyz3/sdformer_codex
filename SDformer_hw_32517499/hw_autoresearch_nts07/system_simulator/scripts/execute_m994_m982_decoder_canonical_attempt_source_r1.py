#!/usr/bin/env python3
"""M994 additive repair: canonical ATTEMPT mkdir is attempt consumption.

M981 payload, row execution, atomic seal, and failure-quarantine semantics are
frozen. M994 changes only the one-attempt boundary found unsafe by M982. No
model prefix or external tool runs in source validation/self-test modes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
BASE_PATH = HERE / "execute_m981_m977_decoder_d2d3_10k_atomic_evidence_r1.py"
BASE_SHA = "dfd626e292077efc1d447ceb870a5c113e531c2086b0001ccbffbf1ec8ff86b2"
CONTRACT = HW / "contracts/m994_m982_decoder_canonical_attempt_source_contract_r1_20260829.json"
SOURCE_HAMMER = HW / "reviews/m995_m994_decoder_canonical_attempt_source_hammer_r1_20260829"
RELEASE = HW / "contracts/m996_m994_decoder_canonical_attempt_release_r1_20260829.json"
RELEASE_HAMMER = HW / "reviews/m997_m996_m994_decoder_canonical_attempt_release_hammer_r1_20260829"
RESULT = HW / "results/m998_m994_decoder_d2d3_10k_canonical_attempt_r1_20260829"
ATTEMPT = HW / "results/.m998_m994_decoder_d2d3_10k_canonical_attempt_consumed"
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
SOURCE_SCHEMA = "m994_m982_decoder_canonical_attempt_source_contract_v1"
RELEASE_SCHEMA = "m996_m994_decoder_canonical_attempt_release_v1"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M982 = HW / "reviews/m982_m981_decoder_d2d3_10k_atomic_evidence_source_hammer_r1_20260829"
M982_ID = (
    "a5f6063f23d8ad3c33861767559ec3bd8a2ae6781d370c9380f6d3a1cea39757",
    "63990eb5abae1dc40c67f6635edbb07b5774d64afa4b52573e66c1d87471bfd3",
    "196c43b14614387d76edac32ca57b18d6bf76354bde475f131fbf06d83d60b50",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


require(sha256(BASE_PATH) == BASE_SHA, "M994 frozen M981 driver drift")
_spec = importlib.util.spec_from_file_location("m994_frozen_m981", BASE_PATH)
require(_spec is not None and _spec.loader is not None, "cannot load M981")
B = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = B
_spec.loader.exec_module(B)

# Repoint only result namespaces. Frozen row/seal/quarantine functions resolve
# these module globals at call time.
B.RESULT = RESULT
B.ATTEMPT = ATTEMPT
B.FAILURE_PREFIX = FAILURE_PREFIX


def canonical_paths():
    return {
        "source_contract": str(CONTRACT.relative_to(REPO)),
        "source_hammer": str(SOURCE_HAMMER.relative_to(REPO)),
        "release": str(RELEASE.relative_to(REPO)),
        "release_hammer": str(RELEASE_HAMMER.relative_to(REPO)),
        "run_result": str(RESULT.relative_to(REPO)),
        "run_attempt": str(ATTEMPT.relative_to(REPO)),
        "run_failure_prefix": "hw_autoresearch_nts07/results/" + FAILURE_PREFIX,
    }


def strict_json(path):
    return B.strict_json(path)


def verify_m982():
    sealed = B.M946.M785.verify_sealed_directory(M982)
    require(sha256(M982 / "review.json") == M982_ID[0] and
            sealed["manifest_sha256"] == M982_ID[1] and
            sealed["outer_seal_file_sha256"] == M982_ID[2], "M982 identity drift")
    review = strict_json(M982 / "review.json")
    require(review.get("status") == "STOP_M982_M981_ATTEMPT_CONSUMPTION_NOT_FAIL_CLOSED" and
            review.get("verdict") == "STOP" and review.get("p0_count") == 1,
            "M982 decision drift")
    return sealed


def validate_source_contract(contract, runner, require_fresh=True):
    require(Path(contract).resolve() == CONTRACT.resolve(), "M994 contract path drift")
    value = strict_json(contract)
    require(value.get("schema") == SOURCE_SCHEMA and value.get("launch_now") is False and
            value.get("canonical") == canonical_paths(), "M994 source contract drift")
    verify_m982()
    for name, item in value["source_identity"].items():
        path = HW / item["path"]
        require(path.is_file() and not path.is_symlink() and sha256(path) == item["sha256"],
                "M994 source drift: " + name)
    require(Path(runner).resolve() ==
            (HW / value["source_identity"]["m998_runner"]["path"]).resolve(),
            "M998 runner path drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA,
            "docs359 drift")
    if require_fresh:
        require(not ATTEMPT.exists() and not RESULT.exists(), "M998 namespace not fresh")
    return {"status": "PASS_M994_SOURCE__NO_REAL_10K",
            "contract_sha256": sha256(contract), "runner_sha256": sha256(runner)}


def _safe_attempt(attempt, result, allowed_parent=None):
    parent = RESULT.parent if allowed_parent is None else Path(allowed_parent)
    require(Path(attempt).parent.resolve() == parent.resolve() and
            Path(result).parent.resolve() == parent.resolve() and
            Path(attempt).name == ATTEMPT.name and Path(result).name == RESULT.name and
            not Path(attempt).is_symlink() and not Path(result).is_symlink(),
            "M994 unsafe canonical attempt namespace")


def consume_attempt(authority, attempt=ATTEMPT, result=RESULT,
                    allowed_parent=None, inject_fault=""):
    """Consume exactly once; canonical mkdir precedes every receipt write."""
    attempt, result = Path(attempt), Path(result)
    _safe_attempt(attempt, result, allowed_parent)
    require(not attempt.exists() and not result.exists(), "M998 attempt already consumed")
    os.mkdir(attempt, 0o700)  # atomic, canonical, irreversible consumption point
    B.fsync_dir(attempt.parent)
    if inject_fault == "after_canonical_mkdir":
        raise RuntimeError("injected after canonical mkdir")
    receipt = {
        "schema": "m998_canonical_attempt_v1",
        "status": "CONSUMED_AT_CANONICAL_MKDIR_BEFORE_D2_MODEL_CALL",
        "max_attempts": 1, "retry": False,
        "release_sha256": authority["release_sha256"],
        "release_hammer_review_sha256": authority["release_hammer_review_sha256"],
        "d2_or_d3_100k_authorized": False, "full_row_authorized": False,
    }
    B.write_exclusive(attempt / "attempt.json",
                      (json.dumps(receipt, sort_keys=True) + "\n").encode())
    B.fsync_dir(attempt)
    if inject_fault == "after_attempt_receipt":
        raise RuntimeError("injected after attempt receipt")
    seal = B.atomic_seal(attempt)
    if inject_fault == "after_attempt_seal":
        raise RuntimeError("injected after attempt seal")
    require(B.verify_atomic_seal(attempt) == seal, "M998 attempt seal drift")
    return {"receipt": receipt, "seal": seal}


def validate_attempt(authority, attempt=ATTEMPT):
    seal = B.verify_atomic_seal(attempt)
    receipt = strict_json(Path(attempt) / "attempt.json")
    require(receipt.get("status") == "CONSUMED_AT_CANONICAL_MKDIR_BEFORE_D2_MODEL_CALL" and
            receipt.get("max_attempts") == 1 and receipt.get("retry") is False and
            receipt.get("release_sha256") == authority["release_sha256"] and
            receipt.get("release_hammer_review_sha256") ==
                authority["release_hammer_review_sha256"], "M998 attempt drift")
    return {"receipt": receipt, "seal": seal}


B.validate_attempt = validate_attempt


def verify_flat_review(directory, identity, label):
    sealed = B.M946.M785.verify_sealed_directory(directory)
    require(sha256(directory / "review.json") == identity[0] and
            sealed["manifest_sha256"] == identity[1] and
            sealed["outer_seal_file_sha256"] == identity[2], label + " identity drift")
    return sealed


def validate_authority(runner, expected_release_sha, source_identity, release_identity):
    source = validate_source_contract(CONTRACT, runner, require_fresh=False)
    require(RELEASE.is_file() and sha256(RELEASE) == expected_release_sha,
            "M996 release identity drift")
    verify_flat_review(SOURCE_HAMMER, source_identity, "M995")
    sr = strict_json(SOURCE_HAMMER / "review.json")
    require(sr.get("status") == "PASS_M995_M994_CANONICAL_ATTEMPT_SOURCE_HAMMER" and
            sr.get("verdict") == "GO_AUTHOR_M996_RELEASE_ONLY", "M995 authority drift")
    release = strict_json(RELEASE)
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == "AUTHORIZE_ONE_M998_D2_THEN_D3_10K_RUN" and
            release.get("release") is True and release.get("launch_now") is False and
            release.get("max_attempts") == 1, "M996 release drift")
    require(release.get("exact_rows") == [
        {"layer": "D2", "sample_id": 0, "config": "A1_OSG", "timestep": 0,
         "expanded_prefix": 10000},
        {"layer": "D3", "sample_id": 0, "config": "A1_OSG", "timestep": 0,
         "expanded_prefix": 10000}], "M996 row order drift")
    binding = release.get("source_binding", {})
    require(binding.get("m994_contract_sha256") == source["contract_sha256"] and
            binding.get("m994_driver_sha256") == sha256(Path(__file__)) and
            binding.get("m998_runner_sha256") == sha256(runner) and
            binding.get("m995_review_sha256") == source_identity[0] and
            binding.get("m982_review_sha256") == M982_ID[0], "M996 binding drift")
    auth = release.get("authorization", {})
    require(auth.get("one_m998_d2_then_d3_10k") is True and
            all(auth.get(key) is False for key in
                ("retry", "d2_or_d3_100k", "full_row", "production",
                 "eda_gpu_remote")), "M996 authorization expansion")
    verify_flat_review(RELEASE_HAMMER, release_identity, "M997")
    rr = strict_json(RELEASE_HAMMER / "review.json")
    require(rr.get("status") == "PASS_M997_M996_M994_CANONICAL_ATTEMPT_RELEASE_HAMMER" and
            rr.get("verdict") == "GO_ONE_M998_RUN_ONLY" and
            rr.get("release_sha256") == expected_release_sha, "M997 authority drift")
    return {"status": "PASS_M994_M998_ONE_RUN_AUTHORITY",
            "release_sha256": expected_release_sha,
            "source_hammer_review_sha256": source_identity[0],
            "release_hammer_review_sha256": release_identity[0]}


def assemble(work, authority):
    B.safe_work(work); validate_attempt(authority)
    rows, seals = [], {}
    for layer in ("D2", "D3"):
        row = Path(work) / layer; seals[layer] = B.verify_atomic_seal(row)
        payload = strict_json(row / "row.json")
        require(payload.get("status") == "PASS_M981_ROW_EXACT__FRESH_HAMMER_REQUIRED" and
                payload.get("summary", {}).get("layer") == layer, "M998 row/order drift")
        rows.append(payload)
    value = {"schema": "m998_decoder_d2d3_10k_canonical_attempt_result_v1",
             "status": "PASS_M998_D2_THEN_D3_10K__RESULT_HAMMER_REQUIRED",
             "release_sha256": authority["release_sha256"], "rows": rows,
             "row_seals": seals,
             "claim_boundary": {"paper_citable": False, "decoder_complete": False,
                 "table_a_row": False, "system_speedup": False}}
    B.write_exclusive(Path(work) / "result.json",
                      (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    B.write_exclusive(Path(work) / "RUN_COMPLETE.txt", b"M998_COMPLETE__HAMMER_REQUIRED\n")
    return {"result": value, "seal": B.atomic_seal(work)}


def publish(work):
    B.safe_work(work); seal = B.verify_atomic_seal(work)
    require(strict_json(Path(work) / "result.json").get("status") ==
            "PASS_M998_D2_THEN_D3_10K__RESULT_HAMMER_REQUIRED" and not RESULT.exists(),
            "M998 publish drift")
    os.rename(work, RESULT); B.fsync_dir(RESULT.parent)
    require(B.verify_atomic_seal(RESULT) == seal, "M998 result publish drift")
    return {"status": "PASS_M998_ATOMIC_RESULT_PUBLICATION", "seal": seal}


def source_self_test():
    authority = {"release_sha256": "a" * 64,
                 "release_hammer_review_sha256": "b" * 64}
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        attempt = parent / ATTEMPT.name; result = parent / RESULT.name
        try:
            consume_attempt(authority, attempt, result, parent, "after_canonical_mkdir")
        except RuntimeError as exc:
            require("injected" in str(exc), "fault injection drift")
        require(attempt.is_dir() and not (attempt / "attempt.json").exists(),
                "canonical mkdir interruption not preserved")
        try:
            consume_attempt(authority, attempt, result, parent)
        except RuntimeError as exc:
            require("already consumed" in str(exc), "retry rejection drift")
        else:
            raise RuntimeError("M994 allowed second attempt")
        require(not any(parent.glob(ATTEMPT.name + ".stage.*")),
                "random attempt stage created")
    return {"status": "PASS_M994_CANONICAL_ATTEMPT_SELF_TEST__NO_REAL_10K",
            "canonical_mkdir_is_consumption": True,
            "interrupted_canonical_attempt_blocks_retry": True,
            "random_precanonical_directory_created": False,
            "real_10k_executed": False, "eda_gpu_remote_used": False}


def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--validate-source", action="store_true")
    p.add_argument("--quarantine-work", action="store_true")
    p.add_argument("--validate-authority", action="store_true")
    p.add_argument("--consume-attempt", action="store_true")
    p.add_argument("--run-row", choices=("D2", "D3"))
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--contract", type=Path, default=CONTRACT)
    p.add_argument("--runner", type=Path); p.add_argument("--work", type=Path)
    p.add_argument("--quarantine", type=Path); p.add_argument("--row-stage", type=Path)
    p.add_argument("--return-code", type=int, default=1)
    p.add_argument("--expected-release-sha", default="")
    for prefix in ("source", "release"):
        p.add_argument("--expected-%s-review-sha" % prefix, default="")
        p.add_argument("--expected-%s-manifest-sha" % prefix, default="")
        p.add_argument("--expected-%s-outer-sha" % prefix, default="")
    a = p.parse_args(argv)
    modes = (a.self_test, a.validate_source, a.quarantine_work, a.validate_authority,
             a.consume_attempt, a.run_row is not None, a.assemble, a.publish)
    require(sum(bool(x) for x in modes) == 1, "M994 requires one mode")
    if a.self_test:
        value = source_self_test()
    elif a.validate_source:
        require(a.runner is not None, "runner required")
        value = validate_source_contract(a.contract, a.runner)
    elif a.quarantine_work:
        require(a.work is not None and a.quarantine is not None, "work/quarantine required")
        value = B.quarantine_work(a.work, a.quarantine, a.return_code)
    else:
        require(a.runner is not None and a.expected_release_sha, "M998 authority required")
        sid = (a.expected_source_review_sha, a.expected_source_manifest_sha,
               a.expected_source_outer_sha)
        rid = (a.expected_release_review_sha, a.expected_release_manifest_sha,
               a.expected_release_outer_sha)
        require(all(sid + rid), "M995/M997 identities required")
        authority = validate_authority(a.runner, a.expected_release_sha, sid, rid)
        if a.validate_authority: value = authority
        elif a.consume_attempt: value = consume_attempt(authority)
        else:
            validate_attempt(authority)
            if a.run_row is not None:
                require(a.row_stage is not None, "row stage required")
                value = B.run_row(a.run_row, a.row_stage)
            elif a.assemble:
                require(a.work is not None, "work required")
                value = assemble(a.work, authority)
            else:
                require(a.work is not None, "work required")
                value = publish(a.work)
    print(json.dumps(value, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
