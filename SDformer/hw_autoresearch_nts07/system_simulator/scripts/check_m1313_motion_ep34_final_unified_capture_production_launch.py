#!/usr/bin/env python3
"""Fail-closed static audit for the M1313 ep34 one-shot capture launch.

This checker is deliberately read-only.  It validates author-time identities,
the canonical-path M1312 successor, the exact forty-sample cohort, and fresh
one-shot namespaces.  It never imports or executes the M1249 capture source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json"
M1182 = HW / "contracts/m1182_m1180_motion_ep29_unified_capture_launch_release_r1_20260830.json"
M1210 = HW / "contracts/m1210_m1208_motion_ep29_unified_capture_launch_release_r1_20260830.json"
STAGED_SELECTION = HW / (
    "system_handoff/incoming/m1306_remote_selection_result_20260830/"
    "hw_autoresearch_nts07/results/"
    "m1257_motion_cross_run_final_checkpoint_selection_r5_20260830")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SCHEMA = "m1249_motion_final_checkpoint_unified_capture_one_shot_production_launch_r1_v1"
STATUS = "M1243_M1244_AND_FINAL_M1237_BOUND__ONE_M1249_GPU_RUN_AUTHORIZED"
TOP_KEYS = {
    "schema", "status", "contract_path", "release_identity", "inputs", "cohort",
    "one_shot", "output", "production_log",
}
RELEASE_KEYS = {
    "source_path", "source_sha256", "test_path", "test_sha256",
    "source_contract_path", "source_contract_sha256",
}
INPUT_KEYS = {
    "m1243_source", "m1243_test", "m1243_source_contract", "m1244_source_hammer",
    "final_selection_result", "final_selection_result_hammer",
}
IDENTITY_KEYS = {"path", "sha256"}
RESULT_KEYS = {
    "result_path", "manifest_sha256", "outer_file_sha256", "selection_member",
    "selection_sha256",
}
HAMMER_KEYS = {"path", "manifest_sha256", "outer_file_sha256", "review_sha256"}
AUTHORITY_KEYS = {
    "result_path", "selection_member", "selection_sha256",
    "selection_manifest_sha256", "selection_outer_file_sha256", "selection_schema",
    "selection_status", "selected_candidate_id", "selected_epoch",
    "selected_profile_sha256", "selected_checkpoint_sha256", "selected_config_sha256",
}

M1249_RELEASE = {
    "source_path": "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1249_motion_final_checkpoint_unified_hardware_one_shot_release_r1.py",
    "source_sha256": "5fbcc4d287f3ffd3b1c9994efa24245e5e3828927cdac925c1a35d8a88a19219",
    "test_path": "hw_autoresearch_nts07/tests/test_m1249_motion_final_checkpoint_unified_capture_one_shot_release_source.py",
    "test_sha256": "fc81e54c6f15f05864ef671bae27e34fbefcf4ea6b965d63ef4d8730ce0a6fce",
    "source_contract_path": "hw_autoresearch_nts07/contracts/m1249_motion_final_checkpoint_unified_capture_one_shot_release_source_contract_r1_20260830.json",
    "source_contract_sha256": "e9d0577b331491269780c8fd511b3cf378d62f4023c392c05924a134b7e35ad0",
}
M1243 = {
    "m1243_source": {
        "path": "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1243_motion_final_checkpoint_unified_hardware_launch_authority_r3.py",
        "sha256": "009c92c22b5429352b0b4dd29c723035744efa828db9c4472d1f4fb4140297e2",
    },
    "m1243_test": {
        "path": "hw_autoresearch_nts07/tests/test_m1243_motion_capture_launch_authority_successor_source.py",
        "sha256": "7529dd988e48926d683c0ea28c1ca5e9e06a2af617febe796a02e09e38c3ded7",
    },
    "m1243_source_contract": {
        "path": "hw_autoresearch_nts07/contracts/m1243_motion_capture_launch_authority_successor_source_contract_r1_20260830.json",
        "sha256": "de558985c0f9a64580060dce90675d8ba4ca771a616fe8152b439483663f26ba",
    },
}
M1244 = {
    "path": "hw_autoresearch_nts07/reviews/m1244_m1243_motion_capture_launch_authority_source_hammer_r1_20260830",
    "manifest_sha256": "8b4e633103098faf140c1660abd1ac6e4745bb7dd3c2838ec9ac88ee6a9adce2",
    "outer_file_sha256": "657af0a531ed95e3abb301f2dd5b5827e3f737dcc34ab3120f3f593ad3ac55f2",
    "review_sha256": "64773f9fc58b67af2caf9cf60642ace071e526ee9de928cfb515c419959edd8a",
}
FINAL_SELECTION = {
    "result_path": "hw_autoresearch_nts07/results/m1257_motion_cross_run_final_checkpoint_selection_r5_20260830",
    "manifest_sha256": "ae4a61f5e79b0d6e308174c00567fff6e25a07a6f065cd7ee3acec2faabcf458",
    "outer_file_sha256": "d0afaea457958752b9d76c21746c0796145a91466cf93ecd20a56d27bd5ef7e4",
    "selection_member": "final_checkpoint_selection.json",
    "selection_sha256": "4af7b7e1b4a174440331268fcfffda44896d86d02c7d20195e7a49d73eae6cd0",
}
M1312 = {
    "path": "hw_autoresearch_nts07/reviews/m1312_m1309_canonical_path_compatibility_successor_r1_20260831",
    "manifest_sha256": "4bc73966ad003054efcf655b8e0fb9da812201ea203e38444842a40470140ea5",
    "outer_file_sha256": "fc0b47bbea57f9a90237d6b72c3d3e9bcd0b0bd9ee555f94ade8f0ba7defbd6b",
    "review_sha256": "ce4c188f1c4fe12a7c4c78f4922cf028170c9087a6fb72cd70659fdd9d1771fd",
}
PINNED_SEALS = {
    "hw_autoresearch_nts07/reviews/m1252_m1249_final_unified_capture_release_source_hammer_r1_20260830/SHA256SUMS.seal.sha256": "4bbf58864d9779448be0a5862b35dcad8a69d0131a77977cbb1f925270f8c68e",
    "hw_autoresearch_nts07/reviews/m1307_m1306_inherited_authority_successor_receipt_blind_hammer_r1_20260830/SHA256SUMS.seal.sha256": "ef91ccfcf77e393df68bb37178a0ff41f61fd65b113a93cd2718a323ec0dbbad",
    "hw_autoresearch_nts07/reviews/m1309_m1306_remote_final_selection_result_independent_hammer_r1_20260831/SHA256SUMS.seal.sha256": "275b8f18538c72819879f921da0d6d56acf356f8f7df3e2ba60e7eb09acdac02",
    "hw_autoresearch_nts07/reviews/m1312_m1309_canonical_path_compatibility_successor_r1_20260831/SHA256SUMS.seal.sha256": "fc0b47bbea57f9a90237d6b72c3d3e9bcd0b0bd9ee555f94ade8f0ba7defbd6b",
    "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_AUTHORITY = {
    "result_path": FINAL_SELECTION["result_path"],
    "selection_member": FINAL_SELECTION["selection_member"],
    "selection_sha256": FINAL_SELECTION["selection_sha256"],
    "selection_manifest_sha256": FINAL_SELECTION["manifest_sha256"],
    "selection_outer_file_sha256": FINAL_SELECTION["outer_file_sha256"],
    "selection_schema": "m1234_motion_cross_run_final_checkpoint_rebind_binder_r2_v1",
    "selection_status": "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R2__INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY",
    "selected_candidate_id": "resume_ep34",
    "selected_epoch": 34,
    "selected_profile_sha256": "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c",
    "selected_checkpoint_sha256": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "selected_config_sha256": "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39",
}
RESULT = HW / "results/m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830"
ATTEMPT = HW / "results/.m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830.attempt_consumed"
LOG = HW / "results/.m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830.production.log"
COHORT_SHA256 = "e9e6443c25a2f3d7ee6994b8c708eaecec7845f70dd920a132adc9276744745f"


class AuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise AuditError("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " is not a regular file")
    require(sha256(path) == expected, label + " SHA mismatch")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> Any:
        raise AuditError("non-finite JSON constant: " + value)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                           parse_constant=reject)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError("invalid JSON: " + str(path)) from exc
    require(isinstance(value, dict), "JSON root is not an object")
    return value


def compact_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_double_seal(entry: dict[str, Any]) -> dict[str, str]:
    require(isinstance(entry, dict) and set(entry) == HAMMER_KEYS, "hammer entry shape")
    root = ROOT / entry["path"]
    require(root.parent == HW / "reviews", "hammer must be a direct reviews child")
    regular_exact(root / "SHA256SUMS", entry["manifest_sha256"], "hammer manifest")
    regular_exact(root / "SHA256SUMS.seal.sha256", entry["outer_file_sha256"],
                  "hammer outer seal")
    rows: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64, "malformed hammer manifest")
        digest, name = parts
        require(name not in rows and "/" not in name and name not in {".", ".."},
                "unsafe or duplicate hammer member")
        rows[name] = digest
        regular_exact(root / name, digest, "hammer member " + name)
    require(rows.get("review.json") == entry["review_sha256"], "hammer review SHA mismatch")
    population = {path.name for path in root.iterdir()
                  if path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(population == set(rows), "hammer manifest population mismatch")
    return rows


def validate_contract(contract: dict[str, Any], *, check_sample_bytes: bool = True,
                      namespace_exists: Callable[[str], bool] = os.path.lexists) -> dict[str, Any]:
    require(set(contract) == TOP_KEYS, "launch top-level keys mismatch")
    require(contract["schema"] == SCHEMA and contract["status"] == STATUS,
            "launch schema/status mismatch")
    require(contract["contract_path"] == str(CONTRACT.relative_to(ROOT)),
            "launch contract path mismatch")
    require(set(contract["release_identity"]) == RELEASE_KEYS and
            contract["release_identity"] == M1249_RELEASE, "M1249 release identity mismatch")
    for key, value in M1249_RELEASE.items():
        if key.endswith("_path"):
            regular_exact(ROOT / value, M1249_RELEASE[key.replace("_path", "_sha256")], key)

    inputs = contract["inputs"]
    require(isinstance(inputs, dict) and set(inputs) == INPUT_KEYS, "launch input keys")
    for key, expected in M1243.items():
        require(isinstance(inputs[key], dict) and set(inputs[key]) == IDENTITY_KEYS and
                inputs[key] == expected, key + " identity mismatch")
        regular_exact(ROOT / expected["path"], expected["sha256"], key)
    require(inputs["m1244_source_hammer"] == M1244, "M1244 exact entry mismatch")
    verify_double_seal(inputs["m1244_source_hammer"])
    require(set(inputs["final_selection_result"]) == RESULT_KEYS and
            inputs["final_selection_result"] == FINAL_SELECTION,
            "final selection canonical entry mismatch")
    require(inputs["final_selection_result_hammer"] == M1312,
            "M1312 exact entry mismatch")
    verify_double_seal(inputs["final_selection_result_hammer"])

    review = strict_json(ROOT / M1312["path"] / "review.json")
    require(review.get("schema") ==
            "m1237_m1234_motion_cross_run_final_checkpoint_binder_result_hammer_r1_v1",
            "M1312 review schema mismatch")
    require(review.get("status") ==
            "PASS_M1237_M1234_FINAL_SELECTION__HARDWARE_REBIND_RELEASE_AUTHORING_ALLOWED",
            "M1312 review status mismatch")
    require(isinstance(review.get("selection_authority"), dict) and
            set(review["selection_authority"]) == AUTHORITY_KEYS and
            review["selection_authority"] == EXPECTED_AUTHORITY,
            "M1312 authority mismatch")
    require(review.get("independence") == {"different_author": True},
            "M1312 independence mismatch")
    require(review.get("authorization") == {
        "hardware_rebind_release_authoring": True, "production_capture": False},
        "M1312 authorization mismatch")

    regular_exact(STAGED_SELECTION / "SHA256SUMS", FINAL_SELECTION["manifest_sha256"],
                  "staged selection manifest")
    regular_exact(STAGED_SELECTION / "SHA256SUMS.seal.sha256",
                  FINAL_SELECTION["outer_file_sha256"], "staged selection outer seal")
    regular_exact(STAGED_SELECTION / FINAL_SELECTION["selection_member"],
                  FINAL_SELECTION["selection_sha256"], "staged selection member")
    selected = strict_json(STAGED_SELECTION / FINAL_SELECTION["selection_member"])["selected"]
    require(selected["candidate_id"] == "resume_ep34" and selected["epoch"] == 34,
            "selected checkpoint pair mismatch")
    require(selected["checkpoint"]["sha256"] == EXPECTED_AUTHORITY["selected_checkpoint_sha256"] and
            selected["configuration"]["sha256"] == EXPECTED_AUTHORITY["selected_config_sha256"] and
            selected["profile"]["sha256"] == EXPECTED_AUTHORITY["selected_profile_sha256"],
            "selected checkpoint/config/profile SHA mismatch")

    regular_exact(M1182, "46450015bcdb3b8c0a32ccd7aaba68a78abf923705a133147202283e7bc7220f",
                  "M1182 cohort contract")
    regular_exact(M1210, "5aeeaf9cab836f32e025f0c329ef1fe90caa4ee3acae691514f4793c1d143829",
                  "M1210 cohort contract")
    samples = contract.get("cohort", {}).get("samples")
    require(isinstance(samples, list) and len(samples) == 40, "cohort must have 40 rows")
    require(samples == strict_json(M1182)["cohort"]["samples"] ==
            strict_json(M1210)["cohort"]["samples"], "cohort differs from M1182/M1210")
    require(compact_sha(samples) == COHORT_SHA256, "cohort compact SHA mismatch")
    require([row["global_sample_id"] for row in samples] == list(range(40)),
            "cohort global order mismatch")
    require(len({row["path"] for row in samples}) == 40 and
            len({row["sha256"] for row in samples}) == 40 and
            len({row["sample_key"] for row in samples}) == 40,
            "cohort identity is not unique")
    require(all(row["bytes"] == 12288128 for row in samples), "cohort byte size mismatch")
    if check_sample_bytes:
        for row in samples:
            path = ROOT / row["path"]
            regular_exact(path, row["sha256"], "cohort sample " + row["sample_key"])
            require(path.stat().st_size == row["bytes"], "cohort sample size mismatch")

    require(contract["one_shot"] == {
        "attempt_marker": str(ATTEMPT.relative_to(ROOT)), "automatic_retry": False},
        "one-shot policy mismatch")
    require(contract["output"] == {"path": str(RESULT.relative_to(ROOT))},
            "result namespace mismatch")
    require(contract["production_log"] == {"path": str(LOG.relative_to(ROOT))},
            "production log namespace mismatch")
    namespaces = [str(RESULT), str(ATTEMPT), str(LOG)]
    require(len(set(namespaces)) == 3, "one-shot namespaces overlap")
    for path in namespaces:
        require(not namespace_exists(path), "one-shot namespace is not fresh: " + path)

    for path, expected in PINNED_SEALS.items():
        regular_exact(ROOT / path, expected, path)
    return {
        "selected_candidate_id": "resume_ep34",
        "selected_epoch": 34,
        "samples": 40,
        "cohort_compact_sha256": COHORT_SHA256,
        "automatic_retry": False,
        "namespaces_fresh": True,
        "remote_gpu_capture_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--skip-sample-byte-hash", action="store_true",
                        help="Only for unit-test diagnostics; author admission hashes all samples.")
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(contract_path == CONTRACT, "only the canonical M1313 contract is admissible")
    result = validate_contract(strict_json(contract_path),
                               check_sample_bytes=not args.skip_sample_byte_hash)
    print("PASS_M1313_EP34_FINAL_UNIFIED_CAPTURE_PRODUCTION_LAUNCH_AUTHOR " +
          json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
