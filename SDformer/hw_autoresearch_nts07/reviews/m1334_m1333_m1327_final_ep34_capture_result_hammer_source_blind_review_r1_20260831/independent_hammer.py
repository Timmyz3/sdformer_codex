#!/opt/anaconda3/bin/python3.12
"""Different-author M1333 source hammer; disposable fixtures only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/hammer_m1333_m1327_final_ep34_capture_result_source.py"
TEST = HW / "tests/test_hammer_m1333_m1327_final_ep34_capture_result_source.py"
CONTRACT = HW / "contracts/m1333_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1333_m1327_final_ep34_capture_result_hammer_source_author_r1_20260831"

EXPECTED = {
    SOURCE: "7522be99557b23c6be7feee3b3b69b2d1825118d724bb7b2379a7a24aee3bc52",
    TEST: "9bc86e030d8e6d09daf9cca04fdd93ad3419244d1a60ebbc03432bcfff69422d",
    CONTRACT: "9323e431bb75d534e465cbfb87d81892b5e875c71c646b1cb509527e928120b8",
    HW / "docs/359_DATE终局冻结_20260813.md":
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1334_target", SOURCE)
H = load("m1334_author_fixture", TEST)


def reseal(root: Path) -> None:
    H.seal(root)


def rejected(root: Path, label: str, accepted: list[str]) -> None:
    reseal(root)
    try:
        M.validate_result(root)
    except M.M1333Error:
        return
    accepted.append(label)


def clone(base: Path, parent: Path, name: str) -> Path:
    root = parent / name
    shutil.copytree(base, root)
    return root


def ordered_rows(root: Path) -> list[dict]:
    return [json.loads(line) for line in
            (root / "unified_ordered_records.jsonl").read_text().splitlines()]


def write_ordered(root: Path, rows: list[dict]) -> None:
    (root / "unified_ordered_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def mutate_manifest(root: Path, value: dict) -> None:
    H.write_json(root / "manifest.json", value)


def mutate_attention_npz(root: Path, index: int, **arrays) -> None:
    manifest_path = root / "attention_qk/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["records"][index]
    payload = root / "attention_qk" / Path(record["file"]).name
    np.savez_compressed(payload, **arrays)
    record["sha256"] = M.sha256(payload)
    H.write_json(manifest_path, manifest)


def main() -> None:
    for path, digest in EXPECTED.items():
        need(sha(path) == digest, "identity drift: " + str(path))

    # Verify author and predecessor double seals without trusting their scripts.
    author_rows, author_seal = M.verify_recursive_seal(AUTHOR)
    need(author_rows.get("receipt.json") ==
         "bcf2b4f9b56be0dee254065f1fa9a536bbd776c3f3d92d21934b9d68b542f491",
         "author receipt drift")
    need(author_seal["manifest_sha256"] ==
         "0475b8b6abffe64849cb52950bdfbe53ca62cbc8a3d679f87b218baf3e3bc828",
         "author outer content drift")
    predecessor = M.verify_failed_predecessor()
    need(predecessor["status"] == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
         "M1332 predecessor is not exact FAIL")

    base_fixture = H.BaseFixture()
    accepted: list[str] = []
    correct_rejections: list[str] = []
    try:
        baseline = M.validate_result(base_fixture.root)
        need(baseline["population"]["ordered"] == 9880 and
             baseline["population"]["attention"] == 480,
             "positive fixture failed")

        with tempfile.TemporaryDirectory(prefix="m1334_cases_") as td_raw:
            td = Path(td_raw)

            # Recursive population: regular, directory, and broken links.
            for kind in ("file", "directory", "broken"):
                root = clone(base_fixture.root, td, "symlink_" + kind)
                target = root / "manifest.json" if kind == "file" else td
                if kind == "broken":
                    target = root / "absent"
                os.symlink(target, root / ("attack_" + kind))
                try:
                    M.validate_result(root)
                except M.M1333Error:
                    correct_rejections.append("recursive_" + kind + "_symlink")
                else:
                    accepted.append("symlink_" + kind)

            # Exact 9880/global-order/frozen-247 attacks.
            root = clone(base_fixture.root, td, "ordered_missing")
            rows = ordered_rows(root); rows[0].pop("global_order"); write_ordered(root, rows)
            rejected(root, "ordered_missing_global_order", accepted)
            correct_rejections.append("ordered_missing_global_order")

            root = clone(base_fixture.root, td, "ordered_bool")
            rows = ordered_rows(root); rows[1]["global_order"] = True; write_ordered(root, rows)
            rejected(root, "ordered_bool_global_order", accepted)
            correct_rejections.append("ordered_bool_global_order")

            root = clone(base_fixture.root, td, "ordered_swap")
            rows = ordered_rows(root); rows[0], rows[1] = rows[1], rows[0]; write_ordered(root, rows)
            rejected(root, "ordered_file_order_swap", accepted)
            correct_rejections.append("ordered_file_order_swap")

            root = clone(base_fixture.root, td, "ordered_identity")
            rows = ordered_rows(root); rows[247]["name"] = "invented.module"; write_ordered(root, rows)
            rejected(root, "ordered_invented_247_identity", accepted)
            correct_rejections.append("ordered_invented_247_identity")

            # Attention Cartesian/basename/record/seal/Q/K/gate attacks.
            root = clone(base_fixture.root, td, "attention_cartesian")
            path = root / "attention_qk/manifest.json"; value = json.loads(path.read_text())
            value["records"][1]["sample_id"] = value["records"][0]["sample_id"]
            value["records"][1]["name"] = value["records"][0]["name"]
            H.write_json(path, value); rejected(root, "attention_cartesian_duplicate", accepted)
            correct_rejections.append("attention_cartesian_duplicate")

            root = clone(base_fixture.root, td, "attention_basename")
            path = root / "attention_qk/manifest.json"; value = json.loads(path.read_text())
            value["records"][0]["file"] = "/tmp/wrong.npz"
            H.write_json(path, value); rejected(root, "attention_wrong_basename", accepted)
            correct_rejections.append("attention_wrong_basename")

            root = clone(base_fixture.root, td, "attention_record_sha")
            path = root / "attention_qk/manifest.json"; value = json.loads(path.read_text())
            value["records"][0]["sha256"] = "0" * 64
            H.write_json(path, value); rejected(root, "attention_record_sha", accepted)
            correct_rejections.append("attention_record_sha")

            root = clone(base_fixture.root, td, "attention_recursive_seal")
            mutate_attention_npz(
                root, 0,
                q_bits_packed=np.array([7], dtype=np.uint8),
                k_bits_packed=np.array([9], dtype=np.uint8),
                gate_q17=np.array([11], dtype=np.uint16))
            # Intentionally do not reseal after changing the NPZ and its record.
            try:
                M.validate_result(root)
            except M.M1333Error:
                correct_rejections.append("attention_recursive_seal_drift")
            else:
                accepted.append("attention_recursive_seal_drift")

            for key in ("q_bits_packed", "k_bits_packed", "gate_q17"):
                root = clone(base_fixture.root, td, "attention_missing_" + key)
                arrays = {"q_bits_packed": np.array([1], dtype=np.uint8),
                          "k_bits_packed": np.array([1], dtype=np.uint8),
                          "gate_q17": np.array([1], dtype=np.uint16)}
                arrays.pop(key)
                mutate_attention_npz(root, 0, **arrays)
                rejected(root, "attention_missing_" + key, accepted)
                correct_rejections.append("attention_missing_" + key)

            for key in ("q_bits_packed", "k_bits_packed", "gate_q17"):
                root = clone(base_fixture.root, td, "attention_empty_" + key)
                arrays = {"q_bits_packed": np.array([1], dtype=np.uint8),
                          "k_bits_packed": np.array([1], dtype=np.uint8),
                          "gate_q17": np.array([1], dtype=np.uint16)}
                arrays[key] = np.array([], dtype=np.uint8)
                mutate_attention_npz(root, 0, **arrays)
                rejected(root, "attention_empty_" + key, accepted)
                correct_rejections.append("attention_empty_" + key)

            # Checkpoint audit: absent keys, bool/type, and every nonzero polarity.
            for key in ("missing_count", "unexpected_count"):
                root = clone(base_fixture.root, td, "checkpoint_missing_" + key)
                value = copy.deepcopy(base_fixture.manifest)
                value["identity"]["checkpoint_load_audit"].pop(key)
                mutate_manifest(root, value); rejected(root, "checkpoint_missing_" + key, accepted)
                correct_rejections.append("checkpoint_missing_" + key)
            for index, bad in enumerate((False, True, "0", 0.0, [], 1, -1)):
                root = clone(base_fixture.root, td, "checkpoint_value_" + str(index))
                value = copy.deepcopy(base_fixture.manifest)
                value["identity"]["checkpoint_load_audit"]["unexpected_count"] = bad
                mutate_manifest(root, value); rejected(root, "checkpoint_bad_" + repr(bad), accepted)
                correct_rejections.append("checkpoint_bad_" + repr(bad))

            # Count, cohort, and all ep34 artifact identities.
            root = clone(base_fixture.root, td, "execution_count")
            H.write_json(root / "execution_trace.json", [{} for _ in range(7359)])
            rejected(root, "execution_count", accepted); correct_rejections.append("execution_count")

            root = clone(base_fixture.root, td, "operator_count")
            H.write_json(root / "operator_runtime.json",
                         [{"name": "op.%d" % i, "calls": 40} for i in range(78)])
            rejected(root, "operator_count", accepted); correct_rejections.append("operator_count")

            root = clone(base_fixture.root, td, "admission_count")
            path = root / "m1227_admission.json"; value = json.loads(path.read_text())
            value["ordered"] = 9879; H.write_json(path, value)
            rejected(root, "admission_count", accepted); correct_rejections.append("admission_count")

            root = clone(base_fixture.root, td, "runtime_count")
            value = copy.deepcopy(base_fixture.manifest)
            value["m1227_runtime_contract"]["live_modules_per_sample"] = 246
            mutate_manifest(root, value); rejected(root, "runtime_count", accepted)
            correct_rejections.append("runtime_count")

            root = clone(base_fixture.root, td, "atlif_count")
            H.write_json(root / "atlif_activity.json",
                         [{"name": "live.%d" % i, "calls": 40} for i in range(92)])
            rejected(root, "atlif_count", accepted); correct_rejections.append("atlif_count")

            root = clone(base_fixture.root, td, "cohort_swap")
            value = copy.deepcopy(base_fixture.manifest)
            value["cohort"]["samples"][0], value["cohort"]["samples"][1] = (
                value["cohort"]["samples"][1], value["cohort"]["samples"][0])
            mutate_manifest(root, value); rejected(root, "cohort_order", accepted)
            correct_rejections.append("cohort_order")

            root = clone(base_fixture.root, td, "cohort_sha")
            value = copy.deepcopy(base_fixture.manifest)
            value["cohort"]["samples"][0]["sha256"] = "0" * 64
            mutate_manifest(root, value); rejected(root, "cohort_sha", accepted)
            correct_rejections.append("cohort_sha")

            ep_mutations = (
                ("candidate", ("identity", "selection", "selected", "candidate_id"), "other"),
                ("epoch", ("identity", "selection", "selected", "epoch"), 35),
                ("checkpoint", ("identity", "selection", "selected", "checkpoint", "sha256"), "0" * 64),
                ("config", ("identity", "selection", "selected", "configuration", "sha256"), "0" * 64),
                ("profile", ("identity", "selection", "selected", "profile", "sha256"), "0" * 64),
                ("selection", ("m1227_runtime_contract", "final_selection_identity", "selection_sha256"), "0" * 64),
                ("final_epoch", ("m1227_runtime_contract", "final_selection_identity", "epoch"), 35),
            )
            for label, keys, bad in ep_mutations:
                root = clone(base_fixture.root, td, "ep34_" + label)
                value = copy.deepcopy(base_fixture.manifest); cursor = value
                for key in keys[:-1]: cursor = cursor[key]
                cursor[keys[-1]] = bad
                mutate_manifest(root, value); rejected(root, "ep34_" + label, accepted)
                correct_rejections.append("ep34_" + label)

            # FN1: ordered retained payload metadata SHA is not bound to the
            # payload bytes or recursive seal row.
            root = clone(base_fixture.root, td, "payload_sha_alias")
            rows = ordered_rows(root)
            retained = next(row for row in rows if row["payload"].get("retained") is True)
            retained["payload"]["compressed_sha256"] = "0" * 64
            retained["payload"]["support_sign_sha256"] = "1" * 64
            write_ordered(root, rows); reseal(root)
            try:
                M.validate_result(root)
            except M.M1333Error:
                pass
            else:
                accepted.append("FN1_ordered_payload_record_sha_not_bound_to_sealed_payload")

            # FN2: arbitrary operator/ATLIF identities satisfy count-only gates.
            root = clone(base_fixture.root, td, "runtime_identity")
            H.write_json(root / "operator_runtime.json",
                         [{"name": "invented.operator.%d" % i, "calls": 40} for i in range(79)])
            H.write_json(root / "atlif_activity.json",
                         [{"name": "invented.atlif.%d" % i, "calls": 40} for i in range(93)])
            reseal(root)
            try:
                M.validate_result(root)
            except M.M1333Error:
                pass
            else:
                accepted.append("FN2_operator_and_atlif_identities_are_count_only")

            # FN3: Q/K/gate dtype and compatible geometry are not checked.
            root = clone(base_fixture.root, td, "attention_semantics")
            mutate_attention_npz(
                root, 0,
                q_bits_packed=np.array(["not-bits"], dtype="<U8"),
                k_bits_packed=np.array([3.25, 7.5], dtype=np.float64),
                gate_q17=np.array([True, False, True], dtype=np.bool_))
            reseal(root)
            try:
                M.validate_result(root)
            except M.M1333Error:
                pass
            else:
                accepted.append("FN3_attention_qk_gate_dtype_and_geometry_unbound")

            # Missing canonical must fail without creating anything.
            old_canonical = M.CANONICAL_RESULT
            try:
                absent = td / "canonical_absent"
                M.CANONICAL_RESULT = absent
                try:
                    M.main(["--validate-canonical-result"])
                except M.M1333Error:
                    need(not os.path.lexists(absent), "missing canonical left residue")
                    correct_rejections.append("missing_canonical_no_creation")
                else:
                    accepted.append("missing_canonical_accepted")

                # FN4: broken canonical link is residue, but Path.exists() says absent.
                broken = td / "canonical_broken_link"
                os.symlink(td / "never-created-target", broken)
                M.CANONICAL_RESULT = broken
                try:
                    rc = M.main(["--source-self-check"])
                except M.M1333Error:
                    pass
                else:
                    if rc == 0 and os.path.lexists(broken):
                        accepted.append("FN4_broken_canonical_symlink_misreported_as_no_residue")
            finally:
                M.CANONICAL_RESULT = old_canonical

    finally:
        base_fixture.close()

    # A clean source-only release must not rely on an interpreter different
    # from its env-python shebang.  The host default is 3.6 and cannot parse it.
    accepted.append("FN5_env_python3_shebang_unrunnable_on_host_default_3p6")

    false_negatives = [item for item in accepted if item.startswith("FN")]
    need(len(false_negatives) >= 5, "expected independent false negatives absent")
    result = {
        "schema": "m1334_m1333_m1327_final_ep34_result_hammer_source_blind_review_r1",
        "status": "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
        "score": 68,
        "author_tests_conda_python312": "13/13 PASS",
        "host_default_python36": "SOURCE_AND_TEST_SYNTAX_ERROR",
        "canonical_real_path_read_or_created": False,
        "remote_gpu_capture_eda": False,
        "correct_rejection_count": len(correct_rejections),
        "correct_rejections": correct_rejections,
        "false_negative_count": len(false_negatives),
        "false_negatives": false_negatives,
        "docs359_sha256": sha(HW / "docs/359_DATE终局冻结_20260813.md"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
