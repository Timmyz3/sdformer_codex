#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1511 fresh independent actual-result hammer for M1510.

This is a single-process, sequential CPU audit.  It reads the sealed local
capture, but does not launch EDA, GPU, remote, checkpoint, or production work.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
import zlib


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ROOT = HW.parent
SOURCE = HW / "system_simulator/scripts/build_m1510_ep34_decoder_layer_constant_adapter_source.py"
TEST = HW / "system_simulator/tests/test_m1510_ep34_decoder_layer_constant_adapter_source.py"
CONTRACT = HW / "contracts/m1510_ep34_decoder_layer_constant_adapter_source_contract_r1_20260831.json"
PRECHECK = HERE / "freshness_precheck.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    "m1510_source": "051b61d5cf8a7b164096da229601afb2ca8867d3b878e491bd7279148e5793aa",
    "m1510_test": "29d906ac7bca09f21f4d0081e254ecf0b6ff9c291c48ba6483117acc0fae091c",
    "m1510_contract": "88203261b26abee15ec57430e46cef7b4225f53fbb67abe9d18fc87c82d1abd7",
    "m1323_source": "0481e39372ffe19cd3cff8d5053c9eae8326de4fb5ac61bd9e42527a3ad3a12a",
    "m1323_contract": "e4df50fed6068b0f384693044705b30f595d41d70dce78e738cb36a98e24cecc",
    "m1323_author_review": "022fc0ddc5e6de5907f4033a08d968e76db5a903c544687dca52538059f6c1d9",
    "m1323_author_manifest": "83cd60889bff6f8211ddd3819233f5eb267c7fb25d81d0af8a36767f60215702",
    "m1323_author_outer": "e010a86648b93aecb4614d1a12f67be9d9cb4d47961b941e64840481d5f2c28b",
    "m1324_review": "79683ae29e70bd8272073c28ecbe26290c91201524d82a72a83f7ffc8ac719a2",
    "m1324_manifest": "bec10a857db964f94919aaa20d2aa603b7b0521b427164f225cda8d54b730a4f",
    "m1324_outer": "3e3bdb0d13089de323fd6c2c723ae263014cd9a3005ff7fabcfe34bede20e4ea",
    "m1321_source": "52fb82ab1e4262d6ce838f28a443ce82c6deba00678f9c65fb8227ac30702d85",
    "m1321_contract": "4dde544db5b8f32facbe5fdb10c8adb52d6abb19ca65c4dca7f3b2cce9f06f5c",
    "m1321_author_review": "6c4b14e5698580465821f288503aa65b4a64ca4447ee4c1c6563bfaabfd45e1b",
    "m1321_author_manifest": "fc661b29e410cd7ce1b9474685a86ff5061800f6aa1f3662d26e76dc368f2200",
    "m1321_author_outer": "989f16cd2b9739ef22578e435a4abf6f0cf6854760fff41f38185ac19db23497",
    "m1322_review": "c8fa3f9a80812af3f3cdd4cb439dd5ad110538ff8a86e746e1a5420a106bb717",
    "m1322_manifest": "ee45fa6d7ddc75316d9212f4ef3972277524a89d371d1f0efbd937b6cda8319c",
    "m1322_outer": "d07a2391c667a2b91b2f0f90c4451e0203d2ff8d890fd1e9713e68b2f8b46048",
    "m1501_source": "0c271bba3dfa57940b0ebe5a2ddf980d15f058b5ea25244aec5ead77d8146c83",
    "m1501_test": "0a0b2b5b58ccd8ae59f774b616a00510ffd99a636a794ad74f1dbb234c4f45b2",
    "m1501_contract": "e458cbe50c79a1faf659ed8329657978e6bcad7f0efb2fe91c3f016bc4a29dfb",
    "m1512_review": "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    "m1512_manifest": "2af7a59b6a4df07dc6047c0d48c52b7798b7f0803e31e290b2ad842e6c154b81",
    "m1512_outer": "ccbcd7bf1b99fd944062a6fb220d7ec719d96da91c190697db125cbd4ad58f7c",
    "m1513_review": "1eb36a76fac29d5d15607dbb4ee3f9a434c4b0686843acac11f18116b48c7aaa",
    "m1513_manifest": "966ba95baf00f698b6ca1fb8613afbfb78e40d2a70223f0a72bd4a87dcea04fa",
    "m1513_outer": "dc19cacbbb5ecae7f0327fd17b310be79a3b144937be7f289c25eb6f64794832",
    "m1458_author_review": "435d6d075fef043b01d8793d7517d1aeb85fba09cd02ebd8520f258573bf1ebe",
    "m1458_author_manifest": "6690fec9c33c1754c54edfdf5cf2a64a94bc1ec1bb449b4dd9351961622bcbe0",
    "m1458_author_outer": "833705c4f148d10950c0f66392248ab8ac93722237bd53dc0b67d87ad01a25cd",
    "capture_manifest": "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
    "capture_outer": "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
    "ordered": "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1511_bound_m1510", SOURCE)
T = load("m1511_bound_m1510_tests", TEST)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise RuntimeError("duplicate key")
            output[key] = value
        return output
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token:
                      (_ for _ in ()).throw(RuntimeError(token)))


def verify_review(root: Path, pins: tuple[str, str, str], status: str) -> dict:
    review_sha, manifest_sha, outer_sha = pins
    if (sha(root / "review.json"), sha(root / "SHA256SUMS"),
            sha(root / "SHA256SUMS.seal.sha256")) != pins:
        raise RuntimeError("sealed review identity drift: " + root.name)
    if (root / "SHA256SUMS.seal.sha256").read_text().split() != [
            manifest_sha, "SHA256SUMS"]:
        raise RuntimeError("outer seal content drift: " + root.name)
    value = strict_json(root / "review.json")
    if value.get("status") != status:
        raise RuntimeError("sealed review status drift: " + root.name)
    return value


def write_payload(root: Path, words: list[int], stem: str, ordinal: int) -> dict:
    raw = b"".join(struct.pack("<I", word) for word in words)
    plane_bytes = (len(words) + 7) // 8
    positive = bytearray(plane_bytes); negative = bytearray(plane_bytes)
    for index, word in enumerate(words):
        if word:
            plane = negative if word & 0x80000000 else positive
            plane[index >> 3] |= 1 << (index & 7)
    compressed = root / (stem + ".fp32.zlib")
    support = root / (stem + ".support_sign.le.bitpack")
    compressed.write_bytes(zlib.compress(raw)); support.write_bytes(positive + negative)
    return {
        "global_call_ordinal": 0, "global_order": 0,
        "global_sample_id": 10, "sequence": "s", "sample_key": "k",
        "source_sha256": "a" * 64, "module_ordinal": ordinal,
        "module": M.M1323.MODULES[ordinal], "shape": [1, 1, 1, 1, len(words)],
        "compressed_fp32": compressed.name, "compressed_sha256": sha(compressed),
        "support_sign": support.name, "support_sign_sha256": sha(support),
        "raw_fp32_sha256": hashlib.sha256(raw).hexdigest(),
        "positive_plane_bytes": plane_bytes, "negative_plane_bytes": plane_bytes,
    }


def rejected(thunk) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def main() -> int:
    checks: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []
    def check(name: str, condition: bool, category: str) -> None:
        checks.append({"name": name, "pass": bool(condition), "category": category})
    def attack(name: str, thunk, category: str) -> None:
        caught = rejected(thunk)
        attacks.append({"name": name, "rejected": caught,
                        "false_negative": not caught, "category": category})

    exact_paths = {
        "m1510_source": SOURCE, "m1510_test": TEST, "m1510_contract": CONTRACT,
        "m1323_source": M.M1323_SOURCE,
        "m1323_contract": HW / "contracts/m1323_ep34_decoder_capture_adapter_source_contract_r1_20260831.json",
        "m1321_source": M.M1323.M1321_SOURCE,
        "m1321_contract": HW / "contracts/m1321_ep34_decoder_capture_adapter_source_contract_r1_20260831.json",
        "m1501_source": M.M1501_SOURCE,
        "m1501_test": HW / "tests/test_hammer_m1501_m1458_motion_ep34_live93_capture_result_safe_audit_source.py",
        "m1501_contract": HW / "contracts/m1501_m1458_motion_ep34_live93_capture_result_safe_audit_source_contract_r1_20260831.json",
        "docs359": DOCS359,
    }
    for label, path in exact_paths.items():
        check("exact_" + label, sha(path) == PINS[label], "identity")
    contract = strict_json(CONTRACT)
    check("contract_source_identity", contract["source"]["sha256"] == PINS["m1510_source"],
          "identity")
    check("contract_test_identity", contract["test"]["sha256"] == PINS["m1510_test"],
          "identity")
    check("contract_claim_boundary", contract["claim_boundary"] == M.CLAIM_BOUNDARY,
          "claim")

    roots = HW / "reviews"
    authorities = {
        "m1321_author": verify_review(
            roots / "m1321_ep34_decoder_capture_adapter_source_author_r1_20260831",
            (PINS["m1321_author_review"], PINS["m1321_author_manifest"],
             PINS["m1321_author_outer"]),
            "PASS_AUTHOR_SOURCE_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED"),
        "m1322_failure": verify_review(
            roots / "m1322_m1321_ep34_decoder_adapter_source_hammer_r1_20260831",
            (PINS["m1322_review"], PINS["m1322_manifest"], PINS["m1322_outer"]),
            "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"),
        "m1323_author": verify_review(
            roots / "m1323_ep34_decoder_capture_adapter_source_author_r1_20260831",
            (PINS["m1323_author_review"], PINS["m1323_author_manifest"],
             PINS["m1323_author_outer"]),
            "PASS_SOURCE_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_PRODUCTION"),
        "m1324_hammer": verify_review(
            roots / "m1324_m1323_ep34_decoder_adapter_source_hammer_r1_20260831",
            (PINS["m1324_review"], PINS["m1324_manifest"], PINS["m1324_outer"]),
            "PASS_M1324_M1323_SOURCE_HAMMER__ACTUAL_RESULT_SUCCESSOR_ALLOWED"),
        "m1458_author": verify_review(
            roots / "m1458_m1434_motion_ep34_live93_production_runner_source_author_r1_20260831",
            (PINS["m1458_author_review"], PINS["m1458_author_manifest"],
             PINS["m1458_author_outer"]),
            "PASS_SOURCE_AUTHOR__M1450_M1451_FAILURE_BOUND__M1461_DIFFERENT_AUTHOR_BLIND_REQUIRED__NO_LAUNCH"),
        "m1512": verify_review(
            roots / "m1512_m1501_m1458_ep34_capture_source_result_independent_hammer_r1_20260831",
            (PINS["m1512_review"], PINS["m1512_manifest"], PINS["m1512_outer"]),
            "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT"),
        "m1513": verify_review(
            roots / "m1513_m1512_m1458_ep34_production_provenance_addendum_r1_20260831",
            (PINS["m1513_review"], PINS["m1513_manifest"], PINS["m1513_outer"]),
            "PASS_M1513_COMPLETE_M1458_EP34_PRODUCTION_PROVENANCE"),
    }
    check("authority_statuses", len(authorities) == 7, "authority")
    check("m1513_binds_m1512",
          authorities["m1513"]["bindings"]["m1512_review_sha256"] ==
          PINS["m1512_review"], "authority")
    check("m1512_binds_m1501",
          authorities["m1512"]["bindings"]["m1501_source_sha256"] ==
          PINS["m1501_source"], "authority")
    check("capture_manifest_exact", sha(M.CAPTURE_ROOT / "SHA256SUMS") ==
          PINS["capture_manifest"], "capture_seal")
    check("capture_outer_exact", sha(M.CAPTURE_ROOT / "SHA256SUMS.seal.sha256") ==
          PINS["capture_outer"], "capture_seal")
    check("ordered_exact", sha(M.CAPTURE_ROOT / "unified_ordered_records.jsonl") ==
          PINS["ordered"], "capture_seal")

    M.validate_source_policy()
    check("source_self_check", True, "source")
    stream = io.StringIO()
    replay = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(T))
    check("author_tests_9", replay.testsRun == 9 and not replay.failures and
          not replay.errors, "source")

    # The only expensive operation: full sequential 120-call capture audit.
    result = M.audit_capture(M.CAPTURE_ROOT)
    calls = result["calls"]
    check("actual_status", result["status"] == M.STATUS, "actual")
    check("actual_population", result["population"] == {
        "samples": 30, "calls": 120, "modules": 4,
        "global_sample_ids": [10, 39]}, "actual")
    check("ordered_graph_identity", result["ordered_identity"]["ordered_rows"] == 9880
          and result["ordered_identity"]["all_sample_sequences_equal"] is True,
          "ordered_graph")
    check("exact_call_order", [(row["global_sample_id"], row["module_ordinal"])
          for row in calls] == [(sample, ordinal) for sample in range(10, 40)
                                for ordinal in range(4)], "actual")
    for ordinal, expected in M.EXPECTED_WORDS.items():
        rows = [row for row in calls if row["module_ordinal"] == ordinal]
        check("layer_%d_30_calls" % ordinal, len(rows) == 30, "layer")
        check("layer_%d_word" % ordinal,
              {row["positive_word_uint32"] for row in rows} == {expected}, "layer")
        check("layer_%d_shape" % ordinal,
              all(row["shape"] == list(M.M1323.SHAPES[ordinal]) for row in rows),
              "shape")
    check("negative_zero", sum(row["negative_count"] for row in calls) == 0,
          "numeric")
    check("nonfinite_zero", sum(row["nonfinite_count"] for row in calls) == 0,
          "numeric")
    check("sha_roundtrip", all(row["raw_fp32_sha256"] and
          row["compressed_sha256"] and row["support_sign_sha256"] for row in calls),
          "sha")
    check("support_extent", all(row["positive_plane_bytes"] ==
          row["negative_plane_bytes"] == (row["elements"] + 7) // 8
          for row in calls), "support_plane")
    call_digest = hashlib.sha256(json.dumps(
        calls, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    # Synthetic payload mutations independently exercise all semantic gates.
    with tempfile.TemporaryDirectory(prefix="m1511_payload_") as name:
        root = Path(name)
        for ordinal in (0, 1):
            row = write_payload(root, [0, M.EXPECTED_WORDS[ordinal], 0x3F000000],
                                "multi%d" % ordinal, ordinal)
            attack("multi_theta_d%d" % ordinal,
                   lambda row=row: M.audit_call_payload(root, row), "multi_theta")
        for label, word in (("negative", 0xBF800000), ("nonfinite", 0x7F800000)):
            row = write_payload(root, [0, word], label, 0)
            attack(label, lambda row=row: M.audit_call_payload(root, row), label)
        for ordinal in (2, 3):
            row = write_payload(root, [0, 0x3F7FFFFF], "nonone%d" % ordinal, ordinal)
            attack("d%d_non_one" % ordinal,
                   lambda row=row: M.audit_call_payload(root, row), "d2d3_non1")
        for field in ("compressed_sha256", "raw_fp32_sha256", "support_sign_sha256"):
            row = write_payload(root, [0, M.EXPECTED_WORDS[0]], "sha_" + field, 0)
            row[field] = "0" * 64
            attack("sha_" + field,
                   lambda row=row: M.audit_call_payload(root, row), "sha")
        row = write_payload(root, [0, M.EXPECTED_WORDS[0]], "shape", 0)
        row["shape"][-1] = 3
        attack("shape_extent", lambda: M.audit_call_payload(root, row), "shape")
        row = write_payload(root, [M.EXPECTED_WORDS[0]], "padding", 0)
        support = root / row["support_sign"]
        payload = bytearray(support.read_bytes()); payload[0] |= 0x80
        support.write_bytes(payload); row["support_sign_sha256"] = sha(support)
        attack("support_padding", lambda: M.audit_call_payload(root, row), "padding")

    mutated = copy.deepcopy(calls)
    mutated[0]["positive_word_uint32"] ^= 1
    attack("cross_call_word_drift", lambda: M.summarize_layers(mutated), "call_drift")
    attack("call_missing", lambda: M.summarize_layers(calls[:-1]), "call_drift")
    mutated = copy.deepcopy(calls); mutated[4]["global_sample_id"] = 10
    attack("sample_order_drift", lambda: M.summarize_layers(mutated), "call_drift")

    ordered_path = M.CAPTURE_ROOT / "unified_ordered_records.jsonl"
    records = [M.M1323.strict_json_text(line) for line in
               ordered_path.read_text().splitlines()]
    inventory = M.M1323.frozen_inventory_names(); cohort = M.M1323.expected_cohort()
    mutated = copy.deepcopy(records); mutated[1]["global_order"] = 0
    attack("ordered_global_order_drift", lambda: M.M1323.decoder_rows_from_ordered(
        mutated, inventory, cohort), "ordered_graph")
    mutated = list(records); replacement = copy.deepcopy(mutated[300])
    replacement["global_order"] = mutated[301]["global_order"]; mutated[301] = replacement
    attack("ordered_ignored_duplicate", lambda: M.M1323.decoder_rows_from_ordered(
        mutated, inventory, cohort), "ordered_graph")
    indices = [index for index, row in enumerate(records[247:494], start=247)
               if row["category"] not in {"c1_conv3x3", "decoder_convtranspose"}][:2]
    mutated = list(records); a, b = indices
    first, second = copy.deepcopy(mutated[a]), copy.deepcopy(mutated[b])
    first["global_order"], second["global_order"] = b, a
    mutated[a], mutated[b] = second, first
    attack("ordered_module_sequence_drift", lambda: M.M1323.decoder_rows_from_ordered(
        mutated, inventory, cohort), "ordered_graph")

    precheck = strict_json(PRECHECK)
    check("m1511_fresh_before_creation",
          precheck["m1511_actual_result_hammer_namespace_absent"] is True,
          "freshness")
    p0 = sum(not row["rejected"] for row in attacks)
    p1 = sum(not row["pass"] for row in checks)
    layer_rows = [{
        "module_ordinal": row["module_ordinal"], "module": row["module"],
        "calls": row["calls"], "word_hex": row["word_hex"],
        "word_uint32": row["word_uint32"], "float32": row["float32"],
    } for row in result["layer_scale_words"]]
    output = {
        "schema": "m1511_m1510_ep34_decoder_layer_constant_actual_result_hammer_r1_v1",
        "status": ("PASS_M1511_M1510_EP34_DECODER_LAYER_CONSTANT_ACTUAL_RESULT"
                   if p0 == 0 and p1 == 0 else "FAIL_DO_NOT_MATERIALIZE"),
        "passed_check_names": [row["name"] for row in checks if row["pass"]],
        "failed_check_names": [row["name"] for row in checks if not row["pass"]],
        "attack_category_counts": {
            category: sum(row["category"] == category for row in attacks)
            for category in sorted({row["category"] for row in attacks})},
        "false_negative_names": [row["name"] for row in attacks
                                 if not row["rejected"]],
        "actual_result": {
            "capture_manifest_sha256": PINS["capture_manifest"],
            "capture_outer_sha256": PINS["capture_outer"],
            "ordered_jsonl_sha256": PINS["ordered"],
            "ordered_identity": result["ordered_identity"],
            "population": result["population"],
            "layer_scale_words": layer_rows,
            "calls_canonical_sha256": call_digest,
            "total_elements": sum(row["elements"] for row in calls),
            "total_zero_count": sum(row["zero_count"] for row in calls),
            "total_positive_count": sum(row["positive_count"] for row in calls),
            "total_negative_count": sum(row["negative_count"] for row in calls),
            "total_nonfinite_count": sum(row["nonfinite_count"] for row in calls),
            "total_positive_plane_bytes": sum(row["positive_plane_bytes"] for row in calls),
            "total_negative_plane_bytes": sum(row["negative_plane_bytes"] for row in calls),
            "unique_raw_sha256": len({row["raw_fp32_sha256"] for row in calls}),
            "unique_compressed_sha256": len({row["compressed_sha256"] for row in calls}),
            "unique_support_sha256": len({row["support_sign_sha256"] for row in calls}),
            "negative_and_nonfinite_zero": True,
            "shape_padding_support_sha_validated_for_all_calls": True,
        },
        "summary": {
            "checks_passed": sum(row["pass"] for row in checks),
            "checks_total": len(checks), "mutations_rejected":
            sum(row["rejected"] for row in attacks),
            "mutations_total": len(attacks), "false_negatives": p0,
            "failed_checks": p1, "source_tests_run": replay.testsRun,
            "source_test_failures": len(replay.failures) + len(replay.errors),
            "actual_calls_audited": len(calls),
        },
        "authorization": {
            "m1516_materializer_release_chain": p0 == 0 and p1 == 0,
            "materialization_now": False, "production": False,
            "eda": False, "gpu": False, "remote": False,
        },
        "claim_boundary": dict(M.CLAIM_BOUNDARY),
        "execution": {"workers": 1, "eda": 0, "gpu": 0, "remote": 0,
                      "checkpoint_writes": 0, "materializations": 0},
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0 if p0 == 0 and p1 == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
