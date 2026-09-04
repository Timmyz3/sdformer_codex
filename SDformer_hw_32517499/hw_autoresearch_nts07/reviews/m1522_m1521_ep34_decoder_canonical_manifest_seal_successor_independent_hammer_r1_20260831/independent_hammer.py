#!/usr/bin/env python3
"""Independent synthetic/static hammer for M1521; no production or capture read.

This review deliberately does not trust the author tests.  It reconstructs a
120-call synthetic authority, invokes the public seal and post-publication
verifier through their internal-derivation path, and attacks both sides with
the M1517 semantic forgeries plus type, population, ordering, path and payload
mutations.  It never invokes the one-shot materializer or reads M1458.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = (HW / "system_simulator/scripts" /
          "build_m1521_ep34_decoder_canonical_manifest_seal_successor_source.py")
TEST = (HW / "system_simulator/tests" /
        "test_m1521_ep34_decoder_canonical_manifest_seal_successor_source.py")
CONTRACT = (HW / "contracts" /
            "m1521_ep34_decoder_canonical_manifest_seal_successor_source_contract_r1_20260831.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "1508607cc42fd3eb6c9aecebfe8fb25819db6b3aa353d533e794fae7e1e82e14",
    "test": "a58424e296e209aaa1b4541a94a8363aad5b131a4d8f3c8ca08e61c68be122cb",
    "contract": "527577c453c2f2dada71dc22332af6479f93d4ad89bb1b0e7c4a64a98da13a24",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
TINY_SHAPES = ((1, 1, 1, 1, 9),) * 4
POSITIVE = bytes((0x05, 0x01))
NEGATIVE = bytes((0x00, 0x00))
POSITIVE_SHA = hashlib.sha256(POSITIVE).hexdigest()
NEGATIVE_SHA = hashlib.sha256(NEGATIVE).hexdigest()
SUPPORT_SHA = hashlib.sha256(POSITIVE + NEGATIVE).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1522_independent_target", SOURCE)


def synthetic_enriched() -> dict:
    calls = []
    for ordinal in range(120):
        sample = 10 + ordinal // 4
        module = ordinal % 4
        calls.append({
            "global_call_ordinal": ordinal,
            "global_order": sample * 257 + 193 + module,
            "global_sample_id": sample,
            "sequence": "blind_sequence_%d" % (sample // 10),
            "sample_key": "blind_%02d" % sample,
            "module_ordinal": module,
            "module": M.M1516.M1510.M1323.MODULES[module],
            "shape": list(TINY_SHAPES[module]),
            "support_sign": "payloads/blind_%03d.support_sign.le.bitpack" % ordinal,
            "support_sign_sha256": SUPPORT_SHA,
            "source_support_sign_sha256": SUPPORT_SHA,
            "positive_plane_sha256": POSITIVE_SHA,
            "negative_zero_plane_sha256": NEGATIVE_SHA,
            "positive_plane_bytes": 2,
            "negative_plane_bytes": 2,
            "plane_bytes": 2,
            "positive_word_uint32": M.M1516.EXPECTED_SCALE_WORDS[module],
            "negative_count": 0,
            "nonfinite_count": 0,
        })
    layers = [{
        "module_ordinal": module,
        "module": M.M1516.M1510.M1323.MODULES[module],
        "calls": 30,
        "word_uint32": M.M1516.EXPECTED_SCALE_WORDS[module],
        "word_hex": "0x{:08x}".format(M.M1516.EXPECTED_SCALE_WORDS[module]),
        "all_calls_same_word": True,
    } for module in range(4)]
    return {
        "schema": M.M1516.M1510.SCHEMA,
        "status": M.M1516.M1510.STATUS,
        "capture_seal": {
            "sha256sums_sha256": M.M1516.CAPTURE_MANIFEST_SHA256,
            "outer_seal_sha256": M.M1516.CAPTURE_OUTER_SHA256,
        },
        "layer_scale_words": layers,
        "calls": calls,
    }


def canonical_manifest() -> dict:
    with mock.patch.object(M.M1516.M1510.M1323, "SHAPES", TINY_SHAPES):
        return M.expected_manifest_from_enriched(synthetic_enriched())


def write_stage(root: Path, manifest: dict) -> None:
    (root / "payloads").mkdir(parents=True)
    for row in manifest["records"]:
        path = root.joinpath(*Path(row["positive_output"]).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            M.M1516.write_exclusive(path, POSITIVE, 0o400)
    M.M1516.write_exclusive(
        root / "manifest.json",
        (json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n").encode(), 0o400)
    M.M1516.write_exclusive(root / "RUN_COMPLETE.txt", M.RUN_TOKEN.encode(), 0o400)


def weak_self_consistent_seal(root: Path) -> None:
    members = M._sealed_payload_files(root)
    lines = [M.sha256(path) + "  " + path.relative_to(root).as_posix()
             for path in members]
    M.M1516.write_exclusive(root / M.M1516.MANIFEST,
                            ("\n".join(lines) + "\n").encode(), 0o400)
    M.M1516.write_exclusive(
        root / M.M1516.OUTER,
        (M.sha256(root / M.M1516.MANIFEST) + "  " +
         M.M1516.MANIFEST + "\n").encode(), 0o400)


def rejected(function) -> bool:
    try:
        function()
    except BaseException:
        return True
    return False


def canonical_derivation_patches(enriched: dict):
    audit_marker = {"independent_audit_marker": True}
    audit = mock.patch.object(M.M1516.M1510, "audit_capture", return_value=audit_marker)
    enrich = mock.patch.object(M.M1516, "enrich_audit", return_value=enriched)
    shapes = mock.patch.object(M.M1516.M1510.M1323, "SHAPES", TINY_SHAPES)
    return audit_marker, audit, enrich, shapes


def public_preseal(root: Path, enriched: dict):
    marker, audit_patch, enrich_patch, shape_patch = canonical_derivation_patches(enriched)
    with audit_patch as audit_call, enrich_patch as enrich_call, shape_patch:
        receipt = M.seal_staging(root)
        audit_call.assert_called_once_with(M.M1516.CAPTURE)
        enrich_call.assert_called_once_with(marker, M.M1516.CAPTURE)
        return receipt


def public_postverify(root: Path, enriched: dict):
    marker, audit_patch, enrich_patch, shape_patch = canonical_derivation_patches(enriched)
    with audit_patch as audit_call, enrich_patch as enrich_call, shape_patch:
        receipt = M.verify_materialized_seal(root)
        audit_call.assert_called_once_with(M.M1516.CAPTURE)
        enrich_call.assert_called_once_with(marker, M.M1516.CAPTURE)
        return receipt


def semantic_attacks():
    return (
        ("scale", lambda x: x["records"][0].update(
            layer_scale_word_uint32=0x3F800000)),
        ("encoding", lambda x: x["records"][0].update(
            numeric_encoding="exact_binary")),
        ("weight_fold", lambda x: x["records"][0].update(weight_folding=True)),
        ("normalize", lambda x: x["records"][0].update(normalized=True)),
        ("coerce", lambda x: x["records"][0].update(coerced=True)),
        ("duplicate_global_order", lambda x: x["records"][1].update(
            capture_global_order=x["records"][0]["capture_global_order"])),
        ("performance_claim", lambda x: x["claim_boundary"].update(cycles=True)),
        ("canonical_path", lambda x: x["records"][0].update(
            positive_output="payloads/renamed_attack.bin")),
    )


def structural_attacks():
    def swap_records(value):
        value["records"][0], value["records"][1] = (
            value["records"][1], value["records"][0])

    def drop_record(value):
        value["records"].pop()

    def add_record(value):
        extra = copy.deepcopy(value["records"][-1])
        extra["global_call_ordinal"] = 120
        extra["positive_output"] = "payloads/c120_s39_d3.positive.le.bitpack"
        value["records"].append(extra)

    return (
        ("bool_as_call_ordinal", lambda x: x["records"][0].update(
            global_call_ordinal=False)),
        ("int_as_bool_flag", lambda x: x["records"][0].update(
            weight_folding=0)),
        ("bool_as_population_int", lambda x: x["population"].update(calls=True)),
        ("record_order_swap", swap_records),
        ("call_count_119", drop_record),
        ("call_count_121", add_record),
        ("record_payload_sha", lambda x: x["records"][0].update(
            positive_output_sha256="0" * 64)),
    )


def run() -> dict:
    checks: dict[str, bool] = {}
    mutations = 0
    checks["source_sha_exact"] = sha256(SOURCE) == EXPECTED["source"]
    checks["test_sha_exact"] = sha256(TEST) == EXPECTED["test"]
    checks["contract_sha_exact"] = sha256(CONTRACT) == EXPECTED["contract"]
    checks["docs359_sha_exact"] = sha256(DOCS359) == EXPECTED["docs359"]

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    checks["contract_source_test_identity"] = (
        contract.get("source", {}).get("sha256") == EXPECTED["source"] and
        contract.get("test", {}).get("sha256") == EXPECTED["test"])
    checks["contract_source_only"] = (
        contract.get("production_authorized") is False and
        contract.get("required_before_release") == {
            "m1522_independent_hammer": True, "m1523_exact_release": True})

    stream = io.StringIO()
    author_suite = unittest.defaultTestLoader.loadTestsFromModule(
        load("m1522_author_tests", TEST))
    author_result = unittest.TextTestRunner(stream=stream, verbosity=1).run(author_suite)
    checks["author_tests_11_pass"] = (
        author_result.wasSuccessful() and author_result.testsRun == 11)

    checks["m1517_failure_exact_bound"] = (
        M.verify_m1517_failure().get("status") == M.M1517_STATUS)
    checks["source_policy_pass"] = bool(M.validate_source_policy())

    seal_parameters = list(inspect.signature(M.seal_staging).parameters)
    verify_parameters = list(inspect.signature(M.verify_materialized_seal).parameters)
    checks["public_external_expected_unavailable"] = (
        seal_parameters == ["root"] and verify_parameters == ["root"] and
        rejected(lambda: M.seal_staging(Path("unused"), {})) and
        rejected(lambda: M.verify_materialized_seal(Path("unused"), {})))
    derive_source = inspect.getsource(M.derive_canonical_expected)
    checks["canonical_m1458_m1510_m1516_chain_static"] = all(token in derive_source for token in (
        "M1516.M1510.audit_capture(M1516.CAPTURE)",
        "M1516.enrich_audit(audit, M1516.CAPTURE)",
        "expected_manifest_from_enriched(enriched)"))
    materialize_source = inspect.getsource(M.materialize_canonical_once)
    checks["materializer_expected_internal_only"] = (
        list(inspect.signature(M.materialize_canonical_once).parameters) ==
        ["output", "attempt"] and "derive_canonical_expected()" in materialize_source)

    enriched = synthetic_enriched()
    expected = canonical_manifest()
    with tempfile.TemporaryDirectory(prefix="m1522_valid_") as directory:
        root = Path(directory) / "stage"; root.mkdir()
        write_stage(root, expected)
        first = public_preseal(root, enriched)
        second = public_postverify(root, enriched)
        checks["public_valid_preseal_and_postverify"] = (
            first == second and first.get("members") == 122 and
            first.get("canonical_paths") == 120 and first.get("full_tree_equal") is True)

    for attack_name, mutation in semantic_attacks() + structural_attacks():
        forged = copy.deepcopy(expected); mutation(forged); mutations += 1
        with tempfile.TemporaryDirectory(prefix="m1522_pre_%s_" % attack_name) as directory:
            root = Path(directory) / "stage"; root.mkdir(); write_stage(root, forged)
            checks["preseal_reject_" + attack_name] = rejected(
                lambda r=root: public_preseal(r, enriched))
            checks["preseal_no_seal_" + attack_name] = not (
                root / M.M1516.MANIFEST).exists()

        forged = copy.deepcopy(expected); mutation(forged); mutations += 1
        with tempfile.TemporaryDirectory(prefix="m1522_post_%s_" % attack_name) as directory:
            root = Path(directory) / "published"; root.mkdir(); write_stage(root, forged)
            weak_self_consistent_seal(root)
            checks["postverify_reject_" + attack_name] = rejected(
                lambda r=root: public_postverify(r, enriched))

    with tempfile.TemporaryDirectory(prefix="m1522_payload_bytes_") as directory:
        root = Path(directory) / "published"; root.mkdir(); write_stage(root, expected)
        weak_self_consistent_seal(root)
        victim = root / expected["records"][0]["positive_output"]
        victim.chmod(0o600); victim.write_bytes(b"forged")
        mutations += 1
        checks["postverify_reject_payload_byte_drift"] = rejected(
            lambda: public_postverify(root, enriched))

    with tempfile.TemporaryDirectory(prefix="m1522_payload_remove_") as directory:
        root = Path(directory) / "published"; root.mkdir(); write_stage(root, expected)
        victim = root / expected["records"][-1]["positive_output"]
        victim.unlink(); weak_self_consistent_seal(root)
        mutations += 1
        checks["postverify_reject_payload_population_119"] = rejected(
            lambda: public_postverify(root, enriched))

    with tempfile.TemporaryDirectory(prefix="m1522_payload_extra_") as directory:
        root = Path(directory) / "published"; root.mkdir(); write_stage(root, expected)
        M.M1516.write_exclusive(root / "payloads/extra.bin", POSITIVE, 0o400)
        weak_self_consistent_seal(root)
        mutations += 1
        checks["postverify_reject_payload_population_121"] = rejected(
            lambda: public_postverify(root, enriched))

    source_text = SOURCE.read_text(encoding="utf-8")
    execute_text = inspect.getsource(M.execute_once)
    checks["cli_has_no_production_switch"] = (
        "--materialize" not in source_text and "--production" not in source_text and
        rejected(lambda: M.main([])))
    checks["release_gate_precedes_materialization"] = (
        execute_text.index("verify_m1522_hammer") <
        execute_text.index("materialize_canonical_once") and
        execute_text.index("verify_m1517_failure") <
        execute_text.index("materialize_canonical_once"))
    checks["release_attempt_output_absent"] = (
        not os.path.lexists(str(M.FUTURE_RELEASE)) and
        not os.path.lexists(str(M.ATTEMPT)) and
        not os.path.lexists(str(M.OUTPUT)) and
        not any(M.OUTPUT.parent.glob(M.WORK_PREFIX + "*")))
    checks["no_remote_gpu_eda_tokens"] = all(token not in source_text for token in (
        "subprocess", "paramiko", "torch.cuda", "ssh ", "vcs", "dc_shell", "pt_shell"))

    failed = sorted(name for name, value in checks.items() if not value)
    return {
        "schema": "m1522_m1521_ep34_decoder_canonical_manifest_seal_independent_hammer_output_r1_v1",
        "target": {
            "source_sha256": EXPECTED["source"],
            "test_sha256": EXPECTED["test"],
            "contract_sha256": EXPECTED["contract"],
        },
        "checks": checks,
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "failed_checks": failed,
        "mutations": mutations,
        "author_tests": {
            "tests_run": author_result.testsRun,
            "passed": author_result.wasSuccessful(),
            "output": stream.getvalue(),
        },
        "verdict": ("PASS_M1522_M1521_CANONICAL_MANIFEST_SEAL__M1523_RELEASE_ONLY"
                    if not failed else
                    "FAIL_CLOSED__M1523_RELEASE_AND_PRODUCTION_BLOCKED"),
        "execution": {
            "capture_read": 0, "production": 0, "gpu": 0,
            "eda": 0, "remote": 0,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
