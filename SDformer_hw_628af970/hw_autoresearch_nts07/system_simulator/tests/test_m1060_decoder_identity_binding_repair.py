#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


HW = Path(__file__).resolve().parents[2]
DRIVER = HW / "system_simulator/scripts/execute_m1060_decoder_identity_binding_repair.py"
RUNNER = HW / "system_simulator/scripts/run_m1062_m1060_decoder_identity_binding_pilot_one_shot.sh"
CONTRACT = HW / "contracts/m1060_decoder_identity_binding_repair_contract_r1_20260830.json"
SPEC = importlib.util.spec_from_file_location("m1060_under_test", DRIVER)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def rejected(call):
    try:
        call()
    except (RuntimeError, OSError, ValueError, TypeError):
        return True
    return False


class M1060IdentityBindingTests(unittest.TestCase):
    def test_contract_freezes_exact_root_and_three_selected_records(self):
        value = M.contract_value()
        frozen = value["frozen_payload"]
        self.assertEqual(frozen["m699_manifest_sha256"],
            "e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0")
        self.assertEqual(frozen["m699_root_manifest_sha256"],
            "27b35748b81d32907410ada0fbecfaa869a6ce1c3039e94ab3da2e52a8f46053")
        self.assertEqual(frozen["m699_outer_seal_file_sha256"],
            "eaf975a9a1a4829b2c0a2251e7ef297abd53b83b30e23630e5ce51db5c5de18c")
        self.assertEqual([(r["layer"], r["relative_path"], r["packed_sha256"])
            for r in frozen["selected_records"]], [
            ("D0", "calls/s00_d0.binary.le.bitpack",
             "2af601cc112e1c39c1e850f7c776f71a28957d52df2164f0e79a988a9dbdf1be"),
            ("D2", "calls/s00_d2.binary.le.bitpack",
             "948d72523e23384e603a83739408aee4decbb1afb2c21cd8d2a77f3bff9a3e64"),
            ("D3", "calls/s00_d3.binary.le.bitpack",
             "0a8567d62df9aaf31ab19d7f1ad78366171be850a63562837d86f12570be86e3")])

    def test_pre_attempt_never_calls_full_payload_verifier(self):
        with mock.patch.object(M.M785, "verify_sealed_directory",
                side_effect=RuntimeError("payload verifier tripwire")):
            out = M.validate_pre_attempt_source(CONTRACT, RUNNER)
        self.assertEqual(out["status"],
            "PASS_M1060_PREATTEMPT_SOURCE_WITH_ZERO_PAYLOAD_MEMBER_ACCESS")
        self.assertFalse(out["payload_members_opened"])
        self.assertFalse(out["payload_members_statted"])
        self.assertFalse(out["payload_members_hashed"])

    def test_all_fake_sha_receipt_rejected_not_just_format_checked(self):
        context = M.synthetic_context()
        receipt = M.make_payload_receipt(context)
        receipt["attempt"] = {key: "f" * 64 for key in
            ("attempt_json_sha256", "runner_sha256", "contract_sha256")}
        receipt["attempt"]["m1061_authority"] = {key: "e" * 64 for key in
            ("review_sha256", "manifest_sha256", "outer_seal_file_sha256")}
        receipt["payload"]["m699_manifest_sha256"] = "d" * 64
        receipt["payload"]["m699_root_manifest_sha256"] = "c" * 64
        receipt["payload"]["m699_outer_seal_file_sha256"] = "b" * 64
        for row in receipt["payload"]["selected_records"]:
            row["packed_sha256"] = "a" * 64
            row["payload_member_sha256"] = "a" * 64
        self.assertTrue(rejected(lambda: M.validate_payload_receipt(receipt, context)))

    def test_nonexistent_path_and_relabel_rehash_receipts_rejected(self):
        context = M.synthetic_context()
        for mutation in ("nonexistent", "relabel_rehash"):
            receipt = M.make_payload_receipt(context)
            row = receipt["payload"]["selected_records"][0]
            if mutation == "nonexistent":
                row["relative_path"] = "calls/FORGED_DOES_NOT_EXIST.bitpack"
            else:
                row.update({"relative_path": "calls/renamed.bitpack",
                            "packed_sha256": "f" * 64,
                            "payload_member_sha256": "f" * 64})
            receipt["canonical_context_sha256"] = M.canonical_sha({
                "attacker_refreshed": receipt["payload"]})
            self.assertTrue(rejected(
                lambda receipt=receipt: M.validate_payload_receipt(receipt, context)))

    def test_verified_member_requires_existing_file_and_frozen_content_sha(self):
        with tempfile.TemporaryDirectory(prefix="m1060_member_") as td:
            root = Path(td); (root / "calls").mkdir()
            data = b"synthetic selected payload"
            target = root / "calls/x.bitpack"; target.write_bytes(data)
            frozen = {"layer": "D0", "population_id": M.M1048.POPULATION_ID,
                "sequence": M.M1048.SEQUENCE, "sample_id": 0, "module_index": 0,
                "route": "EXACT_BINARY_BITPACK", "relative_path": "calls/x.bitpack",
                "packed_sha256": M.sha256(target)}
            self.assertEqual(M.verified_member(root, frozen)["payload_member_sha256"],
                             M.sha256(target))
            bad = copy.deepcopy(frozen); bad["relative_path"] = "calls/missing.bitpack"
            self.assertTrue(rejected(lambda: M.verified_member(root, bad)))
            bad = copy.deepcopy(frozen); bad["packed_sha256"] = "0" * 64
            self.assertTrue(rejected(lambda: M.verified_member(root, bad)))

    def test_raw_layer_identity_is_cross_bound_to_verified_selected_member(self):
        context = M.synthetic_context()
        raw = {"layers": [{"layer": row["layer"],
            "record_identity": M.expected_raw_record(row),
            "verified_payload_member_sha256": row["payload_member_sha256"]}
            for row in context["payload"]["selected_records"]]}
        self.assertTrue(M.bind_raw_records(raw, context))
        for field, value in (("relative_path", "calls/relabel.bitpack"),
                             ("packed_sha256", "f" * 64)):
            attacked = copy.deepcopy(raw)
            attacked["layers"][1]["record_identity"][field] = value
            self.assertTrue(rejected(lambda attacked=attacked:
                                     M.bind_raw_records(attacked, context)))
        attacked = copy.deepcopy(raw)
        attacked["layers"][2]["verified_payload_member_sha256"] = "e" * 64
        self.assertTrue(rejected(lambda: M.bind_raw_records(attacked, context)))

    def test_assemble_requires_canonical_attempt_runner_contract_context(self):
        parameters = set(inspect.signature(M.assemble).parameters)
        self.assertEqual(parameters,
                         {"work", "attempt", "runner", "contract_sha", "authority"})
        self.assertEqual(set(inspect.signature(M.publish).parameters),
            {"work", "result", "attempt", "runner", "contract_sha", "authority"})
        source = DRIVER.read_text(encoding="utf-8")
        body = source[source.index("def assemble("):source.index("def publish(")]
        self.assertIn("build_canonical_context", body)
        self.assertIn("validate_raw(raw, context)", body)
        publish = source[source.index("def publish("):source.index("def quarantine(")]
        self.assertIn("build_canonical_context", publish)
        self.assertIn("validate_raw(raw, context)", publish)

    def test_contract_selected_record_relabel_and_extra_fields_rejected(self):
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        value["frozen_payload"]["selected_records"][0]["relative_path"] = \
            "calls/relabel.bitpack"
        self.assertTrue(rejected(lambda: M.validate_contract(value)))
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        value["frozen_payload"]["selected_records"][0]["attacker"] = "x"
        self.assertTrue(rejected(lambda: M.validate_contract(value)))

    def test_wrong_contract_pin_cannot_consume_attempt(self):
        attempt = M.RESULTS / M.ATTEMPT_NAME
        self.assertFalse(attempt.exists())
        self.assertTrue(rejected(lambda: M.consume_attempt(
            attempt, RUNNER, "0" * 64, {"synthetic": "authority"})))
        self.assertFalse(attempt.exists())

    def test_runtime_namespaces_and_direct_run_rejected(self):
        wrong = [M.RESULTS / ".m1062_wrong_attempt",
                 M.RESULTS / "m1062_wrong_result",
                 M.RESULTS / ".m1062_wrong_work",
                 M.RESULTS / "m1062_wrong_quarantine"]
        for path, role in zip(wrong, ("attempt", "result", "work", "quarantine")):
            self.assertTrue(rejected(lambda path=path, role=role:
                                     M.safe_path(path.resolve(), role)))
        work = M.RESULTS / ("." + M.RESULT_NAME + ".work.m1060test")
        if work.exists(): shutil.rmtree(work)
        work.mkdir(mode=0o700)
        try:
            self.assertTrue(rejected(lambda: M.run_pilot(
                M.RESULTS / M.ATTEMPT_NAME, work.resolve(), RUNNER,
                M.sha256(CONTRACT), {"synthetic": "authority"})))
        finally:
            shutil.rmtree(work)

    def test_runner_orders_attempt_before_payload_and_context_rechecks(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertLess(source.index("--consume-attempt"),
                        source.index("--validate-payload-after-attempt"))
        self.assertLess(source.index("--validate-payload-after-attempt"),
                        source.index("--run-pilot"))
        self.assertLess(source.index("--run-pilot"), source.index("--assemble"))
        assemble = source[source.index("--assemble"):]
        self.assertIn("--attempt", assemble)
        self.assertIn("--runner", assemble)
        self.assertIn("--expected-contract-sha", assemble)


if __name__ == "__main__":
    unittest.main()
