#!/usr/bin/env python3
"""Directed M1057-attack regression for source-only M1064."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/run_m1064_c1_frozen_exact_1rw_replay_source.py"
SPEC = importlib.util.spec_from_file_location("m1064_test_source", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load M1064 source")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1064Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record0 = M.build_frozen_record(0, [1] * 64)

    def test_small_oracle(self):
        self.assertEqual(
            M.small_oracle()["status"],
            "PASS_M1064_SMALL_ORACLE__M1065_REQUIRED_NO_FULL_REPLAY",
        )

    def test_exact_double_sealed_contract(self):
        value = M.validate_sealed_contract()
        self.assertEqual(value["contract_sha256"], M.CONTRACT_SHA)
        self.assertEqual(value["outer_seal_file_sha256"], M.CONTRACT_OUTER_SHA)

    def test_temporary_unsealed_contract_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as stream:
            json.dump({"status": "PASS_M1064_SEALED_CONTRACT_SOURCE_ONLY__M1065_REQUIRED_NO_LAUNCH",
                       "launch_now": False, "max_attempts_now": 0}, stream)
            stream.flush()
            with self.assertRaisesRegex(RuntimeError, "canonical"):
                M.validate_sealed_contract(Path(stream.name))

    def test_capacity_is_zero_argument_internal_derivation(self):
        self.assertEqual(len(inspect.signature(M.derive_physical_capacity).parameters), 0)
        value = M.derive_physical_capacity()
        self.assertEqual(value["psum"]["bytes"], 122_880)
        self.assertEqual(value["weight"]["bytes"], 49_152)
        self.assertEqual(value["parent_plus_other"]["bytes"], 42_880)
        self.assertEqual(value["derived_total_bytes"], 214_912)
        self.assertEqual(value["derived_margin_bytes"], 30_848)
        with self.assertRaises(TypeError):
            M.derive_physical_capacity(0)
        self.assertEqual(list(inspect.signature(M.replay_frozen_sample).parameters),
                         ["records"])
        with self.assertRaisesRegex(RuntimeError, "exact nonempty"):
            M.replay_frozen_sample([])

    def test_boolean_service_count_rejected(self):
        receipt = json.loads(json.dumps(M.M1016.common_receipt(0, 64)))
        receipt["counts"]["dma"] = True
        with self.assertRaisesRegex(RuntimeError, "bool"):
            M.validate_receipt_exact(receipt, 0, 64)

    def test_extra_receipt_key_or_coverage_boolean_rejected(self):
        receipt = json.loads(json.dumps(M.M1016.common_receipt(0, 64)))
        receipt["coverage_pass"] = True
        with self.assertRaisesRegex(RuntimeError, "schema"):
            M.validate_receipt_exact(receipt, 0, 64)

    def test_duplicate_json_key_rejected(self):
        payload = json.dumps(M.M1016.common_receipt(0, 64))
        payload = payload[:-1] + ',"task":0}'
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            M.parse_receipt_json_for_attack(payload, 0, 64)

    def test_equal_empty_coverage_rejected(self):
        self.assertEqual(len(inspect.signature(M.FrozenCoverage).parameters), 0)
        with self.assertRaises(TypeError):
            M.FrozenCoverage(next_task_id=M.TASKS)
        proof = M.FrozenCoverage().proof()
        self.assertFalse(proof["full_coverage_pass"])
        self.assertFalse(proof["checks"]["nonempty"])

    def test_duplicate_and_out_of_order_task_ids_rejected(self):
        coverage = M.FrozenCoverage()
        coverage.consume(self.record0)
        with self.assertRaisesRegex(RuntimeError, "out-of-order"):
            coverage.consume(self.record0)
        record2 = M.build_frozen_record(2, [1] * 64)
        with self.assertRaisesRegex(RuntimeError, "out-of-order"):
            coverage.consume(record2)

    def test_three_design_id_row_preprocess_mismatch_rejected(self):
        receipts = dict(self.record0.design_receipts)
        old = receipts["strongest_zero"]
        receipts["strongest_zero"] = M.DesignTaskReceipt(
            99, 7, old.row_count, old.preprocess_cycles + 1,
            old.common_receipt,
            M.M1056.TaskPlan(99, old.preprocess_cycles + 1,
                             old.plan.work_cycles, 7),
        )
        bad = replace(self.record0, design_receipts=receipts)
        with self.assertRaisesRegex(RuntimeError, "receipt ID/row/preprocess"):
            M.validate_frozen_record(bad)

    def test_three_design_receipt_mismatch_rejected(self):
        receipts = dict(self.record0.design_receipts)
        old = receipts["same_coordinate_bit"]
        bad_common = json.loads(json.dumps(old.common_receipt))
        bad_common["counts"]["dma"] += 1
        receipts["same_coordinate_bit"] = replace(old, common_receipt=bad_common)
        with self.assertRaisesRegex(RuntimeError, "frozen M1016"):
            M.validate_frozen_record(replace(self.record0, design_receipts=receipts))

    def test_missing_or_extra_design_rejected(self):
        missing = dict(self.record0.design_receipts)
        missing.pop("candidate")
        with self.assertRaisesRegex(RuntimeError, "population"):
            M.validate_frozen_record(replace(self.record0, design_receipts=missing))
        extra = dict(self.record0.design_receipts)
        extra["extra"] = next(iter(extra.values()))
        with self.assertRaisesRegex(RuntimeError, "population"):
            M.validate_frozen_record(replace(self.record0, design_receipts=extra))

    def test_bool_mask_and_wrong_tile_length_rejected(self):
        masks = [1] * 64
        masks[0] = True
        with self.assertRaisesRegex(RuntimeError, "mask"):
            M.build_frozen_record(0, masks)
        with self.assertRaisesRegex(RuntimeError, "mask"):
            M.build_frozen_record(0, [1] * 63)

    def test_geometry_boundaries_and_tail_count(self):
        self.assertEqual(M.decode_task_id(0), (0, 0, 0, 0))
        self.assertEqual(M.decode_task_id(M.TASKS - 1), (9, 3, 46, 431))
        self.assertEqual(M.row_count_for_chunk(45), 64)
        self.assertEqual(M.row_count_for_chunk(46), 56)
        self.assertEqual(M.TASKS, 812_160)
        self.assertEqual(M.TASKS_PER_SAMPLE, 81_216)

    def test_frozen_digest_counts_and_claim_boundary(self):
        self.assertEqual(M.EXPECTED_SERVICE_DIGEST,
                         "a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea")
        self.assertEqual(M.EXPECTED_SERVICES, {
            "psum": 12_994_560, "weight": 70_853_184,
            "source": 51_840_000, "dma": 1_476_108, "commit": 960_000,
        })
        boundary = M.small_oracle()["claim_boundary"]
        for key in ("m1065_passed", "launch_now", "full_51840000_replay",
                    "capacity_only_214912B_admitted", "matched_cycles_admitted",
                    "speedup_admitted", "rtl_cycles", "paper_ppa_ready"):
            self.assertFalse(boundary[key])


if __name__ == "__main__":
    unittest.main()
