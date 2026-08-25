from __future__ import print_function

import importlib.util
import json
import os
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT = os.path.join(
    ROOT, "hw_autoresearch_nts07", "system_simulator", "scripts",
    "analyze_m43_tile_resident_parent_delta_schedule.py")


class M43TileResidentParentDeltaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("m43_analyzer", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        self.old_temporal = self.module.ALLOW_TEMPORAL_PARENT
        self.module.ALLOW_TEMPORAL_PARENT = False

    def tearDown(self):
        self.module.ALLOW_TEMPORAL_PARENT = self.old_temporal

    def empty_masks(self):
        return [0] * (self.module.ROWS * self.module.TILES)

    def test_bank_issue_cycles_models_one_source_per_bank(self):
        self.assertEqual(self.module.bank_issue_cycles(0), 0)
        self.assertEqual(self.module.bank_issue_cycles((1 << 0) | (1 << 1)), 1)
        self.assertEqual(self.module.bank_issue_cycles((1 << 0) | (1 << 8)), 2)
        self.assertEqual(self.module.bank_issue_cycles((1 << 7) | (1 << 15)), 2)

    def test_bank_aware_parent_beats_lower_priority_local(self):
        masks = self.empty_masks()
        row = 1
        tile = 0
        masks[(row - 1) * self.module.TILES + tile] = 1 << 0
        masks[row * self.module.TILES + tile] = (1 << 0) | (1 << 8)
        name, parent, add_mask, subtract_mask = self.module.select_parent(
            masks, row, tile)
        self.assertEqual(name, "left")
        self.assertEqual(parent, 1 << 0)
        self.assertEqual(add_mask, 1 << 8)
        self.assertEqual(subtract_mask, 0)

    def test_temporal_parent_is_disabled_for_primary(self):
        masks = self.empty_masks()
        row = self.module.HEIGHT * self.module.WIDTH
        tile = 0
        masks[(row - self.module.HEIGHT * self.module.WIDTH) *
              self.module.TILES + tile] = (1 << 3) | (1 << 11)
        masks[row * self.module.TILES + tile] = (1 << 3) | (1 << 11)
        name = self.module.select_parent(masks, row, tile)[0]
        self.assertEqual(name, "local_zero")
        self.module.ALLOW_TEMPORAL_PARENT = True
        name, parent, add_mask, subtract_mask = self.module.select_parent(
            masks, row, tile)
        self.assertEqual(name, "previous_timestep")
        self.assertNotEqual(parent, 0)
        self.assertEqual(add_mask | subtract_mask, 0)

    def test_signed_delta_algebra(self):
        weights = [5, -7, 11, -13, 17, -19, 23, -29, 31]
        current = {0, 2, 5, 8}
        parent = {0, 1, 5, 7}
        additions = current - parent
        subtractions = parent - current
        direct = sum(weights[index] for index in current)
        reused = (sum(weights[index] for index in parent) +
                  sum(weights[index] for index in additions) -
                  sum(weights[index] for index in subtractions))
        self.assertEqual(direct, reused)

    def test_canonical_contract_and_upstream_identities(self):
        contract = self.module.validate_contract(self.module.DEFAULT_CONTRACT)
        self.assertEqual(contract["geometry"]["peak_product_adds_per_cycle"], 768)
        result, review, total_bytes = self.module.validate_int8_bridge(
            self.module.DEFAULT_M41_RESULT, self.module.DEFAULT_M41_REVIEW)
        self.assertEqual(total_bytes, 21233664)
        self.assertEqual(result["m40_schedule_bridge"][
            "checkpoint_tight_accumulator_signed_bits"], 19)
        self.assertIn("GO_CHECKPOINT_BOUND_MODEL_BRIDGE", review["status"])

    def test_contract_sha_mutation_is_rejected(self):
        old = self.module.EXPECTED_CONTRACT_SHA256
        try:
            self.module.EXPECTED_CONTRACT_SHA256 = "0" * 64
            with self.assertRaises(ValueError):
                self.module.validate_contract(self.module.DEFAULT_CONTRACT)
        finally:
            self.module.EXPECTED_CONTRACT_SHA256 = old

    def test_json_duplicate_and_nan_are_rejected(self):
        with tempfile.TemporaryDirectory() as tempdir:
            duplicate = os.path.join(tempdir, "duplicate.json")
            with open(duplicate, "w") as handle:
                handle.write('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                self.module.read_json(duplicate)
            nan_path = os.path.join(tempdir, "nan.json")
            with open(nan_path, "w") as handle:
                handle.write('{"a":NaN}')
            with self.assertRaises(ValueError):
                self.module.read_json(nan_path)

    def test_output_refuses_overwrite(self):
        payload = {"schema": "unit", "value": 1}
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "result.json")
            self.module.write_output(path, payload)
            with open(path, "rb") as handle:
                before = handle.read()
            with self.assertRaises(ValueError):
                self.module.write_output(path, payload)
            with open(path, "rb") as handle:
                self.assertEqual(before, handle.read())
            self.assertEqual(json.loads(before.decode("utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
