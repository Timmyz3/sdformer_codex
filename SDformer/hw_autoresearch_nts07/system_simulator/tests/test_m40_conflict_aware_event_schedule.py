#!/usr/bin/env python3

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_m40_conflict_aware_event_schedule.py"
SPEC = importlib.util.spec_from_file_location("m40_schedule", str(SCRIPT))
M40 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M40)
CONTRACT = M40.DEFAULT_CONTRACT
FROZEN_RESULT = M40.HW_ROOT / (
    "results/m40_conflict_aware_event_schedule_r3_20260822/"
    "m40_conflict_aware_event_schedule.json")


class M40ConflictAwareScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = M40.read_json(CONTRACT)
        cls.result = M40.build(CONTRACT)
        cls.frozen = M40.read_json(FROZEN_RESULT)

    def write_contract(self, root, payload, name):
        path = root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_frozen_result_rebuilds_type_exact(self):
        self.assertIsNone(M40.mismatch(self.result, self.frozen))
        self.assertEqual(self.result["identity"]["contract_sha256"],
                         M40.EXPECTED_CONTRACT_SHA256)

    def test_all_92160000_values_are_bit_exact_two_code(self):
        trace = self.result["real_source_trace"]
        self.assertEqual(trace["float32_values_bit_exact_mitered"], 92160000)
        self.assertEqual(trace["float32_value_bit_mismatches"], 0)
        self.assertTrue(trace[
            "numeric_values_reconstructable_from_bitmap_plus_layer_amplitude"])
        self.assertIn("40_OF_40", trace["amplitude_codebook_status"])

    def test_four_codes_equal_m35_thresholds(self):
        rows = self.result["real_source_trace"][
            "amplitude_codebook_m35_reconciliation"]
        expected = {
            M40.TARGETS[0]: ("3f7fff87", 16777095, 121),
            M40.TARGETS[1]: ("3f7fff70", 16777072, 144),
            M40.TARGETS[2]: ("3f7fff9f", 16777119, 97),
            M40.TARGETS[3]: ("3f7ffdb4", 16776628, 588),
        }
        self.assertEqual(set(rows), set(expected))
        for name, values in expected.items():
            row = rows[name]
            self.assertEqual((row["float32_bits_hex"], row["uq0p24_raw"],
                              row["m35_delta"]), values)
            self.assertEqual(row["values_mitered"], 23040000)
            self.assertEqual(row["bit_mismatches"], 0)

    def test_real_padding_aware_work_distribution(self):
        rows = self.result["real_source_trace"][
            "exact_work_lower_bound_distribution_by_line"]
        self.assertEqual(rows["Local"]["mean_exact"],
                         {"numerator": 741123776, "denominator": 10})
        self.assertEqual(rows["Local"]["p95_nearest_rank"], 74995872)
        self.assertEqual(rows["Motion"]["mean_exact"],
                         {"numerator": 1099147552, "denominator": 10})
        self.assertEqual(rows["Motion"]["p99_nearest_rank"], 110962768)
        self.assertGreater(rows["Motion"]["mean_exact"]["numerator"],
                           rows["Local"]["mean_exact"]["numerator"])

    def test_padding_table_excludes_pseudo_events(self):
        period, popcount, weighted = M40.bitmap_tables(15, 20)
        self.assertEqual(period, 75)
        self.assertEqual(popcount[1], 1)
        self.assertEqual(weighted[0][1], 4)
        # Spatial index 21 is an interior point.  Byte phase 2 starts at 16,
        # so bit five selects it and has the full nine valid destinations.
        self.assertEqual(weighted[2][1 << 5], 9)

    def test_small_oracle_exercises_conflict_credit_residency(self):
        row = self.result["executable_small_trace_reference"]["synthetic_result"]
        self.assertEqual(row["cycles"], 3)
        self.assertEqual(row["bank_conflict_event_deferrals"], 1)
        self.assertEqual(row["queue_credit_stall_events"], 1)
        self.assertEqual(row["weight_tile_evictions"], 2)
        self.assertEqual(row["events_offered"], row["events_retired"])
        self.assertEqual(row["events_lost"], 0)
        self.assertEqual(row["m35_integer_mismatches"], 0)

    def test_product_accumulator_and_late_scale_mutations_rejected(self):
        oracle = self.contract["synthetic_oracle"]
        cases = []
        events = copy.deepcopy(oracle["events"])
        events[0]["contribution_s16"] += 1
        cases.append((events, "product miter"))
        events = copy.deepcopy(oracle["events"])
        events[1]["accumulator_after_s32"] += 1
        cases.append((events, "accumulator chain"))
        events = copy.deepcopy(oracle["events"])
        events[1]["expected_scaled_s56"] += 1
        cases.append((events, "late-scale integer miter"))
        events = copy.deepcopy(oracle["events"])
        events[2]["motion_delta_direction"] = True
        cases.append((events, "integer type drift"))
        for events, pattern in cases:
            with self.assertRaisesRegex(ValueError, pattern):
                M40.schedule_events(events, oracle["config"])

    def test_flush_reentry_and_tile_overflow_rejected(self):
        oracle = self.contract["synthetic_oracle"]
        events = copy.deepcopy(oracle["events"])
        events[4]["flush_id"] = 0
        with self.assertRaisesRegex(ValueError, "flush boundary"):
            M40.schedule_events(events, oracle["config"])
        events = copy.deepcopy(oracle["events"])
        events[0]["weight_tile_bytes"] = 17
        events[1]["weight_tile_bytes"] = 17
        with self.assertRaisesRegex(ValueError, "exceeds residency"):
            M40.schedule_events(events, oracle["config"])

    def test_m22_is_not_misclassified_as_event_trace(self):
        row = self.result["m22_m23_trace_audit"]
        self.assertFalse(row["has_product_event_coordinates_and_operands"])
        self.assertIn("source_index", row["required_product_event_fields_missing"])
        self.assertTrue(all(value == 540 for value in
                            row["aggregate_rows_by_operator"].values()))

    def test_contract_type_and_population_forgeries_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            payload = copy.deepcopy(self.contract)
            payload["scheduler_contract"]["banks"] = 24.0
            cases.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["required_next_trace"][
                "real_schedule_must_fail_closed_until_complete"] = 1
            cases.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["source_handoff"][
                "individual_file_hashes_are_recursively_retained_in_packed_source_manifest"] = 1
            cases.append(payload)
            payload = copy.deepcopy(self.contract)
            payload["extra"] = False
            cases.append(payload)
            for index, forged in enumerate(cases):
                with self.assertRaisesRegex(ValueError, "recursive type-strict drift"):
                    M40.build(self.write_contract(
                        root, forged, "forged_{}.json".format(index)))

    def test_duplicate_keys_and_nonstandard_numbers_rejected(self):
        canonical = CONTRACT.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema":"forged",' + canonical.lstrip()[1:],
                                 encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                M40.build(duplicate)
            for index, token in enumerate(("NaN", "Infinity", "-Infinity")):
                path = root / "constant_{}.json".format(index)
                path.write_text(canonical.replace(
                    '"banks": 24', '"banks": {}'.format(token), 1),
                    encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "non-standard JSON"):
                    M40.build(path)

    def test_real_schedule_and_headline_claims_stay_closed(self):
        admission = self.result["admission"]
        self.assertTrue(admission[
            "exact_92160000_value_bitmap_codebook_bit_miter_admitted"])
        for key in M40.FORBIDDEN:
            self.assertFalse(admission[key], key)
        self.assertIsNone(self.result["real_source_trace"][
            "executable_cycle_mean_p95_p99"]["Local"])
        self.assertIsNone(self.result["real_source_trace"][
            "executable_cycle_mean_p95_p99"]["Motion"])

    def test_repeat_build_is_byte_identical(self):
        a = json.dumps(M40.build(CONTRACT), indent=2, sort_keys=True) + "\n"
        b = json.dumps(M40.build(CONTRACT), indent=2, sort_keys=True) + "\n"
        self.assertEqual(a.encode("utf-8"), b.encode("utf-8"))

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occupied.json"
            path.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                M40.write_output(path, {})
            self.assertEqual(path.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
