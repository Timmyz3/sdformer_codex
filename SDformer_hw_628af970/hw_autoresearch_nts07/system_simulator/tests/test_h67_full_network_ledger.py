import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_h67_ep35_full_network_ledger.py"
SPEC = importlib.util.spec_from_file_location("h67_full_network_ledger", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FullNetworkLedgerTest(unittest.TestCase):
    def test_multisample_attention_anchor_scales_exact_stage_means(self):
        receipt = {
            "schema": "h67_attention_multisample_vcs_anchor_v1",
            "status": "PASS_FRESH_VCS_RTL",
            "identity": "H67 ep35",
            "sample_count": 10,
            "rows": 1380,
            "rows_per_sample": 138,
            "tokens_per_row": 450,
            "fixed_cycles_total": 1057895,
            "rqtb_cycles_total": 899249,
            "fixed_rqtb_equal_mismatches": 0,
            "fixed_rqtb_emitted_mismatches": 0,
            "rtl_index_emitted_mismatches": 0,
            "stages": [
                {"stage": 0, "fixed_cycles_sum": 27866, "rqtb_cycles_sum": 23857},
                {"stage": 1, "fixed_cycles_sum": 43484, "rqtb_cycles_sum": 36325},
                {"stage": 2, "fixed_cycles_sum": 460806, "rqtb_cycles_sum": 383028},
                {"stage": 3, "fixed_cycles_sum": 525739, "rqtb_cycles_sum": 456039},
            ],
        }
        result = MODULE.attention_cycles_from_multisample_receipt(
            receipt, {"0": 440, "1": 120, "2": 30, "3": 10}
        )
        self.assertEqual(result["fixed_cycles_per_frame"], 3656069)
        self.assertEqual(result["rqtb_cycles_per_frame"], 3090731)
        self.assertAlmostEqual(result["speedup"], 1.1829140096630861)

    def test_atlif_temporal_mac_and_streaming_accumulator_contract(self):
        source = {
            "name": "unit.atlif",
            "calls": "100",
            "elements": "1000",
            "active": "250",
            "activity": "0.25",
            "temporal_steps": "2",
            "input_first_elements": "10",
            "parameter_entries": "7",
            "deployment_dead_result": "False",
        }
        rows = MODULE.build_atlif_rows(
            [source], {"atlif_lanes": 4, "atlif_accumulator_bits": 16}
        )
        row = rows[0]
        self.assertEqual(row["elements_per_frame"], 10)
        self.assertEqual(row["dense_macs_per_frame"], 20)
        self.assertEqual(row["cycles_at_config_lanes"], 5)
        self.assertEqual(row["full_temporal_output_buffer_bytes_per_frame"], 20)
        self.assertEqual(row["minimum_streaming_accumulator_bytes_per_call"], 10)

    def test_atlif_rejects_non_integral_temporal_row(self):
        source = {
            "name": "unit.bad_atlif",
            "calls": "100",
            "elements": "1000",
            "active": "250",
            "activity": "0.25",
            "temporal_steps": "3",
            "input_first_elements": "10",
            "parameter_entries": "13",
            "deployment_dead_result": "False",
        }
        with self.assertRaises(RuntimeError):
            MODULE.build_atlif_rows(
                [source], {"atlif_lanes": 4, "atlif_accumulator_bits": 16}
            )


if __name__ == "__main__":
    unittest.main()
