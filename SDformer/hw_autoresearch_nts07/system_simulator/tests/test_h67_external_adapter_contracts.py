import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_h67_external_adapter_contracts.py"
SPEC = importlib.util.spec_from_file_location("h67_external_adapter_contracts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExternalAdapterContractsTest(unittest.TestCase):
    def test_only_binary_linear_or_conv_maps_to_spiking_gemm(self):
        base = {
            "name": "op",
            "category": "ffn_expand",
            "activity_weighted_macs_per_frame": "100",
            "activity_cycles_at_config_lanes": "10",
        }
        binary = MODULE.map_operator(
            {**base, "operator": "Linear", "input_binary_packed_eligible": "True"}
        )
        nonbinary = MODULE.map_operator(
            {**base, "operator": "Linear", "input_binary_packed_eligible": "False"}
        )
        unsupported = MODULE.map_operator(
            {**base, "operator": "LayerNorm", "input_binary_packed_eligible": "True"}
        )
        self.assertTrue(binary["prosperity_structurally_eligible"])
        self.assertFalse(nonbinary["prosperity_structurally_eligible"])
        self.assertFalse(unsupported["phi_like_structurally_eligible"])

    def test_build_reports_weighted_coverage_without_cycles(self):
        rows = [
            {
                "name": "a", "category": "ffn", "operator": "Linear",
                "input_binary_packed_eligible": "True",
                "activity_weighted_macs_per_frame": "75",
                "activity_cycles_at_config_lanes": "8",
            },
            {
                "name": "b", "category": "ffn", "operator": "Linear",
                "input_binary_packed_eligible": "False",
                "activity_weighted_macs_per_frame": "25",
                "activity_cycles_at_config_lanes": "2",
            },
        ]
        result = MODULE.build(
            {
                "status": MODULE.LEDGER_STATUS,
                "attention": {
                    "fixed_cycles_per_frame": 10,
                    "rqtb_cycles_per_frame": 8,
                },
                "cycles_per_frame_model": {"fixed_total": 20},
            },
            rows,
        )
        self.assertEqual(result["coverage"]["eligible_mac_fraction"], 0.75)
        self.assertTrue(result["status"].endswith("CYCLES_BLOCKED"))


if __name__ == "__main__":
    unittest.main()
