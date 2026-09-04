import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ds_flm", ROOT / "scripts/evaluate_ds_flm_selector.py"
)
assert SPEC and SPEC.loader
ds_flm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ds_flm)


class DsFlmSelectorTest(unittest.TestCase):
    def test_descriptor_factorization_and_boundary_invariance(self):
        values = ds_flm.descriptors(ds_flm.load_rows())
        self.assertEqual(len(values), 36)
        self.assertEqual(
            sum(len(item["lane_major"]) for item in values), 1494
        )
        for item in values:
            self.assertEqual(
                set(item["lane_major"]), set(item["gate_major"])
            )
            self.assertEqual(
                item["lane_major"][0], item["gate_major"][0]
            )
            self.assertEqual(
                item["lane_major"][-1], item["gate_major"][-1]
            )

    def test_scan_bitmap_cost_includes_gate_major_reload(self):
        self.assertEqual(
            ds_flm.scan_state_toggles(7, 3, "lane")["bitmap"], 14
        )
        self.assertEqual(
            ds_flm.scan_state_toggles(7, 3, "gate")["bitmap"], 42
        )

    def test_lru_miss_count_is_existing_strong_baseline(self):
        value = ds_flm.evaluate(ds_flm.load_rows())
        self.assertEqual(value["per_way"]["4"]["product_misses"], 499)
        self.assertEqual(value["per_way"]["6"]["product_misses"], 156)
        self.assertEqual(value["per_way"]["8"]["product_misses"], 156)

    def test_selector_never_exceeds_best_static_for_model(self):
        value = ds_flm.evaluate(ds_flm.load_rows())
        for item in value["per_way"].values():
            for row in item["sweep"]:
                self.assertLessEqual(
                    row["selector_total"], row["best_static"]
                )


if __name__ == "__main__":
    unittest.main()
