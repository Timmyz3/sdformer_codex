import unittest

from scripts.model_fcip_finite_resources import (
    current_postnorm_cycles,
    fcip_cycles,
    materialized_relation_storage,
)
from scripts.model_architecture_innovation_round12 import storage_ledger


class FcipFiniteResourceTest(unittest.TestCase):
    def test_width_only_helps_when_sink_width_also_increases(self):
        row = {
            "active_classes": 4,
            "final_gate_lane_terms": 8,
        }
        serial_sink = fcip_cycles(
            row,
            fragments=64,
            and_width=8,
            emit_width=1,
            product_width=8,
        )
        wide_sink = fcip_cycles(
            row,
            fragments=64,
            and_width=8,
            emit_width=8,
            product_width=8,
        )
        self.assertGreater(serial_sink, wide_sink)

    def test_materialized_relation_costs_more_storage(self):
        self.assertGreater(
            materialized_relation_storage(162),
            storage_ledger(162)["factorized_total_bits"],
        )

    def test_current_postnorm_contract(self):
        self.assertEqual(current_postnorm_cycles(tokens=162), 290)


if __name__ == "__main__":
    unittest.main()
