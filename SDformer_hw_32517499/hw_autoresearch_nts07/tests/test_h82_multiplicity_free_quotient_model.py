import unittest

from scripts.h82_multiplicity_free_quotient_model import (
    PAIRS,
    descriptor_count,
    model_point,
)


class H82MultiplicityFreeQuotientModelTest(unittest.TestCase):
    def test_descriptor_count_reaches_pair_local_lower_bound(self) -> None:
        self.assertEqual(descriptor_count(PAIRS), PAIRS)
        self.assertEqual(descriptor_count(0), 2 * PAIRS)
        self.assertEqual(descriptor_count(200), 250)

    def test_denominator_only_has_no_gate_or_token_gate_access(self) -> None:
        candidate = model_point(128, 212)["candidates"][
            "multiplicity_free_denominator_only_quotient"
        ]
        self.assertEqual(candidate["gate_file_writes"], 0)
        self.assertEqual(candidate["gate_file_reads"], 0)
        self.assertEqual(candidate["token_gate_writes"], 0)
        self.assertEqual(candidate["token_gate_reads"], 0)

    def test_class_stationary_model_charges_reorder(self) -> None:
        point = model_point(128, 212)
        candidate = point["candidates"]["class_stationary_csr_with_reorder"]
        self.assertEqual(candidate["reorder_lower_bound_cycles"], 238)
        self.assertGreater(
            candidate["lower_bound_cycles"],
            point["candidates"]["multiplicity_free_quotient_gate_file"][
                "lower_bound_cycles"
            ],
        )

    def test_profile_gate_denies_high_occupancy(self) -> None:
        self.assertFalse(
            model_point(256, 221)["screen"]["sidecar_rtl_profile_gate"]
        )


if __name__ == "__main__":
    unittest.main()
