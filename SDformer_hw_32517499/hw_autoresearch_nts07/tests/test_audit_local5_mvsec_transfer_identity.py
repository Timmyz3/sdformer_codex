import unittest

from scripts.audit_local5_mvsec_transfer_identity import (
    compare_lower_is_better,
    parse_reinitialized_pe_keys,
)


class Local5MvsecTransferIdentityTest(unittest.TestCase):
    def test_parse_reinitialized_pe_keys_accepts_exact_overlay(self) -> None:
        keys = [f"layer.{index}.attn.positional_encoding" for index in range(12)]
        text = (
            f"dropped 12 shape-mismatched keys: {keys[:5]!r}\n"
            "load audit: checkpoint_overlay_keys=210, missing=12, unexpected=0\n"
            f"missing keys sample: {keys!r}\n"
        )

        self.assertEqual(parse_reinitialized_pe_keys(text), keys)

    def test_parse_reinitialized_pe_keys_rejects_non_pe_key(self) -> None:
        keys = [f"layer.{index}.attn.positional_encoding" for index in range(11)]
        keys.append("layer.11.weight")
        text = (
            f"dropped 12 shape-mismatched keys: {keys[:5]!r}\n"
            "load audit: checkpoint_overlay_keys=210, missing=12, unexpected=0\n"
            f"missing keys sample: {keys!r}\n"
        )

        with self.assertRaisesRegex(ValueError, "exclusively positional"):
            parse_reinitialized_pe_keys(text)

    def test_compare_lower_is_better_rejects_one_losing_sequence(self) -> None:
        candidate = {
            "outdoor_day1": {"AEE": 1.0},
            "indoor_flying1": {"AEE": 2.0},
            "indoor_flying2": {"AEE": 3.0},
            "indoor_flying3": {"AEE": 4.0},
        }
        reference = {
            "outdoor_day1": {"AEE": 1.1},
            "indoor_flying1": {"AEE": 2.1},
            "indoor_flying2": {"AEE": 2.9},
            "indoor_flying3": {"AEE": 4.1},
        }

        result = compare_lower_is_better(candidate, {"baseline": reference})

        self.assertFalse(result["baseline"]["all_four_lower"])
        self.assertGreater(
            result["baseline"]["aee_delta_candidate_minus_reference"][
                "indoor_flying2"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
