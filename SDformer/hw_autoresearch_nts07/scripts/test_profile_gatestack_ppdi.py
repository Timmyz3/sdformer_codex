#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from profile_gatestack_ppdi import profile_stage


class PpdiProfileTest(unittest.TestCase):
    def test_parity_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps({
                "stage": 3, "logical_supertiles": 2
            }), encoding="utf-8")
            (root / "term_destination_counts.memh").write_text(
                "4\n3\n", encoding="utf-8"
            )
            (root / "term_token_offsets.memh").write_text(
                "0\n4\n7\n", encoding="utf-8"
            )
            # term0: 2 even + 2 odd -> 2 commands; term1: 3 even -> 3 commands.
            (root / "term_tokens.memh").write_text(
                "0\n1\n2\n3\n4\n6\n8\n", encoding="utf-8"
            )
            row = profile_stage(root)
            self.assertEqual(row["destinations"], 14)
            self.assertEqual(row["ppdi_commands"], 10)
            self.assertAlmostEqual(row["command_reduction"], 2 / 7)
            self.assertEqual(row["max_term_parity_imbalance"], 3)


if __name__ == "__main__":
    unittest.main()
