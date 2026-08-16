from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_local5_residual_leftover as leftover


class EvidenceParserFailClosedTest(unittest.TestCase):
    def test_leftover_rejects_mismatched_memh_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input_q.memh").write_text("0\n1\n", encoding="ascii")
            (root / "input_candidate_k.memh").write_text("0\n", encoding="ascii")
            (root / "input_valid.memh").write_text("1\n1\n", encoding="ascii")
            (root / "expected_scores.memh").write_text("0\n0\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "memh length mismatch"):
                leftover.analyze(root)

    def test_leftover_accepts_equal_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in {
                "input_q.memh": "0\n",
                "input_candidate_k.memh": "0\n",
                "input_valid.memh": "1\n",
                "expected_scores.memh": "0\n",
            }.items():
                (root / name).write_text(value, encoding="ascii")
            result = leftover.analyze(root)
            self.assertEqual(result["leftover_qnz_not_identk"], 0)


if __name__ == "__main__":
    unittest.main()
