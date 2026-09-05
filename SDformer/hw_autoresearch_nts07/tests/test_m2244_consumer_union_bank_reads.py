"""Small accounting tests, no new hash or release framework."""
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "system_simulator/scripts"
sys.path.insert(0, str(SCRIPTS))
from m2244_consumer_union_bank_reads import masked_reads


class BankReadTest(unittest.TestCase):
    def test_shared_bank_is_read_once(self):
        w = [[1] + [0]*47 for _ in range(4)]
        self.assertEqual(masked_reads(w, "tsbg", 0)[0]["bank_reads"], 6)
        self.assertEqual(masked_reads(w, "ordinary", 4)[0]["bank_reads"], 6)
        self.assertEqual(masked_reads(w, "ordinary", 0)[0]["bank_reads"], 24)

    def test_partial_hit_fetches_only_missing_bank(self):
        w = [[1 << c] + [0]*47 for c in range(4)]
        for mode in ("ordinary", "tsbg"):
            cold, cache = masked_reads(w, mode, 4)
            self.assertEqual(cold["bank_reads"], 24)
            self.assertEqual(masked_reads(w, mode, 4, cache)[0]["bank_reads"], 0)

    def test_sign_bits_are_not_read_requests(self):
        w = [[0xffff0000] * 48 for _ in range(4)]
        for mode in ("ordinary", "tsbg"):
            self.assertEqual(masked_reads(w, mode, 4)[0]["bank_reads"], 0)

    def test_cache_eviction_can_destroy_cross_context_reuse(self):
        w = [[1] * 5 + [0]*43 for _ in range(4)]
        self.assertEqual(masked_reads(w, "ordinary", 4)[0]["bank_reads"], 120)
        self.assertEqual(masked_reads(w, "tsbg", 0)[0]["bank_reads"], 30)


if __name__ == "__main__":
    unittest.main()
