import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "report_local5_qc_cache_ablation.py"
SPEC = importlib.util.spec_from_file_location("report_local5_qc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReportLocal5QcCacheTest(unittest.TestCase):
    def test_parse_rejects_missing_cache_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.log"
            path.write_text(
                "GROUP group=0 cycles=1 terms=1 updates=1 cache_hits=0 "
                "cache_misses=1 tag_compares=4 lru_writes=1 "
                "product_reads=0 product_writes=1 product_starts=1\n"
                "PASS Local5 score-to-projection groups=1 total_cycles=1\n"
            )
            with self.assertRaises(ValueError):
                MODULE.parse_log(path, 1)

    def test_parse_accepts_complete_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ok.log"
            path.write_text(
                "GROUP group=0 cycles=1 terms=1 updates=1 cache_hits=0 "
                "cache_misses=1 tag_compares=4 lru_writes=1 "
                "product_reads=0 product_writes=1 product_starts=1 "
                "weight_reads=2 memory_wait=0\n"
                "PASS Local5 score-to-projection groups=1 total_cycles=1\n"
            )
            parsed = MODULE.parse_log(path, 1)
            self.assertEqual(parsed["totals"]["product_starts"], 1)


if __name__ == "__main__":
    unittest.main()
