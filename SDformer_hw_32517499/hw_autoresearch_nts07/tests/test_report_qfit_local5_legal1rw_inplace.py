import tempfile
import unittest
from pathlib import Path

from scripts.report_qfit_local5_legal1rw_inplace import parse_log, sha256, verify_sha_manifest


class TestReportQfitLocal5Legal1rwInplace(unittest.TestCase):
    def test_parse_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(
                "PASS Local5 multi-tile memo=0 inplace=1 acc_backend=1 "
                "tx_service=1 seed=17717 cycles=123 token=4050 "
                "token_delay_sum=999 weight_delay_sum=777 result_service=43200 "
                "hits=0 fallback=0 replay_records=0 partial=0 final=43200\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_log(path),
                {
                    "memo": 0, "inplace": 1, "backend": 1, "tx": 1,
                    "seed": 17717, "cycles": 123, "token": 4050,
                    "token_delay": 999, "weight_delay": 777,
                    "result_service": 43200,
                },
            )

    def test_verify_sha_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("payload\n", encoding="utf-8")
            manifest = root / "manifest.txt"
            manifest.write_text(f"{sha256(source)}  {source}\n", encoding="utf-8")
            self.assertEqual(verify_sha_manifest(manifest), 1)


if __name__ == "__main__":
    unittest.main()
