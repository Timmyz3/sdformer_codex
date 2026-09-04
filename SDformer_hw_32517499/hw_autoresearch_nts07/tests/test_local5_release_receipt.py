from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_local5_qfsa_profile_after_fullres as watcher


class Local5ReleaseReceiptTest(unittest.TestCase):
    def test_ranking_change_invalidates_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            run_dir.mkdir()
            ranking = run_dir / "ranking.md"
            config = root / "config.yml"
            checkpoint = run_dir / "checkpoint_epoch7.pth"
            status = root / "status.log"
            receipt_path = root / "receipt.json"
            ranking.write_text("| 1 | 7 |\n", encoding="utf-8")
            config.write_text("config\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            prefix = b"WAIT\n"
            marker = (
                "ALL COMPLETE fullres deploy followup H67 H66d\n"
            ).encode("utf-8")
            status.write_bytes(prefix + marker)
            with patch.multiple(
                watcher,
                RANKING=ranking,
                CONFIG=config,
                RUN_DIR=run_dir,
            ):
                binding = watcher.release_artifact_binding()
                self.assertIsNotNone(binding)
                receipt = {
                    "schema": "local5_release_receipt_v2",
                    "watcher_session_uuid": "unit-test",
                    "release_marker": watcher.RELEASE_MARKER,
                    "marker_line": marker.decode("utf-8").rstrip("\n"),
                    "status_path": str(status.resolve()),
                    "status_prefix_bytes": len(prefix),
                    "status_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                    "marker_start_offset": len(prefix),
                    "marker_end_offset": len(prefix) + len(marker),
                    **binding,
                }
                receipt_path.write_text(
                    json.dumps(receipt), encoding="utf-8"
                )
                self.assertIsNotNone(
                    watcher.validate_release_receipt(receipt_path)
                )
                ranking.write_text("| 1 | 8 |\n", encoding="utf-8")
                self.assertIsNone(
                    watcher.validate_release_receipt(receipt_path)
                )


if __name__ == "__main__":
    unittest.main()
