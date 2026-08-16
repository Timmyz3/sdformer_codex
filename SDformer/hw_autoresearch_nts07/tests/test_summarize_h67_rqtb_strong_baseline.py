from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from scripts.summarize_h67_rqtb_strong_baseline import (
    BUILD_RESTART_PASS,
    parse_area,
    parse_cover,
    parse_log,
    require_restart_receipt,
    load_vector_identity,
    sha256,
)


class SummarizeH67RqtbStrongBaselineTest(unittest.TestCase):
    def test_vector_manifest_binds_artifact_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vector = root / "vectors.txt"
            vector.write_text("payload\n", encoding="utf-8")
            config = root / "config.yml"
            config.write_text("model: h67\n", encoding="utf-8")
            source = root / "source_manifest.json"
            source.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            records = []
            for index in range(12):
                record_source = root / f"record{index}.npz"
                record_source.write_bytes(f"record{index}".encode("ascii"))
                records.append(
                    {
                        "stage": 0 if index < 11 else 3,
                        "block": index if index < 11 else 1,
                        "heads": 1 if index < 11 else 24,
                        "rows": 1 if index < 11 else 127,
                        "source": str(record_source),
                        "source_sha256": sha256(record_source),
                    }
                )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "h67_checkpoint_t450_score_shiftmax_vectors_v1",
                        "scope": "test",
                        "vector_file": str(vector),
                        "vector_sha256": sha256(vector),
                        "source_manifest": str(source),
                        "source_manifest_sha256": sha256(source),
                        "row_count": 138,
                        "tokens_per_row": 450,
                        "token_vector_count": 62100,
                        "records": records,
                        "run_context": {
                            "artifact_identity": {
                                "config_path": str(config),
                                "config_sha256": sha256(config),
                                "checkpoint_path": str(checkpoint),
                                "checkpoint_sha256": sha256(checkpoint),
                                "checkpoint_size": checkpoint.stat().st_size,
                            },
                            "eval_protocol": {"tokens_per_window": 450},
                        },
                    }
                ),
                encoding="utf-8",
            )
            identity = load_vector_identity(manifest, vector)
            self.assertEqual(
                identity["artifact_identity"]["checkpoint_sha256"],
                sha256(checkpoint),
            )
            vector.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "向量文件SHA"):
                load_vector_identity(manifest, vector)
            vector.write_text("payload\n", encoding="utf-8")
            checkpoint.write_bytes(b"tampered checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint文件SHA"):
                load_vector_identity(manifest, vector)

    def test_two_slot_log_contract(self) -> None:
        row = (
            "RQTB_ROW row=0 stage=0 block=0 head=0 active=1 equal=225 "
            "fixed_cycles=10 rqtb_cycles=8 fixed_slots=450 rqtb_slots=225 "
            "fixed_desc=450 rqtb_desc=225 fixed_exp=3 rqtb_exp=2"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "two_slot.log"
            rows = [row.replace("row=0", f"row={index}") for index in range(138)]
            rows.extend(
                [
                    "RQTB_OCC fixed=1,2,3",
                    "RQTB_OCC rqtb=3,2,1",
                    "PASS H67 RQTB 2S physical flow rows=138 checked=1 "
                    "fixed_cycles=10 rqtb_cycles=8 fixed_slots=450 "
                    "rqtb_slots=225 fixed_exp=3 rqtb_exp=2 acc32_mismatch=0",
                ]
            )
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            parsed_rows, final, occupancy = parse_log(path)
        self.assertEqual(len(parsed_rows), 138)
        self.assertEqual(final["rqtb_cycles"], 8)
        self.assertEqual(occupancy[1], [3, 2, 1])

    def test_mapping_area_binds_expected_top(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.log"
            path.write_text(
                "Chip area for module '\\h67_temporal_slot_shiftmax_sync_k_2s_top': 123.5\n",
                encoding="utf-8",
            )
            area = parse_area(path, "h67_temporal_slot_shiftmax_sync_k_2s_top")
            with self.assertRaisesRegex(ValueError, "缺少"):
                parse_area(path, "wrong_top")
        self.assertEqual(area, 123.5)

    def test_coverage_receipt_must_be_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cover.log"
            path.write_text(
                "RQTB_2S_COVER cross_pair=2 same_class=3 double_active=4 fifo_both=5 dual_k=6 "
                "fixed_cross_pair=0 rqtb_cross_pair=2 fixed_same_class=1 rqtb_same_class=2 "
                "fixed_double_active=2 rqtb_double_active=2 fixed_fifo_both=2 rqtb_fifo_both=3 "
                "fixed_dual_k=0 rqtb_dual_k=6\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_cover(path)["dual_k"], 6)
            path.write_text(
                "RQTB_2S_COVER cross_pair=0 same_class=3 double_active=4 fifo_both=5 dual_k=6 "
                "fixed_cross_pair=0 rqtb_cross_pair=0 fixed_same_class=1 rqtb_same_class=2 "
                "fixed_double_active=2 rqtb_double_active=2 fixed_fifo_both=2 rqtb_fifo_both=3 "
                "fixed_dual_k=0 rqtb_dual_k=6\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "矩阵"):
                parse_cover(path)

    def test_restart_receipt_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "restart.log"
            path.write_text(
                "PASS H67 RQTB 2S rejected-restart fail-closed outputs=2\n",
                encoding="utf-8",
            )
            require_restart_receipt(path)
            path.write_text("missing\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt"):
                require_restart_receipt(path)

    def test_build_restart_receipt_uses_distinct_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "build_restart.log"
            path.write_text(
                BUILD_RESTART_PASS + " outputs=4\n",
                encoding="utf-8",
            )
            require_restart_receipt(path, BUILD_RESTART_PASS)
            with self.assertRaisesRegex(ValueError, "receipt"):
                require_restart_receipt(path)


if __name__ == "__main__":
    unittest.main()
