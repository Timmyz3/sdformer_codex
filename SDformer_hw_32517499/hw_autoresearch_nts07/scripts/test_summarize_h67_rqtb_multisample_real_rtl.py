#!/usr/bin/env python3
"""Unit tests for the H67 multisample RTL evidence reporter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from scripts import summarize_h67_rqtb_multisample_real_rtl as report
except ModuleNotFoundError:
    import summarize_h67_rqtb_multisample_real_rtl as report


def row_index_rows(sample_count: int = 2) -> list[dict]:
    rows = []
    for row_tag, (sample_id, stage, block, head) in enumerate(
            report.expected_row_sequence(sample_count)):
        record_order = sum(
            report.EXPECTED_BLOCKS[value] for value in range(stage)
        ) + block
        rows.append(
            {
                "row_tag": row_tag,
                "sample_id": sample_id,
                "sample_key": f"sample-{sample_id}",
                "stage": stage,
                "block": block,
                "head": head,
                "record_order": record_order,
                "expected_outputs": 10,
                "expected_folded": 440,
            }
        )
    return rows


def row_receipt(row: dict) -> dict[str, int]:
    return {
        "row": row["row_tag"],
        "stage": row["stage"],
        "block": row["block"],
        "head": row["head"],
        "active": 10,
        "equal": 100,
        "fixed_cycles": 1000 + row["row_tag"],
        "rqtb_cycles": 900 + row["row_tag"],
        "fixed_slots": 450,
        "rqtb_slots": 350,
        "fixed_desc": 0,
        "rqtb_desc": 10,
        "fixed_exp": 20,
        "rqtb_exp": 15,
        "fixed_pair_stall": 1,
        "rqtb_pair_stall": 2,
        "fixed_desc_stall": 3,
        "rqtb_desc_stall": 4,
        "fixed_out_stall": 5,
        "rqtb_out_stall": 6,
        "fixed_fifo_max": 7,
        "rqtb_fifo_max": 8,
    }


def line(prefix: str, values: dict[str, int]) -> str:
    return prefix + " ".join(f"{key}={value}" for key, value in values.items())


def write_fixture(root: Path, sample_count: int = 2):
    rows = row_index_rows(sample_count)
    index = root / "row_index.jsonl"
    index.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="ascii",
    )
    vector = root / "h67_multisample_checkpoint_rows.txt"
    vector.write_text(f"{len(rows)} 450\n", encoding="ascii")
    source_manifest = root / "source_trace_manifest.json"
    source_manifest.write_text("{}\n", encoding="ascii")
    records = []
    for sample_id in range(sample_count):
        for record_id in range(12):
            source = root / f"sample{sample_id}_record{record_id}.npz"
            source.write_bytes(f"trace-{sample_id}-{record_id}".encode("ascii"))
            records.append(
                {
                    "source": str(source),
                    "source_sha256": report.file_sha256(source),
                }
            )
    generator = report.ROOT / "scripts/generate_h67_multisample_checkpoint_row_vectors.py"
    legacy_generator = report.ROOT / "scripts/generate_h67_checkpoint_row_vectors.py"
    manifest = {
        "schema": "h67_multisample_checkpoint_t450_vectors_v1",
        "status": "PASS",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": report.file_sha256(source_manifest),
        "sample_count": sample_count,
        "rows_per_sample": report.ROWS_PER_SAMPLE,
        "row_count": len(rows),
        "tokens_per_row": 450,
        "records": records,
        "artifacts": {
            "vector_file": str(vector),
            "vector_sha256": report.file_sha256(vector),
            "row_index": str(index),
            "row_index_sha256": report.file_sha256(index),
            "generator": str(generator),
            "generator_sha256": report.file_sha256(generator),
            "legacy_semantic_generator": str(legacy_generator),
            "legacy_semantic_generator_sha256": report.file_sha256(legacy_generator),
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    receipts = [row_receipt(row) for row in rows]
    final = {
        "rows": len(rows),
        "checked": sum(row["active"] for row in receipts),
        "fixed_cycles": sum(row["fixed_cycles"] for row in receipts),
        "rqtb_cycles": sum(row["rqtb_cycles"] for row in receipts),
        "fixed_slots": sum(row["fixed_slots"] for row in receipts),
        "rqtb_slots": sum(row["rqtb_slots"] for row in receipts),
        "fixed_exp": sum(row["fixed_exp"] for row in receipts),
        "rqtb_exp": sum(row["rqtb_exp"] for row in receipts),
        "acc32_mismatch": 0,
    }
    text = "\n".join(
        [line("RQTB_ROW ", receipt) for receipt in receipts]
        + [line("PASS H67 RQTB 2S physical flow ", final)]
    ) + "\n"
    icarus = root / "icarus.log"
    verilator = root / "verilator.log"
    icarus.write_text(text, encoding="utf-8")
    verilator.write_text(text, encoding="utf-8")
    return icarus, verilator, index, manifest_path


class ReporterTests(unittest.TestCase):
    def test_positive_cross_simulator_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            result = report.summarize(*paths)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["coverage"]["samples"], 2)
            self.assertEqual(result["coverage"]["rows"], 276)
            self.assertTrue(result["coverage"]["cross_simulator_exact"])
            self.assertEqual([stage["rows"] for stage in result["stages"]], [12, 24, 144, 96])
            self.assertTrue(all(stage["speedup"] > 1.0 for stage in result["stages"]))
            self.assertEqual(
                result["synthetic_acc32_boundary"]["status"],
                "PASS_SYNTHETIC_ONLY",
            )

    def test_rejects_log_divergence_and_nonzero_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            icarus, verilator, index, manifest = write_fixture(root)
            text = verilator.read_text(encoding="utf-8")
            verilator.write_text(text.replace("fixed_cycles=1000", "fixed_cycles=1001", 1))
            with self.assertRaises(ValueError):
                report.summarize(icarus, verilator, index, manifest)
            verilator.write_text(text.replace("acc32_mismatch=0", "acc32_mismatch=1"))
            with self.assertRaises(ValueError):
                report.summarize(icarus, verilator, index, manifest)

    def test_rejects_short_or_misordered_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, index, _ = write_fixture(root, sample_count=1)
            with self.assertRaises(ValueError):
                report.load_row_index(index)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, index, _ = write_fixture(root)
            rows = [json.loads(line) for line in index.read_text().splitlines()]
            rows[1]["head"] = 2
            index.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="ascii"
            )
            with self.assertRaises(ValueError):
                report.load_row_index(index)


if __name__ == "__main__":
    unittest.main()
