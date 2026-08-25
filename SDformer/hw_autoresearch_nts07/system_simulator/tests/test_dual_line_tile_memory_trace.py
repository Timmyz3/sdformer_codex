import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_dual_line_tile_memory_trace.py"
SPEC = importlib.util.spec_from_file_location("dual_line_tile_memory_trace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DualLineTileMemoryTraceTest(unittest.TestCase):
    def make_trace(self, root: Path) -> Path:
        directory = root / "tiles"
        directory.mkdir()
        rows = []
        current = []
        previous = []
        for chunk in range(2):
            for timestep in range(2):
                bits = np.zeros(256, dtype=np.uint8)
                old = np.zeros(256, dtype=np.uint8)
                bits[chunk + timestep] = 1
                if timestep:
                    old[chunk] = 1
                positive = int(np.logical_and(bits, np.logical_not(old)).sum())
                negative = int(np.logical_and(np.logical_not(bits), old).sum())
                rows.append({
                    "record_id": len(rows), "sample_id": 0, "sample_key": "s0",
                    "sequence_key": "seq", "name": "linear", "operator": "Linear",
                    "operator_call_index": 0, "row_id": 0, "chunk_index": chunk,
                    "chunks_per_row": 2, "source_base": chunk * 256,
                    "source_width": 512, "valid_bits": 256,
                    "output_channel_fanout": 96, "weight_group": 0,
                    "output_lane_tile_count_96": 1, "temporal_step": timestep,
                    "state_valid": timestep > 0, "row_current_count": 2,
                    "row_transition_count": 2 if timestep == 0 else 4,
                    "row_use_motion": False, "tile_current_count": int(bits.sum()),
                    "tile_positive_count": positive, "tile_negative_count": negative,
                    "schedule_contract": "test",
                })
                current.append(np.packbits(bits, bitorder="little"))
                previous.append(np.packbits(old, bitorder="little"))
        # Writer order is chunk-major; group_rows must reconstruct complete rows by timestep.
        csv_path = directory / "tile_records.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        npz_path = directory / "packed_tiles.npz"
        np.savez_compressed(npz_path, packed_current_bits=np.stack(current), packed_previous_bits=np.stack(previous))
        manifest = {
            "schema": "dual_line_real_tile_trace_v1",
            "status": "PASS_REAL_BITMAPS_ROW_SELECTOR_TILE_EXECUTION_NOT_ACC32_ORACLE",
            "tile_bits": 256,
            "records": len(rows),
            "row_chunk_identities": 2,
            "run_context": {
                "artifact_identity": {"checkpoint_sha256": "a" * 64, "config_sha256": "b" * 64},
                "checkpoint_load_audit": {
                    "missing_count": 0, "unexpected_count": 0,
                    "overlay_missing_count": 0, "overlay_unexpected_count": 0,
                },
            },
            "sha256": {
                csv_path.name: hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                npz_path.name: hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def test_chunk_complete_grouping_and_state_charge(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_trace(Path(temporary))
            _manifest, records, _current, _previous = MODULE.validate(directory)
            groups = MODULE.group_rows(records)
            self.assertEqual(len(groups), 2)
            local_tx, local = MODULE.schedule_variant(
                groups, variant="local_only", lanes=96, sources_per_cycle=1,
                command_overhead=5, bitmap_bytes_per_cycle=32,
                acc_bytes_per_cycle=64, weight_bytes_per_cycle=96,
                arena=MODULE.AddressArena(),
            )
            shared_tx, shared = MODULE.schedule_variant(
                groups, variant="local_motion_shared_state", lanes=96, sources_per_cycle=1,
                command_overhead=5, bitmap_bytes_per_cycle=32,
                acc_bytes_per_cycle=64, weight_bytes_per_cycle=96,
                arena=MODULE.AddressArena(base=0x80000000),
                motion_enabled=True, state_storage_model="shared_output_state",
            )
            copy_tx, copy = MODULE.schedule_variant(
                groups, variant="local_motion_explicit_copy", lanes=96, sources_per_cycle=1,
                command_overhead=5, bitmap_bytes_per_cycle=32,
                acc_bytes_per_cycle=64, weight_bytes_per_cycle=96,
                arena=MODULE.AddressArena(base=0xC0000000),
                motion_enabled=True, state_storage_model="explicit_copy_state",
            )
        self.assertGreater(copy["cycles"], shared["cycles"])
        self.assertTrue(any(row["phase"] == "previous_bitmap_read" for row in shared_tx))
        self.assertFalse(any(row["phase"] == "state_acc32_write" for row in shared_tx))
        self.assertTrue(any(row["phase"] == "state_acc32_write" for row in copy_tx))
        self.assertFalse(any(row["phase"] == "previous_bitmap_read" for row in local_tx))
        self.assertEqual(local["peak_row_incremental_state_bytes"], 0)
        self.assertEqual(shared["peak_row_incremental_state_bytes"], 0)
        self.assertGreater(copy["peak_row_incremental_state_bytes"], 0)

    def test_missing_chunk_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_trace(Path(temporary))
            _manifest, records, _current, _previous = MODULE.validate(directory)
            with self.assertRaisesRegex(ValueError, "not chunk-complete"):
                MODULE.group_rows(records[:-1])

    def test_weight_groups_never_alias_one_address_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_trace(Path(temporary))
            _manifest, records, _current, _previous = MODULE.validate(directory)
            groups = MODULE.group_rows(records)
            groups[1] = [{**row, "weight_group": "1"} for row in groups[1]]
            transactions, _totals = MODULE.schedule_variant(
                groups, variant="local_only", lanes=96, sources_per_cycle=1,
                command_overhead=5, bitmap_bytes_per_cycle=32,
                acc_bytes_per_cycle=64, weight_bytes_per_cycle=96,
                arena=MODULE.AddressArena(),
            )
        weight_objects = {
            row["object_id"] for row in transactions
            if row["phase"] == "weight_read_and_accumulate"
        }
        self.assertEqual(
            weight_objects,
            {"weight:linear:Linear:g0", "weight:linear:Linear:g1"},
        )

    def test_row_conservation_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_trace(Path(temporary))
            _manifest, records, _current, _previous = MODULE.validate(directory)
            records[0]["row_current_count"] = "99"
            with self.assertRaisesRegex(ValueError, "metadata|conserve"):
                MODULE.group_rows(records)

    def test_checkpoint_audit_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_trace(Path(temporary))
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["run_context"]["checkpoint_load_audit"]["missing_count"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint load audit"):
                MODULE.validate(directory)


if __name__ == "__main__":
    unittest.main()
