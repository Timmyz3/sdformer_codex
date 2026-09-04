#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parent / "generate_tare4_h67_real_trace.py"
SPEC = importlib.util.spec_from_file_location("tare4_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


class Tare4TraceGeneratorTest(unittest.TestCase):
    def test_lane_pack_and_signed10(self) -> None:
        bits = np.zeros(32, dtype=np.bool_)
        bits[[0, 7, 31]] = True
        self.assertEqual(TRACE.pack_lanes(bits), 0x80000081)
        self.assertEqual(TRACE.signed10(-1), 0x3FF)
        self.assertEqual(TRACE.signed10(-256), 0x300)
        self.assertEqual(TRACE.signed10(256), 0x100)
        with self.assertRaises(ValueError):
            TRACE.signed10(512)

    def test_unpack_and_zero_sparse_dense_boundary(self) -> None:
        q = np.zeros((2, 1, 1, 3, 32), dtype=np.bool_)
        k = np.zeros_like(q)
        k[1, 0, 0, 1, :4] = True
        k[1, 0, 0, 2, :5] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.npz"
            np.savez_compressed(
                path,
                q_shape=np.asarray(q.shape, dtype=np.int32),
                q_bits_packed=np.packbits(q.reshape(-1), bitorder="little"),
                k_shape=np.asarray(k.shape, dtype=np.int32),
                k_bits_packed=np.packbits(k.reshape(-1), bitorder="little"),
            )
            payload = np.load(path)
            self.assertTrue(np.array_equal(TRACE.unpack_bits(payload, "q"), q))
            self.assertTrue(np.array_equal(TRACE.unpack_bits(payload, "k"), k))
            record = {
                "name": "S0.B0.attn",
                "file": str(path),
                "sha256": TRACE.sha256(path),
            }
            payload_lines, expected_lines, row = TRACE.process_record(record)
            self.assertEqual(len(payload_lines), 3)
            self.assertEqual(len(expected_lines), 3)
            self.assertEqual(
                (row["kind_zero"], row["kind_sparse"], row["kind_dense"]),
                (1, 1, 1),
            )
            self.assertEqual(row["raw_mismatches"], 0)
            self.assertEqual(row["q7_mismatches"], 0)

            bad = dict(record)
            bad["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                TRACE.process_record(bad)


if __name__ == "__main__":
    unittest.main()
