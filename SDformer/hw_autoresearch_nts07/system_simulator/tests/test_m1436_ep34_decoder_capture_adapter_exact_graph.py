#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1436_ep34_decoder_capture_adapter_exact_graph.py")
SPEC = importlib.util.spec_from_file_location("m1436_source", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class ExactGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1436_")
        self.root = Path(self.temp.name)
        self.path = self.root / "unified_ordered_records.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, mutation=None, terminal_newline=True):
        rows = []
        for ordinal in range(M.EXPECTED_ORDERED_ROWS):
            row = {"global_order": ordinal, "global_sample_id": ordinal // 247,
                   "category": "other", "name": "x"}
            if mutation is not None:
                mutation(ordinal, row)
            rows.append(json.dumps(row, sort_keys=True, allow_nan=False))
        text = "\n".join(rows) + ("\n" if terminal_newline else "")
        self.path.write_text(text, encoding="utf-8")

    def test_complete_exact_graph_passes(self):
        self.write()
        self.assertEqual(M.validate_complete_ordered_graph(self.root),
                         {"rows": 9880, "first_global_order": 0,
                          "last_global_order": 9879})

    def test_selected_duplicate_global_order_rejected(self):
        self.write(lambda ordinal, row: row.update(global_order=2469)
                   if ordinal == 2470 else None)
        with self.assertRaisesRegex(M.AdapterError, "file ordinal"):
            M.validate_complete_ordered_graph(self.root)

    def test_ignored_row_duplicate_rejected(self):
        self.write(lambda ordinal, row: row.update(global_order=7)
                   if ordinal == 8 else None)
        with self.assertRaisesRegex(M.AdapterError, "file ordinal"):
            M.validate_complete_ordered_graph(self.root)

    def test_bool_and_noninteger_global_order_rejected(self):
        for bad in (True, 4.0, "4"):
            with self.subTest(bad=bad):
                self.write(lambda ordinal, row, value=bad:
                           row.update(global_order=value) if ordinal == 4 else None)
                with self.assertRaisesRegex(M.AdapterError, "exact integer"):
                    M.validate_complete_ordered_graph(self.root)

    def test_missing_row_and_terminal_newline_rejected(self):
        self.write()
        lines = self.path.read_text().splitlines()
        self.path.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(M.AdapterError, "9880"):
            M.validate_complete_ordered_graph(self.root)
        self.write(terminal_newline=False)
        with self.assertRaisesRegex(M.AdapterError, "terminal newline"):
            M.validate_complete_ordered_graph(self.root)

    def test_bool_module_ordinal_rejected(self):
        checkpoint = "a" * 64
        rows = []
        for ordinal, shape in enumerate(M.M1321.WEIGHT_SHAPES):
            rows.append({"module_ordinal": ordinal,
                         "module": M.M1321.MODULES[ordinal],
                         "checkpoint_sha256": checkpoint,
                         "weight": {"shape": list(shape),
                                    "dtype": "torch.float32",
                                    "layout": "C_ORDER_CONTIGUOUS",
                                    "byte_order": "little",
                                    "content_bytes": M.M1321.product(shape) * 4,
                                    "content_sha256": str(ordinal + 1) * 64},
                         "bias": None})
        rows[1]["module_ordinal"] = True
        with self.assertRaisesRegex(M.AdapterError, "exact integer"):
            M.validate_weight_identities(rows, checkpoint)

    def test_cli_remains_inert_without_source_audit(self):
        with self.assertRaises(M.AdapterError):
            M.main([])


if __name__ == "__main__":
    unittest.main()
