#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("verify_local5_identity_state_structure_v3.py")
SPEC = importlib.util.spec_from_file_location("state_structure_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StateStructureV3Test(unittest.TestCase):
    @staticmethod
    def write_h1_trace(path: Path, mutate_at: int | None = None) -> None:
        with path.open("wb") as handle:
            handle.write(
                b"cycle,event,tile,head,source,lane,out,delay,index,origin,payload\n"
            )
            for index, row in enumerate(MODULE.expected_all_states(1)):
                value = list(row)
                if mutate_at == index:
                    value[7] = b"31"
                handle.write(str(index).encode("ascii") + b"," + b",".join(value) + b",-\n")

    def test_h3_h6_h12_h24_counts(self) -> None:
        for heads in (3, 6, 12, 24):
            expected = MODULE.expected_counts(heads)
            self.assertEqual(
                expected["tx_state"],
                3 * heads * heads + 43_202 * heads + 1,
            )
            self.assertEqual(
                expected["acc_state"],
                28_800 * heads * heads - 28_800 * heads + 1,
            )
            self.assertEqual(
                expected["head_state"],
                46_157 * heads * heads + 1,
            )

    def test_h1_generators_match_closed_forms(self) -> None:
        heads = 1
        self.assertEqual(
            sum(1 for _ in MODULE.expected_tx_states(heads)),
            MODULE.expected_counts(heads)["tx_state"],
        )
        self.assertEqual(
            sum(1 for _ in MODULE.expected_acc_states(heads)),
            MODULE.expected_counts(heads)["acc_state"],
        )
        self.assertEqual(
            sum(1 for _ in MODULE.expected_head_states(heads)),
            MODULE.expected_counts(heads)["head_state"],
        )
        self.assertEqual(
            sum(1 for _ in MODULE.expected_all_states(heads)),
            sum(MODULE.expected_counts(heads).values()),
        )

    def test_h1_trace_accepts_exact_global_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            self.write_h1_trace(trace)
            report = MODULE.verify_trace(
                trace,
                {"sample": 0, "stage": 0, "block": 0, "window": 0, "heads": 1},
                None,
            )
            self.assertTrue(
                report["structural_oracle"]["global_analytical_order_match"]
            )

    def test_h1_trace_rejects_state_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            self.write_h1_trace(trace, mutate_at=17)
            with self.assertRaisesRegex(ValueError, "global state order differs"):
                MODULE.verify_trace(
                    trace,
                    {"sample": 0, "stage": 0, "block": 0, "window": 0, "heads": 1},
                    None,
                )


if __name__ == "__main__":
    unittest.main()
