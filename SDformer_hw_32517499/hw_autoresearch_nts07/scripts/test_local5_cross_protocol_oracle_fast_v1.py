#!/usr/bin/env python3
"""将 Local5 C 跨存储协议 oracle 与冻结 Python 金参考逐项对拍。"""

from __future__ import annotations

import json
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_local5_phase_summary_contract_v2 as contract  # noqa: E402
import local5_cross_protocol_oracle_fast_v1 as fast_oracle  # noqa: E402


C_SOURCE = SCRIPTS / "local5_cross_protocol_oracle_fast_v1.c"


class FastOracleMiter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory(prefix="local5_cross_oracle_")
        cls.binary = Path(cls.tempdir.name) / "oracle"
        subprocess.run(
            [
                "cc",
                "-O3",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(C_SOURCE),
                "-o",
                str(cls.binary),
            ],
            check=True,
            cwd=Path("/tmp"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def run_fast(
        self,
        target: str,
        heads: int,
        tiles: int,
        addresses: int,
        orders: list[list[int]] | None = None,
    ) -> dict[str, object]:
        command = [
            str(self.binary),
            target,
            str(heads),
            str(tiles),
            str(addresses),
        ]
        if orders is not None:
            order_path = Path(self.tempdir.name) / "orders.bin"
            with order_path.open("wb") as stream:
                for order in orders:
                    for value in order:
                        stream.write(struct.pack("<Q", value))
            command.append(str(order_path))
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path("/tmp"),
        )
        return json.loads(completed.stdout)

    def assert_matches_python(
        self,
        target: str,
        heads: int,
        tiles: int,
        addresses: int,
        orders: list[list[int]] | None = None,
    ) -> None:
        factory = None if orders is None else lambda tile: tuple(orders[tile])
        expected = contract.expected_cross_protocol_ledger(
            target,
            heads=heads,
            output_tiles=tiles,
            addresses_per_tile=addresses,
            address_order_for_tile=factory,
        )
        actual = self.run_fast(target, heads, tiles, addresses, orders)
        self.assertEqual(actual["count"], expected.count)
        self.assertEqual(actual["read_count"], expected.read_count)
        self.assertEqual(actual["write_count"], expected.write_count)
        self.assertEqual(actual["digest0"], f"{expected.digest0:016x}")
        self.assertEqual(actual["digest1"], f"{expected.digest1:016x}")

    def test_canonical_h1(self) -> None:
        self.assert_matches_python("tb.dut.u_acc", 1, 1, 7)

    def test_canonical_h3(self) -> None:
        self.assert_matches_python("tb.dut.u_acc", 3, 3, 37)

    def test_canonical_h6(self) -> None:
        self.assert_matches_python("tb.dut.u_acc", 6, 6, 113)

    def test_canonical_h12(self) -> None:
        self.assert_matches_python("tb.dut.u_acc", 12, 12, 257)

    def test_random_orders_and_utf8_target(self) -> None:
        rng = random.Random(0x10CA15)
        orders: list[list[int]] = []
        for tile in range(5):
            values = [10_000 * tile + index * 3 for index in range(41)]
            rng.shuffle(values)
            orders.append(values)
        self.assert_matches_python("tb.dut.本地累加器", 5, 5, 41, orders)

    def test_random_short_shapes(self) -> None:
        rng = random.Random(0xC055)
        for case in range(12):
            heads = rng.randint(1, 7)
            tiles = rng.randint(1, 7)
            addresses = rng.randint(1, 53)
            orders = []
            for tile in range(tiles):
                values = [case * 100_000 + tile * 1_000 + index for index in range(addresses)]
                rng.shuffle(values)
                orders.append(values)
            with self.subTest(case=case, heads=heads, tiles=tiles, addresses=addresses):
                self.assert_matches_python(
                    f"tb.random.case{case}", heads, tiles, addresses, orders
                )

    def test_duplicate_address_fails_closed(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_fast("tb.dut.u_acc", 2, 1, 3, [[7, 7, 9]])

    def test_trailing_order_data_fails_closed(self) -> None:
        order_path = Path(self.tempdir.name) / "trailing.bin"
        order_path.write_bytes(struct.pack("<QQQ", 0, 1, 2))
        completed = subprocess.run(
            [str(self.binary), "tb.dut.u_acc", "1", "1", "2", str(order_path)],
            capture_output=True,
            text=True,
            cwd=Path("/tmp"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("trailing", completed.stderr)

    def test_invalid_dimension_fails_closed(self) -> None:
        completed = subprocess.run(
            [str(self.binary), "tb.dut.u_acc", "0", "1", "2"],
            capture_output=True,
            text=True,
            cwd=Path("/tmp"),
        )
        self.assertNotEqual(completed.returncode, 0)

    def test_python_wrapper_compile_and_run(self) -> None:
        binary = Path(self.tempdir.name) / "wrapped_oracle"
        compile_report = fast_oracle.compile_oracle(C_SOURCE, binary)
        expected = contract.expected_cross_protocol_ledger(
            "tb.wrapper.u_acc", heads=3, output_tiles=2, addresses_per_tile=19
        )
        actual, run_report = fast_oracle.run_oracle(
            binary,
            "tb.wrapper.u_acc",
            heads=3,
            output_tiles=2,
            addresses_per_tile=19,
        )
        self.assertEqual(actual, expected)
        self.assertEqual(compile_report["binary_sha256"], run_report["binary_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
