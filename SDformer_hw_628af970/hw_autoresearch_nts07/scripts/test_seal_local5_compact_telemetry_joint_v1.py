#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("seal_local5_compact_telemetry_joint_v1.py")
SPEC = importlib.util.spec_from_file_location("compact_joint_v1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompactTelemetryJointTest(unittest.TestCase):
    def test_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x"
            path.write_bytes(b"abc")
            self.assertEqual(
                MODULE.sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_topology_contract_has_twelve_entries(self) -> None:
        self.assertEqual(len(MODULE.EXPECTED_TOPOLOGY), 12)
        self.assertIn((3, 1, 24), MODULE.EXPECTED_TOPOLOGY)


if __name__ == "__main__":
    unittest.main()
