#!/usr/bin/env python3
"""Local5 phase telemetry pilot verifier 的 fail-closed 负测试。"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

try:
    from .verify_local5_phase_telemetry_pilot_v1 import (
        REQUIRED_BINDINGS,
        parse_telemetry,
        verify_bindings,
        verify_identity_contract,
    )
except ImportError:
    from verify_local5_phase_telemetry_pilot_v1 import (
        REQUIRED_BINDINGS,
        parse_telemetry,
        verify_bindings,
        verify_identity_contract,
    )


IDENTITY = {"stage": 0, "block": 0, "window": 249}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_lines() -> list[str]:
    roles = [
        "GROUP_TRANSACTION",
        "TILE_TRANSACTION",
        "HEAD_TRANSACTION",
        "HEAD_WEIGHT",
        "HEAD_FRONTEND",
        "HEAD_READOUT",
        "HEAD_RELEASE",
        "TILE_DRAIN",
    ]
    lines = [
        "SCHEMA,local5_phase_semantic_telemetry_v1",
        "ORIGIN,RTL_DIRECT",
        "COLUMNS_P,record,sequence,stage,block,window,tile,head,role,start_cycle,end_cycle,duration,origin",
        "COLUMNS_R,record,sequence,stage,block,window,tile,head,resource,cycle,identity0,identity1,identity2,origin",
    ]
    for sequence, role in enumerate(roles):
        lines.append(
            f"P,{sequence},0,0,249,0,0,{role},{sequence},{sequence},1,RTL_DIRECT"
        )
    lines.append("R,0,0,0,249,0,0,RELATION_REQ_ACCEPT,9,0,0,0,RTL_DIRECT")
    lines.append(f"END,9,{len(roles)},1,RTL_DIRECT")
    return lines


class TelemetryNegativeTests(unittest.TestCase):
    def parse(self, lines: list[str]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telemetry.csv"
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            parse_telemetry(path, IDENTITY)

    def test_positive_fixture(self) -> None:
        self.parse(valid_lines())

    def test_phase_missing_fails_closed(self) -> None:
        lines = valid_lines()
        del lines[4]
        for index in range(4, 11):
            fields = lines[index].split(",")
            fields[1] = str(index - 4)
            lines[index] = ",".join(fields)
        lines[-1] = "END,9,7,1,RTL_DIRECT"
        with self.assertRaisesRegex(ValueError, "必要 phase role"):
            self.parse(lines)

    def test_phase_reorder_fails_closed(self) -> None:
        lines = valid_lines()
        lines[4], lines[5] = lines[5], lines[4]
        with self.assertRaisesRegex(ValueError, "乱序"):
            self.parse(lines)

    def test_cycle_tamper_fails_closed(self) -> None:
        lines = valid_lines()
        fields = lines[4].split(",")
        fields[10] = "2"
        lines[4] = ",".join(fields)
        with self.assertRaisesRegex(ValueError, "cycle/duration"):
            self.parse(lines)

    def test_origin_forgery_fails_closed(self) -> None:
        lines = valid_lines()
        fields = lines[4].split(",")
        fields[11] = "PARAM_DERIVED"
        lines[4] = ",".join(fields)
        with self.assertRaisesRegex(ValueError, "origin"):
            self.parse(lines)

    def test_digest_receipt_rebind_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bindings = {}
            for name in REQUIRED_BINDINGS:
                path = root / f"{name}.txt"
                path.write_text(name, encoding="ascii")
                bindings[name] = {"path": path.name, "sha256": digest(path)}
            bindings["telemetry"]["sha256"] = bindings["identity_trace"]["sha256"]
            receipt = {
                "schema": "local5_phase_telemetry_pilot_run_receipt_v1",
                "bindings": bindings,
            }
            with self.assertRaisesRegex(ValueError, "重绑"):
                verify_bindings(root, receipt)

    def test_identity_mismatch_fails_closed(self) -> None:
        actual = {"sample": 2, "stage": 0, "block": 0, "window": 249, "heads": 3}
        receipt: dict[str, Any] = {
            "actual_identity": actual,
            "requested_identity": {**actual, "window": 94},
            "requested_tuple_status": "REJECTED_NONCANONICAL_TUPLE",
        }
        run_argv = [
            "+STAGE_ID=0", "+BLOCK_ID=0", "+WINDOW_ID=249",
            "+TELEMETRY_STAGE=0", "+TELEMETRY_BLOCK=0",
            "+TELEMETRY_WINDOW=249",
        ]
        with self.assertRaisesRegex(ValueError, "身份 P0"):
            verify_identity_contract(receipt, actual, run_argv)


if __name__ == "__main__":
    unittest.main()
