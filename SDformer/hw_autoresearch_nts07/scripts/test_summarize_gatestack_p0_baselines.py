#!/usr/bin/env python3
"""GateStack RTL日志汇总器测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_p0_baselines import parse_log


GOOD = (
    "TRACE CHECK tag=1/1 error=0 protocol=0 groups=1 tiles=1 issues=1 "
    "slot=1/1 cache=0/0 proj=1/2 bias=1/0 abort=0/0/0 "
    "final=1 mismatch=0\n"
    "PASS group_cycles=99\n"
)


class GateStackLogParserTest(unittest.TestCase):
    def _write(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        handle.write(text)
        handle.close()
        return Path(handle.name)

    def test_protocol_fields_are_preserved(self) -> None:
        result = parse_log(self._write(GOOD))
        self.assertEqual(result["cycles"], 99)
        self.assertEqual(result["done_error"], 0)
        self.assertEqual(result["protocol_errors"], 0)

    def test_protocol_error_fails_summary(self) -> None:
        with self.assertRaisesRegex(ValueError, "不等价"):
            parse_log(self._write(GOOD.replace("protocol=0", "protocol=1")))


if __name__ == "__main__":
    unittest.main()
