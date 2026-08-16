#!/usr/bin/env python3
"""IBF完整顶层周期日志解析测试。"""

import tempfile
import unittest
from pathlib import Path

from scripts.summarize_single_head_ibf_integration import parse_cycles


class TestSummarizeSingleHeadIbfIntegration(unittest.TestCase):
    def test_parse_last_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.log"
            path.write_text("cycles=10\nPASS cycles=35\n", encoding="utf-8")
            self.assertEqual(parse_cycles(path), 35)


if __name__ == "__main__":
    unittest.main()
