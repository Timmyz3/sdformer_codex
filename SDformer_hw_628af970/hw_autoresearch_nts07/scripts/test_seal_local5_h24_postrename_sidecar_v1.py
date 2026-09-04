#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("seal_local5_h24_postrename_sidecar_v1.py")
SPEC = importlib.util.spec_from_file_location("h24_postrename", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class H24PostrenameSidecarTest(unittest.TestCase):
    def test_relocate_maps_historical_staging_to_final_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "source/test.py"
            target.parent.mkdir()
            target.write_text("pass\n", encoding="utf-8")
            raw = "/tmp/pkg.recovery_staging.553905/source/test.py"
            self.assertEqual(MODULE.relocate_staging_path(raw, root), target)

    def test_relocate_rejects_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                MODULE.relocate_staging_path(
                    "/tmp/pkg.recovery_staging.553905/source/missing.py",
                    Path(directory),
                )

    def test_relocate_rejects_nonhistorical_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                MODULE.relocate_staging_path(
                    "/tmp/pkg/source/test.py", Path(directory)
                )

    def test_relocate_maps_staging_root_without_trailing_slash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = "/tmp/pkg.recovery_staging.553905"
            self.assertEqual(MODULE.relocate_staging_path(raw, root), root)

    def test_recursive_json_path_scan_includes_root_and_nested_argv(self) -> None:
        root = "/tmp/pkg.recovery_staging.553905"
        value = {"root": root, "argv": [root + "/source/a.c", 24]}
        self.assertEqual(MODULE.strings_with_staging_path(value), [root, root + "/source/a.c"])


if __name__ == "__main__":
    unittest.main()
