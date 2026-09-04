#!/usr/bin/env python3
"""Read-only source tests for the M2137 option-aware timing guard.

The tests deliberately do not invoke VCS, lmutil, DC, PT, or a GPU.  They
exercise the exact regression that consumed M2127 plus three contamination
classes which must remain fail-closed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
HW = HERE.parents[1]
RUNNER = (
    HW
    / "dc_handoff/scripts/"
    "run_m2137_m2018_tsbg_rtl_saif_window_diagnostic_one_shot.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("m2137_guard_under_test", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("M2137 import spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M2137 = load_runner()


class M2137OptionAwareTimingSurfaceTest(unittest.TestCase):
    def make_input(self, root: Path, name: str, text: str) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_harmless_sdformer_path_operands_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m2137_positive_") as temp:
            sdformer = Path(temp) / "SDformer"
            active = self.make_input(
                sdformer, "rtl/ordinary_source.sv", "module ordinary_source; endmodule\n"
            )
            filelist = self.make_input(
                sdformer, "filelists/diagnostic.f", f"{active}\n"
            )
            command = [
                "/opt/synopsys/vcs/bin/vcs",
                f"-Mdir={sdformer / 'build/csrc'}",
                "-f",
                str(filelist),
                "-o",
                str(sdformer / "build/simv"),
            ]
            result = M2137.validate_timing_surface(command, [filelist, active])
            self.assertTrue(result["path_operands_may_contain_sdf_substring"])
            self.assertEqual(result["active_input_count"], 2)

    def test_explicit_sdf_option_forms_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m2137_sdf_option_") as temp:
            active = self.make_input(Path(temp), "source.sv", "module source; endmodule\n")
            for option in ("-sdfmax", "-sdf=min:tb.dut:file.sdf", "+sdfverbose"):
                with self.subTest(option=option):
                    with self.assertRaisesRegex(M2137.Failure, "explicit SDF option"):
                        M2137.validate_timing_surface(["vcs", option], [active])

    def test_unit_delay_define_forms_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m2137_unit_delay_define_") as temp:
            active = self.make_input(Path(temp), "source.sv", "module source; endmodule\n")
            for option in (
                "+define+UNIT_DELAY",
                "+define+FOO+UNIT_DELAY=1",
                "+define+unit_delay=1+BAR",
            ):
                with self.subTest(option=option):
                    with self.assertRaisesRegex(M2137.Failure, "UNIT_DELAY define"):
                        M2137.validate_timing_surface(["vcs", option], [active])

    def test_active_source_timing_contamination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m2137_source_contamination_") as temp:
            root = Path(temp)
            sdf_source = self.make_input(
                root, "sdf_source.sv", 'initial $sdf_annotate("gate.sdf", dut);\n'
            )
            unit_source = self.make_input(
                root, "unit_source.sv", "`ifdef UNIT_DELAY\n`endif\n"
            )
            with self.assertRaisesRegex(M2137.Failure, "source-level SDF annotation"):
                M2137.validate_timing_surface(["vcs"], [sdf_source])
            with self.assertRaisesRegex(M2137.Failure, "source-level UNIT_DELAY"):
                M2137.validate_timing_surface(["vcs"], [unit_source])


if __name__ == "__main__":
    unittest.main()
