#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1332_c2_headline_mapped_production_activity_source.py"
SPEC = importlib.util.spec_from_file_location("m1332_checker", CHECKER)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def fake_saif(duration, endpoint_tc=4, tx=0, reset_tc=0):
    return '''(SAIFILE
  (DURATION {duration})
  (INSTANCE tb_m1332_c2_headline_mapped_production_activity
    (INSTANCE core
      (INSTANCE dut
        (PORT
          (clk_core (T0 1) (T1 1) (TX {tx}) (TC 20))
          (rst_core (T0 1) (T1 0) (TX 0) (TC {reset_tc}))
          (raw_valid (T0 1) (T1 1) (TX 0) (TC 2))
          (raw_accept (T0 1) (T1 1) (TX 0) (TC 2))
          (mem_req_accept[0] (T0 1) (T1 1) (TX 0) (TC {endpoint_tc}))
          (mem_rsp_accept[0] (T0 1) (T1 1) (TX 0) (TC {endpoint_tc}))
          (result_accumulator[0] (T0 1) (T1 1) (TX 0) (TC 6))
          (result_accept (T0 1) (T1 1) (TX 0) (TC 4))
          (token_done_accept (T0 1) (T1 1) (TX 0) (TC 2))
        )
      )
    )
  )
)'''.format(duration=duration, endpoint_tc=endpoint_tc, tx=tx,
            reset_tc=reset_tc)


class M1332SourceTest(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".saif", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_static_source(self):
        result = M.validate_static()
        self.assertEqual(result["status"], "PASS_M1332_SOURCE_ONLY__NO_EDA")
        self.assertEqual(result["axes"], ["k8", "k1x8"])
        self.assertEqual(result["cases"], 10)

    def test_filelists_are_axis_separate(self):
        for axis, path in M.FILELISTS.items():
            lines = M.validate_filelist(path.read_text(), axis)
            self.assertGreaterEqual(len(lines), 8)

    def test_k1_or_old_memory_is_rejected(self):
        text = M.FILELISTS["k8"].read_text()
        with self.assertRaisesRegex(RuntimeError, "diagnostic K1"):
            M.validate_filelist(text + "+define+M979_AXIS_K1\n", "k8")
        with self.assertRaisesRegex(RuntimeError, "old memory"):
            M.validate_filelist(text + "tb_m349/m349_fc2_scalar_bank_memory_model.sv\n", "k8")

    def test_k8_saif_passes(self):
        out = M.validate_saif(self.write(fake_saif(153)), "k8", 0, 51)
        self.assertEqual(out["status"], "PASS_M1332_HEADLINE_AXIS_PRODUCTION_SAIF")

    def test_cycle_tx_reset_and_endpoint_fail_closed(self):
        path = self.write(fake_saif(153))
        with self.assertRaisesRegex(RuntimeError, "anchor"):
            M.validate_saif(path, "k8", 0, 52)
        with self.assertRaisesRegex(RuntimeError, "TX"):
            M.validate_saif(self.write(fake_saif(153, tx=1)), "k8", 0, 51)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            M.validate_saif(self.write(fake_saif(153, reset_tc=2)), "k8", 0, 51)
        with self.assertRaisesRegex(RuntimeError, "endpoint"):
            M.validate_saif(self.write(fake_saif(153, endpoint_tc=0)), "k8", 0, 51)

    def test_zero_case_allows_zero_endpoint_only(self):
        out = M.validate_saif(self.write(fake_saif(42, endpoint_tc=0)),
                              "k1x8", 4, 14)
        self.assertEqual(out["major_cone_tc"]["endpoint"], 0)


if __name__ == "__main__":
    unittest.main()
