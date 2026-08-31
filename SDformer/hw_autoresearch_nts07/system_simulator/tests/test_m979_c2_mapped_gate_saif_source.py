#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CHECKER = HERE.parent / "scripts/check_m979_c2_mapped_gate_saif_source.py"
SPEC = importlib.util.spec_from_file_location("m979_checker", CHECKER)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def fake_saif(duration, memory_tc=4, tx=0, reset_tc=0, dut=True):
    scope = "(INSTANCE dut" if dut else "(INSTANCE wrong"
    return f'''(SAIFILE
  (DURATION {duration})
  (INSTANCE tb_m979_c2_three_axis_mapped_gate_case_saif
    {scope}
      (PORT
        (clk_core (T0 1) (T1 1) (TX {tx}) (TC 20))
        (rst_core (T0 1) (T1 0) (TX 0) (TC {reset_tc}))
        (header_accept (T0 1) (T1 1) (TX 0) (TC 2))
        (raw_accept (T0 1) (T1 1) (TX 0) (TC 3))
        (mem_req_accept[0] (T0 1) (T1 1) (TX 0) (TC {memory_tc}))
        (mem_rsp_accept[0] (T0 1) (T1 1) (TX 0) (TC {memory_tc}))
        (result_accumulator[0] (T0 1) (T1 1) (TX 0) (TC 6))
        (result_accept (T0 1) (T1 1) (TX 0) (TC 4))
        (token_done_accept (T0 1) (T1 1) (TX 0) (TC 2))
      )
    )
  )
)'''


class M979SourceTest(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".saif", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_static_source_contract(self):
        out = M.validate_static()
        self.assertEqual(out["status"], "PASS_M979_STATIC_SOURCE__NO_EDA")
        self.assertEqual(out["case_count"], 15)
        self.assertFalse(out["mapped_port_orientation_tool_proven"])

    def test_k8_cycle_and_duration_pass(self):
        out = M.validate_saif(self.write(fake_saif(153)), "k8", 0, 51)
        self.assertEqual(out["tx_nonzero"], 0)
        self.assertGreater(out["major_cone_tc"]["memory"], 0)

    def test_cycle_anchor_and_duration_fail_closed(self):
        path = self.write(fake_saif(153))
        with self.assertRaisesRegex(RuntimeError, "cycle anchor"):
            M.validate_saif(path, "k8", 0, 52)
        with self.assertRaisesRegex(RuntimeError, "duration"):
            M.validate_saif(path, "k1", 0, 52)

    def test_tx_and_scope_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "TX"):
            M.validate_saif(self.write(fake_saif(153, tx=1)), "k8", 0, 51)
        with self.assertRaisesRegex(RuntimeError, "DUT scope"):
            M.validate_saif(self.write(fake_saif(153, dut=False)), "k8", 0, 51)

    def test_reset_must_not_toggle(self):
        with self.assertRaisesRegex(RuntimeError, "reset"):
            M.validate_saif(self.write(fake_saif(153, reset_tc=2)), "k8", 0, 51)

    def test_memory_cone_required_for_nonzero_case(self):
        with self.assertRaisesRegex(RuntimeError, "memory cone"):
            M.validate_saif(self.write(fake_saif(153, memory_tc=0)), "k8", 0, 51)

    def test_zero_event_case_allows_zero_memory_only(self):
        out = M.validate_saif(self.write(fake_saif(42, memory_tc=0)), "k8", 4, 14)
        self.assertEqual(out["major_cone_tc"]["memory"], 0)
        self.assertFalse(out["zero_case_memory_nonzero_required"])


if __name__ == "__main__":
    unittest.main()
