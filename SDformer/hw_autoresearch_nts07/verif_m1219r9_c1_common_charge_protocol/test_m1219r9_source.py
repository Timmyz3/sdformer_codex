#!/usr/bin/env python3
"""Source-only positive and fail-closed mutation tests for M1219/R9."""

import unittest

import check_m1219r9_source as checker


class M1219SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (checker.Path(__file__).with_name(
            "tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9.sv")
            .read_text())

    def assert_rejected(self, mutant: str, reason: str) -> None:
        self.assertTrue(checker.audit_text(mutant), reason)

    def test_canonical_structure_passes(self) -> None:
        self.assertEqual(checker.audit_text(self.text), [])

    def test_unbounded_wait_is_rejected(self) -> None:
        mutant = self.text.replace(
            "while (weight_fire_count != w0 + 1\n"
            "                    && watchdog < R9_RANDOM_WAIT_LIMIT) begin",
            "while (weight_fire_count != w0 + 1) begin", 1)
        self.assert_rejected(mutant, "missing random watchdog was accepted")

    def test_missing_timeout_site_is_rejected(self) -> None:
        mutant = self.text.replace('"normal_prep_ready"', '"normal_prep"', 1)
        self.assert_rejected(mutant, "missing prep timeout site was accepted")

    def test_missing_phase_completion_is_rejected(self) -> None:
        mutant = self.text.replace("PHASE_M1219R9_RANDOM_COMPLETE",
                                   "PHASE_M1219R9_RANDOM_DONE", 1)
        self.assert_rejected(mutant, "missing phase completion was accepted")

    def test_ready_quiesce_removal_is_rejected(self) -> None:
        anchor = self.text.index("R8_RANDOM_REQUEST_READY_QUIESCE_BOUNDARY")
        pos = self.text.index("weight_req_ready = 1'b0;", anchor)
        mutant = self.text[:pos] + "weight_req_ready = 1'b1;" + \
            self.text[pos + len("weight_req_ready = 1'b0;"):]
        self.assert_rejected(mutant, "ready quiesce mutation was accepted")

    def test_random_count_mutation_is_rejected(self) -> None:
        mutant = self.text.replace("test_index < 24", "test_index < 23", 1)
        self.assert_rejected(mutant, "random count mutation was accepted")

    def test_claim_boundary_mutation_is_rejected(self) -> None:
        mutant = self.text.replace("timing_verified=false",
                                   "timing_verified=true", 1)
        self.assert_rejected(mutant, "claim mutation was accepted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
