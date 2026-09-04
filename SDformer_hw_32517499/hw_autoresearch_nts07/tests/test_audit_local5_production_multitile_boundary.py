#!/usr/bin/env python3

from __future__ import annotations

import unittest

from scripts import audit_local5_production_multitile_boundary as audit


class BoundaryAuditTests(unittest.TestCase):
    def test_rejects_term_tape_as_new_object(self) -> None:
        source = {
            "status": "PASS_EXISTING_EXECUTION_OBJECT_BOUND_TO_PRODUCTION_RTL",
            "independent_reconstruction": {
                "totals": {"terms": 74131, "active": 11245}
            },
        }
        out32 = {
            "status": "PASS",
            "physical_width": {"accumulator_payload_bits": 460800},
            "cycles": {"out2_out32_busy_cycle_invariant": True},
        }
        memo0 = {
            "decision": "NO_GO_AS_STANDALONE_DATE_CONTRIBUTION_KEEP_AS_COMPLETENESS_EVIDENCE",
            "comparison": {"speedup": 1.0192316805748824},
        }
        memo25 = {
            "decision": "NO_GO_AS_STANDALONE_DATE_CONTRIBUTION_KEEP_AS_COMPLETENESS_EVIDENCE",
            "comparison": {"speedup": 0.9998366613581608},
        }
        result = audit.analyze(source, out32, memo0, memo25)
        self.assertTrue(result["status"].startswith("NO_GO"))
        self.assertGreater(
            result["materialization"]["expanded_over_factorized_ratio"], 2.0
        )
        self.assertEqual(result["accumulator_context"]["context_ratio"], 16.0)

    def test_fails_if_memo_passes_gate(self) -> None:
        source = {
            "status": "PASS_EXISTING_EXECUTION_OBJECT_BOUND_TO_PRODUCTION_RTL",
            "independent_reconstruction": {
                "totals": {"terms": 74131, "active": 11245}
            },
        }
        out32 = {
            "status": "PASS",
            "physical_width": {"accumulator_payload_bits": 460800},
            "cycles": {"out2_out32_busy_cycle_invariant": True},
        }
        memo = {
            "decision": "NO_GO_AS_STANDALONE_DATE_CONTRIBUTION_KEEP_AS_COMPLETENESS_EVIDENCE",
            "comparison": {"speedup": 1.10},
        }
        with self.assertRaises(ValueError):
            audit.analyze(source, out32, memo, memo)


if __name__ == "__main__":
    unittest.main()
