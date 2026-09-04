import unittest

import numpy as np

from scripts import model_topology_quotient_fission_fusion as MODEL


class TopologyQuotientFissionFusionTest(unittest.TestCase):
    def test_vectorized_ordered_queue_matches_scalar_recurrence(self):
        trace = np.array([0, 1, 9, 0, 7, 2, 0, 16, 1], dtype=np.int64)
        width = 4
        backlog = 0
        max_backlog = 0
        for delta in trace:
            service = MODEL.ceil_div(int(delta), width) if delta else 0
            backlog = max(0, backlog + service - 1)
            max_backlog = max(max_backlog, backlog)
        self.assertEqual(
            MODEL.ordered_queue(trace, width),
            (len(trace) + backlog, max_backlog),
        )

    def test_motion_exact_conservation_and_fusion_dominates_serial(self):
        samples = MODEL.load_motion_samples()
        self.assertEqual(len(samples), 100)
        for sample in samples:
            self.assertEqual(
                sample.expanded_candidates - sample.quotient_candidates,
                sample.exact_collapses,
            )
            for width in (2, 4, 8):
                self.assertLessEqual(
                    sample.ff_cycles_by_slice[width],
                    sample.serial_cycles_by_slice[width],
                )

    def test_local_exact_conservation_and_fission_dominates_serial(self):
        samples = MODEL.load_local_samples()
        self.assertEqual(len(samples), 100)
        for sample in samples:
            self.assertEqual(
                sample.expanded_candidates - sample.quotient_candidates,
                sample.exact_collapses,
            )
            for width in (2, 4, 8):
                self.assertLessEqual(
                    sample.ff_cycles_by_slice[width],
                    sample.serial_cycles_by_slice[width],
                )

    def test_report_keeps_local_ordered_boundary(self):
        report = MODEL.build_report()
        motion = report["lines"]["Motion"]["dse"]["records"]
        local = report["lines"]["Local5"]["dse"]["records"]
        self.assertTrue(
            any(
                record["candidate_status"]
                == "eligible_for_full_boundary_rtl_prototype"
                for record in motion
            )
        )
        self.assertTrue(
            all(
                record["ordered_tail_evidence"] == "[缺失-ordered]"
                for record in local
            )
        )
        self.assertFalse(
            any(
                record["candidate_status"]
                == "eligible_for_full_boundary_rtl_prototype"
                for record in local
            )
        )
        self.assertEqual(
            report["lines"]["Motion"]["trcf"][
                "candidate_status"
            ],
            "rejected",
        )
        motion_w4 = next(
            record for record in motion if record["slice_width"] == 4
        )
        self.assertEqual(
            motion_w4["ff_speedup_vs_equal_lane_monolithic"],
            1.0,
        )
        self.assertEqual(
            motion_w4["fission_fusion_novelty_status"],
            "rejected_no_gain_vs_monolithic",
        )
        self.assertEqual(
            report["lines"]["Local5"]["trcf"][
                "candidate_status"
            ],
            "blocked_by_local_ordered_destination_trace",
        )

    def test_work_reductions_are_nonnegative(self):
        report = MODEL.build_report()
        for line in ("Motion", "Local5"):
            work = report["lines"][line]["work"]
            for field in (
                "candidate_reduction",
                "score_lane_work_reduction",
                "k_payload_reduction",
                "unique_product_generation_reduction",
            ):
                self.assertGreaterEqual(work[field], 0.0)
                self.assertLessEqual(work[field], 1.0)

    def test_acqn_domain_and_evidence_boundary(self):
        report = MODEL.build_report()
        acqn = report["acqn"]
        self.assertEqual(
            acqn["shared_structure"]["score_class_entries"],
            163,
        )
        self.assertLessEqual(acqn["Motion"]["observed_score_max"], 162)
        self.assertGreater(
            acqn["Motion"]["recompute_exp_eval_reduction_vs_current_scs"],
            0.50,
        )
        self.assertGreater(
            acqn["Motion"]["cached_exp_eval_reduction_vs_current_scs"],
            acqn["Motion"]["recompute_exp_eval_reduction_vs_current_scs"],
        )
        self.assertEqual(
            acqn["Local5"]["status"],
            "blocked_by_post_g0_row_score_class_trace",
        )


if __name__ == "__main__":
    unittest.main()
