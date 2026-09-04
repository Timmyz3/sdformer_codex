from __future__ import annotations

import unittest

from scripts.summarize_rqtb_openroad_proxy import (
    build_physical_boundaries,
    parse_unconstrained_endpoints,
    validate_route_complete,
)


class SummarizeRqtbOpenroadProxyTest(unittest.TestCase):
    def test_unconstrained_endpoint_count_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "计数与名称不一致"):
            parse_unconstrained_endpoints(
                "Warning: There are 2 unconstrained endpoints.\n  perf_slots[0]\n"
            )

    def test_incomplete_route_is_rejected(self) -> None:
        route = {
            "detailedroute__route__drc_errors__iter:1": 0,
            "detailedroute__route__drc_errors": 0,
        }
        with self.assertRaisesRegex(ValueError, "未完成"):
            validate_route_complete(route, "Number of violations = 0")

    def test_complete_zero_drc_route_is_accepted(self) -> None:
        route = {
            "detailedroute__route__drc_errors__iter:1": 3,
            "detailedroute__route__drc_errors__iter:2": 0,
            "detailedroute__route__drc_errors": 0,
        }
        validate_route_complete(route, "[INFO DRT-0198] Complete detail routing.")

    def test_report_boundaries_use_current_metrics(self) -> None:
        fixed = {
            "setup_slack_ns": 0.0885,
            "setup_violations": 0,
            "hold_violations": 0,
            "max_cap_violations": 69,
        }
        rqtb = {
            "setup_slack_ns": 0.1511,
            "setup_violations": 0,
            "hold_violations": 0,
            "max_cap_violations": 24,
        }
        negatives, claims = build_physical_boundaries(fixed, rqtb)
        self.assertIn("+0.0885/+0.1511", negatives[0])
        self.assertIn("69/24", negatives[1])
        self.assertIn("均达到", claims[0])


if __name__ == "__main__":
    unittest.main()
