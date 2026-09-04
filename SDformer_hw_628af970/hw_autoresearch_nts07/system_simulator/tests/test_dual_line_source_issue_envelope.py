import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_dual_line_source_issue_envelope.py"
SPEC = importlib.util.spec_from_file_location("dual_line_source_issue_envelope", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DualLineSourceIssueEnvelopeTest(unittest.TestCase):
    def test_system_mapping_replaces_activity_cycles_directly(self):
        result = {
            "sample_count": 10,
            "motion_comparable_operators": 31,
            "local_only_qk_operators": 24,
            "configurations": [{
                "lanes": 96,
                "dense_over_selected": 100.0,
                "selected_cycles": 1000,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "system_summary.json").write_text(json.dumps({
                "config": {"mac_lanes": 96},
                "cycles_per_frame_model": {"fixed_total": 1000},
                "attention": {
                    "fixed_cycles_per_frame": 100,
                    "rqtb_cycles_per_frame": 50,
                },
            }), encoding="utf-8")
            with (root / "operator_transactions.csv").open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "operator", "input_binary_packed_eligible",
                    "replaced_by_attention_rtl_anchor", "activity_cycles_at_config_lanes",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for _ in range(55):
                    writer.writerow({
                        "operator": "Linear",
                        "input_binary_packed_eligible": "True",
                        "replaced_by_attention_rtl_anchor": "False",
                        "activity_cycles_at_config_lanes": "10",
                    })
            MODULE.add_system_envelope(result, root)

        row = result["configurations"][0]
        # frozen = 1000 - 100 - 550 + 50 = 400; candidate = 1000/10 = 100.
        self.assertEqual(row["candidate_eligible_cycles_per_frame"], 100)
        self.assertEqual(row["h67_system_cycles_direct_source_issue"], 500)
        self.assertEqual(row["h67_system_speedup_direct_source_issue"], 2.0)
        self.assertEqual(row["candidate_eligible_speedup_vs_activity_ledger"], 5.5)

    def test_trace_sample_identity_is_required(self):
        trace = [{
            "status": MODULE.PASS,
            "name": "linear",
            "sample_id": "",
            "output_channel_fanout": "1",
            "current_source_count": "1",
            "positive_transition_source_count": "1",
            "negative_transition_source_count": "0",
            "local_work": "1",
            "motion_work": "1",
            "selected_work": "1",
            "selector_saved_work": "0",
            "selector_rows": "1",
            "local_selected_rows": "1",
            "motion_selected_rows": "0",
            "valid_source_work": "1",
        }]
        with self.assertRaisesRegex(ValueError, "sample identity"):
            MODULE.build_identity(trace, [], [16], 4)


if __name__ == "__main__":
    unittest.main()
