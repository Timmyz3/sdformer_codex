from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import local5_erep_statistics_v4 as statistics


def synthetic_validated_rows(
    *, c0: int = 125, c1: int = 105, c2: int = 106, c3: int = 100, c4: int = 105
) -> list[dict[str, object]]:
    rows = []
    stage_order = (0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3)
    block_order = (0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 0, 1)
    for sample in range(100):
        for slot, (stage, block) in enumerate(zip(stage_order, block_order, strict=True)):
            rows.append(
                {
                    "sample": sample,
                    "sample_key": f"sample-{sample}",
                    "sequence_key": f"seq-{sample % 18:02d}",
                    "stage": stage,
                    "block": block,
                    "window": (sample + slot) % statistics.STAGE_WEIGHTS[stage],
                    "weight": statistics.STAGE_WEIGHTS[stage],
                    "command_ledger_sha256": f"{sample * 12 + slot:064x}",
                    "c4_source": "trace_derived_relaxed_oracle_5014_records_v4",
                    "c0": c0,
                    "c1": c1,
                    "c2": c2,
                    "c3": c3,
                    "c4": c4,
                }
            )
    return rows


class Local5ErepStatisticsV4Test(unittest.TestCase):
    def test_production_entry_has_no_input_seed_or_trial_parameters(self) -> None:
        self.assertEqual(tuple(inspect.signature(statistics.evaluate_formal_g0).parameters), ())
        source = inspect.getsource(statistics._evaluate_validated_rows)
        self.assertIn("np.random.PCG64(20260810)", source)
        self.assertIn("range(20_000)", source)

    def test_formal_entry_refuses_while_fixed_admission_artifacts_are_absent(self) -> None:
        self.assertFalse(statistics.ADMISSION_RECEIPT.exists())
        try:
            statistics._verify_runtime()
        except RuntimeError:
            with self.assertRaisesRegex(RuntimeError, "active interpreter"):
                statistics.evaluate_formal_g0()
        else:
            with self.assertRaisesRegex(ValueError, "required frozen artifact is absent"):
                statistics.evaluate_formal_g0()

    def test_exact_weighted_quantile_and_order_statistic(self) -> None:
        self.assertEqual(statistics._weighted_quantile([10, 20, 30], [1, 1, 2], Fraction(1, 4)), 10)
        self.assertEqual(statistics._weighted_quantile([10, 20, 30], [1, 1, 2], Fraction(1, 4) + Fraction(1, 10**9)), 20)
        self.assertEqual(statistics._order_statistic(list(range(80)), Fraction(79, 80)), 78)

    def test_golden_exact_thresholds_pass_with_frozen_bootstrap(self) -> None:
        report = statistics._evaluate_validated_rows(synthetic_validated_rows())
        self.assertTrue(report["g0_passed"])
        self.assertEqual(report["ratios"]["c0_over_c3"]["estimate"], {
            "numerator": 5, "denominator": 4, "decimal": 1.25
        })
        self.assertEqual(report["synergy"]["estimate"]["numerator"], 21)
        self.assertEqual(report["synergy"]["estimate"]["denominator"], 20)
        self.assertEqual(report["determinism"]["trials"], 20_000)

    def test_large_integer_threshold_cannot_round_into_a_pass(self) -> None:
        rows = synthetic_validated_rows()
        # Make C0/C3 exactly smaller than 5/4 by one weighted unit while all
        # individual cycles remain far beyond float64's exact-integer range.
        base = (1 << 60) + 1000
        for row in rows:
            row["c3"] = base
            row["c0"] = 5 * base // 4
        rows[0]["c0"] -= 1
        report = statistics._evaluate_validated_rows(rows)
        exact_c0 = report["weighted_totals_exact_integer"]["c0"]
        exact_c3 = report["weighted_totals_exact_integer"]["c3"]
        self.assertLess(4 * exact_c0, 5 * exact_c3)
        gate = next(row for row in report["g0_gates"] if row["name"] == "primary_speedup_c0_over_c3")
        self.assertFalse(gate["passed"])
        self.assertFalse(report["g0_passed"])

    def test_row_validation_binds_plan_window_cohort_command_and_c4_source(self) -> None:
        rows = synthetic_validated_rows()
        plans = []
        commands = []
        sample_keys = [f"sample-{sample}" for sample in range(100)]
        sequence_keys = [f"seq-{sample % 18:02d}" for sample in range(100)]
        for row in rows:
            stage = int(row["stage"])
            plans.append(
                {
                    "sample": row["sample"], "stage": stage, "block": row["block"],
                    "heads": (3, 6, 12, 24)[stage],
                    "batch_windows": statistics.STAGE_WEIGHTS[stage],
                    "window": row["window"],
                    "inclusion_probability": 1.0 / statistics.STAGE_WEIGHTS[stage],
                    "analysis_weight": float(statistics.STAGE_WEIGHTS[stage]),
                }
            )
            command_body = {
                key: row[key]
                for key in (
                    "sample", "stage", "block", "window",
                    "c0", "c1", "c2", "c3", "c4",
                )
            }
            command_body["window_schedule_sha256"] = statistics._canonical_sha(
                [row["sample"], row["stage"], row["block"], row["window"]]
            )
            command_digest = statistics._canonical_sha(command_body)
            row["command_ledger_sha256"] = command_digest
            commands.append({**command_body, "command_ledger_sha256": command_digest})
        accepted = statistics._validate_rows(rows, plans, sample_keys, sequence_keys, commands)
        self.assertEqual(len(accepted), 1200)
        mutations = (
            ("window", int(rows[0]["window"]) + 1, "selection plan"),
            ("sample_key", "forged", "formal cohort"),
            ("sequence_key", "forged", "formal cohort"),
            ("command_ledger_sha256", "f" * 64, "command ledger"),
            ("c4_source", "caller_supplied", "C4 source"),
        )
        for field, value, message in mutations:
            modified = [dict(row) for row in rows]
            modified[0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    statistics._validate_rows(modified, plans, sample_keys, sequence_keys, commands)

    def test_strict_python_integer_rejects_bool_float_and_numpy_style_subclasses(self) -> None:
        for value in (True, 1.0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    statistics._strict_uint(value, "test", positive=True)

    def test_runtime_receipt_and_pcg64_golden_vector_are_bound(self) -> None:
        receipt = statistics._verify_runtime()
        self.assertEqual(receipt["python_executable"], "/opt/conda/bin/python3.11")
        self.assertEqual(receipt["pcg64_first_draw_18"], [16, 5, 2, 12, 15, 11, 4, 16, 0, 3, 10, 14, 6, 3, 17, 9, 12, 9])

    def test_archive_bindings_reject_missing_bytes_hashes_and_basename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "rtl_trace_archive.npz"
            miter = root / "acc32_miter_archive.npz"
            trace.write_bytes(b"trace-fixture")
            miter.write_bytes(b"miter-fixture")
            trace_sha = statistics._sha256(trace)
            miter_sha = statistics._sha256(miter)
            receipt = {
                "rtl_trace_archive_sha256": trace_sha,
                "acc32_miter_archive_sha256": miter_sha,
            }
            head = {
                "rtl_trace_archive_file": trace.name,
                "rtl_trace_archive_sha256": trace_sha,
                "acc32_miter_archive_file": miter.name,
                "acc32_miter_archive_sha256": miter_sha,
            }
            self.assertEqual(
                statistics._verify_archive_bindings(
                    receipt, head, admission_dir=root
                ),
                (trace, miter),
            )
            mutations = []
            missing = (dict(receipt), dict(head), "missing")
            mutations.append(missing)
            bad_head = (dict(receipt), dict(head), "head_sha")
            bad_head[1]["rtl_trace_archive_sha256"] = "0" * 64
            mutations.append(bad_head)
            bad_receipt = (dict(receipt), dict(head), "receipt_sha")
            bad_receipt[0]["acc32_miter_archive_sha256"] = "0" * 64
            mutations.append(bad_receipt)
            bad_name = (dict(receipt), dict(head), "basename")
            bad_name[1]["rtl_trace_archive_file"] = "other.npz"
            mutations.append(bad_name)
            for changed_receipt, changed_head, name in mutations:
                if name == "missing":
                    trace.unlink()
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "not SHA-bound"
                ):
                    statistics._verify_archive_bindings(
                        changed_receipt, changed_head, admission_dir=root
                    )
                if name == "missing":
                    trace.write_bytes(b"trace-fixture")

            trace.write_bytes(b"changed-trace-fixture")
            with self.assertRaisesRegex(ValueError, "not SHA-bound"):
                statistics._verify_archive_bindings(
                    receipt, head, admission_dir=root
                )


if __name__ == "__main__":
    unittest.main()
