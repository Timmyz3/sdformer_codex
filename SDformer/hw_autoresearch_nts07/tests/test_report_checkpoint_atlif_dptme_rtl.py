from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_checkpoint_atlif_dptme_rtl import parse_protocol_sim, parse_sim  # noqa: E402


class ParseCheckpointAtlifSimulationTest(unittest.TestCase):
    def write_log(
        self,
        sampled_errors: int,
        include_metric: bool = True,
        hidden: int = 25_920,
        assertions_enabled: bool = False,
    ) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".log", delete=False
        )
        suffix = (
            f" sampled_protocol_errors={sampled_errors}" if include_metric else ""
        )
        temporary.write(
            "SIMULATOR=icarus\n"
            + ("ASSERTIONS=enabled\n" if assertions_enabled else "")
            + f"ATLIF_DPTME_RESULT commands=81 hidden={hidden} hidden_mismatches=0 "
            + f"events=25920 event_mismatches=0{suffix}\n"
            "PASS: checkpoint-bound ATLIF DP-TME RTL exact\n"
        )
        temporary.close()
        self.addCleanup(Path(temporary.name).unlink)
        return Path(temporary.name)

    def test_accepts_zero_sampled_protocol_errors(self) -> None:
        result = parse_sim(self.write_log(0), "icarus")
        self.assertEqual(result["sampled_protocol_errors"], 0)

    def test_rejects_nonzero_sampled_protocol_errors(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "RTL mismatch"):
            parse_sim(self.write_log(1), "icarus")

    def test_rejects_legacy_log_without_sampled_error_metric(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "incomplete simulation log"):
            parse_sim(self.write_log(0, include_metric=False), "icarus")

    def test_rejects_wrong_comparison_count(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "comparison count mismatch"):
            parse_sim(self.write_log(0, hidden=0), "icarus")

    def test_verilator_requires_assertion_runtime_marker(self) -> None:
        path = self.write_log(0)
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "SIMULATOR=icarus", "SIMULATOR=verilator"
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "SVA runtime"):
            parse_sim(path, "verilator")


class ParseDirectedProtocolSimulationTest(unittest.TestCase):
    def write_log(
        self,
        errors: int = 3,
        assertions_enabled: bool = False,
        simulator: str = "icarus",
    ) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".log", delete=False
        )
        temporary.write(
            f"SIMULATOR={simulator}\n"
            + ("ASSERTIONS=enabled\n" if assertions_enabled else "")
            +
            f"DPTME_PROTOCOL_RESULT sampled_protocol_errors={errors} "
            "tag_reject=1 early_last_reject=1 single_step_reject=1 "
            "state_advance_errors=0\n"
            "PASS: HIT-Flow DP-TME array\n"
        )
        temporary.close()
        self.addCleanup(Path(temporary.name).unlink)
        return Path(temporary.name)

    def test_accepts_complete_directed_coverage(self) -> None:
        self.assertEqual(
            parse_protocol_sim(self.write_log(), "icarus")["tag_reject"], 1
        )

    def test_rejects_incomplete_directed_coverage(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            parse_protocol_sim(self.write_log(errors=2), "icarus")

    def test_rejects_wrong_directed_simulator_identity(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            parse_protocol_sim(self.write_log(simulator="verilator"), "icarus")

    def test_directed_verilator_requires_assertion_marker(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SVA runtime"):
            parse_protocol_sim(
                self.write_log(simulator="verilator"),
                "verilator",
                require_assertions=True,
            )
        self.assertEqual(
            parse_protocol_sim(
                self.write_log(assertions_enabled=True, simulator="verilator"),
                "verilator",
                require_assertions=True,
            )["sampled_protocol_errors"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
