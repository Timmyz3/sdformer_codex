from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim_qfit"))

import local5_erep_calibration_trace_v4 as trace


MINIMAL_ROW = (
    "EREP_V4 schema=local5_erep_raw_trace_v4 candidate=direct_online "
    "event=cycle_snapshot resource=pipeline kind=state cycle=0 window=0 phase=0 "
    "valid=0 ready=0 fire=0 fifo_occupancy=0 projection_busy=0 projection_done=0 "
    "relation_active=0 relation_done=0 protocol_error=0 time=1 scope=fixture"
)
TERMINAL_ZERO = (
    "PASS Local5 EREP calibration v4 direct_terms=0 direct_updates=0 "
    "tcfm5_terms=0 tcfm5_updates=0 serializer_outputs=0"
)
FINISH = "- tb_qfit/tb_qfit_local5_erep_calibration_v4.sv:1: Verilog $finish"


def parse_text(value: str) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.log"
        path.write_text(value, encoding="utf-8")
        return trace.parse_trace(path)


class Local5ErepCalibrationTraceV4SchemaTest(unittest.TestCase):
    def test_exact_minimal_event_schema_and_terminal_are_accepted(self) -> None:
        rows = parse_text(f"{MINIMAL_ROW}\n{TERMINAL_ZERO}\n{FINISH}\n")
        self.assertEqual(len(rows), 1)

    def test_unknown_output_event_or_missing_field_fails_closed(self) -> None:
        variants = (
            f"warning: forged\n{MINIMAL_ROW}\n{TERMINAL_ZERO}\n{FINISH}\n",
            f"{MINIMAL_ROW.replace(' fifo_occupancy=0', '')}\n{TERMINAL_ZERO}\n{FINISH}\n",
            f"{MINIMAL_ROW.replace('event=cycle_snapshot', 'event=forged')}\n{TERMINAL_ZERO}\n{FINISH}\n",
        )
        for value in variants:
            with self.subTest(value=value[:40]):
                with self.assertRaises(ValueError):
                    parse_text(value)

    def test_terminal_count_mismatch_or_duplicate_terminal_fails(self) -> None:
        wrong = TERMINAL_ZERO.replace("direct_terms=0", "direct_terms=1")
        for value in (
            f"{MINIMAL_ROW}\n{wrong}\n{FINISH}\n",
            f"{MINIMAL_ROW}\n{TERMINAL_ZERO}\n{TERMINAL_ZERO}\n{FINISH}\n",
        ):
            with self.subTest(value=value[-80:]):
                with self.assertRaises(ValueError):
                    parse_text(value)

    def test_formal_label_and_seed_cli_do_not_exist(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "sim_qfit/local5_erep_calibration_trace_v4.py"),
                "--trace",
                "/dev/null",
                "--output",
                "/tmp/unused-local5-erep.json",
                "--evidence",
                "formal_profile",
                "--seed",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)


@unittest.skipUnless(os.environ.get("EREP_TRACE_PATH"), "integration trace not supplied")
class Local5ErepCalibrationTraceV4MutationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(os.environ["EREP_TRACE_PATH"]).read_text(encoding="utf-8")

    def assert_normalizer_rejects(self, value: str) -> None:
        self.assertNotEqual(value, self.source, "mutation unexpectedly produced the original trace")
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "mutated.log"
            output = Path(directory) / "output.json"
            raw.write_text(value, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "sim_qfit/local5_erep_calibration_trace_v4.py"),
                    "--trace",
                    str(raw),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse(output.exists())

    def mutate_first(self, needle: str, replacement: str) -> str:
        self.assertIn(needle, self.source)
        return self.source.replace(needle, replacement, 1)

    def mutate_event_field(self, event: str, field: str, replacement: str) -> str:
        lines = self.source.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if f"event={event} " not in line:
                continue
            marker = f" {field}="
            start = line.find(marker)
            self.assertGreaterEqual(start, 0)
            value_start = start + len(marker)
            value_end = line.find(" ", value_start)
            self.assertGreater(value_end, value_start)
            lines[index] = line[:value_start] + replacement + line[value_end:]
            return "".join(lines)
        self.fail(f"event not found: {event}")

    def replace_line(self, predicate: str, transform) -> str:
        lines = self.source.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if predicate in line:
                lines[index] = transform(line)
                return "".join(lines)
        self.fail(f"line not found: {predicate}")

    def swap_event_payloads(self, event: str, fields: tuple[str, ...]) -> str:
        lines = self.source.splitlines(keepends=True)
        indices = [
            index for index, line in enumerate(lines) if f"event={event} " in line
        ]
        self.assertGreaterEqual(len(indices), 2)

        def field_value(line: str, field: str) -> str:
            marker = f" {field}="
            start = line.index(marker) + len(marker)
            end = line.index(" ", start)
            return line[start:end]

        def replace_field(line: str, field: str, value: str) -> str:
            marker = f" {field}="
            start = line.index(marker) + len(marker)
            end = line.index(" ", start)
            return line[:start] + value + line[end:]

        left, right = indices[:2]
        left_values = {field: field_value(lines[left], field) for field in fields}
        right_values = {field: field_value(lines[right], field) for field in fields}
        for field in fields:
            lines[left] = replace_field(lines[left], field, right_values[field])
            lines[right] = replace_field(lines[right], field, left_values[field])
        return "".join(lines)

    def test_deleted_relation_payload_final_boundary_and_fatal_are_rejected(self) -> None:
        relation_line = next(
            line for line in self.source.splitlines() if "event=relation_accept" in line
        )
        variants = (
            self.source.replace(relation_line + "\n", "", 1),
            self.mutate_event_field("relation_accept", "gates", "dead"),
            self.mutate_event_field("drain_read_response", "data", "999"),
            self.mutate_first("kind=FILL_BEGIN cycle=2", "kind=FILL_BEGIN cycle=3"),
            self.source + "FATAL forged after PASS\n",
        )
        for value in variants:
            with self.subTest(index=variants.index(value)):
                self.assert_normalizer_rejects(value)

    def test_independent_review_semantic_attacks_are_rejected(self) -> None:
        forged_boundary = next(
            line
            for line in self.source.splitlines()
            if "candidate=tcfm5_1rw event=phase_boundary kind=PREPARE_BEGIN" in line
        ).replace("kind=PREPARE_BEGIN", "kind=FORGED_BOUNDARY")
        terminal = next(
            line
            for line in self.source.splitlines()
            if line.startswith("PASS Local5 EREP calibration v4 ")
        )
        terminal_first = terminal + "\n" + self.source.replace(terminal + "\n", "", 1)
        variants = (
            self.replace_line(
                "candidate=direct_online event=relation_accept",
                lambda line: line.replace(
                    "resource=relation_workspace_1rw", "resource=forged_workspace", 1
                ),
            ),
            self.replace_line(
                "candidate=tcfm5_1rw event=phase_boundary kind=PREPARE_BEGIN",
                lambda line: line.replace("phase=0", "phase=4", 1),
            ),
            self.source.replace(terminal + "\n", forged_boundary + "\n" + terminal + "\n", 1),
            terminal_first,
            self.replace_line(
                "candidate=tcfm5_1rw event=stall_observation resource=vector_serializer",
                lambda line: line.replace(" data=15 ", " data=16 ", 1),
            ),
        )
        for index, value in enumerate(variants):
            with self.subTest(index=index):
                self.assert_normalizer_rejects(value)

    def test_cross_segment_commit_and_stall_attacks_are_rejected(self) -> None:
        relation_and_fifo = self.replace_line(
            "candidate=direct_online event=relation_accept resource=relation_workspace_1rw kind=relation_write cycle=2 ",
            lambda line: line.replace(" gates=1 ", " gates=1ff ", 1),
        )
        original = self.source
        self.source = relation_and_fifo
        try:
            relation_and_fifo = self.replace_line(
                "candidate=direct_online event=fifo_enqueue",
                lambda line: line.replace(" gates=1 ", " gates=1ff ", 1),
            )
        finally:
            self.source = original
        commit_zero = self.replace_line(
            "candidate=tcfm5_1rw event=term_accept",
            lambda line: line.replace(" commit=1 ", " commit=0 ", 1),
        )
        direct_stall = next(
            line
            for line in self.source.splitlines()
            if "candidate=direct_online event=stall_observation resource=execute_lane" in line
        )
        deleted_stall = self.source.replace(direct_stall + "\n", "", 1)
        for value in (relation_and_fifo, commit_zero, deleted_stall):
            self.assert_normalizer_rejects(value)

    def test_duplicate_stall_cycle_and_run_shape_attacks_are_rejected(self) -> None:
        lines = self.source.splitlines(keepends=True)
        direct_indices = [
            index
            for index, line in enumerate(lines)
            if "candidate=direct_online event=stall_observation resource=execute_lane" in line
        ]
        self.assertGreaterEqual(len(direct_indices), 2)
        duplicate_direct = list(lines)
        duplicate_direct[direct_indices[1]] = duplicate_direct[direct_indices[0]]

        serializer_indices = [
            index
            for index, line in enumerate(lines)
            if "candidate=tcfm5_1rw event=stall_observation resource=vector_serializer" in line
        ]
        self.assertEqual(len(serializer_indices), 18)
        duplicate_serializer = list(lines)
        for index in serializer_indices:
            duplicate_serializer[index] = duplicate_serializer[serializer_indices[1]]

        broken_serializer_run = list(lines)
        broken_serializer_run[serializer_indices[0]] = broken_serializer_run[
            serializer_indices[0]
        ].replace(" cycle=162 ", " cycle=161 ", 1)

        for value in (
            "".join(duplicate_direct),
            "".join(duplicate_serializer),
            "".join(broken_serializer_run),
        ):
            self.assert_normalizer_rejects(value)

    def test_time_and_rtl_width_attacks_are_rejected(self) -> None:
        variants = (
            self.replace_line(
                "candidate=direct_online event=relation_accept",
                lambda line: line.replace(" time=81000 ", " time=1 ", 1),
            ),
            self.replace_line(
                "candidate=direct_online event=stall_observation resource=execute_lane",
                lambda line: line[: line.index(" time=")]
                + " time=999999999"
                + line[line.index(" scope=") :],
            ),
            self.replace_line(
                "candidate=direct_online event=relation_accept",
                lambda line: line.replace(" candidate_valid=1 ", " candidate_valid=ffffffff ", 1),
            ),
            self.replace_line(
                "candidate=direct_online event=term_accept",
                lambda line: line.replace(" lane=0 ", " lane=4 ", 1),
            ),
            self.replace_line(
                "candidate=direct_online event=acc_update_accept",
                lambda line: line.replace(" address=0 ", " address=3 ", 1),
            ),
            self.replace_line(
                "candidate=tcfm5_1rw event=term_accept",
                lambda line: (
                    line[: line.index(" gate=") + len(" gate=")]
                    + "512"
                    + line[
                        line.index(
                            " ", line.index(" gate=") + len(" gate=")
                        ) :
                    ]
                ),
            ),
        )
        for index, value in enumerate(variants):
            with self.subTest(index=index):
                self.assert_normalizer_rejects(value)

    def test_final_stream_snapshot_and_fixture_mutations_are_rejected(self) -> None:
        coherent_fire = self.replace_line(
            "candidate=direct_online event=cycle_snapshot resource=pipeline kind=state cycle=0 ",
            lambda line: line.replace(" valid=0 ", " valid=2 ", 1).replace(
                " fire=0 ", " fire=2 ", 1
            ),
        )
        coherent_boundary = self.source.replace(
            "candidate=tcfm5_1rw event=phase_boundary kind=EXECUTE_BEGIN cycle=1 ",
            "candidate=tcfm5_1rw event=phase_boundary kind=EXECUTE_BEGIN cycle=2 ",
            1,
        ).replace(
            "candidate=tcfm5_1rw event=cycle_snapshot resource=pipeline kind=state cycle=1 window=0 phase=0 state=2 valid=0 ready=3 fire=0 run_busy=1",
            "candidate=tcfm5_1rw event=cycle_snapshot resource=pipeline kind=state cycle=1 window=0 phase=0 state=0 valid=0 ready=3 fire=0 run_busy=0",
            1,
        ).replace(
            "candidate=tcfm5_1rw event=cycle_snapshot resource=pipeline kind=state cycle=2 window=0 phase=2 state=2",
            "candidate=tcfm5_1rw event=cycle_snapshot resource=pipeline kind=state cycle=2 window=0 phase=0 state=2",
            1,
        )
        variants = (
            self.mutate_first(
                "source_id=8 plane=0 y=2 x=2 out=1 data=0 last=1",
                "source_id=8 plane=0 y=2 x=2 out=1 data=0 last=0",
            ),
            self.swap_event_payloads(
                "serializer_output",
                ("source_id", "plane", "y", "x", "out", "data", "last"),
            ),
            self.swap_event_payloads(
                "drain_read_response", ("source_id", "out", "data")
            ),
            self.replace_line(
                "candidate=direct_online event=cycle_snapshot",
                lambda line: line.replace(" fire=0 ", " fire=1 ", 1),
            ),
            self.replace_line(
                "candidate=direct_online event=cycle_snapshot",
                lambda line: line.replace(
                    " projection_busy=0 ", " projection_busy=1 ", 1
                ),
            ),
            self.replace_line(
                "candidate=direct_online event=relation_accept",
                lambda line: line.replace(
                    " candidate_valid=1 ", " candidate_valid=3 ", 1
                ),
            ),
            coherent_fire,
            coherent_boundary,
        )
        for index, value in enumerate(variants):
            with self.subTest(index=index):
                self.assert_normalizer_rejects(value)


if __name__ == "__main__":
    unittest.main()
