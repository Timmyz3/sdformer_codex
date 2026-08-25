import copy
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


WRITER = None
REFERENCE = None
ENTRYPOINTS = (
    Path(__file__).resolve().parents[3]
    / "neuron_experiments/H9_bipolar_self_attention/entrypoints"
)
if torch is not None:
    WRITER_SPEC = importlib.util.spec_from_file_location(
        "h67_dual_line_cohort_census", ENTRYPOINTS / "h67_dual_line_cohort_census.py"
    )
    WRITER = importlib.util.module_from_spec(WRITER_SPEC)
    assert WRITER_SPEC.loader is not None
    WRITER_SPEC.loader.exec_module(WRITER)
    REFERENCE_SPEC = importlib.util.spec_from_file_location(
        "h67_dual_line_trace", ENTRYPOINTS / "h67_dual_line_trace.py"
    )
    REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
    assert REFERENCE_SPEC.loader is not None
    REFERENCE_SPEC.loader.exec_module(REFERENCE)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


class ProfilerCohortIntegrationContractTest(unittest.TestCase):
    def test_profiler_has_fail_closed_reference_reconciled_wiring(self):
        source = (ENTRYPOINTS / "profile_nts11_hardware_p0.py").read_text(
            encoding="utf-8"
        )
        required_fragments = (
            "--dual-line-cohort-census-dir",
            "--cohort-census-source-chunk-size",
            "--cohort-census-row-block-size",
            "--cohort-census-max-working-set-mib",
            "temporal_work = profile_operator_temporal_work(",
            "reference_work=temporal_work",
            '"cohort_census_writer": file_sha256(cohort_writer_source.resolve())',
            '"dual_line_reference": file_sha256(',
            "if profile_completed and close_completed:",
            'dual_line_cohort_writer.abort("profiler did not complete")',
            '"dual_line_cohort_census_manifest"',
        )
        for fragment in required_fragments:
            self.assertIn(fragment, source)

    def test_r2_padding_exact_gate_and_manifest_split_are_source_locked(self):
        reference_source = (ENTRYPOINTS / "h67_dual_line_trace.py").read_text(
            encoding="utf-8"
        )
        writer_source = (
            ENTRYPOINTS / "h67_dual_line_cohort_census.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _require_zero_padding", reference_source)
        self.assertIn(
            "dual-line exact Conv2d requires padding_mode='zeros'",
            reference_source,
        )
        self.assertIn(
            "M24C exact Conv2d requires padding_mode='zeros'", writer_source
        )
        self.assertIn("if exact_calls <= 0:", writer_source)
        self.assertIn('"exact_calls": exact_calls', writer_source)
        self.assertIn('"rejected_calls": rejected_calls', writer_source)
        self.assertIn(
            '"dual_line_reference",', writer_source
        )


@unittest.skipUnless(torch is not None, "algorithm-side torch is optional on hardware server")
class StreamingCohortCensusTest(unittest.TestCase):
    @staticmethod
    def _bind(writer):
        writer.bind_run_context({
            "artifact_identity": {
                "checkpoint_sha256": "a" * 64,
                "config_sha256": "b" * 64,
            },
            "eval_protocol": {"temporal_steps": 10, "batch_size": 1},
            "checkpoint_load_audit": {"missing": 0, "unexpected": 0},
            "source_sha256": {
                "profiler": "c" * 64,
                "cohort_census_writer": "d" * 64,
                "dual_line_reference": "e" * 64,
            },
        })

    @staticmethod
    def _reference(module, value):
        return REFERENCE.profile_operator_temporal_work(
            module, value, temporal_steps=10
        )

    @staticmethod
    def _record(writer, module, value, reference=None, name="synthetic"):
        if reference is None:
            reference = StreamingCohortCensusTest._reference(module, value)
        writer.record_operator(
            module,
            value,
            reference_work=reference,
            name=name,
            sample_id=7,
            sample_key="sample-7",
            sequence_key="sequence-A",
            operator_call_index=3,
            temporal_steps=10,
        )
        return reference

    @staticmethod
    def _read_csv(output_dir):
        with (Path(output_dir) / "call_reconciliation.csv").open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _read_histograms(output_dir):
        with (Path(output_dir) / "cohort_histograms.jsonl").open(
            "r", encoding="utf-8"
        ) as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _assert_reference_conservation(self, row, reference, fanout):
        self.assertEqual(
            int(row["valid_source_count"]),
            sum(int(item["valid_source_work"]) // fanout for item in reference),
        )
        for field in (
            "current_source_count",
            "positive_transition_source_count",
            "negative_transition_source_count",
        ):
            self.assertEqual(
                int(row[field]), sum(int(item[field]) for item in reference)
            )
        self.assertEqual(
            int(row["selected_source_count"]),
            sum(int(item["selected_work"]) // fanout for item in reference),
        )
        self.assertEqual(
            int(row["destination_scalar_updates"]),
            sum(int(item["selected_work"]) for item in reference),
        )
        self.assertEqual(
            int(row["selected_positive_source_count"])
            + int(row["selected_negative_source_count"]),
            int(row["selected_source_count"]),
        )
        self.assertEqual(
            int(row["local_cohort_coefficient_scalar_reads"]),
            int(row["local_cohort_coefficient_vectors"]) * fanout,
        )
        self.assertEqual(
            int(row["motion_cohort_coefficient_scalar_reads"]),
            int(row["motion_cohort_coefficient_vectors"]) * fanout,
        )

    @staticmethod
    def _expected_linear_masks(value, row_index, source_start, source_stop):
        full = value[:, row_index].eq(1)
        previous = torch.zeros_like(full[0])
        local_masks = [0 for _ in range(int(full.shape[1]))]
        positive_masks = [0 for _ in range(int(full.shape[1]))]
        negative_masks = [0 for _ in range(int(full.shape[1]))]
        for timestep in range(10):
            current = full[timestep]
            positive = current & ~previous
            negative = previous & ~current
            choose_motion = bool(
                timestep > 0
                and int((positive | negative).sum().item())
                < int(current.sum().item())
            )
            for source in range(int(full.shape[1])):
                if bool(current[source].item()):
                    local_masks[source] |= 1 << timestep
                if choose_motion:
                    if bool(positive[source].item()):
                        positive_masks[source] |= 1 << timestep
                    if bool(negative[source].item()):
                        negative_masks[source] |= 1 << timestep
                elif bool(current[source].item()):
                    positive_masks[source] |= 1 << timestep
            previous = current

        def sparse(values):
            return [
                [int(mask), int(count)]
                for mask, count in sorted(Counter(values).items())
            ]

        selected = slice(source_start, source_stop)
        pairs = [
            positive | (negative << 10)
            for positive, negative in zip(
                positive_masks[selected], negative_masks[selected]
            )
        ]
        return {
            "local_presence_histogram_nonzero_bins": sparse(local_masks[selected]),
            "motion_positive_histogram_nonzero_bins": sparse(positive_masks[selected]),
            "motion_negative_histogram_nonzero_bins": sparse(negative_masks[selected]),
            "motion_signed_pair_histogram_nonzero_bins": sparse(pairs),
        }

    def test_linear_all_rows_tail_chunks_and_exact_mask_census(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "linear_census"
            writer = WRITER.StreamingCohortCensusWriter(
                output_dir,
                source_chunk_size=3,
                requested_row_block_size=1,
                max_working_set_mib=1,
            )
            self._bind(writer)
            linear = torch.nn.Linear(5, 4, bias=False)
            temporal = torch.arange(10).reshape(10, 1, 1)
            rows = torch.arange(2).reshape(1, 2, 1)
            sources = torch.arange(5).reshape(1, 1, 5)
            value = ((temporal + 2 * rows + sources) % 4 < 2).to(torch.float32)
            reference = self._record(writer, linear, value, name="linear5")
            writer.close()

            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], WRITER.PASS_STATUS)
            self.assertTrue(manifest["working_set_bound_satisfied"])
            self.assertFalse(manifest["raw_activation_or_npz_saved"])
            self.assertEqual(manifest["calls"], 1)
            self.assertEqual(manifest["exact_calls"], 1)
            self.assertEqual(manifest["rejected_calls"], 0)
            self.assertEqual(
                manifest["run_context"]["source_sha256"]["cohort_census_writer"],
                "d" * 64,
            )
            self.assertEqual(
                set(path.name for path in output_dir.iterdir()),
                {"manifest.json", "cohort_histograms.jsonl", "call_reconciliation.csv"},
            )
            for name, identity in manifest["files"].items():
                self.assertEqual(_sha256(output_dir / name), identity["sha256"])

            reconciliation = self._read_csv(output_dir)
            self.assertEqual(len(reconciliation), 1)
            row = reconciliation[0]
            self.assertEqual(int(row["rows"]), 2)
            self.assertEqual(int(row["source_width"]), 5)
            self.assertEqual(int(row["source_chunks"]), 2)
            self.assertEqual(int(row["histogram_records"]), 4)
            self._assert_reference_conservation(row, reference, fanout=4)

            records = self._read_histograms(output_dir)
            self.assertEqual(len(records), 4)
            self.assertEqual({item["sample_key"] for item in records}, {"sample-7"})
            self.assertEqual({item["sequence_key"] for item in records}, {"sequence-A"})
            self.assertEqual({item["operator_call_index"] for item in records}, {3})
            self.assertEqual({item["weight_group"] for item in records}, {0})
            self.assertEqual({item["output_channel_fanout"] for item in records}, {4})
            self.assertEqual({item["valid_bits"] for item in records}, {2, 3})
            for item in records:
                population = (item["row_stop_exclusive"] - item["row_start"]) * item["valid_bits"]
                self.assertEqual(item["row_source_identity_count"], population)
                signed_population = sum(
                    count for code, count in item["motion_signed_pair_histogram_nonzero_bins"]
                )
                self.assertEqual(signed_population, population)
                for code, count in item["motion_signed_pair_histogram_nonzero_bins"]:
                    self.assertGreater(count, 0)
                    self.assertEqual((code & 0x3FF) & ((code >> 10) & 0x3FF), 0)
                expected = self._expected_linear_masks(
                    value,
                    item["row_start"],
                    item["source_base"],
                    item["source_base"] + item["valid_bits"],
                )
                for field, histogram in expected.items():
                    self.assertEqual(item[field], histogram)

    def test_grouped_conv2d_padding_batches_and_tail_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "conv_census"
            writer = WRITER.StreamingCohortCensusWriter(
                output_dir,
                source_chunk_size=7,
                requested_row_block_size=4,
                max_working_set_mib=1,
            )
            self._bind(writer)
            conv = torch.nn.Conv2d(
                4, 6, kernel_size=3, stride=1, padding=1, groups=2, bias=False
            )
            temporal = torch.arange(10).reshape(10, 1, 1, 1, 1)
            batch = torch.arange(2).reshape(1, 2, 1, 1, 1)
            channel = torch.arange(4).reshape(1, 1, 4, 1, 1)
            y = torch.arange(3).reshape(1, 1, 1, 3, 1)
            x = torch.arange(3).reshape(1, 1, 1, 1, 3)
            value = ((temporal + batch + channel + 2 * y + x) % 5 < 2).to(
                torch.float32
            )
            reference = self._record(writer, conv, value, name="grouped_conv")
            writer.close()

            reconciliation = self._read_csv(output_dir)
            self.assertEqual(len(reconciliation), 1)
            row = reconciliation[0]
            self.assertEqual(int(row["rows"]), 36)
            self.assertEqual(int(row["source_width"]), 18)
            self.assertEqual(int(row["source_chunks"]), 3)
            self.assertEqual(int(row["weight_groups"]), 2)
            self.assertEqual(int(row["output_channel_fanout"]), 3)
            self.assertEqual(int(row["histogram_records"]), 36)
            self.assertLess(int(row["valid_source_count"]), 36 * 18 * 10)
            self._assert_reference_conservation(row, reference, fanout=3)

            records = self._read_histograms(output_dir)
            self.assertEqual(len(records), 36)
            self.assertEqual({item["weight_group"] for item in records}, {0, 1})
            self.assertEqual({item["valid_bits"] for item in records}, {4, 7})
            self.assertEqual({item["output_channel_fanout"] for item in records}, {3})
            self.assertEqual(max(item["row_stop_exclusive"] for item in records), 36)

    def test_all_rejections_cannot_publish_pass_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "reject_census"
            writer = WRITER.StreamingCohortCensusWriter(output_dir, max_working_set_mib=1)
            self._bind(writer)
            linear = torch.nn.Linear(2, 2, bias=False)
            temporal_bad = torch.zeros((9, 1, 2), dtype=torch.float32)
            nonbinary = torch.zeros((10, 1, 2), dtype=torch.float32)
            nonbinary[4, 0, 1] = -1.0
            self._record(writer, linear, temporal_bad, name="bad_temporal")
            self._record(writer, linear, nonbinary, name="bad_binary")
            with self.assertRaisesRegex(
                ValueError, "cannot publish without at least one exact call"
            ):
                writer.close()

            self.assertFalse(output_dir.exists())
            self.assertFalse((writer.staging_dir / "manifest.json").exists())
            interrupted = json.loads(
                (writer.staging_dir / "INTERRUPTED.json").read_text()
            )
            self.assertEqual(interrupted["exact_calls_before_abort"], 0)
            self.assertEqual(interrupted["rejected_calls_before_abort"], 2)
            self.assertFalse(interrupted["pass_manifest_written"])

    def test_nonzero_conv_padding_modes_fail_before_exact_or_rejection(self):
        value = torch.zeros((10, 1, 1, 4, 4), dtype=torch.float32)
        for padding_mode in ("reflect", "replicate", "circular"):
            with self.subTest(padding_mode=padding_mode):
                with tempfile.TemporaryDirectory() as directory:
                    output_dir = Path(directory) / "padding_census"
                    writer = WRITER.StreamingCohortCensusWriter(
                        output_dir, max_working_set_mib=1
                    )
                    self._bind(writer)
                    conv = torch.nn.Conv2d(
                        1, 2, kernel_size=3, padding=1,
                        padding_mode=padding_mode, bias=False,
                    )
                    with self.assertRaisesRegex(ValueError, "padding_mode='zeros'"):
                        self._reference(conv, value)
                    with self.assertRaisesRegex(ValueError, "padding_mode='zeros'"):
                        self._record(writer, conv, value, reference=[])
                    self.assertEqual(writer.calls, 0)
                    self.assertEqual(writer.histogram_records, 0)
                    writer.abort("unsupported padding mode")
                    self.assertFalse(output_dir.exists())
                    self.assertTrue(
                        (writer.staging_dir / "INTERRUPTED.json").is_file()
                    )

    def test_reference_mismatch_aborts_without_pass_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "mismatch_census"
            writer = WRITER.StreamingCohortCensusWriter(output_dir, max_working_set_mib=1)
            self._bind(writer)
            linear = torch.nn.Linear(4, 3, bias=False)
            value = (torch.arange(80).reshape(10, 2, 4) % 3 == 0).to(torch.float32)
            reference = copy.deepcopy(self._reference(linear, value))
            reference[2]["selected_work"] += 1
            with self.assertRaisesRegex(ValueError, "M24C/reference mismatch"):
                self._record(writer, linear, value, reference=reference)
            writer.abort("synthetic reference mismatch")

            self.assertFalse(output_dir.exists())
            self.assertTrue((writer.staging_dir / "INTERRUPTED.json").is_file())
            self.assertFalse((writer.staging_dir / "manifest.json").exists())
            interrupted = json.loads(
                (writer.staging_dir / "INTERRUPTED.json").read_text()
            )
            self.assertEqual(interrupted["status"], "INTERRUPTED_NOT_ADMITTED")
            self.assertFalse(interrupted["pass_manifest_written"])

    def test_shape_mismatch_fails_closed_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "shape_census"
            writer = WRITER.StreamingCohortCensusWriter(output_dir, max_working_set_mib=1)
            self._bind(writer)
            linear = torch.nn.Linear(4, 3, bias=False)
            good = torch.zeros((10, 1, 4), dtype=torch.float32)
            bad = torch.zeros((10, 1, 5), dtype=torch.float32)
            reference = self._reference(linear, good)
            with self.assertRaisesRegex(ValueError, "feature dimension mismatch"):
                self._record(writer, linear, bad, reference=reference)
            writer.abort("synthetic shape mismatch")
            self.assertFalse(output_dir.exists())
            self.assertTrue((writer.staging_dir / "INTERRUPTED.json").is_file())

    def test_memory_model_adapts_block_and_fails_if_one_row_cannot_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "bounded_census"
            writer = WRITER.StreamingCohortCensusWriter(
                output_dir,
                source_chunk_size=256,
                requested_row_block_size=100000,
                max_working_set_mib=1,
            )
            self._bind(writer)
            linear = torch.nn.Linear(128, 2, bias=False)
            value = (torch.arange(10 * 100 * 128).reshape(10, 100, 128) % 7 == 0).to(
                torch.float32
            )
            self._record(writer, linear, value)
            writer.close()
            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertLess(manifest["maximum_row_block_size"], 100000)
            self.assertLessEqual(
                manifest["maximum_estimated_working_set_bytes"], 1 << 20
            )
            self.assertTrue(manifest["working_set_bound_satisfied"])

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "impossible_census"
            writer = WRITER.StreamingCohortCensusWriter(
                output_dir,
                requested_row_block_size=1,
                max_working_set_mib=1,
            )
            self._bind(writer)
            linear = torch.nn.Linear(8000, 1, bias=False)
            value = torch.zeros((10, 1, 8000), dtype=torch.float32)
            reference = self._reference(linear, value)
            with self.assertRaisesRegex(ValueError, "cannot hold one source row"):
                self._record(writer, linear, value, reference=reference)
            writer.abort("synthetic memory bound")
            self.assertFalse(output_dir.exists())
            self.assertTrue((writer.staging_dir / "INTERRUPTED.json").is_file())


if __name__ == "__main__":
    unittest.main()
