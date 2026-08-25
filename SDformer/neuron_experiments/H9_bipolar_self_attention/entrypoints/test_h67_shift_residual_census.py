import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


MODULE = None
if torch is not None:
    source = Path(__file__).with_name("h67_shift_residual_census.py")
    spec = importlib.util.spec_from_file_location("h67_shift_residual_census", source)
    MODULE = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(MODULE)


@unittest.skipUnless(torch is not None, "PyTorch is optional on the hardware server")
class ShiftResidualCensusTest(unittest.TestCase):
    @staticmethod
    def _bind(writer, batch_size=1):
        writer.bind_run_context({
            "artifact_identity": {
                "checkpoint_sha256": "a" * 64,
                "config_sha256": "b" * 64,
            },
            "eval_protocol": {
                "temporal_steps": 10,
                "eval_batch_size": batch_size,
                "requested_profile_samples": writer.expected_samples,
                "expected_operator_calls": writer.expected_operator_calls,
                "expected_exact_calls": writer.expected_exact_calls,
                "temporal_axis_contract": "hook_input_dim0_is_T10_and_dim1_is_eval_batch",
            },
            "checkpoint_load_audit": {"missing": 0, "unexpected": 0},
            "source_sha256": {
                "profiler": "c" * 64,
                "shift_residual_census_writer": "d" * 64,
            },
        })

    @staticmethod
    def _moving_binary(steps, batches, channels, height, width, step_y=0, step_x=1):
        base = torch.zeros((batches, channels, height, width), dtype=torch.bool)
        for batch in range(batches):
            for channel in range(channels):
                base[batch, channel, (channel + batch) % height, :] = True
                base[batch, channel, :, (2 * channel + batch) % width] ^= True
        frames = []
        current = base
        for _ in range(steps):
            frames.append(current)
            shifted = torch.zeros_like(current)
            y_src_start = max(0, step_y)
            y_src_stop = min(height, height + step_y)
            x_src_start = max(0, step_x)
            x_src_stop = min(width, width + step_x)
            y_dst_start = max(0, -step_y)
            y_dst_stop = min(height, height - step_y)
            x_dst_start = max(0, -step_x)
            x_dst_stop = min(width, width - step_x)
            shifted[:, :, y_dst_start:y_dst_stop, x_dst_start:x_dst_stop] = current[
                :, :, y_src_start:y_src_stop, x_src_start:x_src_stop
            ]
            current = shifted
        return torch.stack(frames)

    def _assert_dense_exact(self, module, value, tile, radius=1):
        candidates = MODULE.canonical_shift_candidates(radius)
        binary, geometry = MODULE._conv_geometry(module, value)
        dense = torch.stack([module(value[t].to(torch.float32)) for t in range(10)])
        selected_nonzero = 0
        selected_border = 0
        for timestep in range(10):
            for batch in range(geometry["batches"]):
                for group in range(geometry["groups"]):
                    output_first = group * geometry["fanout"]
                    for y0 in range(0, geometry["output_h"], tile[0]):
                        y1 = min(geometry["output_h"], y0 + tile[0])
                        for x0 in range(0, geometry["output_w"], tile[1]):
                            x1 = min(geometry["output_w"], x0 + tile[1])
                            record = MODULE.analyze_conv_tile(
                                module, binary, timestep, batch, group,
                                y0, y1, x0, x1, candidates, source_chunk_size=2,
                            )
                            self.assertEqual(
                                record["selected_source_count"],
                                record["positive_residual_source_count"]
                                + record["negative_residual_source_count"],
                            )
                            fallback_current = (
                                record["local_current_source_count"]
                                - record["valid_current_source_count"]
                            )
                            self.assertEqual(
                                record["valid_current_source_count"]
                                - record["shifted_previous_source_count"],
                                record["positive_residual_source_count"]
                                - fallback_current
                                - record["negative_residual_source_count"],
                            )
                            dy = record["selected_dy"]
                            dx = record["selected_dx"]
                            selected_nonzero += int(timestep > 0 and (dy != 0 or dx != 0))
                            selected_border += int(
                                timestep > 0 and (dy != 0 or dx != 0)
                                and record["border_fallback_rows"] > 0
                            )
                            output_y, output_x = MODULE._tile_positions(
                                geometry, y0, y1, x0, x1, binary.device
                            )
                            current_sources = MODULE._gather_source_chunk(
                                binary[timestep], geometry, batch, group,
                                output_y, output_x, (0, 0), 0, geometry["source_width"],
                            ).to(torch.int64)
                            if timestep > 0:
                                previous_sources = MODULE._gather_source_chunk(
                                    binary[timestep - 1], geometry, batch, group,
                                    output_y, output_x, (dy, dx), 0, geometry["source_width"],
                                ).to(torch.int64)
                            else:
                                previous_sources = torch.zeros_like(current_sources)
                            for row in range(int(output_y.numel())):
                                oy = int(output_y[row].item())
                                ox = int(output_x[row].item())
                                base_valid = (
                                    timestep > 0
                                    and 0 <= oy + dy < geometry["output_h"]
                                    and 0 <= ox + dx < geometry["output_w"]
                                )
                                for local_output in range(geometry["fanout"]):
                                    output_channel = output_first + local_output
                                    weights = module.weight[output_channel].reshape(-1).to(torch.int64)
                                    bias = (
                                        int(module.bias[output_channel].item())
                                        if module.bias is not None else 0
                                    )
                                    if base_valid:
                                        base = int(dense[
                                            timestep - 1, batch, output_channel,
                                            oy + dy, ox + dx,
                                        ].item())
                                        residual = int(torch.dot(
                                            current_sources[row] - previous_sources[row], weights
                                        ).item())
                                        reconstructed = base + residual
                                    else:
                                        reconstructed = bias + int(torch.dot(
                                            current_sources[row], weights
                                        ).item())
                                    self.assertEqual(
                                        reconstructed,
                                        int(dense[timestep, batch, output_channel, oy, ox].item()),
                                    )
        return selected_nonzero, selected_border

    def test_exact_dense_reconstruction_across_conv_geometries(self):
        cases = [
            dict(channels=1, out=2, kernel=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            dict(channels=2, out=3, kernel=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            dict(channels=4, out=4, kernel=3, stride=2, padding=2, dilation=2, groups=2, bias=True),
        ]
        saw_nonzero = 0
        saw_border = 0
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                module = torch.nn.Conv2d(
                    case["channels"], case["out"], case["kernel"],
                    stride=case["stride"], padding=case["padding"],
                    dilation=case["dilation"], groups=case["groups"], bias=case["bias"],
                )
                weights = torch.arange(module.weight.numel(), dtype=torch.int64)
                weights = (weights % 5) - 2
                module.weight.data.copy_(weights.reshape_as(module.weight).to(torch.float32))
                if module.bias is not None:
                    module.bias.data.copy_(
                        torch.arange(case["out"], dtype=torch.float32) - 1
                    )
                value = self._moving_binary(
                    10, 1, case["channels"], 7 + index, 8 + index,
                    step_y=0, step_x=case["stride"],
                )
                nonzero, border = self._assert_dense_exact(module, value, tile=(3, 3))
                saw_nonzero += nonzero
                saw_border += border
        self.assertGreater(saw_nonzero, 0)
        self.assertGreater(saw_border, 0)

    def test_atomic_histogram_only_publish_and_explicit_linear_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "census"
            writer = MODULE.StreamingShiftResidualCensusWriter(
                output, shift_radius=1, output_tile=(2, 3), source_chunk_size=2,
                expected_samples=1, expected_operator_calls=2,
                expected_exact_calls=1,
            )
            self._bind(writer)
            conv = torch.nn.Conv2d(2, 4, 3, padding=1, groups=2, bias=True)
            value = self._moving_binary(10, 1, 2, 5, 6)
            writer.record_operator(
                conv, value, "conv", 0, "sample", "sequence", 0,
            )
            linear = torch.nn.Linear(4, 3)
            writer.record_operator(
                linear, torch.zeros((10, 2, 4)), "linear", 0, "sample", "sequence", 0,
            )
            writer.close()
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"manifest.json", "shift_residual_histograms.jsonl", "call_reconciliation.csv"},
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], MODULE.PASS_STATUS)
            self.assertFalse(manifest["headline_admitted"])
            self.assertFalse(manifest["raw_activation_residual_or_tile_map_saved"])
            self.assertEqual(manifest["exact_calls"], 1)
            self.assertEqual(manifest["rejected_calls"], 1)
            self.assertEqual(manifest["coverage"]["observed_operator_calls"], 2)
            self.assertEqual(manifest["coverage"]["observed_sample_ids"], [0])
            self.assertTrue(manifest["algebraic_binary_source_conservation_exact"])
            self.assertFalse(manifest["fixed_point_bit_exact"])
            self.assertIn("previous_output_state_bits", manifest["peak_live_capacity_bits"])
            self.assertNotIn("previous_output_state_bits", manifest["workload_totals"])
            with (output / "call_reconciliation.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], MODULE.EXACT_STATUS)
            self.assertEqual(rows[1]["status"], "TEMPORAL_AXIS_UNQUALIFIED")
            self.assertGreater(int(rows[0]["search_bit_comparisons"]), 0)
            self.assertGreater(int(rows[0]["previous_output_state_bits"]), 0)
            self.assertGreater(int(rows[0]["output_scalar_writes"]), 0)
            self.assertEqual(rows[0]["headline_admitted"], "False")
            histogram = json.loads((output / "shift_residual_histograms.jsonl").read_text())
            self.assertFalse(histogram["raw_tensor_or_tile_map_saved"])

    def test_fail_closed_invalid_inputs_and_all_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.canonical_shift_candidates(-1)
        with self.assertRaises(ValueError):
            MODULE.canonical_shift_candidates(1, [(0, 0), (2, 0)])
        with self.assertRaises(ValueError):
            MODULE.canonical_shift_candidates(1, [(1, 0)])
        with self.assertRaises(ValueError):
            MODULE.canonical_shift_candidates(1, [(0, 0), (0, 0)])
        with self.assertRaises(ValueError):
            MODULE.canonical_shift_candidates(1, [(0, 0), (0.5, 0)])

        rejection_cases = []
        rejection_cases.append((
            torch.nn.Conv2d(1, 1, 3, padding=1, padding_mode="reflect"),
            torch.zeros((10, 1, 1, 5, 5)),
            "UNSUPPORTED_PADDING",
        ))
        nonbinary = torch.zeros((10, 1, 1, 5, 5))
        nonbinary[3, 0, 0, 2, 2] = 0.5
        rejection_cases.append((torch.nn.Conv2d(1, 1, 1), nonbinary, "NON_BINARY_BYPASS"))
        rejection_cases.append((
            torch.nn.Conv2d(1, 1, 1), torch.zeros((2, 1, 5, 5)),
            "TEMPORAL_AXIS_UNQUALIFIED",
        ))
        rejection_cases.append((
            torch.nn.Linear(4, 2), torch.zeros((10, 3, 4)),
            "TEMPORAL_AXIS_UNQUALIFIED",
        ))
        for index, (module, value, status) in enumerate(rejection_cases):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "all_rejected"
                writer = MODULE.StreamingShiftResidualCensusWriter(
                    output, expected_samples=1, expected_operator_calls=1,
                    expected_exact_calls=0,
                )
                self._bind(writer)
                writer.record_operator(module, value, "bad", 0, "s", "q", index)
                with self.assertRaisesRegex(ValueError, "all-rejected"):
                    writer.close()
                self.assertFalse(output.exists())
                marker = json.loads((writer.staging_dir / "INTERRUPTED.json").read_text())
                self.assertEqual(marker["status"], "INTERRUPTED_NOT_ADMITTED")
                self.assertFalse(marker["headline_admitted"])
                with writer.calls_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]["status"], status)
                writer.discard_staging_for_test_only()

    def test_negative_vertical_asymmetric_batch2_and_random_property_oracle(self):
        module = torch.nn.Conv2d(
            2, 3, (2, 3), stride=(1, 1), padding=(1, 2),
            dilation=(2, 1), bias=True,
        )
        weights = (torch.arange(module.weight.numel()) % 255) - 127
        module.weight.data.copy_(weights.reshape_as(module.weight).to(torch.float32))
        module.bias.data.copy_(torch.tensor([-1000.0, 0.0, 1000.0]))
        moving = self._moving_binary(
            10, 2, 2, 8, 9, step_y=-1, step_x=0
        )
        nonzero, _ = self._assert_dense_exact(module, moving, tile=(2, 3))
        self.assertGreater(nonzero, 0)

        for seed in range(3):
            torch.manual_seed(seed)
            random_value = torch.randint(
                0, 2, (10, 2, 2, 7, 8), dtype=torch.int64
            ).bool()
            random_module = torch.nn.Conv2d(
                2, 4, (3, 2), stride=(2, 1), padding=(2, 1),
                dilation=(2, 1), groups=2, bias=True,
            )
            random_module.weight.data.copy_(
                torch.randint(
                    -128, 128, random_module.weight.shape, dtype=torch.int64
                ).to(torch.float32)
            )
            random_module.bias.data.copy_(
                torch.randint(-10000, 10001, (4,), dtype=torch.int64).to(torch.float32)
            )
            self._assert_dense_exact(
                random_module, random_value, tile=(2, 3)
            )

    def test_context_coverage_and_concurrent_writer_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "locked"
            first = MODULE.StreamingShiftResidualCensusWriter(
                output, expected_samples=1, expected_operator_calls=1,
                expected_exact_calls=1,
            )
            with self.assertRaisesRegex(ValueError, "owns output lock"):
                MODULE.StreamingShiftResidualCensusWriter(
                    output, expected_samples=1, expected_operator_calls=1,
                    expected_exact_calls=1,
                )
            first.discard_staging_for_test_only()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad_sha"
            writer = MODULE.StreamingShiftResidualCensusWriter(
                output, expected_samples=1, expected_operator_calls=1,
                expected_exact_calls=1,
            )
            context = {
                "artifact_identity": {
                    "checkpoint_sha256": "g" * 64,
                    "config_sha256": "b" * 64,
                },
                "eval_protocol": {
                    "temporal_steps": 10,
                    "eval_batch_size": 1,
                    "requested_profile_samples": 1,
                    "expected_operator_calls": 1,
                    "expected_exact_calls": 1,
                    "temporal_axis_contract": "hook_input_dim0_is_T10_and_dim1_is_eval_batch",
                },
                "checkpoint_load_audit": {"missing": 0, "unexpected": 0},
                "source_sha256": {
                    "profiler": "c" * 64,
                    "shift_residual_census_writer": "d" * 64,
                },
            }
            with self.assertRaisesRegex(ValueError, "SHA"):
                writer.bind_run_context(context)
            writer.discard_staging_for_test_only()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "coverage"
            writer = MODULE.StreamingShiftResidualCensusWriter(
                output, expected_samples=2, expected_operator_calls=1,
                expected_exact_calls=1,
            )
            self._bind(writer)
            writer.record_operator(
                torch.nn.Conv2d(1, 1, 1),
                torch.zeros((10, 1, 1, 2, 2), dtype=torch.bool),
                "conv", 0, "sample0", "sequence", 0,
            )
            with self.assertRaisesRegex(ValueError, "sample coverage"):
                writer.close()
            self.assertFalse(output.exists())
            marker = json.loads(
                (writer.staging_dir / "INTERRUPTED.json").read_text()
            )
            self.assertEqual(marker["status"], "INTERRUPTED_NOT_ADMITTED")
            writer.discard_staging_for_test_only()

    def test_zero_cost_tie_uses_canonical_zero_shift_and_profiler_wiring_is_locked(self):
        module = torch.nn.Conv2d(1, 1, 3, padding=1, bias=True)
        value = torch.zeros((10, 1, 1, 4, 4), dtype=torch.bool)
        record = MODULE.analyze_conv_tile(
            module, value, 1, 0, 0, 0, 4, 0, 4,
            MODULE.canonical_shift_candidates(1), source_chunk_size=3,
        )
        self.assertEqual((record["selected_dy"], record["selected_dx"]), (0, 0))
        profiler = Path(__file__).with_name("profile_nts11_hardware_p0.py").read_text()
        for fragment in (
            "--shift-residual-census-dir",
            "shift_residual_writer.record_operator(",
            '"shift_residual_census_writer": file_sha256(',
            'shift_residual_writer.abort("profiler did not complete")',
            '"shift_residual_census_manifest"',
            "--shift-residual-expected-operator-calls",
            "--shift-residual-expected-exact-calls",
        ):
            self.assertIn(fragment, profiler)


if __name__ == "__main__":
    unittest.main()
