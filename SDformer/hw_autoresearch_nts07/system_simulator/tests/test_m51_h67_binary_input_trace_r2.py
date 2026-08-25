#!/usr/bin/env python3
"""Pure-CPU protocol and static runner tests for the M51-r2 P1 repairs."""

from __future__ import print_function

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
ENTRYPOINTS = (REPO /
               "neuron_experiments/H9_bipolar_self_attention/entrypoints")
RUNNER_PATH = ENTRYPOINTS / "capture_h67_full_network_binary_inputs_r2.py"
R1_TEST_PATH = (REPO / "hw_autoresearch_nts07/system_simulator/tests"
                / "test_m51_h67_binary_input_trace.py")
sys.path.insert(0, str(ENTRYPOINTS))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R2 = _load("m51_writer_r2", ENTRYPOINTS / "h67_binary_input_trace_r2.py")
R1_TEST = _load("m51_r1_test_helpers", R1_TEST_PATH)


def _memory(phase, allocated, reserved, max_allocated, max_reserved):
    return {
        "phase": phase,
        "cuda_available": True,
        "capture_device_type": "cuda",
        "memory_allocated_bytes": allocated,
        "memory_reserved_bytes": reserved,
        "max_memory_allocated_bytes": max_allocated,
        "max_memory_reserved_bytes": max_reserved,
    }


class FakeTensor(object):
    def __init__(self, contiguous):
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous


class M51R2WriterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="m51_r2_cpu_")
        self.root = Path(self.temporary.name)
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(
            json.dumps(R1_TEST._synthetic_plan(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def writer(self, output_name="trace", exact_sync=True):
        writer = R2.ExactBinaryInputTraceWriterR2(
            self.plan_path, self.root / output_name,
            expected_plan_sha256=None)
        writer.bind_module_identities(R1_TEST._identities())
        sync = ({
            "before_capture": 1,
            "per_sample_post_forward": 10,
            "final_pre_manifest": 1,
        } if exact_sync else {
            "before_capture": 1,
            "per_sample_post_forward": 9,
            "final_pre_manifest": 0,
        })
        writer.bind_run_context({
            "test": "pure_cpu_r2_protocol",
            "cuda_synchronization": sync,
        })
        return writer

    @staticmethod
    def payload(values):
        def produce(handle, digest):
            return R2.write_binary_value_chunks([values], handle, digest)
        return produce

    def complete_calls(self, writer):
        writer.begin_sample(0, "sample_0.npy", "sequence_a")
        writer.capture("synthetic.conv", "Conv2d", [1, 1, 2, 3],
                       [1, 1, 2, 3], self.payload([1, 0, 1, 0, 1, 0]))
        writer.capture("synthetic.linear", "Linear", [1, 4], [1, 2],
                       self.payload([0, 1, 0, 1]))
        writer.end_sample()

    def test_non_contiguous_input_is_rejected_before_copy(self):
        self.assertIs(R2.require_c_order_contiguous(FakeTensor(True)).contiguous,
                      True)
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            R2.require_c_order_contiguous(FakeTensor(False))
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            R2.require_c_order_contiguous(object())

    def test_abort_removes_all_partial_and_writes_only_failed_status(self):
        writer = self.writer()
        orphan_a = writer.output_root / "calls/a.partial"
        orphan_b = writer.output_root / "calls/b.partial"
        orphan_a.write_bytes(b"partial-a")
        orphan_b.write_bytes(b"partial-b")
        failed = writer.abort("injected synchronize failure",
                              failure_memory={"phase": "injected"})
        receipt = json.loads(failed.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"],
                         "FAIL_CLOSED_PARTIAL_CLEANED_NO_PASS_MANIFEST")
        self.assertEqual(receipt["partial_files_remaining"], 0)
        self.assertEqual(receipt["partial_files_removed"],
                         ["calls/a.partial", "calls/b.partial"])
        self.assertFalse((writer.output_root / "manifest.json").exists())
        self.assertEqual(set(path.name for path in writer.output_root.iterdir()),
                         {"calls", "FAILED.json"})

    def test_injected_post_forward_sync_failure_has_no_manifest(self):
        writer = self.writer()
        self.complete_calls(writer)

        def injected_synchronize():
            raise RuntimeError("deferred CUDA execution error")

        try:
            injected_synchronize()
        except BaseException as error:
            writer.abort("{}: {}".format(type(error).__name__, error))
        self.assertTrue((writer.output_root / "FAILED.json").is_file())
        self.assertFalse((writer.output_root / "manifest.json").exists())

    def test_pass_manifest_requires_memory_and_all_synchronization_counts(self):
        before = _memory("BEFORE_CAPTURE", 100, 200, 100, 200)
        after = _memory("AFTER_FINAL_SYNCHRONIZE", 110, 220, 180, 300)

        missing_memory = self.writer("missing_memory")
        self.complete_calls(missing_memory)
        with self.assertRaisesRegex(ValueError, "memory telemetry"):
            missing_memory.close()

        missing_sync = self.writer("missing_sync", exact_sync=False)
        self.complete_calls(missing_sync)
        missing_sync.record_capture_memory(before, after)
        with self.assertRaisesRegex(ValueError, "synchronization counts"):
            missing_sync.close()

        complete = self.writer("complete")
        self.complete_calls(complete)
        complete.record_capture_memory(before, after)
        manifest = json.loads(complete.close().read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_context"]["capture_memory"]["before"],
                         before)
        self.assertEqual(manifest["run_context"]["capture_memory"]["after"],
                         after)

    def test_invalid_or_decreasing_memory_telemetry_is_rejected(self):
        writer = self.writer()
        before = _memory("BEFORE_CAPTURE", 100, 200, 100, 200)
        decreasing = _memory(
            "AFTER_FINAL_SYNCHRONIZE", 50, 100, 90, 190)
        with self.assertRaisesRegex(ValueError, "decreased"):
            writer.record_capture_memory(before, decreasing)
        invalid = dict(before)
        invalid["cuda_available"] = False
        with self.assertRaisesRegex(ValueError, "snapshot identity"):
            writer.record_capture_memory(invalid, before)

    def test_runner_static_order_and_no_whole_call_contiguous(self):
        text = RUNNER_PATH.read_text(encoding="utf-8")
        stream_start = text.index("def stream_torch_binary_r2")
        stream_end = text.index("\ndef validate_frozen_protocol", stream_start)
        stream = text[stream_start:stream_end]
        self.assertIn("require_c_order_contiguous(tensor)", stream)
        self.assertIn("tensor.detach().view(-1)", stream)
        self.assertNotIn("tensor.detach().contiguous()", stream)
        self.assertNotIn("flat = tensor.detach().reshape", stream)

        model_call = text.index("                model(x)\n")
        sample_sync = text.index(
            "                torch.cuda.synchronize(device)\n", model_call)
        sample_end = text.index("                writer.end_sample()\n", sample_sync)
        final_sync = text.index(
            "        torch.cuda.synchronize(device)\n", sample_end)
        memory_after = text.index("        memory_after = cuda_memory_snapshot",
                                  final_sync)
        manifest_close = text.index("        manifest = writer.close()\n",
                                    memory_after)
        self.assertTrue(model_call < sample_sync < sample_end < final_sync <
                        memory_after < manifest_close)
        self.assertIn("except BaseException as error:", text)
        self.assertIn("writer.abort(", text)
        self.assertIn("torch.cuda.max_memory_allocated(device)", text)
        self.assertIn("torch.cuda.max_memory_reserved(device)", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
