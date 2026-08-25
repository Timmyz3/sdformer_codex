#!/usr/bin/env python3
"""Pure-CPU tests for the M51 exact-binary streaming trace protocol."""

from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
WRITER_PATH = (
    REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints"
    / "h67_binary_input_trace.py")
SPEC = importlib.util.spec_from_file_location("m51_writer", str(WRITER_PATH))
WRITER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRITER)


def _call(sample_id, module_index, order, input_shape, output_shape):
    elements = WRITER.product(input_shape)
    return {
        "sample_id": sample_id,
        "sample_key": "sample_{}.npy".format(sample_id),
        "sequence_key": "sequence_a",
        "frozen_execution_call_index": 2 + module_index,
        "dual_line_operator_call_index": sample_id,
        "dual_line_temporal_steps": list(range(10)),
        "target_order_index": order,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "input_elements": elements,
        "output_elements": WRITER.product(output_shape),
        "packed_bytes": (elements + 7) // 8,
        "relative_output_path": (
            "calls/s{:02d}_m{:02d}.activation.le.bitpack".format(
                sample_id, module_index)),
    }


def _synthetic_plan():
    calls = [
        _call(0, 0, 0, [1, 1, 2, 3], [1, 1, 2, 3]),
        _call(0, 1, 1, [1, 4], [1, 2]),
    ]
    modules = [
        {
            "module_index": 0,
            "name": "synthetic.conv",
            "operator": "Conv2d",
            "expected_hook_calls": 1,
            "expected_weight_elements": 6,
            "runtime_weight_and_bias_content_sha256_required": True,
            "calls": [calls[0]],
        },
        {
            "module_index": 1,
            "name": "synthetic.linear",
            "operator": "Linear",
            "expected_hook_calls": 1,
            "expected_weight_elements": 8,
            "runtime_weight_and_bias_content_sha256_required": True,
            "calls": [calls[1]],
        },
    ]
    return {
        "schema": "m51_h67_ep35_binary_input_trace_target_plan_v1",
        "identity": {"contract_sha256": "1" * 64},
        "samples": [{
            "sample_id": 0,
            "sample_key": "sample_0.npy",
            "sequence_key": "sequence_a",
        }],
        "modules": modules,
        "population": {
            "samples": 1,
            "modules": 2,
            "hook_calls": 2,
            "dual_line_rows": 20,
            "input_elements": 10,
            "packed_bytes": 2,
        },
        "packing": {
            "layout": "C_ORDER_FLAT",
            "bit_order": "LITTLE_WITHIN_BYTE",
            "tail_padding_high_bits_zero": True,
            "file_granularity": "ONE_RAW_FILE_PER_HOOK_CALL",
            "float_payload_retained": False,
            "delta_payload_retained": False,
        },
    }


def _identities():
    return {
        "synthetic.conv": {
            "operator": "Conv2d",
            "weight": {
                "shape": [1, 1, 2, 3],
                "dtype": "synthetic.int32",
                "content_bytes": 24,
                "content_sha256": "2" * 64,
                "byte_order": "little",
                "layout": "C_ORDER_CONTIGUOUS",
            },
            "bias": {
                "shape": [1],
                "dtype": "synthetic.int32",
                "content_bytes": 4,
                "content_sha256": "3" * 64,
                "byte_order": "little",
                "layout": "C_ORDER_CONTIGUOUS",
            },
        },
        "synthetic.linear": {
            "operator": "Linear",
            "weight": {
                "shape": [2, 4],
                "dtype": "synthetic.int32",
                "content_bytes": 32,
                "content_sha256": "4" * 64,
                "byte_order": "little",
                "layout": "C_ORDER_CONTIGUOUS",
            },
            "bias": None,
        },
    }


class M51WriterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="m51_cpu_test_")
        self.root = Path(self.temporary.name)
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(
            json.dumps(_synthetic_plan(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def writer(self, name="trace"):
        writer = WRITER.ExactBinaryInputTraceWriter(
            self.plan_path, self.root / name, expected_plan_sha256=None)
        writer.bind_module_identities(_identities())
        writer.bind_run_context({"test": "pure_cpu_synthetic"})
        return writer

    @staticmethod
    def payload(values, groups=None):
        if groups is None:
            groups = [values]

        def produce(handle, digest):
            return WRITER.write_binary_value_chunks(groups, handle, digest)
        return produce

    def test_pack_unpack_exact_little_bit_and_tail(self):
        values = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
        payload, count, active = WRITER.pack_binary_little(values)
        self.assertEqual(payload, bytes((0x4d, 0x13)))
        self.assertEqual((count, active), (13, 7))
        self.assertEqual(WRITER.unpack_binary_little(payload, count), values)
        with self.assertRaisesRegex(ValueError, "tail padding"):
            WRITER.unpack_binary_little(bytes((0x4d, 0xf3)), count)

    def test_conv_and_linear_happy_path_manifest_and_hashes(self):
        writer = self.writer()
        writer.begin_sample(0, "sample_0.npy", "sequence_a")
        conv = [1, 0, 1, 1, 0, 0]
        linear = [0, 1, 1, 0]
        writer.capture("synthetic.conv", "Conv2d", [1, 1, 2, 3],
                       [1, 1, 2, 3], self.payload(conv, [conv[:2], conv[2:]]))
        writer.capture("synthetic.linear", "Linear", [1, 4], [1, 2],
                       self.payload(linear, [linear[:1], linear[1:3], linear[3:]]))
        writer.end_sample()
        manifest_path = writer.close()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["population"], {
            "active_elements": 5,
            "hook_calls": 2,
            "input_elements": 10,
            "modules": 2,
            "packed_bytes": 2,
            "samples": 1,
        })
        self.assertEqual(set(manifest["module_identities"]),
                         {"synthetic.conv", "synthetic.linear"})
        expected_values = [conv, linear]
        for record, values in zip(manifest["records"], expected_values):
            path = self.root / "trace" / record["relative_path"]
            payload = path.read_bytes()
            self.assertEqual(record["file_sha256"],
                             hashlib.sha256(payload).hexdigest())
            self.assertEqual(
                WRITER.unpack_binary_little(payload, record["input_elements"]),
                values)

    def test_nonbinary_is_rejected_and_partial_is_removed(self):
        writer = self.writer()
        writer.begin_sample(0, "sample_0.npy", "sequence_a")
        with self.assertRaisesRegex(ValueError, "non-binary"):
            writer.capture("synthetic.conv", "Conv2d", [1, 1, 2, 3],
                           [1, 1, 2, 3], self.payload([0, 1, 2, 0, 1, 0]))
        call_dir = self.root / "trace" / "calls"
        self.assertEqual(list(call_dir.iterdir()), [])
        self.assertFalse((self.root / "trace" / "manifest.json").exists())

    def test_shape_call_and_sample_order_are_fail_closed(self):
        writer = self.writer()
        with self.assertRaisesRegex(ValueError, "sample order"):
            writer.begin_sample(1, "sample_1.npy", "sequence_a")
        writer.begin_sample(0, "sample_0.npy", "sequence_a")
        with self.assertRaisesRegex(ValueError, "call order"):
            writer.capture("synthetic.linear", "Linear", [1, 4], [1, 2],
                           self.payload([0, 1, 1, 0]))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            writer.capture("synthetic.conv", "Conv2d", [1, 1, 3, 2],
                           [1, 1, 2, 3], self.payload([0] * 6))
        writer.capture("synthetic.conv", "Conv2d", [1, 1, 2, 3],
                       [1, 1, 2, 3], self.payload([0] * 6))
        with self.assertRaisesRegex(ValueError, "call order"):
            writer.capture("synthetic.conv", "Conv2d", [1, 1, 2, 3],
                           [1, 1, 2, 3], self.payload([0] * 6))
        with self.assertRaisesRegex(ValueError, "missing target"):
            writer.end_sample()

    def test_overwrite_and_second_close_are_rejected(self):
        output = self.root / "occupied"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "existing M51 output root"):
            WRITER.ExactBinaryInputTraceWriter(
                self.plan_path, output, expected_plan_sha256=None)

        writer = self.writer("complete")
        writer.begin_sample(0, "sample_0.npy", "sequence_a")
        writer.capture("synthetic.conv", "Conv2d", [1, 1, 2, 3],
                       [1, 1, 2, 3], self.payload([0] * 6))
        writer.capture("synthetic.linear", "Linear", [1, 4], [1, 2],
                       self.payload([0] * 4))
        writer.end_sample()
        writer.close()
        with self.assertRaisesRegex(ValueError, "second close"):
            writer.close()

    def test_call_file_collision_is_rejected(self):
        writer = self.writer()
        writer.begin_sample(0, "sample_0.npy", "sequence_a")
        collision = (self.root / "trace" / "calls"
                     / "s00_m00.activation.le.bitpack")
        collision.write_bytes(b"do-not-overwrite")
        with self.assertRaisesRegex(ValueError, "existing call output"):
            writer.capture("synthetic.conv", "Conv2d", [1, 1, 2, 3],
                           [1, 1, 2, 3], self.payload([0] * 6))
        self.assertEqual(collision.read_bytes(), b"do-not-overwrite")


if __name__ == "__main__":
    unittest.main(verbosity=2)
