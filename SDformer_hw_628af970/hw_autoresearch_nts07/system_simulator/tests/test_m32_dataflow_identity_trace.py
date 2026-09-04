#!/usr/bin/env python3

import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "m32_dataflow_identity_trace.py"
)
SPEC = importlib.util.spec_from_file_location("m32_identity", str(SCRIPT))
M32 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M32)
WRAPPER_SCRIPT = SCRIPT.with_name("run_m32_dataflow_identity_profile.py")
import sys
sys.path.insert(0, str(SCRIPT.parent))
WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "m32_identity_wrapper", str(WRAPPER_SCRIPT)
)
WRAPPER = importlib.util.module_from_spec(WRAPPER_SPEC)
WRAPPER_SPEC.loader.exec_module(WRAPPER)


class FakeDType(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class FakeNumpy(object):
    def __init__(self, raw):
        self.raw = raw

    def tobytes(self, order="C"):
        if order != "C":
            raise AssertionError("unexpected byte order")
        return self.raw


class FakeStorage(object):
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class FakeTensor(object):
    def __init__(
        self, raw, pointer, shape=(2,), dtype="torch.float32", numel_value=2
    ):
        self.raw = raw
        self.pointer = pointer
        self.shape = shape
        self.dtype = FakeDType(dtype)
        self.device = "cuda:0"
        self.numel_value = numel_value

    def detach(self):
        return self

    def contiguous(self):
        return self

    def reshape(self, *shape):
        return self

    def view(self, *args, **kwargs):
        return FakeTensor(
            self.raw,
            self.pointer,
            shape=(len(self.raw),),
            dtype="torch.uint8",
            numel_value=len(self.raw),
        )

    def cpu(self):
        return self

    def numpy(self):
        return FakeNumpy(self.raw)

    def numel(self):
        return self.numel_value

    def element_size(self):
        return 4

    def data_ptr(self):
        return self.pointer

    def untyped_storage(self):
        return FakeStorage(self.pointer)

    def storage_offset(self):
        return 0

    def stride(self):
        return (1,)


class FakeTorch(object):
    uint8 = FakeDType("torch.uint8")

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)


def write_candidate_report(path):
    candidates = []
    for index in range(10):
        candidates.append({
            "producer": "producer.{}".format(index),
            "name": "consumer.{}".format(index),
            "calls": 1,
            "output_elements_per_sample": 2,
            "semantic_admission": False,
        })
    path.write_text(json.dumps({
        "schema": "m32_threshold_carry_late_scale_audit_v2",
        "identity": {"samples": 1, "checkpoint_sha256": "a" * 64},
        "candidate_census": {"candidates": candidates},
        "semantic_admission": False,
        "headline_admitted": False,
    }), encoding="utf-8")


class M32DataflowIdentityTraceTest(unittest.TestCase):
    def make_writer(self, directory, name):
        report = Path(directory) / "candidate.json"
        if not report.exists():
            write_candidate_report(report)
        profile_output = Path(directory) / (name + "_profile")
        profile_output.mkdir()
        writer = M32.M32DataflowIdentityWriter(
            Path(directory) / name,
            report,
            expected_samples=1,
            run_identity={
                "test": True,
                "candidate_report_sha256": M32.sha256(report),
                "checkpoint_sha256": "a" * 64,
                "config_sha256": "c" * 64,
                "trace_contract_sha256": "b" * 64,
                "samples": 1,
                "profile_output_dir": str(profile_output),
            },
            torch_module=FakeTorch,
        )
        return writer

    def bind_postrun(self, writer, sample_key="sample.npy"):
        profile_output = Path(writer.run_identity["profile_output_dir"])
        profile_path = profile_output / "nts11_hardware_p0_profile.json"
        workload_path = profile_output / "sample_workload.csv"
        profile_path.write_text(json.dumps({
            "samples": 1,
            "artifact_identity": {
                "checkpoint_sha256": "a" * 64,
                "config_sha256": "c" * 64,
            },
            "checkpoint_load_audit": {
                "missing_count": 0,
                "unexpected_count": 0,
                "overlay_missing_count": 0,
                "overlay_unexpected_count": 0,
            },
            "summary": {"sample_records": [{"sample_key": sample_key}]},
        }), encoding="utf-8")
        with workload_path.open("w", encoding="utf-8", newline="") as handle:
            out = csv.DictWriter(
                handle, fieldnames=["sample_id", "sample_key", "sequence_key"]
            )
            out.writeheader()
            out.writerow({
                "sample_id": 0,
                "sample_key": sample_key,
                "sequence_key": "sequence",
            })
        writer.bind_postrun_evidence(profile_path, workload_path)

    def drive(self, writer, consumer_pointer_delta=0, skip_last=False):
        writer._root_prehook(None, None)
        for index, pair in enumerate(writer.pairs):
            if skip_last and index == len(writer.pairs) - 1:
                continue
            key = (pair["producer"], pair["consumer"])
            raw = bytes([index, 0, 0, 0, index, 0, 0, 0])
            producer = FakeTensor(raw, pointer=1000 + index)
            consumer = (
                producer
                if consumer_pointer_delta == 0
                else FakeTensor(raw, pointer=1000 + index + consumer_pointer_delta)
            )
            writer._producer_hook(key, pair)(None, None, producer)
            writer._consumer_hook(key, pair)(
                None, (consumer,), FakeTensor(raw, pointer=2000 + index)
            )
        if not skip_last:
            writer._root_hook(None, None, None)

    def test_exact_identity_is_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, "pass")
            self.drive(writer)
            self.bind_postrun(writer)
            manifest = writer.close()
            self.assertEqual(
                manifest["status"],
                "PASS_EXACT_PRODUCER_CONSUMER_TENSOR_IDENTITY",
            )
            self.assertEqual(manifest["records"], 10)
            self.assertEqual(manifest["identity_admitted_records"], 10)
            self.assertEqual(manifest["same_tensor_object_records"], 10)
            self.assertEqual(manifest["root_forwards"], 1)
            self.assertTrue(
                manifest["instrumentation"]["instrumentation_intrusive"]
            )
            self.assertEqual(
                manifest["postrun_evidence"]["sample_workload"]["records"], 1
            )

    def test_same_values_in_different_storage_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, "bad_storage")
            self.drive(writer, consumer_pointer_delta=100)
            self.bind_postrun(writer)
            with self.assertRaisesRegex(RuntimeError, "tensor_identity"):
                writer.close()
            manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "FAIL_CLOSED")
            self.assertIn("tensor_identity", manifest["failures"])

    def test_missing_call_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, "missing")
            self.drive(writer, skip_last=True)
            with self.assertRaisesRegex(RuntimeError, "call_population"):
                writer.close()

    def test_checkpoint_binding_drift_fails_before_attach(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "candidate.json"
            write_candidate_report(report)
            with self.assertRaisesRegex(ValueError, "binding drift"):
                M32.M32DataflowIdentityWriter(
                    Path(directory) / "bad_checkpoint",
                    report,
                    expected_samples=1,
                    run_identity={
                        "candidate_report_sha256": M32.sha256(report),
                        "checkpoint_sha256": "c" * 64,
                        "config_sha256": "c" * 64,
                        "trace_contract_sha256": "b" * 64,
                        "samples": 1,
                        "profile_output_dir": str(Path(directory) / "profile"),
                    },
                    torch_module=FakeTorch,
                )

    def test_postrun_sample_identity_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, "bad_sample")
            self.drive(writer)
            profile_output = Path(writer.run_identity["profile_output_dir"])
            profile_path = profile_output / "nts11_hardware_p0_profile.json"
            workload_path = profile_output / "sample_workload.csv"
            profile_path.write_text(json.dumps({
                "samples": 1,
                "artifact_identity": {
                    "checkpoint_sha256": "a" * 64,
                    "config_sha256": "c" * 64,
                },
                "checkpoint_load_audit": {
                    "missing_count": 0,
                    "unexpected_count": 0,
                    "overlay_missing_count": 0,
                    "overlay_unexpected_count": 0,
                },
                "summary": {"sample_records": [{"sample_key": "a.npy"}]},
            }), encoding="utf-8")
            workload_path.write_text(
                "sample_id,sample_key,sequence_key\n0,b.npy,sequence\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sample identity mismatch"):
                writer.bind_postrun_evidence(profile_path, workload_path)

    def test_duplicate_profile_option_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            WRAPPER._required_option(
                ["--config", "a.yml", "--config", "b.yml"], "--config"
            )

    def test_consumer_output_population_drift_fails_in_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, "bad_output")
            writer._root_prehook(None, None)
            pair = writer.pairs[0]
            key = (pair["producer"], pair["consumer"])
            tensor = FakeTensor(b"12345678", pointer=123)
            writer._producer_hook(key, pair)(None, None, tensor)
            with self.assertRaisesRegex(RuntimeError, "output population drift"):
                writer._consumer_hook(key, pair)(
                    None,
                    (tensor,),
                    FakeTensor(b"123456789012", pointer=456, numel_value=3),
                )


if __name__ == "__main__":
    unittest.main()
