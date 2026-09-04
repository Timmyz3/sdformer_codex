#!/usr/bin/env python3
"""Fail-closed producer-to-consumer tensor identity tracing for M32."""

import hashlib
import csv
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_tensors(value, torch_module):
    if torch_module.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            for tensor in _iter_tensors(value[key], torch_module):
                yield tensor
    elif isinstance(value, (list, tuple)):
        for item in value:
            for tensor in _iter_tensors(item, torch_module):
                yield tensor


def _storage_pointer(tensor):
    if hasattr(tensor, "untyped_storage"):
        return int(tensor.untyped_storage().data_ptr())
    return int(tensor.storage().data_ptr())


def _logical_raw_bytes(tensor, torch_module):
    """Return bytes in logical C order, excluding unrelated backing storage."""
    value = tensor.detach().contiguous().reshape(-1)
    try:
        value = value.view(torch_module.uint8)
    except TypeError:
        value = value.view(dtype=torch_module.uint8)
    return value.cpu().numpy().tobytes(order="C")


def tensor_identity(tensor, torch_module):
    raw = _logical_raw_bytes(tensor, torch_module)
    return {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": [int(value) for value in tensor.shape],
        "numel": int(tensor.numel()),
        "element_size_bytes": int(tensor.element_size()),
        "logical_nbytes": len(raw),
        "logical_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_data_pointer": int(tensor.data_ptr()),
        "storage_pointer": _storage_pointer(tensor),
        "storage_offset": int(tensor.storage_offset()),
        "stride": [int(value) for value in tensor.stride()],
    }


class M32DataflowIdentityWriter(object):
    """Trace exact outputs and downstream inputs for frozen M32 pairs."""

    def __init__(
        self,
        output_dir,
        candidate_report,
        expected_samples,
        run_identity,
        torch_module=None,
    ):
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists():
            raise ValueError(
                "refusing to overwrite M32 dataflow output: {}".format(
                    self.output_dir
                )
            )
        self.output_dir.mkdir(parents=True)
        self.rows_path = self.output_dir / "m32_dataflow_identity.jsonl"
        self.manifest_path = self.output_dir / "m32_dataflow_identity_manifest.json"
        self.candidate_report_path = Path(candidate_report).resolve()
        self.expected_samples = int(expected_samples)
        if self.expected_samples <= 0:
            raise ValueError("M32 dataflow trace requires positive samples")
        self.run_identity = dict(run_identity)
        self.torch = torch_module
        self.handles = []
        self.rows = []
        self.pending = {}
        self.producer_calls = {}
        self.consumer_calls = {}
        self.active_sample_id = None
        self.root_forwards = 0
        self.postrun_evidence = None
        self.closed = False

        report = json.loads(
            self.candidate_report_path.read_text(encoding="utf-8")
        )
        if (
            report.get("schema") != "m32_threshold_carry_late_scale_audit_v2"
            or report.get("semantic_admission") is not False
            or report.get("headline_admitted") is not False
            or int(report["identity"]["samples"]) != self.expected_samples
        ):
            raise ValueError("unexpected or admitted M32 candidate report")
        self.candidate_report_sha256 = sha256(self.candidate_report_path)
        if (
            self.run_identity.get("candidate_report_sha256")
            != self.candidate_report_sha256
            or self.run_identity.get("checkpoint_sha256")
            != report["identity"]["checkpoint_sha256"]
            or int(self.run_identity.get("samples", -1))
            != self.expected_samples
            or not self.run_identity.get("trace_contract_sha256")
        ):
            raise ValueError("M32 candidate/run identity binding drift")
        self.pairs = []
        seen_producers = set()
        seen_consumers = set()
        for row in report["candidate_census"]["candidates"]:
            producer = str(row["producer"])
            consumer = str(row["name"])
            if producer in seen_producers or consumer in seen_consumers:
                raise ValueError("M32 trace requires one-to-one module pairs")
            if row.get("semantic_admission") is not False:
                raise ValueError("M32 candidate unexpectedly semantically admitted")
            seen_producers.add(producer)
            seen_consumers.add(consumer)
            self.pairs.append({
                "producer": producer,
                "consumer": consumer,
                "expected_calls": int(row["calls"]),
                "expected_consumer_output_elements_per_sample": int(
                    row["output_elements_per_sample"]
                ),
            })
        if len(self.pairs) != 10:
            raise ValueError("M32 trace requires exactly ten candidate pairs")
        for pair in self.pairs:
            if pair["expected_calls"] != self.expected_samples:
                raise ValueError("M32 candidate call population drift")
            key = (pair["producer"], pair["consumer"])
            self.pending[key] = []
            self.producer_calls[key] = 0
            self.consumer_calls[key] = 0

    def _require_one_tensor(self, value, role, module_name):
        tensors = list(_iter_tensors(value, self.torch))
        if len(tensors) != 1:
            raise RuntimeError(
                "M32 {} {} yielded {} tensors, expected one".format(
                    role, module_name, len(tensors)
                )
            )
        return tensors[0]

    def _producer_hook(self, key, pair):
        def hook(_module, _inputs, output):
            if self.active_sample_id is None:
                raise RuntimeError("M32 producer observed outside root forward")
            call_index = self.producer_calls[key]
            if call_index != self.active_sample_id:
                raise RuntimeError("M32 producer is not one call per root forward")
            if call_index >= self.expected_samples or self.pending[key]:
                raise RuntimeError("M32 producer call population exceeded contract")
            tensor = self._require_one_tensor(output, "producer", pair["producer"])
            identity = tensor_identity(tensor, self.torch)
            self.pending[key].append({
                "call_index": call_index,
                "object_id": id(tensor),
                "tensor": tensor,
                "identity": identity,
            })
            self.producer_calls[key] += 1
        return hook

    def _consumer_hook(self, key, pair):
        def hook(_module, inputs, output):
            if self.active_sample_id is None:
                raise RuntimeError("M32 consumer observed outside root forward")
            call_index = self.consumer_calls[key]
            if call_index != self.active_sample_id:
                raise RuntimeError("M32 consumer is not one call per root forward")
            if call_index >= self.expected_samples:
                raise RuntimeError("M32 consumer call population exceeded contract")
            if not self.pending[key]:
                raise RuntimeError("M32 consumer observed without pending producer")
            producer = self.pending[key].pop(0)
            if producer["call_index"] != call_index:
                raise RuntimeError("M32 producer/consumer call order drift")
            tensor = self._require_one_tensor(inputs, "consumer", pair["consumer"])
            output_tensor = self._require_one_tensor(
                output, "consumer output", pair["consumer"]
            )
            consumer_output_numel = int(output_tensor.numel())
            if (
                consumer_output_numel
                != pair["expected_consumer_output_elements_per_sample"]
            ):
                raise RuntimeError("M32 consumer output population drift")
            consumer = tensor_identity(tensor, self.torch)
            source = producer["identity"]
            row = {
                "sample_id": self.active_sample_id,
                "producer": pair["producer"],
                "consumer": pair["consumer"],
                "producer_call_index": producer["call_index"],
                "consumer_call_index": call_index,
                "dtype": source["dtype"],
                "shape": source["shape"],
                "numel": source["numel"],
                "producer_raw_value_sha256": source["logical_raw_sha256"],
                "consumer_raw_value_sha256": consumer["logical_raw_sha256"],
                "consumer_output_numel": consumer_output_numel,
                "expected_consumer_output_numel": pair[
                    "expected_consumer_output_elements_per_sample"
                ],
                "same_tensor_object": producer["tensor"] is tensor,
                "same_storage_pointer": (
                    source["storage_pointer"] == consumer["storage_pointer"]
                ),
                "same_data_pointer": (
                    source["tensor_data_pointer"] == consumer["tensor_data_pointer"]
                ),
                "same_storage_offset": (
                    source["storage_offset"] == consumer["storage_offset"]
                ),
                "same_stride": source["stride"] == consumer["stride"],
                "same_dtype": source["dtype"] == consumer["dtype"],
                "same_device": source["device"] == consumer["device"],
                "same_shape": source["shape"] == consumer["shape"],
                "same_numel": source["numel"] == consumer["numel"],
                "same_logical_nbytes": (
                    source["logical_nbytes"] == consumer["logical_nbytes"]
                ),
                "same_value_digest": (
                    source["logical_raw_sha256"]
                    == consumer["logical_raw_sha256"]
                ),
            }
            row["identity_admitted"] = all([
                row["same_tensor_object"],
                row["same_storage_pointer"],
                row["same_data_pointer"],
                row["same_storage_offset"],
                row["same_stride"],
                row["same_dtype"],
                row["same_device"],
                row["same_shape"],
                row["same_numel"],
                row["same_logical_nbytes"],
                row["same_value_digest"],
            ])
            self.rows.append(row)
            self.consumer_calls[key] += 1
        return hook

    def _root_prehook(self, _module, _inputs):
        if self.active_sample_id is not None:
            raise RuntimeError("M32 nested root forward is not supported")
        if self.root_forwards >= self.expected_samples:
            raise RuntimeError("M32 root forward population exceeded contract")
        if any(self.pending[key] for key in self.pending):
            raise RuntimeError("M32 pending producer crossed a sample boundary")
        self.active_sample_id = self.root_forwards

    def _root_hook(self, _module, _inputs, _output):
        if self.active_sample_id != self.root_forwards:
            raise RuntimeError("M32 root forward boundary state drift")
        expected_count = self.root_forwards + 1
        for pair in self.pairs:
            key = (pair["producer"], pair["consumer"])
            if (
                self.producer_calls[key] != expected_count
                or self.consumer_calls[key] != expected_count
                or self.pending[key]
            ):
                raise RuntimeError(
                    "M32 pair is not exactly once within root forward: {} -> {}"
                    .format(pair["producer"], pair["consumer"])
                )
        self.root_forwards += 1
        self.active_sample_id = None

    def attach(self, model):
        if self.torch is None:
            import torch
            self.torch = torch
        modules = dict(model.named_modules())
        self.handles.append(model.register_forward_pre_hook(self._root_prehook))
        self.handles.append(model.register_forward_hook(self._root_hook))
        for pair in self.pairs:
            producer = pair["producer"]
            consumer = pair["consumer"]
            if producer not in modules or consumer not in modules:
                raise ValueError(
                    "M32 module pair missing: {} -> {}".format(
                        producer, consumer
                    )
                )
            key = (producer, consumer)
            self.handles.append(
                modules[producer].register_forward_hook(
                    self._producer_hook(key, pair)
                )
            )
            self.handles.append(
                modules[consumer].register_forward_hook(
                    self._consumer_hook(key, pair)
                )
            )

    def _remove_handles(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def bind_postrun_evidence(self, profile_json_path, sample_workload_path):
        profile_json_path = Path(profile_json_path).resolve()
        sample_workload_path = Path(sample_workload_path).resolve()
        if self.postrun_evidence is not None:
            raise RuntimeError("M32 postrun evidence already bound")
        if self.active_sample_id is not None:
            raise RuntimeError("M32 cannot bind evidence during a root forward")
        for path in (profile_json_path, sample_workload_path):
            if not path.is_file():
                raise ValueError("missing M32 postrun evidence: {}".format(path))
        expected_parent = Path(self.run_identity["profile_output_dir"]).resolve()
        if (
            profile_json_path.parent != expected_parent
            or sample_workload_path.parent != expected_parent
        ):
            raise ValueError("M32 postrun evidence is outside profile output")

        profile = json.loads(profile_json_path.read_text(encoding="utf-8"))
        artifact = profile.get("artifact_identity") or {}
        load_audit = profile.get("checkpoint_load_audit") or {}
        if (
            int(profile.get("samples", -1)) != self.expected_samples
            or artifact.get("checkpoint_sha256")
            != self.run_identity.get("checkpoint_sha256")
            or artifact.get("config_sha256")
            != self.run_identity.get("config_sha256")
            or int(load_audit.get("missing_count", -1)) != 0
            or int(load_audit.get("unexpected_count", -1)) != 0
            or int(load_audit.get("overlay_missing_count", -1)) != 0
            or int(load_audit.get("overlay_unexpected_count", -1)) != 0
        ):
            raise ValueError("M32 postrun profile identity/load audit drift")

        with sample_workload_path.open("r", encoding="utf-8", newline="") as handle:
            workload_rows = list(csv.DictReader(handle))
        if (
            len(workload_rows) != self.expected_samples
            or [int(row["sample_id"]) for row in workload_rows]
            != list(range(self.expected_samples))
            or any(not row.get("sample_key") or not row.get("sequence_key")
                   for row in workload_rows)
            or len(set(row["sample_key"] for row in workload_rows))
            != self.expected_samples
        ):
            raise ValueError("M32 sample workload identity drift")
        summary_rows = (profile.get("summary") or {}).get("sample_records") or []
        if (
            len(summary_rows) != self.expected_samples
            or [str(row.get("sample_key")) for row in summary_rows]
            != [row["sample_key"] for row in workload_rows]
        ):
            raise ValueError("M32 profile/workload sample identity mismatch")
        sample_digest = hashlib.sha256()
        for row in workload_rows:
            sample_digest.update(
                ("{}\t{}\t{}\n".format(
                    row["sample_id"], row["sample_key"], row["sequence_key"]
                )).encode("utf-8")
            )
        self.postrun_evidence = {
            "profile_json": {
                "path": str(profile_json_path),
                "sha256": sha256(profile_json_path),
            },
            "sample_workload": {
                "path": str(sample_workload_path),
                "sha256": sha256(sample_workload_path),
                "records": len(workload_rows),
                "ordered_sample_identity_sha256": sample_digest.hexdigest(),
                "sample_keys": [row["sample_key"] for row in workload_rows],
                "sequence_keys": [row["sequence_key"] for row in workload_rows],
            },
            "checkpoint_load_audit": load_audit,
            "status": "PASS_FROZEN_POSTRUN_PROFILE_AND_SAMPLE_IDENTITY",
        }

    def close(self):
        if self.closed:
            raise RuntimeError("M32 dataflow writer already closed")
        self._remove_handles()
        failures = []
        if self.postrun_evidence is None:
            failures.append("postrun_evidence_missing")
        if self.root_forwards != self.expected_samples:
            failures.append("root_forward_population")
        if self.active_sample_id is not None:
            failures.append("unterminated_root_forward")
        for pair in self.pairs:
            key = (pair["producer"], pair["consumer"])
            if self.producer_calls[key] != self.expected_samples:
                failures.append("producer_call_population")
            if self.consumer_calls[key] != self.expected_samples:
                failures.append("consumer_call_population")
            if self.pending[key]:
                failures.append("unconsumed_producer_output")
        if len(self.rows) != len(self.pairs) * self.expected_samples:
            failures.append("row_population")
        if not all(row["identity_admitted"] for row in self.rows):
            failures.append("tensor_identity")
        with self.rows_path.open("x", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        manifest = {
            "schema": "m32_dataflow_identity_manifest_v1",
            "status": (
                "PASS_EXACT_PRODUCER_CONSUMER_TENSOR_IDENTITY"
                if not failures else "FAIL_CLOSED"
            ),
            "candidate_report": {
                "path": str(self.candidate_report_path),
                "sha256": self.candidate_report_sha256,
            },
            "run_identity": self.run_identity,
            "postrun_evidence": self.postrun_evidence,
            "expected_samples": self.expected_samples,
            "candidate_pairs": len(self.pairs),
            "records": len(self.rows),
            "identity_admitted_records": sum(
                bool(row["identity_admitted"]) for row in self.rows
            ),
            "same_tensor_object_records": sum(
                bool(row["same_tensor_object"]) for row in self.rows
            ),
            "root_forwards": self.root_forwards,
            "instrumentation": {
                "instrumentation_intrusive": True,
                "gpu_to_cpu_digest_copy": True,
                "cuda_synchronization_expected": True,
                "performance_use_forbidden": [
                    "profile wall time", "CUDA allocator behavior", "cycle count",
                    "FPS", "power", "energy", "PPA",
                ],
            },
            "rows": {
                "path": str(self.rows_path),
                "sha256": sha256(self.rows_path),
            },
            "failures": sorted(set(failures)),
            "semantic_scope": (
                "proves only that each frozen producer tensor is the exact object, "
                "logical value, and storage view read by its paired Linear/Conv2d "
                "consumer once per outer model forward"
            ),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.closed = True
        if failures:
            raise RuntimeError(
                "M32 dataflow identity failed closed: {}".format(
                    ",".join(sorted(set(failures)))
                )
            )
        return manifest

    def abort(self, reason):
        self._remove_handles()
        if not self.manifest_path.exists():
            self.manifest_path.write_text(
                json.dumps({
                    "schema": "m32_dataflow_identity_manifest_v1",
                    "status": "ABORTED_NOT_ADMITTED",
                    "reason": str(reason),
                    "records_captured": len(self.rows),
                    "candidate_report_sha256": self.candidate_report_sha256,
                    "run_identity": self.run_identity,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self.closed = True
