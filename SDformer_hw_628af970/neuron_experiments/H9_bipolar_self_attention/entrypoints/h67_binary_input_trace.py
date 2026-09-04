"""Fail-closed streaming writer for exact binary operator-input planes.

The core writer has no torch/numpy dependency so its protocol can be tested on
CPU-only hosts.  The GPU entry point supplies a bounded, exact-binary torch
payload writer for each hook call.
"""

from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path


EXPECTED_TARGET_PLAN_SHA256 = (
    "bf0827d32896a871d9ea4c91afe49014bb5c236d619764b5c3f8a2804dc595e3")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def is_sha256(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def pack_binary_little(values):
    """Pack an iterable of exact integer/bool 0/1 values, little bit first."""
    output = bytearray()
    current = 0
    used = 0
    count = active = 0
    for value in values:
        require(value is False or value is True or
                (isinstance(value, int) and not isinstance(value, bool) and
                 value in (0, 1)),
                "non-binary value rejected: {!r}".format(value))
        bit = int(value)
        current |= bit << used
        used += 1
        count += 1
        active += bit
        if used == 8:
            output.append(current)
            current = 0
            used = 0
    if used:
        output.append(current)
    return bytes(output), count, active


def unpack_binary_little(payload, elements):
    require(elements >= 0 and len(payload) == (elements + 7) // 8,
            "payload/element length mismatch")
    values = []
    for index in range(elements):
        values.append((payload[index // 8] >> (index % 8)) & 1)
    if elements % 8:
        used_mask = (1 << (elements % 8)) - 1
        require((payload[-1] & ~used_mask) == 0,
                "nonzero high tail padding")
    return values


def write_binary_value_chunks(chunks, handle, digest):
    """Small/reference streaming packer used by CPU tests."""
    current = 0
    used = count = active = packed_bytes = 0
    for chunk in chunks:
        for value in chunk:
            require(value is False or value is True or
                    (isinstance(value, int) and not isinstance(value, bool) and
                     value in (0, 1)),
                    "non-binary value rejected: {!r}".format(value))
            bit = int(value)
            current |= bit << used
            used += 1
            count += 1
            active += bit
            if used == 8:
                payload = bytes((current,))
                handle.write(payload)
                digest.update(payload)
                packed_bytes += 1
                current = used = 0
    if used:
        payload = bytes((current,))
        handle.write(payload)
        digest.update(payload)
        packed_bytes += 1
    return {"elements": count, "active": active,
            "packed_bytes": packed_bytes, "tail_used_bits": used or 8}


def validate_parameter_identity(identity, expected_operator,
                                expected_weight_elements):
    require(identity["operator"] == expected_operator,
            "module operator identity mismatch")
    weight = identity["weight"]
    require(product(weight["shape"]) == expected_weight_elements and
            isinstance(weight["dtype"], str) and weight["dtype"] and
            weight["content_bytes"] > 0 and
            is_sha256(weight["content_sha256"]) and
            weight["byte_order"] == "little" and
            weight["layout"] == "C_ORDER_CONTIGUOUS",
            "module weight identity mismatch")
    bias = identity["bias"]
    if bias is not None:
        require(product(bias["shape"]) > 0 and
                isinstance(bias["dtype"], str) and bias["dtype"] and
                bias["content_bytes"] > 0 and
                is_sha256(bias["content_sha256"]) and
                bias["byte_order"] == "little" and
                bias["layout"] == "C_ORDER_CONTIGUOUS",
                "module bias identity mismatch")


class ExactBinaryInputTraceWriter(object):
    """One-call-at-a-time raw bit-plane writer with strict plan matching."""

    def __init__(self, target_plan_path, output_root,
                 expected_plan_sha256=EXPECTED_TARGET_PLAN_SHA256):
        self.target_plan_path = Path(target_plan_path).resolve()
        self.output_root = Path(output_root).resolve()
        require(self.target_plan_path.is_file(), "missing target plan")
        actual_plan_sha = sha256_path(self.target_plan_path)
        if expected_plan_sha256 is not None:
            require(actual_plan_sha == expected_plan_sha256,
                    "target plan SHA mismatch")
        self.plan = strict_json(self.target_plan_path)
        require(self.plan["schema"] ==
                "m51_h67_ep35_binary_input_trace_target_plan_v1",
                "target plan schema mismatch")
        require(not self.output_root.exists(),
                "refusing existing M51 output root")
        self.output_root.mkdir(parents=True)
        (self.output_root / "calls").mkdir()
        self.plan_sha256 = actual_plan_sha
        self.modules = self.plan["modules"]
        self.module_by_name = dict((row["name"], row) for row in self.modules)
        require(len(self.module_by_name) == len(self.modules),
                "duplicate module in target plan")
        self.module_identities = None
        self.run_context = None
        self.current_sample = None
        self.current_order = 0
        self.completed_samples = []
        self.records = []
        self.closed = False
        self.aborted = False

    def bind_module_identities(self, identities):
        require(self.module_identities is None and not self.records and
                self.current_sample is None, "module identities already bound")
        require(set(identities) == set(self.module_by_name),
                "module identity population mismatch")
        normalized = {}
        for name, target in self.module_by_name.items():
            identity = identities[name]
            validate_parameter_identity(
                identity, target["operator"], target["expected_weight_elements"])
            normalized[name] = identity
        self.module_identities = normalized

    def bind_run_context(self, context):
        require(self.run_context is None and not self.records and
                self.current_sample is None, "run context already bound")
        self.run_context = dict(context)

    def begin_sample(self, sample_id, sample_key, sequence_key):
        require(not self.closed and not self.aborted,
                "writer is closed/aborted")
        require(self.module_identities is not None and self.run_context is not None,
                "writer identities/context not bound")
        require(self.current_sample is None, "previous sample not ended")
        require(sample_id == len(self.completed_samples),
                "sample order mismatch")
        expected = self.plan["samples"][sample_id]
        require(expected == {"sample_id": sample_id,
                             "sample_key": sample_key,
                             "sequence_key": sequence_key},
                "sample identity mismatch")
        self.current_sample = sample_id
        self.current_order = 0

    def _expected_call(self, name):
        require(self.current_sample is not None, "capture outside sample")
        require(name in self.module_by_name, "unexpected target module")
        require(self.current_order < len(self.modules), "too many target calls")
        expected_module = self.modules[self.current_order]
        require(expected_module["name"] == name,
                "target call order mismatch: expected {} observed {}".format(
                    expected_module["name"], name))
        return expected_module, expected_module["calls"][self.current_sample]

    def capture(self, name, operator, input_shape, output_shape, payload_writer):
        require(not self.closed and not self.aborted, "writer is closed/aborted")
        module, expected = self._expected_call(name)
        require(module["operator"] == operator,
                "target operator mismatch")
        require(list(input_shape) == expected["input_shape"] and
                list(output_shape) == expected["output_shape"],
                "target input/output shape mismatch")
        final_path = self.output_root / expected["relative_output_path"]
        partial_path = final_path.with_name(final_path.name + ".partial")
        require(not final_path.exists() and not partial_path.exists(),
                "refusing existing call output")
        digest = hashlib.sha256()
        try:
            with partial_path.open("xb") as handle:
                stats = payload_writer(handle, digest)
            require(stats["elements"] == expected["input_elements"] and
                    stats["packed_bytes"] == expected["packed_bytes"],
                    "captured element/byte population mismatch")
            require(partial_path.stat().st_size == expected["packed_bytes"],
                    "captured file size mismatch")
            # Hard-link publication gives an atomic no-replace guarantee.  A
            # concurrent pre-existing final path therefore fails closed.
            os.link(str(partial_path), str(final_path))
            partial_path.unlink()
        except Exception:
            if partial_path.exists():
                partial_path.unlink()
            raise
        record = {
            "sample_id": self.current_sample,
            "sample_key": expected["sample_key"],
            "sequence_key": expected["sequence_key"],
            "module_index": module["module_index"],
            "name": name,
            "operator": operator,
            "frozen_execution_call_index":
                expected["frozen_execution_call_index"],
            "target_order_index": expected["target_order_index"],
            "input_shape": expected["input_shape"],
            "output_shape": expected["output_shape"],
            "input_elements": stats["elements"],
            "active_elements": stats["active"],
            "packed_bytes": stats["packed_bytes"],
            "tail_used_bits": stats["tail_used_bits"],
            "relative_path": expected["relative_output_path"],
            "file_sha256": digest.hexdigest(),
        }
        self.records.append(record)
        self.current_order += 1
        return record

    def end_sample(self):
        require(self.current_sample is not None, "no active sample")
        require(self.current_order == len(self.modules),
                "missing target module call(s)")
        self.completed_samples.append(self.current_sample)
        self.current_sample = None
        self.current_order = 0

    def close(self):
        require(not self.closed and not self.aborted, "second close/aborted writer")
        require(self.current_sample is None, "sample still active at close")
        population = self.plan["population"]
        require(self.completed_samples == list(range(population["samples"])) and
                len(self.records) == population["hook_calls"] and
                sum(row["input_elements"] for row in self.records) ==
                population["input_elements"] and
                sum(row["packed_bytes"] for row in self.records) ==
                population["packed_bytes"],
                "final M51 population mismatch")
        manifest = {
            "schema": "m51_h67_ep35_binary_input_trace_manifest_v1",
            "status": "PASS_EXACT_BINARY_INPUT_TRACE_NO_OUTPUT_OR_PERFORMANCE_CLAIM",
            "identity": {
                "target_plan_path": str(self.target_plan_path),
                "target_plan_sha256": self.plan_sha256,
                "contract_sha256": self.plan["identity"]["contract_sha256"],
            },
            "run_context": self.run_context,
            "module_identities": self.module_identities,
            "population": {
                "samples": len(self.completed_samples),
                "modules": len(self.modules),
                "hook_calls": len(self.records),
                "input_elements": sum(row["input_elements"]
                                      for row in self.records),
                "active_elements": sum(row["active_elements"]
                                       for row in self.records),
                "packed_bytes": sum(row["packed_bytes"]
                                    for row in self.records),
            },
            "packing": self.plan["packing"],
            "records": self.records,
            "claim_boundary": (
                "raw exact-binary input activation planes only; no output, "
                "cycle, speedup, RTL, PPA, energy or system claim"),
        }
        manifest_path = self.output_root / "manifest.json"
        require(not manifest_path.exists(), "refusing existing manifest")
        temporary = manifest_path.with_name(
            manifest_path.name + ".tmp.{}".format(os.getpid()))
        with temporary.open("x") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.link(str(temporary), str(manifest_path))
        temporary.unlink()
        self.closed = True
        return manifest_path

    def abort(self, reason):
        require(not self.closed and not self.aborted, "second abort/closed writer")
        receipt = self.output_root / "FAILED.json"
        require(not receipt.exists(), "failure receipt already exists")
        receipt.write_text(json.dumps({
            "schema": "m51_binary_input_trace_failure_v1",
            "status": "FAIL_CLOSED_PARTIAL_PAYLOAD_NOT_ADMITTED",
            "reason": str(reason),
            "completed_records": len(self.records),
            "manifest_written": False,
        }, indent=2, sort_keys=True) + "\n")
        self.aborted = True
        return receipt
