#!/usr/bin/env python3
"""Capture the H67 layer-static decoder payload under M686-r6 semantics.

M686-r6 preserves the exact M511/M649 model, data, checkpoint, call order and
native cuDNN-TF32 evaluation semantics.  M681 proved that disabling cuDNN TF32
changes 264,066 of 4,608,000 sample-0/d0 bits, while deterministic execution
with cuDNN TF32 retained reproduces the frozen M511 bitpack byte-for-byte.
D0/D2/D3 are
captured as exact binary inputs.  D1 has a stricter dual result: only an exact
runtime ``{0, scalar_theta}`` S10 gate may publish a mask and FP32
theta-folded weight; otherwise D1 remains the opaque FP32 fallback described
by M659 and no D1 payload is published.  No thresholding, rounding or coercion
is allowed.  This producer emits payload and identity evidence only; it emits
no cycle, speedup, RTL, EDA, energy or PPA claim.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import struct
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import torch
import torch.nn.functional as torch_functional
from spikingjelly.activation_based import functional


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
M511_PRODUCER_SHA256 = (
    "e16a454d532acd15d96527cfddf43ebf9f95338a34ce9aeedbb10032cb26230a")
M511_CONTRACT_SHA256 = (
    "e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e")
M649_RESULT_SHA256 = (
    "fb8cd63f3af2becafc8cd6f72aded3a2b82d11ecab30e2214f725fb09176fdf3")
M658_REVIEW_SHA256 = (
    "3edece8a6a98364eae2b1e0c4722690291ae48bd09a19770d63c4533c89903d0")
M662_REVIEW_SHA256 = (
    "58b105f4b1e696be232e7dfe5a85df93696a58922e141f87ee582a9467de3722")
M666_REVIEW_OUTER_SEAL_FILE_SHA256 = (
    "455447d9693f57fc5b1ddf5610009bdfbcb2af8b57f6473e3f546e3865cff82a")
M666_REVIEW_SHA256 = (
    "30337db281b7bf8e591ad29a47fe9eaaa10743618272c2e314b6c2403a20c716")
BINARY_MODULE_INDICES = (0, 2, 3)
FALLBACK_MODULE_INDEX = 1


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(token):
        raise RuntimeError("M660 non-standard JSON token: " + token)

    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "M660 duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def reject_symlink_chain(path, allow_missing_leaf=False):
    absolute = Path(os.path.abspath(str(path)))
    cursor = Path(absolute.parts[0])
    for index, part in enumerate(absolute.parts[1:], 1):
        cursor = cursor / part
        leaf = index == len(absolute.parts) - 1
        if os.path.lexists(str(cursor)):
            require(not cursor.is_symlink(),
                    "M660 rejects symlink path component: " + str(cursor))
        else:
            require(leaf and allow_missing_leaf,
                    "M660 missing path component: " + str(cursor))


def checked_path(path, allow_missing_leaf=False, label="path"):
    raw = Path(path)
    require(".." not in raw.parts,
            "M660 rejects parent traversal in {}: {}".format(label, raw))
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    reject_symlink_chain(absolute, allow_missing_leaf=allow_missing_leaf)
    return absolute.resolve(strict=not allow_missing_leaf)


def checked_contract_path(relative, allow_missing_leaf=False, label="input"):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "M660 unsafe contract {} path: {}".format(label, relative))
    return checked_path(ROOT / relative, allow_missing_leaf, label)


def checked_path_match(raw, expected, allow_missing_leaf=False, label="path"):
    observed = checked_path(raw, allow_missing_leaf, "runtime " + label)
    wanted = checked_path(expected, allow_missing_leaf, "expected " + label)
    require(observed == wanted, "M660 {} path drift".format(label))
    return observed


def _seal_members(directory):
    excluded = {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    return [
        path.relative_to(directory)
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file() and path.relative_to(directory).as_posix()
        not in excluded
    ]


def write_double_seal(directory):
    directory = Path(directory)
    members = _seal_members(directory)
    seal = directory / "SHA256SUMS"
    seal.write_text("".join(
        "{}  {}\n".format(sha256(directory / member), member.as_posix())
        for member in members), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text("{}  SHA256SUMS\n".format(sha256(seal)),
                     encoding="utf-8")


def verify_double_seal(directory):
    directory = checked_path(directory, label="sealed directory")
    seal = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(seal.is_file() and not seal.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "M660 missing predecessor double seal")
    expected, name = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(name == "SHA256SUMS" and sha256(seal) == expected,
            "M660 outer seal mismatch")
    sealed = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(name not in sealed and name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256") and
            ".." not in Path(name).parts and not Path(name).is_absolute(),
            "M660 unsafe/duplicate sealed member: " + name)
        member = directory / name
        require(member.is_file() and not member.is_symlink() and
                sha256(member) == expected,
                "M660 sealed member mismatch: " + name)
        sealed.add(name)
    actual = {path.relative_to(directory).as_posix()
              for path in directory.rglob("*") if path.is_file() and
              path.relative_to(directory).as_posix() not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == sealed, "M660 sealed population mismatch")


def verify_contract_inputs(contract, launcher_path):
    require(set(contract["inputs"]) == set(contract["required_input_names"]),
            "M660 contract input population drift")
    identities = {}
    for name, entry in contract["inputs"].items():
        path = checked_contract_path(entry["path"], label="input " + name)
        require(path.is_file() and not path.is_symlink(),
                "M660 missing/symlink input: " + name)
        observed = sha256(path)
        require(observed == entry["sha256"],
                "M660 input identity drift: " + name)
        identities[name] = {"path": str(path), "bytes": path.stat().st_size,
                            "sha256": observed}
    require(checked_contract_path(
        contract["inputs"]["launcher"]["path"], label="input launcher") ==
            checked_path(launcher_path, label="runtime launcher"),
            "M660 launcher path drift")
    require(identities["m511_producer"]["sha256"] == M511_PRODUCER_SHA256 and
            identities["m511_contract"]["sha256"] == M511_CONTRACT_SHA256 and
            identities["m649_result"]["sha256"] == M649_RESULT_SHA256 and
            identities["m658_review"]["sha256"] == M658_REVIEW_SHA256 and
            identities["m662_review"]["sha256"] == M662_REVIEW_SHA256 and
            identities["m666_review"]["sha256"] == M666_REVIEW_SHA256 and
            identities["m666_outer_seal"]["sha256"] ==
            M666_REVIEW_OUTER_SEAL_FILE_SHA256 and
            identities["docs359"]["sha256"] == DOCS359_SHA256,
            "M660 critical frozen identity drift")
    return identities


def load_frozen_m511(path):
    entrypoint = str(path.parent)
    if entrypoint not in sys.path:
        sys.path.insert(0, entrypoint)
    spec = importlib.util.spec_from_file_location("m660_frozen_m511", str(path))
    require(spec is not None and spec.loader is not None,
            "M660 cannot construct M511 import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(sha256(path) == M511_PRODUCER_SHA256,
            "M660 M511 producer drift across import")
    return module


def verify_predecessor_evidence(contract):
    m649_dir = checked_contract_path(contract["predecessors"]["m649_directory"],
                                     label="M649 canonical")
    m658_dir = checked_contract_path(contract["predecessors"]["m658_directory"],
                                     label="M658 review")
    m659_dir = checked_contract_path(contract["predecessors"]["m659_directory"],
                                     label="M659 plan")
    m662_dir = checked_contract_path(contract["predecessors"]["m662_directory"],
                                     label="M662 review")
    m666_dir = checked_contract_path(contract["predecessors"]["m666_directory"],
                                     label="M666 review")
    for directory in (m649_dir, m658_dir, m659_dir, m662_dir, m666_dir):
        verify_double_seal(directory)
    m649 = strict_json(m649_dir / "m649_typed_numeric_audit.json")
    m658 = strict_json(m658_dir / "review.json")
    m659 = strict_json(m659_dir / "plan.json")
    m662 = strict_json(m662_dir / "review.json")
    m666 = strict_json(m666_dir / "review.json")
    require(m649["status"] ==
            "PASS_NUMERIC_AUDIT__NO_GO_EXACT_TYPED_SPLIT" and
            m649["population"] == {"samples": 10, "modules": 4,
                                    "records": 40},
            "M660 M649 semantic identity drift")
    require(m658["status"] ==
            "PASS_NUMERIC_RESULT__NO_GO_GLOBAL_SPLIT_CONFIRMED" and
            m658["severity"] == {"p0": 0, "p1": 0, "p2": 1} and
            m658["score"] == 99 and
            m658["decision"]["global_exact_typed_split"] == "NO_GO_CORRECT" and
            not m658["decision"]["speedup_or_rtl_authorized"],
            "M660 M658 verdict drift")
    require(m659["claim_boundary"]["plan_only"] and
            not m659["claim_boundary"]["gpu_run"] and
            m659["minimum_successor_payload"]["binary_bitpack_records"] == 30 and
            m659["minimum_successor_payload"]["D1_raw_payload_saved"] is False,
            "M660 M659 boundary drift")
    require(m662["status"] ==
            "CONDITIONAL_GO_D1_THRESHOLD_FOLD_MEASUREMENT__NO_PERFORMANCE_OR_RTL_ADMISSION" and
            m662["scope"] == {
                "primary": "H67 ep35 decoder-1 scalar-threshold folding into ConvTranspose2d weights",
                "conditional_generalization": "scale-coded binary producer groups across Conv2d, Linear and ConvTranspose2d consumers",
                "gpu_run": False, "eda_run": False, "simulator_run": False} and
            m662["d1_static_chain"]["threshold"] == {
                "shape": [], "numel": 1, "dtype": "torch.float32",
                "value": 0.9999954104423523,
                "little_endian_hex": "b3ff7f3f"} and
            not m662["separate_admission_bits"]["exact_scb_representation"] and
            not m662["separate_admission_bits"]
            ["exact_fp32_folded_weight_execution"] and
            not m662["claim_boundary"]["speedup"],
            "M660 M662 conditional measurement boundary drift")
    require(m666["status"] ==
            "NO_GO_P1__DO_NOT_EXECUTE_OR_CONSUME_ONE_SHOT" and
            m666["severity"] == {"p0": 0, "p1": 4, "p2": 2} and
            not m666["go"] and
            [row["id"] for row in m666["p1_findings"]] ==
            ["P1-1", "P1-2", "P1-3", "P1-4"] and
            not m666["claim_boundary"]["payload_admitted"],
            "M660-r2 M666 repair-root boundary drift")

    expected = {}
    binary_totals = {index: {"elements": 0, "ones": 0} for index in
                     BINARY_MODULE_INDICES}
    d1_totals = {"elements": 0, "zero_count": 0, "one_count": 0,
                 "nonbinary_finite_count": 0, "nonfinite_count": 0}
    records = m649["records"]
    require(len(records) == 40, "M660 M649 record population drift")
    for order, record in enumerate(records):
        sample_id = order // 4
        module_index = order % 4
        require(record["sample_id"] == sample_id and
                record["module_index"] == module_index,
                "M660 M649 record order drift")
        full = record["input_numeric"]["full_tensor"]
        key = (sample_id, module_index)
        expected[key] = {
            "shape": record["input_numeric"]["shape"],
            "dtype": record["input_numeric"]["dtype"],
            "stride": record["input_numeric"]["stride"],
            "elements": int(full["elements"]),
            "zero_count": int(full["zero_count"]),
            "one_count": int(full["one_count"]),
            "nonbinary_finite_count": int(full["nonbinary_finite_count"]),
            "nonfinite_count": int(full["nonfinite_count"]),
        }
        if module_index in BINARY_MODULE_INDICES:
            require(full["all_exact_binary"] and full["all_finite"] and
                    full["nonbinary_finite_count"] == 0,
                    "M660 M649 binary-module fact drift")
            binary_totals[module_index]["elements"] += int(full["elements"])
            binary_totals[module_index]["ones"] += int(full["one_count"])
        else:
            require(not full["all_exact_binary"] and full["all_finite"],
                    "M660 M649 fallback-module fact drift")
            for name in d1_totals:
                d1_totals[name] += int(full[name])
    for label, index in (("d0", 0), ("d2", 2), ("d3", 3)):
        fact = m658["module_facts"][label]
        require(binary_totals[index] == {
            "elements": int(fact["elements"]), "ones": int(fact["ones"])},
            "M660 M658 binary aggregate drift: " + label)
    d1_fact = m658["module_facts"]["d1"]
    require(d1_totals == {
        "elements": int(d1_fact["elements"]),
        "zero_count": int(d1_fact["exact_binary"]),
        "one_count": int(d1_fact["exact_ones"]),
        "nonbinary_finite_count": int(d1_fact["finite_nonbinary"]),
        "nonfinite_count": int(d1_fact["nonfinite"]),
    }, "M660 M658 D1 aggregate drift")
    return {
        "m649_result_sha256": sha256(
            m649_dir / "m649_typed_numeric_audit.json"),
        "m649_outer_seal_file_sha256": sha256(
            m649_dir / "SHA256SUMS.seal.sha256"),
        "m658_review_sha256": sha256(m658_dir / "review.json"),
        "m658_outer_seal_file_sha256": sha256(
            m658_dir / "SHA256SUMS.seal.sha256"),
        "m659_plan_sha256": sha256(m659_dir / "plan.json"),
        "m659_outer_seal_file_sha256": sha256(
            m659_dir / "SHA256SUMS.seal.sha256"),
        "m662_review_sha256": sha256(m662_dir / "review.json"),
        "m662_outer_seal_file_sha256": sha256(
            m662_dir / "SHA256SUMS.seal.sha256"),
        "m666_review_sha256": sha256(m666_dir / "review.json"),
        "m666_outer_seal_file_sha256": sha256(
            m666_dir / "SHA256SUMS.seal.sha256"),
        "expected_records": expected,
    }


def _canonical_cpu_chunk(chunk):
    return chunk.detach().to(device="cpu").contiguous().numpy()


def stream_raw_content_hash(tensor, chunk_elements):
    require(torch.is_tensor(tensor) and chunk_elements > 0,
            "M660 invalid raw hash input")
    flat = tensor.detach().contiguous().view(-1)
    digest = hashlib.sha256()
    total_bytes = 0
    for begin in range(0, int(flat.numel()), chunk_elements):
        values = _canonical_cpu_chunk(
            flat[begin:min(int(flat.numel()), begin + chunk_elements)])
        payload = values.tobytes(order="C")
        digest.update(payload)
        total_bytes += len(payload)
    return {
        "elements": int(flat.numel()),
        "content_bytes": total_bytes,
        "content_sha256": digest.hexdigest(),
        "canonical_layout": "C_ORDER_LOGICAL_CONTIGUOUS",
        "byte_order": sys.byteorder,
    }


def stream_binary_input(tensor, chunk_elements, path):
    require(torch.is_tensor(tensor) and tensor.dtype == torch.float32,
            "M660 binary input must be a float32 tensor")
    require(chunk_elements > 0 and chunk_elements % 8 == 0,
            "M660 binary chunk size must be a positive multiple of eight")
    flat = tensor.detach().contiguous().view(-1)
    elements = int(flat.numel())
    require(elements % 8 == 0,
            "M660 binary payload requires byte-aligned call population")
    packed_digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    ones = packed_bytes = raw_bytes = 0
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    require(not path.exists() and not partial.exists(),
            "M660 refuses an existing binary payload")
    try:
        with partial.open("xb") as handle:
            for begin in range(0, elements, chunk_elements):
                chunk = flat[begin:min(elements, begin + chunk_elements)]
                exact = torch.logical_or(chunk == 0, chunk == 1)
                require(bool(torch.all(exact).item()),
                        "M660 binary-module input is not exact {0,1}")
                raw = _canonical_cpu_chunk(chunk)
                raw_payload = raw.tobytes(order="C")
                raw_digest.update(raw_payload)
                raw_bytes += len(raw_payload)
                values = raw.astype(np.uint8, copy=False)
                ones += int(values.sum(dtype=np.uint64))
                payload = np.packbits(values, bitorder="little").tobytes(
                    order="C")
                require(len(payload) == int(chunk.numel()) // 8,
                        "M660 packed chunk byte mismatch")
                handle.write(payload)
                packed_digest.update(payload)
                packed_bytes += len(payload)
        require(packed_bytes == elements // 8 and
                partial.stat().st_size == packed_bytes,
                "M660 final packed byte mismatch")
        os.link(str(partial), str(path))
        partial.unlink()
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return {
        "elements": elements,
        "zero_count": elements - ones,
        "one_count": ones,
        "exact_binary_count": elements,
        "nonbinary_finite_count": 0,
        "nonfinite_count": 0,
        "packed_bytes": packed_bytes,
        "packed_sha256": packed_digest.hexdigest(),
        "raw_content_bytes": raw_bytes,
        "raw_content_sha256": raw_digest.hexdigest(),
        "bit_order": "little",
        "packing_order": "C_ORDER_FLAT",
    }


def summarize_d1_fallback(tensor, chunk_elements):
    """Hash/count D1 without accepting a path or emitting any payload."""
    require(torch.is_tensor(tensor) and tensor.dtype == torch.float32,
            "M660 D1 fallback must be float32")
    flat = tensor.detach().contiguous().view(-1)
    digest = hashlib.sha256()
    counts = {"zero_count": 0, "one_count": 0,
              "nonbinary_finite_count": 0, "nonfinite_count": 0}
    content_bytes = 0
    for begin in range(0, int(flat.numel()), chunk_elements):
        values = _canonical_cpu_chunk(
            flat[begin:min(int(flat.numel()), begin + chunk_elements)])
        payload = values.tobytes(order="C")
        digest.update(payload)
        content_bytes += len(payload)
        finite = np.isfinite(values)
        zero = values == 0
        one = values == 1
        counts["zero_count"] += int(np.count_nonzero(zero))
        counts["one_count"] += int(np.count_nonzero(one))
        counts["nonbinary_finite_count"] += int(np.count_nonzero(
            finite & ~zero & ~one))
        counts["nonfinite_count"] += int(np.count_nonzero(~finite))
    return {
        "elements": int(flat.numel()),
        "content_bytes": content_bytes,
        "content_sha256": digest.hexdigest(),
        "canonical_layout": "C_ORDER_LOGICAL_CONTIGUOUS",
        "byte_order": sys.byteorder,
        **counts,
        "route": "COMMON_FP32_DENSE_FALLBACK",
        "raw_payload_saved": False,
        "thresholded": False,
        "coerced_to_binary": False,
    }


def stream_theta_binary_candidate(tensor, theta, chunk_elements, path):
    """Audit D1 against exact {0, theta}; publish a candidate only on pass."""
    require(torch.is_tensor(tensor) and tensor.dtype == torch.float32 and
            torch.is_tensor(theta) and theta.numel() == 1 and
            theta.dtype == torch.float32,
            "M660 invalid D1 theta-binary input")
    require(chunk_elements > 0 and chunk_elements % 8 == 0,
            "M660 invalid D1 theta-binary chunk size")
    flat = tensor.detach().contiguous().view(-1)
    elements = int(flat.numel())
    require(elements % 8 == 0, "M660 D1 bit population is not byte aligned")
    digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    counts = {"zero_count": 0, "theta_count": 0,
              "other_finite_count": 0, "nonfinite_count": 0}
    packed_bytes = raw_bytes = 0
    gate_pass = True
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    require(not path.exists() and not partial.exists(),
            "M660 refuses an existing D1 candidate payload")
    try:
        with partial.open("xb") as handle:
            for begin in range(0, elements, chunk_elements):
                chunk = flat[begin:min(elements, begin + chunk_elements)]
                finite = torch.isfinite(chunk)
                zero = chunk == 0
                active = chunk == theta
                valid = torch.logical_or(zero, active)
                counts["zero_count"] += int(torch.count_nonzero(zero).item())
                counts["theta_count"] += int(torch.count_nonzero(active).item())
                counts["other_finite_count"] += int(torch.count_nonzero(
                    torch.logical_and(finite, torch.logical_not(valid))).item())
                counts["nonfinite_count"] += int(torch.count_nonzero(
                    torch.logical_not(finite)).item())
                raw = _canonical_cpu_chunk(chunk)
                raw_payload = raw.tobytes(order="C")
                raw_digest.update(raw_payload)
                raw_bytes += len(raw_payload)
                if not bool(torch.all(valid).item()):
                    gate_pass = False
                if gate_pass:
                    mask = _canonical_cpu_chunk(active).astype(
                        np.uint8, copy=False)
                    payload = np.packbits(mask, bitorder="little").tobytes(
                        order="C")
                    require(len(payload) == int(chunk.numel()) // 8,
                            "M660 D1 candidate packed chunk mismatch")
                    handle.write(payload)
                    digest.update(payload)
                    packed_bytes += len(payload)
        if gate_pass:
            require(packed_bytes == elements // 8 and
                    partial.stat().st_size == packed_bytes,
                    "M660 D1 candidate packed population mismatch")
            os.link(str(partial), str(path))
        partial.unlink()
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return {
        "elements": elements, **counts,
        "theta_gate_pass": bool(gate_pass),
        "packed_bytes": packed_bytes if gate_pass else 0,
        "packed_sha256": digest.hexdigest() if gate_pass else None,
        "raw_content_bytes": raw_bytes,
        "raw_content_sha256": raw_digest.hexdigest(),
        "bit_order": "little" if gate_pass else None,
        "packing_order": "C_ORDER_FLAT" if gate_pass else None,
        "raw_payload_saved": False,
        "thresholded": False,
        "rounded": False,
        "comparison": "BIT_EXACT_X_EQ_0_OR_X_EQ_RUNTIME_SCALAR_THETA",
    }


def _ordered_float32_bits(bits):
    """Map raw FP32 uint32 patterns to monotonic integers for ULP distance."""
    value = bits.to(dtype=torch.int64) & 0xFFFFFFFF
    negative = (value & 0x80000000) != 0
    return torch.where(negative, 0xFFFFFFFF - value,
                       value + 0x80000000)


def compare_tensors_streaming(original, reference, chunk_elements):
    require(torch.is_tensor(original) and torch.is_tensor(reference) and
            original.shape == reference.shape and
            original.dtype == reference.dtype,
            "M660 folded miter tensor identity drift")
    left = original.detach().contiguous().view(-1)
    right = reference.detach().contiguous().view(-1)
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    mismatches = 0
    signed_zero_mismatches = 0
    max_abs_error = 0.0
    max_ulp_error = 0
    for begin in range(0, int(left.numel()), chunk_elements):
        end = min(int(left.numel()), begin + chunk_elements)
        left_chunk = left[begin:end]
        right_chunk = right[begin:end]
        left_bits = left_chunk.view(torch.int32)
        right_bits = right_chunk.view(torch.int32)
        bit_mismatch = left_bits != right_bits
        mismatches += int(torch.count_nonzero(bit_mismatch).item())
        signed_zero_mismatches += int(torch.count_nonzero(
            (left_chunk == 0) & (right_chunk == 0) & bit_mismatch).item())
        finite_pair = torch.isfinite(left_chunk) & torch.isfinite(right_chunk)
        if bool(torch.any(finite_pair).item()):
            max_abs_error = max(max_abs_error, float(torch.max(torch.abs(
                left_chunk[finite_pair] - right_chunk[finite_pair])).item()))
        ulp = torch.abs(_ordered_float32_bits(left_bits) -
                        _ordered_float32_bits(right_bits))
        if int(ulp.numel()):
            max_ulp_error = max(max_ulp_error, int(torch.max(ulp).item()))
        left_digest.update(_canonical_cpu_chunk(left_chunk).tobytes(order="C"))
        right_digest.update(_canonical_cpu_chunk(right_chunk).tobytes(order="C"))
    left_sha = left_digest.hexdigest()
    right_sha = right_digest.hexdigest()
    exact = mismatches == 0 and left_sha == right_sha
    return {
        "elements": int(left.numel()),
        "original_output_sha256": left_sha,
        "folded_reference_output_sha256": right_sha,
        "bit_exact_mismatch_count": mismatches,
        "signed_zero_bit_mismatch_count": signed_zero_mismatches,
        "max_abs_error": max_abs_error,
        "max_ulp_error": max_ulp_error,
        "hashes_equal": left_sha == right_sha,
        "bit_exact": exact,
        "comparison": "FLOAT32_RAW_UINT32_AND_CANONICAL_BYTES",
        "reference_definition":
            "conv_transpose2d(float32(mask), float32(theta*W), original ordered parameters)",
    }


def folded_miter_admitted(d1_records, theta_gate_pass):
    """One typed deployment gate shared by main and independent unit attacks."""
    return bool(theta_gate_pass and d1_records and all(
        row["folded_weight_miter"] is not None and
        row["folded_weight_miter"]["bit_exact"] and
        row["folded_weight_miter"]["bit_exact_mismatch_count"] == 0 and
        row["folded_weight_miter"]["signed_zero_bit_mismatch_count"] == 0 and
        row["folded_weight_miter"]["max_ulp_error"] == 0 and
        row["folded_weight_miter"]["hashes_equal"] and
        row["folded_weight_miter"]["original_output_sha256"] ==
        row["folded_weight_miter"]["folded_reference_output_sha256"]
        for row in d1_records))


def save_weight_payloads(model, modules, directory):
    directory = Path(directory)
    directory.mkdir()
    named = dict(model.named_modules())
    result = {}
    for expected in modules:
        index = int(expected["module_index"])
        weight = named[expected["name"]].weight.detach().to(
            device="cpu").contiguous()
        require(weight.dtype == torch.float32 and
                [int(item) for item in weight.shape] == expected["weight_shape"],
                "M660 weight type/shape drift")
        relative = "d{}.weight.f32le".format(index)
        path = directory / relative
        payload = weight.numpy().tobytes(order="C")
        with path.open("xb") as handle:
            handle.write(payload)
        result[str(index)] = {
            "module_index": index, "name": expected["name"],
            "relative_path": "weights/" + relative,
            "shape": expected["weight_shape"], "dtype": "torch.float32",
            "layout": "C_ORDER_CONTIGUOUS", "byte_order": "little",
            "content_bytes": len(payload), "content_sha256": sha256(path),
        }
    return result


def decoder_threshold_identity(model, expected):
    named = dict(model.named_modules())
    suffix = ".deconv.0"
    require(expected["name"].endswith(suffix),
            "M660 decoder module naming drift")
    owner_name = expected["name"][:-len(suffix)]
    require(owner_name in named and hasattr(named[owner_name], "sn"),
            "M660 decoder owner/neuron missing")
    wrapper = named[owner_name].sn
    leaf_name = owner_name + ".sn.spiking_neuron"
    require(hasattr(wrapper, "spiking_neuron") and leaf_name in named and
            named[leaf_name] is wrapper.spiking_neuron,
            "M660 D1 owner.sn must be the frozen Spiking_neuron wrapper with a named spiking_neuron leaf")
    neuron = wrapper.spiking_neuron
    require(hasattr(neuron, "thresh") and
            torch.is_tensor(neuron.thresh) and neuron.thresh.numel() == 1 and
            neuron.thresh.dtype == torch.float32 and
            getattr(neuron, "threshold_mode", None) == "official_atlif" and
            getattr(neuron, "output_mode", None) == "binary",
            "M660 D1 is not the frozen scalar official-ATLIF binary neuron")
    require(bool(torch.isfinite(neuron.thresh.detach()).item()) and
            bool((neuron.thresh.detach() > 0).item()),
            "M660 D1 scalar threshold must be finite and positive")
    frozen = neuron.thresh.detach().clone(memory_format=torch.contiguous_format)
    cpu = frozen.to(device="cpu").contiguous()
    payload = cpu.numpy().tobytes(order="C")
    require(len(payload) == 4, "M660 D1 threshold is not scalar FP32")
    return frozen, {
        "owner_name": owner_name,
        "wrapper_name": owner_name + ".sn",
        "wrapper_class": wrapper.__class__.__name__,
        "wrapper_class_module": wrapper.__class__.__module__,
        "leaf_name": leaf_name,
        "leaf_class": neuron.__class__.__name__,
        "leaf_class_module": neuron.__class__.__module__,
        "parameter_name": leaf_name + ".thresh",
        "parameter_device": str(neuron.thresh.device),
        "parameter_requires_grad": bool(neuron.thresh.requires_grad),
        "shape": [], "dtype": "torch.float32",
        "value": float(cpu.item()),
        "ieee754_le_hex": payload.hex(),
        "ieee754_uint32": int(struct.unpack("<I", payload)[0]),
        "content_bytes": 4,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "threshold_mode": "official_atlif",
        "output_mode": "binary",
        "source_semantics": "OfficialATLIFSurrogate returns out * thre",
    }


def build_folded_weight_device(module, theta):
    return (module.weight.detach() * theta.to(
        device=module.weight.device, dtype=module.weight.dtype)).contiguous()


def save_folded_weight_payload(module, theta, folded_device, directory,
                               identity):
    """Serialize candidates only after the complete S10 theta gate."""
    folded_cpu = folded_device.to(device="cpu").contiguous()
    require(folded_cpu.dtype == torch.float32,
            "M660 folded D1 weight is not float32")
    path = Path(directory) / "d1.weight.folded_theta.f32le"
    payload = folded_cpu.numpy().tobytes(order="C")
    with path.open("xb") as handle:
        handle.write(payload)
    identity["d1_folded_theta"] = {
        "module_index": 1,
        "name": "sttmultires_unet.decoders.1.deconv.0",
        "relative_path": "weights/d1.weight.folded_theta.f32le",
        "definition": "float32(theta_d1 * frozen_d1_weight)",
        "shape": [int(item) for item in folded_cpu.shape],
        "dtype": "torch.float32", "layout": "C_ORDER_CONTIGUOUS",
        "byte_order": "little", "content_bytes": len(payload),
        "content_sha256": sha256(path),
    }
    sidecar = {
        "schema": "m660_d1_original_weight_output_scale_sidecar_v1",
        "status": "CANDIDATE_REQUIRES_SEPARATE_NUMERIC_MITER",
        "definition": "conv_transpose2d(float32(mask), original_FP32_weight) then float32 output scale by runtime theta",
        "original_weight_key": "1",
        "theta_ieee754_le_hex": theta.detach().to(
            device="cpu").contiguous().numpy().tobytes(order="C").hex(),
        "admitted": False,
        "cycles": False,
        "speedup": False,
    }
    sidecar_path = Path(directory) / "d1.original_weight_output_scale.sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    identity["d1_original_weight_output_scale_sidecar"] = {
        "module_index": 1,
        "relative_path":
            "weights/d1.original_weight_output_scale.sidecar.json",
        "content_bytes": sidecar_path.stat().st_size,
        "content_sha256": sha256(sidecar_path),
        "admitted": False,
    }
    return folded_device


def configure_deterministic_execution():
    """Freeze replay controls without changing the checkpoint's eval math."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # M511/M649 were evaluated with the PyTorch 2.7.1 cuDNN default enabled.
    # M681 isolated this bit as the sole cause of the M660-r4 replay mismatch.
    torch.backends.cudnn.allow_tf32 = True
    return observe_execution_controls()


def observe_execution_controls():
    """Read the live backend controls without mutating them."""
    return {
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"),
    }


def require_deterministic_execution(observed):
    require(observed == {
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": ":4096:8",
    }, "M686-r6 deterministic/native-TF32 execution controls drift")


def threshold_identity_matches(model, expected, frozen_identity, phase):
    """Re-read exact live leaf bytes and compare with the immutable identity."""
    _snapshot, observed = decoder_threshold_identity(model, expected)
    require(observed == frozen_identity,
            "M660-r2 D1 threshold drift at " + phase)
    return observed


def scrub_d1_candidates(staging):
    """Remove every D1 candidate from a failed, noncanonical staging tree."""
    staging = Path(staging)
    removed = []
    candidate = staging / "d1_candidate"
    if candidate.exists():
        require(candidate.is_dir() and not candidate.is_symlink(),
                "M660-r2 unsafe D1 candidate directory during scrub")
        for path in sorted(candidate.iterdir()):
            require(path.is_file() and not path.is_symlink(),
                    "M660-r2 unsafe D1 candidate member during scrub")
            path.unlink()
            removed.append(path.relative_to(staging).as_posix())
        candidate.rmdir()
    for relative in (
            "weights/d1.weight.folded_theta.f32le",
            "weights/d1.original_weight_output_scale.sidecar.json"):
        path = staging / relative
        if os.path.lexists(str(path)):
            require(path.is_file() and not path.is_symlink(),
                    "M660-r2 unsafe D1 weight candidate during scrub")
            path.unlink()
            removed.append(relative)
    calls = staging / "calls"
    if calls.is_dir() and not calls.is_symlink():
        for path in sorted(calls.glob("s??_d1.activation.theta.le.bitpack")):
            require(path.is_file() and not path.is_symlink(),
                    "M660-r2 unsafe promoted D1 candidate during scrub")
            path.unlink()
            removed.append(path.relative_to(staging).as_posix())
    # Any seal or success marker created before the exception no longer
    # describes the scrubbed population and must not survive as valid evidence.
    for relative in (
            "weights/SHA256SUMS", "weights/SHA256SUMS.seal.sha256",
            "SHA256SUMS", "SHA256SUMS.seal.sha256",
            "manifest.json", "RUN_COMPLETE.txt"):
        path = staging / relative
        if os.path.lexists(str(path)):
            require(path.is_file() and not path.is_symlink(),
                    "M660-r2 unsafe stale evidence during scrub")
            path.unlink()
            removed.append(relative)
    return removed


def nvidia_smi_identity(tool):
    query = subprocess.run([
        str(tool), "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    require(len(rows) == 1, "M660 requires exactly one visible physical GPU")
    fields = [item.strip() for item in rows[0].split(",")]
    require(len(fields) == 5, "M660 malformed nvidia-smi identity row")
    return {"index": int(fields[0]), "name": fields[1], "uuid": fields[2],
            "driver_version": fields[3],
            "memory_total_mib": int(fields[4])}


def runtime_receipt(contract, directory, contract_sha256, determinism):
    allowed = set(contract["runtime_provenance"]["allowed_environment_names"])
    observed_names = set(os.environ)
    require(observed_names <= allowed,
            "M660 runtime environment exceeds allowlist: " +
            ",".join(sorted(observed_names - allowed)))
    observed_environment = {name: os.environ[name]
                            for name in sorted(observed_names)}
    expected_environment = dict(
        contract["runtime_provenance"]["expected_environment"])
    require(expected_environment["M660R2_EXPECTED_CONTRACT_SHA256"] ==
            "DERIVED_EQUAL_TO_RUNNING_CONTRACT_SHA256",
            "M660 contract-SHA environment policy drift")
    expected_environment["M660R2_EXPECTED_CONTRACT_SHA256"] = contract_sha256
    require(observed_environment == expected_environment,
            "M660 runtime environment value/population drift")
    executable = checked_path(sys.executable, label="Python executable")
    expected_python = contract["runtime_provenance"]["python"]
    require(str(executable) == expected_python["path"] and
            sha256(executable) == expected_python["sha256"],
            "M660 Python executable identity drift")
    expected_argv = contract["runtime_provenance"]["exact_python_argv"]
    require([str(executable)] + sys.argv == expected_argv,
            "M660 exact Python argv drift")
    hostname_tool = checked_path(
        contract["runtime_provenance"]["hostname_tool"]["path"],
        label="hostname tool")
    nvidia_tool = checked_path(
        contract["runtime_provenance"]["nvidia_smi_tool"]["path"],
        label="nvidia-smi tool")
    require(sha256(hostname_tool) == contract["runtime_provenance"]
            ["hostname_tool"]["sha256"] and
            sha256(nvidia_tool) == contract["runtime_provenance"]
            ["nvidia_smi_tool"]["sha256"],
            "M660 runtime identity-tool drift")
    hostname = subprocess.run([str(hostname_tool)], check=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True).stdout.strip()
    gpu = nvidia_smi_identity(nvidia_tool)
    expected_host = contract["runtime_provenance"]["expected_host_gpu"]
    require(hostname == expected_host["hostname"] and
            gpu == expected_host["gpu"], "M660 host/GPU identity drift")
    packages = {}
    for name in contract["runtime_provenance"]["package_names"]:
        packages[name] = importlib.metadata.version(name)
    require(packages == contract["runtime_provenance"]["package_versions"],
            "M660 Python package identity drift")
    require(torch.cuda.is_available(), "M660 CUDA is unavailable")
    current = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(current)
    require(props.name == gpu["name"], "M660 torch/nvidia GPU-name drift")
    receipt = {
        "schema": "m660_h67_layer_static_decoder_runtime_receipt_v1",
        "status": "PASS_RUNTIME_IDENTITY_BEFORE_CAPTURE",
        "hostname": hostname,
        "platform_node": platform.node(),
        "socket_hostname": socket.gethostname(),
        "python": {"executable": str(executable),
                   "executable_sha256": sha256(executable),
                   "version": platform.python_version(),
                   "version_full": sys.version},
        "packages": packages,
        "torch_cuda": {
            "torch_version": torch.__version__,
            "compiled_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": bool(torch.cuda.is_available()),
            "current_device": current,
            "device_name": props.name,
            "compute_capability": [int(props.major), int(props.minor)],
            "total_memory_bytes": int(props.total_memory),
        },
        "deterministic_execution": determinism,
        "nvidia_smi": {"tool": str(nvidia_tool),
                       "tool_sha256": sha256(nvidia_tool), **gpu},
        "command": {"argv": [str(executable)] + sys.argv,
                    "shell": False},
        "environment": {"allowlist": sorted(allowed),
                        "observed": observed_environment,
                        "all_observed_names_allowlisted": True},
        "claim_boundary": {"runtime_identity_only": True,
                           "cycles": False, "speedup": False,
                           "rtl": False, "eda": False,
                           "energy": False, "ppa": False},
    }
    directory = Path(directory)
    directory.mkdir()
    (directory / "runtime_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_double_seal(directory)
    verify_double_seal(directory)
    return receipt


def cpu_exact_load_preflight(contract, contract_path, contract_sha256, m511,
                             m511_contract, config_path, checkpoint_path,
                             output, determinism):
    """Exact-load the frozen checkpoint on CPU and prove the real D1 topology."""
    expected_output = ROOT / contract["cpu_exact_load_preflight"][
        "canonical_directory"]
    output = checked_path_match(output, expected_output,
                                allow_missing_leaf=True,
                                label="CPU exact-load preflight output")
    require(output.parent.is_dir() and not os.path.lexists(str(output)),
            "M660-r2 CPU preflight output must be fresh")
    staging = Path(tempfile.mkdtemp(
        prefix=output.name + ".staging.", dir=str(output.parent)))
    try:
        config, _configured_device = m511.profile.load_config(config_path)
        require(config["model"]["name"] == "MS_SpikingformerFlowNet_en4" and
                config["model"]["use_upsample_conv"] is False and
                int(config["model"]["kernel_size"]) == 3,
                "M660-r2 CPU preflight H67 config drift")
        device = torch.device("cpu")
        model = m511.profile.build_model(config, checkpoint_path, device)
        load_audit = m511.profile.validate_h9_load_audit(model, config)
        require(load_audit is not None and
                int(load_audit.get("missing_count", 0)) == 0 and
                int(load_audit.get("unexpected_count", 0)) == 0,
                "M660-r2 CPU preflight checkpoint load is not exact")
        observed_convtranspose = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.ConvTranspose2d)]
        expected_convtranspose = [row["name"]
                                  for row in m511_contract["modules"]]
        require(observed_convtranspose == expected_convtranspose,
                "M660-r2 CPU preflight ConvTranspose topology drift")
        module_identities = m511.module_identities(
            model, m511_contract["modules"])
        d1_expected = m511_contract["modules"][1]
        _theta, d1_identity = decoder_threshold_identity(model, d1_expected)
        require(d1_identity["wrapper_class"] == "Spiking_neuron" and
                d1_identity["parameter_name"] ==
                ("sttmultires_unet.decoders.1.sn.spiking_neuron.thresh"),
                "M660-r2 CPU preflight did not reach the frozen wrapper/leaf")
        receipt = {
            "schema": "m660r2_h67_cpu_exact_load_preflight_v1",
            "status": "PASS_CPU_EXACT_LOAD_REAL_WRAPPER_AND_ATLIF_LEAF",
            "contract": {"path": str(contract_path),
                         "sha256": contract_sha256},
            "config": {"path": str(config_path),
                       "sha256": sha256(config_path)},
            "checkpoint": {"path": str(checkpoint_path),
                           "sha256": sha256(checkpoint_path),
                           "size_bytes": checkpoint_path.stat().st_size},
            "checkpoint_load_audit": load_audit,
            "convtranspose_names": observed_convtranspose,
            "module_identities": module_identities,
            "d1_threshold_identity": d1_identity,
            "device": "cpu",
            "forward_executed": False,
            "deterministic_execution": determinism,
            "claim_boundary": {
                "topology_and_checkpoint_exact_load_only": True,
                "gpu": False, "one_shot": False, "capture": False,
                "cycles": False, "speedup": False, "rtl": False,
                "eda": False, "energy": False, "ppa": False,
            },
        }
        (staging / "preflight.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M660R2_CPU_EXACT_LOAD_PREFLIGHT\n", encoding="utf-8")
        write_double_seal(staging)
        verify_double_seal(staging)
        require(sha256(contract_path) == contract_sha256 and
                sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
                DOCS359_SHA256,
                "M660-r2 CPU preflight frozen root drift")
        os.replace(staging, output)
        verify_double_seal(output)
        print("PASS M660-r2 CPU exact-load preflight {}".format(
            output / "preflight.json"), flush=True)
        return 0
    except BaseException:
        # This preflight never writes a candidate payload. Preserve only a
        # plainly failed noncanonical staging directory for diagnosis.
        failure = staging / "FAILED.json"
        if not failure.exists():
            failure.write_text(json.dumps({
                "schema": "m660r2_cpu_exact_load_preflight_failure_v1",
                "status": "FAIL_CLOSED_NO_PREFLIGHT_RESULT",
                "candidate_payload_written": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def take_exact(iterable, count):
    iterator = iter(iterable)
    for index in range(count):
        try:
            yield next(iterator)
        except StopIteration:
            raise RuntimeError("M660 loader exhausted at item {} of {}".format(
                index + 1, count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--m511-contract", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cpu-preflight-only", action="store_true")
    parser.add_argument("--cpu-preflight-output", type=Path)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--chunk-elements", required=True, type=int)
    args = parser.parse_args()
    require(sys.byteorder == "little", "M660 requires little endian")
    require(args.samples == 10 and args.num_workers == 0 and
            args.chunk_elements == 8388608,
            "M660 requires S10, num-workers=0, chunk-elements=8388608")

    launcher_path = checked_path(Path(__file__), label="runtime launcher")
    contract_path = checked_path(args.contract, label="runtime contract")
    contract_start = sha256(contract_path)
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m660r2_h67_ep35_layer_static_decoder_payload_contract_v2" and
            contract.get("status") ==
            "STATIC_AUTHOR_HANDOFF_R2__FRESH_HAMMER_REQUIRED_BEFORE_GPU",
            "M660-r2 contract schema/status drift")
    identities = verify_contract_inputs(contract, launcher_path)
    predecessor = verify_predecessor_evidence(contract)

    m511_contract_path = checked_path_match(
        args.m511_contract,
        ROOT / contract["inputs"]["m511_contract"]["path"],
        label="M511 contract")
    config_path = checked_path_match(
        args.config, ROOT / contract["inputs"]["config"]["path"],
        label="config")
    checkpoint_path = checked_path_match(
        args.checkpoint, ROOT / contract["inputs"]["checkpoint"]["path"],
        label="checkpoint")
    m511_contract = strict_json(m511_contract_path)
    require(checkpoint_path.stat().st_size ==
            m511_contract["checkpoint_identity"]["size_bytes"],
            "M660 checkpoint size drift")
    m511_path = checked_contract_path(
        contract["inputs"]["m511_producer"]["path"], label="M511 producer")
    m511 = load_frozen_m511(m511_path)
    frozen_m511_inputs = m511.verify_inputs(
        m511_contract, M511_PRODUCER_SHA256)
    determinism = configure_deterministic_execution()
    require_deterministic_execution(determinism)

    if args.cpu_preflight_only:
        require(args.output_dir is None and
                args.cpu_preflight_output is not None,
                "M660-r2 CPU preflight CLI boundary drift")
        return cpu_exact_load_preflight(
            contract, contract_path, contract_start, m511, m511_contract,
            config_path, checkpoint_path, args.cpu_preflight_output,
            determinism)

    require(args.output_dir is not None and
            args.cpu_preflight_output is None,
            "M660-r2 capture CLI boundary drift")
    output = checked_path_match(
        args.output_dir, ROOT / contract["output"]["canonical_directory"],
        allow_missing_leaf=True, label="output")
    require(output.parent.is_dir() and not os.path.lexists(str(output)),
            "M660-r2 output must be a fresh child of an existing directory")
    preflight = checked_contract_path(
        contract["cpu_exact_load_preflight"]["canonical_directory"],
        label="CPU exact-load preflight")
    require(preflight.is_dir() and not preflight.is_symlink(),
            "M660-r2 CPU exact-load preflight is missing")
    verify_double_seal(preflight)
    preflight_receipt = strict_json(preflight / "preflight.json")
    require(preflight_receipt.get("status") ==
            "PASS_CPU_EXACT_LOAD_REAL_WRAPPER_AND_ATLIF_LEAF" and
            preflight_receipt["contract"]["sha256"] == contract_start and
            preflight_receipt["d1_threshold_identity"]["parameter_name"] ==
            "sttmultires_unet.decoders.1.sn.spiking_neuron.thresh",
            "M660-r2 CPU exact-load preflight receipt drift")
    attempt = checked_contract_path(
        contract["one_shot"]["attempt_directory"], label="attempt directory")
    require(attempt.is_dir() and not attempt.is_symlink(),
            "M660-r2 independently consumed attempt receipt is missing")
    verify_double_seal(attempt / "initial")
    require(checked_path(os.environ.get("M660R2_ATTEMPT_DIRECTORY", ""),
                         label="runtime attempt directory") == attempt,
            "M660-r2 attempt environment identity drift")
    runner_path = checked_path(os.environ.get("M660R2_RUNNER_PATH", ""),
                               label="runtime runner")
    require(runner_path == checked_contract_path(
        contract["inputs"]["runner"]["path"], label="contract runner") and
        sha256(runner_path) == os.environ.get(
            "M660R2_EXPECTED_RUNNER_SHA256"),
        "M660-r2 runner environment identity drift")

    staging = Path(tempfile.mkdtemp(
        prefix=output.name + ".staging.", dir=str(output.parent)))
    (staging / "calls").mkdir()
    (staging / "d1_candidate").mkdir()
    binary_records = []
    d1_records = []
    global_order = []
    handles = []
    current = {"sample_id": None, "order": 0}
    published = False
    quarantine = None
    try:
        runtime = runtime_receipt(contract, staging / "runtime_receipt",
                                  contract_start, determinism)
        config, device = m511.profile.load_config(config_path)
        require_deterministic_execution(observe_execution_controls())
        require(torch.cuda.is_available() and torch.device(device).type == "cuda",
                "M660 requires CUDA model execution")
        require(config["model"]["name"] == "MS_SpikingformerFlowNet_en4" and
                config["model"]["use_upsample_conv"] is False and
                int(config["model"]["kernel_size"]) == 3,
                "M660 H67 decoder config drift")
        dataset = m511.profile.DSECDatasetLite(
            config, file_list="valid", stereo=False,
            scale_factor=config.get("test", {}).get("scale_factor", 1))
        sequence_file = Path(dataset.sequence_file).resolve()
        sequence_identity = {"path": str(sequence_file),
                             "bytes": sequence_file.stat().st_size,
                             "sha256": sha256(sequence_file)}
        sample_sources = [m511.sample_source_identity(dataset, index)
                          for index in range(10)]
        require([row["sample_key"] for row in sample_sources] ==
                [row["sample_key"] for row in m511_contract["samples"]],
                "M660 raw source/sample cohort drift")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False,
            pin_memory=False, num_workers=args.num_workers)
        transform_valid = None
        if config["loader"].get("crop") is not None:
            transform_valid = m511.profile.Compose([m511.profile.CenterCrop((
                config["loader"]["crop"][0], config["loader"]["crop"][1]))])
        model = m511.profile.build_model(config, checkpoint_path, device)
        load_audit = m511.profile.validate_h9_load_audit(model, config)
        require(load_audit is not None and
                int(load_audit.get("missing_count", 0)) == 0 and
                int(load_audit.get("unexpected_count", 0)) == 0,
                "M660 checkpoint load is not exact")
        module_counts = m511.profile.h9_module_counts(model)
        bn_policy = config.get("test", {}).get("bn_policy", "running")
        bn_changed = m511.profile.configure_batch_norm_evaluation(model,
                                                                   bn_policy)
        require_deterministic_execution(observe_execution_controls())
        backend_resolver = m511.profile.configure_snn_backend.__globals__[
            "resolve_snn_backend"]
        resolved_backend, backend_reason = backend_resolver(config)
        psn_targets = [name for name, module in model.named_modules()
                       if module.__class__.__name__ == "PSN"]
        atlif_targets = [name for name, module in model.named_modules()
                         if module.__class__.__name__ == "ATLIFTernaryPSN"]
        atlif_backend_attributes = [
            name for name, module in model.named_modules()
            if module.__class__.__name__ == "ATLIFTernaryPSN" and
            hasattr(module, "backend")]
        backend_attribute_inventory = {}
        for _name, module in model.named_modules():
            if hasattr(module, "backend"):
                key = "{}:{}".format(
                    module.__class__.__name__, str(module.backend))
                backend_attribute_inventory[key] = (
                    backend_attribute_inventory.get(key, 0) + 1)
        backend_identity = {
            "configured_request": config.get("runtime", {}).get(
                "snn_backend"),
            "resolver_result": resolved_backend,
            "resolver_reason": backend_reason,
            "cupy_installed_distribution": "cupy-cuda12x",
            "cupy_installed_version": importlib.metadata.version(
                "cupy-cuda12x"),
            "set_backend_target_class": "PSN",
            "set_backend_target_module_count": len(psn_targets),
            "set_backend_target_module_names": psn_targets,
            "effective_cupy_assignment_count": 0,
            "complete_backend_attribute_inventory":
                backend_attribute_inventory,
            "actual_spike_execution": {
                "class": "ATLIFTernaryPSN",
                "module_count": len(atlif_targets),
                "backend_attribute_count": len(atlif_backend_attributes),
                "forward_primitive": "torch.addmm",
                "source_path": identities["atlif_impl"]["path"],
                "source_sha256": identities["atlif_impl"]["sha256"],
                "remaining_spikingjelly_ifnode_count":
                    backend_attribute_inventory.get("IFNode:torch", 0),
                "remaining_spikingjelly_ifnode_backend": "torch",
            },
            "resolver_label_is_not_claimed_as_actual_cupy_execution": True,
        }
        require(backend_identity == {
            "configured_request": "cupy",
            "resolver_result": "cupy",
            "resolver_reason": "explicit config",
            "cupy_installed_distribution": "cupy-cuda12x",
            "cupy_installed_version": "14.2.0",
            "set_backend_target_class": "PSN",
            "set_backend_target_module_count": 0,
            "set_backend_target_module_names": [],
            "effective_cupy_assignment_count": 0,
            "complete_backend_attribute_inventory": {
                "Dropout:torch": 49, "IFNode:torch": 4},
            "actual_spike_execution": {
                "class": "ATLIFTernaryPSN",
                "module_count": 105,
                "backend_attribute_count": 0,
                "forward_primitive": "torch.addmm",
                "source_path": identities["atlif_impl"]["path"],
                "source_sha256":
                    "d9ee7e172f941a53ad1c031b0d5cdbbf7819f521c807e5bc54001a80c41b57f3",
                "remaining_spikingjelly_ifnode_count": 4,
                "remaining_spikingjelly_ifnode_backend": "torch",
            },
            "resolver_label_is_not_claimed_as_actual_cupy_execution": True,
        }, "M686-r6 resolved-versus-actual SNN execution identity drift")
        observed_protocol = {
            "resolution": list(config["loader"]["resolution"]),
            "crop": config["loader"].get("crop"),
            "window_size": list(config["swin_transformer"]["window_size"]),
            "pretrained_window_size": config["swin_transformer"].get(
                "pretrained_window_size"),
            "tokens_per_window": int(np.prod(
                config["swin_transformer"]["window_size"])),
            "remap": config["loader"].get("remap"),
            "bn_policy": bn_policy, "bn_modules_changed": bn_changed,
            "eval_batch_size": 1, "num_workers": args.num_workers,
            "module_counts": module_counts,
        }
        require(observed_protocol == m511_contract["eval_protocol"],
                "M660 evaluation protocol mismatch")
        module_identities = m511.module_identities(
            model, m511_contract["modules"])
        named = dict(model.named_modules())
        require([name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.ConvTranspose2d)] ==
                [item["name"] for item in m511_contract["modules"]],
                "M660 complete ConvTranspose2d module set drift")
        weights = save_weight_payloads(
            model, m511_contract["modules"], staging / "weights")
        for index, identity in weights.items():
            require(identity["content_sha256"] == module_identities[
                m511_contract["modules"][int(index)]["name"]]["weight"][
                    "content_sha256"], "M660 weight payload identity drift")
        d1_expected = m511_contract["modules"][1]
        d1_theta, d1_threshold_identity = decoder_threshold_identity(
            model, d1_expected)
        d1_module = named[d1_expected["name"]]
        d1_folded_weight_device = build_folded_weight_device(
            d1_module, d1_theta)
        theta_stability_checks = {
            "initial_identity": 1,
            "leaf_pre_forward": 0,
            "leaf_post_forward": 0,
            "d1_deconv_pre_hook": 0,
            "d1_deconv_post_hook": 0,
            "sample_pre_forward": 0,
            "sample_post_forward": 0,
            "final_identity": 0,
        }

        def check_theta_stable(phase, counter):
            threshold_identity_matches(
                model, d1_expected, d1_threshold_identity, phase)
            theta_stability_checks[counter] += 1

        def make_hook(expected):
            def hook(module, inputs, output_tensor):
                sample_id = current["sample_id"]
                index = int(expected["module_index"])
                require(sample_id is not None and current["order"] == index,
                        "M660 decoder call order drift")
                require(isinstance(inputs, tuple) and len(inputs) == 1 and
                        torch.is_tensor(inputs[0]) and
                        torch.is_tensor(output_tensor),
                        "M660 hook tensor arity drift")
                value = inputs[0]
                require([int(item) for item in value.shape] ==
                        expected["input_shape"] and
                        [int(item) for item in output_tensor.shape] ==
                        expected["output_shape"],
                        "M660 hook shape drift")
                prior = predecessor["expected_records"][(sample_id, index)]
                require([int(item) for item in value.shape] == prior["shape"] and
                        str(value.dtype) == prior["dtype"] and
                        [int(item) for item in value.stride()] == prior["stride"],
                        "M660 M649 tensor identity drift")
                output_hash = stream_raw_content_hash(
                    output_tensor, args.chunk_elements)
                order_row = {"global_call_index": sample_id * 4 + index,
                             "sample_id": sample_id, "module_index": index,
                             "name": expected["name"]}
                if index in BINARY_MODULE_INDICES:
                    relative = "calls/s{:02d}_d{}.activation.le.bitpack".format(
                        sample_id, index)
                    stats = stream_binary_input(
                        value, args.chunk_elements, staging / relative)
                    for key in ("elements", "zero_count", "one_count",
                                "nonbinary_finite_count", "nonfinite_count"):
                        require(stats[key] == prior[key],
                                "M660/M649 binary count mismatch {}".format(key))
                    if sample_id == 0 and index == 0:
                        require(stats["one_count"] == 839586 and
                                stats["zero_count"] == 3768414 and
                                stats["packed_sha256"] ==
                                identities["m511_failed_d0_payload"]["sha256"],
                                "M686-r6 S00/D0 frozen bit-exact sentinel drift")
                    binary_records.append({
                        **order_row, "sample_key": m511_contract["samples"]
                        [sample_id]["sample_key"],
                        "sequence_key": m511_contract["samples"]
                        [sample_id]["sequence_key"],
                        "route": "EXACT_BINARY_BITPACK",
                        "input_shape": expected["input_shape"],
                        "input_dtype": str(value.dtype),
                        "input_stride": [int(item) for item in value.stride()],
                        "relative_path": relative,
                        "input": stats, "output": output_hash,
                    })
                    order_row["route"] = "EXACT_BINARY_BITPACK"
                else:
                    require(index == FALLBACK_MODULE_INDEX,
                            "M660 unexpected decoder route")
                    check_theta_stable(
                        "sample_{:02d}_d1_deconv_post_hook".format(sample_id),
                        "d1_deconv_post_hook")
                    summary = summarize_d1_fallback(value, args.chunk_elements)
                    for key in ("elements", "zero_count", "one_count",
                                "nonbinary_finite_count", "nonfinite_count"):
                        require(summary[key] == prior[key],
                                "M660/M649 D1 count mismatch {}".format(key))
                    candidate_relative = (
                        "d1_candidate/s{:02d}_d1.activation.theta.le.bitpack"
                        .format(sample_id))
                    theta_binary = stream_theta_binary_candidate(
                        value, d1_theta, args.chunk_elements,
                        staging / candidate_relative)
                    require(theta_binary["zero_count"] == prior["zero_count"] and
                            (theta_binary["theta_count"] +
                             theta_binary["other_finite_count"]) ==
                            prior["nonbinary_finite_count"] and
                            theta_binary["nonfinite_count"] ==
                            prior["nonfinite_count"],
                            "M660/M649 D1 theta accounting mismatch")
                    folded_miter = None
                    if theta_binary["theta_gate_pass"]:
                        mask = (value == d1_theta).to(dtype=value.dtype)
                        flattened = mask.flatten(0, 1)
                        reference = torch_functional.conv_transpose2d(
                            flattened, d1_folded_weight_device,
                            bias=d1_module.bias,
                            stride=d1_module.stride,
                            padding=d1_module.padding,
                            output_padding=d1_module.output_padding,
                            groups=d1_module.groups,
                            dilation=d1_module.dilation).reshape(
                                output_tensor.shape)
                        folded_miter = compare_tensors_streaming(
                            output_tensor, reference, args.chunk_elements)
                        require(folded_miter["original_output_sha256"] ==
                                output_hash["content_sha256"],
                                "M660 D1 original-output hash disagreement")
                    d1_records.append({
                        **order_row, "sample_key": m511_contract["samples"]
                        [sample_id]["sample_key"],
                        "sequence_key": m511_contract["samples"]
                        [sample_id]["sequence_key"],
                        "route": "COMMON_FP32_DENSE_FALLBACK",
                        "input_shape": expected["input_shape"],
                        "input_dtype": str(value.dtype),
                        "input_stride": [int(item) for item in value.stride()],
                        "input": summary,
                        "theta_binary_candidate": theta_binary,
                        "folded_weight_miter": folded_miter,
                        "candidate_relative_path": (candidate_relative
                            if theta_binary["theta_gate_pass"] else None),
                        "output": output_hash,
                    })
                    order_row["route"] = "D1_DUAL_RESULT_PENDING"
                global_order.append(order_row)
                current["order"] += 1
            return hook

        for expected in m511_contract["modules"]:
            handles.append(named[expected["name"]].register_forward_hook(
                make_hook(expected)))

        def require_d1_hook_context(label):
            sample_id = current["sample_id"]
            require(isinstance(sample_id, int) and 0 <= sample_id < 10 and
                    current["order"] == 1,
                    "M660-r2 D1 {} sample/order drift".format(label))
            return sample_id

        def d1_deconv_pre_hook(_module, _inputs):
            sample_id = require_d1_hook_context("deconv-pre")
            check_theta_stable(
                "sample_{:02d}_d1_deconv_pre_hook".format(
                    sample_id),
                "d1_deconv_pre_hook")

        handles.append(d1_module.register_forward_pre_hook(
            d1_deconv_pre_hook))

        def d1_leaf_pre_hook(_module, _inputs):
            sample_id = require_d1_hook_context("leaf-pre")
            check_theta_stable(
                "sample_{:02d}_d1_leaf_pre".format(sample_id),
                "leaf_pre_forward")

        def d1_leaf_post_hook(_module, _inputs, _output):
            sample_id = require_d1_hook_context("leaf-post")
            check_theta_stable(
                "sample_{:02d}_d1_leaf_post".format(sample_id),
                "leaf_post_forward")

        d1_leaf = named[d1_threshold_identity["leaf_name"]]
        handles.append(d1_leaf.register_forward_pre_hook(d1_leaf_pre_hook))
        handles.append(d1_leaf.register_forward_hook(d1_leaf_post_hook))
        sync_counts = {"before_capture": 0, "per_sample_post_forward": 0,
                       "final_pre_manifest": 0}
        torch.cuda.synchronize(device)
        sync_counts["before_capture"] += 1
        processed = 0
        with torch.no_grad():
            for chunk, mask, label in take_exact(loader, args.samples):
                functional.reset_net(model)
                require_deterministic_execution(observe_execution_controls())
                check_theta_stable(
                    "sample_{:02d}_pre_forward".format(processed),
                    "sample_pre_forward")
                sample_key, sequence_key = m511.sample_identity(
                    dataset, processed)
                require(m511_contract["samples"][processed] == {
                    "sample_id": processed, "sample_key": sample_key,
                    "sequence_key": sequence_key},
                    "M660 sample identity drift")
                current.update({"sample_id": processed, "order": 0})
                x, _label, _mask = m511.profile.preprocess_chunk(
                    config, chunk, label, mask, transform_valid, device)
                model(x)
                torch.cuda.synchronize(device)
                sync_counts["per_sample_post_forward"] += 1
                require_deterministic_execution(observe_execution_controls())
                check_theta_stable(
                    "sample_{:02d}_post_forward".format(processed),
                    "sample_post_forward")
                require(current["order"] == 4, "M660 missing decoder call")
                current.update({"sample_id": None, "order": 0})
                processed += 1
                print("[M660] captured sample {}/10".format(processed),
                      flush=True)
        require(processed == 10 and len(binary_records) == 30 and
                len(d1_records) == 10 and len(global_order) == 40,
                "M660 capture lattice drift")
        torch.cuda.synchronize(device)
        sync_counts["final_pre_manifest"] += 1
        require_deterministic_execution(observe_execution_controls())
        while handles:
            handles.pop().remove()
        check_theta_stable("final_pre_manifest", "final_identity")
        require(theta_stability_checks == {
            "initial_identity": 1,
            "leaf_pre_forward": 10,
            "leaf_post_forward": 10,
            "d1_deconv_pre_hook": 10,
            "d1_deconv_post_hook": 10,
            "sample_pre_forward": 10,
            "sample_post_forward": 10,
            "final_identity": 1,
        }, "M660-r2 D1 threshold stability-check lattice drift")

        binary_records.sort(key=lambda row: (row["sample_id"],
                                             row["module_index"]))
        d1_records.sort(key=lambda row: row["sample_id"])
        expected_binary_lattice = [(sample_id, index)
                                   for sample_id in range(10)
                                   for index in BINARY_MODULE_INDICES]
        require([(row["sample_id"], row["module_index"])
                 for row in binary_records] == expected_binary_lattice,
                "M660 binary 30-cell order drift")
        require([(row["sample_id"], row["module_index"])
                 for row in d1_records] ==
                [(sample_id, 1) for sample_id in range(10)],
                "M660 D1 fallback lattice drift")
        require([(row["sample_id"], row["module_index"])
                 for row in global_order] ==
                [(sample_id, index) for sample_id in range(10)
                 for index in range(4)],
                "M660 global 40-call order drift")
        d1_theta_gate_pass = all(row["theta_binary_candidate"]
                                 ["theta_gate_pass"] for row in d1_records)
        d1_folded_miter_bit_exact = folded_miter_admitted(
            d1_records, d1_theta_gate_pass)
        candidate_dir = staging / "d1_candidate"
        if d1_theta_gate_pass:
            save_folded_weight_payload(
                d1_module, d1_theta, d1_folded_weight_device,
                staging / "weights", weights)
            for row in d1_records:
                source = staging / row.pop("candidate_relative_path")
                relative = "calls/s{:02d}_d1.activation.theta.le.bitpack".format(
                    row["sample_id"])
                target = staging / relative
                require(source.is_file() and not target.exists(),
                        "M660 D1 candidate promotion precondition failed")
                os.replace(source, target)
                row["relative_path"] = relative
                row["route"] = "EXACT_SCALED_BINARY_BITPACK"
                row["input"]["route"] = (
                    "EXACT_SCALED_BINARY_BITPACK")
            candidate_dir.rmdir()
            for row in global_order:
                if row["module_index"] == 1:
                    row["route"] = (
                        "EXACT_SCALED_BINARY_BITPACK")
            weights["d1_folded_theta"]["deployment_admitted"] = bool(
                d1_folded_miter_bit_exact)
            weights["d1_folded_theta"]["role"] = (
                "BIT_EXACT_DEPLOYMENT" if d1_folded_miter_bit_exact else
                "DIAGNOSTIC_CANDIDATE_NOT_ADMITTED")
            weights["d1_original_weight_output_scale_sidecar"][
                "deployment_admitted"] = False
            weights["d1_original_weight_output_scale_sidecar"]["role"] = (
                "UNMITERED_CANDIDATE_NOT_ADMITTED")
        else:
            for path in candidate_dir.iterdir():
                require(path.is_file() and not path.is_symlink(),
                        "M660 unsafe D1 candidate cleanup population")
                path.unlink()
            candidate_dir.rmdir()
            require("d1_folded_theta" not in weights and
                    "d1_original_weight_output_scale_sidecar" not in weights,
                    "M660-r2 negative route serialized a delayed candidate")
            for row in d1_records:
                row.pop("candidate_relative_path", None)
                row["route"] = "COMMON_FP32_DENSE_FALLBACK"
            for row in global_order:
                if row["module_index"] == 1:
                    row["route"] = "COMMON_FP32_DENSE_FALLBACK"
        write_double_seal(staging / "weights")
        verify_double_seal(staging / "weights")

        by_module_bytes = {str(index): sum(
            row["input"]["packed_bytes"] for row in binary_records
            if row["module_index"] == index) for index in BINARY_MODULE_INDICES}
        if d1_theta_gate_pass:
            by_module_bytes["1"] = sum(
                row["theta_binary_candidate"]["packed_bytes"]
                for row in d1_records)
        packed_bytes = sum(by_module_bytes.values())
        expected_population = contract["expected_population"]
        expected_case = ("theta_binary_go" if d1_theta_gate_pass
                         else "fp32_fallback")
        require(by_module_bytes == expected_population[expected_case]
                ["binary_packed_bytes_by_module"] and
                packed_bytes == expected_population[expected_case]
                ["binary_packed_bytes_total"],
            "M660 binary payload byte population drift")
        call_files = sorted(path.relative_to(staging).as_posix()
                            for path in (staging / "calls").iterdir()
                            if path.is_file())
        expected_call_files = [row["relative_path"] for row in binary_records]
        if d1_theta_gate_pass:
            expected_call_files.extend(row["relative_path"]
                                       for row in d1_records)
        require(call_files == sorted(expected_call_files) and
                (sum("_d1." in name for name in call_files) ==
                 (10 if d1_theta_gate_pass else 0)),
                "M660 D1 dual-result payload population drift")

        require(sha256(launcher_path) ==
                contract["inputs"]["launcher"]["sha256"] and
                sha256(contract_path) == contract_start,
                "M660 launcher/contract mutated during capture")
        verify_contract_inputs(contract, launcher_path)
        verify_predecessor_evidence(contract)
        m511.verify_inputs(m511_contract, M511_PRODUCER_SHA256)
        m511.rehash_sample_sources(
            sequence_file, sequence_identity, sample_sources)
        verify_double_seal(staging / "runtime_receipt")
        verify_double_seal(staging / "weights")
        require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
                DOCS359_SHA256, "M660 docs/359 drift")
        if d1_theta_gate_pass and d1_folded_miter_bit_exact:
            result_status = (
                "PASS_S10_ALL4_SCALED_BINARY__D1_FOLDED_WEIGHT_MITER_BIT_EXACT")
        elif d1_theta_gate_pass:
            result_status = (
                "PASS_S10_ALL4_SCALED_BINARY__D1_FOLDED_WEIGHT_MITER_NONEXACT")
        else:
            result_status = (
                "PASS_S10_D0_D2_D3_BINARY__D1_COMMON_FP32_FALLBACK")
        manifest = {
            "schema": "m660_h67_ep35_layer_static_decoder_payload_v1",
            "status": result_status,
            "identity": {"contract": {"path": str(contract_path),
                                         "sha256": contract_start},
                         "inputs": identities,
                         "frozen_m511_inputs": frozen_m511_inputs,
                         "predecessor_evidence": {
                             key: value for key, value in predecessor.items()
                             if key != "expected_records"},
                         "checkpoint_load_audit": load_audit},
            "eval_protocol": observed_protocol,
            "raw_validation_sources": {"sequence_list": sequence_identity,
                                       "samples": sample_sources},
            "runtime_receipt": {
                "relative_directory": "runtime_receipt",
                "outer_seal_file_sha256": sha256(
                    staging / "runtime_receipt/SHA256SUMS.seal.sha256"),
                "hostname": runtime["hostname"],
                "gpu_uuid": runtime["nvidia_smi"]["uuid"]},
            "cpu_exact_load_preflight": {
                "relative_to_repository": preflight.relative_to(ROOT).as_posix(),
                "outer_seal_file_sha256": sha256(
                    preflight / "SHA256SUMS.seal.sha256"),
                "status": preflight_receipt["status"]},
            "deterministic_execution": determinism,
            "snn_backend_identity": backend_identity,
            "cuda_synchronization": sync_counts,
            "layer_static_route_table": {
                "d0": "EXACT_BINARY_BITPACK",
                "d1": ("EXACT_SCALED_BINARY_BITPACK"
                       if d1_theta_gate_pass else
                       "COMMON_FP32_DENSE_FALLBACK"),
                "d2": "EXACT_BINARY_BITPACK",
                "d3": "EXACT_BINARY_BITPACK"},
            "module_identities": module_identities,
            "weight_payloads": weights,
            "d1_scalar_threshold_identity": d1_threshold_identity,
            "d1_threshold_stability_checks": theta_stability_checks,
            "d1_dual_result_decision": {
                "exact_zero_or_runtime_scalar_theta_s10": d1_theta_gate_pass,
                "folded_weight_convtranspose_miter_bit_exact_s10":
                    d1_folded_miter_bit_exact,
                "scaled_binary_representation_admitted": d1_theta_gate_pass,
                "folded_weight_deployment_admitted":
                    d1_folded_miter_bit_exact,
                "folded_weight_payload_role": (
                    "BIT_EXACT_DEPLOYMENT" if d1_folded_miter_bit_exact else
                    ("DIAGNOSTIC_CANDIDATE_NOT_ADMITTED"
                     if d1_theta_gate_pass else "ABSENT")),
                "original_weight_output_scale_sidecar_role": (
                    "UNMITERED_CANDIDATE_NOT_ADMITTED"
                    if d1_theta_gate_pass else "ABSENT"),
                "fallback_selected": not d1_theta_gate_pass,
                "miter_nonexact_is_not_silently_admitted":
                    bool(d1_theta_gate_pass and
                         not d1_folded_miter_bit_exact)},
            "population": {
                "samples": 10, "hook_calls": 40,
                "binary_payload_records": (40 if d1_theta_gate_pass else 30),
                "d0_d2_d3_binary_payload_records": 30,
                "d1_metadata_records": 10,
                "d1_theta_binary_payload_records":
                    (10 if d1_theta_gate_pass else 0),
                "binary_packed_bytes_by_module": by_module_bytes,
                "binary_packed_bytes_total": packed_bytes,
                "d1_raw_payload_files": 0,
                "d1_raw_payload_bytes": 0},
            "global_call_order": global_order,
            "d0_d2_d3_binary_records": binary_records,
            "d1_records": d1_records,
            "packing": {"values": [0, 1], "bit_order": "little",
                        "order": "C_ORDER_FLAT",
                        "whole_call_contiguous_copy_allowed": False},
            "d1_policy": {
                "route": ("EXACT_SCALED_BINARY_BITPACK"
                          if d1_theta_gate_pass else
                          "COMMON_FP32_DENSE_FALLBACK"),
                "bitpack": bool(d1_theta_gate_pass), "raw_payload": False,
                "threshold": False, "round": False,
                "binary_coercion": False,
                "exact_comparison": "x==0 or x==runtime_scalar_theta",
                "folded_weight_definition":
                    "float32(runtime_scalar_theta * frozen_FP32_weight)",
                "folded_weight_deployment_admitted_only_if_bit_exact_miter":
                    d1_folded_miter_bit_exact,
                "original_weight_plus_output_scale_sidecar_admitted": False,
                "fallback_on_any_non_zero_non_theta": True,
                "same_fallback_required_for_all_future_baselines_if_selected":
                    True},
            "claim_boundary": {
                "capture_payload": True,
                "d0_d2_d3_exact_binary_observed_s10": True,
                "d1_exact_scaled_binary_observed_s10": d1_theta_gate_pass,
                "d1_common_fp32_dense_fallback": not d1_theta_gate_pass,
                "d1_folded_weight_miter_bit_exact":
                    d1_folded_miter_bit_exact,
                "decoder_numeric_equivalence": False,
                "cycles": False, "speedup": False,
                "rtl": False, "vcs": False, "eda": False,
                "dc": False, "formality": False, "ptpx": False,
                "energy": False, "ppa": False,
                "system_speedup": False, "date_headline": False},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M660_S10_LAYER_STATIC_DECODER_PAYLOAD\n", encoding="utf-8")
        write_double_seal(staging)
        verify_double_seal(staging)
        require(not output.exists(), "M660 output appeared during capture")
        quarantine = output.with_name(
            output.name + ".quarantine.failed.{}.{}".format(
                os.getpid(), uuid.uuid4().hex))
        require(not quarantine.exists(), "M660 quarantine target exists")
        os.replace(staging, output)
        published = True
        verify_double_seal(output)
        print("PASS M660 {} {}".format(
            output / "manifest.json", sha256(output / "manifest.json")),
            flush=True)
    except BaseException as error:
        scrubbed = scrub_d1_candidates(staging)
        if published:
            os.replace(output, quarantine)
            require(not output.exists() and quarantine.is_dir(),
                    "M660 failed to quarantine canonical output")
            failure_root = quarantine
            scrubbed.extend(scrub_d1_candidates(failure_root))
            failure_name = "FAILED_POSTPUBLICATION.json"
        else:
            failure_root = staging
            failure_name = "FAILED.json"
        failure = failure_root / failure_name
        if not failure.exists():
            failure.write_text(json.dumps({
                "schema": "m660_h67_layer_static_decoder_capture_failure_v1",
                "status": "FAIL_CLOSED_NO_CANONICAL_RESULT",
                "reason": "{}: {}".format(type(error).__name__, error),
                "binary_records_completed": len(binary_records),
                "d1_records_completed": len(d1_records),
                "d1_payload_permitted_only_after_all_s10_theta_gate": True,
                "d1_candidate_scrubbed_before_failure_receipt": True,
                "scrubbed_relative_paths": sorted(set(scrubbed)),
                "staging_directory": str(staging),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        weights_failure = failure_root / "weights"
        if weights_failure.is_dir() and not weights_failure.is_symlink():
            write_double_seal(weights_failure)
            verify_double_seal(weights_failure)
        write_double_seal(failure_root)
        verify_double_seal(failure_root)
        raise
    finally:
        for handle in handles:
            try:
                handle.remove()
            except BaseException:
                pass
    return 0


if __name__ == "__main__":
    os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")
    raise SystemExit(main())
