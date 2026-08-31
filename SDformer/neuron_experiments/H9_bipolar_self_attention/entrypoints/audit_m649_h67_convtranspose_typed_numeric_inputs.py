#!/usr/bin/env python3
"""Audit the numeric types at the four frozen H67 decoder inputs.

M649 is a diagnostic successor to the failed-closed M511 one-shot.  It imports
the exact frozen M511 producer only after all producer/model identities have
been checked, and then reuses the same config, dataset, checkpoint loader,
preprocessing, model construction, BN policy, sample order and forward path.
It writes numeric summaries only: no activation payload, cycle, speedup, RTL,
energy or PPA result is produced.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np
import torch
from spikingjelly.activation_based import functional


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M511_PRODUCER_SHA256 = (
    "e16a454d532acd15d96527cfddf43ebf9f95338a34ce9aeedbb10032cb26230a")
M511_CONTRACT_SHA256 = (
    "e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
EXPECTED_M511_STAGING_POPULATION = {
    "FAILED.json": (403,
                    "343a29e2932345e83d9da2410eb070f22520b9c0e4302e4940a25633e2250863"),
    "calls/s00_d0.activation.le.bitpack": (
        576000,
        "ad2251f1fb8a470651044456e0b7182bd6db0e0a89fb63018efa3a9e6fcd6447"),
}


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
        raise RuntimeError("M649 non-standard JSON token: " + token)

    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "M649 duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def reject_symlink_chain(path, allow_missing_leaf=False):
    """Reject symlinks in an existing path chain, including dangling leaves."""
    path = Path(os.path.abspath(str(path)))
    parts = path.parts
    cursor = Path(parts[0])
    for index, part in enumerate(parts[1:], 1):
        cursor = cursor / part
        is_leaf = index == len(parts) - 1
        if os.path.lexists(str(cursor)):
            require(not cursor.is_symlink(),
                    "M649 rejects symlink path component: " + str(cursor))
        else:
            require(is_leaf and allow_missing_leaf,
                    "M649 missing path component: " + str(cursor))


def checked_path(path, allow_missing_leaf=False, label="path"):
    """Reject lexical aliases/symlinks before returning a resolved path.

    Path.resolve() must never precede the symlink test: doing so would hide both
    an input alias and a dangling output leaf.  Rejecting ``..`` also prevents a
    symlink-before-parent traversal from being collapsed by abspath semantics.
    """
    raw = Path(path)
    require(".." not in raw.parts,
            "M649 rejects parent traversal in {}: {}".format(label, raw))
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    reject_symlink_chain(absolute, allow_missing_leaf=allow_missing_leaf)
    return absolute.resolve(strict=not allow_missing_leaf)


def checked_path_match(raw, expected, allow_missing_leaf=False, label="path"):
    """Fail closed on either raw chain, then require canonical equality."""
    observed = checked_path(raw, allow_missing_leaf=allow_missing_leaf,
                            label="runtime " + label)
    wanted = checked_path(expected, allow_missing_leaf=allow_missing_leaf,
                          label="expected " + label)
    require(observed == wanted, "M649 {} path drift".format(label))
    return observed


def checked_contract_path(entry, allow_missing_leaf=False, label="input"):
    """Resolve a repository-relative contract member without aliasing."""
    relative = Path(entry)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "M649 unsafe contract {} path: {}".format(label, relative))
    return checked_path(ROOT / relative,
                        allow_missing_leaf=allow_missing_leaf, label=label)


def verify_double_seal(directory):
    directory = Path(directory)
    reject_symlink_chain(directory)
    seal = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(seal.is_file() and outer.is_file(),
            "M649 missing predecessor double seal")
    expected, name = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(name == "SHA256SUMS" and sha256(seal) == expected,
            "M649 predecessor outer seal mismatch")
    sealed = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(name not in sealed and name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256"),
            "M649 unsafe/duplicate sealed member: " + name)
        member = directory / name
        require(member.is_file() and not member.is_symlink() and
                sha256(member) == expected,
                "M649 predecessor sealed member mismatch: " + name)
        sealed.add(name)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256")
    }
    require(actual == sealed,
            "M649 predecessor sealed/actual population mismatch")


def write_double_seal(directory):
    directory = Path(directory)
    members = [
        path.relative_to(directory)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256")
    ]
    seal = directory / "SHA256SUMS"
    seal.write_text("".join(
        "{}  {}\n".format(sha256(directory / member), member.as_posix())
        for member in members), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text("{}  SHA256SUMS\n".format(sha256(seal)),
                     encoding="utf-8")


def verify_own_double_seal(directory):
    verify_double_seal(directory)


def verify_failed_m511_state(contract):
    attempt = checked_contract_path(
        contract["failed_m511_state"]["attempt_initial_directory"],
        label="failed M511 attempt")
    staging = checked_contract_path(
        contract["failed_m511_state"]["failed_staging_directory"],
        label="failed M511 staging")
    canonical = checked_contract_path(
        contract["failed_m511_state"]["forbidden_m511_canonical"],
        allow_missing_leaf=True, label="forbidden M511 canonical")
    require(attempt.is_dir() and staging.is_dir(),
            "M649 requires the frozen failed M511 attempt")
    require(not os.path.lexists(str(canonical)),
            "M649 refuses to coexist with an M511 canonical payload")
    verify_double_seal(attempt)
    actual = {
        path.relative_to(staging).as_posix(): (path.stat().st_size, sha256(path))
        for path in staging.rglob("*") if path.is_file()
    }
    require(actual == EXPECTED_M511_STAGING_POPULATION,
            "M649 failed M511 staging identity/population drift")
    failure = strict_json(staging / "FAILED.json")
    require(failure == {
        "completed_records": 1,
        "reason": "RuntimeError: M511 raw ConvTranspose2d input is not exact binary",
        "schema": "m511_h67_ep35_convtranspose_capture_failure_v1",
        "staging_directory": str(staging.resolve()),
        "status": "FAIL_CLOSED_NO_PASS_MANIFEST",
    }, "M649 unexpected M511 failure receipt")
    return {
        "attempt_initial_directory": str(attempt.resolve()),
        "attempt_seal_sha256": sha256(attempt / "SHA256SUMS"),
        "attempt_outer_seal_file_sha256": sha256(
            attempt / "SHA256SUMS.seal.sha256"),
        "failed_staging_directory": str(staging.resolve()),
        "failed_staging_population": {
            name: {"bytes": value[0], "sha256": value[1]}
            for name, value in sorted(actual.items())
        },
        "original_m511_canonical_absent": True,
    }


def verify_prior_failed_m649_state(contract):
    """Freeze the completed-40-record/no-result first diagnostic attempt."""
    state = contract["prior_failed_m649_execution"]
    staging = checked_contract_path(
        state["failed_staging_directory"],
        label="prior failed M649 staging")
    canonical = checked_contract_path(
        contract["output"]["canonical_directory"],
        allow_missing_leaf=True, label="M649 canonical")
    require(staging.is_dir() and not staging.is_symlink(),
            "M649 prior failed staging is not a regular directory")
    require(not os.path.lexists(str(canonical)),
            "M649 retry refuses an existing canonical result")
    entries = list(staging.iterdir())
    require(len(entries) == 1 and entries[0].name == "FAILED.json" and
            entries[0].is_file() and not entries[0].is_symlink(),
            "M649 prior failed staging population drift")
    receipt = entries[0]
    require(receipt.stat().st_size == int(state["failed_receipt_bytes"]) and
            sha256(receipt) == state["failed_receipt_sha256"],
            "M649 prior failed receipt identity drift")
    failure = strict_json(receipt)
    require(failure == {
        "completed_records": 40,
        "original_m511_staging_preserved": True,
        "reason": state["expected_reason"],
        "schema": "m649_h67_ep35_convtranspose_typed_numeric_failure_v1",
        "staging_directory": str(staging),
        "status": "FAIL_CLOSED_NO_RESULT",
    }, "M649 prior failed receipt semantic drift")
    return {
        "failed_staging_directory": str(staging),
        "failed_receipt_bytes": receipt.stat().st_size,
        "failed_receipt_sha256": sha256(receipt),
        "completed_records_before_prefetch_failure": 40,
        "canonical_result_absent": True,
        "retry_reason": "STRICT_TEN_NEXT_CALLS_NO_ELEVENTH_DATALOADER_FETCH",
    }


def verify_contract_inputs(contract, launcher_path):
    expected_keys = set(contract["required_input_names"])
    require(set(contract["inputs"]) == expected_keys,
            "M649 contract input population drift")
    identities = {}
    for name, entry in contract["inputs"].items():
        path = checked_contract_path(entry["path"], label="input " + name)
        require(path.is_file() and not path.is_symlink(),
                "M649 missing/symlink input: " + name)
        observed = sha256(path)
        require(observed == entry["sha256"],
                "M649 input identity drift: " + name)
        identities[name] = {
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": observed,
        }
    require(checked_contract_path(
        contract["inputs"]["launcher"]["path"], label="input launcher") ==
            checked_path(launcher_path, label="runtime launcher"),
            "M649 launcher path drift")
    require(identities["m511_producer"]["sha256"] == M511_PRODUCER_SHA256 and
            identities["m511_contract"]["sha256"] == M511_CONTRACT_SHA256 and
            identities["docs359"]["sha256"] == DOCS359_SHA256,
            "M649 critical frozen identity drift")
    return identities


def load_frozen_m511(producer_path):
    """Import M511 only after its full trust root has been checked."""
    entrypoint = str(producer_path.parent)
    if entrypoint not in sys.path:
        sys.path.insert(0, entrypoint)
    spec = importlib.util.spec_from_file_location(
        "m649_frozen_m511_producer", str(producer_path))
    require(spec is not None and spec.loader is not None,
            "M649 cannot construct frozen M511 import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(sha256(producer_path) == M511_PRODUCER_SHA256,
            "M649 M511 producer drift across import")
    return module


def reduce_counts(tensor, channel_block):
    """Return exact per-channel numeric counts without coercing the input."""
    require(torch.is_tensor(tensor) and tensor.ndim == 5,
            "M649 expects T_B_C_H_W tensor")
    require(channel_block > 0, "M649 invalid channel block")
    value = tensor.detach()
    channels = int(value.shape[2])
    per_channel = []
    reduction_dims = (0, 1, 3, 4)
    elements_per_channel = int(value.shape[0] * value.shape[1] *
                               value.shape[3] * value.shape[4])
    for begin in range(0, channels, channel_block):
        end = min(channels, begin + channel_block)
        block = value[:, :, begin:end, :, :]
        finite = torch.isfinite(block)
        zero = torch.sum(block == 0, dim=reduction_dims,
                         dtype=torch.int64).cpu().tolist()
        one = torch.sum(block == 1, dim=reduction_dims,
                        dtype=torch.int64).cpu().tolist()
        finite_count = torch.sum(finite, dim=reduction_dims,
                                 dtype=torch.int64).cpu().tolist()
        integer = torch.sum(torch.logical_and(finite, block == torch.trunc(block)),
                            dim=reduction_dims,
                            dtype=torch.int64).cpu().tolist()
        for offset in range(end - begin):
            binary = int(zero[offset]) + int(one[offset])
            finite_item = int(finite_count[offset])
            per_channel.append({
                "channel": begin + offset,
                "elements": elements_per_channel,
                "zero_count": int(zero[offset]),
                "one_count": int(one[offset]),
                "exact_binary_count": binary,
                "nonbinary_finite_count": finite_item - binary,
                "nonfinite_count": elements_per_channel - finite_item,
                "integer_count": int(integer[offset]),
                "all_exact_binary": binary == elements_per_channel,
                "all_finite": finite_item == elements_per_channel,
                "all_integer": int(integer[offset]) == elements_per_channel,
            })
    return per_channel


def aggregate_channel_rows(rows):
    keys = ("elements", "zero_count", "one_count", "exact_binary_count",
            "nonbinary_finite_count", "nonfinite_count", "integer_count")
    result = {key: sum(int(row[key]) for row in rows) for key in keys}
    result.update({
        "channels": len(rows),
        "all_exact_binary": result["exact_binary_count"] == result["elements"],
        "all_finite": result["nonfinite_count"] == 0,
        "all_integer": result["integer_count"] == result["elements"],
    })
    return result


def safe_analog_channel_stats(tensor, channel):
    """Bounded CPU float64 statistics; JSON output never contains NaN/Inf."""
    values = tensor.detach()[:, :, channel:channel + 1, :, :].to(
        device="cpu", dtype=torch.float64).contiguous().numpy().reshape(-1)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    nan_count = int(np.isnan(values).sum(dtype=np.int64))
    posinf_count = int(np.isposinf(values).sum(dtype=np.int64))
    neginf_count = int(np.isneginf(values).sum(dtype=np.int64))
    if finite.size:
        finite_sum = float(np.sum(finite, dtype=np.float64))
        finite_abs_sum = float(np.sum(np.abs(finite), dtype=np.float64))
        finite_square_sum = float(np.sum(np.square(finite), dtype=np.float64))
        mean = finite_sum / int(finite.size)
        mean_abs = finite_abs_sum / int(finite.size)
        rms = math.sqrt(finite_square_sum / int(finite.size))
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
    else:
        finite_sum = finite_abs_sum = finite_square_sum = None
        mean = mean_abs = rms = minimum = maximum = None
    return {
        "channel": int(channel),
        "elements": int(values.size),
        "accumulator_dtype": "float64",
        "nonfinite_policy": "EXCLUDED_FROM_AGGREGATES_AND_COUNTED_SEPARATELY",
        "finite_count": int(finite.size),
        "nan_count": nan_count,
        "positive_infinity_count": posinf_count,
        "negative_infinity_count": neginf_count,
        "nonfinite_count": nan_count + posinf_count + neginf_count,
        "exact_zero_count": int(np.equal(values, 0).sum(dtype=np.int64)),
        "exact_one_count": int(np.equal(values, 1).sum(dtype=np.int64)),
        "exact_integer_count": int(np.logical_and(
            finite_mask, np.equal(values, np.trunc(values))).sum(dtype=np.int64)),
        "finite_min": minimum,
        "finite_max": maximum,
        "finite_sum": finite_sum,
        "finite_abs_sum": finite_abs_sum,
        "finite_square_sum": finite_square_sum,
        "finite_mean": mean,
        "finite_mean_abs": mean_abs,
        "finite_rms": rms,
    }


def audit_decoder_input(tensor, expected, channel_block):
    shape = [int(item) for item in tensor.shape]
    require(shape == expected["input_shape"],
            "M649 hook input shape drift: " + expected["name"])
    per_channel = reduce_counts(tensor, channel_block)
    channels = shape[2]
    full = aggregate_channel_rows(per_channel)
    result = {
        "shape": shape,
        "dtype": str(tensor.dtype),
        "device_type": tensor.device.type,
        "is_contiguous": bool(tensor.is_contiguous()),
        "stride": [int(item) for item in tensor.stride()],
        "channel_axis": 2,
        "full_tensor": full,
        "per_channel_exactness": per_channel,
    }
    if expected["module_index"] == 0:
        result["typed_partition"] = {
            "hypothesis": "D0_ALL_CHANNELS_BINARY",
            "binary": full,
            "gate_pass": bool(full["all_exact_binary"]),
        }
    else:
        first2 = per_channel[:2]
        suffix = per_channel[2:]
        prefix = per_channel[:-2]
        last2 = per_channel[-2:]
        analog_first2 = [safe_analog_channel_stats(tensor, index)
                         for index in (0, 1)]
        analog_last2 = [safe_analog_channel_stats(tensor, index)
                        for index in (channels - 2, channels - 1)]
        first2_summary = aggregate_channel_rows(first2)
        suffix_summary = aggregate_channel_rows(suffix)
        prefix_summary = aggregate_channel_rows(prefix)
        last2_summary = aggregate_channel_rows(last2)
        first2_analog = (first2_summary["all_finite"] and
                         first2_summary["nonbinary_finite_count"] > 0)
        last2_analog = (last2_summary["all_finite"] and
                        last2_summary["nonbinary_finite_count"] > 0)
        result["typed_partition"] = {
            "source_order_expectation": {
                "flow_channels": [0, 1],
                "reason": "skip_concat(predictions[-1], x, dim=2) is torch.cat([x1, x2], dim)",
            },
            "first2_flow_hypothesis": {
                "flow_channel_indices": [0, 1],
                "flow_summary": first2_summary,
                "flow_channel_safe_stats": analog_first2,
                "binary_suffix_range": [2, channels],
                "binary_suffix_summary": suffix_summary,
                "flow_is_finite_and_observably_nonbinary": first2_analog,
                "gate_pass": bool(first2_analog and
                                  suffix_summary["all_exact_binary"]),
            },
            "last2_flow_hypothesis": {
                "binary_prefix_range": [0, channels - 2],
                "binary_prefix_summary": prefix_summary,
                "flow_channel_indices": [channels - 2, channels - 1],
                "flow_summary": last2_summary,
                "flow_channel_safe_stats": analog_last2,
                "flow_is_finite_and_observably_nonbinary": last2_analog,
                "gate_pass": bool(prefix_summary["all_exact_binary"] and
                                  last2_analog),
            },
        }
    return result


def typed_split_decision(records, expected_records=40):
    checks = []
    checks.append({
        "id": "POPULATION_10X4",
        "pass": len(records) == expected_records,
        "observed": len(records),
        "expected": expected_records,
    })
    by_module = {index: [] for index in range(4)}
    for record in records:
        index = int(record["module_index"])
        require(index in by_module, "M649 unexpected module index")
        by_module[index].append(record)
        checks.append({
            "id": "S{:02d}_D{}_DTYPE_FLOAT32".format(
                int(record["sample_id"]), index),
            "pass": record["input_numeric"]["dtype"] == "torch.float32",
        })
        if index == 0:
            passed = record["input_numeric"]["typed_partition"]["gate_pass"]
            check_id = "S{:02d}_D0_ALL_BINARY".format(
                int(record["sample_id"]))
        else:
            passed = record["input_numeric"]["typed_partition"][
                "first2_flow_hypothesis"]["gate_pass"]
            check_id = "S{:02d}_D{}_FIRST2_FLOW_SUFFIX_BINARY".format(
                int(record["sample_id"]), index)
        checks.append({"id": check_id, "pass": bool(passed)})
    for index in range(4):
        checks.append({
            "id": "D{}_S10_POPULATION".format(index),
            "pass": len(by_module[index]) == 10,
            "observed": len(by_module[index]),
            "expected": 10,
        })
    last2_passes = []
    first2_passes = []
    for index in (1, 2, 3):
        first2_nonbinary = sum(
            row["input_numeric"]["typed_partition"][
                "first2_flow_hypothesis"]["flow_summary"][
                    "nonbinary_finite_count"] for row in by_module[index])
        last2_nonbinary = sum(
            row["input_numeric"]["typed_partition"][
                "last2_flow_hypothesis"]["flow_summary"][
                    "nonbinary_finite_count"] for row in by_module[index])
        first2_pass = (len(by_module[index]) == 10 and first2_nonbinary > 0 and
                       all(row["input_numeric"]["typed_partition"][
                           "first2_flow_hypothesis"]["gate_pass"]
                           for row in by_module[index]))
        last2_pass = (len(by_module[index]) == 10 and last2_nonbinary > 0 and
                      all(row["input_numeric"]["typed_partition"][
                          "last2_flow_hypothesis"]["gate_pass"]
                          for row in by_module[index]))
        first2_passes.append(first2_pass)
        last2_passes.append(last2_pass)
        checks.extend([
            {"id": "D{}_S10_FIRST2_FLOW_TYPED_SPLIT".format(index),
             "pass": first2_pass,
             "observed_nonbinary_finite": first2_nonbinary},
            {"id": "D{}_S10_LAST2_FLOW_HYPOTHESIS".format(index),
             "pass": last2_pass,
             "observed_nonbinary_finite": last2_nonbinary,
             "admission_role": "DIAGNOSTIC_ONLY_NOT_SOURCE_EXPECTED"},
        ])
    all_hard = all(item["pass"] for item in checks
                   if item.get("admission_role") !=
                   "DIAGNOSTIC_ONLY_NOT_SOURCE_EXPECTED")
    return {
        "status": ("GO_EXACT_TYPED_SPLIT__D0_BINARY__D1_D3_FIRST2_FLOW_SUFFIX_BINARY"
                   if all_hard else "NO_GO_EXACT_TYPED_SPLIT"),
        "typed_split_authorized": bool(all_hard),
        "authorized_layout": ({
            "d0": "all channels exact binary",
            "d1_d3": "channels [0,2) finite analog flow; channels [2,C) exact binary",
        } if all_hard else None),
        "last2_flow_hypothesis_all_modules_pass": all(last2_passes),
        "first2_flow_hypothesis_all_modules_pass": all(first2_passes),
        "checks": checks,
    }


def take_exact(iterable, count):
    """Yield exactly ``count`` items without probing item count+1."""
    require(count >= 0, "M649 invalid exact iteration count")
    iterator = iter(iterable)
    for index in range(count):
        try:
            yield next(iterator)
        except StopIteration:
            raise RuntimeError(
                "M649 data loader exhausted before item {} of {}".format(
                    index + 1, count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--m511-contract", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--channel-block", type=int, default=64)
    args = parser.parse_args()

    require(args.samples == 10 and args.num_workers == 0 and
            args.channel_block == 64,
            "M649 requires S10, num-workers=0, channel-block=64")
    launcher_path = checked_path(Path(__file__), label="runtime launcher")
    contract_path = checked_path(args.contract, label="runtime contract")
    contract_start = sha256(contract_path)
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m649_h67_ep35_convtranspose_typed_numeric_audit_contract_v1" and
            contract.get("status") ==
            "STATIC_AUTHOR_HANDOFF_ONLY__FRESH_HAMMER_REQUIRED_BEFORE_GPU",
            "M649 contract schema/status drift")
    identities = verify_contract_inputs(contract, launcher_path)
    failed_state = verify_failed_m511_state(contract)
    prior_m649_failure = verify_prior_failed_m649_state(contract)

    expected_output_raw = ROOT / contract["output"]["canonical_directory"]
    output = checked_path_match(
        args.output_dir, expected_output_raw, allow_missing_leaf=True,
        label="output")
    expected_output = checked_path(
        expected_output_raw, allow_missing_leaf=True,
        label="expected output")
    require(output.parent.is_dir(), "M649 output parent missing")
    require(not os.path.lexists(str(output)), "M649 output already exists")
    m511_contract_path = checked_path_match(
        args.m511_contract,
        ROOT / contract["inputs"]["m511_contract"]["path"],
        label="M511 contract")
    require(
            sha256(m511_contract_path) == M511_CONTRACT_SHA256,
            "M649 runtime M511 contract drift")
    m511_contract = strict_json(m511_contract_path)
    config_path = checked_path_match(
        args.config, ROOT / contract["inputs"]["config"]["path"],
        label="config")
    checkpoint_path = checked_path_match(
        args.checkpoint, ROOT / contract["inputs"]["checkpoint"]["path"],
        label="checkpoint")
    require(checkpoint_path.stat().st_size ==
            m511_contract["checkpoint_identity"]["size_bytes"],
            "M649 checkpoint size drift")

    producer_path = checked_contract_path(
        contract["inputs"]["m511_producer"]["path"],
        label="input m511_producer")
    m511 = load_frozen_m511(producer_path)
    require(m511.strict_json(m511_contract_path) == m511_contract,
            "M649/M511 strict contract parse disagreement")
    frozen_m511_inputs = m511.verify_inputs(
        m511_contract, M511_PRODUCER_SHA256)

    staging = Path(tempfile.mkdtemp(
        prefix=output.name + ".staging.", dir=str(output.parent)))
    records = []
    handles = []
    current = {"sample_id": None, "order": 0}
    published = False
    quarantine = None
    try:
        config, device = m511.profile.load_config(config_path)
        require(torch.cuda.is_available() and torch.device(device).type == "cuda",
                "M649 requires an available CUDA device")
        expected_protocol = m511_contract["eval_protocol"]
        require(config["model"]["name"] == "MS_SpikingformerFlowNet_en4" and
                config["model"]["use_upsample_conv"] is False and
                int(config["model"]["kernel_size"]) == 3,
                "M649 H67 decoder config drift")
        dataset = m511.profile.DSECDatasetLite(
            config, file_list="valid", stereo=False,
            scale_factor=config.get("test", {}).get("scale_factor", 1))
        sequence_file = Path(dataset.sequence_file).resolve()
        sequence_identity = {
            "path": str(sequence_file), "bytes": sequence_file.stat().st_size,
            "sha256": sha256(sequence_file),
        }
        sample_sources = [m511.sample_source_identity(dataset, sample_id)
                          for sample_id in range(10)]
        require([row["sample_key"] for row in sample_sources] ==
                [row["sample_key"] for row in m511_contract["samples"]],
                "M649 raw source/sample cohort drift")
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
                "M649 checkpoint load is not exact")
        module_counts = m511.profile.h9_module_counts(model)
        bn_policy = config.get("test", {}).get("bn_policy", "running")
        bn_changed = m511.profile.configure_batch_norm_evaluation(model, bn_policy)
        observed_protocol = {
            "resolution": list(config["loader"]["resolution"]),
            "crop": config["loader"].get("crop"),
            "window_size": list(config["swin_transformer"]["window_size"]),
            "pretrained_window_size": config["swin_transformer"].get(
                "pretrained_window_size"),
            "tokens_per_window": int(np.prod(
                config["swin_transformer"]["window_size"])),
            "remap": config["loader"].get("remap"),
            "bn_policy": bn_policy,
            "bn_modules_changed": bn_changed,
            "eval_batch_size": 1,
            "num_workers": args.num_workers,
            "module_counts": module_counts,
        }
        require(observed_protocol == expected_protocol,
                "M649 frozen evaluation protocol mismatch")
        module_identities = m511.module_identities(
            model, m511_contract["modules"])
        named = dict(model.named_modules())
        require([name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.ConvTranspose2d)] ==
                [item["name"] for item in m511_contract["modules"]],
                "M649 complete ConvTranspose2d module set drift")

        def make_hook(expected):
            def hook(module, inputs, output_tensor):
                require(current["sample_id"] is not None and
                        current["order"] == expected["module_index"],
                        "M649 decoder call order drift")
                require(isinstance(inputs, tuple) and len(inputs) == 1 and
                        torch.is_tensor(inputs[0]) and
                        torch.is_tensor(output_tensor),
                        "M649 hook tensor arity drift")
                require([int(item) for item in output_tensor.shape] ==
                        expected["output_shape"],
                        "M649 hook output shape drift")
                numeric = audit_decoder_input(
                    inputs[0], expected, args.channel_block)
                records.append({
                    "sample_id": current["sample_id"],
                    "sample_key": m511_contract["samples"][
                        current["sample_id"]]["sample_key"],
                    "sequence_key": m511_contract["samples"][
                        current["sample_id"]]["sequence_key"],
                    "module_index": expected["module_index"],
                    "name": expected["name"],
                    "operator": "ConvTranspose2d",
                    "input_numeric": numeric,
                    "output_shape": [int(item) for item in output_tensor.shape],
                    "output_dtype": str(output_tensor.dtype),
                })
                current["order"] += 1
            return hook

        for expected in m511_contract["modules"]:
            handles.append(named[expected["name"]].register_forward_hook(
                make_hook(expected)))

        sync_counts = {"before_audit": 0, "per_sample_post_forward": 0,
                       "final_pre_result": 0}
        torch.cuda.synchronize(device)
        sync_counts["before_audit"] += 1
        processed = 0
        with torch.no_grad():
            for chunk, mask, label in take_exact(loader, args.samples):
                functional.reset_net(model)
                sample_key, sequence_key = m511.sample_identity(dataset, processed)
                require(m511_contract["samples"][processed] == {
                    "sample_id": processed, "sample_key": sample_key,
                    "sequence_key": sequence_key,
                }, "M649 sample identity drift")
                current.update({"sample_id": processed, "order": 0})
                x, _label, _mask = m511.profile.preprocess_chunk(
                    config, chunk, label, mask, transform_valid, device)
                model(x)
                torch.cuda.synchronize(device)
                sync_counts["per_sample_post_forward"] += 1
                require(current["order"] == len(m511_contract["modules"]),
                        "M649 missing decoder call")
                current.update({"sample_id": None, "order": 0})
                processed += 1
                print("[M649] audited sample {}/10".format(processed),
                      flush=True)
        require(processed == 10 and len(records) == 40,
                "M649 audit population drift")
        torch.cuda.synchronize(device)
        sync_counts["final_pre_result"] += 1
        while handles:
            handles.pop().remove()

        decision = typed_split_decision(records, expected_records=40)
        require(sha256(launcher_path) ==
                contract["inputs"]["launcher"]["sha256"],
                "M649 launcher mutated during audit")
        require(sha256(contract_path) == contract_start,
                "M649 contract mutated during audit")
        verify_contract_inputs(contract, launcher_path)
        verify_failed_m511_state(contract)
        verify_prior_failed_m649_state(contract)
        m511.verify_inputs(m511_contract, M511_PRODUCER_SHA256)
        m511.rehash_sample_sources(
            sequence_file, sequence_identity, sample_sources)

        result = {
            "schema": "m649_h67_ep35_convtranspose_typed_numeric_audit_v1",
            "status": "PASS_NUMERIC_AUDIT__" + decision["status"],
            "identity": {
                "contract": {"path": str(contract_path),
                             "sha256": contract_start},
                "inputs": identities,
                "frozen_m511_inputs": frozen_m511_inputs,
                "checkpoint_load_audit": load_audit,
                "failed_m511_state": failed_state,
                "prior_failed_m649_execution": prior_m649_failure,
            },
            "eval_protocol": observed_protocol,
            "raw_validation_sources": {
                "sequence_list": sequence_identity,
                "samples": sample_sources,
            },
            "module_identities": module_identities,
            "cuda_synchronization": sync_counts,
            "population": {"samples": 10, "modules": 4, "records": 40},
            "numeric_semantics": {
                "comparison": "EXACT_FLOAT_EQUALITY_NO_THRESHOLD_NO_COERCION",
                "layout": "T_B_C_H_W_CHANNEL_AXIS_2",
                "safe_analog_aggregates":
                    "FINITE_VALUES_ONLY_FLOAT64__NONFINITE_COUNTED_SEPARATELY",
                "source_order_evidence": {
                    "spiking_stswinnet_sha256": identities[
                        "spiking_stswinnet"]["sha256"],
                    "model_util_sha256": identities["model_util"]["sha256"],
                    "decoder_concat":
                        "skip_ftn(predictions[-1], x, dim=2)",
                    "skip_concat": "torch.cat([x1, x2], dim=dim)",
                    "expected_flow_channels_d1_d3": [0, 1],
                },
            },
            "decision": decision,
            "records": records,
            "claim_boundary": {
                "numeric_audit": True,
                "typed_split_authorized": decision["typed_split_authorized"],
                "activation_payload": False,
                "cycles": False,
                "speedup": False,
                "rtl": False,
                "vcs": False,
                "synopsys": False,
                "energy": False,
                "ppa": False,
                "system_speedup": False,
                "date_headline": False,
            },
        }
        result_path = staging / "m649_typed_numeric_audit.json"
        result_path.write_text(json.dumps(
            result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M649_NUMERIC_AUDIT__{}\n".format(decision["status"]),
            encoding="utf-8")
        write_double_seal(staging)
        verify_own_double_seal(staging)
        require(not os.path.lexists(str(output)),
                "M649 output appeared during audit")
        os.replace(str(staging), str(output))
        published = True
        verify_own_double_seal(output)
        print("PASS M649 {} {}".format(
            result_path.name, decision["status"]), flush=True)
    except BaseException as error:
        failure_root = staging
        failure_name = "FAILED.json"
        if published:
            quarantine = output.with_name(
                output.name + ".quarantine.failed.{}.{}".format(
                    os.getpid(), uuid.uuid4().hex))
            quarantine = checked_path(
                quarantine, allow_missing_leaf=True,
                label="post-publication quarantine")
            require(not os.path.lexists(str(quarantine)),
                    "M649 post-publication quarantine target exists")
            os.replace(str(output), str(quarantine))
            require(not os.path.lexists(str(output)) and quarantine.is_dir(),
                    "M649 failed to quarantine post-publication output")
            failure_root = quarantine
            failure_name = "FAILED_POSTPUBLICATION.json"
        failure = failure_root / failure_name
        if not failure.exists():
            failure.write_text(json.dumps({
                "schema": "m649_h67_ep35_convtranspose_typed_numeric_failure_v1",
                "status": "FAIL_CLOSED_NO_RESULT",
                "reason": "{}: {}".format(type(error).__name__, error),
                "completed_records": len(records),
                "staging_directory": str(staging),
                "original_m511_staging_preserved": True,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
