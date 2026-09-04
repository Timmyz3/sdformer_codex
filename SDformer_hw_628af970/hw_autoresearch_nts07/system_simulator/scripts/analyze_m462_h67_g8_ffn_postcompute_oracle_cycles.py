#!/usr/bin/env python3
"""Fail-closed M462 cycle-opportunity audit for the M460R5 G8 capture.

This analyzer deliberately reports a post-compute oracle, not an executable
skip and not a measured system speedup.  It keeps two non-interchangeable
masks:

* token: strict [T,N,H,W] eligibility, charging only FC1/FC2 opportunity;
* site: eligibility must hold at all T=10 steps of one [N,H,W] site before
  FC1/FC2 and the two FFN-local ATLIF services may all be charged.

The frozen profile100 Linear baselines are normalized independently for every
FFN pair and role with integer floor: saved=(B*S)//D.  There is no global or
floating-point activity-ratio shortcut.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
LANES = 96
SAMPLES = 10
TIMESTEPS = 10
PAIRS = 12
RECORDS = SAMPLES * PAIRS
TOKENS = 5_580_000
SITES = TOKENS // TIMESTEPS
ENVELOPE = 620_302_905
FC1_BASELINE = 118_370_114
FC2_BASELINE = 41_413_997
LINEAR_BASELINE = 159_784_111
SN1_ATLIF_BASELINE = 9_120_000
SN2_ATLIF_BASELINE = 36_480_000
ATLIF_BASELINE = 45_600_000
FFN_ACCOUNTED = 205_384_111
GATES = (("1.15", 80_909_075), ("1.20", 103_383_818),
         ("1.30", 143_146_825))
TAU_GRID = (
    ("zero_exact", 0.0),
    ("2^-16", 2.0 ** -16),
    ("2^-14", 2.0 ** -14),
    ("2^-12", 2.0 ** -12),
    ("2^-10", 2.0 ** -10),
    ("2^-8", 2.0 ** -8),
    ("2^-6", 2.0 ** -6),
)
ARRAY_DTYPES = {
    "x_l1": "<f8", "x_l2_sq": "<f8", "x_linf": "<f4",
    "sn1_nnz": "<i4", "sn2_nnz": "<i4", "pre_bn2_l1": "<f8",
    "f_exact_zero": "|b1", "f_l1": "<f8", "f_l2_sq": "<f8",
    "f_linf": "<f4", "finite": "|b1", "rho": "<f8",
}
PAIR_RE = re.compile(
    r"^sttmultires_unet\.encoders\.swin3d\.layers\.(\d+)\."
    r"swin_blocks\.(\d+)\.mlp$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


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
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def finite_float(value, label):
    result = float(value)
    require(math.isfinite(result), label + " is non-finite")
    return result


def logical_array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode(
        "ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def array_receipt(array):
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": [int(item) for item in value.shape],
        "elements": int(value.size),
        "bytes": int(value.nbytes),
        "logical_sha256": logical_array_sha256(value),
    }


def manifest_entries(directory, manifest_name):
    """Verify a sha256 manifest without accepting aliases or path escape."""
    directory = Path(directory).resolve()
    manifest = directory / manifest_name
    require(manifest.is_file() and not manifest.is_symlink(),
            "manifest absent/symlink: " + str(manifest))
    entries = []
    names = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            require(line and raw.endswith("\n"),
                    "manifest blank/unterminated line {}".format(number))
            parts = line.split("  ")
            require(len(parts) == 2 and HEX64.match(parts[0]) is not None,
                    "malformed manifest line {}".format(number))
            expected, relative = parts
            posix = PurePosixPath(relative)
            require(not posix.is_absolute() and relative == str(posix) and
                    "\\" not in relative and ".." not in posix.parts and
                    "." not in posix.parts and relative not in names,
                    "manifest unsafe/duplicate path: " + relative)
            names.add(relative)
            leaf = directory.joinpath(*posix.parts)
            require(leaf.is_file() and not leaf.is_symlink(),
                    "manifest leaf absent/symlink: " + relative)
            require(sha256(leaf) == expected,
                    "manifest leaf SHA drift: " + relative)
            entries.append(relative)
    require(entries, "empty manifest: " + manifest_name)
    return entries


def verify_double_seal(directory, manifest_name="manifest.sha256",
                       outer_name="manifest.sha256.outer.seal.sha256",
                       exact_leaf_population=True):
    directory = Path(directory).resolve()
    leaves = manifest_entries(directory, manifest_name)
    outer = manifest_entries(directory, outer_name)
    require(outer == [manifest_name], "outer seal must bind only inner manifest")
    if exact_leaf_population:
        actual = set()
        for path in directory.rglob("*"):
            if path.is_file():
                require(not path.is_symlink(), "sealed tree symlink: " + str(path))
                relative = path.relative_to(directory).as_posix()
                if relative not in (manifest_name, outer_name):
                    actual.add(relative)
        require(actual == set(leaves), "sealed tree leaf population drift")
    return leaves


def resolve_identity(record):
    root = record["root"]
    require(root in ("code_repo", "code_hw"), "unsupported identity root")
    base = ROOT if root == "code_repo" else HW
    return (base / record["path"]).resolve()


def validate_contract(path):
    contract = strict_json(path)
    require(contract.get("schema") ==
            "m462_h67_g8_postcompute_oracle_cycle_audit_contract_v1",
            "M462 contract schema drift")
    require(contract.get("status") == "READY_EXACT_SHA_CPU_ORACLE_AUDIT",
            "M462 contract status drift")
    identities = {}
    for name, record in contract["identity"].items():
        target = resolve_identity(record)
        require(target.is_file() and not target.is_symlink(),
                "M462 identity absent/symlink: " + name)
        actual = sha256(target)
        require(actual == record["sha256"], "M462 identity SHA drift: " + name)
        identities[name] = target
    require(sha256(Path(__file__).resolve()) ==
            contract["identity"]["analyzer"]["sha256"],
            "M462 analyzer self SHA drift")
    binding = contract["capture_binding"]
    for key in ("top_manifest_sha256", "top_outer_seal_file_sha256",
                "capture_manifest_sha256", "capture_outer_seal_file_sha256"):
        require(HEX64.match(binding[key]) is not None,
                "M462 malformed capture binding: " + key)
    require(contract["cycle_model"] == {
        "lanes": LANES,
        "samples": SAMPLES,
        "timesteps": TIMESTEPS,
        "global_envelope_cycles": ENVELOPE,
        "profile100_fc1_cycles": FC1_BASELINE,
        "profile100_fc2_cycles": FC2_BASELINE,
        "ffn_local_sn1_atlif_cycles": SN1_ATLIF_BASELINE,
        "ffn_local_sn2_atlif_cycles": SN2_ATLIF_BASELINE,
        "profile_normalization": "per_pair_per_role_integer_floor_B_times_S_div_D",
    }, "M462 cycle model contract drift")
    admission = contract["admission"]
    for key in ("executable_skip", "delta_aee", "valid825_accuracy",
                "measured_cycle_speedup", "system_speedup", "energy",
                "ppa", "headline"):
        require(admission[key] is False, "M462 forbidden admission true: " + key)
    return contract, identities


def load_ffn_ledger(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == PAIRS, "FFN ledger pair population drift")
    result = {}
    for row in rows:
        pair = row["pair_id"]
        match = PAIR_RE.match(pair)
        require(match is not None and pair not in result, "FFN pair identity drift")
        stage, block = (int(match.group(1)), int(match.group(2)))
        channels = int(row["input_channels"])
        require(channels % LANES == 0 and
                int(row["expanded_channels"]) == 4 * channels and
                int(row["output_channels"]) == channels,
                "FFN ledger geometry drift: " + pair)
        result[pair] = {
            "stage": stage, "block": block, "channels": channels,
            "height": int(row["height"]), "width": int(row["width"]),
            "fc1_baseline": int(row["fc1_cycles_model"]),
            "fc2_baseline": int(row["fc2_cycles_model"]),
        }
    require(sum(row["fc1_baseline"] for row in result.values()) == FC1_BASELINE,
            "profile100 FC1 baseline drift")
    require(sum(row["fc2_baseline"] for row in result.values()) == FC2_BASELINE,
            "profile100 FC2 baseline drift")
    return result


def load_operator_runtime(path, pairs):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row["name"]: row for row in rows}
    result = {}
    for pair in pairs:
        for role in ("fc1", "fc2"):
            name = pair + "." + role
            require(name in by_name, "operator_runtime FFN role absent: " + name)
            row = by_name[name]
            require(row["operator"] == "Linear" and int(row["calls"]) == SAMPLES,
                    "operator_runtime role/call drift: " + name)
            active = int(row["input_active"])
            require(active > 0, "operator_runtime input_active nonpositive: " + name)
            result[(pair, role)] = active
    require(len(result) == 2 * PAIRS, "operator_runtime role population drift")
    return result


def masks_for_tau(arrays, tau):
    finite = arrays["finite"]
    if tau == 0.0:
        strict = finite & arrays["f_exact_zero"]
        equal = strict.copy()
        inclusive = strict.copy()
    else:
        strict = finite & (arrays["rho"] < tau)
        equal = finite & (arrays["rho"] == tau)
        inclusive = strict | equal
    return strict, equal, inclusive


def all_t_site_mask(token_mask):
    require(token_mask.ndim == 4 and token_mask.shape[0] == TIMESTEPS,
            "site mask requires literal [T=10,N,H,W]")
    return np.all(token_mask, axis=0)


def normalize_profile_cycles(baseline, selected_issue, denominator_issue):
    baseline = int(baseline)
    selected_issue = int(selected_issue)
    denominator_issue = int(denominator_issue)
    require(baseline >= 0 and denominator_issue > 0 and
            0 <= selected_issue <= denominator_issue,
            "invalid profile normalization operands")
    return (baseline * selected_issue) // denominator_issue


def validate_r5_root(root, contract, identities):
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), "M460R5 root absent/symlink")
    binding = contract["capture_binding"]
    require(sha256(root / "manifest.sha256") ==
            binding["top_manifest_sha256"], "R5 top manifest binding drift")
    require(sha256(root / "manifest.sha256.outer.seal.sha256") ==
            binding["top_outer_seal_file_sha256"], "R5 top outer binding drift")
    verify_double_seal(root)
    capture = root / "capture_payload"
    require(sha256(capture / "manifest.sha256") ==
            binding["capture_manifest_sha256"], "R5 payload manifest binding drift")
    require(sha256(capture / "manifest.sha256.outer.seal.sha256") ==
            binding["capture_outer_seal_file_sha256"],
            "R5 payload outer binding drift")
    leaves = verify_double_seal(capture)
    npz_names = sorted(path.name for path in capture.glob("*.npz"))
    expected_capture = set(npz_names + [
        "samples.csv", "per_sample_module_manifest.json",
        "m460_h67_g8_ffn_token_residual_s10_capture.json"])
    require(len(npz_names) == RECORDS and set(leaves) == expected_capture,
            "R5 payload sealed population drift")

    author = strict_json(root / "m460r5_one_shot_capture_author_receipt.json")
    require(author.get("schema") == "m460r5_one_shot_capture_author_receipt_v1" and
            author.get("status") ==
            "PASS_M460R5_ONE_SHOT_S10_POSTCOMPUTE_ORACLE_CAPTURE",
            "R5 author receipt schema/status drift")
    require(author["contract_sha256"] == sha256(identities["m460r5_contract"]),
            "R5 author/contract binding drift")
    require(author["launch_outer_seal_sha256"] ==
            sha256(identities["m460r5_launch_outer_seal"]),
            "R5 author/launch binding drift")
    summary_path = capture / "m460_h67_g8_ffn_token_residual_s10_capture.json"
    require(author["capture_summary_sha256"] == sha256(summary_path) and
            author["capture_inner_manifest_sha256"] ==
            sha256(capture / "manifest.sha256") and
            author["capture_outer_seal_file_sha256"] ==
            sha256(capture / "manifest.sha256.outer.seal.sha256"),
            "R5 author/payload binding drift")
    require(author["postcompute_oracle_only"] is True and
            author["one_shot_attempts_consumed"] == 1 and
            author["reduction_npz"] == RECORDS,
            "R5 author opportunity/population drift")
    for key in ("executable_skip", "delta_aee", "cycle_speedup", "energy",
                "ppa", "system_speedup", "headline", "training"):
        require(author[key] is False, "R5 author forbidden admission true: " + key)

    summary = strict_json(summary_path)
    require(summary.get("schema") == "m460r5_h67_g8_one_shot_capture_v1" and
            summary.get("status") ==
            "PASS_M460R5_H67_EP35_NO_RUNNING_S10_ONE_SHOT_POSTCOMPUTE_ORACLE",
            "R5 summary schema/status drift")
    pop = summary["population"]
    require((int(pop["samples"]), int(pop["ffn_modules"]),
             int(pop["sample_module_records"]), int(pop["tokens"]),
             int(pop["expected_tokens"])) ==
            (SAMPLES, PAIRS, RECORDS, TOKENS, TOKENS) and
            pop["sequence_keys"] == ["zurich_city_09_a"],
            "R5 summary population drift")
    audit = summary["identity"]["checkpoint_load_audit"]
    require(int(audit["missing_count"]) == 0 and
            int(audit["unexpected_count"]) == 0 and
            summary["identity"]["capture_bn_policy"] ==
            "no_running/current-batch",
            "R5 checkpoint/BN identity drift")
    require(summary["strict_runtime_state_machine"]["order"] ==
            ["pre", "sn1", "sn2", "fc2", "full_output"] and
            summary["strict_runtime_state_machine"][
                "sn2_fc2_sn1_attack_accepted"] is False,
            "R5 state-machine receipt drift")
    return capture, summary, author


def read_samples(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == SAMPLES and
            [int(row["sample_id"]) for row in rows] == list(range(SAMPLES)) and
            all(row["sequence_key"] == "zurich_city_09_a" for row in rows),
            "R5 samples.csv S10 identity drift")
    return {int(row["sample_id"]): row for row in rows}


def validate_npz_structure(path):
    with zipfile.ZipFile(str(path), "r") as archive:
        names = archive.namelist()
    expected = sorted(name + ".npy" for name in ARRAY_DTYPES)
    require(len(names) == len(set(names)) and sorted(names) == expected,
            "NPZ member population/duplicate drift: " + path.name)


def validate_tau_receipt(row, arrays, channels):
    tau = finite_float(row["tau"], "tau")
    strict, equal, inclusive = masks_for_tau(arrays, tau)
    expected_rule = ("numeric_exact_zero_and_finite" if tau == 0.0 else
                     "finite_and_rho_strictly_less_than_tau")
    require(row["rule"] == expected_rule and
            int(row["strict_skip_tokens"]) == int(np.count_nonzero(strict)) and
            int(row["equal_boundary_tokens"]) == int(np.count_nonzero(equal)) and
            int(row["inclusive_skip_tokens"]) == int(np.count_nonzero(inclusive)),
            "R5 tau count/rule receipt drift: " + row["tau_name"])
    source = (arrays["sn1_nnz"].astype(np.int64) * (4 * channels) +
              arrays["sn2_nnz"].astype(np.int64) * channels)
    require(int(row["strict_source_work_oracle_saved"]) ==
            int(source[strict].sum()) and
            int(row["strict_dense_mac_oracle_saved"]) ==
            int(np.count_nonzero(strict) * 8 * channels * channels),
            "R5 tau source/dense work receipt drift: " + row["tau_name"])
    l1 = float(arrays["f_l1"][strict].sum())
    l2 = float(arrays["f_l2_sq"][strict].sum())
    linf = float(arrays["f_linf"][strict].max()) if np.any(strict) else 0.0
    require(float(row["strict_selected_f_l1_sum"]) == l1 and
            float(row["strict_selected_f_l2_sq_sum"]) == l2 and
            float(row["strict_selected_f_linf_max"]) == linf,
            "R5 tau norm receipt drift: " + row["tau_name"])
    return strict, equal


def validate_and_accumulate(capture, pairs, runtime, record_rows_path):
    payload = strict_json(capture / "per_sample_module_manifest.json")
    require(payload.get("schema") ==
            "m460_h67_g8_ffn_token_residual_payload_manifest_v1" and
            len(payload.get("records", [])) == RECORDS,
            "R5 logical record manifest drift")
    samples = read_samples(capture / "samples.csv")
    seen = set()
    runtime_nnz = defaultdict(int)
    denominators = defaultdict(int)
    selected_issue = defaultdict(int)
    selected_tokens = defaultdict(int)
    equal_tokens = defaultdict(int)
    selected_sites = defaultdict(int)
    per_record = []
    site_rhos = []
    site_pairs = []
    site_fc1 = []
    site_fc2 = []

    for record in payload["records"]:
        sample_id = int(record["sample_id"])
        pair = record["module"]
        require(pair in pairs and sample_id in samples and
                (sample_id, pair) not in seen,
                "R5 duplicate/unknown sample-module record")
        seen.add((sample_id, pair))
        info = pairs[pair]
        require((int(record["stage"]), int(record["block"]),
                 int(record["channels"])) ==
                (info["stage"], info["block"], info["channels"]),
                "R5 record stage/block/channel drift")
        shape = (TIMESTEPS, 1, info["height"], info["width"])
        require(tuple(int(x) for x in record["token_shape_t_n_h_w"]) == shape and
                int(record["tokens"]) == int(np.prod(shape)) and
                record["sample_key"] == samples[sample_id]["sample_key"] and
                record["sequence_key"] == samples[sample_id]["sequence_key"] and
                record["dynamic_bn_policy"] == "no_running/current-batch" and
                record["residual_boundary"] ==
                "post_bn2_before_parent_sew_add" and
                record["pre_bn2_is_residual"] is False,
                "R5 record identity/semantics drift")
        npz_path = capture / record["npz"]
        require(npz_path.name ==
                "s{:02d}_stage{}_block{}_ffn_metrics.npz".format(
                    sample_id, info["stage"], info["block"]) and
                npz_path.is_file() and not npz_path.is_symlink() and
                sha256(npz_path) == record["npz_sha256"],
                "R5 record NPZ identity drift")
        validate_npz_structure(npz_path)
        with np.load(str(npz_path), allow_pickle=False) as loaded:
            require(set(loaded.files) == set(ARRAY_DTYPES),
                    "R5 NPZ array population drift")
            arrays = {name: np.ascontiguousarray(loaded[name])
                      for name in loaded.files}
        for name, dtype in ARRAY_DTYPES.items():
            value = arrays[name]
            require(value.dtype.str == dtype and tuple(value.shape) == shape and
                    record["arrays"][name] == array_receipt(value),
                    "R5 array receipt/dtype/shape drift: {} {}".format(
                        record["npz"], name))
        require(set(record["arrays"]) == set(ARRAY_DTYPES),
                "R5 logical array receipt population drift")
        require(np.all(arrays["sn1_nnz"] >= 0) and
                np.all(arrays["sn1_nnz"] <= info["channels"]) and
                np.all(arrays["sn2_nnz"] >= 0) and
                np.all(arrays["sn2_nnz"] <= 4 * info["channels"]),
                "R5 source nnz range drift")
        for name in ("x_l1", "x_l2_sq", "x_linf", "pre_bn2_l1",
                     "f_l1", "f_l2_sq", "f_linf", "rho"):
            require(np.all(np.isfinite(arrays[name])) and
                    np.all(arrays[name] >= 0),
                    "R5 norm/rho finite/nonnegative drift: " + name)
        expected_rho = arrays["f_l1"] / np.maximum(
            arrays["x_l1"], 2.0 ** -24)
        require(np.array_equal(expected_rho, arrays["rho"]) and
                np.array_equal(arrays["f_exact_zero"], arrays["f_l1"] == 0) and
                int(record["finite_tokens"]) ==
                int(np.count_nonzero(arrays["finite"])),
                "R5 rho/exact-zero/finite logical drift")

        c = info["channels"]
        coeff1, coeff2 = (4 * c) // LANES, c // LANES
        issue1 = arrays["sn1_nnz"].astype(np.int64) * coeff1
        issue2 = arrays["sn2_nnz"].astype(np.int64) * coeff2
        runtime_nnz[(pair, "fc1")] += int(arrays["sn1_nnz"].sum())
        runtime_nnz[(pair, "fc2")] += int(arrays["sn2_nnz"].sum())
        denominators[(pair, "fc1")] += int(issue1.sum())
        denominators[(pair, "fc2")] += int(issue2.sum())

        require(len(record["tau_grid"]) == len(TAU_GRID),
                "R5 per-record tau population drift")
        for receipt, (tau_name, tau) in zip(record["tau_grid"], TAU_GRID):
            require(receipt["tau_name"] == tau_name and
                    float(receipt["tau"]) == tau,
                    "R5 tau order/value drift")
            strict, equal = validate_tau_receipt(receipt, arrays, c)
            for mode, mask in (("strict_token_tnhw", strict),
                               ("t10_all_spatial_site", np.broadcast_to(
                                   all_t_site_mask(strict)[None, ...], shape))):
                selected_issue[(tau_name, mode, pair, "fc1")] += int(
                    issue1[mask].sum())
                selected_issue[(tau_name, mode, pair, "fc2")] += int(
                    issue2[mask].sum())
                selected_tokens[(tau_name, mode)] += int(np.count_nonzero(mask))
            equal_tokens[tau_name] += int(np.count_nonzero(equal))
            sites = all_t_site_mask(strict)
            selected_sites[tau_name] += int(np.count_nonzero(sites))
            per_record.append({
                "sample_id": sample_id, "pair_id": pair,
                "stage": info["stage"], "block": info["block"],
                "tau_name": tau_name, "tau": tau,
                "strict_token_count": int(np.count_nonzero(strict)),
                "equal_boundary_token_count": int(np.count_nonzero(equal)),
                "t10_all_site_count": int(np.count_nonzero(sites)),
                "token_fc1_issue": int(issue1[strict].sum()),
                "token_fc2_issue": int(issue2[strict].sum()),
                "site_fc1_issue": int(issue1[np.broadcast_to(
                    sites[None, ...], shape)].sum()),
                "site_fc2_issue": int(issue2[np.broadcast_to(
                    sites[None, ...], shape)].sum()),
            })

        finite_site = np.all(arrays["finite"], axis=0)
        max_rho = np.max(arrays["rho"], axis=0)
        max_rho = np.where(finite_site, max_rho, np.inf).reshape(-1)
        site_rhos.append(max_rho)
        site_pairs.append(np.full(max_rho.size, pair, dtype=object))
        site_fc1.append(issue1.sum(axis=0).reshape(-1))
        site_fc2.append(issue2.sum(axis=0).reshape(-1))

    require(len(seen) == RECORDS and
            seen == set((sid, pair) for sid in range(SAMPLES) for pair in pairs),
            "R5 complete sample-module Cartesian population drift")
    for key, expected in runtime.items():
        require(runtime_nnz[key] == expected,
                "R5 NPZ/operator_runtime input_active mismatch: {} {}".format(
                    key, (runtime_nnz[key], expected)))
    require(sum(info["fc1_baseline"] for info in pairs.values()) == FC1_BASELINE and
            sum(info["fc2_baseline"] for info in pairs.values()) == FC2_BASELINE,
            "profile100 baseline sum drift")

    with Path(record_rows_path).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(per_record[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_record)
    site_data = {
        "rho": np.concatenate(site_rhos),
        "pair": np.concatenate(site_pairs),
        "fc1_issue": np.concatenate(site_fc1),
        "fc2_issue": np.concatenate(site_fc2),
    }
    require(site_data["rho"].size == SITES,
            "extended site diagnostic population drift")
    return (denominators, selected_issue, selected_tokens, equal_tokens,
            selected_sites, site_data)


def aggregate_frozen_grid(pairs, denominators, selected_issue,
                          selected_tokens, equal_tokens, selected_sites):
    rows = []
    for tau_name, tau in TAU_GRID:
        for mode in ("strict_token_tnhw", "t10_all_spatial_site"):
            saved1 = 0
            saved2 = 0
            for pair, info in pairs.items():
                saved1 += normalize_profile_cycles(
                    info["fc1_baseline"],
                    selected_issue[(tau_name, mode, pair, "fc1")],
                    denominators[(pair, "fc1")])
                saved2 += normalize_profile_cycles(
                    info["fc2_baseline"],
                    selected_issue[(tau_name, mode, pair, "fc2")],
                    denominators[(pair, "fc2")])
            if mode == "t10_all_spatial_site":
                sn1 = 0
                sn2 = 0
                # selected_sites is global, but ATLIF cost is channel weighted;
                # recompute below from the FC site masks is not recoverable here.
                # The caller replaces these two fields from per-pair site counts.
            else:
                sn1 = sn2 = 0
            rows.append({
                "tau_name": tau_name, "tau": tau, "mask_mode": mode,
                "strict_selected_tokens": selected_tokens[(tau_name, mode)],
                "strict_selected_sites": (selected_sites[tau_name]
                                          if mode == "t10_all_spatial_site" else 0),
                "equal_boundary_tokens": equal_tokens[tau_name],
                "fc1_profile100_oracle_saved_cycles": saved1,
                "fc2_profile100_oracle_saved_cycles": saved2,
                "sn1_atlif_oracle_saved_cycles": sn1,
                "sn2_atlif_oracle_saved_cycles": sn2,
            })
    return rows


def attach_atlif_to_frozen_rows(rows, per_record_path, pairs):
    by_tau_pair_sites = defaultdict(int)
    with Path(per_record_path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_tau_pair_sites[(row["tau_name"], row["pair_id"])] += int(
                row["t10_all_site_count"])
    for row in rows:
        if row["mask_mode"] == "t10_all_spatial_site":
            for pair, info in pairs.items():
                count = by_tau_pair_sites[(row["tau_name"], pair)]
                row["sn1_atlif_oracle_saved_cycles"] += (
                    count * TIMESTEPS * info["channels"] // LANES)
                row["sn2_atlif_oracle_saved_cycles"] += (
                    count * 4 * TIMESTEPS * info["channels"] // LANES)
        total = (row["fc1_profile100_oracle_saved_cycles"] +
                 row["fc2_profile100_oracle_saved_cycles"] +
                 row["sn1_atlif_oracle_saved_cycles"] +
                 row["sn2_atlif_oracle_saved_cycles"])
        require(0 <= total < ENVELOPE, "oracle savings outside envelope")
        row["total_accounted_postcompute_oracle_saved_cycles"] = total
        row["remaining_global_envelope_cycles_if_oracle_were_free"] = ENVELOPE - total
        row["ideal_amdahl_ceiling_ratio_not_speedup"] = ENVELOPE / float(
            ENVELOPE - total)
        row["meets_1p15_opportunity_gate"] = total >= GATES[0][1]
        row["postcompute_oracle_only"] = True
        row["executable_skip"] = False
        row["delta_aee_available"] = False
        row["system_speedup"] = False
    return rows


def full_mask_invariants(pairs, denominators):
    full1 = sum(normalize_profile_cycles(
        info["fc1_baseline"], denominators[(pair, "fc1")],
        denominators[(pair, "fc1")]) for pair, info in pairs.items())
    full2 = sum(normalize_profile_cycles(
        info["fc2_baseline"], denominators[(pair, "fc2")],
        denominators[(pair, "fc2")]) for pair, info in pairs.items())
    sn1 = sum(SAMPLES * info["height"] * info["width"] * TIMESTEPS *
              info["channels"] // LANES for info in pairs.values())
    sn2 = sum(SAMPLES * info["height"] * info["width"] * TIMESTEPS * 4 *
              info["channels"] // LANES for info in pairs.values())
    require((full1, full2, sn1, sn2) ==
            (FC1_BASELINE, FC2_BASELINE, SN1_ATLIF_BASELINE,
             SN2_ATLIF_BASELINE), "M462 full-mask invariant drift")
    return {"fc1": full1, "fc2": full2, "linear": full1 + full2,
            "sn1_atlif": sn1, "sn2_atlif": sn2,
            "ffn_accounted": full1 + full2 + sn1 + sn2}


def site_savings_at_k(k, sorted_data, pairs, denominators):
    saved1 = saved2 = 0
    pair_array = sorted_data["pair"][:k]
    for pair, info in pairs.items():
        mask = pair_array == pair
        issue1 = int(sorted_data["fc1_issue"][:k][mask].sum())
        issue2 = int(sorted_data["fc2_issue"][:k][mask].sum())
        saved1 += normalize_profile_cycles(
            info["fc1_baseline"], issue1, denominators[(pair, "fc1")])
        saved2 += normalize_profile_cycles(
            info["fc2_baseline"], issue2, denominators[(pair, "fc2")])
    channels = sorted_data["channels"][:k]
    sn1 = int((TIMESTEPS * channels // LANES).sum())
    sn2 = int((4 * TIMESTEPS * channels // LANES).sum())
    return saved1, saved2, sn1, sn2


def extended_cliff_diagnostic(site_data, pairs, denominators):
    finite = np.isfinite(site_data["rho"])
    order = np.argsort(site_data["rho"][finite], kind="stable")
    sorted_data = {
        "rho": site_data["rho"][finite][order],
        "pair": site_data["pair"][finite][order],
        "fc1_issue": site_data["fc1_issue"][finite][order],
        "fc2_issue": site_data["fc2_issue"][finite][order],
    }
    sorted_data["channels"] = np.asarray(
        [pairs[pair]["channels"] for pair in sorted_data["pair"]],
        dtype=np.int64)
    rho = sorted_data["rho"]
    require(rho.size > 0 and np.all(rho[:-1] <= rho[1:]),
            "extended rho sort drift")
    ends = np.flatnonzero(np.r_[rho[:-1] != rho[1:], True]) + 1
    rows = []
    for label, required in GATES:
        lo, hi = 0, len(ends) - 1
        found = None
        found_values = None
        while lo <= hi:
            mid = (lo + hi) // 2
            k = int(ends[mid])
            values = site_savings_at_k(k, sorted_data, pairs, denominators)
            total = sum(values)
            if total >= required:
                found, found_values = mid, values
                hi = mid - 1
            else:
                lo = mid + 1
        if found is None:
            rows.append({
                "target_ideal_amdahl_ratio_not_speedup": label,
                "required_oracle_savings": required,
                "reachable": False,
                "postcompute_oracle_only": True,
                "outside_frozen_tau_grid": True,
                "delta_aee_available": False,
                "admitted": False,
            })
            continue
        k = int(ends[found])
        boundary = float(rho[k - 1])
        next_tau = float(np.nextafter(np.float64(boundary), np.float64(np.inf)))
        saved1, saved2, sn1, sn2 = found_values
        total = saved1 + saved2 + sn1 + sn2
        rows.append({
            "target_ideal_amdahl_ratio_not_speedup": label,
            "required_oracle_savings": required,
            "reachable": True,
            "strict_tau_relation": "tau_strictly_greater_than_boundary",
            "boundary_max_t_rho": boundary,
            "smallest_binary64_tau_above_boundary": next_tau,
            "selected_spatial_sites": k,
            "eligible_finite_spatial_sites": int(rho.size),
            "all_spatial_sites": SITES,
            "selected_site_fraction_of_all": k / float(SITES),
            "fc1_profile100_oracle_saved_cycles": saved1,
            "fc2_profile100_oracle_saved_cycles": saved2,
            "sn1_atlif_oracle_saved_cycles": sn1,
            "sn2_atlif_oracle_saved_cycles": sn2,
            "total_accounted_postcompute_oracle_saved_cycles": total,
            "ideal_amdahl_ceiling_ratio_not_speedup": ENVELOPE / float(
                ENVELOPE - total),
            "postcompute_oracle_only": True,
            "outside_frozen_tau_grid": True,
            "delta_aee_available": False,
            "executable_skip": False,
            "system_speedup": False,
            "admitted": False,
        })
    return rows


def write_csv(path, rows):
    require(rows, "refusing empty CSV")
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def seal_output(output):
    leaves = sorted(path for path in output.iterdir() if path.is_file() and
                    path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(leaves, "M462 output has no evidence")
    manifest = output / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                                for path in leaves), encoding="utf-8")
    outer = output / "SHA256SUMS.seal.sha256"
    outer.write_text("{}  {}\n".format(sha256(manifest), manifest.name),
                     encoding="utf-8")
    verify_double_seal(output, manifest.name, outer.name)
    return manifest, outer


def execute(contract_path, capture_root, output):
    output = Path(output).resolve()
    require(not output.exists(), "refusing to overwrite M462 output")
    script_start = sha256(Path(__file__).resolve())
    contract, identities = validate_contract(contract_path)
    capture, capture_summary, author = validate_r5_root(
        capture_root, contract, identities)
    m159 = strict_json(identities["m159"])
    require(m159["schema"] == "m159_h67_full_ffn_subgraph_scope_v1" and
            m159["accounted_compute_cycles_per_frame"] == {
                "fc1_plus_fc2": LINEAR_BASELINE,
                "full_ffn_subgraph_excluding_bn_residual": FFN_ACCOUNTED,
                "global_envelope_cycles": ENVELOPE,
                "partition_rule": "The 45.6M FFN-local ATLIF cycles are moved conceptually from the global ATLIF bucket; they are not added to 620.3M.",
                "perfect_removal_amdahl_ceiling_not_design_speedup": 1.4949983321314677,
                "share_of_current_compute_envelope": 0.33110293268737795,
                "sn1_atlif": SN1_ATLIF_BASELINE,
                "sn2_atlif": SN2_ATLIF_BASELINE,
            }, "M159 accounted cycle identity drift")
    pairs = load_ffn_ledger(identities["ffn_ledger"])
    runtime = load_operator_runtime(identities["operator_runtime"], pairs)

    output.mkdir(parents=True)
    per_record_path = output / "m462_per_record_tau_mask_audit.csv"
    try:
        (denominators, selected_issue, selected_tokens, equal_tokens,
         selected_sites, site_data) = validate_and_accumulate(
             capture, pairs, runtime, per_record_path)
        invariants = full_mask_invariants(pairs, denominators)
        frozen_rows = aggregate_frozen_grid(
            pairs, denominators, selected_issue, selected_tokens,
            equal_tokens, selected_sites)
        attach_atlif_to_frozen_rows(frozen_rows, per_record_path, pairs)
        extended = extended_cliff_diagnostic(site_data, pairs, denominators)
        frozen_path = output / "m462_frozen_tau_dual_mask_cycle_oracle.csv"
        extended_path = output / "m462_extended_posthoc_cliff_diagnostic.csv"
        write_csv(frozen_path, frozen_rows)
        write_csv(extended_path, extended)
        max_frozen = max(row["total_accounted_postcompute_oracle_saved_cycles"]
                         for row in frozen_rows)
        best_frozen = [row for row in frozen_rows
                       if row["total_accounted_postcompute_oracle_saved_cycles"] ==
                       max_frozen][0]
        summary = {
            "schema": "m462_h67_g8_postcompute_oracle_cycle_audit_v1",
            "status": ("NO_GO_FROZEN_TAU_GRID_BELOW_1P15_OPPORTUNITY_GATE"
                       if max_frozen < GATES[0][1] else
                       "GO_FROZEN_TAU_GRID_REACHES_1P15_OPPORTUNITY_GATE"),
            "identity": {
                "contract_sha256": sha256(contract_path),
                "analyzer_start_end_sha256": script_start,
                "capture_top_manifest_sha256": sha256(
                    Path(capture_root) / "manifest.sha256"),
                "capture_top_outer_seal_file_sha256": sha256(
                    Path(capture_root) / "manifest.sha256.outer.seal.sha256"),
                "capture_payload_manifest_sha256": sha256(
                    capture / "manifest.sha256"),
                "capture_payload_outer_seal_file_sha256": sha256(
                    capture / "manifest.sha256.outer.seal.sha256"),
                "r5_author_receipt_sha256": sha256(
                    Path(capture_root) /
                    "m460r5_one_shot_capture_author_receipt.json"),
                "input_sha256": {name: sha256(path)
                                  for name, path in identities.items()},
            },
            "population": {
                "samples": SAMPLES, "ffn_pairs": PAIRS,
                "sample_pair_records": RECORDS, "tokens": TOKENS,
                "spatial_sites": SITES, "frozen_tau_points": len(TAU_GRID),
                "dual_mask_rows": len(frozen_rows),
            },
            "cycle_model": {
                "lanes": LANES,
                "profile_normalization":
                    "per_pair_per_role_integer_floor_saved_equals_B_times_S_div_D",
                "token_mask":
                    "strict [T,N,H,W] postcompute mask; FC1/FC2 only",
                "site_mask":
                    "AND across literal T=10 then broadcast; FC1/FC2 plus ATLIF",
                "global_envelope_cycles": ENVELOPE,
                "required_savings": {name: value for name, value in GATES},
                "full_mask_invariants": invariants,
            },
            "frozen_tau_result": {
                "best_row": best_frozen,
                "maximum_accounted_postcompute_oracle_saved_cycles": max_frozen,
                "meets_1p15_opportunity_gate": max_frozen >= GATES[0][1],
            },
            "extended_posthoc_cliff_diagnostic": {
                "purpose": "Find the first strict max_T rho boundary at each opportunity gate.",
                "part_of_frozen_capture_tau_receipts": False,
                "delta_aee_available": False,
                "admitted": False,
                "rows": extended,
            },
            "source_receipt_checks": {
                "r5_double_seal_and_author_binding": True,
                "npz_byte_and_logical_array_receipts": True,
                "npz_dtype_shape_population": True,
                "rho_tau_and_source_work_recomputed": True,
                "operator_runtime_s10_input_active_exact": True,
                "profile100_full_linear_mask_exact": LINEAR_BASELINE,
                "full_site_atlif_mask_exact": ATLIF_BASELINE,
            },
            "admission": {
                "sealed_checkpoint_bound_s10_postcompute_oracle_audit": True,
                "frozen_tau_opportunity_counts": True,
                "extended_posthoc_cliff_diagnostic": True,
                "executable_skip": False,
                "delta_aee": False,
                "valid825_accuracy": False,
                "measured_cycle_speedup": False,
                "system_speedup": False,
                "energy": False, "ppa": False, "headline": False,
            },
            "files": {
                "per_record_tau_mask_audit": per_record_path.name,
                "frozen_tau_dual_mask_cycle_oracle": frozen_path.name,
                "extended_posthoc_cliff_diagnostic": extended_path.name,
            },
            "claim_boundary": (
                "M462 is a frozen H67-ep35/no-running S10 post-compute "
                "oracle opportunity audit. It admits no executable skip, "
                "Delta-AEE, valid825 accuracy, measured cycle speedup, "
                "system speedup, energy, PPA, or headline. Extended rho "
                "cliff rows are post-hoc diagnostics outside the captured "
                "tau grid and are not admitted performance points."),
        }
        summary_path = output / "m462_h67_g8_postcompute_oracle_cycle_audit.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) +
                                "\n", encoding="utf-8")
        require(sha256(Path(__file__).resolve()) == script_start,
                "M462 analyzer changed during execution")
        require(sha256(identities["docs359"]) ==
                contract["identity"]["docs359"]["sha256"],
                "protected docs/359 changed during M462")
        manifest, outer = seal_output(output)
    except Exception:
        # Preserve diagnostics for debugging, but never leave a sealed-looking
        # failed result.
        raise
    print(json.dumps({
        "status": summary["status"],
        "maximum_frozen_tau_oracle_saved_cycles": max_frozen,
        "meets_1p15_opportunity_gate": max_frozen >= GATES[0][1],
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
        "postcompute_oracle_only": True,
        "executable_skip": False,
        "system_speedup": False,
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    execute(args.contract.resolve(), args.capture_root.resolve(),
            args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
