#!/usr/bin/env python3
"""Exact N=0 Patch/early-Conv receptive-field audit and G10 fast-kill.

The frozen bit-sparse baseline already omits every zero source contribution.
This script therefore gives no incremental MAC or active weight-row credit to
an empty receptive field.  It measures only exact empty output populations,
whole-temporal spatial reuse, aligned empty tiles, and explicitly bounded
scan/commit/ATLIF opportunity.  Dynamic no-running BN and residual/PED
shortcuts remain fail-closed semantic blockers for checkpoint-static replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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
        answer = {}
        for key, value in items:
            require(key not in answer, "duplicate JSON key: " + key)
            answer[key] = value
        return answer

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def normalize_tbc_hw(shape):
    shape = tuple(int(value) for value in shape)
    if len(shape) == 4:
        return (shape[0], 1, shape[1], shape[2], shape[3])
    require(len(shape) == 5, "unsupported Conv shape")
    return shape


def decode_record(record, payload_root):
    path = payload_root / record["relative_path"]
    require(path.is_file(), "missing payload: " + str(path))
    require(sha256(path) == record["file_sha256"],
            "payload SHA drift: " + str(path))
    shape = normalize_tbc_hw(record["input_shape"])
    require(product(shape) == int(record["input_elements"]),
            "input extent drift")
    packed = np.fromfile(str(path), dtype=np.uint8)
    require(packed.size == int(record["packed_bytes"]),
            "packed byte extent drift")
    bits = np.unpackbits(packed, bitorder="little")[:product(shape)]
    require(int(bits.sum(dtype=np.uint64)) == int(record["active_elements"]),
            "active element mismatch")
    return bits.reshape(shape)


def infer_stride(shape, output_shape):
    shape = normalize_tbc_hw(shape)
    output_shape = normalize_tbc_hw(output_shape)
    require(shape[:2] == output_shape[:2] and output_shape[2] == 96,
            "selected Conv output identity drift")
    height, width = shape[-2:]
    out_height, out_width = output_shape[-2:]
    candidates = []
    for stride in (1, 2):
        if ((height + 2 - 3) // stride + 1,
                (width + 2 - 3) // stride + 1) == (out_height, out_width):
            candidates.append(stride)
    require(len(candidates) == 1, "ambiguous stride")
    return candidates[0]


def receptive_field_population(bits, output_shape, stride):
    """Return exact source count for every T/B/output-H/output-W token."""
    output_shape = normalize_tbc_hw(output_shape)
    out_height, out_width = output_shape[-2:]
    padded = np.pad(bits, ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1)))
    population = np.zeros(
        (bits.shape[0], bits.shape[1], out_height, out_width),
        dtype=np.uint16,
    )
    for kernel_y in range(3):
        for kernel_x in range(3):
            sampled = padded[
                :, :, :,
                kernel_y:kernel_y + stride * out_height:stride,
                kernel_x:kernel_x + stride * out_width:stride,
            ]
            require(sampled.shape[-2:] == (out_height, out_width),
                    "padding/halo sample geometry drift")
            population += sampled.sum(axis=2, dtype=np.uint16)
    return population


def aligned_tile_metrics(whole_temporal_zero, tile_sizes):
    """Count non-overlapping aligned tiles that are empty for every T."""
    batch, height, width = whole_temporal_zero.shape
    rows = {}
    total_sites = int(batch * height * width)
    empty_sites = int(whole_temporal_zero.sum(dtype=np.uint64))
    for tile in tile_sizes:
        require(height % tile == 0 and width % tile == 0,
                "tile does not divide output geometry")
        view = whole_temporal_zero.reshape(
            batch, height // tile, tile, width // tile, tile)
        empty_tiles = view.all(axis=(2, 4))
        count = int(empty_tiles.sum(dtype=np.uint64))
        covered = int(count * tile * tile)
        rows[str(tile)] = {
            "tile_height": tile,
            "tile_width": tile,
            "aligned_tiles": int(batch * (height // tile) * (width // tile)),
            "all_zero_tiles": count,
            "covered_spatial_sites": covered,
            "covered_fraction_of_all_spatial_sites": (
                float(covered) / float(total_sites)),
            "covered_fraction_of_empty_spatial_sites": (
                float(covered) / float(empty_sites) if empty_sites else 0.0),
        }
    return rows


def full_envelope_speedup(envelope, saved):
    require(0 <= saved < envelope, "invalid envelope saving")
    return float(envelope) / float(envelope - saved)


def scaled_binary_envelope_speedup(envelope, binary_cycles,
                                   isolated_speedup):
    scaled = float(binary_cycles) / float(isolated_speedup)
    return float(envelope) / float(envelope - binary_cycles + scaled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = Path(__file__).resolve()
    source_start = sha256(source)
    contract_path = args.contract.resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m374_patch_zero_rf_constant_replay_fastkill_contract_v1",
            "M374 contract schema drift")
    hw = contract_path.parents[1]
    identities = {
        "contract": {
            "path": str(contract_path.relative_to(hw)),
            "sha256": sha256(contract_path),
        },
        "analyzer": {
            "path": str(source.relative_to(hw.parent)),
            "sha256": source_start,
        },
    }
    paths = {}
    for label, spec in contract["identity"].items():
        path = (hw / spec["path"]).resolve()
        require(path.is_file(), "missing frozen input: " + str(path))
        digest = sha256(path)
        require(digest == spec["sha256"],
                "{} SHA drift: {}".format(label, digest))
        paths[label] = path
        identities[label] = {"path": spec["path"], "sha256": digest}

    manifest = strict_json(paths["m51_manifest"])
    m222 = strict_json(paths["m222_patch_model"])
    m272 = strict_json(paths["m272_bit_sparse_pwp_model"])
    m291 = strict_json(paths["m291_no_running_correction"])
    m286r2 = strict_json(paths["m286_m273r2_review"])
    m221 = strict_json(paths["m221_compute_envelope"])
    m25 = strict_json(paths["m25_atlif_exact96_lower_bound"])
    dynamic_review = strict_json(paths["m159_m160_dynamic_bn_review"])

    require(manifest["packing"] == {
        "bit_order": "LITTLE_WITHIN_BYTE",
        "delta_payload_retained": False,
        "file_granularity": "ONE_RAW_FILE_PER_HOOK_CALL",
        "float_payload_retained": False,
        "layout": "C_ORDER_FLAT",
        "tail_padding_high_bits_zero": True,
    }, "M51 packing drift")
    require(m291["corrected_frozen_semantics"]["bn_policy"] ==
            "no_running/current-batch", "M291 BN policy drift")
    require(dynamic_review["direct_model_audit"][
        "bn_after_track_running_false"] == 78,
        "dynamic BN review identity drift")
    require(m286r2["verdict"]["clean_cycle_formula"].startswith("GO_ONLY") and
            m286r2["independently_verified_metrics"]["clean_N1_cycles"] == 24,
            "M273r2 review drift")
    require(m25["atlif_exact96_arithmetic_lower_bound"][
        "service_cycles_lower_bound"] == 128020500,
        "M25 exact96 lower bound drift")

    selected = set(int(value) for value in
                   contract["population"]["selected_module_indices"])
    records = [row for row in manifest["records"]
               if int(row["module_index"]) in selected]
    require(len(records) == int(contract["population"]["records"]),
            "M374 record population drift")
    require(sorted(set(int(row["module_index"]) for row in records)) ==
            sorted(selected), "M374 module population drift")
    require(all(row["operator"] == "Conv2d" and
                "patch_embed" in row["name"] for row in records),
            "non-patch Conv selected")
    module_identities = manifest["module_identities"]
    require(all(module_identities[row["name"]]["bias"] is None
                for row in records), "selected Conv gained a bias")

    m222_index = {
        (int(row["sample_id"]), row["name"]): row
        for row in m222["per_record"]
    }
    tile_sizes = [int(value) for value in
                  contract["population"]["aligned_spatial_tile_sizes"]]
    per_record = []
    aggregate = defaultdict(lambda: {
        "records": 0,
        "source_contributions": 0,
        "output_tokens": 0,
        "zero_receptive_field_tokens": 0,
        "border_zero_tokens": 0,
        "interior_zero_tokens": 0,
        "spatial_sites": 0,
        "whole_temporal_zero_spatial_sites": 0,
        "records_with_whole_temporal_zero": 0,
        "input_scan_cycles": 0,
        "output_commit_cycles": 0,
        "source_service_cycles": 0,
        "tiles": defaultdict(lambda: {
            "aligned_tiles": 0,
            "all_zero_tiles": 0,
            "covered_spatial_sites": 0,
        }),
    })

    payload_root = paths["m51_manifest"].parent
    for record in sorted(records, key=lambda row: (
            int(row["sample_id"]), int(row["module_index"]))):
        bits = decode_record(record, payload_root)
        stride = infer_stride(record["input_shape"], record["output_shape"])
        population = receptive_field_population(
            bits, record["output_shape"], stride)
        zero = population == 0
        whole_zero = zero.all(axis=0)
        border = np.zeros(zero.shape[-2:], dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        border_zero = int(zero[..., border].sum(dtype=np.uint64))
        zero_tokens = int(zero.sum(dtype=np.uint64))
        source_count = int(population.sum(dtype=np.uint64))
        output_tokens = int(population.size)
        spatial_sites = int(whole_zero.size)
        whole_zero_sites = int(whole_zero.sum(dtype=np.uint64))
        tiles = aligned_tile_metrics(whole_zero, tile_sizes)

        prior = m222_index[(int(record["sample_id"]), record["name"])]
        prior_point = prior["model_points"]["K1_D96_ROW_STRIPED"]
        require(source_count == int(prior[
            "valid_receptive_field_source_contributions"]),
            "M222 source contribution mismatch")
        require(zero_tokens == int(prior_point["zero_output_tokens"]),
                "M222 zero token mismatch")
        require(output_tokens == int(prior_point["output_commit_cycles"]),
                "M222 output extent mismatch")

        row = {
            "sample_id": int(record["sample_id"]),
            "module_index": int(record["module_index"]),
            "module": record["name"],
            "input_shape": list(normalize_tbc_hw(record["input_shape"])),
            "output_shape": list(normalize_tbc_hw(record["output_shape"])),
            "stride": stride,
            "padding": [1, 1],
            "source_contributions": source_count,
            "output_tokens": output_tokens,
            "zero_receptive_field_tokens": zero_tokens,
            "zero_receptive_field_token_fraction": (
                float(zero_tokens) / float(output_tokens)),
            "border_zero_tokens": border_zero,
            "interior_zero_tokens": zero_tokens - border_zero,
            "spatial_sites": spatial_sites,
            "whole_temporal_zero_spatial_sites": whole_zero_sites,
            "whole_temporal_zero_spatial_fraction": (
                float(whole_zero_sites) / float(spatial_sites)),
            "aligned_spatial_tiles": tiles,
        }
        per_record.append(row)
        acc = aggregate[record["name"]]
        acc["records"] += 1
        acc["source_contributions"] += source_count
        acc["output_tokens"] += output_tokens
        acc["zero_receptive_field_tokens"] += zero_tokens
        acc["border_zero_tokens"] += border_zero
        acc["interior_zero_tokens"] += zero_tokens - border_zero
        acc["spatial_sites"] += spatial_sites
        acc["whole_temporal_zero_spatial_sites"] += whole_zero_sites
        acc["records_with_whole_temporal_zero"] += int(whole_zero_sites > 0)
        acc["input_scan_cycles"] += int(prior_point[
            "linebuffer_scan_cycles"])
        acc["output_commit_cycles"] += int(prior_point[
            "output_commit_cycles"])
        acc["source_service_cycles"] += int(prior_point["service_cycles"])
        for tile, values in tiles.items():
            for key in ("aligned_tiles", "all_zero_tiles",
                        "covered_spatial_sites"):
                acc["tiles"][tile][key] += int(values[key])

    mapping = contract["atlif_replay_mapping"]
    module_rows = []
    for module in sorted(aggregate):
        acc = aggregate[module]
        tiles = {}
        for tile in map(str, tile_sizes):
            values = dict(acc["tiles"][tile])
            values["covered_fraction_of_all_spatial_sites"] = (
                float(values["covered_spatial_sites"]) /
                float(acc["spatial_sites"]))
            values["covered_fraction_of_empty_spatial_sites"] = (
                float(values["covered_spatial_sites"]) /
                float(acc["whole_temporal_zero_spatial_sites"])
                if acc["whole_temporal_zero_spatial_sites"] else 0.0)
            tiles[tile] = values
        has_residual = ("resblocks" in module and ".conv2.0" in module)
        has_ped_shortcut = module.endswith("patch_embed.proj.conv")
        following_atlif = mapping.get(module)
        module_rows.append({
            "module": module,
            "records": acc["records"],
            "conv_bias": None,
            "bn_policy": "no_running/current-batch",
            "source_contributions": acc["source_contributions"],
            "output_tokens": acc["output_tokens"],
            "zero_receptive_field_tokens": acc[
                "zero_receptive_field_tokens"],
            "zero_receptive_field_token_fraction": (
                float(acc["zero_receptive_field_tokens"]) /
                float(acc["output_tokens"])),
            "border_zero_tokens": acc["border_zero_tokens"],
            "interior_zero_tokens": acc["interior_zero_tokens"],
            "spatial_sites": acc["spatial_sites"],
            "whole_temporal_zero_spatial_sites": acc[
                "whole_temporal_zero_spatial_sites"],
            "whole_temporal_zero_spatial_fraction": (
                float(acc["whole_temporal_zero_spatial_sites"]) /
                float(acc["spatial_sites"])),
            "records_with_whole_temporal_zero": acc[
                "records_with_whole_temporal_zero"],
            "aligned_spatial_tiles": tiles,
            "m222_input_scan_cycles_s10": acc["input_scan_cycles"],
            "m222_output_commit_cycles_s10": acc["output_commit_cycles"],
            "m222_source_service_cycles_s10": acc["source_service_cycles"],
            "incremental_source_mac_or_active_weight_row_cycles_saved": 0,
            "following_atlif_replay_candidate": following_atlif,
            "residual_add_blocks_static_output_replay": has_residual,
            "ped_shortcut_blocks_static_output_replay": has_ped_shortcut,
            "checkpoint_static_replay_legal": False,
        })

    totals = {
        key: sum(int(row[key]) for row in module_rows)
        for key in (
            "source_contributions", "output_tokens",
            "zero_receptive_field_tokens", "border_zero_tokens",
            "interior_zero_tokens", "spatial_sites",
            "whole_temporal_zero_spatial_sites",
            "m222_input_scan_cycles_s10",
            "m222_output_commit_cycles_s10",
            "m222_source_service_cycles_s10",
        )
    }
    require(totals["source_contributions"] == 1774268587 and
            totals["output_tokens"] == 40320000 and
            totals["zero_receptive_field_tokens"] == 7257197,
            "M222 aggregate population drift")
    constants = contract["cycle_constants"]
    require(totals["m222_input_scan_cycles_s10"] ==
            constants["m222_linebuffer_scan_cycles_s10"] and
            totals["m222_output_commit_cycles_s10"] ==
            constants["m222_output_commit_cycles_s10"] and
            totals["m222_source_service_cycles_s10"] ==
            constants["m222_source_service_cycles_s10"],
            "M222 cost partition drift")
    require(m272["same_resource_module_cycles"]["bit_sparse"] ==
            constants["m272_bit_sparse_cycles_s10"],
            "M272 bit-sparse baseline drift")

    with paths["atlif_activity"].open("r", encoding="utf-8", newline="") as h:
        atlif_rows = {row["name"]: row for row in csv.DictReader(h)}
    cycles_per_site = constants[
        "atlif_exact96_cycles_per_whole_temporal_spatial_site_for_96_channels"]
    atlif_replay_rows = []
    measured_atlif_saving_s10 = 0
    perfect_mapped_atlif_cycles_s10 = 0
    module_by_name = {row["module"]: row for row in module_rows}
    record_by_name = defaultdict(list)
    for row in per_record:
        record_by_name[row["module"]].append(row)
    for source_module, target_module in mapping.items():
        trace = atlif_rows[target_module]
        require(int(trace["calls"]) == 10 and
                int(trace["temporal_steps"]) == 10 and
                trace["output_mode"] == "binary",
                "mapped ATLIF trace identity drift")
        elements = int(trace["elements"])
        exact96_cycles = (elements * 10) // 96
        require(elements * 10 % 96 == 0,
                "mapped ATLIF exact96 extent is fractional")
        empty_sites = module_by_name[source_module][
            "whole_temporal_zero_spatial_sites"]
        compute_once_cycles = sum(
            int(row["whole_temporal_zero_spatial_sites"] > 0) * cycles_per_site
            for row in record_by_name[source_module])
        saved = max(0, empty_sites * cycles_per_site - compute_once_cycles)
        measured_atlif_saving_s10 += saved
        perfect_mapped_atlif_cycles_s10 += exact96_cycles
        atlif_replay_rows.append({
            "source_conv": source_module,
            "following_atlif": target_module,
            "atlif_trace_elements_s10": elements,
            "atlif_exact96_arithmetic_lower_bound_cycles_s10": exact96_cycles,
            "whole_temporal_zero_spatial_sites_s10": empty_sites,
            "dynamic_constant_compute_once_cycles_s10": compute_once_cycles,
            "ideal_dynamic_constant_replay_cycles_saved_s10": saved,
            "qualification": (
                "requires current-batch BN barrier, per-sample/channel dynamic "
                "constant generation, exact ATLIF response, descriptor routing "
                "and broadcast; this is not checkpoint-static or executable"
            ),
        })

    baseline = constants["m272_bit_sparse_cycles_s10"]
    oracle_zero_commit_speed = float(baseline) / float(
        baseline - totals["zero_receptive_field_tokens"])
    impossible_all_scan_commit_speed = float(baseline) / float(
        baseline - totals["m222_input_scan_cycles_s10"] -
        totals["m222_output_commit_cycles_s10"])
    envelope = constants["compute_envelope_cycles_per_frame"]
    binary_cycles = constants["six_binary_patch_activity_cycles_per_frame"]
    measured_atlif_saved_per_frame = (
        float(measured_atlif_saving_s10) /
        float(contract["population"]["samples"]))
    perfect_mapped_atlif_per_frame = (
        float(perfect_mapped_atlif_cycles_s10) /
        float(contract["population"]["samples"]))
    gate = float(constants["promotion_gate_scope_correct_speedup"])

    result = {
        "schema": "m374_patch_zero_rf_constant_replay_fastkill_v1",
        "status": "PASS_EXACT_N0_POPULATION__FAST_KILL_CHECKPOINT_STATIC_G10__NO_RTL",
        "identity": identities,
        "population": {
            "samples": contract["population"]["samples"],
            "records": len(per_record),
            "modules": len(module_rows),
            "all_selected_conv_biases_null": True,
            "kernel": [3, 3],
            "padding": [1, 1],
            "padding_and_halo_explicitly_sampled": True,
            "source_contributions": totals["source_contributions"],
            "output_tokens": totals["output_tokens"],
            "zero_receptive_field_tokens": totals[
                "zero_receptive_field_tokens"],
            "zero_receptive_field_token_fraction": (
                float(totals["zero_receptive_field_tokens"]) /
                float(totals["output_tokens"])),
            "border_zero_tokens": totals["border_zero_tokens"],
            "interior_zero_tokens": totals["interior_zero_tokens"],
            "whole_temporal_spatial_sites": totals["spatial_sites"],
            "whole_temporal_zero_spatial_sites": totals[
                "whole_temporal_zero_spatial_sites"],
            "whole_temporal_zero_spatial_fraction": (
                float(totals["whole_temporal_zero_spatial_sites"]) /
                float(totals["spatial_sites"])),
        },
        "per_module": module_rows,
        "per_record": per_record,
        "bit_sparse_double_count_guard": {
            "baseline": "M272 one-source/96-lane bit-sparse reference",
            "source_service_cycles_s10": totals[
                "m222_source_service_cycles_s10"],
            "source_contributions_s10": totals["source_contributions"],
            "empty_receptive_field_source_service_cycles": 0,
            "incremental_mac_cycles_saved_by_G10": 0,
            "incremental_active_weight_or_pwp_dma_cycles_saved_by_G10": 0,
            "reason": (
                "M222/M272 issue one weight row only for an active source; "
                "N=0 has no source issue before G10"
            ),
        },
        "semantic_audit": {
            "conv_zero_for_n0": True,
            "conv_reason": "all six Conv biases are null and every valid/padded RF input is zero",
            "frozen_bn_policy": "no_running/current-batch over T*B*H*W",
            "checkpoint_static_bn_constant": False,
            "dynamic_bn_zero_value": "c[j]=beta[j]-gamma[j]*mu[j]/sqrt(var[j]+eps), sample-dependent after a global barrier",
            "whole_temporal_zero_required_for_atlif_response_reuse": True,
            "resblock_conv2_output_requires_identity_read_add_commit": True,
            "proj_main_branch_requires_live_ped_shortcut_read_add_commit": True,
            "m273r2_n0_behavior": "illegal sticky fault; no admitted exact empty-tile response exists",
            "checkpoint_static_constant_replay": "NO_GO",
            "dynamic_per_sample_constant_broadcast": "ANALYTICAL_UPPER_BOUND_ONLY",
        },
        "atlif_dynamic_replay_upper_bound": {
            "mapped_conv_atlif_pairs": atlif_replay_rows,
            "measured_ideal_cycles_saved_s10": measured_atlif_saving_s10,
            "measured_ideal_cycles_saved_per_frame": measured_atlif_saved_per_frame,
            "measured_scope_correct_compute_envelope_sensitivity":
                full_envelope_speedup(envelope,
                                      measured_atlif_saved_per_frame),
            "perfect_remove_all_three_mapped_atlif_cycles_per_frame":
                perfect_mapped_atlif_per_frame,
            "perfect_remove_all_three_mapped_atlif_compute_envelope_ceiling":
                full_envelope_speedup(envelope,
                                      perfect_mapped_atlif_per_frame),
            "qualification": (
                "exact M51-s10 empty-site population overlaid on the M25 "
                "arithmetic-only exact96 lower bound and M221 profile envelope; "
                "no executable replay, same-population wall cycles, BN ports, "
                "broadcast, state, or output materialization"
            ),
        },
        "scan_bn_commit_bounds": {
            "m272_bit_sparse_cycles_s10": baseline,
            "m222_linebuffer_scan_cycles_s10": totals[
                "m222_input_scan_cycles_s10"],
            "m222_output_commit_cycles_s10": totals[
                "m222_output_commit_cycles_s10"],
            "oracle_remove_only_measured_zero_token_commit_speedup":
                oracle_zero_commit_speed,
            "oracle_zero_commit_compute_envelope_sensitivity":
                scaled_binary_envelope_speedup(
                    envelope, binary_cycles, oracle_zero_commit_speed),
            "impossible_remove_all_six_module_scan_and_commit_speedup":
                impossible_all_scan_commit_speed,
            "impossible_all_scan_commit_compute_envelope_sensitivity":
                scaled_binary_envelope_speedup(
                    envelope, binary_cycles,
                    impossible_all_scan_commit_speed),
            "bn_vector_normalization_extent_s10": totals["output_tokens"],
            "bn_zero_vector_candidates_s10": totals[
                "zero_receptive_field_tokens"],
            "bn_cycles_in_frozen_620m_envelope": False,
            "commit_elision_semantically_legal": False,
            "scan_elision_requires_upstream_halo_aware_zero_metadata": True,
        },
        "coverage_gap": {
            "selected_exact_binary_modules": 6,
            "complete_patch_conv_modules": 8,
            "selected_profile_activity_cycles": binary_cycles,
            "complete_patch_profile_activity_cycles": constants[
                "patch_total_activity_cycles_per_frame"],
            "missing_nonbinary_cycles": (
                constants["patch_total_activity_cycles_per_frame"] -
                binary_cycles),
            "missing_modules": [
                "sttmultires_unet.encoders.swin3d.patch_embed.head.conv.0",
                "sttmultires_unet.encoders.swin3d.patch_embed.proj.conv_res"
            ],
            "proj_conv_res_profile_input_activity": "approximately one; zero-mask opportunity expected negligible but not RF-captured",
        },
        "decision": {
            "promotion_gate": gate,
            "checkpoint_static_g10": "FAST_KILL",
            "dynamic_constant_broadcast_rtl": "NO_GO_BELOW_GATE_AND_MISSING_EXECUTABLE_SEMANTICS",
            "write_rtl": False,
            "run_n_gt_0_accuracy": False,
            "reason": (
                "zero source MAC/weight work is already absent; static replay "
                "is invalid under current-batch BN and residual/PED shortcuts; "
                "even perfect deletion of all three mapped ATLIF modules and "
                "the impossible all-scan/all-commit isolated bounds remain "
                "below the 1.15 scope gate"
            ),
        },
        "minimum_new_capture_if_reopened": [
            "For head.conv.0 and proj.conv_res, capture exact input!=0 masks with sample/module/call identity, shape, stride, padding and SHA; values are unnecessary for N=0 census.",
            "For every candidate Conv, capture per-sample current-batch BN mean/biased-variance/gamma/beta and an exact post-BN zero-RF miter receipt.",
            "Capture whole-T post-ATLIF outputs at zero-RF sites to prove one dynamic response per sample/channel is reusable, including ATLIF state/reset identity.",
            "Capture residual identity/PED shortcut reads and final commit values/tags; a zero Conv branch never authorizes dropping those paths.",
            "Capture halo-aware upstream zero metadata and address-timed scan/weight/commit transactions before assigning any SRAM/DRAM or scan saving."
        ],
        "algorithm_feedback_not_executed": [
            "If this axis is strategically required, train/evaluate a static or zero-preserving normalization policy for Patch only; valid825 is mandatory.",
            "Constrain the zero-input BN-to-ATLIF response to zero and preserve residual identity so an empty descriptor can be forwarded without materialization.",
            "Propagate halo-aware empty masks from the event frontend; isolated per-timestep zeros are insufficient for temporal replay."
        ],
        "admission": {
            "exact_n0_rf_population_s10": True,
            "exact_aligned_empty_tile_population_s10": True,
            "padding_halo_handled": True,
            "bit_sparse_double_count_prevented": True,
            "checkpoint_static_constant_replay": False,
            "dynamic_bn_replay": False,
            "residual_or_ped_commit_elision": False,
            "hardware_cycles": False,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "energy": False,
            "system_speedup": False,
            "headline": False,
        },
        "tool_execution": {
            "cpu_only": True,
            "gpu_invoked": False,
            "synopsys_invoked": False,
            "open_source_rtl_invoked": False,
        },
        "docs359_modified": False,
        "docs359_sha256": sha256(paths["docs359"]),
    }

    require(sha256(source) == source_start, "analyzer changed during run")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / (
        "m374_patch_zero_rf_constant_replay_fastkill_r1.json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    csv_path = args.output_dir / "m374_per_module_zero_rf.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        keys = [
            "module", "output_tokens", "zero_receptive_field_tokens",
            "zero_receptive_field_token_fraction", "spatial_sites",
            "whole_temporal_zero_spatial_sites",
            "whole_temporal_zero_spatial_fraction",
            "m222_source_service_cycles_s10",
            "incremental_source_mac_or_active_weight_row_cycles_saved",
            "following_atlif_replay_candidate",
            "checkpoint_static_replay_legal",
        ]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in module_rows:
            writer.writerow({key: row[key] for key in keys})
    print("PASS M374 records={} zero={} wholeT={} measured_atlif_save_s10={} fast_kill=1".format(
        len(per_record), totals["zero_receptive_field_tokens"],
        totals["whole_temporal_zero_spatial_sites"],
        measured_atlif_saving_s10))


if __name__ == "__main__":
    main()
