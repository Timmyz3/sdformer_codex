#!/usr/bin/env python3
"""Independent M375 hammer of M374 using channel-OR receptive fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extent(shape):
    answer = 1
    for value in shape:
        answer *= int(value)
    return answer


def shape5(shape):
    shape = tuple(int(value) for value in shape)
    return ((shape[0], 1, shape[1], shape[2], shape[3])
            if len(shape) == 4 else shape)


def decode(record, root):
    path = root / record["relative_path"]
    require(path.is_file(), "missing payload")
    require(sha256(path) == record["file_sha256"], "payload SHA drift")
    shape = shape5(record["input_shape"])
    raw = np.fromfile(str(path), dtype=np.uint8)
    require(raw.size == int(record["packed_bytes"]), "payload bytes drift")
    bits = np.unpackbits(raw, bitorder="little")[:extent(shape)]
    require(int(bits.sum(dtype=np.uint64)) == int(record["active_elements"]),
            "payload active drift")
    return bits.reshape(shape), int(raw.size)


def independent_zero_mask(bits, output_shape):
    """Different construction from M374: channel OR before halo sampling."""
    out_shape = shape5(output_shape)
    out_height, out_width = out_shape[-2:]
    height, width = bits.shape[-2:]
    candidates = [stride for stride in (1, 2)
                  if ((height + 2 - 3) // stride + 1,
                      (width + 2 - 3) // stride + 1) ==
                  (out_height, out_width)]
    require(len(candidates) == 1, "stride ambiguity")
    stride = candidates[0]
    any_channel = bits.any(axis=2)
    padded = np.pad(any_channel,
                    ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant", constant_values=False)
    nonzero = np.zeros((bits.shape[0], bits.shape[1],
                        out_height, out_width), dtype=bool)
    for kernel_y in range(3):
        for kernel_x in range(3):
            sampled = padded[
                :, :,
                kernel_y:kernel_y + stride * out_height:stride,
                kernel_x:kernel_x + stride * out_width:stride,
            ]
            require(sampled.shape[-2:] == (out_height, out_width),
                    "halo geometry drift")
            nonzero |= sampled
    return ~nonzero, stride


def tile_counts(whole_zero, sizes):
    batch, height, width = whole_zero.shape
    answer = {}
    for size in sizes:
        require(height % size == 0 and width % size == 0,
                "tile divisibility drift")
        view = whole_zero.reshape(batch, height // size, size,
                                  width // size, size)
        count = int(view.all(axis=(2, 4)).sum(dtype=np.uint64))
        answer[str(size)] = {
            "aligned_tiles": int(batch * (height // size) * (width // size)),
            "all_zero_tiles": count,
            "covered_spatial_sites": int(count * size * size),
        }
    return answer


def close(left, right, tolerance=1e-12):
    return math.isclose(float(left), float(right), rel_tol=tolerance,
                        abs_tol=tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    script = Path(__file__).resolve()
    script_start = sha256(script)
    contract_path = args.contract.resolve()
    contract = load_json(contract_path)
    require(contract.get("schema") ==
            "m375_m374_patch_zero_rf_independent_hammer_contract_v1",
            "M375 contract schema drift")
    hw = contract_path.parents[1]
    paths = {}
    identities = {
        "contract": {"path": str(contract_path.relative_to(hw)),
                     "sha256": sha256(contract_path)},
        "auditor": {"path": str(script.relative_to(hw.parent)),
                    "sha256": script_start},
    }
    for label, spec in contract["identity"].items():
        path = (hw / spec["path"]).resolve()
        require(path.is_file(), "missing input: " + str(path))
        digest = sha256(path)
        require(digest == spec["sha256"], label + " SHA drift")
        paths[label] = path
        identities[label] = {"path": spec["path"], "sha256": digest}

    manifest = load_json(paths["m51_manifest"])
    producer = load_json(paths["m374_result"])
    m222 = load_json(paths["m222_patch_model"])
    m272 = load_json(paths["m272_bit_sparse_pwp_model"])
    m291 = load_json(paths["m291_no_running_correction"])
    require(producer["status"].startswith("PASS_EXACT_N0"),
            "M374 result status drift")
    require(m291["corrected_frozen_semantics"]["bn_policy"] ==
            "no_running/current-batch", "BN policy drift")

    selected = set(range(6))
    records = [row for row in manifest["records"]
               if int(row["module_index"]) in selected]
    require(len(records) == 60, "selected record drift")
    producer_index = {
        (int(row["sample_id"]), row["module"]): row
        for row in producer["per_record"]
    }
    m222_index = {
        (int(row["sample_id"]), row["name"]): row
        for row in m222["per_record"]
    }
    sizes = [1, 2, 4, 8]
    raw_payload_bytes = 0
    per_record = []
    per_module = defaultdict(lambda: {
        "records": 0,
        "output_tokens": 0,
        "zero_tokens": 0,
        "spatial_sites": 0,
        "whole_t_zero_sites": 0,
        "tiles": defaultdict(lambda: {
            "aligned_tiles": 0,
            "all_zero_tiles": 0,
            "covered_spatial_sites": 0,
        }),
    })
    mismatches = []
    for record in sorted(records, key=lambda row: (
            int(row["sample_id"]), int(row["module_index"]))):
        bits, payload_bytes = decode(record, paths["m51_manifest"].parent)
        raw_payload_bytes += payload_bytes
        zero, stride = independent_zero_mask(bits, record["output_shape"])
        whole_zero = zero.all(axis=0)
        counts = tile_counts(whole_zero, sizes)
        key = (int(record["sample_id"]), record["name"])
        expected = producer_index[key]
        prior = m222_index[key]
        observed = {
            "sample_id": key[0],
            "module": key[1],
            "stride": stride,
            "output_tokens": int(zero.size),
            "zero_receptive_field_tokens": int(
                zero.sum(dtype=np.uint64)),
            "spatial_sites": int(whole_zero.size),
            "whole_temporal_zero_spatial_sites": int(
                whole_zero.sum(dtype=np.uint64)),
            "aligned_spatial_tiles": counts,
        }
        for field in ("stride", "output_tokens",
                      "zero_receptive_field_tokens", "spatial_sites",
                      "whole_temporal_zero_spatial_sites"):
            if int(observed[field]) != int(expected[field]):
                mismatches.append({"key": key, "field": field,
                                   "observed": observed[field],
                                   "expected": expected[field]})
        for size in map(str, sizes):
            for field in ("aligned_tiles", "all_zero_tiles",
                          "covered_spatial_sites"):
                if int(counts[size][field]) != int(
                        expected["aligned_spatial_tiles"][size][field]):
                    mismatches.append({
                        "key": key, "field": "tile." + size + "." + field,
                        "observed": counts[size][field],
                        "expected": expected[
                            "aligned_spatial_tiles"][size][field],
                    })
        require(observed["zero_receptive_field_tokens"] == int(
            prior["model_points"]["K1_D96_ROW_STRIPED"][
                "zero_output_tokens"]), "M222 independent zero mismatch")
        per_record.append(observed)
        acc = per_module[key[1]]
        acc["records"] += 1
        acc["output_tokens"] += observed["output_tokens"]
        acc["zero_tokens"] += observed["zero_receptive_field_tokens"]
        acc["spatial_sites"] += observed["spatial_sites"]
        acc["whole_t_zero_sites"] += observed[
            "whole_temporal_zero_spatial_sites"]
        for size in map(str, sizes):
            for field in counts[size]:
                acc["tiles"][size][field] += counts[size][field]

    require(not mismatches, "M374 independent record mismatch")
    require(raw_payload_bytes == 645120000, "selected payload byte drift")
    module_rows = []
    for name in sorted(per_module):
        acc = per_module[name]
        module_rows.append({
            "module": name,
            "records": acc["records"],
            "output_tokens": acc["output_tokens"],
            "zero_receptive_field_tokens": acc["zero_tokens"],
            "spatial_sites": acc["spatial_sites"],
            "whole_temporal_zero_spatial_sites": acc["whole_t_zero_sites"],
            "aligned_spatial_tiles": {
                key: dict(value) for key, value in acc["tiles"].items()
            },
        })
    output_tokens = sum(row["output_tokens"] for row in module_rows)
    zero_tokens = sum(row["zero_receptive_field_tokens"]
                      for row in module_rows)
    spatial_sites = sum(row["spatial_sites"] for row in module_rows)
    whole_t = sum(row["whole_temporal_zero_spatial_sites"]
                  for row in module_rows)
    require((output_tokens, zero_tokens, spatial_sites, whole_t) ==
            (40320000, 7257197, 4032000, 156),
            "M374 aggregate independent mismatch")
    module_whole_t = {row["module"]:
                      row["whole_temporal_zero_spatial_sites"]
                      for row in module_rows}
    require(sum(value for name, value in module_whole_t.items()
                if ".conv2.0" not in name) == 0 and
            module_whole_t[
                "sttmultires_unet.encoders.swin3d.patch_embed."
                "residual_encoding.resblocks.0.conv2.0"] == 156,
            "whole-T location classification drift")

    baseline = int(m272["same_resource_module_cycles"]["bit_sparse"])
    scan = int(producer["scan_bn_commit_bounds"][
        "m222_linebuffer_scan_cycles_s10"])
    commit = int(producer["scan_bn_commit_bounds"][
        "m222_output_commit_cycles_s10"])
    envelope = 620302905
    binary = 172321077
    oracle_commit = float(baseline) / float(baseline - zero_tokens)
    impossible_scan_commit = float(baseline) / float(baseline - scan - commit)
    oracle_scaled = float(envelope) / float(
        envelope - binary + float(binary) / oracle_commit)
    impossible_scaled = float(envelope) / float(
        envelope - binary + float(binary) / impossible_scan_commit)
    perfect_atlif = float(envelope) / float(envelope - 23040000)
    bounds = producer["scan_bn_commit_bounds"]
    require(close(oracle_commit, bounds[
        "oracle_remove_only_measured_zero_token_commit_speedup"]) and
            close(impossible_scan_commit, bounds[
                "impossible_remove_all_six_module_scan_and_commit_speedup"]) and
            close(oracle_scaled, bounds[
                "oracle_zero_commit_compute_envelope_sensitivity"]) and
            close(impossible_scaled, bounds[
                "impossible_all_scan_commit_compute_envelope_sensitivity"]) and
            close(perfect_atlif, producer[
                "atlif_dynamic_replay_upper_bound"][
                    "perfect_remove_all_three_mapped_atlif_compute_envelope_ceiling"]),
            "M374 upper-bound arithmetic mismatch")

    model_text = paths["model_source"].read_text(encoding="utf-8")
    profile_text = paths["profile_source"].read_text(encoding="utf-8")
    semantic_source_checks = {
        "profile_disables_running_stats":
            "module.track_running_stats = False" in profile_text,
        "profile_erases_running_mean": "module.running_mean = None" in profile_text,
        "profile_erases_running_var": "module.running_var = None" in profile_text,
        "sew_residual_add_present": "out = out + identity" in model_text,
        "ped_shortcut_add_present": "x = (x+x_res)" in model_text,
        "producer_analyzer_imported": False,
    }
    require(all(value for key, value in semantic_source_checks.items()
                if key != "producer_analyzer_imported") and
            semantic_source_checks["producer_analyzer_imported"] is False,
            "semantic source attack failed")

    review = {
        "schema": "m375_m374_patch_zero_rf_independent_hammer_review_v1",
        "status": "PASS_INDEPENDENT_RECOMPUTE__CONFIRM_FAST_KILL__NO_RTL",
        "review_date": "2026-08-25",
        "reviewer_role": contract["reviewer_role"],
        "overall_score": 96,
        "scores": {
            "evidence_quality": {
                "score": 98,
                "maximum": 100,
                "components": {
                    "frozen_identity_and_exact_sha": "20/20",
                    "independent_60_payload_decode": "20/20",
                    "independent_padding_halo_rf_method": "20/20",
                    "per_record_and_tile_exact_match": "20/20",
                    "scope_correct_bound_recompute": "10/10",
                    "semantic_source_attack": "8/10"
                }
            },
            "hardware_admission": {
                "score": 12,
                "maximum": 100,
                "components": {
                    "exact_n0_population": "10/10",
                    "incremental_mac_or_weight_dma": "0/20",
                    "checkpoint_static_semantics": "0/20",
                    "useful_whole_t_population": "0/15",
                    "executable_cycles_or_rtl": "0/20",
                    "scope_complete_capture": "0/10",
                    "claim_boundary": "2/5"
                }
            }
        },
        "verdict": {
            "exact_per_timestep_zero_rf_population": "GO_7257197_OF_40320000",
            "exact_whole_temporal_zero_population": "GO_156_OF_4032000",
            "mapped_post_bn_atlif_replay_population": "NO_GO_ZERO_SITES",
            "aligned_2x2_4x4_8x8_tiles": "12_1_0_ALL_BEHIND_RESIDUAL_CONV2",
            "incremental_mac_or_active_weight_dma": "NO_GO_ALREADY_ZERO_IN_BIT_SPARSE_BASELINE",
            "checkpoint_static_constant": "NO_GO_CURRENT_BATCH_BN",
            "residual_or_ped_commit_elision": "NO_GO",
            "scope_correct_1p15_gate": "NO_GO",
            "rtl": "DO_NOT_WRITE"
        },
        "independent_recompute": {
            "producer_analyzer_imported": False,
            "payloads_rehashed": 60,
            "payload_bytes_rehashed": raw_payload_bytes,
            "per_record_scalar_mismatches": len(mismatches),
            "per_record_tile_mismatches": 0,
            "output_tokens": output_tokens,
            "zero_receptive_field_tokens": zero_tokens,
            "zero_receptive_field_fraction": float(zero_tokens) / output_tokens,
            "whole_temporal_spatial_sites": spatial_sites,
            "whole_temporal_zero_spatial_sites": whole_t,
            "whole_temporal_zero_fraction": float(whole_t) / spatial_sites,
            "whole_temporal_sites_outside_resblock0_conv2": 0,
            "module_rows": module_rows
        },
        "independent_bound_recompute": {
            "oracle_zero_commit_isolated_speedup": oracle_commit,
            "oracle_zero_commit_compute_envelope_sensitivity": oracle_scaled,
            "impossible_all_scan_commit_isolated_speedup": impossible_scan_commit,
            "impossible_all_scan_commit_compute_envelope_sensitivity": impossible_scaled,
            "perfect_delete_three_mapped_atlif_compute_envelope_ceiling": perfect_atlif,
            "promotion_gate": 1.15,
            "all_below_gate": max(oracle_scaled, impossible_scaled,
                                  perfect_atlif) < 1.15
        },
        "semantic_source_checks": semantic_source_checks,
        "findings": {
            "p0": [],
            "p1": [
                {
                    "id": "M375-P1-1",
                    "finding": "Frozen H67 uses no-running/current-batch BN, so the zero-Conv response depends on each sample's global T*B*H*W moments.",
                    "impact": "Checkpoint-static bias/BN/ATLIF replay is not exact; dynamic coefficients and a barrier are mandatory."
                },
                {
                    "id": "M375-P1-2",
                    "finding": "All three Conv-to-ATLIF mappings have zero whole-T empty spatial sites across the ten frozen samples.",
                    "impact": "Measured dynamic ATLIF response reuse is exactly zero, despite 17.999% per-timestep empty RFs."
                },
                {
                    "id": "M375-P1-3",
                    "finding": "The only 156 whole-T empty sites occur at resblock0.conv2, whose branch must still be normalized and added to the live residual identity.",
                    "impact": "They do not authorize output commit elision; 2x2/4x4 tile counts cannot become static-output replay."
                },
                {
                    "id": "M375-P1-4",
                    "finding": "M51 has exact RF inputs for six of eight Patch Conv modules and no address-timed halo-mask/BN/residual transaction trace.",
                    "impact": "No complete-Patch, SRAM/DRAM, executable-cycle, or energy claim is available."
                }
            ],
            "p2": [
                {
                    "id": "M375-P2-1",
                    "finding": "The bit-sparse baseline already issues exactly 1,774,268,587 active source/weight rows and zero for N=0 RFs.",
                    "impact": "Do not count N=0 as an additional MAC or PWP/weight-DMA saving."
                },
                {
                    "id": "M375-P2-2",
                    "finding": "Only 12 aligned 2x2 tiles, one 4x4 tile and zero 8x8 tiles are whole-T empty.",
                    "impact": "A halo-aware empty-tile descriptor has negligible measured population before its metadata and broadcast cost."
                },
                {
                    "id": "M375-P2-3",
                    "finding": "Oracle zero-commit, impossible all-scan/all-commit, and perfect three-ATLIF deletion sensitivities are 1.00107x, 1.01640x and 1.03858x on the frozen compute envelope.",
                    "impact": "Every disclosed upper bound is below 1.15x; fast-kill before RTL is correct."
                },
                {
                    "id": "M375-P2-4",
                    "finding": "M273r2 treats N=0 release as an illegal sticky fault rather than an exact response.",
                    "impact": "No existing empty-tile RTL protocol can be reused as proof of G10 behavior."
                }
            ]
        },
        "required_action": {
            "current_axis": "CLOSE_G10_NO_RTL_NO_GPU",
            "reopen_only_if": [
                "Patch normalization is retrained/validated as static or zero-preserving",
                "a new population produces whole-T empty sites on an ATLIF-mapped path",
                "halo-aware metadata plus BN/residual/address-timed transactions are captured",
                "a scope-correct executable upper bound exceeds 1.15x"
            ]
        },
        "claim_boundary": {
            "exact_n0_population": True,
            "exact_tile_population": True,
            "independent_recompute": True,
            "dynamic_bn_constant_replay": False,
            "hardware_cycles": False,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "energy": False,
            "system_speedup": False,
            "headline": False
        },
        "tool_execution": {
            "cpu_only": True,
            "gpu_invoked": False,
            "synopsys_invoked": False,
            "open_source_rtl_invoked": False
        },
        "producer_directory_modified": False,
        "docs359_modified": False,
        "docs359_sha256": sha256(paths["docs359"]),
        "identity": identities
    }
    recompute = {
        "schema": "m375_m374_independent_recompute_v1",
        "method": contract["independent_method"],
        "payloads_rehashed": 60,
        "payload_bytes_rehashed": raw_payload_bytes,
        "mismatches": mismatches,
        "per_record": per_record,
        "per_module": module_rows,
        "bounds": review["independent_bound_recompute"],
        "docs359_sha256": review["docs359_sha256"]
    }
    require(sha256(script) == script_start, "auditor changed during run")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "m375_m374_independent_recompute_r1.json").write_text(
        json.dumps(recompute, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (args.output_dir /
     "m375_m374_patch_zero_rf_independent_hammer_review_r1.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("PASS M375 payloads=60 mismatches=0 zero={} wholeT={} score=96 p0=0 p1=4 p2=4".format(
        zero_tokens, whole_t))


if __name__ == "__main__":
    main()
