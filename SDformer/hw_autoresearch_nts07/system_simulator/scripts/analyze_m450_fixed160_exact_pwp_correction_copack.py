#!/usr/bin/env python3
"""M450 fixed-160-byte exact PWP/correction atomic co-pack screen.

The screen never changes the M430 catalog and never reads the raw M40 payload.
It consumes the already sealed M430 per-phase heldout ledger exactly once.
The primary point is deliberately atomic: one PWP issue may eliminate one
separate correction issue only when the complete exact correction vector fits
in that same issue's unused physical payload.  Fragment pooling across rows,
destinations, blocks, or phases is not admitted as an executable point.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED = {
    "m430_contract": "261cb8fc3fec3d08570f55423da71188b3b8c17b5537f695309075d16f72c912",
    "m430_result": "6cf413e93d8159d9516ad048eaa26c741e49c2c9a3b330fb1d6dd20ba64dab2a",
    "m430_phase_csv": "0717e2c4ffd33cf95184df5acc2cb04751edbe42789f8b9d63ed5fbc6a20d006",
    "m430_static_codec": "4658f7f1dbfb64d4f3a7db13b8e29e8170c609889661f7514b1b17ec0aafbbb1",
    "m430_manifest": "ae5461ec7dbb39261c3631fa7aeccb0ffd7076bc60c84fd6b458d9ac30d7c893",
    "m430_seal": "462501b849f42f1a0690d2fe8dbe3dc226e83ae05dea86f7cb0396d60e9faf7e",
    "m430_catalog": "3ff522ff2296a021b005ca5733d846cc169560c125c8713c814b22a14d372f78",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "weight_o0": "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "weight_o1": "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "weight_o2": "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "weight_o3": "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
    "m433_rtl": "75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046",
    "m104_rtl": "7ea7978f431e917ee1a7835b8474af59e8f294587b1f115441388de8fb9c1ec5",
    "m442_seal": "43ee4f0aaa241e7e649c083438733a55da454b41c6c48b2b9fcabaada016209c",
    "m449_seal": "a7fe306a91a1efc7b05340fdfa4bfd859e9f7aa830db01e022b046e1fb14b96a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def signed_width(values):
    minimum = int(values.min())
    maximum = int(values.max())
    for bits in range(1, 9):
        if minimum >= -(1 << (bits - 1)) and maximum <= (1 << (bits - 1)) - 1:
            return bits
    raise RuntimeError("checkpoint weight vector exceeds signed INT8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M450 output overwrite")
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m450_fixed160_exact_pwp_correction_copack_contract_v1" and
            contract["status"] ==
            "FROZEN_BEFORE_SINGLE_PASS_SEALED_M430_HELDOUT_LEDGER",
            "M450 contract status drift")
    root = args.contract.resolve().parents[1]
    script_start = sha256(Path(__file__).resolve())
    require(contract["inputs"]["analyzer"]["sha256"] == script_start,
            "M450 analyzer self identity drift")

    paths = {
        "m430_contract": root / "contracts/m430b_h67_dualaware_q32_heldout_once_contract_r1_20260826.json",
        "m430_result": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/m430b_h67_dualaware_q32_heldout_r1.json",
        "m430_phase_csv": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/per_phase_heldout_dual_replay.csv",
        "m430_static_codec": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/static_codec_audit.csv",
        "m430_manifest": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/SHA256SUMS",
        "m430_seal": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/SHA256SUMS.seal.sha256",
        "m430_catalog": root / "results/m430a_trainonly_dualaware_q32_catalog_r1_20260826/m430_trainonly_dualaware_q32_catalog_r1.json",
        "m41_result": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/m41_h67_ep35_bottleneck_int8_bridge.json",
        "weight_o0": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o0_weight_i_ky_kx_o_s8.bin",
        "weight_o1": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o1_weight_i_ky_kx_o_s8.bin",
        "weight_o2": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o2_weight_i_ky_kx_o_s8.bin",
        "weight_o3": root / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o3_weight_i_ky_kx_o_s8.bin",
        "m433_rtl": root / "rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv",
        "m104_rtl": root / "rtl_m104/m104_held_weight_correction_broadcaster.sv",
        "m442_seal": root / "results/m442b_m430_full_static_codec_m433_vcs_r1_20260826/RUN_MANIFEST.seal.sha256",
        "m449_seal": root / "results/m449_m447_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256",
        "docs359": root / "docs/359_DATE终局冻结_20260813.md",
    }
    identities = {}
    for name, path in paths.items():
        actual = sha256(path)
        require(actual == EXPECTED[name], "M450 input SHA drift: " + name)
        identities[name] = {"path": str(path.relative_to(root)),
                            "sha256": actual}

    m430_contract = strict_json(paths["m430_contract"])
    m430 = strict_json(paths["m430_result"])
    m41 = strict_json(paths["m41_result"])
    require(m430["status"] ==
            "PASS_M430B_ONE_COMPLETED_M40_HELDOUT_DUAL_REPLAY",
            "M450 M430 status drift")
    require(m430["scope"] ==
            "four frozen H67 ep35 bottleneck Conv3x3 operators only",
            "M450 M430 scope drift")
    require(m430_contract["cycle_model"]
            ["dual_pwp_padded_signal_bytes_per_issue"] == 160 and
            m430_contract["cycle_model"]
            ["dual_pwp_logical_bytes_per_issue"] == 144 and
            m430_contract["cycle_model"]
            ["correction_bytes_per_issue"] == 96,
            "M450 input width model drift")

    # Enumerate both actual correction directions.  A shared negate bit acts on
    # one complete 96-lane vector, so the stored payload stays INT8.  The
    # sign-conditioned minimal bit width is nevertheless measured to give the
    # co-pack proposal every exact fixed-width advantage.
    width_rows = []
    width_histogram = Counter()
    reserved_negative_128 = 0
    global_minimum = 999
    global_maximum = -999
    for operator in range(4):
        weights = np.fromfile(paths[f"weight_o{operator}"], dtype=np.int8)
        require(weights.size == 6912 * 768,
                "M450 weight extent drift")
        values = weights.reshape(6912, 8, 96).astype(np.int16)
        reserved_negative_128 += int(np.count_nonzero(values == -128))
        global_minimum = min(global_minimum, int(values.min()))
        global_maximum = max(global_maximum, int(values.max()))
        for direction, multiplier in (("positive", 1), ("negative", -1)):
            for source_term in range(6912):
                for output_block in range(8):
                    vector = values[source_term, output_block] * multiplier
                    bits = signed_width(vector)
                    width_histogram[(operator, direction, bits)] += 1
    for (operator, direction, bits), count in sorted(width_histogram.items()):
        width_rows.append({
            "operator": operator,
            "direction": direction,
            "signed_bits_per_lane": bits,
            "exact_payload_bits": bits * 96,
            "exact_payload_bytes": bits * 12,
            "vector_cases": count,
            "fits_wide_pwp_16B_slack": int(bits * 12 <= 16),
            "fits_narrow_pwp_64B_slack": int(bits * 12 <= 64),
        })
    minimum_bits = min(bits for _, _, bits in width_histogram)
    maximum_bits = max(bits for _, _, bits in width_histogram)
    fitting_wide_cases = sum(count for (_, _, bits), count in
                             width_histogram.items() if bits * 12 <= 16)
    fitting_narrow_cases = sum(count for (_, _, bits), count in
                               width_histogram.items() if bits * 12 <= 64)
    require(reserved_negative_128 == 0 and global_minimum == -127 and
            global_maximum == 127 and minimum_bits == 6 and
            maximum_bits == 8 and fitting_wide_cases == 0 and
            fitting_narrow_cases == 0,
            "M450 correction-width proof drift")

    # Audit the complete static PWP codec population without using it to alter
    # the catalog.  Narrow consumes only the 96-byte low side; wide consumes
    # 96+48 logical bytes.  Both retain the frozen 160-byte physical envelope.
    static_rows = static_narrow = static_wide = 0
    with paths["m430_static_codec"].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            static_rows += 1
            if int(row["narrow"]):
                static_narrow += 1
            else:
                static_wide += 1
    require(static_rows == 442368 and static_narrow == 70503 and
            static_wide == 371865,
            "M450 static codec population drift")

    # Exactly one pass over the already sealed heldout phase ledger.  No raw
    # M40 packed/value payload is opened, preserving the M430 one-shot history.
    phase_count = 0
    heldout = Counter()
    with paths["m430_phase_csv"].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phase_count += 1
            for field in ("active_rows", "eligible_rows", "pwp_rows",
                          "exact_pwp_rows", "fallback_rows",
                          "correction_ops_per_block", "used_pwp_patterns",
                          "used_center_runs", "early_matcher"):
                heldout[field] += int(row[field])
    require(phase_count == 17280 and
            heldout["pwp_rows"] ==
                m430["runtime_population"]["pwp_rows"] and
            heldout["correction_ops_per_block"] ==
                m430["runtime_population"]["correction_ops_per_block"] and
            heldout["fallback_rows"] ==
                m430["runtime_population"]["fallback_rows"] and
            heldout["early_matcher"] ==
                m430["runtime_population"]["q32_early_matcher_cycles"],
            "M450 single-pass heldout ledger mismatch")

    pwp_issues = heldout["pwp_rows"] * 8
    correction_issues = heldout["correction_ops_per_block"] * 8
    require(pwp_issues ==
            m430["traffic_and_port_ledger"]["pwp_output_block_issues"] and
            correction_issues ==
            m430["traffic_and_port_ledger"]["correction_output_block_issues"],
            "M450 issue population drift")
    runtime_narrow = (
        m430["runtime_population"]["narrow_block_descriptors_tile0"] +
        m430["runtime_population"]["narrow_block_descriptors_tile1"])
    runtime_wide = pwp_issues - runtime_narrow
    require(runtime_narrow >= 0 and runtime_wide >= 0,
            "M450 runtime PWP split invalid")

    # Primary executable screen: no sign-conditioned vector fits even the
    # largest 64-byte atomic slack, so no separate correction issue disappears.
    atomic_copacked_correction_issues = 0
    m430_cycles = m430["comparisons"]["m430_catalog_dual_cycles"]
    strong_zero = m430["comparisons"]["strong_zero_cycles"]
    fixed160_cycles = m430_cycles - atomic_copacked_correction_issues
    speedup_vs_m430 = m430_cycles / fixed160_cycles
    speedup_vs_strong_zero = strong_zero / fixed160_cycles

    # Non-executable generosity bound: pool every payload slack byte globally,
    # ignore row/block/destination/phase identity, buffering, dual-destination
    # accumulation and fragment metadata, and charge no assembly cost.  Even
    # this deliberately impossible bound cannot reach the 1.10x decision gate.
    pooled_slack_bytes = runtime_narrow * 64 + runtime_wide * 16
    pooled_hidden_corrections = min(correction_issues,
                                    pooled_slack_bytes // 96)
    pooled_cycles = m430_cycles - pooled_hidden_corrections
    pooled_speedup = m430_cycles / pooled_cycles

    threshold = float(contract["decision_rule"]["minimum_speedup_vs_m430"])
    decision = ("GO_FIXED160_ATOMIC_COPACK" if speedup_vs_m430 >= threshold
                else "NO_GO_FIXED160_ATOMIC_COPACK_BELOW_1P10")
    require(decision == "NO_GO_FIXED160_ATOMIC_COPACK_BELOW_1P10" and
            pooled_speedup < threshold,
            "M450 frozen decision rule drift")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "m450_correction_vector_width_histogram.csv",
              width_rows,
              ["operator", "direction", "signed_bits_per_lane",
               "exact_payload_bits", "exact_payload_bytes", "vector_cases",
               "fits_wide_pwp_16B_slack",
               "fits_narrow_pwp_64B_slack"])
    result = {
        "schema": "m450_fixed160_exact_pwp_correction_copack_result_v1",
        "status": "PASS_M450_SINGLE_PASS_FIXED160_SCREEN",
        "decision": decision,
        "identity": identities,
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "packing_contract": {
            "total_physical_payload_bytes_per_cycle": 160,
            "metadata_sideband_is_not_charged_to_160B": True,
            "metadata_treatment": "most optimistic: reuse existing fixed sideband and charge zero payload bytes; no new signal is allowed",
            "pwp_low_payload_bits": 768,
            "pwp_low_payload_bytes": 96,
            "pwp_wide_high_logical_bits": 384,
            "pwp_wide_high_logical_bytes": 48,
            "pwp_wide_padding_bits": 128,
            "pwp_wide_atomic_slack_bytes": 16,
            "pwp_narrow_high_side_unused_bits": 512,
            "pwp_narrow_atomic_slack_bytes": 64,
            "pwp_low_identity_bits": 33,
            "pwp_high_duplicated_identity_bits": 33,
            "pwp_request_narrow_bits": 1,
            "correction_stored_weight_lanes": 96,
            "correction_stored_weight_bits_per_lane": 8,
            "correction_stored_payload_bits": 768,
            "correction_stored_payload_bytes": 96,
            "correction_shared_negate_bits": 1,
            "correction_m104_event_metadata_bits": 41,
            "correction_m104_event_metadata_fields":
                "source4+block3+negate1+last1+tag32",
            "correction_arithmetic_output_bits_per_lane": 12,
            "correction_arithmetic_output_payload_bits": 1152,
            "atomic_rule": "a separate correction issue is removed only if one complete exact sign-conditioned 96-lane vector fits in the unused payload of the same PWP issue",
            "fragment_pooling_admitted": False,
            "catalog_changed": False,
            "port_expanded": False,
        },
        "correction_width_audit": {
            "weight_global_minimum": global_minimum,
            "weight_global_maximum": global_maximum,
            "reserved_negative_128_count": reserved_negative_128,
            "sign_conditioned_vector_cases": sum(width_histogram.values()),
            "minimum_signed_bits_per_lane": minimum_bits,
            "maximum_signed_bits_per_lane": maximum_bits,
            "minimum_exact_vector_payload_bytes": minimum_bits * 12,
            "maximum_exact_vector_payload_bytes": maximum_bits * 12,
            "fits_wide_16B_cases": fitting_wide_cases,
            "fits_narrow_64B_cases": fitting_narrow_cases,
            "histogram": {
                str(bits): sum(count for (_, _, width), count in
                               width_histogram.items() if width == bits)
                for bits in range(minimum_bits, maximum_bits + 1)
            },
        },
        "pwp_population": {
            "static_codec_blocks": static_rows,
            "static_narrow_blocks": static_narrow,
            "static_wide_blocks": static_wide,
            "runtime_pwp_output_block_issues": pwp_issues,
            "runtime_narrow_pwp_issues": runtime_narrow,
            "runtime_wide_pwp_issues": runtime_wide,
        },
        "single_pass_heldout": {
            "raw_m40_payload_reads": 0,
            "sealed_phase_csv_passes": 1,
            "phases": phase_count,
            "source_rows": 51840000,
            "pwp_rows": heldout["pwp_rows"],
            "correction_ops_per_block": heldout["correction_ops_per_block"],
            "correction_output_block_issues": correction_issues,
            "atomic_copacked_correction_issues":
                atomic_copacked_correction_issues,
        },
        "cycle_points": {
            "strong_zero_cycles": strong_zero,
            "m430_separate_cycles": m430_cycles,
            "fixed160_atomic_copack_cycles": fixed160_cycles,
            "fixed160_atomic_copack_speedup_vs_m430": speedup_vs_m430,
            "fixed160_atomic_copack_speedup_vs_strong_zero":
                speedup_vs_strong_zero,
        },
        "non_executable_global_fragment_pooling_ceiling": {
            "admitted": False,
            "runtime_slack_bytes": pooled_slack_bytes,
            "maximum_hidden_96B_corrections": pooled_hidden_corrections,
            "optimistic_cycles": pooled_cycles,
            "optimistic_speedup_vs_m430": pooled_speedup,
            "reason_not_executable": "globally pools fragments across row/block/destination/phase and assumes free assembly plus free multi-destination compute/accumulation",
        },
        "decision_rule": {
            "minimum_speedup_vs_m430": threshold,
            "passes": speedup_vs_m430 >= threshold,
        },
        "claim_boundary": {
            "exact_width_and_trace_screen": True,
            "four_h67_bottleneck_conv_scope": True,
            "new_rtl": False,
            "rtl_measured_speedup": False,
            "resource_normalized_speedup": False,
            "sram_or_interconnect": False,
            "power_or_energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
    }
    result_path = args.output_dir / "m450_fixed160_exact_copack_result_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    audit = {
        "schema": "m450_single_pass_audit_v1",
        "phase_csv_passes": 1,
        "raw_payload_reads": 0,
        "phase_count": phase_count,
        "aggregates": dict(heldout),
        "input_script_sha256_start": script_start,
        "input_script_sha256_end": sha256(Path(__file__).resolve()),
        "contract_sha256_start": sha256(args.contract),
        "contract_sha256_end": sha256(args.contract),
    }
    require(audit["input_script_sha256_start"] ==
            audit["input_script_sha256_end"],
            "M450 analyzer mutated during run")
    (args.output_dir / "m450_single_pass_audit_r1.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evidence = [result_path,
                args.output_dir / "m450_correction_vector_width_histogram.csv",
                args.output_dir / "m450_single_pass_audit_r1.json"]
    manifest = args.output_dir / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in evidence),
        encoding="utf-8")
    seal = args.output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text(f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "strong_zero_cycles": strong_zero,
        "m430_separate_cycles": m430_cycles,
        "fixed160_atomic_copack_cycles": fixed160_cycles,
        "speedup_vs_m430": speedup_vs_m430,
        "minimum_correction_payload_bytes": minimum_bits * 12,
        "maximum_atomic_slack_bytes": 64,
        "pooled_ceiling_speedup_vs_m430": pooled_speedup,
        "output_dir": str(args.output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
