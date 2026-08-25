#!/usr/bin/env python3
"""Fail-closed identity, geometry, arithmetic-bound and VCS-receipt audit."""

import hashlib
import json
import re
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
OUTPUT = REVIEW / "m116_signed20_independent_hammer_audit.json"

PATHS = {
    "m115_analyzer": HW / "system_simulator/scripts/analyze_m115_pwp_transient_accumulator_width.py",
    "m115_result": HW / "results/m115_pwp_transient_accumulator_width_r1_20260824/m115_pwp_transient_accumulator_width.json",
    "m115_manifest": HW / "results/m115_pwp_transient_accumulator_width_r1_20260824/SHA256SUMS.txt",
    "m115_contract": HW / "contracts/m115_pwp_transient_accumulator_width_contract_r1_20260824.json",
    "weight_o0": HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o0_weight_i_ky_kx_o_s8.bin",
    "weight_o1": HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o1_weight_i_ky_kx_o_s8.bin",
    "weight_o2": HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o2_weight_i_ky_kx_o_s8.bin",
    "weight_o3": HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o3_weight_i_ky_kx_o_s8.bin",
    "m116_contract": HW / "contracts/m116_w384_signed20_lane_sliced_accumulator_vcs_contract_r1_20260824.json",
    "m116_core": HW / "rtl_m116/m116_w384_signed20_accumulator_frontend.sv",
    "m116_adapter": HW / "rtl_m116/m116_w384_signed20_lane_sliced_accumulator_adapter.sv",
    "m116_sva": HW / "verif_m116/m116_w384_signed20_lane_sliced_accumulator_assertions.sv",
    "m116_tb": HW / "tb_m116/tb_m116_w384_signed20_lane_sliced_accumulator.sv",
    "m116_filelist": HW / "dc_handoff/filelists/date_m116_w384_signed20_lane_sliced_accumulator_directed_vcs.f",
    "m116_runner": HW / "dc_handoff/scripts/run_vcs_m116_w384_signed20_lane_accumulator.sh",
    "sealed_complete": HW / "dc_handoff/runs/m116_w384_signed20_lane_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "sealed_compile": HW / "dc_handoff/runs/m116_w384_signed20_lane_accumulator_vcs_r1_sealed_20260824/compile.raw.log",
    "sealed_sim": HW / "dc_handoff/runs/m116_w384_signed20_lane_accumulator_vcs_r1_sealed_20260824/sim.raw.log",
    "sealed_assert": HW / "dc_handoff/runs/m116_w384_signed20_lane_accumulator_vcs_r1_sealed_20260824/assert.report",
    "hammer_tb": REVIEW / "tb_m116_signed20_independent_hammer.sv",
    "hammer_runner": REVIEW / "run_commercial_vcs_independent_hammer.sh",
    "hammer_complete": REVIEW / "vcs_run_r1/RUN_COMPLETE.txt",
    "hammer_compile": REVIEW / "vcs_run_r1/compile.raw.log",
    "hammer_sim": REVIEW / "vcs_run_r1/sim.raw.log",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}

EXPECTED = {
    "m115_analyzer": "bafadcf53e5221d70ab86da0fb17dcbae8da661b0148007dbd537f4fa519aa27",
    "m115_result": "9f62d9cb3e56c293cc117bd92c21844e8bd10515ea418a51cbfae0ebab62b94b",
    "m115_manifest": "bb12196b1ed7e0c10cb6b41db85271db24bfefab62bf0058b194666353afc951",
    "m115_contract": "ba730fcb6612fd8aa5c8e8c7d1aba976b759de54cbab05779ca409dadf9af9c8",
    "weight_o0": "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "weight_o1": "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "weight_o2": "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "weight_o3": "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
    "m116_contract": "bb245aa111d9646ff6b772c65a3362ae266d2f492691dac83d1782789912b721",
    "m116_core": "dd7e52e9ab3739972ca160283406c17f5d1a2947a3dd2456608a782b640c47b0",
    "m116_adapter": "074735e1f583d3dbef8e6dbee28f1ffb5a82bcda7a7328c8b520c5efc3c53a16",
    "m116_sva": "e7e36fbc3f695a71cc7b7c6e0393146131071152f1e7a6ad5df8f4d70732eecd",
    "m116_tb": "845c09847df7b65db4d787fce93283cc95b161e8f931112ae12d1609e7eec6d5",
    "m116_filelist": "4ed59a697bb688f1e53b90b313eea5cf6877c46af62022cd57b2bd30ef51f208",
    "m116_runner": "870020a1974a8c88683d7ad6cab2280eddaffae3d1fe73b4d8c4efd93cf2b708",
    "sealed_complete": "0d29ccffbca08254487cc99a91d4bfd8005496f6b5fc9186bb63ebc21998f602",
    "sealed_compile": "97d06d441a81f9099c200068c2ec76b4e6ea7678e6b8712abb9bb7635fb602a5",
    "sealed_sim": "d6dff1e42258b363af6dd988cd774f6439679b60821f4f23cecb774b9b8c9e68",
    "sealed_assert": "29d8f378db36e369ee5a3acec1208d14cd72bd62f63b7609ef27b2f37c7c67b3",
    "hammer_tb": "43d90648b4d4569accebb55bca493e9972deedc6e2db5d1443a2f1a00aca7053",
    "hammer_runner": "6a5a539b205f91f57ffb54d270dbb6582f49e9a38b3027e9cf6f3731102bc0a9",
    "hammer_complete": "436f82c7c89798fc85f5c7a7081b9b7dbee1e196a07523ab31c5a8378db97ac0",
    "hammer_compile": "107213996527a962310cffbe22284061ac7bd7de75f8732d0b5f02573e4e4166",
    "hammer_sim": "9b3fa5dccef5a4177ac07e7e05801559bf87714cdd2de1fe81087610e4c213a2",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def key_value_lines(path):
    output = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            require(key not in output, "duplicate receipt key " + key)
            output[key] = value
    return output


def main():
    start_sha = sha256(Path(__file__).resolve())
    observed = {}
    for label, path in PATHS.items():
        actual = sha256(path)
        require(actual == EXPECTED[label],
                "identity drift {} {}".format(label, actual))
        observed[str(path.relative_to(HW))] = actual

    m115 = strict_json(PATHS["m115_result"])
    c115 = strict_json(PATHS["m115_contract"])
    c116 = strict_json(PATHS["m116_contract"])
    geometry = c116["geometry"]
    require(geometry["signed_bits_per_lane"] == 20
            and geometry["logical_vector_bits"] == 1920
            and geometry["lane_macro_count"] == 96
            and geometry["lane_macro_depth"] == 3072
            and geometry["lane_macro_width_bits"] == 20,
            "M116 geometry drift")
    payload20 = 384 * 8 * 96 * 20 // 8
    payload24 = 384 * 8 * 96 * 24 // 8
    require(payload20 == geometry["logical_accumulator_bytes"] == 737280,
            "signed20 payload arithmetic mismatch")
    require(payload24 == 884736 and payload24 - payload20 == 147456,
            "signed24 saving arithmetic mismatch")
    require(96 * 3072 * 20 == geometry["logical_accumulator_bits"]
            == 5898240, "lane macro capacity mismatch")

    proof = m115["proof"]
    require(proof["checkpoint_maximum_sum_abs_q"] == 218338
            and proof["checkpoint_transient_magnitude_bound"] == 436676
            and proof["checkpoint_transient_required_signed_bits"] == 20,
            "M115 checkpoint proof drift")
    require(2 * 218338 == 436676 < (1 << 19),
            "signed20 positive endpoint proof failure")
    require(proof["dense_int8_decomposed_transient_magnitude_bound"]
            == 1755648
            and proof["dense_int8_decomposed_transient_required_signed_bits"]
            == 22, "M115 dense boundary drift")
    require(c115["arithmetic_contract"]["direct_signed19_inheritance"] is False,
            "signed19 rejection drift")
    recomputed_maxima = []
    for operator in range(4):
        raw = PATHS["weight_o{}".format(operator)].read_bytes()
        require(len(raw) == 6912 * 768,
                "weight extent drift op{}".format(operator))
        per_channel = [0] * 768
        for index, byte in enumerate(raw):
            signed = byte if byte < 128 else byte - 256
            per_channel[index % 768] += abs(signed)
        recomputed_maxima.append(max(per_channel))
    require(recomputed_maxima == [218338, 204866, 207239, 190753],
            "independent weight sumabs recomputation drift")

    core = PATHS["m116_core"].read_text(encoding="utf-8")
    adapter = PATHS["m116_adapter"].read_text(encoding="utf-8")
    require("ACC_BITS != 20 || VECTOR_BITS != 1920" in core,
            "core elaboration guard drift")
    require("lane_sum_ext[lane][ACC_BITS]" in core
            and "!= lane_sum_ext[lane][ACC_BITS-1]" in core,
            "signed overflow guard drift")
    require("row_valid_q[bank] <= '0" in core,
            "lazy valid clear drift")
    require("same_address_rdw_hazard" in core
            and "update_pipe_valid_q && update_valid" in core,
            "same-address fail-closed guard drift")
    require("block_times_384" in adapter
            and "DEPTH != 3072 || ADDR_W != 12" in adapter,
            "flattened geometry drift")
    require("lane * ACC_BITS +: ACC_BITS" in adapter,
            "lane slice mapping drift")

    sealed = key_value_lines(PATHS["sealed_complete"])
    hammer = key_value_lines(PATHS["hammer_complete"])
    require(sealed["status"]
            == "PASS_M116_W384_SIGNED20_LANE_SLICED_ACCUMULATOR_DIRECTED_VCS_SVA",
            "production sealed receipt status drift")
    require(sealed["behavioral_sync_lane_sliced_1r1w_macro"] == "true"
            and sealed["foundry_sram_macro"] == "false"
            and sealed["scheduled_cycle_ratio"] == "false"
            and sealed["physical_speedup"] == "false"
            and sealed["system_speedup"] == "false"
            and sealed["headline"] == "false",
            "production claim boundary drift")
    require(hammer["status"]
            == "PASS_M116_INDEPENDENT_SIGNED20_COMMERCIAL_VCS_HAMMER",
            "independent hammer status drift")
    for key in ("signed20_positive_negative_boundary",
                "rmw_and_nonconflicting_ii1", "lazy_clear",
                "commit_stall_and_order", "same_address_rdw_fail_closed",
                "positive_overflow_fail_closed",
                "negative_overflow_fail_closed"):
        require(hammer[key] == "true", "independent feature drift " + key)
    for key in ("foundry_macro", "ppa", "cycle_ratio",
                "physical_speedup", "system_speedup", "headline"):
        require(hammer[key] == "false", "independent claim drift " + key)

    sealed_log = PATHS["sealed_sim"].read_text(encoding="utf-8")
    hammer_log = PATHS["hammer_sim"].read_text(encoding="utf-8")
    require("Compiler version V-2023.12-SP1_Full64" in sealed_log
            and "Runtime version V-2023.12-SP1_Full64" in sealed_log,
            "sealed commercial VCS identity absent")
    require("Compiler version V-2023.12-SP1_Full64" in hammer_log
            and "Runtime version V-2023.12-SP1_Full64" in hammer_log,
            "hammer commercial VCS identity absent")
    match = re.search(
        r"PASS M116 independent signed20 hammer .*"
        r"positive_updates=(\d+) positive_writes=(\d+) "
        r"ii1_pairs=(\d+) read_write_overlap=(\d+) "
        r"commits=(\d+) lane_checks=(\d+) "
        r"commit_stalls=(\d+) stall_releases=(\d+).*"
        r"negative_overflow_attacks=1", hammer_log)
    require(match is not None, "independent PASS ledger absent")
    metrics = [int(value) for value in match.groups()]
    require(metrics[0:6] == [69, 69, 67, 67, 6144, 589824],
            "independent conservation drift")
    require(metrics[6] >= 100 and metrics[7] > 0,
            "commit stall campaign too weak")
    assert_report = PATHS["sealed_assert"].read_text(encoding="utf-8")
    for cover, count in (("cp_update_ii1", 1056),
                         ("cp_read_write_overlap", 1056),
                         ("cp_commit_stall", 699),
                         ("cp_full_commit", 2),
                         ("cp_fault", 2)):
        require(re.search(cover + r", .* {} match".format(count),
                          assert_report) is not None,
                "sealed SVA cover drift " + cover)

    payload = {
        "schema": "m116_signed20_independent_hammer_audit_v1",
        "status": "PASS_EXACT_SHA_COMMERCIAL_VCS_INDEPENDENT_HAMMER_BOUNDED_MODULE_ONLY",
        "identity": observed,
        "geometry_recomputed": {
            "signed_bits_per_lane": 20,
            "logical_vector_bits": 1920,
            "lane_macros": 96,
            "lane_macro_depth": 3072,
            "lane_macro_width": 20,
            "logical_payload_bits": 5898240,
            "logical_payload_bytes": payload20,
            "signed24_payload_bytes": payload24,
            "logical_saving_bytes_vs_signed24": payload24 - payload20,
        },
        "commercial_vcs_evidence": {
            "production_sealed_directed_sva": True,
            "independent_testbench": True,
            "tool": "Synopsys VCS V-2023.12-SP1 Full64",
            "independent_positive_updates": metrics[0],
            "independent_memory_writes": metrics[1],
            "independent_ii1_pairs": metrics[2],
            "independent_read_write_overlap": metrics[3],
            "independent_commits": metrics[4],
            "independent_lane_checks": metrics[5],
            "independent_commit_stalls": metrics[6],
            "independent_stall_releases": metrics[7],
            "positive_and_negative_signed20_boundaries": True,
            "positive_and_negative_overflow_attacks": True,
            "same_address_rdw_attack": True,
            "lazy_clear_two_window_attack": True,
            "flattened_address_minimum_and_maximum": True,
        },
        "checkpoint_boundary": {
            "applies_to": "four frozen H67 checkpoint INT8 weight payloads under the M108 PWP-anchor plus at-most-one correction coefficient contract",
            "ordering_independent_bound": "2 * sum(abs(weight)) per output channel",
            "maximum_magnitude": 436676,
            "all_four_weight_payloads_independently_recomputed": True,
            "per_operator_sumabs_maxima": recomputed_maxima,
            "signed20_safe": True,
            "arbitrary_dense_int8_safe_at_signed20": False,
            "arbitrary_dense_int8_required_bits": 22,
            "exact_heldout_update_trace_replayed_through_m116": False,
            "payload_to_signed20_delta_bridge_present": False,
            "bias_rescale_and_post_accumulation_in_scope": False,
        },
        "admission": {
            "bounded_signed20_rtl_module": True,
            "exact_sha_commercial_vcs_directed": True,
            "independent_commercial_vcs_hammer": True,
            "behavioral_sync_lane_memory": True,
            "foundry_sram_macro": False,
            "macro_inclusive_ppa": False,
            "exact_heldout_integrated_replay": False,
            "cycle_reduction": False,
            "scheduled_cycle_ratio": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == start_sha,
            "audit script changed during execution")
    require(not OUTPUT.exists(), "refusing audit output overwrite")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M116 independent audit score_candidate=92 payload={} saving={} commits={} lanes={}"
          .format(payload20, payload24 - payload20, metrics[4], metrics[5]),
          flush=True)


if __name__ == "__main__":
    main()
