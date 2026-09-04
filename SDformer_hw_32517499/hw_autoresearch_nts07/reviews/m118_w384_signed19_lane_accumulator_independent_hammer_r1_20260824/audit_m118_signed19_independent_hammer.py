#!/usr/bin/env python3
"""Fail-closed independent audit of M118 signed19 RTL and VCS evidence."""

import hashlib
import itertools
import json
import re
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
OUTPUT = REVIEW / "m118_signed19_independent_hammer_audit.json"
R2 = HW / "dc_handoff/runs/m118_w384_signed19_lane_accumulator_vcs_r2_sealed_20260824"
R1 = HW / "dc_handoff/runs/m118_w384_signed19_lane_accumulator_vcs_r1_sealed_20260824"

PATHS = {
    "m115r2_analyzer": HW / "system_simulator/scripts/analyze_m115r2_pwp_prefix_coefficient_width.py",
    "m115r2_result": HW / "results/m115r2_pwp_prefix_coefficient_width_r1_20260824/m115r2_pwp_prefix_coefficient_width.json",
    "m115r2_manifest": HW / "results/m115r2_pwp_prefix_coefficient_width_r1_20260824/SHA256SUMS.complete_r1.txt",
    "m115r2_contract": HW / "contracts/m115r2_pwp_prefix_coefficient_width_contract_r1_20260824.json",
    "weight_o0": HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o0_weight_i_ky_kx_o_s8.bin",
    "weight_o1": HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o1_weight_i_ky_kx_o_s8.bin",
    "weight_o2": HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o2_weight_i_ky_kx_o_s8.bin",
    "weight_o3": HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o3_weight_i_ky_kx_o_s8.bin",
    "m118_contract": HW / "contracts/m118_w384_signed19_lane_sliced_accumulator_vcs_contract_r1_20260824.json",
    "m118_core": HW / "rtl_m118/m118_w384_signed19_accumulator_frontend.sv",
    "m118_adapter": HW / "rtl_m118/m118_w384_signed19_lane_sliced_accumulator_adapter.sv",
    "m118_sva": HW / "verif_m118/m118_w384_signed19_lane_sliced_accumulator_assertions.sv",
    "m118_tb": HW / "tb_m118/tb_m118_w384_signed19_lane_sliced_accumulator.sv",
    "m118_filelist": HW / "dc_handoff/filelists/date_m118_w384_signed19_lane_sliced_accumulator_directed_vcs.f",
    "m118_runner": HW / "dc_handoff/scripts/run_vcs_m118_w384_signed19_lane_accumulator.sh",
    "r2_complete": R2 / "RUN_COMPLETE.txt",
    "r2_compile": R2 / "compile.raw.log",
    "r2_sim": R2 / "sim.raw.log",
    "r2_assert": R2 / "assert.report",
    "r1_failed": R1 / "RUN_FAILED_OR_INCOMPLETE.txt",
    "r1_compile": R1 / "compile.raw.log",
    "r1_sim": R1 / "sim.raw.log",
    "r1_assert": R1 / "assert.report",
    "hammer_tb": REVIEW / "tb_m118_signed19_independent_hammer.sv",
    "hammer_runner": REVIEW / "run_commercial_vcs_independent_hammer.sh",
    "hammer_complete": REVIEW / "vcs_run_r1/RUN_COMPLETE.txt",
    "hammer_compile": REVIEW / "vcs_run_r1/compile.raw.log",
    "hammer_sim": REVIEW / "vcs_run_r1/sim.raw.log",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}

EXPECTED = {
    "m115r2_analyzer": "2f3512f2c664daea6430c1360838c7496228b49ae2dd5a648db9af361fbf0f31",
    "m115r2_result": "b0e7fbb0573473ad854ca856d5eab3eaf15af1ba79ea2ce3a958810575bc6708",
    "m115r2_manifest": "6b9af5e9e7de61edc770e1d4d738d6c0b0070e7947f6aec12633da7181f96326",
    "m115r2_contract": "9edd6aac10186e24f21fffa5ce1b5a28da292258ad30df1d6934a7b1d1927eec",
    "weight_o0": "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "weight_o1": "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "weight_o2": "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "weight_o3": "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
    "m118_contract": "c79f55a15e03bbf26c22e9da2f0eb35d53b1a9795ab02b24a6b3c951c729903e",
    "m118_core": "0903a295f056f69067792e20f40acdae5cb8a38471c4be82077bf5b0c086d482",
    "m118_adapter": "cbccbe2611f7be31c305fb4032c0d518bc7eb51025b6d66cecc157693b6554af",
    "m118_sva": "ccea5ca611265c4970ceda9dee7d714ba154730102940931c3549473d186b07c",
    "m118_tb": "3f084d0c3a406dbdb36d82f0230c3e6f4e2e194fe6d43224f982288d6ab3d66c",
    "m118_filelist": "a5042955a8dc9eae93b61aa1ba14bb2a93a79b6791504dc3e04bbc53bf811af0",
    "m118_runner": "b8cac09b65f4c239d4dfa0ba915c0d15a1e1af81744a135f985fb8cdcb90d367",
    "r2_complete": "f45baa3c322a439377aa9c0c3e919440020294c9392b81343c7fae1bc1e605ff",
    "r2_compile": "6b5b7bff8b85bcc2168e6595f107a98af2be43d58d42dde8b806b4022bac88f3",
    "r2_sim": "363390673687c49de8ef486d98e8176c663fd3afe88ebef2af13f9765b7ec039",
    "r2_assert": "8971afe86225e56c26ad2e628369edac57d1dd7f39472afbd07c420b16623fe8",
    "r1_failed": "a1bbaa0205b4cbe7d793e5525ca93da242f0e14e11e64eb7383903559c0126a0",
    "r1_compile": "ac177d68c5167c34498a2ec4e93c33908bcef975ace3a96e796e4cfe135f7b1c",
    "r1_sim": "98cf86aa76ffd9f2bba744b34feab100aef2b158fdfd46efc8c297946681b4bc",
    "r1_assert": "8971afe86225e56c26ad2e628369edac57d1dd7f39472afbd07c420b16623fe8",
    "hammer_tb": "06ec7e609fa64723918b74fc691ef9ecabcc3f0fd129b7cf901f2ef2d14c7c84",
    "hammer_runner": "857e9df508d6aafc969f7a381da64d54be2c95f76cfea910a9fce250c5be4685",
    "hammer_complete": "9fdc54442c289d4a8db4161b2906d69a4911e69d2e068c7ba96d733bbb5ce266",
    "hammer_compile": "17fa48c7d91c3bad5c4913096178c2cdb7083a751a34d7049c00912eaea31440",
    "hammer_sim": "1692d68ad76ce4c0b0095b2922ad3704de6253cbf6b27ca310233c2469dc6054",
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
        raise ValueError("non-standard JSON constant " + raw)

    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def receipt(path):
    output = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            require(key not in output, "duplicate receipt key " + key)
            output[key] = value
    return output


def prefix_case(center, target):
    operations = []
    if center:
        operations.append(1)
    if target and not center:
        operations.append(1)
    if center and not target:
        operations.append(-1)
    orders = sorted(set(itertools.permutations(operations)))
    if not orders:
        orders = [()]
    rows = []
    for order in orders:
        prefixes = [0]
        for coefficient in order:
            prefixes.append(prefixes[-1] + coefficient)
        require(prefixes[-1] == target, "prefix final mismatch")
        rows.append({"order": list(order), "prefixes": prefixes})
    return rows


def main():
    start_sha = sha256(Path(__file__).resolve())
    observed = {}
    for label, path in PATHS.items():
        actual = sha256(path)
        require(actual == EXPECTED[label],
                "identity mismatch {} {}".format(label, actual))
        observed[str(path.relative_to(HW))] = actual

    c115 = strict_json(PATHS["m115r2_contract"])
    r115 = strict_json(PATHS["m115r2_result"])
    c118 = strict_json(PATHS["m118_contract"])
    require(c115["admission"]["mathematical_prefix_bound"] is True
            and c115["admission"][
                "integrated_accepted_transaction_exact_once_miter"] is False,
            "M115-r2 mathematical/integrated boundary drift")
    require(r115["checkpoint"]["mathematical_candidate_signed_bits"] == 19
            and r115["prefix_coefficient_proof"][
                "maximum_absolute_prefix_coefficient"] == 1,
            "M115-r2 candidate drift")

    prefix_rows = {}
    max_prefix = 0
    for center in (0, 1):
        for target in (0, 1):
            rows = prefix_case(center, target)
            prefix_rows["{}{}".format(center, target)] = rows
            for row in rows:
                max_prefix = max(max_prefix,
                                 *(abs(value) for value in row["prefixes"]))
    require(max_prefix == 1, "independent prefix enumeration drift")

    maxima = []
    for operator in range(4):
        raw = PATHS["weight_o{}".format(operator)].read_bytes()
        require(len(raw) == 6912 * 768,
                "weight extent drift op{}".format(operator))
        per_channel = [0] * 768
        for index, byte in enumerate(raw):
            signed = byte if byte < 128 else byte - 256
            per_channel[index % 768] += abs(signed)
        maxima.append(max(per_channel))
    require(maxima == [218338, 204866, 207239, 190753],
            "weight sumabs drift")
    require(max(maxima) == 218338 < (1 << 18),
            "checkpoint bound does not fit signed19")

    payload19_bits = 384 * 8 * 96 * 19
    payload19_bytes = payload19_bits // 8
    payload24_bytes = 384 * 8 * 96 * 24 // 8
    descriptor_bits = 2 * 128 * 384 * 2
    metadata_bits = 314
    valid_bits = 8 * 384
    combined_bits = payload19_bits + descriptor_bits + metadata_bits + valid_bits
    combined_bytes = (combined_bits + 7) // 8
    geometry = c118["geometry"]
    require(payload19_bits == 5603328
            and payload19_bytes == geometry["logical_accumulator_bytes"]
            == 700416, "signed19 payload geometry mismatch")
    require(96 * 3072 * 19 == payload19_bits
            and geometry["logical_vector_bits"] == 1824,
            "lane/vector geometry mismatch")
    require(combined_bytes == geometry[
                "combined_descriptor_valid_accumulator_logical_bytes"]
            == 725416, "combined logical storage mismatch")
    require(payload24_bytes == 884736
            and payload24_bytes - payload19_bytes == 184320,
            "signed24 saving mismatch")

    core = PATHS["m118_core"].read_text(encoding="utf-8")
    adapter = PATHS["m118_adapter"].read_text(encoding="utf-8")
    require("ACC_BITS != 19 || VECTOR_BITS != 1824" in core,
            "core geometry guard drift")
    require("same_address_rdw_hazard" in core
            and "lane_sum_ext[lane][ACC_BITS]" in core
            and "!= lane_sum_ext[lane][ACC_BITS-1]" in core,
            "RMW/overflow guards drift")
    require("row_valid_q[bank] <= '0" in core,
            "lazy clear drift")
    require("block_times_384" in adapter
            and "DEPTH != 3072 || ADDR_W != 12" in adapter
            and "lane * ACC_BITS +: ACC_BITS" in adapter,
            "lane mapping drift")

    r2 = receipt(PATHS["r2_complete"])
    require(r2["status"]
            == "PASS_M118_W384_SIGNED19_LANE_SLICED_ACCUMULATOR_DIRECTED_VCS_SVA",
            "r2 sealed receipt drift")
    require(r2["integrated_accepted_transaction_exact_once_miter"] == "false"
            and r2["foundry_sram_macro"] == "false"
            and r2["scheduled_cycle_ratio"] == "false"
            and r2["physical_speedup"] == "false"
            and r2["system_speedup"] == "false"
            and r2["headline"] == "false", "r2 claim boundary drift")
    r2_log = PATHS["r2_sim"].read_text(encoding="utf-8")
    require("Compiler version V-2023.12-SP1_Full64" in r2_log
            and "Runtime version V-2023.12-SP1_Full64" in r2_log,
            "r2 commercial VCS identity absent")
    assert_report = PATHS["r2_assert"].read_text(encoding="utf-8")
    for cover, count in (("cp_update_ii1", 1058),
                         ("cp_read_write_overlap", 1058),
                         ("cp_commit_stall", 699),
                         ("cp_full_commit", 2),
                         ("cp_fault", 3)):
        require(re.search(cover + r", .* {} match".format(count),
                          assert_report) is not None,
                "r2 cover mismatch " + cover)

    r1 = receipt(PATHS["r1_failed"])
    require(r1["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            and r1["runner_exit_code"] == "1",
            "r1 failure boundary drift")
    require(not (R1 / "RUN_COMPLETE.txt").exists(),
            "r1 unexpectedly has citable completion")
    require((R1 / "compile.rc").read_text(encoding="utf-8").strip() == "0"
            and (R1 / "sim.rc").read_text(encoding="utf-8").strip() == "0"
            and "PASS M118 W384" in PATHS["r1_sim"].read_text(encoding="utf-8"),
            "r1 was not a post-simulation seal failure")

    hammer = receipt(PATHS["hammer_complete"])
    require(hammer["status"]
            == "PASS_M118_INDEPENDENT_SIGNED19_COMMERCIAL_VCS_HAMMER",
            "independent hammer receipt drift")
    require(hammer["integrated_accepted_transaction_exact_once_miter"]
            == "false" and hammer["r1_receipt_citable"] == "false",
            "independent boundary drift")
    for key in ("foundry_macro", "ppa", "cycle_ratio",
                "physical_speedup", "system_speedup", "headline"):
        require(hammer[key] == "false", "hammer claim drift " + key)
    hammer_log = PATHS["hammer_sim"].read_text(encoding="utf-8")
    require("Compiler version V-2023.12-SP1_Full64" in hammer_log
            and "Runtime version V-2023.12-SP1_Full64" in hammer_log,
            "hammer commercial VCS identity absent")
    match = re.search(
        r"PASS M118 independent signed19 hammer .*"
        r"positive_updates=(\d+) positive_writes=(\d+) "
        r"ii1_pairs=(\d+) read_write_overlap=(\d+) "
        r"commits=(\d+) lane_checks=(\d+) "
        r"commit_stalls=(\d+) stall_releases=(\d+).*"
        r"negative_overflow_attacks=1", hammer_log)
    require(match is not None, "independent PASS ledger absent")
    metrics = [int(value) for value in match.groups()]
    require(metrics[:6] == [37, 37, 35, 35, 6144, 589824],
            "independent conservation mismatch")
    require(metrics[6] >= 100 and metrics[7] > 0,
            "independent backpressure campaign weak")

    output = {
        "schema": "m118_signed19_independent_hammer_audit_v1",
        "status": "PASS_STANDALONE_SIGNED19_RTL_P0_INTEGRATED_EXACT_ONCE_REMAINS_FALSE",
        "identity": observed,
        "m115r2_boundary": {
            "mathematical_candidate": True,
            "prefix_cases_and_orders": prefix_rows,
            "maximum_absolute_prefix_coefficient": max_prefix,
            "all_four_weights_recomputed": True,
            "per_operator_sumabs_maxima": maxima,
            "checkpoint_bound": 218338,
            "checkpoint_signed19_candidate": True,
            "dense_int8_required_bits": 21,
            "integrated_accepted_transaction_exact_once_miter": False,
            "duplicate_or_replayed_accepted_operation_invalidates_bound": True,
        },
        "geometry_recomputed": {
            "signed_bits_per_lane": 19,
            "logical_vector_bits": 1824,
            "lane_geometry": "96x3072x19",
            "logical_payload_bits": payload19_bits,
            "logical_payload_bytes": payload19_bytes,
            "combined_descriptor_valid_accumulator_bits": combined_bits,
            "combined_logical_bytes": combined_bytes,
            "signed24_payload_bytes": payload24_bytes,
            "logical_saving_bytes_vs_signed24": payload24_bytes - payload19_bytes,
        },
        "sealed_run_audit": {
            "r2_citable": True,
            "r2_exact_sha": True,
            "r2_tool": "Synopsys VCS V-2023.12-SP1 Full64",
            "r2_assert_cover_matches": {
                "cp_update_ii1": 1058,
                "cp_read_write_overlap": 1058,
                "cp_commit_stall": 699,
                "cp_full_commit": 2,
                "cp_fault": 3,
            },
            "r1_citable": False,
            "r1_status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
            "r1_compile_and_sim_returned_zero": True,
            "r1_failure_stage": "post-simulation receipt/seal gate",
            "r1_has_run_complete": False,
        },
        "independent_commercial_vcs": {
            "separate_testbench": True,
            "production_runner_invoked": False,
            "positive_updates": metrics[0],
            "positive_memory_writes": metrics[1],
            "ii1_pairs": metrics[2],
            "read_write_overlap": metrics[3],
            "commits": metrics[4],
            "lane_checks": metrics[5],
            "commit_stalls": metrics[6],
            "stall_releases": metrics[7],
            "signed19_maximum_and_minimum": True,
            "positive_and_negative_overflow": True,
            "same_address_rdw_fail_closed": True,
            "lazy_clear": True,
            "address_zero_and_3071": True,
            "full_commit_order": True,
        },
        "admission": {
            "standalone_signed19_accumulator_rtl": True,
            "exact_sha_production_commercial_vcs_r2": True,
            "independent_commercial_vcs_hammer": True,
            "integrated_accepted_transaction_exact_once_miter": False,
            "foundry_sram_macro": False,
            "macro_inclusive_ppa": False,
            "scheduled_cycle_ratio": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == start_sha,
            "audit changed during execution")
    require(not OUTPUT.exists(), "refusing audit output overwrite")
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M118 audit standalone=93 integrated=58 payload={} combined={} saving={} r1_citable=false"
          .format(payload19_bytes, combined_bytes,
                  payload24_bytes - payload19_bytes), flush=True)


if __name__ == "__main__":
    main()
