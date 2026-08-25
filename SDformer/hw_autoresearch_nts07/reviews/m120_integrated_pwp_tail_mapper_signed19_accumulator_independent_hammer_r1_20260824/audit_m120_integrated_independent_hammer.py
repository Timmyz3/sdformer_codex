#!/usr/bin/env python3
"""Fail-closed audit of M120 production evidence and independent VCS hammer."""

import hashlib
import json
import re
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
PROD = HW / "dc_handoff/runs/m120_integrated_pwp_tail_mapper_signed19_accumulator_vcs_r1_sealed_20260824"
IND = REVIEW / "vcs_run_r1"
OUTPUT = REVIEW / "m120_integrated_independent_hammer_audit.json"

PATHS = {
    "m120_contract": HW / "contracts/m120_integrated_pwp_tail_mapper_signed19_accumulator_vcs_contract_r1_20260824.json",
    "m119_contract": HW / "contracts/m119_pwp_weight_tail_bypass_mapper_vcs_contract_r1_20260824.json",
    "m118_contract": HW / "contracts/m118_w384_signed19_lane_sliced_accumulator_vcs_contract_r1_20260824.json",
    "m120_rtl": HW / "rtl_m120/m120_pwp_tail_mapper_signed19_accumulator_island.sv",
    "m119_rtl": HW / "rtl_m119/m119_pwp_weight_tail_bypass_mapper.sv",
    "m118_core": HW / "rtl_m118/m118_w384_signed19_accumulator_frontend.sv",
    "m118_adapter": HW / "rtl_m118/m118_w384_signed19_lane_sliced_accumulator_adapter.sv",
    "m120_sva": HW / "verif_m120/m120_pwp_tail_mapper_signed19_accumulator_island_assertions.sv",
    "m120_tb": HW / "tb_m120/tb_m120_pwp_tail_mapper_signed19_accumulator_island.sv",
    "m120_filelist": HW / "dc_handoff/filelists/date_m120_integrated_pwp_tail_mapper_signed19_accumulator_directed_vcs.f",
    "m120_runner": HW / "dc_handoff/scripts/run_vcs_m120_integrated_pwp_tail_mapper_signed19_accumulator.sh",
    "prod_complete": PROD / "RUN_COMPLETE.txt",
    "prod_compile": PROD / "compile.raw.log",
    "prod_sim": PROD / "sim.raw.log",
    "prod_assert": PROD / "assert.report",
    "m119_complete": HW / "dc_handoff/runs/m119_pwp_weight_tail_bypass_mapper_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m118_complete": HW / "dc_handoff/runs/m118_w384_signed19_lane_accumulator_vcs_r2_sealed_20260824/RUN_COMPLETE.txt",
    "m115r2_manifest": HW / "reviews/m115r2_pwp_prefix_coefficient_width_independent_hammer_r1_20260824/manifest.sha256",
    "hammer_tb": REVIEW / "tb_m120_independent_hammer.sv",
    "hammer_filelist": REVIEW / "m120_independent.f",
    "hammer_input": REVIEW / "input_manifest.sha256",
    "hammer_runner": REVIEW / "run_vcs_m120_independent_hammer.sh",
    "hammer_complete": IND / "RUN_COMPLETE.txt",
    "hammer_compile": IND / "compile.raw.log",
    "hammer_sim": IND / "sim.raw.log",
    "hammer_assert": IND / "assert.report",
    "hammer_preflight": IND / "preflight_sha_checks.txt",
    "hammer_outputs": IND / "output_sha256.txt",
    "hammer_runner_sha": IND / "runner_sha256.txt",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}

EXPECTED = {
    "m120_contract": "0ce38d33e4885bd3c5b79f81117acec54df6e0e8b753359b172b6031403a947a",
    "m119_contract": "5ccdebb50ae7149bd51a7b767ae3176758c9617dc6e751d04255814e001e3cd8",
    "m118_contract": "c79f55a15e03bbf26c22e9da2f0eb35d53b1a9795ab02b24a6b3c951c729903e",
    "m120_rtl": "f37ed1f9ea1f6c26c80327c620e219bbfb3863f29337c754d50ae85068236316",
    "m119_rtl": "2077c5abe1a5a54e586a59e6e0335db0b76655f7be22bee2b626e8f3671ef337",
    "m118_core": "0903a295f056f69067792e20f40acdae5cb8a38471c4be82077bf5b0c086d482",
    "m118_adapter": "cbccbe2611f7be31c305fb4032c0d518bc7eb51025b6d66cecc157693b6554af",
    "m120_sva": "89d6d0f8a71e60b2f2b5daa5152ca230bc935aa0390ba4ca858612186d94c908",
    "m120_tb": "1b3d3ae2b060573ca516906b20c968c17608791f1aef0edaf5ffe82b05c3a758",
    "m120_filelist": "80ca152b62e1dbfae4de9ce7bc5fca63fbc8473ab51f33ee0890defe5f32e982",
    "m120_runner": "6c84912b2647d61eb149f27501d9c20fe7a7c1280fbc41122bd6c92fd2f150ce",
    "prod_complete": "1cce8b2e7a09bd193baeb703d25e2b25e1d263f80d3cd273f4bedd1a35b032ac",
    "prod_compile": "d496ad7457706fa4f6fa9cd32361808da712de4ea0e630ee90fc9b6e10ef4ad5",
    "prod_sim": "9bc7cf0cf405fb680076d788e6c45e15492074337f8e16937a213d65ebb39331",
    "prod_assert": "0a9f902ec42e4b0e20e48cbc3e442ce102733efcbbabe3ce62074ec287755eb9",
    "m119_complete": "88b36867e9ba4cd67e3d1ff8265351de40a54a42843e4b4cf9c4e7f2a2c9d423",
    "m118_complete": "f45baa3c322a439377aa9c0c3e919440020294c9392b81343c7fae1bc1e605ff",
    "m115r2_manifest": "d0c7067f599c8e24b77099ffec4624c533bbbc098c1d5123bf444ef467237790",
    "hammer_tb": "cc61b8fc5e1581eb2ad53f3aa933f9363bd200effbf9f71297feba27326c8f6b",
    "hammer_filelist": "52eb53065c7aa10e268f078ef6d1c683959bc4a5668c5ac320379b6194c2ceee",
    "hammer_input": "d7f15532b3c987abb1830bec933e22095878343678b563b3b52b41ba2ac67e84",
    "hammer_runner": "b48ee109eaaa8112a777183eec87f8088ef58ee08162ee68093166c0c5489fbe",
    "hammer_complete": "27ed3cd47836cd88779d5cf0832eeccd28b5e167fcc6a2596a9fe40c21d37790",
    "hammer_compile": "06b29c353e735c4063bbb8c75c7d5d5ae7f35c50245536e6b9920ff78bd6c4f1",
    "hammer_sim": "a3a8d0e5b6a05310478196704d3842309127f6586b68ad19c2e1e4382f88019c",
    "hammer_assert": "9d592a17e671251ead95268a314e8ab732c5ce18cad5bd2ece2fa4cbaee94d38",
    "hammer_preflight": "8cb36919ddd1b4e762d0662e24156853512da572bb20e64b391c3645e08133a1",
    "hammer_outputs": "cad800b6a34813305e7e45c8e633c340dfbcf9a72e73f674e0dbc6dee9b64140",
    "hammer_runner_sha": "fb149170e570985fb054960868b53a345f6b01bd5fe18214b4cf245a02c97b54",
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


def main():
    self_start = sha256(Path(__file__).resolve())
    observed = {}
    for label, path in PATHS.items():
        actual = sha256(path)
        require(actual == EXPECTED[label],
                "identity mismatch {} {}".format(label, actual))
        observed[str(path.relative_to(HW))] = actual

    contract = strict_json(PATHS["m120_contract"])
    c119 = strict_json(PATHS["m119_contract"])
    c118 = strict_json(PATHS["m118_contract"])
    require(contract["frozen_sources"]["wrapper_rtl_sha256"]
            == EXPECTED["m120_rtl"], "M120 wrapper identity drift")
    require(contract["frozen_sources"]["wrapper_sva_sha256"]
            == EXPECTED["m120_sva"], "M120 SVA identity drift")
    require(contract["frozen_sources"]["testbench_sha256"]
            == EXPECTED["m120_tb"], "M120 TB identity drift")
    require(contract["frozen_sources"]["filelist_sha256"]
            == EXPECTED["m120_filelist"], "M120 filelist identity drift")
    require(contract["motivation"]["m115r2_independent_manifest_sha256"]
            == EXPECTED["m115r2_manifest"], "M115-r2 input drift")
    require(c118["arithmetic_basis"]["maximum_absolute_prefix_coefficient"] == 1
            and c118["arithmetic_basis"]["checkpoint_sumabs_bound"] == 218338
            and c118["arithmetic_basis"]["checkpoint_mathematical_candidate_signed_bits"] == 19,
            "M118 signed19 basis drift")
    require(c119["architecture"]["tail_bypass"].startswith("beat2 response"),
            "M119 tail-bypass contract drift")

    accumulator_bits = 384 * 8 * 96 * 19
    accumulator_bytes = accumulator_bits // 8
    vector_bits = 96 * 19
    descriptor_bits = 2 * 128 * 384 * 2
    metadata_bits = 314
    valid_bits = 8 * 384
    combined_bits = accumulator_bits + descriptor_bits + metadata_bits + valid_bits
    combined_bytes = (combined_bits + 7) // 8
    require(vector_bits == 1824 and accumulator_bits == 5603328
            and accumulator_bytes == 700416 and combined_bits == 5803322
            and combined_bytes == 725416, "independent geometry mismatch")
    require(contract["architecture"]["mapped_delta_bits_internal_only"] == vector_bits
            and contract["architecture"]["accumulator_payload_bytes"] == accumulator_bytes
            and contract["architecture"]["combined_descriptor_valid_accumulator_logical_lower_bound_bytes"] == combined_bytes,
            "M120 geometry contract mismatch")

    m119_rtl = PATHS["m119_rtl"].read_text(encoding="utf-8")
    m118_rtl = PATHS["m118_core"].read_text(encoding="utf-8")
    m120_rtl = PATHS["m120_rtl"].read_text(encoding="utf-8")
    require("output_slot_available = !update_valid_q || update_ready" in m119_rtl
            and "service_ready = !protocol_error && token_shape_valid" in m119_rtl,
            "M119 elastic accept logic drift")
    require("same_address_rdw_hazard = update_pipe_valid_q && update_valid" in m118_rtl
            and "&& !same_address_rdw_hazard" in m118_rtl,
            "M118 same-address hazard guard drift")
    require(".update_ready(mapper_update_ready)" in m120_rtl
            and ".update_valid(mapper_update_valid)" in m120_rtl,
            "M120 direct mapper/accumulator wiring drift")

    prod = receipt(PATHS["prod_complete"])
    require(prod["status"] == "PASS_M120_INTEGRATED_PWP_TAIL_MAPPER_SIGNED19_ACCUMULATOR_DIRECTED_VCS_SVA",
            "production receipt status drift")
    require([int(prod[key]) for key in ("accepted_events",
                                         "mapped_accumulator_updates",
                                         "accumulator_writes")]
            == [1024, 1024, 1024], "production token ledger mismatch")
    for key in ("m117_scheduler_integrated",
                "heldout_trace_duplicate_retry_escape_replay",
                "foundry_weight_sram_macro", "foundry_accumulator_sram_macro",
                "scheduled_cycle_ratio", "physical_speedup",
                "system_speedup", "headline"):
        require(prod[key] == "false", "production boundary drift " + key)
    prod_log = PATHS["prod_sim"].read_text(encoding="utf-8")
    require("Compiler version V-2023.12-SP1_Full64" in prod_log
            and "Runtime version V-2023.12-SP1_Full64" in prod_log,
            "production VCS identity absent")
    prod_assert = PATHS["prod_assert"].read_text(encoding="utf-8")
    for cover, count in (("cp_three_loads_tail_event", 256),
                         ("cp_event_update_chain", 1024),
                         ("cp_update_ii1", 768),
                         ("cp_lane_read_write_overlap", 768),
                         ("cp_commit_stall_release", 699),
                         ("cp_full_window", 2), ("cp_fault", 1),
                         ("cp_busy", 8680)):
        require(re.search(cover + r", .* {} match".format(count), prod_assert),
                "production cover mismatch " + cover)

    hammer = receipt(PATHS["hammer_complete"])
    require(hammer["status"]
            == "PASS_M120_INDEPENDENT_HAMMER_P0_ACCEPTED_THEN_LOST_CONFIRMED",
            "independent receipt status drift")
    require([int(hammer[key]) for key in ("positive_events",
                                           "positive_mapped_updates",
                                           "positive_accumulator_writes")]
            == [1024, 1024, 1024], "independent positive ledger mismatch")
    require(hammer["same_address_events_accepted"] == "2"
            and hammer["same_address_mapped_updates"] == "1"
            and hammer["same_address_accumulator_writes"] == "1"
            and hammer["same_address_accept_then_loss_p0"] == "true",
            "same-address accepted/lost P0 not demonstrated")
    require(hammer["retry_events_accepted"] == "3"
            and hammer["retry_updates_written"] == "3"
            and hammer["retry_dedup_absent"] == "true",
            "retry/dedup finding drift")
    require(hammer["reset_events_accepted"] == "1"
            and hammer["reset_updates_written"] == "0"
            and hammer["reset_exact_once_undefined"] == "true",
            "reset finding drift")
    require(hammer["full_integrated_exact_once_p0_closed"] == "false",
            "independent P0 boundary drift")
    hammer_log = PATHS["hammer_sim"].read_text(encoding="utf-8")
    require("Compiler version V-2023.12-SP1_Full64" in hammer_log
            and "Runtime version V-2023.12-SP1_Full64" in hammer_log,
            "independent commercial VCS identity absent")
    pass_match = re.search(
        r"PASS M120 independent hammer .*"
        r"positive_loads=(\d+) positive_weight_reads=(\d+) "
        r"positive_events=(\d+) positive_updates=(\d+) positive_writes=(\d+) "
        r"positive_ii1_pairs=(\d+) positive_rw_overlap=(\d+) "
        r"mapper_lane_checks=(\d+).* commits=(\d+) commit_lane_checks=(\d+) "
        r"commit_stalls=(\d+) stall_releases=(\d+).*"
        r"same_address_events_accepted=2 same_address_updates_written=1.*"
        r"reset_events_accepted=1 reset_updates_written=0", hammer_log)
    require(pass_match is not None, "independent PASS ledger absent")
    metrics = [int(value) for value in pass_match.groups()]
    require(metrics[:10] == [768, 768, 1024, 1024, 1024, 768, 768,
                             98304, 6144, 589824],
            "independent numeric ledger mismatch")
    require(metrics[10] >= 100 and metrics[11] > 0,
            "independent commit backpressure weak")
    hammer_assert = PATHS["hammer_assert"].read_text(encoding="utf-8")
    for cover, count in (("cp_three_loads_tail_event", 256),
                         ("cp_event_update_chain", 1024),
                         ("cp_update_ii1", 768),
                         ("cp_lane_read_write_overlap", 768),
                         ("cp_full_window", 2)):
        require(re.search(cover + r", .* {} match".format(count), hammer_assert),
                "independent positive SVA cover mismatch " + cover)

    output = {
        "schema": "m120_integrated_independent_hammer_audit_v1",
        "status": "FAIL_FULL_EXACT_ONCE_P0_FOUND_DIRECTED_DISTINCT_ADDRESS_SUBSCOPE_PASS",
        "identity": observed,
        "production_evidence": {
            "exact_sha": True,
            "commercial_vcs": "V-2023.12-SP1_Full64",
            "directed_counts": {
                "accepted_events": 1024,
                "mapped_updates": 1024,
                "accumulator_writes": 1024,
                "commit_vectors": 6144,
                "commit_lane_checks": 589824,
            },
            "directed_legal_distinct_consecutive_address_scope_pass": True,
            "heldout_duplicate_retry_reset_scope": False,
        },
        "independent_hammer": {
            "commercial_vcs": "V-2023.12-SP1_Full64",
            "positive_counts": {
                "loads": 768,
                "weight_reads": 768,
                "accepted_events": 1024,
                "mapped_updates": 1024,
                "accumulator_writes": 1024,
                "ii1_pairs": 768,
                "read_write_overlap": 768,
                "mapper_lane_checks": 98304,
                "commit_vectors": 6144,
                "commit_lane_checks": 589824,
                "tail_bypass_hits": 256,
                "negated_events": 512,
            },
            "int8_endpoints_and_negation_checked": True,
            "load_never_writes_checked": True,
            "malformed_load_beat_and_key_fail_closed": True,
            "premature_end_fault_with_older_update_drain_checked": True,
            "two_window_lazy_clear_and_full_commit_order_checked": True,
            "same_address_attack": {
                "legal_shaped_events_accepted": 2,
                "mapped_updates": 1,
                "accumulator_writes": 1,
                "protocol_error": True,
                "finding": "P0_ACCEPTED_EVENT_IS_LOST_AFTER_ACCEPT",
            },
            "separated_retry_attack": {
                "events_accepted": 3,
                "updates_written": 3,
                "deduplication_present": False,
            },
            "reset_attack": {
                "event_accepted_before_reset": 1,
                "updates_written": 0,
                "recovery_or_replay_contract_present": False,
            },
        },
        "geometry_recomputed": {
            "vector_bits_96x19": vector_bits,
            "accumulator_bits_384x8x96x19": accumulator_bits,
            "accumulator_bytes": accumulator_bytes,
            "descriptor_bits": descriptor_bits,
            "metadata_bits": metadata_bits,
            "valid_bits": valid_bits,
            "combined_bits": combined_bits,
            "combined_bytes_ceiling": combined_bytes,
        },
        "m118_p0_closure": {
            "closed_scope": "reset-free directed traffic with no consecutive same-address accepted events",
            "closed": True,
            "full_integrated_exact_once_closed": False,
            "remaining_p0": [
                "Backpressure service acceptance using the accumulator's current ready, or add a same-address forwarding/interlock, so a second same-address event cannot be accepted and then lost.",
                "Define and verify transaction identity/retry/reset recovery with M117 and heldout trace replay before claiming full accepted-transaction exact-once.",
            ],
        },
        "claim_boundary": {
            "foundry_weight_sram_macro": False,
            "foundry_accumulator_sram_macro": False,
            "macro_inclusive_ppa": False,
            "scheduled_cycle_ratio": False,
            "physical_speedup": False,
            "system_speedup": False,
            "m109_projected_ratio": 2.53546204172554,
            "m109_projected_ratio_headline_admitted": False,
            "headline": False,
        },
        "self_sha256_at_start": self_start,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    require(sha256(Path(__file__).resolve()) == self_start,
            "audit script changed during execution")
    print("PASS M120 integrated independent audit")


if __name__ == "__main__":
    main()
