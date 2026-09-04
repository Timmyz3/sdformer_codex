#!/usr/bin/env python3
"""Fail-closed machine audit for the M128 independent hammer review."""

import hashlib
import json
import re
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
OUTPUT = REVIEW / "m128_independent_machine_audit.json"

EXPECTED = {
    "contract": ("contracts/m128_descriptor_streamed_k4_row_fold_vcs_contract_r1_20260824.json",
                 "7b08459cbba96f14666c57b5db274b850b58546c25d7d42e52210bf9e4228bf1"),
    "m127_correction": ("contracts/m127_r1_throughput_scope_correction_r1_20260824.json",
                        "a64a00f443d691b1295a4eb14a92edbc9d41ce448d83fd3a8c3ca4f59d2b365d"),
    "rtl": ("rtl_m128/m128_descriptor_streamed_k4_row_fold.sv",
            "b7c5c4c329bc4f1a7011398c5d3c20933dd8badfc4b2bbf3b213b15efe01e54d"),
    "sva": ("verif_m128/m128_descriptor_streamed_k4_row_fold_assertions.sv",
            "334c366289690bff624e8a3976dd602ed45f6046b7b1ed6314143922e5a06a50"),
    "production_tb": ("tb_m128/tb_m128_descriptor_streamed_k4_row_fold.sv",
                      "30cc18e83a00173a9f0e17ea5116f5429a340fbea88f3decb4d28073e8cbee94"),
    "production_filelist": ("dc_handoff/filelists/date_m128_descriptor_streamed_k4_row_fold_directed_vcs.f",
                            "685e547c610acbbf8f9298bb32f9ced1035aff158192d9f882e2c519f5f9cf7c"),
    "production_runner": ("dc_handoff/scripts/run_vcs_m128_descriptor_streamed_k4_row_fold.sh",
                          "d4fa2311c4d7674fc808a3ad1dc09c9f266000660bb913cb17908c4d2098931c"),
    "sealed_run_complete": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
                            "d9e320092d381999ec158fa31d8aaf32be47c02283d50e3e7ba463cfd7751f28"),
    "sealed_input_sha_list": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/input_sha256.txt",
                              "beac6d8e22d48a32a2909eb7348e605e3fcfe0ba580698811a0536ced7582d85"),
    "sealed_output_sha_list": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/output_sha256.txt",
                               "90c6569c8063aeccbc1e831853f3afd8cc296b2155f864198e5b1f94201501be"),
    "sealed_preflight": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/preflight_sha_checks.txt",
                         "21d80fbe266a63aa1dc67e2199a299adb29b0997d00c58bc8550370ee2f4bb68"),
    "sealed_runner_sha": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/runner_sha256.txt",
                          "26603b4e474722eb6ae9e18680a29195499d14b98c6b58677417bac56b3f292f"),
    "sealed_compile_rc": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/compile.rc",
                          "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"),
    "sealed_sim_rc": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/sim.rc",
                      "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"),
    "sealed_compile_log": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/compile.raw.log",
                           "2f908d12bd4def784c97895ea0d2d24651876249d4b32f0127278e0325f4c1fc"),
    "sealed_sim_log": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/sim.raw.log",
                       "ceed0d6fb747ca28973e99642aa19a5599fcf0308d45b6e7a7d91c00498d15d0"),
    "sealed_assert_report": ("dc_handoff/runs/m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/assert.report",
                             "86dc81ace8966288ac31758c3d6e62f21d92b501b70c9e19fe5c89be86c6b723"),
    "m127_review": ("reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824/manifest.sha256",
                    "8bea333f44528044f251a48ebf9d20e261e4919bc63ed9f262b01004d25c7947"),
    "docs359": ("docs/359_DATE终局冻结_20260813.md",
                "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
    "review_tb": ("reviews/m128_descriptor_streamed_k4_row_fold_independent_hammer_r1_20260824/tb_m128_independent_hammer.sv",
                  "e66f3d447b8b52ec3587ea2b24d7b0612d3e7417f011d2764a26d5de2a6e807d"),
    "review_filelist": ("reviews/m128_descriptor_streamed_k4_row_fold_independent_hammer_r1_20260824/m128_independent.f",
                        "d42b58c414fae0fd179078bc394724fdf09cf9071ab5947c5f8ae5389b3165b5"),
    "review_runner": ("reviews/m128_descriptor_streamed_k4_row_fold_independent_hammer_r1_20260824/run_vcs_m128_independent_hammer.sh",
                      "4576ccb160ec8e4d334da0400b646f8a621f56a00132c01f54feb28d4949136d"),
}


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL M128 independent audit: " + message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def strict_json(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result

    return json.loads(read(path), object_pairs_hook=pairs_hook,
                      parse_constant=lambda value: require(False, value))


def tokens_from_pass(log, prefix):
    match = re.search(r"^" + re.escape(prefix) + r" (.+)$", log, re.M)
    require(match is not None, "missing pass line " + prefix)
    return dict(item.split("=", 1) for item in match.group(1).split()
                if "=" in item)


def verify_sha_list(path):
    count = 0
    for line in read(path).splitlines():
        digest, label = line.split(None, 1)
        target = Path(label.strip())
        require(target.is_absolute(), "expected absolute VCS output label")
        require(target.is_file(), "missing VCS output " + str(target))
        require(sha256(target) == digest, "VCS output digest " + str(target))
        count += 1
    return count


observed = {}
for name, (label, expected) in EXPECTED.items():
    actual = sha256(HW / label)
    require(actual == expected, name + " SHA drift")
    observed[name] = actual

contract = strict_json(HW / EXPECTED["contract"][0])
review = strict_json(REVIEW / "m128_descriptor_streamed_k4_row_fold_independent_hammer_review.json")
require(review["score"] == {
    "overall": 88,
    "exact_sha_commercial_vcs": 20,
    "descriptor_local_numeric_and_protocol_function": 28,
    "elastic_stall_reset_and_cross_row_ii1": 18,
    "interface_and_row_semantics": 10,
    "physical_system_and_claim_discipline": 12,
}, "score drift")
require(review["severity_counts"] == {"P0": 0, "P1": 2, "P2": 4},
        "severity drift")
for flag in ("descriptor_producer_implemented", "descriptor_bandwidth_accounted",
             "foundry_weight_macro", "dc_frequency_improvement",
             "macro_inclusive_ppa", "physical_speedup", "system_speedup",
             "headline"):
    require(contract["admission"][flag] is False,
            "unsafe production contract admission " + flag)
require(contract["architecture"]["descriptor_predecode_external"] is True,
        "external predecode boundary drift")
require(contract["architecture"]["descriptor_predecode_cost_modeled"] is False,
        "external predecode cost unexpectedly admitted")

for directory in (REVIEW / "sealed_vcs_replay", REVIEW / "independent_vcs"):
    require(read(directory / "compile.rc").strip() == "0",
            str(directory) + " compile rc")
    require(read(directory / "sim.rc").strip() == "0",
            str(directory) + " sim rc")
    combined = read(directory / "sim.raw.log") + "\n" + read(directory / "assert.report")
    require(re.search(r"failed at|Offending|^Error|^Fatal|watchdog timeout",
                      combined, re.I | re.M) is None,
            str(directory) + " failure marker")
    require("Version V-2023.12-SP1_Full64" in read(directory / "compile.raw.log"),
            str(directory) + " VCS version")

sealed_log = read(REVIEW / "sealed_vcs_replay/sim.raw.log")
require(contract["expected_pass_line"] in sealed_log,
        "exact production pass line")
sealed_covers = {
    "cp_k4_descriptor": 144,
    "cp_tail_descriptor": 240,
    "cp_reset_quiesce": 6,
    "cp_cross_row_replace": 60,
    "cp_update_stall_release": 78,
}
sealed_assert = read(REVIEW / "sealed_vcs_replay/assert.report")
for name, count in sealed_covers.items():
    require(re.search(re.escape(name) + r", .* " + str(count) + r" match",
                      sealed_assert) is not None, "sealed cover " + name)

independent_log = read(REVIEW / "independent_vcs/sim.raw.log")
tokens = tokens_from_pass(independent_log, "PASS M128 independent hammer")
expected_tokens = {
    "descriptor_bits": "53", "groups": "140", "updates": "139",
    "reset_aborted_descriptors": "1", "sources": "543", "lanes": "13344",
    "k1": "3", "k2": "3", "k3": "2", "k4": "132",
    "cross_group_ii1_intervals": "127",
    "cross_update_ii1_intervals": "127",
    "output_stall_cycles": "97", "max_output_stall": "97",
    "group_stall_cycles": "97", "long_stall_replace": "1",
    "plus512": "9", "minus512": "9", "row_done_checks": "138",
    "row_done_overlap_next_row": "134", "semantic_ready_probes": "2",
    "duplicate_attacks": "1", "dirty_source_attacks": "1",
    "dirty_negate_attacks": "1", "mask_attacks": "1",
    "cache_miss_attacks": "1", "block_attacks": "1", "empty_attacks": "1",
    "fill_collision_attacks": "1", "reset_checks": "1",
    "noncanonical_order_accepts": "1", "holey_valid_accepts": "1",
    "cross_descriptor_duplicate_accepts": "2",
    "internal_combinational_loop_observed": "false",
    "canonical_order_enforced": "false", "valid_left_packing_enforced": "false",
    "cross_descriptor_source_conservation_enforced": "false",
    "descriptor_predecode_external": "true",
    "descriptor_predecode_cost_modeled": "false",
    "descriptor_bandwidth_accounted": "false",
    "dc_frequency_improvement": "false", "physical_speedup": "false",
    "system_speedup": "false", "headline": "false",
}
for key, value in expected_tokens.items():
    require(tokens.get(key) == value, "independent token " + key)

independent_covers = {
    "cp_k4_descriptor": 132,
    "cp_tail_descriptor": 8,
    "cp_reset_quiesce": 30,
    "cp_cross_row_replace": 125,
    "cp_update_stall_release": 1,
}
independent_assert = read(REVIEW / "independent_vcs/assert.report")
for name, count in independent_covers.items():
    require(re.search(re.escape(name) + r", .* " + str(count) + r" match",
                      independent_assert) is not None,
            "independent cover " + name)

require(verify_sha_list(REVIEW / "vcs_output.sha256") == 6,
        "VCS output manifest count")

rtl = read(HW / EXPECTED["rtl"][0])
descriptor_bits = 3 + 9 + 4 + 4 * 4 + 4 + 16 + 1
require(descriptor_bits == 53, "descriptor arithmetic")
require(contract["architecture"]["descriptor_bits"] == descriptor_bits,
        "contract descriptor width")
require("assign raw_pipe_can_advance = !pipe_valid_q || update_ready;" in rtl,
        "elastic advance dependency drift")
require("assign group_ready = !rst_core && !quarantine" in rtl,
        "group_ready dependency drift")
require("&& group_semantically_valid && !weight_fill_valid" in rtl,
        "semantic-ready dependency drift")
require("assign group_accept = group_valid && group_ready;" in rtl,
        "group_accept dependency drift")
request_audit = rtl.split("always_comb begin : request_audit", 1)[1].split(
    "assign protocol_error", 1)[0]
require("group_ready" not in request_audit and "group_accept" not in request_audit,
        "ready/accept feedback into request audit")
descriptor_audit = rtl.split("always_comb begin : descriptor_audit", 1)[1].split(
    "always_comb begin : request_audit", 1)[0]
require("group_ready" not in descriptor_audit and "group_accept" not in descriptor_audit,
        "ready/accept feedback into descriptor audit")
require("group_source_valid != 0" in request_audit,
        "nonempty descriptor audit drift")
require("descriptor_duplicate" in request_audit and
        "descriptor_padding_dirty" in request_audit and
        "descriptor_derived_mask == group_selected_mask" in request_audit,
        "descriptor-local audit drift")
require("group_source[0] < group_source[1]" not in rtl,
        "unexpected local canonical order enforcement")
require("group_source_valid == 4'b0001" not in request_audit,
        "unexpected valid packing enforcement")
rtl_without_line_comments = re.sub(r"//.*", "", rtl)
require("row_source_mask_q" not in rtl_without_line_comments and
        "source_conservation" not in rtl_without_line_comments,
        "unexpected cross-descriptor conservation state")
require("if (!quarantine && update_accept && pipe_last_q)" in rtl and
        "row_done <= 1'b1;" in rtl,
        "row_done timing implementation drift")

machine = {
    "schema": "m128_independent_machine_audit_v1",
    "status": "PASS_DESCRIPTOR_LOCAL_AND_CROSS_ROW_II1_WITH_EXTERNAL_PREDECODE_AND_ROW_DONE_FINDINGS",
    "score": {"overall": 88, "P0": 0, "P1": 2, "P2": 4},
    "frozen_sha256": observed,
    "production_exact_sha_vcs": {
        "compile_rc": 0, "sim_rc": 0,
        "pass_line_exact": True, "covers": sealed_covers,
    },
    "independent_vcs": {
        "compile_rc": 0, "sim_rc": 0,
        "pass_tokens": tokens, "covers": independent_covers,
        "descriptor_local_numeric_equivalence": True,
        "negative_descriptor_attacks_fail_closed": True,
        "cross_row_descriptor_and_update_ii1": True,
        "long_stall_and_same_cycle_replace": True,
        "reset_isolation_and_flush": True,
    },
    "descriptor_width": {
        "payload_bits": descriptor_bits,
        "valid_ready_excluded": 2,
        "producer_and_transport_cost_excluded": True,
    },
    "static_dependency_audit": {
        "internal_ready_error_combinational_loop": False,
        "group_ready_payload_semantic_when_valid_zero": True,
        "external_valid_or_payload_must_not_depend_on_ready": True,
    },
    "external_predecode_boundary": {
        "producer_implemented": False,
        "cost_modeled": False,
        "bandwidth_accounted": False,
        "canonical_order_locally_enforced": False,
        "valid_left_packing_locally_enforced": False,
        "cross_descriptor_source_conservation_locally_enforced": False,
    },
    "row_done_boundary": {
        "overlap_next_row_update_observed": True,
        "overlap_count": 134,
        "done_has_block_row_tag": False,
        "safe_completion_identity": "accepted update_last handshake",
    },
    "claim_boundary": {
        "dc_frequency_improvement": False,
        "foundry_macro": False,
        "macro_ppa": False,
        "physical_speedup": False,
        "system_speedup": False,
        "headline": False,
    },
}
OUTPUT.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
print("PASS M128 independent machine audit score=88 p0=0 p1=2 p2=4 "
      "cross_row_ii1=true external_predecode_unmodeled=true "
      "row_done_ownership_open=true physical_speedup=false system_speedup=false")
