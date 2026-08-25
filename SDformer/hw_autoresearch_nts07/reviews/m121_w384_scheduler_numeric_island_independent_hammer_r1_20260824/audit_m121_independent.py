#!/usr/bin/env python3
"""Independent exact-SHA, receipt, RTL-boundary and counterexample audit for M121."""

import argparse
import hashlib
import json
import re
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
PROD_RUN = HW / "dc_handoff/runs/m121_w384_scheduler_numeric_island_vcs_r1_sealed_20260824"
REVIEW = Path(__file__).resolve().parent
IND_RUN = REVIEW / "vcs_counterexamples_r1"
PATHS = {
    "contract": HW / "contracts/m121_w384_scheduler_numeric_island_vcs_contract_r1_20260824.json",
    "wrapper": HW / "rtl_m121/m121_w384_scheduler_numeric_island.sv",
    "sva": HW / "verif_m121/m121_w384_scheduler_numeric_island_assertions.sv",
    "tb": HW / "tb_m121/tb_m121_w384_scheduler_numeric_island.sv",
    "filelist": HW / "dc_handoff/filelists/date_m121_w384_scheduler_numeric_island_directed_vcs.f",
    "runner": HW / "dc_handoff/scripts/run_vcs_m121_w384_scheduler_numeric_island.sh",
    "scheduler": HW / "rtl_m117/m117_w384_prefetch_transpose_scheduler.sv",
    "mapper": HW / "rtl_m119/m119_pwp_weight_tail_bypass_mapper.sv",
    "accumulator": HW / "rtl_m118/m118_w384_signed19_accumulator_frontend.sv",
    "adapter": HW / "rtl_m118/m118_w384_signed19_lane_sliced_accumulator_adapter.sv",
    "numeric": HW / "rtl_m120/m120_pwp_tail_mapper_signed19_accumulator_island.sv",
    "m117_contract": HW / "contracts/m117_w384_prefetch_transpose_vcs_contract_r1_20260824.json",
    "m117_receipt": HW / "dc_handoff/runs/m117_w384_prefetch_transpose_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m120_contract": HW / "contracts/m120_integrated_pwp_tail_mapper_signed19_accumulator_vcs_contract_r1_20260824.json",
    "m120_receipt": HW / "dc_handoff/runs/m120_integrated_pwp_tail_mapper_signed19_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m117_review_manifest": HW / "reviews/m117_w384_prefetch_transpose_independent_hammer_r1_20260824/manifest.sha256",
    "m119_review_manifest": HW / "reviews/m119_pwp_weight_tail_bypass_mapper_independent_hammer_r1_20260824/manifest.sha256",
    "m119_review": HW / "reviews/m119_pwp_weight_tail_bypass_mapper_independent_hammer_r1_20260824/m119_pwp_weight_tail_bypass_mapper_independent_hammer_review.json",
    "prod_receipt": PROD_RUN / "RUN_COMPLETE.txt",
    "prod_input_manifest": PROD_RUN / "input_sha256.txt",
    "prod_output_manifest": PROD_RUN / "output_sha256.txt",
    "prod_preflight": PROD_RUN / "preflight_sha_checks.txt",
    "prod_runner_manifest": PROD_RUN / "runner_sha256.txt",
    "prod_compile_log": PROD_RUN / "compile.raw.log",
    "prod_sim_log": PROD_RUN / "sim.raw.log",
    "prod_assert_report": PROD_RUN / "assert.report",
    "prod_compile_rc": PROD_RUN / "compile.rc",
    "prod_sim_rc": PROD_RUN / "sim.rc",
    "ind_filelist": REVIEW / "m121_independent.f",
    "ind_tb": REVIEW / "tb_m121_independent_counterexamples.sv",
    "ind_runner": REVIEW / "run_vcs_m121_independent_counterexamples.sh",
    "ind_receipt": IND_RUN / "RUN_COMPLETE.txt",
    "ind_input_manifest": IND_RUN / "input_sha256.txt",
    "ind_output_manifest": IND_RUN / "output_sha256.txt",
    "ind_compile_log": IND_RUN / "compile.raw.log",
    "ind_sim_log": IND_RUN / "sim.raw.log",
    "ind_compile_rc": IND_RUN / "compile.rc",
    "ind_sim_rc": IND_RUN / "sim.rc",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED_SHA = {
    "contract": "a4a2d2aac9838c30cf28c841add479472e3287db087763dc6b1535cc5bcd10ad",
    "wrapper": "a448e4cc530a1885f92e413e74f2e9b06df7a5fc5338cc5771f1130bf746be85",
    "sva": "84801c33ed2c59d4cc1404cfd9339e4903d01c407bd1e8c7ff2d301470db41a8",
    "tb": "23304b03a148daa8c368bebea9b0baff525d206e768357c10fea10be005a39a5",
    "filelist": "0a65fb3ddda0bf430a61ca0d9025688f1ce93404fff66b47c8bca9ff09687d65",
    "runner": "e81b52b30d2d16c00b2b6d19b07d62ec98399ebee7eadd40141861c86bce77d8",
    "scheduler": "4e640770349fa2d95ac09731efe7f8587d8bb108bd89169c204200cf41f3983a",
    "mapper": "2077c5abe1a5a54e586a59e6e0335db0b76655f7be22bee2b626e8f3671ef337",
    "accumulator": "0903a295f056f69067792e20f40acdae5cb8a38471c4be82077bf5b0c086d482",
    "adapter": "cbccbe2611f7be31c305fb4032c0d518bc7eb51025b6d66cecc157693b6554af",
    "numeric": "f37ed1f9ea1f6c26c80327c620e219bbfb3863f29337c754d50ae85068236316",
    "m117_contract": "b327f0e14d83ecf1df18fcbedb2d5986a1b53971b54a972892f6552b44ca1fef",
    "m117_receipt": "92f991f06f8a4d80ef2fc0d2fdd96cb473a7b6a2e29e687627ac3f531814c927",
    "m120_contract": "0ce38d33e4885bd3c5b79f81117acec54df6e0e8b753359b172b6031403a947a",
    "m120_receipt": "1cce8b2e7a09bd193baeb703d25e2b25e1d263f80d3cd273f4bedd1a35b032ac",
    "m117_review_manifest": "3fc112ae72769f4dbba8aed8450fe2b840327292a112825956be8160e93137b2",
    "m119_review_manifest": "b73e8a8a6d23a12edc62300ca6ad04d5ccd128e89e6e085a04832621d1e43abf",
    "m119_review": "a544da61ecc4188f1f2c0b90815e513caef8a27dd181998e148726124e47bb91",
    "prod_receipt": "4b3e0d1bf249bff14dc18a6de05cc7ddf5bca4e2d384a7ef160650702fbee986",
    "prod_input_manifest": "823a4c78ddd5773addf32ad82cfb5d23c0fcff13ca0725b32ef831876d131a17",
    "prod_output_manifest": "fbd996bbcaa9a9f54a68aab9d66f27bc0ac6d7fb50cc27f20930a5f0a4aaab6c",
    "prod_preflight": "681e4909cd4c47b74e19771bd28c8202a5c28b99c7b9784386a93fa3c7d74dd6",
    "prod_runner_manifest": "b364ee92999dd2c6dca203db12829c00cc41192372ddc7d43aaa07f5ffaa9bd4",
    "prod_compile_log": "01097c44ac68ac2d3122feb69884e63dcf66512b042d324b1d914015c8fce7cd",
    "prod_sim_log": "68169201c149325f35580e35d4d980829e40f5c3a0ba38b2f3ae4abd97750307",
    "prod_assert_report": "db7ec013bc08846b80beb7b8a760888f21749debab1f1c655f1851cc4740af4a",
    "prod_compile_rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "prod_sim_rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ind_filelist": "aacf59f1c0e6b265555138c46cc83cc41402c1ea29a76833bda7dda7e5a7112c",
    "ind_tb": "2c167ede7889ba51febff0e0dd25d44470e45b921b1faa7224101543d810f7da",
    "ind_runner": "ff202d459d480a9d3dd14c1f7914c301e6469e1faff13d48081edca9acc8e5bf",
    "ind_receipt": "8927f59d4886f9cdfbe322fc3b611f461fb4b998781619d2fd00b1808938abc7",
    "ind_input_manifest": "96295328bdcbee6fb45d5038933edf7ca95ab1c3e7dd7be4876c9fffc8f698d7",
    "ind_output_manifest": "cb41d37a7665c185e7ef1fd64d67aa7a555a2c587618ab4efdb553f587d31c6b",
    "ind_compile_log": "ac89fd15a2713dfcfdedc3408322fa90dbebebf1d104d292d13a27fc75371568",
    "ind_sim_log": "284be8cfd505b4e23d7e429bc62d64871f3017b9d4efc276838de5b0cbecc3b4",
    "ind_compile_rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ind_sim_rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def strict_loads(text):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=hook, parse_constant=reject)


def strict_json(path):
    return strict_loads(Path(path).read_text(encoding="utf-8"))


def attack_raises(function):
    try:
        function()
    except (ValueError, json.JSONDecodeError):
        return True
    return False


def parse_manifest(path, base, allow_absolute=False, absolute_root=None):
    entries = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "malformed manifest line {}".format(number))
        expected, raw = match.groups()
        require(raw not in seen, "duplicate manifest path")
        seen.add(raw)
        candidate = Path(raw)
        if candidate.is_absolute():
            require(allow_absolute, "absolute path not allowed")
            resolved = candidate.resolve()
            if absolute_root is not None:
                require(str(resolved).startswith(str(Path(absolute_root).resolve()) + "/"),
                        "absolute path escaped root")
        else:
            require(".." not in candidate.parts, "manifest traversal")
            resolved = Path(base) / candidate
        entries.append((expected, raw, resolved))
    require(entries, "empty manifest")
    return entries


def verify_manifest(path, base, allow_absolute=False, absolute_root=None):
    entries = parse_manifest(path, base, allow_absolute, absolute_root)
    failed = [raw for expected, raw, resolved in entries
              if not resolved.is_file() or sha256(resolved) != expected]
    return {"entries": len(entries), "failed": failed,
            "sha256": sha256(path), "paths": [raw for _, raw, _ in entries]}


def strict_receipt(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        require(re.fullmatch(r"[^=\s]+=[^\s]+", line) is not None,
                "malformed receipt line")
        key, value = line.split("=", 1)
        require(key not in result, "duplicate receipt key")
        result[key] = value
    require(result, "empty receipt")
    return result


def parse_pass_line(text):
    line = next(row for row in text.splitlines()
                if row.startswith("PASS M121 W384 scheduler numeric island VCS "))
    fields = {}
    for token in line.split()[7:]:
        if "=" in token:
            key, value = token.split("=", 1)
            require(key not in fields, "duplicate pass-line field")
            fields[key] = value
    return line, fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing audit overwrite")
    actual_sha = {label: sha256(path) for label, path in PATHS.items()}
    require(actual_sha == EXPECTED_SHA, "exact SHA drift")

    contract = strict_json(PATHS["contract"])
    m119_review = strict_json(PATHS["m119_review"])
    receipt = strict_receipt(PATHS["prod_receipt"])
    independent_receipt = strict_receipt(PATHS["ind_receipt"])
    strict_attacks = {
        "duplicate_json_key_rejected": attack_raises(
            lambda: strict_loads('{"x":1,"x":2}')),
        "nan_rejected": attack_raises(lambda: strict_loads('{"x":NaN}')),
        "infinity_rejected": attack_raises(lambda: strict_loads('{"x":Infinity}')),
        "duplicate_receipt_key_rejected": attack_raises(
            lambda: _receipt_text("x=1\nx=2\n")),
        "duplicate_manifest_path_rejected": attack_raises(
            lambda: _manifest_text(("0" * 64 + "  x\n") * 2)),
        "malformed_manifest_hash_rejected": attack_raises(
            lambda: _manifest_text("bad  x\n")),
        "manifest_traversal_rejected": attack_raises(
            lambda: _manifest_text("0" * 64 + "  ../x\n")),
        "byte_mutation_changes_sha": sha256_bytes(
            PATHS["wrapper"].read_bytes() + b"x") != EXPECTED_SHA["wrapper"],
    }
    require(all(strict_attacks.values()), "strict attack failure")

    prod_input = verify_manifest(PATHS["prod_input_manifest"], HW)
    prod_output = verify_manifest(PATHS["prod_output_manifest"], HW,
                                  allow_absolute=True, absolute_root=PROD_RUN)
    prod_runner = verify_manifest(PATHS["prod_runner_manifest"], HW)
    ind_input = verify_manifest(PATHS["ind_input_manifest"], HW)
    ind_output = verify_manifest(PATHS["ind_output_manifest"], HW,
                                 allow_absolute=True, absolute_root=IND_RUN)
    require(not prod_input["failed"] and not prod_output["failed"]
            and not prod_runner["failed"] and not ind_input["failed"]
            and not ind_output["failed"], "manifest verification failure")
    require(PATHS["prod_compile_rc"].read_text().strip() == "0"
            and PATHS["prod_sim_rc"].read_text().strip() == "0"
            and PATHS["ind_compile_rc"].read_text().strip() == "0"
            and PATHS["ind_sim_rc"].read_text().strip() == "0",
            "nonzero compile/sim receipt")

    sim_text = PATHS["prod_sim_log"].read_text(encoding="utf-8")
    pass_line, pass_fields = parse_pass_line(sim_text)
    require(pass_line == contract["expected_pass_line"], "pass line drift")
    expected_arithmetic = {
        "descriptors": 2,
        "ingress_events": 2 * 128 * 384,
        "active_keys": 2 * 128,
        "weight_prefetches": 2 * 128,
        "weight_load_tokens": 2 * 128 * 3,
        "service_event_tokens": 2 * 128 * 384,
        "service_tokens": 2 * 128 * (3 + 384),
        "tail_bypassed_first_events": 2 * 128,
        "zero_bubble_key_transitions": 2 * 127,
        "mapped_accumulator_updates": 2 * 128 * 384,
        "accumulator_writes": 2 * 128 * 384,
        "mapped_update_ii1_pairs": 2 * 128 * 383,
        "lane_read_write_overlap_cycles": 2 * 128 * 383,
        "descriptor_done_pulses": 2,
        "commit_vectors": 8 * 384,
        "commit_lane_checks": 8 * 384 * 96,
        "logical_accumulator_bytes": 384 * 8 * 96 * 19 // 8,
    }
    for key, value in expected_arithmetic.items():
        contract_value = contract["directed_scope"].get(key,
            contract["architecture"].get(key))
        if key == "logical_accumulator_bytes":
            contract_value = contract["architecture"]["accumulator_payload_bytes"]
        require(contract_value == value,
                "contract arithmetic mismatch " + key)

    assert_text = PATHS["prod_assert_report"].read_text(encoding="utf-8")
    cover_expected = contract["directed_scope"]["sva_cover_expected_matches"]
    cover_actual = {}
    for name, expected in cover_expected.items():
        match = re.search(re.escape(name) + r", .*? (\d+) match", assert_text)
        require(match is not None, "missing cover " + name)
        cover_actual[name] = int(match.group(1))
        require(cover_actual[name] == expected, "cover mismatch " + name)

    wrapper_text = PATHS["wrapper"].read_text(encoding="utf-8")
    numeric_text = PATHS["numeric"].read_text(encoding="utf-8")
    prod_tb_text = PATHS["tb"].read_text(encoding="utf-8")
    sva_text = PATHS["sva"].read_text(encoding="utf-8")
    require("logic [11:0] service_destination_row;" in wrapper_text,
            "destination row declaration drift")
    require(".service_destination_row(service_destination_row)" in wrapper_text,
            "scheduler destination connection drift")
    require(".service_row_offset(service_row_offset)" in wrapper_text,
            "numeric local row connection drift")
    require("service_destination_row" not in numeric_text,
            "numeric suddenly consumed destination identity")
    require("weight_prefetch_context" in wrapper_text
            and "output logic [6:0]" in wrapper_text
            and "weight_rd_key" in wrapper_text,
            "weight identity interface drift")
    require("downstream_backpressure_cycles=0" in pass_line,
            "positive run downstream stall boundary drift")
    require("weight_value(weight_rd_key" in prod_tb_text,
            "synthetic key-only weight model drift")
    require("protocol_attacks=1" in pass_line
            and "missing-window attack" in prod_tb_text,
            "production negative scope drift")
    require("protocol_error |-> !commit_valid" not in sva_text,
            "fault commit suppression assertion unexpectedly present")
    require("scheduler_protocol_error" not in numeric_text,
            "scheduler fault unexpectedly propagated to numeric island")

    ind_sim = PATHS["ind_sim_log"].read_text(encoding="utf-8")
    counterexamples = {
        "held_valid_grace": "PASS M121 independent counterexamples held_valid_grace=1" in ind_sim,
        "whole_descriptor_replay_accepted": "COUNTEREXAMPLE descriptor_replay_accepted accepts=2 closes=2 loads=6 events=2 updates=2 protocol_error=0" in ind_sim,
        "scheduler_fault_commit_escape": "COUNTEREXAMPLE scheduler_fault_commit_escape end_accept=1 commit_valid_under_top_error=1 scheduler_error=1 numeric_error=0" in ind_sim,
        "delayed_response_data_corruption": "COUNTEREXAMPLE delayed_weight_response counters_ok=1 protocol_error=0 loads=3 events=1 updates=1 got_lane0=0 expected_lane0=-128 got_lane95=-93 expected_lane95=67" in ind_sim,
    }
    require(all(counterexamples.values()), "independent counterexample missing")
    require(independent_receipt["commercial_vcs"] == "true",
            "independent VCS receipt drift")
    require(m119_review["severity_counts"]["P0"] == 1,
            "M119 triggering P0 drift")

    false_claims = (
        "heldout_trace_duplicate_retry_escape_replay",
        "foundry_weight_sram_macro", "foundry_accumulator_sram_macro",
        "macro_inclusive_ppa", "module_cycle_projection", "physical_speedup",
        "system_speedup", "headline")
    require(all(contract["admission"][key] is False for key in false_claims),
            "claim boundary overreach")
    require(receipt["heldout_trace_duplicate_retry_escape_replay"] == "false"
            and receipt["module_cycle_projection_admitted"] == "false"
            and receipt["physical_speedup"] == "false"
            and receipt["system_speedup"] == "false"
            and receipt["headline"] == "false", "receipt boundary overreach")

    payload = {
        "schema": "m121_w384_scheduler_numeric_island_independent_audit_v1",
        "status": "P0_COMBINED_FAULT_NOT_FAIL_CLOSED_DIRECTED_HAPPY_PATH_AND_COUNTEREXAMPLES_VCS_VERIFIED",
        "identity": actual_sha,
        "strict_attacks": strict_attacks,
        "production_sealed_run": {
            "commercial_vcs": True,
            "compile_rc": 0,
            "sim_rc": 0,
            "input_manifest": prod_input,
            "output_manifest": prod_output,
            "runner_manifest": prod_runner,
            "pass_line_exact_contract_match": True,
            "cover_matches": cover_actual,
            "assertion_failures": 0,
        },
        "independent_arithmetic": expected_arithmetic,
        "independent_commercial_vcs_counterexamples": {
            "compile_rc": 0,
            "sim_rc": 0,
            "input_manifest": ind_input,
            "output_manifest": ind_output,
            "counterexamples": counterexamples,
            "delayed_response_observation": {
                "token_counts_correct": True,
                "protocol_error": False,
                "got_lane0": 0,
                "expected_lane0": -128,
                "got_lane95": -93,
                "expected_lane95": 67,
            },
        },
        "static_boundary_audit": {
            "scheduler_destination_row_generated_but_not_consumed_by_numeric": True,
            "weight_read_request_has_key_and_beat_but_no_context_or_response_valid": True,
            "production_weight_model_key_only_not_context_indexed": True,
            "positive_numeric_downstream_backpressure_cycles": 0,
            "production_protocol_attack_classes": 1,
            "production_attack_is_missing_window_only": True,
            "scheduler_fault_not_propagated_into_numeric_abort": True,
            "top_commit_not_gated_by_combined_protocol_error": True,
            "sva_fault_commit_suppression_absent": True,
        },
        "m119_p0_status": {
            "same_valid_held_after_accept_is_grace_suppressed": True,
            "scheduler_owns_each_generated_service_token": True,
            "whole_descriptor_same_identity_replay_after_low_gap_is_accepted_twice": True,
            "heldout_duplicate_retry_escape_replay_admitted": False,
            "verdict": "DIRECTED_HANDSHAKE_CUT_CLOSED_GENERAL_RETRY_EXACT_ONCE_REMAINS_OPEN_AND_CORRECTLY_UNADMITTED",
        },
        "claim_boundary": {
            "directed_scheduler_numeric_island_vcs": True,
            "heldout_trace_duplicate_retry_escape_replay": False,
            "foundry_weight_sram_macro": False,
            "foundry_accumulator_sram_macro": False,
            "macro_inclusive_ppa": False,
            "module_cycle_projection_admitted": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
            "performance_overclaim_found": False,
            "fault_policy_overclaim_found": True,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("P0 M121 independent production_inputs={} outputs={} counterexamples={} commit_escape=1".format(
        prod_input["entries"], prod_output["entries"], len(counterexamples)),
        flush=True)


def _receipt_text(text):
    result = {}
    for line in text.splitlines():
        require("=" in line, "malformed receipt line")
        key, value = line.split("=", 1)
        require(key not in result, "duplicate receipt key")
        result[key] = value
    return result


def _manifest_text(text):
    seen = set()
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "malformed manifest line")
        raw = match.group(2)
        require(".." not in Path(raw).parts, "manifest traversal")
        require(raw not in seen, "duplicate manifest path")
        seen.add(raw)
    return seen


if __name__ == "__main__":
    main()
