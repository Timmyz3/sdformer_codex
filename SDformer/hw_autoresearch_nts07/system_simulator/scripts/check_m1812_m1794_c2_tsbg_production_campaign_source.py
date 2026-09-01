#!/usr/bin/env python3
"""Fail-closed M1812 source/runtime checker; never launches EDA."""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
M1794_CHECKER = HW / "system_simulator/scripts/check_m1794_c2_tsbg_reviewer_repair_source.py"
BASE_SPEC = importlib.util.spec_from_file_location("m1794_checker_for_m1812",
                                                   str(M1794_CHECKER))
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("M1794 checker unavailable")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

RTL = BASE.RTL
TB = BASE.TB
SVA = BASE.SVA
FILELIST = BASE.FILELIST
M1794_CONTRACT = BASE.CONTRACT
M1794_AUTHOR = HW / "reviews/m1794_m1788_c2_tsbg_reviewer_repair_source_author_receipt_r1_20260902"
M1795 = HW / "reviews/m1795_m1794_c2_tsbg_reviewer_repair_source_hammer_r1_20260902"
DOC359 = BASE.DOC359

RUNNER = HW / "dc_handoff/scripts/run_m1812_m1794_c2_tsbg_directed_vcs_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1812_m1794_c2_tsbg_production_campaign_source.py"
CONTRACT = HW / "contracts/m1812_m1795_m1794_c2_tsbg_production_campaign_source_contract_r1_20260902.json"

CLAIMS = dict((key, False) for key in (
    "vcs", "dc", "ptpx", "area", "energy", "same_resource_result",
    "paper_admitted", "component_speedup", "system_speedup", "headline"))

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    RTL: "283ef29727095255c8502a6f12f66170a41147f16358e09862a46d1d30dc4365",
    TB: "4d064937b53ea6b23a4917a1fe13051ea31c06ab45872496e2611f3d3ba6b6a3",
    SVA: "21e6d7e164fc57cc2f9ea40b584311fc65633ff3e3931ece9a7c8f9893ed631f",
    FILELIST: "f9c1b48634ab802b50dd53c432ae9c2dd54efaebd93efef4bfce084210b2db86",
    M1794_CHECKER: "7b774276c3f257147c2081b8abf17814ed8393d55b3219d72ddb2e63eb7b8ee5",
    M1794_CONTRACT: "263529c4bfbdae896a69320a7ddf306c2f3b1f05739c09ee8b4fea5ca12dad18",
    M1794_AUTHOR / "receipt.json": "87352467fcfad64c7631a53cc265ac4d59b128acbeb8a45b54d649cd77416f62",
    M1794_AUTHOR / "SHA256SUMS": "6ff306d16c224e82a2cb2dc2a041dd7bf81a27dae14049093a404163bd1c9b01",
    M1794_AUTHOR / "SHA256SUMS.seal.sha256": "e5f2c46e66d47679341e5f10ffe94e4804887bebcf1e7e46ec45831961d37f5a",
    M1795 / "review.json": "4a8ba47c085920e047e0db4ac1a75fefce0eb99f515efe94acda8a2b7f639a0e",
    M1795 / "SHA256SUMS": "50a027e0c6ac0732e305821835de6029835150fd70766ebc963f3973c8902aab",
    M1795 / "SHA256SUMS.seal.sha256": "bed2a3dac746ddb84643c413098e72974e832ce3d9200ee58958a3dfb26b4c53",
}

TB_TOKENS = (
    "if (tsbg.mem_rsp_accept[3] && !tsbg.inject_replay[3]",
    "saved_rsp_epoch <= tsbg.mem_rsp_epoch[3]",
    "saved_rsp_slot <= tsbg.mem_rsp_slot[3]",
    "saved_rsp_generation <= tsbg.mem_rsp_generation[3]",
    "saved_rsp_tag <= tsbg.mem_rsp_tag[3]",
    "saved_rsp_weight[lane] <= tsbg.mem_rsp_weight[3][lane]",
    "tsbg.replay_epoch[3] = saved_rsp_epoch",
    "tsbg.replay_slot[3] = saved_rsp_slot",
    "tsbg.replay_generation[3] = saved_rsp_generation",
    "tsbg.replay_tag[3] = saved_rsp_tag",
    "tsbg.replay_weight[3][lane] = saved_rsp_weight[lane]",
    "if (tsbg.mem_rsp_accept[3])",
    "replay_accept_count != 0",
    "!tsbg.protocol_error || !tsbg.stale_response_seen",
    "load_minimal_legal_workload();",
    "base.issue_count != 96 || tsbg.issue_count != 96",
    "base.commit_count != 48 || tsbg.commit_count != 48",
    "terminal_base != 8 || terminal_tsbg != 8",
    "post_reset_legal_service_count = post_reset_legal_service_count + 1",
    "reset_recovery_count != 2",
    "full_base_done_cycle * 1.0 / full_tsbg_done_cycle < 1.15",
    "PASS_M1794_C2_TSBG_B8_REAL_M803_TYPED_SIGNED_DIRECTED",
)
SVA_TOKENS = (
    "cp_reset_recovery_minimum_one_cycle",
    "disable iff (1'b0)",
    "protocol_error ##[1:8] rst_core[*1:8] ##1 !rst_core",
    "##[1:300000] (commit_accept && commit_terminal && !protocol_error)",
    "ap_fault_is_sticky",
    "ap_no_legal_overflow",
)
RUNNER_TOKENS = (
    "results/.m1812_m1794_tsbg_directed_vcs_attempt_consumed",
    "date_dual_synopsys_same_uid_eda_queue.lock",
    "M1812_EXPECTED_RUNNER_SHA256",
    "M1812_EXPECTED_SOURCE_CONTRACT_SHA256",
    "M1812_EXPECTED_M1813_MANIFEST_SHA256",
    "M1812_EXPECTED_M1813_OUTER_FILE_SHA256",
    "M1812_EXPECTED_M1813_REVIEW_SHA256",
    "M1812_EXPECTED_M1814_RELEASE_SHA256",
    "M1812_EXPECTED_M1814_SIDECAR_SHA256",
    "M1812_EXPECTED_M1814_OUTER_FILE_SHA256",
    "PASS_M1813_M1812_TSBG_PRODUCTION_CAMPAIGN_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_DIRECTED_VCS",
    "m1814_m1813_m1812_m1794_c2_tsbg_directed_vcs_launch_release_r1_v1",
    "AUTHORIZE_ONE_FRESH_M1812_M1794_TSBG_DIRECTED_VCS_CAMPAIGN",
    "mapping.get(\"review.json\")",
    "verify_contract_double_seal()",
    "release.get(\"identity\") != expected_identity",
    "release.get(\"prelaunch_claim_boundary\") != PRELAUNCH_CLAIMS",
    "release.get(\"measurement_boundary\") != RELEASE_BOUNDARY",
    "release.get(\"attempt_uniqueness\") != ATTEMPT_UNIQUENESS",
    "release.get(\"fresh_execution_budget\") != dict(",
    "verify_file_double_seal(",
    "failure_quarantine_only_after_attempt_consumed",
    "if state[\"attempt\"] and not state[\"complete\"]",
    "publish_no_replace(STAGE, RESULT)",
    "automatic_retry\": False",
    "reuse_prior_simv\": False",
    "-assert", "svaext",
)

# Each mutation is evaluated against in-memory source text by the M1812 test;
# contract SHA drift is deliberately not counted as semantic rejection.
MUTATION_SPECS = (
    ("capture_guard", "tb", "if (tsbg.mem_rsp_accept[3] && !tsbg.inject_replay[3]", "if (tsbg.mem_rsp_valid[3]"),
    ("capture_epoch", "tb", "saved_rsp_epoch <= tsbg.mem_rsp_epoch[3]", "saved_rsp_epoch <= 16'h0"),
    ("capture_slot", "tb", "saved_rsp_slot <= tsbg.mem_rsp_slot[3]", "saved_rsp_slot <= 3'h0"),
    ("capture_generation", "tb", "saved_rsp_generation <= tsbg.mem_rsp_generation[3]", "saved_rsp_generation <= 32'h0"),
    ("capture_tag", "tb", "saved_rsp_tag <= tsbg.mem_rsp_tag[3]", "saved_rsp_tag <= 24'h0"),
    ("capture_payload", "tb", "saved_rsp_weight[lane] <= tsbg.mem_rsp_weight[3][lane]", "saved_rsp_weight[lane] <= 8'sh0"),
    ("replay_epoch", "tb", "tsbg.replay_epoch[3] = saved_rsp_epoch", "tsbg.replay_epoch[3] = 16'h0"),
    ("replay_slot", "tb", "tsbg.replay_slot[3] = saved_rsp_slot", "tsbg.replay_slot[3] = 3'h0"),
    ("replay_generation", "tb", "tsbg.replay_generation[3] = saved_rsp_generation", "tsbg.replay_generation[3] = 32'h0"),
    ("replay_tag", "tb", "tsbg.replay_tag[3] = saved_rsp_tag", "tsbg.replay_tag[3] = 24'h0"),
    ("replay_payload", "tb", "tsbg.replay_weight[3][lane] = saved_rsp_weight[lane]", "tsbg.replay_weight[3][lane] = 8'sh0"),
    ("zero_accept_immediate", "tb", "if (tsbg.mem_rsp_accept[3])", "if (1'b0 && tsbg.mem_rsp_accept[3])"),
    ("zero_accept_ledger", "tb", "replay_accept_count != 0", "replay_accept_count != 999"),
    ("sticky_protocol_stale", "tb", "!tsbg.protocol_error || !tsbg.stale_response_seen", "!tsbg.protocol_error"),
    ("three_clock_resets", "tb", "repeat (3) @(posedge clk_core)", "repeat (0) @(posedge clk_core)"),
    ("post_reset_service", "tb", "load_minimal_legal_workload();", "load_workload();"),
    ("post_reset_issue", "tb", "base.issue_count != 96 || tsbg.issue_count != 96", "base.issue_count != 0 || tsbg.issue_count != 0"),
    ("post_reset_commit", "tb", "base.commit_count != 48 || tsbg.commit_count != 48", "base.commit_count != 0 || tsbg.commit_count != 0"),
    ("post_reset_terminal", "tb", "terminal_base != 8 || terminal_tsbg != 8", "terminal_base != 0 || terminal_tsbg != 0"),
    ("post_reset_recovery_count", "tb", "post_reset_legal_service_count = post_reset_legal_service_count + 1", "post_reset_legal_service_count = 1"),
    ("reset_count_gate", "tb", "reset_recovery_count != 2", "reset_recovery_count != 0"),
    ("directed_cycle_gate", "tb", "full_base_done_cycle * 1.0 / full_tsbg_done_cycle < 1.15", "full_base_done_cycle * 1.0 / full_tsbg_done_cycle < 0.0"),
    ("sva_reset_range", "sva", "protocol_error ##[1:8] rst_core[*1:8] ##1 !rst_core", "protocol_error ##1 rst_core ##1 !rst_core"),
    ("sva_terminal", "sva", "##[1:300000] (commit_accept && commit_terminal && !protocol_error)", "##1 !protocol_error"),
    ("sva_disable", "sva", "disable iff (1'b0)", "disable iff (rst_core)"),
    ("runner_attempt", "runner", "results/.m1812_m1794_tsbg_directed_vcs_attempt_consumed", "results/.wrong_attempt"),
    ("runner_contract_pin", "runner", "M1812_EXPECTED_SOURCE_CONTRACT_SHA256", "M1812_UNPINNED_CONTRACT"),
    ("runner_review_pin", "runner", "M1812_EXPECTED_M1813_REVIEW_SHA256", "M1812_UNPINNED_REVIEW"),
    ("runner_review_manifest_pin", "runner", "M1812_EXPECTED_M1813_MANIFEST_SHA256", "M1812_UNPINNED_REVIEW_MANIFEST"),
    ("runner_review_outer_pin", "runner", "M1812_EXPECTED_M1813_OUTER_FILE_SHA256", "M1812_UNPINNED_REVIEW_OUTER"),
    ("runner_release_pin", "runner", "M1812_EXPECTED_M1814_RELEASE_SHA256", "M1812_UNPINNED_RELEASE"),
    ("runner_release_sidecar_pin", "runner", "M1812_EXPECTED_M1814_SIDECAR_SHA256", "M1812_UNPINNED_RELEASE_SIDECAR"),
    ("runner_release_outer_pin", "runner", "M1812_EXPECTED_M1814_OUTER_FILE_SHA256", "M1812_UNPINNED_RELEASE_OUTER"),
    ("runner_review_status", "runner", "PASS_M1813_M1812_TSBG_PRODUCTION_CAMPAIGN_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_DIRECTED_VCS", "WRONG_M1813_STATUS"),
    ("runner_release_schema", "runner", "m1814_m1813_m1812_m1794_c2_tsbg_directed_vcs_launch_release_r1_v1", "wrong_release_schema"),
    ("runner_release_status", "runner", "AUTHORIZE_ONE_FRESH_M1812_M1794_TSBG_DIRECTED_VCS_CAMPAIGN", "WRONG_M1814_STATUS"),
    ("runner_review_transitive_seal", "runner", "mapping.get(\"review.json\")", "mapping.get(\"unsealed.json\")"),
    ("runner_contract_double_seal", "runner", "verify_contract_double_seal()", "verify_contract_single_seal()"),
    ("runner_identity", "runner", "release.get(\"identity\") != expected_identity", "False"),
    ("runner_claims", "runner", "release.get(\"prelaunch_claim_boundary\") != PRELAUNCH_CLAIMS", "False"),
    ("runner_measurement_boundary", "runner", "release.get(\"measurement_boundary\") != RELEASE_BOUNDARY", "False"),
    ("runner_attempt_identity", "runner", "release.get(\"attempt_uniqueness\") != ATTEMPT_UNIQUENESS", "False"),
    ("runner_budget", "runner", "release.get(\"fresh_execution_budget\") != dict(", "False and dict("),
    ("runner_double_seal", "runner", "verify_file_double_seal(", "verify_file_single_seal("),
    ("runner_no_retry", "runner", "automatic_retry\": False", "automatic_retry\": True"),
    ("runner_failure_after_attempt", "runner", "if state[\"attempt\"] and not state[\"complete\"]", "if not state[\"complete\"]"),
    ("runner_atomic_publish", "runner", "publish_no_replace(STAGE, RESULT)", "STAGE.rename(RESULT)"),
    ("runner_sva_enabled", "runner", "svaext", "no_sva"),
)


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_sealed_directory(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal")
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*")); name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts
             and name not in mapping, "unsafe manifest")
        need((root / rel).is_file() and not (root / rel).is_symlink()
             and sha(root / rel) == fields[0], "manifest drift " + name)
        mapping[name] = fields[0]
    need(mapping.get("review.json") == sha(root / "review.json")
         if (root / "review.json").exists() else True,
         "review not transitively sealed")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def validate_semantics(texts):
    tb = texts[TB]
    sva = texts[SVA]
    runner = texts[RUNNER]
    active_tb = strip_comments(tb).lower()
    need("force " not in active_tb and "release " not in active_tb
         and "$root" not in active_tb, "TB hierarchy bypass")
    for token in TB_TOKENS:
        need(token in tb, "TB omits " + token)
    need(tb.count("repeat (3) @(posedge clk_core)") == 2,
         "TB must retain both three-clock resets")
    for token in SVA_TOKENS:
        need(token in sva, "SVA omits " + token)
    for token in RUNNER_TOKENS:
        need(token in runner, "runner omits " + token)
    for forbidden in ("+initreg", "+notimingcheck", "+no_notifier",
                      "+nospecify", "deposit(", "vpi_handle_by_name"):
        need(forbidden not in runner.lower(), "runner bypass " + forbidden)
    need(runner.count("state[\"license_queries\"] += 1") == 1
         and runner.count("state[\"vcs_compiles\"] += 1") == 1
         and runner.count("state[\"simv_runs\"] += 1") == 1,
         "runner one-shot budget")


def validate_sources():
    BASE.validate_sources()
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift " + str(path))
    verify_sealed_directory(M1794_AUTHOR)
    verify_sealed_directory(M1795)
    review = strict_json(M1795 / "review.json")
    need(review.get("status") ==
         "FAIL_CLOSED_M1795_M1794_C2_TSBG_SOURCE_HAMMER__P1_2__NO_VCS_NO_EDA"
         and review.get("severity_counts") == {"p0": 0, "p1": 2, "p2": 0},
         "M1795 disposition")

    source_paths = (RUNNER, CHECKER, TEST)
    for path in source_paths:
        need(path.is_file() and not path.is_symlink(), "source absent " + str(path))
    texts = {RTL: RTL.read_text(), TB: TB.read_text(), SVA: SVA.read_text(),
             RUNNER: RUNNER.read_text()}
    validate_semantics(texts)
    contract = strict_json(CONTRACT)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract outer")
    need(contract.get("schema") ==
         "m1812_m1795_m1794_c2_tsbg_production_campaign_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1795_P1_FIXED__M1813_REVIEW_AND_M1814_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIMS, "claim promotion")
    need(contract.get("execution_budget") == {
        "license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
        "automatic_retry": False, "reuse_prior_simv": False},
        "contract budget")
    mapping = dict((row.get("path"), row.get("sha256"))
                   for row in contract.get("source_files", []))
    need(len(mapping) == len(source_paths), "source inventory cardinality")
    for path in source_paths:
        need(mapping.get(str(path.relative_to(HW))) == sha(path),
             "source inventory drift " + str(path))
    return {"status": "PASS_M1812_PRODUCTION_CAMPAIGN_SOURCE_STATIC",
            "source_files": len(source_paths),
            "semantic_mutations_required": len(MUTATION_SPECS),
            "claim_boundary": dict(CLAIMS), "eda_runs": 0}


def validate_runtime(path):
    text = Path(path).read_text(errors="strict")
    token = "PASS_M1794_C2_TSBG_B8_REAL_M803_TYPED_SIGNED_DIRECTED"
    need(text.count(token) == 1, "runtime PASS count")
    need("$fatal" not in text and "Assertion failed" not in text
         and "Error-[" not in text and "Fatal:" not in text,
         "runtime failure signature")
    pattern = (r"PASS_M1794_C2_TSBG_B8_REAL_M803_TYPED_SIGNED_DIRECTED\s+"
               r"rows=(\d+) issues=(\d+) products=(\d+) commits=(\d+)\s+"
               r"bundles_base=(\d+) bundles_tsbg=(\d+) scalar_base=(\d+) scalar_tsbg=(\d+)\s+"
               r"stale=(\d+) retired_replay=(\d+) replay_accept=(\d+) reset=(\d+) recovery=(\d+)")
    hits = re.findall(pattern, text)
    need(hits == [("96", "1152", "18432", "48", "1152", "144",
                   "9216", "1152", "1", "1", "0", "2", "1")],
         "runtime ledger")
    return {
        "status": "PASS_M1812_M1794_DIRECTED_RUNTIME_PENDING_RESULT_HAMMER",
        "rows_each": 96, "issues_each": 1152,
        "signed_products_each": 18432, "commits_each": 48,
        "aggregate_bundle_beats_baseline": 1152,
        "aggregate_bundle_beats_tsbg": 144,
        "scalar_bank_beats_baseline": 9216,
        "scalar_bank_beats_tsbg": 1152,
        "retired_identity_replay_accepts": 0,
        "resets": 2, "post_reset_complete_services": 1,
        "directed_local_cycle_gate_ge_1p15": True,
        "claim_boundary": {
            "directed_behavioral_vcs": True,
            "numeric_and_protocol_checks": True,
            "directed_local_cycle_gate": True,
            "dc": False, "ptpx": False, "area": False, "energy": False,
            "same_resource_result": False, "paper_admitted": False,
            "component_speedup": False, "system_speedup": False,
            "headline": False}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-self-check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate_sources(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
