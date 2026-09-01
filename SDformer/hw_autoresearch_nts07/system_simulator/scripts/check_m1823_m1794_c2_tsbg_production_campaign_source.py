#!/usr/bin/env python3
"""Fail-closed M1823 source/runtime checker; never launches EDA."""
from __future__ import print_function

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
M1812_CHECKER = HW / "system_simulator/scripts/check_m1812_m1794_c2_tsbg_production_campaign_source.py"
BASE_SPEC = importlib.util.spec_from_file_location("m1812_checker_for_m1823",
                                                   str(M1812_CHECKER))
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("M1812 checker unavailable")
M1812_BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(M1812_BASE)
BASE = M1812_BASE.BASE

RTL = BASE.RTL
TB = BASE.TB
SVA = BASE.SVA
FILELIST = BASE.FILELIST
M1794_CONTRACT = BASE.CONTRACT
M1794_AUTHOR = HW / "reviews/m1794_m1788_c2_tsbg_reviewer_repair_source_author_receipt_r1_20260902"
M1795 = HW / "reviews/m1795_m1794_c2_tsbg_reviewer_repair_source_hammer_r1_20260902"
DOC359 = BASE.DOC359
M1812_CONTRACT = M1812_BASE.CONTRACT
M1812_AUTHOR = HW / "reviews/m1812_m1795_m1794_c2_tsbg_production_campaign_source_author_receipt_r1_20260902"
M1813 = HW / "reviews/m1813_m1812_m1794_c2_tsbg_production_campaign_source_hammer_r1_20260902"

RUNNER = HW / "dc_handoff/scripts/run_m1823_m1794_c2_tsbg_directed_vcs_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1823_m1794_c2_tsbg_production_campaign_source.py"
CONTRACT = HW / "contracts/m1823_m1813_m1812_m1794_c2_tsbg_production_campaign_source_contract_r1_20260902.json"

CLAIMS = dict((key, False) for key in (
    "vcs", "dc", "ptpx", "area", "energy", "same_resource_result",
    "paper_admitted", "component_speedup", "system_speedup", "headline"))

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1812_CHECKER: "10db56b250575f7d15abf7e614b3d9c38c34a3a2a4199c3f9d18c8a1364470e6",
    M1812_CONTRACT: "e4c79c016cc11248b203383e97ace197b6fd544ba8f2903e9182b35e46156fd1",
    Path(str(M1812_CONTRACT) + ".sha256"): "75f4af0c9dda9c963218d966ea5d57085608e56085c8bd8439b7c7eb9fd88ea3",
    Path(str(M1812_CONTRACT) + ".sha256.seal.sha256"): "9658f290699ed41f045217a305f0125923c228a523ad58c7f96af1eed3c3ed8a",
    M1812_AUTHOR / "receipt.json": "31ea59bf2bc67836975d6ba39d1084cd5fbf3acefc80f6ddeb1aa81ea6bbf24a",
    M1812_AUTHOR / "SHA256SUMS": "97eadbff39df61715bfdda0a785ea17775d93e4b4480391bc535926ee6550358",
    M1812_AUTHOR / "SHA256SUMS.seal.sha256": "7a84920089f32846138de053d5abf34ebf3f1df0e87aa7e874ace3433928b67b",
    M1813 / "review.json": "f044d0f31819acc3fbf39d75fdcd9269bd87639bee989a55727d55035f23e0ea",
    M1813 / "SHA256SUMS": "e745de007f2f6a7be157e0f85380a59f7b9a2188f583e06bb66b1f580b62f861",
    M1813 / "SHA256SUMS.seal.sha256": "8b94d590088cbb4d42d2c13fbaa81da363ea716decaa2b9e5eda583c87492d5a",
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
    "results/.m1823_m1794_tsbg_directed_vcs_attempt_consumed",
    "date_dual_synopsys_same_uid_eda_queue.lock",
    "M1823_EXPECTED_RUNNER_SHA256",
    "M1823_EXPECTED_SOURCE_CONTRACT_SHA256",
    "M1823_EXPECTED_M1824_MANIFEST_SHA256",
    "M1823_EXPECTED_M1824_OUTER_FILE_SHA256",
    "M1823_EXPECTED_M1824_REVIEW_SHA256",
    "M1823_EXPECTED_M1825_RELEASE_SHA256",
    "M1823_EXPECTED_M1825_SIDECAR_SHA256",
    "M1823_EXPECTED_M1825_OUTER_FILE_SHA256",
    "PASS_M1824_M1823_TSBG_PRODUCTION_CAMPAIGN_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_DIRECTED_VCS",
    "m1825_m1824_m1823_m1794_c2_tsbg_directed_vcs_launch_release_r1_v1",
    "AUTHORIZE_ONE_FRESH_M1823_M1794_TSBG_DIRECTED_VCS_CAMPAIGN",
    "mapping.get(\"review.json\")",
    "verify_contract_double_seal()",
    "release.get(\"identity\") != expected_identity",
    "release.get(\"prelaunch_claim_boundary\") != PRELAUNCH_CLAIMS",
    "release.get(\"measurement_boundary\") != RELEASE_BOUNDARY",
    "release.get(\"attempt_uniqueness\") != ATTEMPT_UNIQUENESS",
    "release.get(\"fresh_execution_budget\") != dict(",
    "verify_file_double_seal(",
    "m1812_source_contract_sha256",
    "m1813_review_sha256",
    "m1794_source_contract_sha256",
    "m1795_review_sha256",
    "docs359_sha256",
    "failure_quarantine_only_after_attempt_consumed",
    "if state[\"attempt\"] and not state[\"complete\"]",
    "publish_no_replace(STAGE, RESULT)",
    "automatic_retry\": False",
    "reuse_prior_simv\": False",
    "-assert", "svaext",
)

# Each mutation is evaluated against in-memory source text by the M1823 test;
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
    ("runner_attempt", "runner", "results/.m1823_m1794_tsbg_directed_vcs_attempt_consumed", "results/.wrong_attempt"),
    ("runner_contract_pin", "runner", "M1823_EXPECTED_SOURCE_CONTRACT_SHA256", "M1823_UNPINNED_CONTRACT"),
    ("runner_review_pin", "runner", "M1823_EXPECTED_M1824_REVIEW_SHA256", "M1823_UNPINNED_REVIEW"),
    ("runner_review_manifest_pin", "runner", "M1823_EXPECTED_M1824_MANIFEST_SHA256", "M1823_UNPINNED_REVIEW_MANIFEST"),
    ("runner_review_outer_pin", "runner", "M1823_EXPECTED_M1824_OUTER_FILE_SHA256", "M1823_UNPINNED_REVIEW_OUTER"),
    ("runner_release_pin", "runner", "M1823_EXPECTED_M1825_RELEASE_SHA256", "M1823_UNPINNED_RELEASE"),
    ("runner_release_sidecar_pin", "runner", "M1823_EXPECTED_M1825_SIDECAR_SHA256", "M1823_UNPINNED_RELEASE_SIDECAR"),
    ("runner_release_outer_pin", "runner", "M1823_EXPECTED_M1825_OUTER_FILE_SHA256", "M1823_UNPINNED_RELEASE_OUTER"),
    ("runner_review_status", "runner", "PASS_M1824_M1823_TSBG_PRODUCTION_CAMPAIGN_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_DIRECTED_VCS", "WRONG_M1824_STATUS"),
    ("runner_release_schema", "runner", "m1825_m1824_m1823_m1794_c2_tsbg_directed_vcs_launch_release_r1_v1", "wrong_release_schema"),
    ("runner_release_status", "runner", "AUTHORIZE_ONE_FRESH_M1823_M1794_TSBG_DIRECTED_VCS_CAMPAIGN", "WRONG_M1825_STATUS"),
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
    ("governance_reachable_verify_authority", "runner",
     "        verify_authority()\n        CHECK.validate_sources()",
     "        if False: verify_authority()\n        CHECK.validate_sources()"),
    ("governance_reachable_validate_sources", "runner",
     "        CHECK.validate_sources()\n        namespaces_fresh()",
     "        if False: CHECK.validate_sources()\n        namespaces_fresh()"),
    ("governance_reachable_namespaces_fresh", "runner",
     "        namespaces_fresh()\n        fcntl.flock",
     "        if False: namespaces_fresh()\n        fcntl.flock"),
    ("governance_reachable_collision_gate", "runner",
     "        collision_gate()\n        resource_gate()",
     "        if False: collision_gate()\n        resource_gate()"),
    ("governance_reachable_resource_gate", "runner",
     "        resource_gate()\n        namespaces_fresh()",
     "        if False: resource_gate()\n        namespaces_fresh()"),
    ("governance_attempt_state_transition", "runner",
     "        ATTEMPT.mkdir()\n        state[\"attempt\"] = True",
     "        ATTEMPT.mkdir()\n        state[\"attempt\"] = False"),
    ("governance_identity_m1794", "runner",
     "\"m1794_source_contract_sha256\": sha(M1794_CONTRACT)",
     "\"m1794_source_contract_omitted\": sha(M1794_CONTRACT)"),
    ("governance_identity_m1795", "runner",
     "\"m1795_review_sha256\": sha(M1795_REVIEW)",
     "\"m1795_review_omitted\": sha(M1795_REVIEW)"),
    ("governance_identity_docs359", "runner",
     "\"docs359_sha256\": sha(DOC359)",
     "\"docs359_omitted\": sha(DOC359)"),
    ("governance_self_runner_pin", "runner",
     "exact(RUNNER, authority_pin(\"M1823_EXPECTED_RUNNER_SHA256\"))",
     "exact(CONTRACT, authority_pin(\"M1823_EXPECTED_RUNNER_SHA256\"))"),
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


def function_node(tree, name):
    hits = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name]
    need(len(hits) == 1, "runner function " + name)
    return hits[0]


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return prefix + "." + node.attr if prefix else node.attr
    return ""


def direct_call_name(statement):
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return dotted_name(statement.value.func)
    return ""


def string_value(node):
    if isinstance(node, ast.Str):
        return node.s
    constant = getattr(ast, "Constant", None)
    if constant is not None and isinstance(node, constant):
        return node.value if isinstance(node.value, str) else None
    return None


def state_attempt_target(node):
    if not isinstance(node, ast.Subscript):
        return False
    if not isinstance(node.value, ast.Name) or node.value.id != "state":
        return False
    value = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
    return string_value(value) == "attempt"


def is_true_node(node):
    if isinstance(node, ast.NameConstant):
        return node.value is True
    constant = getattr(ast, "Constant", None)
    return constant is not None and isinstance(node, constant) and node.value is True


def sha_of_name(node, name):
    return (isinstance(node, ast.Call) and dotted_name(node.func) == "sha"
            and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
            and node.args[0].id == name and not node.keywords)


def validate_runner_reachability(runner):
    try:
        tree = ast.parse(runner)
    except SyntaxError as error:
        raise RuntimeError("runner AST syntax " + str(error))

    main_node = function_node(tree, "main")
    main_tries = [node for node in main_node.body if isinstance(node, ast.Try)]
    need(len(main_tries) == 1, "main reachable try")
    body = main_tries[0].body
    direct = [direct_call_name(statement) for statement in body]
    required_prefix = [
        "verify_authority", "CHECK.validate_sources", "namespaces_fresh",
        "fcntl.flock", "fcntl.flock", "collision_gate", "resource_gate",
        "namespaces_fresh"]
    need(direct[:len(required_prefix)] == required_prefix,
         "main governance call path/order")

    mkdir_hits = [index for index, statement in enumerate(body)
                  if direct_call_name(statement) == "ATTEMPT.mkdir"]
    need(len(mkdir_hits) == 1 and mkdir_hits[0] + 1 < len(body),
         "attempt creation reachable")
    transition = body[mkdir_hits[0] + 1]
    need(isinstance(transition, ast.Assign)
         and len(transition.targets) == 1
         and state_attempt_target(transition.targets[0])
         and is_true_node(transition.value),
         "attempt state transition must immediately follow ATTEMPT.mkdir")

    run_node = function_node(tree, "run")
    run_direct = [direct_call_name(statement) for statement in run_node.body]
    need(run_direct[:2] == ["CHECK.validate_sources", "collision_gate"],
         "tool wrapper source/collision gates reachable")

    authority = function_node(tree, "verify_authority")
    authority_direct = [statement for statement in authority.body
                        if direct_call_name(statement) == "exact"]
    self_pins = []
    for statement in authority_direct:
        call = statement.value
        if (len(call.args) == 2 and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "RUNNER"
                and isinstance(call.args[1], ast.Call)
                and dotted_name(call.args[1].func) == "authority_pin"
                and len(call.args[1].args) == 1
                and string_value(call.args[1].args[0]) ==
                    "M1823_EXPECTED_RUNNER_SHA256"):
            self_pins.append(statement)
    need(len(self_pins) == 1, "exact external self-runner pin")

    assignments = [node for node in authority.body
                   if isinstance(node, ast.Assign)
                   and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "expected_identity"]
    need(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict),
         "expected_identity assignment")
    identity = {}
    for key, value in zip(assignments[0].value.keys, assignments[0].value.values):
        text = string_value(key)
        need(text is not None and text not in identity,
             "expected_identity duplicate/nonliteral key")
        identity[text] = value
    for key, source in (
            ("m1794_source_contract_sha256", "M1794_CONTRACT"),
            ("m1795_review_sha256", "M1795_REVIEW"),
            ("docs359_sha256", "DOC359")):
        need(key in identity and sha_of_name(identity[key], source),
             "expected_identity omits/misbinds " + key)


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
    validate_runner_reachability(runner)
    need(runner.count("state[\"license_queries\"] += 1") == 1
         and runner.count("state[\"vcs_compiles\"] += 1") == 1
         and runner.count("state[\"simv_runs\"] += 1") == 1,
         "runner one-shot budget")


def validate_sources():
    M1812_BASE.validate_sources()
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift " + str(path))
    verify_sealed_directory(M1812_AUTHOR)
    verify_sealed_directory(M1813)
    review = strict_json(M1813 / "review.json")
    need(review.get("status") ==
         "FAIL_CLOSED_M1813_M1812_TSBG_PRODUCTION_CAMPAIGN_SOURCE_HAMMER__P1_1__NO_VCS_NO_EDA"
         and review.get("severity_counts") == {"p0": 0, "p1": 1, "p2": 0},
         "M1813 disposition")
    need(review.get("independent_escape_probe", {}).get("attacks") == 9
         and review.get("independent_escape_probe", {}).get("rejected") == 0
         and review.get("independent_escape_probe", {}).get("escaped") == 9,
         "M1813 governance escape evidence")

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
         "m1823_m1813_m1812_m1794_c2_tsbg_production_campaign_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1813_P1_FIXED__M1824_REVIEW_AND_M1825_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIMS, "claim promotion")
    need(contract.get("execution_budget") == {
        "license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
        "automatic_retry": False, "reuse_prior_simv": False},
        "contract budget")
    need(contract.get("semantic_mutation_plan", {}).get("mutations") ==
         len(MUTATION_SPECS) and len(MUTATION_SPECS) >= 58,
         "mutation cardinality")
    need(contract.get("governance_closure", {}).get(
         "reachable_call_gates") == [
             "verify_authority", "CHECK.validate_sources",
             "namespaces_fresh", "collision_gate", "resource_gate"]
         and contract.get("governance_closure", {}).get(
             "attempt_state_transition") == "ATTEMPT.mkdir_then_state_attempt_true"
         and contract.get("governance_closure", {}).get(
             "release_identity_keys") == [
                 "m1794_source_contract_sha256",
                 "m1795_review_sha256", "docs359_sha256"]
         and contract.get("governance_closure", {}).get(
             "external_self_runner_pin") is True,
         "contract governance closure")
    mapping = dict((row.get("path"), row.get("sha256"))
                   for row in contract.get("source_files", []))
    need(len(mapping) == len(source_paths), "source inventory cardinality")
    for path in source_paths:
        need(mapping.get(str(path.relative_to(HW))) == sha(path),
             "source inventory drift " + str(path))
    return {"status": "PASS_M1823_PRODUCTION_CAMPAIGN_SOURCE_STATIC",
            "source_files": len(source_paths),
            "semantic_mutations_required": len(MUTATION_SPECS),
            "m1813_governance_escapes_closed": 9,
            "self_runner_pin_mutation": 1,
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
        "status": "PASS_M1823_M1794_DIRECTED_RUNTIME_PENDING_RESULT_HAMMER",
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
