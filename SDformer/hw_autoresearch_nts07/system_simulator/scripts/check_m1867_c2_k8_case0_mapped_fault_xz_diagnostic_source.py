#!/usr/bin/env python3
"""Mutation-resilient static checker for the M1867 diagnostic source."""
from __future__ import print_function

import ast
import hashlib
import json
from pathlib import Path
import re
import sys


HW = Path(__file__).resolve().parents[2]
PATHS = {
    "runner": HW / "dc_handoff/scripts/run_m1867_c2_k8_case0_mapped_fault_xz_diagnostic_one_shot.py",
    "tb": HW / "dc_handoff/tb/tb_m1867_c2_k8_case0_mapped_fault_xz_diagnostic.sv",
    "filelist": HW / "dc_handoff/filelists/date_m1867_c2_k8_case0_mapped_fault_xz_diagnostic.f",
    "checker": Path(__file__).resolve(),
    "test": HW / "system_simulator/tests/test_m1867_c2_k8_case0_mapped_fault_xz_diagnostic_source.py",
    "contract": HW / "contracts/m1867_m1863_m1862_c2_k8_case0_mapped_fault_xz_diagnostic_source_contract_r1_20260902.json",
}
CONTRACT = PATHS["contract"]
MAPPED_REL = "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902/k8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v"
MAPPED = HW / MAPPED_REL
MAPPED_SHA256 = "63605469818c36574ce9719130877610e79cf0c3b7317c0e69848539afa6b792"
FILELIST_REL = "dc_handoff/filelists/date_m1867_c2_k8_case0_mapped_fault_xz_diagnostic.f"
FILELIST = HW / FILELIST_REL
FILELIST_SHA256 = "e13944a7c57806f340cc8bc145be3b2aad3b8e2d08dcd3fa86518c8a689eaa13"
TOP = "tb_m1867_c2_k8_case0_mapped_fault_xz_diagnostic"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M1863_REVIEW_SHA256 = "b6b493b44cf5505ca9b2f70310f37827ad0d4da988316edae5365ad770d04810"
M1863_MANIFEST_SHA256 = "7d014d5ee955d9baf1f7719fa76879133d56770bde4644a89f98ede9300cd8c4"
M1863_OUTER_SHA256 = "30587c2a246827b4fd01d682a5d341c26e13c6b4949f22009838affde2198196"

CLAIMS = {
    "diagnostic_source_only": True,
    "m1845_retry": False,
    "m1856_launch": False,
    "mapped_functionality": False,
    "production_functionality": False,
    "power": False,
    "energy": False,
    "performance": False,
    "speedup": False,
    "system_speedup": False,
    "paper_citable": False,
    "headline": False,
}

NAMESPACES = {
    "ATTEMPT": 'ATTEMPT = HW / "results/.m1867_c2_k8_case0_mapped_fault_xz_diagnostic_attempt_consumed"',
    "RESULT": 'RESULT = HW / "results/m1867_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902"',
    "FAILURE": 'FAILURE = HW / "results/m1867_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902.failed_or_incomplete.quarantine"',
    "WORK": 'WORK = HW / ("results/.m1867_c2_k8_case0_mapped_fault_xz_diagnostic_work." + str(os.getpid()))',
    "STAGE": 'STAGE = HW / ("results/.m1867_c2_k8_case0_mapped_fault_xz_diagnostic_stage." + str(os.getpid()))',
    "FAIL_STAGE": 'FAIL_STAGE = HW / ("results/.m1867_c2_k8_case0_mapped_fault_xz_diagnostic_failure_stage." + str(os.getpid()))',
    "QUEUE": 'QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")',
    "LOCAL_LOCK": 'LOCAL_LOCK = Path("/tmp/m1867_c2_k8_case0_mapped_fault_xz_diagnostic.lock")',
}


class CheckFailure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_text(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise CheckFailure("duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           CheckFailure("nonfinite JSON " + token)))
    if type(value) is not dict:
        raise CheckFailure("JSON root")
    return value


def source_map():
    values = {}
    for name, path in PATHS.items():
        if not path.is_file() or path.is_symlink():
            raise CheckFailure("missing source " + str(path))
        values[name] = path.read_text()
    return values


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return (prefix + "." if prefix else "") + node.attr
    return ""


def literal(node):
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.NameConstant):
        return node.value
    constant = getattr(ast, "Constant", ())
    if constant and isinstance(node, constant):
        return node.value
    return None


def function(tree, name):
    rows = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name]
    if len(rows) != 1:
        raise CheckFailure("function cardinality " + name)
    return rows[0]


def calls(node, name):
    return [item for item in ast.walk(node)
            if isinstance(item, ast.Call) and call_name(item.func) == name]


def assignment_call(function_node, target, call):
    rows = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if (isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == target
                and isinstance(node.value, ast.Call)
                and call_name(node.value.func) == call):
            rows.append(node)
    if len(rows) != 1:
        raise CheckFailure("assignment-call cardinality " + target + "=" + call)
    return rows[0]


def direct_try_body(function_node):
    rows = [node for node in function_node.body if isinstance(node, ast.Try)]
    if len(rows) != 1:
        raise CheckFailure("main direct try cardinality")
    node = rows[0]
    if len(node.handlers) != 1 or node.orelse or len(node.finalbody) != 2:
        raise CheckFailure("main direct try shape")
    return node.body


def direct_assignment_call(statements, target, call):
    rows = []
    for node in statements:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == target
                and isinstance(node.value, ast.Call)
                and call_name(node.value.func) == call):
            rows.append(node)
    if len(rows) != 1:
        raise CheckFailure("direct assignment-call cardinality " + target + "=" + call)
    return rows[0], node_index(statements, rows[0])


def direct_expr_calls(statements, call):
    rows = []
    for index, node in enumerate(statements):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and call_name(node.value.func) == call):
            rows.append((node.value, index))
    return rows


def node_index(statements, target):
    for index, node in enumerate(statements):
        if node is target:
            return index
    raise CheckFailure("direct statement absent")


def arg_is_string(call, index, value):
    return len(call.args) > index and literal(call.args[index]) == value


def slash_string(node, left_name, right_value):
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Name) and node.left.id == left_name
            and literal(node.right) == right_value)


def check_runner(text):
    tree = ast.parse(text)
    main = function(tree, "main")
    runtime = function(tree, "run")
    main_body = direct_try_body(main)

    for name, exact_line in NAMESPACES.items():
        if text.count(exact_line) != 1:
            raise CheckFailure("exact namespace " + name)
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == name
                               for target in node.targets)]
        if len(assignments) != 1:
            raise CheckFailure("namespace AST cardinality " + name)

    authority, authority_index = direct_assignment_call(
        main_body, "release_sha", "verify_authority")
    fresh_direct = direct_expr_calls(main_body, "namespaces_fresh")
    fresh = calls(main, "namespaces_fresh")
    if len(fresh) != 2 or len(fresh_direct) != 2:
        raise CheckFailure("main freshness cardinality")
    flock_direct = direct_expr_calls(main_body, "fcntl.flock")
    flock = calls(main, "fcntl.flock")
    if len(flock) != 2 or len(flock_direct) != 2:
        raise CheckFailure("main lock cardinality")
    lock_texts = [ast.dump(call) for call in flock]
    if (sum("queue_handle" in row and "LOCK_EX" in row for row in lock_texts) != 1
            or sum("local_handle" in row and "LOCK_EX" in row and "LOCK_NB" in row
                   for row in lock_texts) != 1):
        raise CheckFailure("global/local lock semantics")
    main_collision_direct = direct_expr_calls(main_body, "collision_gate")
    main_collision = calls(main, "collision_gate")
    runtime_collision = calls(runtime, "collision_gate")
    if (len(main_collision) != 1 or len(main_collision_direct) != 1
            or len(runtime_collision) != 1):
        raise CheckFailure("collision gate cardinality")
    if len(calls(runtime, "CHECK.validate_sources")) != 1:
        raise CheckFailure("runtime source validation")
    runtime_subprocess = calls(runtime, "subprocess.run")
    if len(runtime_subprocess) != 1:
        raise CheckFailure("runtime subprocess cardinality")
    if not (calls(runtime, "CHECK.validate_sources")[0].lineno
            < runtime_collision[0].lineno < runtime_subprocess[0].lineno):
        raise CheckFailure("runtime source/collision/subprocess order")

    attempt_direct = direct_expr_calls(main_body, "ATTEMPT.mkdir")
    attempt_mkdir = [call for call in calls(main, "ATTEMPT.mkdir")]
    if len(attempt_mkdir) != 1 or len(attempt_direct) != 1:
        raise CheckFailure("attempt mkdir cardinality")
    direct_run_calls = direct_expr_calls(main_body, "run")
    run_calls = calls(main, "run")
    if len(run_calls) != 2 or len(direct_run_calls) != 2:
        raise CheckFailure("main tool call cardinality")
    direct_run_values = [row[0] for row in direct_run_calls]
    compile_calls = [call for call in direct_run_values if call.args
                     and isinstance(call.args[0], ast.Call)
                     and call_name(call.args[0].func) == "compile_command"]
    sim_calls = [call for call in direct_run_values if call.args
                 and isinstance(call.args[0], (ast.List, ast.Tuple))]
    if len(compile_calls) != 1 or len(sim_calls) != 1:
        raise CheckFailure("compile/sim call identity")
    sim_args = [literal(item) for item in sim_calls[0].args[0].elts]
    if sim_args != ["./simv", "-lca", "+M979_CASE=0"]:
        raise CheckFailure("sim exact case command")
    parser_call, parser_index = direct_assignment_call(
        main_body, "result", "CHECK.validate_diagnostic_log")
    parser_calls = calls(main, "CHECK.validate_diagnostic_log")
    if len(parser_calls) != 1 or parser_calls[0] is not parser_call.value:
        raise CheckFailure("diagnostic parser cardinality")
    publish_calls = calls(main, "publish_no_replace")
    result_publish = [call for call in publish_calls
                      if len(call.args) == 2
                      and isinstance(call.args[0], ast.Name)
                      and call.args[0].id == "STAGE"
                      and isinstance(call.args[1], ast.Name)
                      and call.args[1].id == "RESULT"]
    if len(result_publish) != 1:
        raise CheckFailure("result publication cardinality")
    direct_result_publish = [row for row in direct_expr_calls(main_body, "publish_no_replace")
                             if row[0] is result_publish[0]]
    if len(direct_result_publish) != 1:
        raise CheckFailure("direct result publication")

    receipt_writes = []
    for call in calls(main, "write_json"):
        if call.args and slash_string(call.args[0], "STAGE", "receipt.json"):
            receipt_writes.append(call)
    if len(receipt_writes) != 1 or len(receipt_writes[0].args) != 2:
        raise CheckFailure("receipt write cardinality")
    direct_receipt_writes = [row for row in direct_expr_calls(main_body, "write_json")
                             if row[0] is receipt_writes[0]]
    if len(direct_receipt_writes) != 1:
        raise CheckFailure("direct receipt write")
    receipt = receipt_writes[0].args[1]
    if not isinstance(receipt, ast.Dict):
        raise CheckFailure("receipt dictionary")
    pairs = {literal(key): value for key, value in zip(receipt.keys, receipt.values)}
    claim = pairs.get("claim_boundary")
    if not (isinstance(claim, ast.Attribute) and isinstance(claim.value, ast.Name)
            and claim.value.id == "CHECK" and claim.attr == "CLAIMS"):
        raise CheckFailure("receipt exact claim boundary")

    compile_index = [index for call, index in direct_run_calls if call is compile_calls[0]][0]
    sim_index = [index for call, index in direct_run_calls if call is sim_calls[0]][0]
    ordered = [authority_index, fresh_direct[0][1], flock_direct[0][1],
               flock_direct[1][1], main_collision_direct[0][1], fresh_direct[1][1],
               attempt_direct[0][1], compile_index, sim_index, parser_index,
               direct_receipt_writes[0][1], direct_result_publish[0][1]]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise CheckFailure("main direct authority/freshness/lock/tool/publication order")
    for node in main_body[:direct_result_publish[0][1]]:
        if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            raise CheckFailure("main direct terminal before publication")

    required = (
        "M1867_EXPECTED_M1868_REVIEW_SHA256",
        "M1867_EXPECTED_M1869_RELEASE_SHA256",
        "M1867_DIAGNOSTIC_ATTEMPT_CONSUMED",
        "M1867_DIAGNOSTIC_LOCALIZATION_COMPLETE_DO_NOT_CITE_AS_PRODUCTION",
        "M1867_DIAGNOSTIC_FAILED_OR_INCOMPLETE_DO_NOT_RETRY",
    )
    for token in required:
        if text.count(token) != 1:
            raise CheckFailure("runner exact token cardinality " + token)
    for token in ("+M979_UCLI_SAIF", '"-ucli"', "/opt/synopsys/prime",
                  "reuse_prior_simv", "M1856_EXPECTED_", "M1858", "M1864"):
        if token in text:
            raise CheckFailure("runner forbidden token " + token)


def task_body(text, name):
    match = re.search(r"task automatic " + re.escape(name)
                      + r"\b(.*?)endtask", text, flags=re.DOTALL)
    if match is None:
        raise CheckFailure("TB task absent " + name)
    return match.group(1)


def check_tb(text):
    if text.count("module " + TOP) != 1:
        raise CheckFailure("TB top identity")
    if text.count("(value === 1'b0) || (value === 1'b1)") != 1:
        raise CheckFailure("TB case-equality binary test")
    if text.count("$finish;") != 4:
        raise CheckFailure("TB stop cardinality")
    expected = {
        "stop_protocol": ("name=protocol_error value=%b", "edge_name, core.protocol_error);"),
        "stop_numeric": ("name=numeric_overflow value=%b", "edge_name, core.numeric_overflow);"),
        "stop_stale": ("name=stale_response_seen value=%b", "edge_name, core.stale_response_seen);"),
        "stop_endpoint": ("name=endpoint_fault[%0d] value=%b", "edge_name, bank, endpoint_fault[bank]);"),
    }
    for name, tokens in expected.items():
        body = task_body(text, name)
        if body.count("$finish;") != 1:
            raise CheckFailure("TB stop per branch " + name)
        if (len(re.findall(r"\bbegin\b", body)) != 1
                or len(re.findall(r"\bend\b", body)) != 1
                or re.search(r"\b(if|case|casex|casez|for|foreach|while|repeat|forever|fork|disable|return)\b",
                             body) is not None):
            raise CheckFailure("TB unconditional stop-task structure " + name)
        display_index = body.find("$display(")
        if display_index < 0 or re.fullmatch(
                r"\$display\(.*?\);\s*\$finish;\s*end\s*",
                body[display_index:], flags=re.DOTALL) is None:
            raise CheckFailure("TB direct display-finish terminal sequence " + name)
        for token in tokens:
            if body.count(token) != 1:
                raise CheckFailure("TB token/value binding " + name)
    localize = task_body(text, "print_and_localize")
    branches = (
        "if (!is_binary(core.protocol_error))\n                stop_protocol(edge_name);",
        "if (!is_binary(core.numeric_overflow))\n                stop_numeric(edge_name);",
        "if (!is_binary(core.stale_response_seen))\n                stop_stale(edge_name);",
        "if (!is_binary(endpoint_fault[bank]))\n                    stop_endpoint(edge_name, bank);",
    )
    for branch in branches:
        if localize.count(branch) != 1:
            raise CheckFailure("TB first-X/Z branch " + branch.splitlines()[0])
    if re.search(r"if\s*\(\s*!is_binary\(mapped_", text):
        raise CheckFailure("internal mapped tap decides localization")
    if text.count('print_and_localize("posedge")') != 1:
        raise CheckFailure("posedge monitor")
    if text.count('print_and_localize("negedge")') != 1:
        raise CheckFailure("negedge monitor")
    for token in ("M979_UCLI_SAIF", "$assertoff", "force ", "release "):
        if token in text:
            raise CheckFailure("TB forbidden token " + token)


def check_filelist(text):
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    if len(rows) != 6 or rows[0] != "+define+M1831_AXIS_K8":
        raise CheckFailure("filelist identity/row count")
    expected = (
        "tcbn28hpcplusbwp35p140.v",
        "m1334_c2_production_activity_reset_safe_memory_model.sv",
        "tb_m1831_c2_fresh_mapped_gate_case_core.sv",
        "tb_m1867_c2_k8_case0_mapped_fault_xz_diagnostic.sv",
    )
    for suffix in expected:
        if sum(row.endswith(suffix) for row in rows) != 1:
            raise CheckFailure("filelist exact member " + suffix)
    for token in ("M1831_AXIS_K1X8", "SVA_RUNTIME_ENABLED",
                  "m1831_c2_registered_public_fault_production_assertions.sv",
                  "tb_m1831_c2_fresh_mapped_production_energy.sv", ".ucli"):
        if token in text:
            raise CheckFailure("filelist forbidden token " + token)


def check_contract(text, texts):
    value = strict_json_text(text)
    if (value.get("schema") !=
            "m1867_m1863_m1862_c2_k8_case0_mapped_fault_xz_diagnostic_source_contract_r1_v1"
            or value.get("status") !=
            "SOURCE_ONLY_M1867_C2_K8_CASE0_MAPPED_FAULT_XZ_DIAGNOSTIC_SUCCESSOR__NO_EDA_NO_LICENSE_NO_ATTEMPT"):
        raise CheckFailure("contract identity/status")
    if value.get("authorization_now") != {
            "license_queries": 0, "attempts_created": 0, "vcs_compiles": 0,
            "simv_runs": 0, "ucli_runs": 0, "saif_files": 0,
            "ptpx_runs": 0, "all_other_eda_runs": 0,
            "results_created": 0, "releases_created": 0}:
        raise CheckFailure("contract authoring authorization")
    if value.get("future_execution_budget") != {
            "vcs_compiles_exact": 1, "simv_runs_exact": 1,
            "case": 0, "axis": "K8", "ucli_runs": 0, "saif_files": 0,
            "ptpx_runs": 0, "all_other_eda_runs": 0,
            "automatic_retry": False, "reuse_m1845_or_m1856_simv": False}:
        raise CheckFailure("contract future budget")
    if value.get("claim_boundary") != CLAIMS:
        raise CheckFailure("contract claim boundary")
    if value.get("failed_predecessor_review") != {
            "path": "reviews/m1863_m1862_c2_k8_case0_mapped_fault_xz_diagnostic_source_hammer_r1_20260902",
            "review_sha256": M1863_REVIEW_SHA256,
            "manifest_sha256": M1863_MANIFEST_SHA256,
            "outer_seal_file_sha256": M1863_OUTER_SHA256,
            "p0": 0, "p1": 1, "p2": 0,
            "m1864_release_authorized": False}:
        raise CheckFailure("contract failed predecessor")
    identity = value.get("exact_diagnostic_identity")
    expected_identity = {
        "axis": "K8", "arch_mode": 0, "m979_case": 0, "top": TOP,
        "mapped_netlist": MAPPED_REL,
        "mapped_netlist_sha256": MAPPED_SHA256,
        "filelist": FILELIST_REL, "filelist_sha256": FILELIST_SHA256,
        "m979_stimulus": "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv",
        "m979_stimulus_sha256": "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
        "m1334_reset_safe_memory_model_sha256": "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
        "m1831_gate_case_adapter_sha256": "0c96db6f2d79fb8c716f99240405f27b65a690293bb39f7d9a7a92886f61e642",
    }
    if identity != expected_identity:
        raise CheckFailure("contract exact diagnostic identity")
    source_files = value.get("source_files")
    if type(source_files) is not dict or len(source_files) != 5:
        raise CheckFailure("contract source inventory")
    for name in ("runner", "tb", "filelist", "checker", "test"):
        rel = PATHS[name].relative_to(HW).as_posix()
        if source_files.get(rel) != hashlib.sha256(texts[name].encode()).hexdigest():
            raise CheckFailure("contract source hash " + name)
    future = value.get("future_authority", {})
    if (future.get("source_review") !=
            "reviews/m1868_m1867_c2_k8_case0_mapped_fault_xz_diagnostic_source_hammer_r1_20260902"
            or future.get("launch_release") !=
            "contracts/m1869_m1868_m1867_c2_k8_case0_mapped_fault_xz_diagnostic_launch_release_r1_20260902.json"
            or future.get("source_review_and_release_present_now") is not False):
        raise CheckFailure("contract future authority")
    if value.get("docs359_sha256") != DOCS359_SHA256:
        raise CheckFailure("docs359 identity")


def validate_diagnostic_log(path):
    text = Path(path).read_text(errors="strict")
    matches = re.findall(
        r"M1867_FIRST_NONBINARY time_ps=(\d+) edge=(posedge|negedge) "
        r"name=(protocol_error|numeric_overflow|stale_response_seen|endpoint_fault\[[0-7]\]) value=([xz])",
        text, flags=re.IGNORECASE)
    if len(matches) != 1:
        raise CheckFailure("diagnostic first-nonbinary token cardinality/value")
    if text.count("M1867_SAMPLE") < 1 or text.count("M1867_AUX") < 1:
        raise CheckFailure("diagnostic samples absent")
    return {"time_ps": int(matches[0][0]), "edge": matches[0][1],
            "name": matches[0][2], "value": matches[0][3].lower(),
            "diagnostic_only": True}


def check(overrides=None):
    texts = source_map()
    if overrides:
        texts.update(overrides)
    check_runner(texts["runner"])
    check_tb(texts["tb"])
    check_filelist(texts["filelist"])
    check_contract(texts["contract"], texts)
    if sha(MAPPED) != MAPPED_SHA256 or sha(FILELIST) != FILELIST_SHA256:
        raise CheckFailure("mapped/filelist live identity")
    return {"status": "PASS_M1867_DIAGNOSTIC_SUCCESSOR_SOURCE_STATIC",
            "eda_or_license_run": False, "launch_authorized": False,
            "future_vcs_compiles": 1, "future_simv_runs": 1,
            "future_ucli_saif_ptpx": 0, "paper_claim": False}


def validate_sources():
    return check()


if __name__ == "__main__":
    try:
        print(json.dumps(check(), sort_keys=True))
    except Exception as error:
        print("FAIL_M1867_DIAGNOSTIC_SUCCESSOR_SOURCE_STATIC: " + str(error), file=sys.stderr)
        raise
