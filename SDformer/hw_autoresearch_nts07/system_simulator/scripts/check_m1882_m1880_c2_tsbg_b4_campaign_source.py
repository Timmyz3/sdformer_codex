#!/usr/bin/env python3
"""Fail-closed, source-only checker for the inert M1882 B4 VCS campaign."""
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
M1880_CHECKER = HW / "system_simulator/scripts/check_m1880_c2_tsbg_b4_source.py"
SPEC = importlib.util.spec_from_file_location("m1880_checker_for_m1882", str(M1880_CHECKER))
M1880 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M1880)

RTL = M1880.RTL
SVA = M1880.SVA
TB = M1880.TB
FILELIST = M1880.FILELIST
M803 = M1880.M803
DOC359 = M1880.DOC359
M1880_CONTRACT = M1880.CONTRACT
M1880_AUTHOR = HW / "reviews/m1880_m1875_m1874_c2_tsbg_b4_source_author_receipt_r1_20260902"
M1881 = HW / "reviews/m1881_m1880_c2_tsbg_b4_source_hammer_r1_20260902"
M1866 = M1880.M1866
M1871 = M1880.M1871
M1875 = M1880.M1875

RUNNER = HW / "dc_handoff/scripts/run_m1882_m1880_c2_tsbg_b4_directed_vcs_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1882_m1880_c2_tsbg_b4_campaign_source.py"
CONTRACT = HW / "contracts/m1882_m1881_m1880_c2_tsbg_b4_campaign_source_contract_r1_20260902.json"

CLAIMS = dict((key, False) for key in (
    "source_review_pass", "vcs", "simv", "dc", "ptpx", "area", "energy",
    "same_area", "same_resource_result", "rtl_executed", "paper_admitted",
    "component_speedup", "system_speedup", "headline"))

UPSTREAM_IDENTITY = {
    "m803_adapter_sha256": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "m1880_rtl_sha256": "8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05",
    "m1880_sva_sha256": "e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2",
    "m1880_tb_sha256": "07f638b3a6a2ae99c3d24fcf96088ed84bfa61ab3c34bd626f65965fa1fed2d5",
    "m1880_filelist_sha256": "300702cdfec07ba83d1b85c5464002e411ea838846d623d3a09b1045391e71d2",
    "m1880_checker_sha256": "496c20e7daecccaa5df24519aaa45ee052e82aa193bcc2b6bfd27faa4982bf4c",
    "m1880_tests_sha256": "a8d702a40423796b8b8e0b45a6036fb6a368aadfc73f03ae89e2b24deebf20b7",
    "m1880_contract_sha256": "cf5ab7edb90c1477fb81773a6613957ab389601a34bc517f348c4b2087079f3d",
    "m1880_contract_sidecar_file_sha256": "a357725681727be872678f9c53c0b740da1e97fbf6c0cfc7112b314eb8ad9602",
    "m1880_contract_outer_file_sha256": "b5b3e0b9ceeda14c019f780f5aee5b50271cd162a7032335b445c1c78b27f630",
    "m1880_author_receipt_sha256": "c400ac05a209372150a140735968cf1a5c9618e2a1ef57c29fc22f1d9777d47a",
    "m1880_author_manifest_sha256": "b2f5cb717535376f67f0b66fb1e7dd7f4f7b52a31d136d8f80f2fdfa820ad273",
    "m1880_author_outer_file_sha256": "bb9ab6d39304c3f63140fdc70cb5d5d2922c6931f21a5463279eecd278d512fb",
    "m1881_review_sha256": "62d44419bbe240fe4d2874c87d82ceb67a923b47e1f21e9e5844c6c9f94a1281",
    "m1881_manifest_sha256": "28bb0efd64def451d49fa1749ddef36bfca2da6a6d622e7b567c7aa59e870a1c",
    "m1881_outer_file_sha256": "74fcb1b67b1e65ae6ec32ffe7888e6413e76a64fcbefb58b528f5c8b2fb16e67",
    "m1866_review_sha256": "6560b3660d247440691d31dea7cccd0ca0294cd203c7f2d957a183116eb81830",
    "m1866_manifest_sha256": "12e466e667cf133a4a4953199817180d24054b4aa39ec1ef4a277e602c18b897",
    "m1866_outer_file_sha256": "da826a3797d7586508f9f95dfa06430a47a59c9f3e328320453e83777e587fb7",
    "m1871_review_sha256": "fb7d0e0d322111bcfaabf74bae0d640c50fe00ea9d7327ae0e3ac883065ad5a8",
    "m1871_manifest_sha256": "fbbf43b4614ca9fb90494d9087b13bbf3ca751b34c8c8d6b35c5fd655be4577a",
    "m1871_outer_file_sha256": "decd92229b18483577abd867f4ad4028b4d231f7da47642e3e5db3f488e4e8c4",
    "m1875_review_sha256": "92f95021d9a127a3149e820e8c86110ecec8ee1c8f21673f6d043cc6d9239bee",
    "m1875_manifest_sha256": "0c39e1d299bef6c3302e943fe15ac4889d636b8ce945388debb514ddc5be704f",
    "m1875_outer_file_sha256": "7fb52ce7c5f9391d82603711cf90cdf7f882c29caecfabc13a48a2b84b0e673d",
    "docs359_sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

SOURCE_PATHS = (M803, RTL, SVA, TB, FILELIST, M1880_CHECKER, RUNNER, CHECKER, TEST)
SOURCE_SHA256 = {}


class CheckFailure(RuntimeError):
    pass


def need(value, message):
    if not value:
        raise CheckFailure(message)


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
            need(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           CheckFailure("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_sealed_directory(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "sealed directory absent")
    need(outer.read_text(encoding="ascii").split() == [sha(manifest), "SHA256SUMS"],
         "outer seal " + str(root))
    listed = set()
    for row in manifest.read_text(encoding="ascii").splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
             "manifest syntax")
        rel = Path(fields[1].lstrip("*")); name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest")
        need((root / rel).is_file() and not (root / rel).is_symlink()
             and sha(root / rel) == fields[0], "manifest member " + name)
        listed.add(name)
    if (root / "review.json").exists():
        need("review.json" in listed, "review not sealed")


def compact(text):
    return re.sub(r"\s+", "", text)


def need_code_once(text, snippet, label):
    count = compact(text).count(compact(snippet))
    need(count == 1, label + " cardinality " + str(count))


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


def function_node(tree, name):
    hits = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name]
    need(len(hits) == 1, "runner function " + name)
    return hits[0]


def validate_runner_semantics(text):
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise CheckFailure("runner syntax " + str(error))
    need("if False:" not in text and "if 0:" not in text,
         "unreachable governance wrapper")
    for forbidden in ("os.replace(", ".rename(", "shutil.move(", "LOCK_SH",
                      "automatic_retry\": True", "reuse_prior_simv\": True"):
        need(forbidden not in text, "forbidden runner primitive " + forbidden)

    required = (
        "M1882_EXPECTED_RUNNER_SHA256",
        "M1882_EXPECTED_SOURCE_CONTRACT_SHA256",
        "M1882_EXPECTED_M1884_REVIEW_SHA256",
        "M1882_EXPECTED_M1884_MANIFEST_SHA256",
        "M1882_EXPECTED_M1884_OUTER_FILE_SHA256",
        "M1882_EXPECTED_M1885_RELEASE_SHA256",
        "M1882_EXPECTED_M1885_SIDECAR_SHA256",
        "M1882_EXPECTED_M1885_OUTER_FILE_SHA256",
        "M1882_EXPECTED_M1886_REVIEW_SHA256",
        "M1882_EXPECTED_M1886_MANIFEST_SHA256",
        "M1882_EXPECTED_M1886_OUTER_FILE_SHA256",
        "PASS_M1884_M1882_C2_TSBG_B4_CAMPAIGN_SOURCE_HAMMER__",
        "AUTHORIZE_RELEASE_SOURCE_ONLY",
        "m1885_m1884_m1882_m1880_c2_tsbg_b4_directed_vcs_",
        "launch_release_r1_v1",
        "AUTHORIZE_ONE_FRESH_M1882_M1880_C2_TSBG_B4_DIRECTED_VCS_CAMPAIGN",
        "PASS_M1886_M1885_C2_TSBG_B4_LAUNCH_RELEASE_AUDIT__",
        "AUTHORIZE_ONE_M1882_ATTEMPT",
        "release.get(\"identity\") != expected_release_identity()",
        "release.get(\"prelaunch_claim_boundary\") != CLAIMS",
        "release.get(\"measurement_boundary\") != MEASUREMENT_BOUNDARY",
        "release.get(\"fresh_execution_budget\") != dict(",
        "release_audit.get(\"audited_identity\") != {",
        "result_hammer_still_required\": True",
        "results/.m1882_m1880_c2_tsbg_b4_directed_vcs_attempt_consumed",
        "results/m1882_m1880_c2_tsbg_b4_directed_vcs_r1_20260902",
        "failed_or_incomplete.quarantine",
        "private_build.unsealed_do_not_cite",
        "/tmp/m1882_m1880_c2_tsbg_b4_directed_vcs.lock",
        "/tmp/date_dual_synopsys_same_uid_eda_queue.lock",
        "prior private build or simv namespace",
        "same-UID EDA collision",
        "MemAvailable below 16 GiB",
        "SwapFree below 8 GiB",
        "commit headroom below 16 GiB",
        "result disk free below 12 GiB",
        "-assert", "svaext",
        "PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED",
        "RAW_PASS_AWAIT_DIFFERENT_AUTHOR_RESULT_HAMMER",
        "FAILED_OR_INCOMPLETE_DO_NOT_CITE_NO_RETRY",
        "result_hammer_required\": True",
    )
    for token in required:
        need(token in text, "runner omits " + token)

    snippets = (
        ("canonical result path", "RESULT = HW / \"results/m1882_m1880_c2_tsbg_b4_directed_vcs_r1_20260902\""),
        ("private work path", "WORK = HW / (\"results/.m1882_m1880_c2_tsbg_b4_directed_vcs_work.\" + str(os.getpid()))"),
        ("success stage path", "STAGE = HW / (\"results/.m1882_m1880_c2_tsbg_b4_directed_vcs_stage.\" + str(os.getpid()))"),
        ("failure stage path", "FAIL_STAGE = HW / (\"results/.m1882_m1880_c2_tsbg_b4_directed_vcs_failure_stage.\" + str(os.getpid()))"),
        ("self runner pin", "exact(RUNNER, authority_pin(\"M1882_EXPECTED_RUNNER_SHA256\"))"),
        ("source contract pin", "exact(CONTRACT, authority_pin(\"M1882_EXPECTED_SOURCE_CONTRACT_SHA256\"))"),
        ("shared exclusive lock", "fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)"),
        ("local exclusive lock", "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)"),
        ("attempt transition", "ATTEMPT.mkdir()\n        state[\"attempt\"] = True"),
        ("success publication", "seal_dir(STAGE)\n        publish_no_replace(STAGE, RESULT)\n        state[\"complete\"] = True"),
        ("failure only after attempt", "if state[\"attempt\"] and not state[\"complete\"]:"),
        ("existing failure blocks overwrite", "if os.path.lexists(str(FAILURE)):\n                raise"),
        ("failure publication", "seal_dir(FAIL_STAGE)\n            publish_no_replace(FAIL_STAGE, FAILURE)\n            attempt_terminal_gate(state)"),
        ("no-replace syscall", "renameat2(-100, os.fsencode(source), -100,\n                 os.fsencode(destination), 1)"),
        ("attempt terminal xor", "if success == failure:\n        raise Failure(\"attempt must terminate in exactly one sealed namespace\")"),
        ("one license query", "state[\"license_queries\"] += 1"),
        ("one compile", "state[\"vcs_compiles\"] += 1"),
        ("one simv", "state[\"simv_runs\"] += 1"),
        ("compile assertion", "str(VCS), \"-full64\", \"-sverilog\", \"-assert\", \"svaext\""),
        ("namespace residue gate", "if os.path.lexists(str(path)):\n            raise Failure(\"namespace residue \" + str(path))"),
        ("prior simv glob gate", "if any((HW / \"results\").glob(pattern)):\n            raise Failure(\"prior private build or simv namespace \" + pattern)"),
        ("collision inventory", "blocked = {\"vcs\", \"vcs1\", \"vlogan\", \"simv\", \"dc_shell\", \"dc_shell-t\",\n               \"pt_shell\", \"fm_shell\", \"icc2_shell\", \"common_shell_exec\",\n               \"common_shell_exe\"}"),
        ("memory resource threshold", "if values.get(\"MemAvailable\", 0) < 16 * 1024 * 1024:"),
        ("future release schema", "if release.get(\"schema\") != (\n            \"m1885_m1884_m1882_m1880_c2_tsbg_b4_directed_vcs_\"\n            \"launch_release_r1_v1\")"),
    )
    for label, snippet in snippets:
        need_code_once(text, snippet, label)

    main = function_node(tree, "main")
    tries = [node for node in main.body if isinstance(node, ast.Try)]
    need(len(tries) == 1, "main try")
    direct = [direct_call_name(node) for node in tries[0].body]
    direct = [name for name in direct if name]
    required_sequence = [
        "verify_authority", "CHECK.validate_sources", "namespaces_fresh",
        "fcntl.flock", "fcntl.flock", "collision_gate", "resource_gate",
        "namespaces_fresh", "ATTEMPT.mkdir", "WORK.mkdir", "STAGE.mkdir",
        "run_tool", "run_tool", "shutil.copy2", "shutil.copy2",
        "shutil.copy2", "write_json", "seal_dir", "publish_no_replace",
        "attempt_terminal_gate"]
    cursor = 0
    for name in direct:
        if cursor < len(required_sequence) and name == required_sequence[cursor]:
            cursor += 1
    need(cursor == len(required_sequence), "main direct-call reachability/order")


MUTATION_SPECS = (
    ("call_verify_authority_unreachable", "        verify_authority()\n        CHECK.validate_sources()", "        if False: verify_authority()\n        CHECK.validate_sources()"),
    ("call_validate_sources_unreachable", "CHECK.validate_sources()\n        namespaces_fresh()", "if False: CHECK.validate_sources()\n        namespaces_fresh()"),
    ("call_first_namespaces_unreachable", "namespaces_fresh()\n        fcntl.flock(queue_handle", "if False: namespaces_fresh()\n        fcntl.flock(queue_handle"),
    ("call_collision_unreachable", "collision_gate()\n        resource_gate()", "if False: collision_gate()\n        resource_gate()"),
    ("call_resource_unreachable", "resource_gate()\n        namespaces_fresh()", "if False: resource_gate()\n        namespaces_fresh()"),
    ("call_second_namespaces_unreachable", "resource_gate()\n        namespaces_fresh()\n\n        state", "resource_gate()\n        if False: namespaces_fresh()\n\n        state"),
    ("call_compile_unreachable", "        run_tool([\n            str(VCS)", "        if False: run_tool([\n            str(VCS)"),
    ("call_simv_unreachable", "        run_tool([str(simv)]", "        if False: run_tool([str(simv)]"),
    ("call_success_terminal_unreachable", "        attempt_terminal_gate(state)\n        return 0", "        if False: attempt_terminal_gate(state)\n        return 0"),
    ("path_attempt_changed", "results/.m1882_m1880_c2_tsbg_b4_directed_vcs_attempt_consumed", "results/.wrong_attempt"),
    ("path_result_changed", "results/m1882_m1880_c2_tsbg_b4_directed_vcs_r1_20260902\"", "results/wrong_result\""),
    ("path_failure_changed", "failed_or_incomplete.quarantine", "wrong_failure.quarantine"),
    ("path_private_changed", "private_build.unsealed_do_not_cite", "private_build.citable"),
    ("path_work_prefix_changed", "results/.m1882_m1880_c2_tsbg_b4_directed_vcs_work.\" + str(os.getpid())", "results/.wrong_work.\" + str(os.getpid())"),
    ("path_stage_prefix_changed", "results/.m1882_m1880_c2_tsbg_b4_directed_vcs_stage.\" + str(os.getpid())", "results/.wrong_stage.\" + str(os.getpid())"),
    ("path_failure_stage_changed", "results/.m1882_m1880_c2_tsbg_b4_directed_vcs_failure_stage.\" + str(os.getpid())", "results/.wrong_failure_stage.\" + str(os.getpid())"),
    ("lock_local_path_changed", "/tmp/m1882_m1880_c2_tsbg_b4_directed_vcs.lock", "/tmp/wrong.lock"),
    ("lock_queue_path_changed", "/tmp/date_dual_synopsys_same_uid_eda_queue.lock", "/tmp/wrong_queue.lock"),
    ("lock_queue_downgraded", "fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)", "fcntl.flock(queue_handle.fileno(), fcntl.LOCK_SH)"),
    ("lock_local_downgraded", "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)", "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)"),
    ("lock_local_wrong_handle", "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)", "fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)"),
    ("path_provenance_parent_weakened", "resolved_parent != (HW / \"results\").resolve(strict=True)", "False"),
    ("namespace_residue_weakened", "if os.path.lexists(str(path)):", "if False and os.path.lexists(str(path)):"),
    ("prior_simv_glob_omitted", "if any((HW / \"results\").glob(pattern)):", "if False and any((HW / \"results\").glob(pattern)):"),
    ("collision_set_emptied", "    blocked = {\"vcs\", \"vcs1\", \"vlogan\", \"simv\", \"dc_shell\", \"dc_shell-t\",", "    blocked = set(); _old = {\"vcs\", \"vcs1\", \"vlogan\", \"simv\", \"dc_shell\", \"dc_shell-t\","),
    ("resource_mem_zeroed", "if values.get(\"MemAvailable\", 0) < 16 * 1024 * 1024:", "if values.get(\"MemAvailable\", 0) < 0:"),
    ("attempt_latch_omitted", "ATTEMPT.mkdir()", "WORK.mkdir()"),
    ("attempt_state_false", "state[\"attempt\"] = True", "state[\"attempt\"] = False"),
    ("attempt_failure_guard_bypassed", "if state[\"attempt\"] and not state[\"complete\"]:", "if False and state[\"attempt\"] and not state[\"complete\"]:"),
    ("attempt_terminal_xor_bypassed", "if success == failure:", "if False and success == failure:"),
    ("attempt_success_complete_early", "publish_no_replace(STAGE, RESULT)\n        state[\"complete\"] = True", "state[\"complete\"] = True\n        publish_no_replace(STAGE, RESULT)"),
    ("attempt_failure_existing_overwrite", "if os.path.lexists(str(FAILURE)):\n                raise", "if False and os.path.lexists(str(FAILURE)):\n                raise"),
    ("publish_success_plain_rename", "publish_no_replace(STAGE, RESULT)", "STAGE.rename(RESULT)"),
    ("publish_failure_plain_rename", "publish_no_replace(FAIL_STAGE, FAILURE)", "FAIL_STAGE.rename(FAILURE)"),
    ("publish_no_replace_flag_zero", "os.fsencode(destination), 1)", "os.fsencode(destination), 0)"),
    ("publish_success_unsealed", "seal_dir(STAGE)\n        publish_no_replace(STAGE, RESULT)", "publish_no_replace(STAGE, RESULT)"),
    ("publish_failure_unsealed", "seal_dir(FAIL_STAGE)\n            publish_no_replace(FAIL_STAGE, FAILURE)", "publish_no_replace(FAIL_STAGE, FAILURE)"),
    ("future_runner_pin_changed", "M1882_EXPECTED_RUNNER_SHA256", "M1882_UNPINNED_RUNNER"),
    ("future_contract_pin_changed", "exact(CONTRACT, authority_pin(\"M1882_EXPECTED_SOURCE_CONTRACT_SHA256\"))", "exact(RUNNER, authority_pin(\"M1882_EXPECTED_SOURCE_CONTRACT_SHA256\"))"),
    ("future_m1884_review_pin_changed", "M1882_EXPECTED_M1884_REVIEW_SHA256", "M1882_UNPINNED_M1884_REVIEW"),
    ("future_m1884_manifest_pin_changed", "M1882_EXPECTED_M1884_MANIFEST_SHA256", "M1882_UNPINNED_M1884_MANIFEST"),
    ("future_m1884_outer_pin_changed", "M1882_EXPECTED_M1884_OUTER_FILE_SHA256", "M1882_UNPINNED_M1884_OUTER"),
    ("future_m1885_release_pin_changed", "M1882_EXPECTED_M1885_RELEASE_SHA256", "M1882_UNPINNED_M1885_RELEASE"),
    ("future_m1885_sidecar_pin_changed", "M1882_EXPECTED_M1885_SIDECAR_SHA256", "M1882_UNPINNED_M1885_SIDECAR"),
    ("future_m1885_outer_pin_changed", "M1882_EXPECTED_M1885_OUTER_FILE_SHA256", "M1882_UNPINNED_M1885_OUTER"),
    ("future_m1886_review_pin_changed", "M1882_EXPECTED_M1886_REVIEW_SHA256", "M1882_UNPINNED_M1886_REVIEW"),
    ("future_m1886_manifest_pin_changed", "M1882_EXPECTED_M1886_MANIFEST_SHA256", "M1882_UNPINNED_M1886_MANIFEST"),
    ("future_m1886_outer_pin_changed", "M1882_EXPECTED_M1886_OUTER_FILE_SHA256", "M1882_UNPINNED_M1886_OUTER"),
    ("future_m1884_status_changed", "PASS_M1884_M1882_C2_TSBG_B4_CAMPAIGN_SOURCE_HAMMER__", "WRONG_M1884_STATUS__"),
    ("future_m1885_schema_changed", "            \"m1885_m1884_m1882_m1880_c2_tsbg_b4_directed_vcs_\"", "            \"wrong_release_schema_\""),
    ("future_m1885_status_changed", "AUTHORIZE_ONE_FRESH_M1882_M1880_C2_TSBG_B4_DIRECTED_VCS_CAMPAIGN", "WRONG_M1885_STATUS"),
    ("future_m1885_identity_bypassed", "release.get(\"identity\") != expected_release_identity()", "False"),
    ("future_m1885_claims_bypassed", "release.get(\"prelaunch_claim_boundary\") != CLAIMS", "False"),
    ("future_m1885_budget_bypassed", "release.get(\"fresh_execution_budget\") != dict(", "False and dict("),
    ("future_m1886_status_changed", "PASS_M1886_M1885_C2_TSBG_B4_LAUNCH_RELEASE_AUDIT__", "WRONG_M1886_STATUS__"),
    ("future_m1886_identity_bypassed", "release_audit.get(\"audited_identity\") != {", "False and {") ,
    ("count_license_zero", "state[\"license_queries\"] += 1", "state[\"license_queries\"] += 0"),
    ("count_compile_zero", "state[\"vcs_compiles\"] += 1", "state[\"vcs_compiles\"] += 0"),
    ("count_simv_zero", "state[\"simv_runs\"] += 1", "state[\"simv_runs\"] += 0"),
    ("compile_sva_disabled", "\"-assert\", \"svaext\"", "\"-assert\", \"no_sva\""),
    ("success_result_hammer_false", "\"result_hammer_required\": True", "\"result_hammer_required\": False"),
    ("failure_retry_enabled", "                \"automatic_retry\": False,\n            })", "                \"automatic_retry\": True,\n            })"),
)


def source_texts():
    return {RTL: RTL.read_text(encoding="utf-8"),
            SVA: SVA.read_text(encoding="utf-8"),
            TB: TB.read_text(encoding="utf-8"),
            RUNNER: RUNNER.read_text(encoding="utf-8")}


def validate_semantics(texts):
    M1880.validate_rtl_text(texts[RTL])
    M1880.validate_sva_text(texts[SVA])
    M1880.validate_tb_text(texts[TB])
    validate_runner_semantics(texts[RUNNER])


def validate_contract():
    value = strict_json(CONTRACT)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text(encoding="ascii").split() == [sha(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text(encoding="ascii").split() == [sha(sidecar), sidecar.name],
         "contract outer")
    need(value.get("schema") ==
         "m1882_m1881_m1880_c2_tsbg_b4_campaign_source_contract_r1_v1",
         "contract schema")
    need(value.get("status") ==
         "SOURCE_ONLY_M1882_C2_TSBG_B4_ONE_SHOT_CAMPAIGN__M1884_REVIEW_M1885_RELEASE_M1886_AUDIT_REQUIRED__NO_EDA",
         "contract status")
    need(value.get("source_sha256") == SOURCE_SHA256, "contract source inventory")
    need(value.get("upstream_identity") == UPSTREAM_IDENTITY, "contract upstream identity")
    need(value.get("claim_boundary") == CLAIMS, "contract claims")
    need(value.get("authorization") == {
        "run_vcs": False, "run_simv": False, "run_dc": False,
        "run_ptpx": False, "query_license": False,
        "create_attempt": False, "create_result": False,
        "create_release": False, "automatic_retry": False},
        "contract authorization")
    need(value.get("future_chain") == {
        "campaign_source_review": "M1884",
        "launch_release": "M1885",
        "launch_release_audit": "M1886",
        "all_three_required_before_attempt": True,
        "one_license_query_one_compile_one_simv": True,
        "result_hammer_required": True,
        "naked_release_forbidden": True}, "contract future chain")
    return value


def validate_sources():
    global SOURCE_SHA256
    SOURCE_SHA256 = dict((str(path.relative_to(ROOT)), sha(path))
                         for path in SOURCE_PATHS)
    for key, digest in UPSTREAM_IDENTITY.items():
        path = {
            "m803_adapter_sha256": M803,
            "m1880_rtl_sha256": RTL,
            "m1880_sva_sha256": SVA,
            "m1880_tb_sha256": TB,
            "m1880_filelist_sha256": FILELIST,
            "m1880_checker_sha256": M1880_CHECKER,
            "m1880_tests_sha256": M1880.TEST,
            "m1880_contract_sha256": M1880_CONTRACT,
            "m1880_contract_sidecar_file_sha256": Path(str(M1880_CONTRACT) + ".sha256"),
            "m1880_contract_outer_file_sha256": Path(str(M1880_CONTRACT) + ".sha256.seal.sha256"),
            "m1880_author_receipt_sha256": M1880_AUTHOR / "author_receipt.json",
            "m1880_author_manifest_sha256": M1880_AUTHOR / "SHA256SUMS",
            "m1880_author_outer_file_sha256": M1880_AUTHOR / "SHA256SUMS.seal.sha256",
            "m1881_review_sha256": M1881 / "review.json",
            "m1881_manifest_sha256": M1881 / "SHA256SUMS",
            "m1881_outer_file_sha256": M1881 / "SHA256SUMS.seal.sha256",
            "m1866_review_sha256": M1866 / "review.json",
            "m1866_manifest_sha256": M1866 / "SHA256SUMS",
            "m1866_outer_file_sha256": M1866 / "SHA256SUMS.seal.sha256",
            "m1871_review_sha256": M1871 / "review.json",
            "m1871_manifest_sha256": M1871 / "SHA256SUMS",
            "m1871_outer_file_sha256": M1871 / "SHA256SUMS.seal.sha256",
            "m1875_review_sha256": M1875 / "review.json",
            "m1875_manifest_sha256": M1875 / "SHA256SUMS",
            "m1875_outer_file_sha256": M1875 / "SHA256SUMS.seal.sha256",
            "docs359_sha256": DOC359,
        }[key]
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "upstream identity " + key)
    for root in (M1880_AUTHOR, M1881, M1866, M1871, M1875):
        verify_sealed_directory(root)
    m1881 = strict_json(M1881 / "review.json")
    need(m1881.get("status") ==
         "PASS_M1881_M1880_C2_TSBG_B4_SOURCE_HAMMER__P0_P1_P2_0_0_0__M1882_CAMPAIGN_SOURCE_ONLY_NEXT__NO_NAKED_RELEASE_NO_EDA",
         "M1881 status")
    need(m1881.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
         "M1881 severity")
    m1866 = strict_json(M1866 / "review.json")
    need(m1866.get("rtl_source_ruling", {}).get("single_selected_bundle") == 4,
         "M1866 B4 selection")
    need(m1866.get("authorization", {}).get("b4_rtl_execution") is False,
         "M1866 no execution")
    need(strict_json(M1871 / "review.json").get("severity_counts") ==
         {"p0": 0, "p1": 1, "p2": 0}, "M1871 authority")
    need(strict_json(M1875 / "review.json").get("severity_counts") ==
         {"p0": 0, "p1": 1, "p2": 0}, "M1875 authority")
    M1880.validate_sources()
    validate_semantics(source_texts())
    contract = validate_contract()
    need(len(MUTATION_SPECS) >= 60, "mutation inventory below 60")
    return {
        "status": "PASS_M1882_C2_TSBG_B4_CAMPAIGN_SOURCE_STATIC_NO_EDA",
        "source_sha256": SOURCE_SHA256,
        "upstream_identity": UPSTREAM_IDENTITY,
        "claim_boundary": CLAIMS,
        "future_chain": contract["future_chain"],
        "author_execution": {"license_queries": 0, "vcs": 0, "simv": 0,
                             "eda": 0, "attempts": 0, "results": 0,
                             "releases": 0},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    need(args.self_check, "M1882 checker requires --self-check")
    print(json.dumps(validate_sources(), indent=2, sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
