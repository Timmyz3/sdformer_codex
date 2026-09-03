#!/usr/bin/env python3
"""Fail-closed, source-only checker for the inert M1887 B4 VCS campaign."""
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
SPEC = importlib.util.spec_from_file_location("m1880_checker_for_m1887", str(M1880_CHECKER))
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
M1882_RUNNER = HW / "dc_handoff/scripts/run_m1882_m1880_c2_tsbg_b4_directed_vcs_one_shot.py"
M1882_CHECKER = HW / "system_simulator/scripts/check_m1882_m1880_c2_tsbg_b4_campaign_source.py"
M1882_TEST = HW / "system_simulator/tests/test_m1882_m1880_c2_tsbg_b4_campaign_source.py"
M1882_CONTRACT = HW / "contracts/m1882_m1881_m1880_c2_tsbg_b4_campaign_source_contract_r1_20260902.json"
M1882_AUTHOR = HW / "reviews/m1882_m1881_m1880_c2_tsbg_b4_campaign_source_author_receipt_r1_20260902"
M1884 = HW / "reviews/m1884_m1882_m1880_c2_tsbg_b4_campaign_source_hammer_r1_20260902"

RUNNER = HW / "dc_handoff/scripts/run_m1887_m1880_c2_tsbg_b4_directed_vcs_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1887_m1880_c2_tsbg_b4_campaign_source.py"
CONTRACT = HW / "contracts/m1887_m1884_m1882_m1880_c2_tsbg_b4_campaign_successor_source_contract_r1_20260902.json"

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
    "m1882_runner_sha256": "29cdc882cc419b7dca525751d8e243a77316cb8f1ac79088004c4a8ab142bea1",
    "m1882_checker_sha256": "70b8c6d8882b469b27ab2abb17a908f13ea7a4e7c59a501f5d502320ae14ef84",
    "m1882_tests_sha256": "ea6a2e72f438c6bcd950d8b8ca9589c816f1aeecc5b91f69bf407b6863dece1f",
    "m1882_contract_sha256": "06c74e771b5dc51780386fa619527e7e4a9fb25db270f58c4aa02dcbc61bcb3c",
    "m1882_contract_sidecar_file_sha256": "90b8528b228b2dc4145c6433a64517dbf530da0ae6c1c5f556c2b4dfc2a23ac4",
    "m1882_contract_outer_file_sha256": "4b523254917539d1ef7e894a9cc561c5a4e4abc8a2c2b373ba995a9ec75baf40",
    "m1882_author_receipt_sha256": "3e9119d5ec843c4ab6e37659c5c330e6aa6bab7c6220d3ffe4be2f5c6d465fa4",
    "m1882_author_manifest_sha256": "84c07c2af66a345a7395a72a3f05a83ccab689d826e142ffe234c4676cbe0de3",
    "m1882_author_outer_file_sha256": "bda165759b2b10c2b19b4c8a6d815ffea81faf48926f13fc53ad08a06fea6058",
    "m1884_review_sha256": "72d360d6610730cf07ebc1d8bb1ff62fdff5f9ace24514f299df9829dc96dc71",
    "m1884_manifest_sha256": "63a2f943dd97d0d31d41eccae2e1faa47e83e0c9ddfda82abf7915aa29a33469",
    "m1884_outer_file_sha256": "df9777ceb536d19d34e213043a1f61ea3700ce78612247637b8cfa951f2df201",
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


def statement_call_name(statement):
    value = None
    if isinstance(statement, ast.Expr):
        value = statement.value
    elif isinstance(statement, ast.Assign):
        value = statement.value
    if isinstance(value, ast.Call):
        return dotted_name(value.func)
    return ""


def function_node(tree, name):
    hits = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name]
    need(len(hits) == 1, "runner function " + name)
    return hits[0]


def string_value(node):
    if isinstance(node, ast.Str):
        return node.s
    constant = getattr(ast, "Constant", None)
    if constant is not None and isinstance(node, constant):
        return node.value if isinstance(node.value, str) else None
    return None


def validate_governance_helper_reachability(tree):
    no_return_helpers = (
        "verify_authority", "namespaces_fresh", "collision_gate",
        "resource_gate", "seal_dir", "publish_no_replace",
        "attempt_terminal_gate")
    for name in no_return_helpers:
        node = function_node(tree, name)
        need(not any(isinstance(item, ast.Return) for item in ast.walk(node)),
             "governance helper early return " + name)

    external = function_node(tree, "run_external_once")
    returns = [item for item in ast.walk(external) if isinstance(item, ast.Return)]
    need(len(returns) == 1 and external.body[-1] is returns[0],
         "run_external_once must have one final return")
    calls = [item for item in ast.walk(external) if isinstance(item, ast.Call)]
    subprocess_calls = [item for item in calls
                         if dotted_name(item.func) == "subprocess.run"]
    need(len(subprocess_calls) == 1, "one live subprocess.run implementation")
    need(subprocess_calls[0].lineno < returns[0].lineno,
         "external subprocess must precede final return")
    attempts = []
    for item in ast.walk(external):
        if not isinstance(item, ast.Compare) or not isinstance(item.left, ast.Call):
            continue
        if dotted_name(item.left.func) != "state.get" or not item.left.args:
            continue
        if string_value(item.left.args[0]) == "attempt":
            attempts.append(item)
    need(len(attempts) >= 1 and min(item.lineno for item in attempts)
         < subprocess_calls[0].lineno,
         "external helper must gate durable attempt before subprocess")


def validate_external_call_closure(tree):
    forbidden = {"subprocess.Popen", "subprocess.call", "subprocess.check_call",
                 "subprocess.check_output", "subprocess.CompletedProcess",
                 "os.system", "os.popen", "os.execl", "os.execle",
                 "os.execlp", "os.execlpe", "os.execv", "os.execve",
                 "os.execvp", "os.execvpe", "pty.spawn",
                 "asyncio.create_subprocess_exec", "asyncio.create_subprocess_shell"}
    all_calls = [item for item in ast.walk(tree) if isinstance(item, ast.Call)]
    need(not any(dotted_name(item.func) in forbidden for item in all_calls),
         "forbidden/faked external invocation")
    subprocess_calls = [item for item in all_calls
                         if dotted_name(item.func) == "subprocess.run"]
    need(len(subprocess_calls) == 1, "external subprocess call-site cardinality")
    owner = function_node(tree, "run_external_once")
    need(subprocess_calls[0] in list(ast.walk(owner)),
         "subprocess.run must be owned by run_external_once")
    need(not any(keyword.arg == "shell" for keyword in subprocess_calls[0].keywords),
         "shell execution forbidden")

    owner_text = ast.dump(owner)
    for key in ("license", "vcs_compile", "simv"):
        need(key in owner_text, "external counter mapping " + key)
    increments = [item for item in ast.walk(owner)
                  if isinstance(item, ast.AugAssign)
                  and isinstance(item.op, ast.Add)
                  and isinstance(item.target, ast.Subscript)
                  and isinstance(item.target.value, ast.Name)
                  and item.target.value.id == "state"
                  and getattr(item.value, "n", getattr(item.value, "value", None)) == 1]
    need(len(increments) == 1, "one accounted dynamic counter increment")

    main = function_node(tree, "main")
    calls = [item for item in ast.walk(main)
             if isinstance(item, ast.Call)
             and dotted_name(item.func) == "run_external_once"]
    calls.sort(key=lambda item: item.lineno)
    need(len(calls) == 3, "exact three accounted external calls")
    kinds = [string_value(item.args[1]) if len(item.args) > 1 else None
             for item in calls]
    need(kinds == ["license", "vcs_compile", "simv"],
         "external call kinds/order")
    need(all(len(item.args) == 6 and not item.keywords for item in calls),
         "external call signature")

    license_command = calls[0].args[2]
    compile_command = calls[1].args[2]
    simv_command = calls[2].args[2]
    need(isinstance(license_command, (ast.List, ast.Tuple))
         and len(license_command.elts) == 5
         and string_value(license_command.elts[1]) == "lmstat"
         and string_value(license_command.elts[2]) == "-a"
         and string_value(license_command.elts[3]) == "-c",
         "sole lmstat command shape")
    need(isinstance(compile_command, (ast.List, ast.Tuple))
         and sum(string_value(item) == "-assert" for item in compile_command.elts) == 1
         and sum(string_value(item) == "svaext" for item in compile_command.elts) == 1,
         "sole VCS compile keeps SVA")
    need(isinstance(simv_command, (ast.List, ast.Tuple))
         and len(simv_command.elts) == 1,
         "sole simv command shape")

    attempt_calls = [item for item in ast.walk(main)
                     if isinstance(item, ast.Call)
                     and dotted_name(item.func) == "ATTEMPT.mkdir"]
    need(len(attempt_calls) == 1 and attempt_calls[0].lineno < calls[0].lineno,
         "durable attempt must precede first external call")
    attempt_true = [item for item in ast.walk(main)
                    if isinstance(item, ast.Assign)
                    and 'attempt' in ast.dump(item)
                    and (isinstance(item.value, ast.NameConstant)
                         and item.value.value is True
                         or getattr(item.value, "value", None) is True)]
    need(len(attempt_true) == 1
         and attempt_calls[0].lineno < attempt_true[0].lineno < calls[0].lineno,
         "attempt state transition order")


def validate_runner_semantics(text):
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise CheckFailure("runner syntax " + str(error))
    need("if False:" not in text and "if 0:" not in text,
         "unreachable governance wrapper")
    validate_governance_helper_reachability(tree)
    validate_external_call_closure(tree)
    for forbidden in ("os.replace(", ".rename(", "shutil.move(", "LOCK_SH",
                      "automatic_retry\": True", "reuse_prior_simv\": True"):
        need(forbidden not in text, "forbidden runner primitive " + forbidden)

    required = (
        "M1887_EXPECTED_RUNNER_SHA256",
        "M1887_EXPECTED_SOURCE_CONTRACT_SHA256",
        "M1887_EXPECTED_M1888_REVIEW_SHA256",
        "M1887_EXPECTED_M1888_MANIFEST_SHA256",
        "M1887_EXPECTED_M1888_OUTER_FILE_SHA256",
        "M1887_EXPECTED_M1889_RELEASE_SHA256",
        "M1887_EXPECTED_M1889_SIDECAR_SHA256",
        "M1887_EXPECTED_M1889_OUTER_FILE_SHA256",
        "M1887_EXPECTED_M1890_REVIEW_SHA256",
        "M1887_EXPECTED_M1890_MANIFEST_SHA256",
        "M1887_EXPECTED_M1890_OUTER_FILE_SHA256",
        "PASS_M1888_M1887_C2_TSBG_B4_CAMPAIGN_SOURCE_HAMMER__",
        "AUTHORIZE_RELEASE_SOURCE_ONLY",
        "m1889_m1888_m1887_m1880_c2_tsbg_b4_directed_vcs_",
        "launch_release_r1_v1",
        "AUTHORIZE_ONE_FRESH_M1887_M1880_C2_TSBG_B4_DIRECTED_VCS_CAMPAIGN",
        "PASS_M1890_M1889_C2_TSBG_B4_LAUNCH_RELEASE_AUDIT__",
        "AUTHORIZE_ONE_M1887_ATTEMPT",
        "release.get(\"identity\") != expected_release_identity()",
        "release.get(\"prelaunch_claim_boundary\") != CLAIMS",
        "release.get(\"measurement_boundary\") != MEASUREMENT_BOUNDARY",
        "release.get(\"fresh_execution_budget\") != dict(",
        "release_audit.get(\"audited_identity\") != {",
        "result_hammer_still_required\": True",
        "results/.m1887_m1880_c2_tsbg_b4_directed_vcs_attempt_consumed",
        "results/m1887_m1880_c2_tsbg_b4_directed_vcs_r1_20260902",
        "failed_or_incomplete.quarantine",
        "private_build.unsealed_do_not_cite",
        "/tmp/m1887_m1880_c2_tsbg_b4_directed_vcs.lock",
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
        ("canonical result path", "RESULT = HW / \"results/m1887_m1880_c2_tsbg_b4_directed_vcs_r1_20260902\""),
        ("private work path", "WORK = HW / (\"results/.m1887_m1880_c2_tsbg_b4_directed_vcs_work.\" + str(os.getpid()))"),
        ("success stage path", "STAGE = HW / (\"results/.m1887_m1880_c2_tsbg_b4_directed_vcs_stage.\" + str(os.getpid()))"),
        ("failure stage path", "FAIL_STAGE = HW / (\"results/.m1887_m1880_c2_tsbg_b4_directed_vcs_failure_stage.\" + str(os.getpid()))"),
        ("self runner pin", "exact(RUNNER, authority_pin(\"M1887_EXPECTED_RUNNER_SHA256\"))"),
        ("source contract pin", "exact(CONTRACT, authority_pin(\"M1887_EXPECTED_SOURCE_CONTRACT_SHA256\"))"),
        ("shared exclusive lock", "fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)"),
        ("local exclusive lock", "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)"),
        ("attempt transition", "ATTEMPT.mkdir()\n        state[\"attempt\"] = True"),
        ("success publication", "seal_dir(STAGE)\n        publish_no_replace(STAGE, RESULT)\n        state[\"complete\"] = True"),
        ("failure only after attempt", "if state[\"attempt\"] and not state[\"complete\"]:"),
        ("existing failure blocks overwrite", "if os.path.lexists(str(FAILURE)):\n                raise"),
        ("failure publication", "seal_dir(FAIL_STAGE)\n            publish_no_replace(FAIL_STAGE, FAILURE)\n            attempt_terminal_gate(state)"),
        ("no-replace syscall", "renameat2(-100, os.fsencode(source), -100,\n                 os.fsencode(destination), 1)"),
        ("attempt terminal xor", "if success == failure:\n        raise Failure(\"attempt must terminate in exactly one sealed namespace\")"),
        ("dynamic counter mapping", "counters = {\"license\": \"license_queries\", \"vcs_compile\": \"vcs_compiles\",\n                \"simv\": \"simv_runs\"}"),
        ("one dynamic external count", "state[counter] += 1"),
        ("external attempt gate", "if state.get(\"attempt\") is not True:\n        raise Failure(\"external call before durable attempt\")"),
        ("compile assertion", "str(VCS), \"-full64\", \"-sverilog\", \"-assert\", \"svaext\""),
        ("namespace residue gate", "if os.path.lexists(str(path)):\n            raise Failure(\"namespace residue \" + str(path))"),
        ("prior simv glob gate", "if any((HW / \"results\").glob(pattern)):\n            raise Failure(\"prior private build or simv namespace \" + pattern)"),
        ("collision inventory", "blocked = {\"vcs\", \"vcs1\", \"vlogan\", \"simv\", \"dc_shell\", \"dc_shell-t\",\n               \"pt_shell\", \"fm_shell\", \"icc2_shell\", \"common_shell_exec\",\n               \"common_shell_exe\"}"),
        ("memory resource threshold", "if values.get(\"MemAvailable\", 0) < 16 * 1024 * 1024:"),
        ("future release schema", "if release.get(\"schema\") != (\n            \"m1889_m1888_m1887_m1880_c2_tsbg_b4_directed_vcs_\"\n            \"launch_release_r1_v1\")"),
    )
    for label, snippet in snippets:
        need_code_once(text, snippet, label)

    main = function_node(tree, "main")
    tries = [node for node in main.body if isinstance(node, ast.Try)]
    need(len(tries) == 1, "main try")
    direct = [statement_call_name(node) for node in tries[0].body]
    direct = [name for name in direct if name]
    required_sequence = [
        "verify_authority", "CHECK.validate_sources", "namespaces_fresh",
        "fcntl.flock", "fcntl.flock", "collision_gate", "resource_gate",
        "namespaces_fresh", "ATTEMPT.mkdir", "WORK.mkdir", "STAGE.mkdir",
        "run_external_once", "run_external_once", "run_external_once",
        "shutil.copy2", "shutil.copy2",
        "shutil.copy2", "write_json", "seal_dir", "publish_no_replace",
        "attempt_terminal_gate"]
    cursor = 0
    for name in direct:
        if cursor < len(required_sequence) and name == required_sequence[cursor]:
            cursor += 1
    need(cursor == len(required_sequence), "main direct-call reachability/order")
    main_returns = [node for node in ast.walk(main) if isinstance(node, ast.Return)]
    need(len(main_returns) == 1, "main must have exactly one success return")
    terminal_calls = [node for node in ast.walk(main)
                      if isinstance(node, ast.Call)
                      and dotted_name(node.func) == "attempt_terminal_gate"]
    need(len(terminal_calls) == 2,
         "success and failure must each reach terminal gate")
    need(main_returns[0].lineno > min(node.lineno for node in terminal_calls),
         "main success return before terminal gate")


MUTATION_SPECS = (
    ("call_verify_authority_unreachable", "        verify_authority()\n        CHECK.validate_sources()", "        if False: verify_authority()\n        CHECK.validate_sources()"),
    ("call_validate_sources_unreachable", "CHECK.validate_sources()\n        namespaces_fresh()", "if False: CHECK.validate_sources()\n        namespaces_fresh()"),
    ("call_first_namespaces_unreachable", "namespaces_fresh()\n        fcntl.flock(queue_handle", "if False: namespaces_fresh()\n        fcntl.flock(queue_handle"),
    ("call_collision_unreachable", "collision_gate()\n        resource_gate()", "if False: collision_gate()\n        resource_gate()"),
    ("call_resource_unreachable", "resource_gate()\n        namespaces_fresh()", "if False: resource_gate()\n        namespaces_fresh()"),
    ("call_second_namespaces_unreachable", "resource_gate()\n        namespaces_fresh()\n\n        state", "resource_gate()\n        if False: namespaces_fresh()\n\n        state"),
    ("call_license_unreachable", "        license_check = run_external_once(\n            state, \"license\"", "        if False: license_check = run_external_once(\n            state, \"license\""),
    ("call_compile_unreachable", "        run_external_once(state, \"vcs_compile\", [\n            str(VCS)", "        if False: run_external_once(state, \"vcs_compile\", [\n            str(VCS)"),
    ("call_simv_unreachable", "        run_external_once(state, \"simv\", [str(simv)]", "        if False: run_external_once(state, \"simv\", [str(simv)]"),
    ("call_success_terminal_unreachable", "        attempt_terminal_gate(state)\n        return 0", "        if False: attempt_terminal_gate(state)\n        return 0"),
    ("path_attempt_changed", "results/.m1887_m1880_c2_tsbg_b4_directed_vcs_attempt_consumed", "results/.wrong_attempt"),
    ("path_result_changed", "results/m1887_m1880_c2_tsbg_b4_directed_vcs_r1_20260902\"", "results/wrong_result\""),
    ("path_failure_changed", "failed_or_incomplete.quarantine", "wrong_failure.quarantine"),
    ("path_private_changed", "private_build.unsealed_do_not_cite", "private_build.citable"),
    ("path_work_prefix_changed", "results/.m1887_m1880_c2_tsbg_b4_directed_vcs_work.\" + str(os.getpid())", "results/.wrong_work.\" + str(os.getpid())"),
    ("path_stage_prefix_changed", "results/.m1887_m1880_c2_tsbg_b4_directed_vcs_stage.\" + str(os.getpid())", "results/.wrong_stage.\" + str(os.getpid())"),
    ("path_failure_stage_changed", "results/.m1887_m1880_c2_tsbg_b4_directed_vcs_failure_stage.\" + str(os.getpid())", "results/.wrong_failure_stage.\" + str(os.getpid())"),
    ("lock_local_path_changed", "/tmp/m1887_m1880_c2_tsbg_b4_directed_vcs.lock", "/tmp/wrong.lock"),
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
    ("future_runner_pin_changed", "M1887_EXPECTED_RUNNER_SHA256", "M1887_UNPINNED_RUNNER"),
    ("future_contract_pin_changed", "exact(CONTRACT, authority_pin(\"M1887_EXPECTED_SOURCE_CONTRACT_SHA256\"))", "exact(RUNNER, authority_pin(\"M1887_EXPECTED_SOURCE_CONTRACT_SHA256\"))"),
    ("future_m1888_review_pin_changed", "M1887_EXPECTED_M1888_REVIEW_SHA256", "M1887_UNPINNED_M1888_REVIEW"),
    ("future_m1888_manifest_pin_changed", "M1887_EXPECTED_M1888_MANIFEST_SHA256", "M1887_UNPINNED_M1888_MANIFEST"),
    ("future_m1888_outer_pin_changed", "M1887_EXPECTED_M1888_OUTER_FILE_SHA256", "M1887_UNPINNED_M1888_OUTER"),
    ("future_m1889_release_pin_changed", "M1887_EXPECTED_M1889_RELEASE_SHA256", "M1887_UNPINNED_M1889_RELEASE"),
    ("future_m1889_sidecar_pin_changed", "M1887_EXPECTED_M1889_SIDECAR_SHA256", "M1887_UNPINNED_M1889_SIDECAR"),
    ("future_m1889_outer_pin_changed", "M1887_EXPECTED_M1889_OUTER_FILE_SHA256", "M1887_UNPINNED_M1889_OUTER"),
    ("future_m1890_review_pin_changed", "M1887_EXPECTED_M1890_REVIEW_SHA256", "M1887_UNPINNED_M1890_REVIEW"),
    ("future_m1890_manifest_pin_changed", "M1887_EXPECTED_M1890_MANIFEST_SHA256", "M1887_UNPINNED_M1890_MANIFEST"),
    ("future_m1890_outer_pin_changed", "M1887_EXPECTED_M1890_OUTER_FILE_SHA256", "M1887_UNPINNED_M1890_OUTER"),
    ("future_m1888_status_changed", "PASS_M1888_M1887_C2_TSBG_B4_CAMPAIGN_SOURCE_HAMMER__", "WRONG_M1888_STATUS__"),
    ("future_m1889_schema_changed", "            \"m1889_m1888_m1887_m1880_c2_tsbg_b4_directed_vcs_\"", "            \"wrong_release_schema_\""),
    ("future_m1889_status_changed", "AUTHORIZE_ONE_FRESH_M1887_M1880_C2_TSBG_B4_DIRECTED_VCS_CAMPAIGN", "WRONG_M1889_STATUS"),
    ("future_m1889_identity_bypassed", "release.get(\"identity\") != expected_release_identity()", "False"),
    ("future_m1889_claims_bypassed", "release.get(\"prelaunch_claim_boundary\") != CLAIMS", "False"),
    ("future_m1889_budget_bypassed", "release.get(\"fresh_execution_budget\") != dict(", "False and dict("),
    ("future_m1890_status_changed", "PASS_M1890_M1889_C2_TSBG_B4_LAUNCH_RELEASE_AUDIT__", "WRONG_M1890_STATUS__"),
    ("future_m1890_identity_bypassed", "release_audit.get(\"audited_identity\") != {", "False and {") ,
    ("count_dynamic_zero", "state[counter] += 1", "state[counter] += 0"),
    ("count_license_mapping_wrong", "\"license\": \"license_queries\"", "\"license\": \"simv_runs\""),
    ("count_compile_mapping_wrong", "\"vcs_compile\": \"vcs_compiles\"", "\"vcs_compile\": \"license_queries\""),
    ("count_simv_mapping_wrong", "\"simv\": \"simv_runs\"", "\"simv\": \"vcs_compiles\""),
    ("compile_sva_disabled", "\"-assert\", \"svaext\"", "\"-assert\", \"no_sva\""),
    ("success_result_hammer_false", "\"result_hammer_required\": True", "\"result_hammer_required\": False"),
    ("failure_retry_enabled", "                \"automatic_retry\": False,\n            })", "                \"automatic_retry\": True,\n            })"),
    ("m1884_return_verify_authority", "def verify_authority():\n    exact(", "def verify_authority():\n    return\n    exact("),
    ("m1884_return_namespaces_fresh", "def namespaces_fresh():\n    fixed =", "def namespaces_fresh():\n    return\n    fixed ="),
    ("m1884_return_collision_gate", "def collision_gate():\n    blocked =", "def collision_gate():\n    return\n    blocked ="),
    ("m1884_return_resource_gate", "def resource_gate():\n    values =", "def resource_gate():\n    return\n    values ="),
    ("m1884_return_external_helper", "def run_external_once(state, kind, command, cwd, timeout, output):\n    CHECK.validate_sources()", "def run_external_once(state, kind, command, cwd, timeout, output):\n    return None\n    CHECK.validate_sources()"),
    ("m1884_return_seal_dir", "def seal_dir(root):\n    rows =", "def seal_dir(root):\n    return\n    rows ="),
    ("m1884_return_publish_no_replace", "def publish_no_replace(source, destination):\n    libc =", "def publish_no_replace(source, destination):\n    return\n    libc ="),
    ("return_attempt_terminal_gate", "def attempt_terminal_gate(state):\n    if state.get", "def attempt_terminal_gate(state):\n    return\n    if state.get"),
    ("m1884_extra_uncounted_license", "        if state[\"license_queries\"] != COUNTS[\"license_queries\"]:", "        subprocess.run([str(LMUTIL), \"lmstat\", \"-a\", \"-c\", LICENSE_SERVER])\n        if state[\"license_queries\"] != COUNTS[\"license_queries\"]:"),
    ("m1884_fake_license_success", "        license_check = run_external_once(", "        license_check = subprocess.CompletedProcess("),
    ("m1884_extra_uncounted_simv", "        if state[\"simv_runs\"] != COUNTS[\"simv_runs\"]:", "        subprocess.run([str(simv)])\n        if state[\"simv_runs\"] != COUNTS[\"simv_runs\"]:"),
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
         "m1887_m1884_m1882_m1880_c2_tsbg_b4_campaign_successor_source_contract_r1_v1",
         "contract schema")
    need(value.get("status") ==
         "SOURCE_ONLY_M1887_C2_TSBG_B4_ONE_SHOT_CAMPAIGN__M1888_REVIEW_M1889_RELEASE_M1890_AUDIT_REQUIRED__NO_EDA",
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
        "campaign_source_review": "M1888",
        "launch_release": "M1889",
        "launch_release_audit": "M1890",
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
            "m1882_runner_sha256": M1882_RUNNER,
            "m1882_checker_sha256": M1882_CHECKER,
            "m1882_tests_sha256": M1882_TEST,
            "m1882_contract_sha256": M1882_CONTRACT,
            "m1882_contract_sidecar_file_sha256": Path(str(M1882_CONTRACT) + ".sha256"),
            "m1882_contract_outer_file_sha256": Path(str(M1882_CONTRACT) + ".sha256.seal.sha256"),
            "m1882_author_receipt_sha256": M1882_AUTHOR / "author_receipt.json",
            "m1882_author_manifest_sha256": M1882_AUTHOR / "SHA256SUMS",
            "m1882_author_outer_file_sha256": M1882_AUTHOR / "SHA256SUMS.seal.sha256",
            "m1884_review_sha256": M1884 / "review.json",
            "m1884_manifest_sha256": M1884 / "SHA256SUMS",
            "m1884_outer_file_sha256": M1884 / "SHA256SUMS.seal.sha256",
            "docs359_sha256": DOC359,
        }[key]
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "upstream identity " + key)
    for root in (M1880_AUTHOR, M1881, M1866, M1871, M1875,
                 M1882_AUTHOR, M1884):
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
    m1884 = strict_json(M1884 / "review.json")
    need(m1884.get("status") ==
         "FAIL_CLOSED_M1884_M1882_C2_TSBG_B4_CAMPAIGN_SOURCE_HAMMER__P1_3__NO_M1885_NO_M1886_NO_VCS_NO_EDA",
         "M1884 fail-closed status")
    need(m1884.get("severity_counts") == {"p0": 0, "p1": 3, "p2": 0},
         "M1884 severity")
    need(m1884.get("authorization") == {
        "author_additive_successor_campaign_source": True,
        "automatic_retry": False,
        "create_attempt": False,
        "create_m1885_release": False,
        "create_m1886_release_audit": False,
        "create_result": False,
        "modify_m1882_author_source_in_place": False,
        "paper_admission": False,
        "query_license": False,
        "run_dc": False,
        "run_ptpx": False,
        "run_simv": False,
        "run_vcs": False}, "M1884 additive-successor-only authority")
    need(Path(str(M1882_CONTRACT) + ".sha256").read_text(encoding="ascii").split()
         == [sha(M1882_CONTRACT), M1882_CONTRACT.name], "M1882 contract sidecar")
    need(Path(str(M1882_CONTRACT) + ".sha256.seal.sha256").read_text(
         encoding="ascii").split() ==
         [sha(Path(str(M1882_CONTRACT) + ".sha256")),
          M1882_CONTRACT.name + ".sha256"], "M1882 contract outer seal")
    M1880.validate_sources()
    validate_semantics(source_texts())
    contract = validate_contract()
    need(len(MUTATION_SPECS) >= 60, "mutation inventory below 60")
    return {
        "status": "PASS_M1887_C2_TSBG_B4_CAMPAIGN_SOURCE_STATIC_NO_EDA",
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
    need(args.self_check, "M1887 checker requires --self-check")
    print(json.dumps(validate_sources(), indent=2, sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
