#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind bounded/static hammer for M1095. Never runs production replay."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HW = Path(__file__).resolve().parents[2]
LAUNCHER = HW / "system_simulator/scripts/run_m1095_m1094r2_c1_zero_work_full_replay_zero_arg.py"
ENGINE = HW / "system_simulator/scripts/execute_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot.py"
CONTRACT = HW / "contracts/m1095_m1095a_m1094r2_c1_zero_arg_launch_wrapper_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1095_m1094r2_c1_zero_arg_launch_wrapper_source_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
OUT = Path(__file__).with_name("mechanical_checks.json")

LAUNCHER_SHA = "74576584bcf3140a17d935f7f2bce2fb7fe6a373e8e4b2b0666f5e797e0a5f3b"
CONTRACT_OUTER_SHA = "806d93871cf3a5edac956b657f0a219c9aebc5f8acf79939f274e4f101133d1f"
AUTHOR_OUTER_SHA = "00b1d7a1dcc70225a8df49d030d879ede427999756966e6d53f2dd9c21eeeca9"
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_flat(directory: Path, outer_file_sha: str) -> bool:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    if (not directory.is_dir() or directory.is_symlink() or
            sha(outer) != outer_file_sha or
            outer.read_text().split() != [sha(manifest), "SHA256SUMS"]):
        return False
    seen = set()
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        member = directory / relative
        if (relative in seen or not member.is_file() or member.is_symlink() or
                sha(member) != digest):
            return False
        seen.add(relative)
    return True


def verify_double(path: Path, expected_outer_file_sha: str) -> bool:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    return (
        path.is_file() and not path.is_symlink() and
        side.is_file() and not side.is_symlink() and
        outer.is_file() and not outer.is_symlink() and
        outer.read_text().split() == [sha(side), side.name] and
        side.read_text().split() == [sha(path), path.name] and
        sha(outer) == expected_outer_file_sha
    )


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise RuntimeError("missing function " + name)


def called_name(call: ast.Call) -> str:
    target = call.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        pieces = [target.attr]
        value = target.value
        while isinstance(value, ast.Attribute):
            pieces.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            pieces.append(value.id)
        return ".".join(reversed(pieces))
    return ""


launcher_text = LAUNCHER.read_text(encoding="utf-8")
engine_text = ENGINE.read_text(encoding="utf-8")
launcher_tree = ast.parse(launcher_text)
engine_tree = ast.parse(engine_text)
main = function(launcher_tree, "main")
execute_full = function(engine_tree, "execute_full")

main_calls = sorted(
    [(node.lineno, called_name(node)) for node in ast.walk(main) if isinstance(node, ast.Call)],
    key=lambda item: item[0],
)
execute_calls = sorted(
    [(node.lineno, called_name(node)) for node in ast.walk(execute_full) if isinstance(node, ast.Call)],
    key=lambda item: item[0],
)

def lines_for(calls, suffix):
    return [line for line, name in calls if name.endswith(suffix)]

attempt_lines = lines_for(main_calls, "consume_attempt_atomically")
full_lines = lines_for(main_calls, "M1094.execute_full")
publish_lines = lines_for(main_calls, "M1094.publish_result")
lock_lines = lines_for(main_calls, "acquire_lock")
preflight_lines = lines_for(execute_calls, "canonical_work_domain_preflight")
iterator_lines = lines_for(execute_calls, "iter_canonical_full_replay_results")

environment_authority_reads = []
for node in ast.walk(launcher_tree):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr in {"environ", "getenv"}:
        environment_authority_reads.append({"line": node.lineno, "name": "os." + node.attr})

result = {
    "schema": "m1098_m1095_c1_zero_arg_launch_hammer_checks_v1",
    "receipt_blind": True,
    "production_preflight_called": False,
    "production_iterator_called": False,
    "attempt_consumed": False,
    "full_replay_executed": False,
    "identity": {},
    "static": {},
    "bounded": {},
    "synthetic": {},
    "attacks": {},
}

result["identity"] = {
    "launcher_sha256": sha(LAUNCHER),
    "launcher_matches_expected": sha(LAUNCHER) == LAUNCHER_SHA,
    "contract_double_seal": verify_double(CONTRACT, CONTRACT_OUTER_SHA),
    "contract_outer_seal_file_sha256": sha(Path(str(CONTRACT) + ".sha256.seal.sha256")),
    "author_receipt_recursive_seal": verify_flat(AUTHOR, AUTHOR_OUTER_SHA),
    "author_receipt_outer_seal_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
    "python_sha256": sha(PYTHON),
    "python_matches_expected": sha(PYTHON) == PYTHON_SHA,
    "python_direct_symlink": PYTHON.is_symlink(),
    "docs359_sha256": sha(DOCS359),
    "docs359_unchanged": sha(DOCS359) == DOCS359_SHA,
}

result["static"] = {
    "zero_argument_gate": 'len(sys.argv) == 1' in launcher_text,
    "isolated_python_gate": "sys.flags.isolated == 1" in launcher_text,
    "no_user_site_gate": "sys.flags.no_user_site == 1" in launcher_text,
    "exact_python_path_sha_version": all(token in launcher_text for token in ("PYTHON_SHA", "(3, 10, 18)", "Path(sys.executable).resolve() == PYTHON")),
    "environment_authority_reads": environment_authority_reads,
    "hardcoded_authority_function": "def hardcoded_result_authority" in launcher_text,
    "caller_authority_or_metric_cli": "argparse" in launcher_text,
    "population_constants": {"tasks": 812160, "designs": 3, "values": 2436480},
    "population_literals_present": all(token in launcher_text for token in ("TASKS = 812160", 'DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")', "VALUES = 2436480")),
    "lock_is_atomic_mkdir": 'M1094.LOCK.mkdir(mode=0o700)' in launcher_text and "except FileExistsError" in launcher_text,
    "freshness_before_lock": launcher_text.index("validate_process_resource_freshness()") < launcher_text.index("acquire_lock(); locked = True"),
    "freshness_under_lock": launcher_text.index("acquire_lock(); locked = True") < launcher_text.index('phase = "CONSUME_ATTEMPT"'),
    "attempt_before_first_production_payload": len(attempt_lines) == 1 and len(full_lines) == 1 and attempt_lines[0] < full_lines[0],
    "publish_after_full": len(publish_lines) == 1 and full_lines[0] < publish_lines[0],
    "preflight_calls_in_execute_full": len(preflight_lines),
    "iterator_calls_in_execute_full": len(iterator_lines),
    "preflight_before_iterator": len(preflight_lines) == 1 and len(iterator_lines) == 1 and preflight_lines[0] < iterator_lines[0],
    "iterator_yield_count_checked": "M1094 full iterator yielded more than once" in engine_text,
    "no_replace_publish": "rename_noreplace(work, RESULT)" in engine_text,
    "automatic_retry_false": '"automatic_retry": False' in launcher_text,
    "caught_post_attempt_quarantine": "if attempt_consumed:" in launcher_text and "M1094.quarantine_work(work, quarantine, 1, phase)" in launcher_text,
    "result_claims_false": all(token in launcher_text for token in ('"speedup_admitted": False', '"paper_citable": False')),
    "resource_minimums": {"mem_available_kib": 4194304, "commit_headroom_kib": 8388608},
    "process_scan": 'Path("/proc")' in launcher_text and "competing_launcher_pids" in launcher_text,
}

# Read-only exact-authority validation under the production interpreter and -I.
probe = """
import importlib.util,json,os,sys
sys.dont_write_bytecode=True
p=%r
s=importlib.util.spec_from_file_location('m1098_probe',p)
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
print(json.dumps({'authority':m.validate_hardcoded_authorities(),'resource':m.validate_process_resource_freshness(),'map':m.hardcoded_result_authority()},sort_keys=True))
""" % str(LAUNCHER)
poison_env = os.environ.copy()
poison_env.update({
    "M1095_EXPECTED_SPEEDUP": "999999",
    "M1095_EXPECTED_AUTHORITY": "attacker",
    "M1098_OUTER": "0" * 64,
    "PYTHONPATH": "/tmp/attacker",
})
probe_run = subprocess.run(
    [str(PYTHON), "-I", "-B", "-c", probe],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    env=poison_env, timeout=60, check=False,
)
result["bounded"]["isolated_readonly_authority_probe"] = {
    "returncode": probe_run.returncode,
    "stdout": probe_run.stdout[-6000:],
    "attempt_absent_after": not (HW / "results/.m1094_m1086_c1_zero_work_exact_1rw_full_replay_attempt_consumed").exists(),
}

# Invalid argv must reject before attempt.
argv_run = subprocess.run(
    [str(PYTHON), "-I", "-B", str(LAUNCHER), "--forged-authority"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    timeout=60, check=False,
)
result["bounded"]["argv_attack"] = {
    "returncode": argv_run.returncode,
    "rejected": argv_run.returncode != 0 and "M1095 accepts zero arguments" in argv_run.stdout,
    "attempt_absent_after": not (HW / "results/.m1094_m1086_c1_zero_work_exact_1rw_full_replay_attempt_consumed").exists(),
}

# Missing -I must reject before attempt.
nonisolated_run = subprocess.run(
    [str(PYTHON), "-B", str(LAUNCHER)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    timeout=60, check=False,
)
result["bounded"]["nonisolated_python_attack"] = {
    "returncode": nonisolated_run.returncode,
    "rejected": nonisolated_run.returncode != 0 and "M1095 isolated Python identity drift" in nonisolated_run.stdout,
    "attempt_absent_after": not (HW / "results/.m1094_m1086_c1_zero_work_exact_1rw_full_replay_attempt_consumed").exists(),
}

# Import exact launcher without main; use only synthetic validators.
spec = importlib.util.spec_from_file_location("m1098_synthetic", LAUNCHER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
engine = module.M1094

def rejects(callable_):
    try:
        callable_()
    except Exception:
        return True
    return False

population_attacks = {}
for name, mutate in {
    "tasks": lambda x: x.__setitem__("tasks", 812159),
    "tasks_bool": lambda x: x.__setitem__("tasks", True),
    "values": lambda x: x.__setitem__("values_checked", 2436479),
    "values_nonfinite": lambda x: x.__setitem__("values_checked", float("nan")),
    "designs": lambda x: x.__setitem__("designs", ["candidate"]),
    "caller_work": lambda x: x.__setitem__("caller_supplied_work", True),
}.items():
    value = engine.synthetic_preflight()
    mutate(value)
    population_attacks[name] = rejects(lambda value=value: engine.validate_preflight(value))

claim_attacks = {}
for name, mutate in {
    "speedup_true": lambda x: x["claim_boundary"].__setitem__("speedup_admitted", True),
    "rtl_cycles_true": lambda x: x["claim_boundary"].__setitem__("rtl_cycles", True),
    "hammer_false": lambda x: x["claim_boundary"].__setitem__("independent_result_hammer_required", False),
    "capacity_bool": lambda x: x["capacity"].__setitem__("derived_total_bytes", True),
}.items():
    value = engine.synthetic_raw_result()
    mutate(value)
    claim_attacks[name] = rejects(lambda value=value: engine.normalize_raw(value))

result["synthetic"] = {
    "population_attacks": population_attacks,
    "claim_boundary_attacks": claim_attacks,
    "production_preflight_called": False,
    "production_iterator_called": False,
}

# Namespace/lock attacks are bounded in a temporary directory using the exact
# attempt function. No production namespace is patched or opened.
with tempfile.TemporaryDirectory(prefix="m1098_hammer_") as temp_name:
    temp = Path(temp_name)
    old_attempt, old_result, old_lock = engine.ATTEMPT, engine.RESULT, engine.LOCK
    try:
        engine.ATTEMPT = temp / old_attempt.name
        engine.RESULT = temp / old_result.name
        engine.LOCK = temp / old_lock.name
        authority = module.hardcoded_result_authority()
        first = module.consume_attempt_atomically(authority)
        duplicate_rejected = rejects(lambda: module.consume_attempt_atomically(authority))
        engine.LOCK.mkdir()
        stale_lock_rejected = rejects(module.acquire_lock)
        result["attacks"]["temporary_namespace"] = {
            "first_attempt_sealed": first["receipt"]["status"] == "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS" and (engine.ATTEMPT / engine.SEAL_DIR).is_dir(),
            "duplicate_attempt_rejected": duplicate_rejected,
            "stale_lock_rejected": stale_lock_rejected,
        }
    finally:
        engine.ATTEMPT, engine.RESULT, engine.LOCK = old_attempt, old_result, old_lock

result["attacks"].update({
    "existing_result_rejected_by_pre_and_post_lock_freshness": launcher_text.count("not M1094.RESULT.exists()") >= 3,
    "existing_attempt_rejected_by_pre_and_post_lock_freshness": launcher_text.count("not M1094.ATTEMPT.exists()") >= 3,
    "existing_work_rejected_by_pre_and_post_lock_freshness": launcher_text.count("M1094.WORK_PREFIX") >= 3,
    "existing_quarantine_rejected_by_pre_and_post_lock_freshness": launcher_text.count("M1094.FAILURE_PREFIX") >= 3,
    "partial_seal_recovery_present": "PARTIAL_SEAL_ATTEMPTS" in engine_text,
    "publish_collision_no_replace": "rename_noreplace(work, RESULT)" in engine_text,
    "no_retry_after_attempt": '"maximum_attempts": 1' in launcher_text and '"automatic_retry": False' in launcher_text,
})

all_required = [
    result["identity"]["launcher_matches_expected"],
    result["identity"]["contract_double_seal"],
    result["identity"]["author_receipt_recursive_seal"],
    result["identity"]["python_matches_expected"],
    result["identity"]["docs359_unchanged"],
    result["static"]["zero_argument_gate"],
    result["static"]["isolated_python_gate"],
    not result["static"]["environment_authority_reads"],
    not result["static"]["caller_authority_or_metric_cli"],
    result["static"]["attempt_before_first_production_payload"],
    result["static"]["preflight_before_iterator"],
    result["static"]["preflight_calls_in_execute_full"] == 1,
    result["static"]["iterator_calls_in_execute_full"] == 1,
    probe_run.returncode == 0,
    result["bounded"]["argv_attack"]["rejected"],
    result["bounded"]["nonisolated_python_attack"]["rejected"],
    all(population_attacks.values()),
    all(claim_attacks.values()),
    all(result["attacks"]["temporary_namespace"].values()),
    not (HW / "results/.m1094_m1086_c1_zero_work_exact_1rw_full_replay_attempt_consumed").exists(),
]
result["verdict"] = "PASS" if all(all_required) else "STOP"
OUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
print(result["verdict"] + "_M1098_M1095_LAUNCH_HAMMER")
