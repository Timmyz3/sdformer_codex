#!/usr/bin/env python3
"""Independent M1211 source/release hammer. Strictly local; never invokes network or EDA."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1210_m1208_motion_ep29_unified_capture_secure_remote_one_shot_source.py"
SOURCE_CONTRACT = HW / "contracts/m1210_m1208_motion_ep29_unified_capture_secure_release_source_contract_r1_20260830.json"
TEST = HW / "tests/test_run_m1210_m1208_motion_ep29_unified_capture_secure_remote_one_shot_source.py"
LAUNCH = HW / "contracts/m1210_m1208_motion_ep29_unified_capture_launch_release_r1_20260830.json"
INVENTORY = HW / "contracts/m1210_m1208_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json"
TRANSFER_LIST = HW / "contracts/m1210_m1208_motion_ep29_unified_capture_remote_transfer_files_r1_20260830.txt"
OLD_INVENTORY = HW / "contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json"
AUTHOR = HW / "reviews/m1210_m1208_motion_ep29_unified_capture_release_author_r1_20260830"
M1208_AUTHOR = HW / "reviews/m1208_motion_ep29_unified_capture_symlink_root_successor_author_r1_20260830"
M1209 = HW / "reviews/m1209_m1208_motion_ep29_unified_capture_source_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
LOCAL_ATTEMPT = HW / "results/.m1210_m1208_secure_transfer_and_launch_r1_attempt_consumed"

EXPECTED = {
    SOURCE: "d3ce4b1e7aa1243b266053ed4f26ba452d25505791a9d328b03b5edda6e8432d",
    SOURCE_CONTRACT: "b6bea670083eecbf8a6a73b48996fea6f1ee268d01f7da05738a4f986e361267",
    TEST: "7558fd0af5440317bc5549af6a0b353298bf1eaa10da070215cea2d2d9e2ca00",
    LAUNCH: "5aeeaf9cab836f32e025f0c329ef1fe90caa4ee3acae691514f4793c1d143829",
    INVENTORY: "11483d3c227f28aba8bb8a1dd765db59b8d432383b192f91c967d50f881de0cd",
    TRANSFER_LIST: "2734d71d8555cb131d77a3bea0057634d93264d0fcda38b050416c506f41e220",
    OLD_INVENTORY: "de6ff2b13719580b77674b44f7414a7798cffd3f7cde5e80e88ff3ea8f0d97ae",
    AUTHOR / "author_receipt.json": "06a6b12b1d6869a1bab43b33331649bf2652b0b7445f90c3e63d7fa9a30477c0",
    AUTHOR / "SHA256SUMS": "a7d10fc194fae5efffce5430ac75bc3c57fe4accb6b91ab57d158a396738a8c3",
    AUTHOR / "SHA256SUMS.seal.sha256": "b2a9e6d8dbc90f65435537de8bc4fa3e38ee2e53ffe45bc50f3f658974f0b68f",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks = 0
mutations = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def regular(path: Path, label: str) -> None:
    require(path.exists() and not path.is_symlink(), "regular present " + label)
    require(stat.S_ISREG(path.lstat().st_mode), "regular type " + label)


def strict_json(path: Path) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key " + key)
            out[key] = value
        return out
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(
                           AssertionError("nonfinite JSON " + value)))
    require(isinstance(value, dict), "JSON object " + str(path))
    return value


def safe_rel(text: str) -> Path:
    path = Path(text)
    require(text == path.as_posix() and bool(path.parts) and not path.is_absolute()
            and ".." not in path.parts, "safe relative path " + text)
    return path


def verify_sealed_dir(directory: Path, expected_members: int | None = None) -> dict[str, str]:
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory " + str(directory))
    manifest, outer = directory / "SHA256SUMS", directory / "SHA256SUMS.seal.sha256"
    regular(manifest, "manifest"); regular(outer, "outer")
    parts = outer.read_text(encoding="ascii").split()
    require(parts == [sha(manifest), "SHA256SUMS"], "outer seal exact " + str(directory))
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        pieces = line.split("  ", 1)
        require(len(pieces) == 2 and re.fullmatch(r"[0-9a-f]{64}", pieces[0]) is not None,
                "manifest row")
        name = pieces[1]
        require("/" not in name and name not in rows
                and name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
                "manifest member safe")
        member = directory / name
        regular(member, "sealed member " + name)
        require(sha(member) == pieces[0], "sealed member exact " + name)
        rows[name] = pieces[0]
    actual = {p.name for p in directory.iterdir() if p.is_file() and not p.is_symlink()
              and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "recursive membership exact " + str(directory))
    if expected_members is not None:
        require(len(rows) == expected_members, "sealed member count " + str(directory))
    return rows


def function_text(source: str, name: str) -> tuple[str, ast.FunctionDef]:
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(nodes) == 1, "one function " + name)
    body = ast.get_source_segment(source, nodes[0])
    require(body is not None, "function source " + name)
    return body, nodes[0]


def remote_helper(source: str) -> str:
    tree = ast.parse(source)
    values = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REMOTE_HELPER"
                                                   for t in node.targets):
            values.append(ast.literal_eval(node.value))
    require(len(values) == 1 and isinstance(values[0], str), "one literal REMOTE_HELPER")
    ast.parse(values[0])
    return values[0]


def validate_source(source: str) -> None:
    ast.parse(source)
    require("import shlex" in source, "shlex imported")
    execute, node = function_text(source, "execute_once")
    helper = remote_helper(source)

    # Four transport calls only: mktemp, SCP, exact remote preflight, launch.
    runner_calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name) and call.func.id == "runner"]
    require(len(runner_calls) == 4, "exact four transport runner calls")
    require(execute.count("remote_mktemp = runner(") == 1, "one mktemp call")
    require(execute.count("copied = runner(") == 1, "one SCP call")
    require(execute.count("checked = runner(") == 1, "one remote preflight call")
    require(execute.count("launched = runner(") == 1, "one launch call")
    require(execute.count("shlex.join(") == 3, "all SSH remote commands shell-quoted")
    require("REMOTE_HOST, checked_command])" in execute,
            "remote preflight uses the quoted command string")
    require(not any(isinstance(item, (ast.For, ast.AsyncFor, ast.While)) for item in ast.walk(node)),
            "no execute retry loop")

    absent = execute.index('require(not os.path.lexists(LOCAL_ATTEMPT)')
    mktemp = execute.index('remote_mktemp = runner(')
    checked = execute.index('checked = runner(')
    checked_pass = execute.index('checked.stdout.count("PASS_M1210_REMOTE_EXACT_TRANSFER_PREFLIGHT")')
    marker = execute.index('descriptor = os.open(LOCAL_ATTEMPT')
    launch = execute.index('launched = runner(')
    require(absent < mktemp < checked < checked_pass < marker < launch,
            "permanent no-retry marker after transfer preflight before launch")
    require("os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400" in execute,
            "marker O_EXCL mode0400")
    require("M1210_TRANSFER_COMPLETE__M1208_REMOTE_LAUNCH_ATTEMPT_CONSUMED__NO_RETRY" in execute,
            "permanent no-retry marker token")
    require("unlink(LOCAL_ATTEMPT" not in source and "rmtree(LOCAL_ATTEMPT" not in source,
            "marker never removed")
    require("single M1208 remote launch failed; no retry authorized" in execute,
            "launch failure is terminal")

    # Remote preflight validates the 95 inherited rows and all old/fresh
    # namespaces before staging or publishing any new file.
    old_pos = helper.index("for row in plan['old_dependencies']")
    m1180_pos = helper.index("plan['m1180_attempt']")
    namespace_pos = helper.index("for rel in [plan['m1180_result']")
    stage_pos = helper.index("stage=temp/'stage'")
    publish_pos = helper.index("for row in plan['members']", stage_pos + 1)
    replace_pos = helper.index("os.replace(tmp,dst)")
    require(old_pos < m1180_pos < namespace_pos < stage_pos < publish_pos < replace_pos,
            "old post-SHA and namespaces precede publication")
    require("p.stat().st_size!=row['size_bytes'] or sha(p)!=row['sha256']" in helper,
            "old dependency size plus post-SHA")
    require("M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\\n" in helper,
            "M1180 exact immutable attempt token")
    require("[plan['m1180_result'],plan['m1180_log'],plan['m1208_attempt'],plan['m1208_result'],plan['m1208_log']]"
            in helper, "M1180 failure and M1208 fresh namespaces")
    require("target_drift:" in helper and "continue" in helper
            and "except FileNotFoundError: st=None" in helper,
            "ABSENT-or-EXACT target policy")
    require("not stat.S_ISREG(st.st_mode) or dst.is_symlink()" in helper,
            "existing symlink/nonregular rejected")
    require("not stat.S_ISDIR(pst.st_mode) or cursor.is_symlink()" in helper,
            "destination parent symlink rejected")
    require("os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o444" in helper,
            "atomic nofollow temporary publication")
    require("tmp.stat().st_size!=row['size_bytes'] or sha(tmp)!=row['sha256']" in helper
            and "dst.stat().st_size!=row['size_bytes'] or sha(dst)!=row['sha256']" in helper,
            "pre/post publication exact SHA")

    # Remote mktemp and archive identity/type/order are authenticated before use.
    require("/usr/bin/mktemp\", \"-d\", REMOTE_TEMP_TEMPLATE" in execute,
            "remote mktemp exact argv")
    require("REMOTE_TEMP_RE.fullmatch(remote_temp_text)" in execute,
            "remote mktemp anchored response")
    require("st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o700" in helper,
            "remote mktemp owner0 mode0700")
    require("archive.stat().st_size!=plan['archive_size'] or sha(archive)!=plan['archive_sha256']"
            in helper, "archive size and SHA")
    require("[m.name for m in members]!=[r['path'] for r in plan['members']]" in helper,
            "archive member exact order")
    require("not m.isfile() or m.issym() or m.islnk() or m.size!=row['size_bytes']" in helper,
            "archive member type and size")
    require("n!=row['size_bytes'] or h.hexdigest()!=row['sha256']" in helper,
            "archive extracted content SHA")

    # M1180 is strictly read-only: no old child invocation or namespace cleanup.
    require("run_m1180" not in execute and "unlink(M1180" not in source
            and "rmtree(M1180" not in source, "M1180 no retry/no mutation")
    require("M1208_ATTEMPT_REL" in execute and "M1180_ATTEMPT_REL" in execute,
            "both disjoint namespace identities conveyed")


def reject_mutation(source: str, old: str, new: str, label: str) -> None:
    global mutations
    require(old in source, "mutation anchor " + label)
    changed = source.replace(old, new, 1)
    try:
        validate_source(changed)
    except (AssertionError, SyntaxError, ValueError):
        mutations += 1
        return
    raise AssertionError("mutation accepted " + label)


def main() -> None:
    for path, digest in EXPECTED.items():
        regular(path, "frozen identity")
        require(sha(path) == digest, "identity drift " + str(path))
    require(not os.path.lexists(LOCAL_ATTEMPT), "local M1210 attempt namespace fresh")

    source_contract = strict_json(SOURCE_CONTRACT)
    launch = strict_json(LAUNCH)
    inventory = strict_json(INVENTORY)
    old = strict_json(OLD_INVENTORY)
    receipt = strict_json(AUTHOR / "author_receipt.json")
    require(source_contract["schema"] ==
            "m1210_m1208_motion_ep29_unified_capture_secure_release_source_contract_r1_v1"
            and source_contract["status"] ==
            "INERT_SOURCE_ONLY__FRESH_M1211_HAMMER_REQUIRED__NO_REMOTE_NO_GPU",
            "source contract semantics")
    require(source_contract["source"]["sha256"] == sha(SOURCE)
            and source_contract["test"]["sha256"] == sha(TEST)
            and source_contract["launch_contract"]["sha256"] == sha(LAUNCH)
            and source_contract["inventory"]["sha256"] == sha(INVENTORY)
            and source_contract["transfer_list"]["sha256"] == sha(TRANSFER_LIST),
            "source contract exact bindings")
    require(launch["schema"] == "m1208_motion_ep29_unified_capture_launch_r1_v1"
            and launch["status"] == "M1175_AND_M1209_BOUND__ONE_M1208_GPU_RUN_AUTHORIZED",
            "launch contract semantics")
    require(launch["one_shot"] == {
        "attempt_consumed_by_child_after_identity_preflight": True,
        "attempt_marker": "hw_autoresearch_nts07/results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed",
        "attempt_token": "M1208_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE",
        "automatic_retry": False,
        "child_process_count": 1,
    }, "launch exactly once no retry")
    require(launch["prior_m1180_failure"]["automatic_retry"] is False,
            "M1180 retry forbidden by launch contract")
    require(launch["claim_boundary"]["remote_execution_authorized_now"] is False,
            "launch remains inert until M1211")

    require(inventory["schema"] ==
            "m1210_m1208_motion_ep29_unified_capture_remote_dependency_inventory_r1_v1"
            and inventory["status"] == "EXACT_NEW_RELEASE_TRANSFER__OLD_DEPENDENCIES_POST_SHA",
            "inventory semantics")
    rows = inventory["transfer_required"]
    listed = TRANSFER_LIST.read_text(encoding="utf-8").splitlines()
    require(len(rows) == 21 and inventory["fixed_transfer_member_count"] == 21,
            "exact 21 fixed transfer members")
    require(listed == [row["path"] for row in rows] and len(listed) == len(set(listed)),
            "transfer list exact order and uniqueness")
    for row in rows:
        path = ROOT / safe_rel(row["path"])
        regular(path, "transfer member")
        require(path.stat().st_size == row["size_bytes"] and sha(path) == row["sha256"],
                "transfer member size/SHA " + row["path"])
    require(inventory["new_m1208_files"] == 4
            and inventory["launch_contract_files"] == 1
            and inventory["m1208_author_files"] == 7
            and inventory["m1209_hammer_files"] == 9,
            "21-member category closure")

    dependencies = old["dependencies"]
    require(len(dependencies) == 95 and inventory["old_dependency_inventory"]["row_count"] == 95,
            "95 inherited remote dependency rows")
    require(len({row["path"] for row in dependencies}) == 95,
            "95 dependency paths unique")
    for row in dependencies:
        safe_rel(row["path"])
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None
                and isinstance(row["size_bytes"], int) and row["size_bytes"] > 0,
                "old dependency exact identity row")
    require(inventory["old_dependency_inventory"]["policy"] ==
            "ALL_ROWS_REMOTE_POST_SIZE_SHA_BEFORE_ANY_PUBLICATION",
            "95-row post-SHA policy")

    author_rows = verify_sealed_dir(AUTHOR, 4)
    verify_sealed_dir(M1208_AUTHOR, 5)
    verify_sealed_dir(M1209, 7)
    require(author_rows["author_receipt.json"] == sha(AUTHOR / "author_receipt.json"),
            "author receipt sealed")
    require(receipt["status"] ==
            "AUTHOR_COMPLETE__INERT_UNTIL_FRESH_M1211_HAMMER__NO_REMOTE_NO_GPU_NO_CAPTURE",
            "author status")
    require(receipt["bindings"]["old_m1182_dependency_rows_post_sha_before_publish"] == 95
            and receipt["bindings"]["fixed_transfer_members"] == 21,
            "author inventory closure")
    require(receipt["one_shot"]["remote_launcher_calls"] == 1
            and receipt["one_shot"]["automatic_retry"] is False,
            "author one-shot boundary")
    require(all(value is False for value in receipt["execution"].values()),
            "author performed no execution")

    # Launch contract has its own exact two-level sidecar seal.
    launch_sidecar = LAUNCH.with_name(LAUNCH.name + ".sha256")
    launch_outer = LAUNCH.with_name(LAUNCH.name + ".sha256.seal.sha256")
    regular(launch_sidecar, "launch sidecar"); regular(launch_outer, "launch outer")
    require(launch_sidecar.read_text().split() == [sha(LAUNCH), LAUNCH.name],
            "launch sidecar exact")
    require(launch_outer.read_text().split() == [sha(launch_sidecar), launch_sidecar.name],
            "launch outer exact")

    source = SOURCE.read_text(encoding="utf-8")
    validate_source(source)
    mutations_to_test = [
        ("REMOTE_HOST, shlex.join([REMOTE_INTERPRETER, \"-c\", launch_code])",
         "REMOTE_HOST, REMOTE_INTERPRETER, \"-c\", launch_code", "unquoted_launch_argv"),
        ("REMOTE_HOST, checked_command])", "REMOTE_HOST, REMOTE_INTERPRETER, \"-c\", REMOTE_HELPER])",
         "unquoted_preflight_argv"),
        ("descriptor = os.open(LOCAL_ATTEMPT", "launched = runner([])\n        descriptor = os.open(LOCAL_ATTEMPT",
         "launch_before_marker"),
        ("launched = runner(", "launched = runner(\n            # second launch\n            runner(",
         "second_launch"),
        ("for row in plan['old_dependencies']:", "for row in []:", "skip_old_post_sha"),
        ("plan['m1180_result'],plan['m1180_log'],plan['m1208_attempt'],plan['m1208_result'],plan['m1208_log']",
         "plan['m1208_attempt'],plan['m1208_result'],plan['m1208_log']", "drop_m1180_failure_boundary"),
        ("plan['m1180_result'],plan['m1180_log'],plan['m1208_attempt'],plan['m1208_result'],plan['m1208_log']",
         "plan['m1180_result'],plan['m1180_log']", "drop_fresh_m1208_namespaces"),
        ("or dst.is_symlink() or dst.stat().st_size", "or False or dst.stat().st_size",
         "accept_existing_symlink"),
        ("os.replace(tmp,dst)", "dst.write_bytes(tmp.read_bytes())", "nonmonotonic_publish"),
        ("stat.S_IMODE(st.st_mode)!=0o700", "False", "relax_remote_temp_mode"),
        ("archive.stat().st_size!=plan['archive_size'] or sha(archive)!=plan['archive_sha256']",
         "False", "relax_archive_identity"),
        ("os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400",
         "os.O_WRONLY | os.O_CREAT, 0o600", "remove_permanent_marker_exclusive"),
    ]
    for old_text, new_text, label in mutations_to_test:
        reject_mutation(source, old_text, new_text, label)
    require(mutations == 12, "all 12 independent mutations rejected")

    print(json.dumps({
        "schema": "m1211_m1210_m1208_motion_ep29_unified_capture_release_hammer_mechanical_r1_v1",
        "status": "PASS_M1210_SECURE_TRANSFER_AND_ONE_M1208_REMOTE_LAUNCH_AUTHORIZED",
        "checks_passed": checks,
        "mutations_rejected": mutations,
        "score": 100,
        "p0_count": 0,
        "p1_count": 0,
        "fixed_transfer_members": 21,
        "old_dependency_post_sha_rows": 95,
        "remote_mktemp_owner0_mode0700": True,
        "archive_exact_type_order_size_sha": True,
        "absent_or_exact_monotonic_publish": True,
        "m1180_read_only_failure_boundary": True,
        "m1208_fresh_namespaces": True,
        "local_no_retry_marker_before_launch": True,
        "launch_count": 1,
        "automatic_retry": False,
        "remote_runs": 0,
        "network_runs": 0,
        "gpu_runs": 0,
        "capture_runs": 0,
        "eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
        "bindings": {str(path.relative_to(ROOT)): digest for path, digest in EXPECTED.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
