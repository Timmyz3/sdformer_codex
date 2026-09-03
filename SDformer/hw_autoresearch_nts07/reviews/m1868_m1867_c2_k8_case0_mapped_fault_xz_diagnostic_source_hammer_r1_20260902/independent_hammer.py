#!/usr/bin/env python3
"""Different-author source-only M1868 hammer for M1867.

This program performs static identity, seal, namespace, and synchronized-source
mutation checks only.  It never invokes the runner, VCS, simv, EDA, a license
query, UCLI, SAIF, or PTPX.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1867_c2_k8_case0_mapped_fault_xz_diagnostic_source.py"
SPEC = importlib.util.spec_from_file_location("m1867_checker_for_m1868", str(CHECKER))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

CONTRACT = C.CONTRACT
AUTHOR = HW / "reviews/m1867_m1863_m1862_c2_k8_case0_mapped_fault_xz_diagnostic_source_author_receipt_r1_20260902"
PREDECESSOR = HW / "reviews/m1863_m1862_c2_k8_case0_mapped_fault_xz_diagnostic_source_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1867_c2_k8_case0_mapped_fault_xz_diagnostic_attempt_consumed"
RESULT = HW / "results/m1867_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902"
FAILURE = HW / "results/m1867_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902.failed_or_incomplete.quarantine"
RELEASE = HW / "contracts/m1869_m1868_m1867_c2_k8_case0_mapped_fault_xz_diagnostic_launch_release_r1_20260902.json"

EXPECTED = {
    "contract": "954229788bc67ad5b1fdb09a83fe341a04a9099305e5cbd4c3634446f82f64d5",
    "contract_sidecar": "cc96d6a01f035cdab108d374d83e001872a1e95094ae76118983d51af8bf1bc0",
    "contract_outer": "98a4f646bf50e011d87d0807f57391189e216b9f6b775fc97496f52b4542ae14",
    "runner": "39a9561a6544b38cbb3e9a0363a3de54cfc2a08ddb59e257f8707e22c9af23dc",
    "tb": "7f9481279599f6e9146c5b3e9434188d312e09e1b3892666b580d31417c6aeb5",
    "filelist": "e13944a7c57806f340cc8bc145be3b2aad3b8e2d08dcd3fa86518c8a689eaa13",
    "checker": "ff752e5719020334418dc03ab2facf451b9ba3f1df6b0d1dfc660cad79efa639",
    "test": "3cd1d8c9edcbf410cdb2b9f92049a4aaf94f0572a7eadfa042e8f57b9648c8b0",
    "author_receipt": "c68c5237a12d2b68a1bab2423819edba7b9301bb22ee84a9bb89fcb928f4a46c",
    "author_manifest": "0c506c5f5781416e59b72c99e2793cc47765cbf2177d5b7440181126b2ff9f2b",
    "author_outer": "af06ac43f6c6e10c05787271d5a3ceb8d702b350081e1ace48685fbaeb9ce7a1",
    "predecessor_review": "b6b493b44cf5505ca9b2f70310f37827ad0d4da988316edae5365ad770d04810",
    "predecessor_manifest": "7d014d5ee955d9baf1f7719fa76879133d56770bde4644a89f98ede9300cd8c4",
    "predecessor_outer": "30587c2a246827b4fd01d682a5d341c26e13c6b4949f22009838affde2198196",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerFailure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, expected):
    path = Path(path)
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode) or sha(path) != expected):
        raise HammerFailure("identity drift: " + str(path))


def strict(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise HammerFailure("duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           HammerFailure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise HammerFailure("JSON root")
    return value


def sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise HammerFailure("sealed directory absent: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text() != manifest_sha + "  SHA256SUMS\n":
        raise HammerFailure("outer seal semantics: " + str(root))
    listed = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise HammerFailure("manifest syntax: " + str(root))
        name = fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise HammerFailure("unsafe member: " + name)
        exact(root / rel, fields[0])
        listed[name] = fields[0]
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise HammerFailure("symlink in sealed directory: " + str(path))
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if set(listed) != actual:
        raise HammerFailure("sealed population drift: " + str(root))
    return listed


def verify_file_seal(path, file_sha, sidecar_sha, outer_sha):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    exact(path, file_sha)
    exact(sidecar, sidecar_sha)
    exact(outer, outer_sha)
    if sidecar.read_text() != file_sha + "  " + path.name + "\n":
        raise HammerFailure("contract sidecar semantics")
    if outer.read_text() != sidecar_sha + "  " + sidecar.name + "\n":
        raise HammerFailure("contract outer semantics")


def verify_chain():
    verify_file_seal(CONTRACT, EXPECTED["contract"], EXPECTED["contract_sidecar"],
                     EXPECTED["contract_outer"])
    for name in ("runner", "tb", "filelist", "checker", "test"):
        exact(C.PATHS[name], EXPECTED[name])
    exact(DOCS359, EXPECTED["docs359"])

    members = sealed_directory(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    if members.get("author_receipt.json") != EXPECTED["author_receipt"]:
        raise HammerFailure("author receipt member drift")
    receipt = strict(AUTHOR / "author_receipt.json")
    expected_auth = {
        "license_queries": 0, "attempts_created": 0, "vcs_compiles": 0,
        "simv_runs": 0, "ucli_runs": 0, "saif_files": 0, "ptpx_runs": 0,
        "all_other_eda_runs": 0, "results_created": 0,
        "releases_created": 0, "paper_claim_now": False,
    }
    if (receipt.get("status") !=
            "PASS_SOURCE_AUTHORING_ONLY_M1867_DIAGNOSTIC_SUCCESSOR__M1868_REVIEW_M1869_RELEASE_REQUIRED__NO_EDA_NO_LICENSE_NO_ATTEMPT"
            or receipt.get("authorization") != expected_auth):
        raise HammerFailure("author receipt semantics")

    members = sealed_directory(PREDECESSOR, EXPECTED["predecessor_manifest"],
                               EXPECTED["predecessor_outer"])
    if members.get("review.json") != EXPECTED["predecessor_review"]:
        raise HammerFailure("predecessor review member drift")
    failed = strict(PREDECESSOR / "review.json")
    if (failed.get("audit_status") != "FAIL_CLOSED"
            or failed.get("severity_counts") != {"p0": 0, "p1": 1, "p2": 0}
            or failed.get("authorization", {}).get("m1864_release") is not False):
        raise HammerFailure("predecessor semantics")

    for path in (ATTEMPT, RESULT, FAILURE, RELEASE,
                 Path(str(RELEASE) + ".sha256"),
                 Path(str(RELEASE) + ".sha256.seal.sha256")):
        if path.exists() or path.is_symlink():
            raise HammerFailure("pre-review namespace already exists: " + str(path))
    C.check()


def synchronized_override(texts, name, old, new):
    if old not in texts[name]:
        raise HammerFailure("attack anchor absent: " + name + " / " + old[:64])
    changed = texts[name].replace(old, new, 1)
    contract = json.loads(texts["contract"])
    rel = C.PATHS[name].relative_to(HW).as_posix()
    contract["source_files"][rel] = hashlib.sha256(changed.encode()).hexdigest()
    return {name: changed, "contract": json.dumps(contract, sort_keys=True)}


def contract_override(texts, mutate):
    contract = json.loads(texts["contract"])
    mutate(contract)
    return {"contract": json.dumps(contract, sort_keys=True)}


def rejected(overrides):
    try:
        C.check(overrides)
    except (C.CheckFailure, SyntaxError, ValueError):
        return True
    return False


def run_attacks():
    texts = C.source_map()
    rows = []

    def attack(label, name, old, new, group):
        ok = rejected(synchronized_override(texts, name, old, new))
        rows.append({"name": label, "group": group,
                     "result": "REJECTED" if ok else "ESCAPED"})

    # The exact twelve M1857 synchronized-inventory escape classes.
    attack("main_path_verify_authority_call", "runner",
           "        release_sha = verify_authority()", '        release_sha = "0" * 64', "m1857_exact12")
    attack("attempt_namespace", "runner",
           "results/.m1867_c2_k8_case0_mapped_fault_xz_diagnostic_attempt_consumed",
           "results/.m1845_c2_fresh_mapped_production_energy_attempt_consumed", "m1857_exact12")
    attack("result_namespace", "runner",
           "results/m1867_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902",
           "results/m1845_c2_fresh_mapped_production_energy_r1_20260902", "m1857_exact12")
    attack("first_namespace_freshness", "runner",
           "        namespaces_fresh()\n        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)",
           "        # removed\n        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)", "m1857_exact12")
    attack("second_namespace_freshness", "runner",
           "        collision_gate()\n        namespaces_fresh()\n        ATTEMPT.mkdir()",
           "        collision_gate()\n        # removed\n        ATTEMPT.mkdir()", "m1857_exact12")
    attack("global_queue_lock", "runner",
           "        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)", "        # removed", "m1857_exact12")
    attack("local_nonblocking_lock", "runner",
           "        fcntl.flock(local_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
           "        # removed", "m1857_exact12")
    attack("per_tool_runtime_collision_gate", "runner",
           "    CHECK.validate_sources()\n    collision_gate()\n    env = {",
           "    CHECK.validate_sources()\n    # removed\n    env = {", "m1857_exact12")
    attack("published_result_claim_boundary", "runner",
           '            "claim_boundary": CHECK.CLAIMS})', '            "claim_boundary": {}})', "m1857_exact12")
    attack("first_nonbinary_stop", "tb", "            $finish;", "            ;", "m1857_exact12")
    attack("first_nonbinary_token_sampled_value", "tb",
           "edge_name, core.protocol_error);", "edge_name, 1'bx);", "m1857_exact12")
    drift = contract_override(texts, lambda row: row["exact_diagnostic_identity"].update(
        {"mapped_netlist": "evil.v"}))
    rows.append({"name": "contract_mapped_path_identity", "group": "m1857_exact12",
                 "result": "REJECTED" if rejected(drift) else "ESCAPED"})

    # The exact four M1863 constant-false reachability attacks.
    attack("authority_call_guarded_by_constant_false", "runner",
           "        release_sha = verify_authority()",
           "        if False:\n            release_sha = verify_authority()", "m1863_exact4")
    attack("attempt_creation_guarded_by_constant_false", "runner",
           "        ATTEMPT.mkdir()", "        if False:\n            ATTEMPT.mkdir()", "m1863_exact4")
    attack("compile_guarded_by_constant_false", "runner",
           '        run(compile_command(), WORK, WORK / "compile.log", 7200)',
           '        if False:\n            run(compile_command(), WORK, WORK / "compile.log", 7200)', "m1863_exact4")
    attack("first_finish_guarded_by_constant_false", "tb",
           "            $finish;", "            if (1'b0) $finish;", "m1863_exact4")

    # Independent predecessor-terminal attacks.  All counted critical actions
    # remain direct statements and in the expected order, but become unreachable.
    attack("nested_true_return_before_authority", "runner",
           "        release_sha = verify_authority()",
           "        if True:\n            return 0\n        release_sha = verify_authority()", "independent_terminal4")
    attack("nested_true_raise_before_attempt", "runner",
           "        ATTEMPT.mkdir()",
           '        if True:\n            raise Failure("stop")\n        ATTEMPT.mkdir()', "independent_terminal4")
    attack("nested_true_return_before_compile", "runner",
           '        run(compile_command(), WORK, WORK / "compile.log", 7200)',
           '        if True:\n            return 0\n        run(compile_command(), WORK, WORK / "compile.log", 7200)',
           "independent_terminal4")
    attack("runtime_true_return_before_subprocess", "runner",
           '    with Path(output).open("wb") as stream:',
           '    if True:\n        return\n    with Path(output).open("wb") as stream:',
           "independent_terminal4")
    return rows


def main():
    verify_chain()
    rows = run_attacks()
    groups = {}
    for group in ("m1857_exact12", "m1863_exact4", "independent_terminal4"):
        selected = [row for row in rows if row["group"] == group]
        groups[group] = {
            "total": len(selected),
            "rejected": sum(row["result"] == "REJECTED" for row in selected),
            "escaped": sum(row["result"] == "ESCAPED" for row in selected),
        }
    expected = {
        "m1857_exact12": {"total": 12, "rejected": 12, "escaped": 0},
        "m1863_exact4": {"total": 4, "rejected": 4, "escaped": 0},
        "independent_terminal4": {"total": 4, "rejected": 0, "escaped": 4},
    }
    if groups != expected:
        raise HammerFailure("unexpected attack matrix: " + repr(groups))
    print(json.dumps({
        "status": "FAIL_CLOSED_M1868_M1867_SOURCE__HISTORICAL_16_REJECTED__NEW_TERMINAL_4_ESCAPED",
        "groups": groups,
        "attacks": rows,
        "eda_or_license_run": False,
        "m1869_release_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
