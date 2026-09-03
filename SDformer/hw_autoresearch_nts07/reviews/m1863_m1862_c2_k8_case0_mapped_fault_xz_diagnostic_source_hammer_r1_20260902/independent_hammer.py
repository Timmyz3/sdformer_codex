#!/usr/bin/env python3
"""Different-author, source-only M1863 hammer of the M1862 diagnostic successor.

This program never invokes a simulator, EDA executable, license query, or the
M1862 runner.  Source mutations synchronize the contract inventory hash before
calling the candidate checker, so rejection cannot be caused merely by a stale
inventory entry.
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
CHECKER = HW / "system_simulator/scripts/check_m1862_c2_k8_case0_mapped_fault_xz_diagnostic_source.py"
SPEC = importlib.util.spec_from_file_location("m1862_checker_for_m1863", str(CHECKER))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

CONTRACT = C.CONTRACT
AUTHOR = HW / "reviews/m1862_m1857_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_source_author_receipt_r1_20260902"
PREDECESSOR = HW / "reviews/m1857_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_source_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1862_c2_k8_case0_mapped_fault_xz_diagnostic_attempt_consumed"
RESULT = HW / "results/m1862_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902"
FAILURE = HW / "results/m1862_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902.failed_or_incomplete.quarantine"
RELEASE = HW / "contracts/m1864_m1863_m1862_c2_k8_case0_mapped_fault_xz_diagnostic_launch_release_r1_20260902.json"

EXPECTED = {
    "contract": "85d834c671689a9e4c08af0d310b89c5b18d811e2b9233d554df1d3efe0aa83e",
    "contract_sidecar": "55bb5ad1afb776245ff3accac7263bb0cc2c3c13bddcb19fd1bde55c0479c16b",
    "contract_outer": "3d0d60d8eb2292b40aae26c58dd72dc13837d2684ab9584c2d14e343f3581200",
    "runner": "4e3a5123dc54c2a4521deb2a7fdbe022e580ed80eb3485cadb98a5a002cf3bbc",
    "tb": "d05f8475ea4349e658ee8a2efb93568232d0d3c53d24ed7ba0904375e6f6b3a2",
    "filelist": "e1bbb62961c551ecca9b2429e105ae71a7a88aa6905b16a0f7f694ea76c46474",
    "checker": "382fb0d6066f39097195f520c994a8d0bbfba14c1cd2fd7d19ff4bd8e7bc900e",
    "test": "37edfc75e964dba7c16a158a59b6571868b2f01988a0c9998d7c5552bb5ee99d",
    "author_receipt": "749a52dc1328b89812dbad7968a0391bd32b6e5f574a9fda60a6c58955790e08",
    "author_manifest": "00d68cf2feb0d2652a200f0513a9814bf7710ed3bbbb3edc147aeea119c8dc3c",
    "author_outer": "ffcf621ad2354f23a1ed327373a525cb0759ec6f597aeade687555d0a0da5da3",
    "predecessor_review": "dce664b6efe5897b4d646234bb0dbf40c7c768fa8a955ab72d22c2081eb3a2a1",
    "predecessor_manifest": "fb2ec220afdaf6fdd40e19319fd8845b8500337a404109b986aaa103473fc885",
    "predecessor_outer": "8cb2b3b6d922545a86c7e050e0d7572f6c666d6c2653f4478d423688bcb32921",
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
        result = {}
        for key, value in items:
            if key in result:
                raise HammerFailure("duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           HammerFailure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise HammerFailure("JSON root")
    return value


def sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise HammerFailure("sealed directory absent/nonregular: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text() != manifest_sha + "  SHA256SUMS\n":
        raise HammerFailure("outer seal semantic drift: " + str(root))
    listed = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise HammerFailure("manifest syntax: " + str(root))
        name = fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise HammerFailure("unsafe/duplicate member: " + name)
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
    if (receipt.get("status") !=
            "PASS_SOURCE_AUTHORING_ONLY_M1862_DIAGNOSTIC_SUCCESSOR__M1863_REVIEW_M1864_RELEASE_REQUIRED__NO_EDA_NO_LICENSE_NO_ATTEMPT"
            or receipt.get("authorization") != {
                "license_queries": 0, "attempts_created": 0, "vcs_compiles": 0,
                "simv_runs": 0, "ucli_runs": 0, "saif_files": 0,
                "ptpx_runs": 0, "all_other_eda_runs": 0,
                "results_created": 0, "releases_created": 0,
                "paper_claim_now": False}):
        raise HammerFailure("author receipt semantics")

    members = sealed_directory(PREDECESSOR, EXPECTED["predecessor_manifest"],
                               EXPECTED["predecessor_outer"])
    if members.get("review.json") != EXPECTED["predecessor_review"]:
        raise HammerFailure("M1857 predecessor member drift")
    failed = strict(PREDECESSOR / "review.json")
    if (failed.get("audit_status") != "FAIL_CLOSED"
            or failed.get("severity_counts") != {"p0": 0, "p1": 1, "p2": 0}
            or failed.get("authorization", {}).get("m1858_release") is not False):
        raise HammerFailure("M1857 predecessor semantics")

    for path in (ATTEMPT, RESULT, FAILURE, RELEASE,
                 Path(str(RELEASE) + ".sha256"),
                 Path(str(RELEASE) + ".sha256.seal.sha256")):
        if path.exists() or path.is_symlink():
            raise HammerFailure("pre-review authority/namespace already exists: " + str(path))
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
    results = []

    def attack(label, name, old, new, group):
        ok = rejected(synchronized_override(texts, name, old, new))
        results.append({"name": label, "group": group,
                        "result": "REJECTED" if ok else "ESCAPED"})

    # Exact twelve M1857 synchronized-inventory escapes.
    attack("main_path_verify_authority_call", "runner",
           "        release_sha = verify_authority()", '        release_sha = "0" * 64', "m1857_exact12")
    attack("attempt_namespace", "runner",
           "results/.m1862_c2_k8_case0_mapped_fault_xz_diagnostic_attempt_consumed",
           "results/.m1845_c2_fresh_mapped_production_energy_attempt_consumed", "m1857_exact12")
    attack("result_namespace", "runner",
           "results/m1862_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902",
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
           '            "claim_boundary": CHECK.CLAIMS})', '            "claim_boundary": {}})',
           "m1857_exact12")
    attack("first_nonbinary_stop", "tb", "            $finish;", "            ;", "m1857_exact12")
    attack("first_nonbinary_token_sampled_value", "tb",
           "edge_name, core.protocol_error);", "edge_name, 1'bx);", "m1857_exact12")
    drift = contract_override(texts, lambda row: row["exact_diagnostic_identity"].update(
        {"mapped_netlist": "evil.v"}))
    results.append({"name": "contract_mapped_path_identity", "group": "m1857_exact12",
                    "result": "REJECTED" if rejected(drift) else "ESCAPED"})

    # Independent direct-control-flow probes.  These preserve the nodes/tokens
    # counted by the candidate checker but render the guarded action unreachable.
    attack("authority_call_guarded_by_constant_false", "runner",
           "        release_sha = verify_authority()",
           "        if False:\n            release_sha = verify_authority()", "independent_control_flow")
    attack("attempt_creation_guarded_by_constant_false", "runner",
           "        ATTEMPT.mkdir()", "        if False:\n            ATTEMPT.mkdir()", "independent_control_flow")
    attack("compile_guarded_by_constant_false", "runner",
           '        run(compile_command(), WORK, WORK / "compile.log", 7200)',
           '        if False:\n            run(compile_command(), WORK, WORK / "compile.log", 7200)',
           "independent_control_flow")
    attack("first_finish_guarded_by_constant_false", "tb",
           "            $finish;", "            if (1'b0) $finish;", "independent_control_flow")
    return results


def main():
    verify_chain()
    attacks = run_attacks()
    exact12 = [row for row in attacks if row["group"] == "m1857_exact12"]
    independent = [row for row in attacks if row["group"] == "independent_control_flow"]
    result = {
        "status": "FAIL_CLOSED_M1863_M1862_SOURCE_HAMMER",
        "source_identity_and_seals": "PASS",
        "m1857_exact12_rejected": sum(row["result"] == "REJECTED" for row in exact12),
        "m1857_exact12_escaped": sum(row["result"] == "ESCAPED" for row in exact12),
        "independent_rejected": sum(row["result"] == "REJECTED" for row in independent),
        "independent_escaped": sum(row["result"] == "ESCAPED" for row in independent),
        "attacks": attacks,
        "m1864_release_authorized": False,
        "eda_or_license_run": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
