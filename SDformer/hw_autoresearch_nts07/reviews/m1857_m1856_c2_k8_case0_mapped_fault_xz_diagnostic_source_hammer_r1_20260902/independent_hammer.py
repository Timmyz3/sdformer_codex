#!/usr/bin/env python3
"""Different-author, read-only source hammer for M1856 diagnostic source."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_source.py"
SPEC = importlib.util.spec_from_file_location("m1856_checker_for_m1857_hammer", str(CHECKER))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

CONTRACT = HW / "contracts/m1856_m1854_m1845_c2_k8_case0_mapped_fault_xz_diagnostic_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1856_m1854_m1845_c2_k8_case0_mapped_fault_xz_diagnostic_source_author_receipt_r1_20260902"
M1854 = HW / "reviews/m1854_m1845_c2_mapped_energy_failure_hammer_r1_20260902"
M1845_ATTEMPT = HW / "results/.m1845_c2_fresh_mapped_production_energy_attempt_consumed"
M1845_FAILURE = HW / "results/m1845_c2_fresh_mapped_production_energy_r1_20260902.failed_or_incomplete.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1858_RELEASE = HW / "contracts/m1858_m1857_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_launch_release_r1_20260902.json"
M1856_ATTEMPT = HW / "results/.m1856_c2_k8_case0_mapped_fault_xz_diagnostic_attempt_consumed"
M1856_RESULT = HW / "results/m1856_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902"
M1856_FAILURE = HW / "results/m1856_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902.failed_or_incomplete.quarantine"

EXPECTED = {
    "contract": "3fbb5d000adbc28117a88d44fd24e50173f92b9ebbb69db009c4d486df410800",
    "contract_sidecar": "cc2bed7b9c8d4f1443f40a424c648b61853baf4559beb1a887f69428060c934e",
    "contract_outer": "5a9adf1c94407869331c372615590252802ca28187cdee1eadc026b87dd91d49",
    "runner": "c472cc4a8f0079e5067fc33d532b0ab146ec3190c823606e43cc4f7bf78e80d2",
    "tb": "c3641d4184fa69b669bbae3ae5de88095b2c639ec664677067a3c29d4682c616",
    "filelist": "3d879d295c5a45e763001c0403f43a16091f2a2172b2dcaf2b2987f098262afe",
    "checker": "34fd3a3e8f150105a1a9ec8d0caa59441f5dbaeec1b434217eca9c8f03b6e80f",
    "test": "ab104b2836ebf2fa86036d378e441bd1be1ebe1b89b5b84f4c15d0ce217294ad",
    "author_receipt": "74735384e768614ae3039b097221d2aae8cd44e28a5d987bf289bc7be92cb597",
    "author_manifest": "7489e65046e563db21a91efe4a5453ff41c5751113fd0ef845bdea8ef1eb8877",
    "author_outer": "d47dce0a9c0ae078c6e749897d94b66f04ae9b9e91a68ee2a688978ab0b4a137",
    "m1854_review": "9ef9f00091b145e438a03c9039123af198f28ef16f6d3180169361ca6470d0a6",
    "m1854_manifest": "49176d9165cbfe449f243fe4f76b2e2ae3af1e388398d8556f843e75dbfd10b8",
    "m1854_outer": "28089b517fbe5a7f052dfef98b031cfc9887f8f288d610f917cd3234acc2c1f4",
    "m1845_attempt_json": "ed24cf02e484364189abb0dc02fb9ca9a59064aa22393adb7892db9231975a18",
    "m1845_attempt_manifest": "b43356a0ba2851b96a6226f850f1bb8bf8a7d039f96b0a47f992e167f6fdc62d",
    "m1845_attempt_outer": "ff2885f6013702d2f6a2976b2800391c7a591cdecad46a7b57917ce9efb8565f",
    "m1845_failure_json": "108e3e312f3e7650ea4e4aa2283a37a2404f7890755cbe5d7a95624fade23b5c",
    "m1845_failure_manifest": "d87ecfe15696bc6167f1f19d192052a92b34f042579e2cae8cf4ab2e1ee7e10c",
    "m1845_failure_outer": "f13da71243208c743493067b4f9205ace57df6d57b89a4c0df2143168889db2e",
    "mapped": "63605469818c36574ce9719130877610e79cf0c3b7317c0e69848539afa6b792",
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


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


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
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise HammerFailure("sealed directory absent/nonregular: " + str(root))
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
            raise HammerFailure("unsafe/duplicate manifest member: " + name)
        exact(root / rel, fields[0])
        listed[name] = fields[0]
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise HammerFailure("symlink in sealed directory: " + str(path))
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(listed):
        raise HammerFailure("sealed population drift: " + str(root))
    return listed


def verify_sources_and_chain():
    exact(CONTRACT, EXPECTED["contract"])
    exact(Path(str(CONTRACT) + ".sha256"), EXPECTED["contract_sidecar"])
    exact(Path(str(CONTRACT) + ".sha256.seal.sha256"), EXPECTED["contract_outer"])
    if Path(str(CONTRACT) + ".sha256").read_text() != EXPECTED["contract"] + "  " + CONTRACT.name + "\n":
        raise HammerFailure("contract sidecar semantics")
    if Path(str(CONTRACT) + ".sha256.seal.sha256").read_text() != EXPECTED["contract_sidecar"] + "  " + CONTRACT.name + ".sha256\n":
        raise HammerFailure("contract outer semantics")
    for name in ("runner", "tb", "filelist", "checker", "test"):
        exact(C.PATHS[name], EXPECTED[name])
    members = sealed_directory(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    if members.get("author_receipt.json") != EXPECTED["author_receipt"]:
        raise HammerFailure("author receipt member drift")
    receipt = strict(AUTHOR / "author_receipt.json")
    if (receipt.get("authorization", {}).get("vcs_compiles") != 0
            or receipt.get("authorization", {}).get("simv_runs") != 0
            or receipt.get("authorization", {}).get("license_queries") != 0):
        raise HammerFailure("author receipt execution drift")
    members = sealed_directory(M1854, EXPECTED["m1854_manifest"], EXPECTED["m1854_outer"])
    if members.get("review.json") != EXPECTED["m1854_review"]:
        raise HammerFailure("M1854 review member drift")
    failure_review = strict(M1854 / "review.json")
    if (failure_review.get("production_admission") != "FAIL_CLOSED"
            or failure_review.get("severity_counts") != {"p0": 0, "p1": 1, "p2": 0}
            or failure_review.get("execution_audit", {}).get("automatic_retry") is not False):
        raise HammerFailure("M1854 failure semantics")
    members = sealed_directory(M1845_ATTEMPT, EXPECTED["m1845_attempt_manifest"], EXPECTED["m1845_attempt_outer"])
    if members.get("attempt.json") != EXPECTED["m1845_attempt_json"]:
        raise HammerFailure("M1845 attempt member drift")
    attempt = strict(M1845_ATTEMPT / "attempt.json")
    if attempt.get("status") != "M1845_ATTEMPT_CONSUMED" or attempt.get("automatic_retry") is not False:
        raise HammerFailure("M1845 attempt/retry boundary")
    members = sealed_directory(M1845_FAILURE, EXPECTED["m1845_failure_manifest"], EXPECTED["m1845_failure_outer"])
    if members.get("failure.json") != EXPECTED["m1845_failure_json"]:
        raise HammerFailure("M1845 failure member drift")
    failure = strict(M1845_FAILURE / "failure.json")
    if (failure.get("status") != "FAILED_OR_INCOMPLETE_DO_NOT_RETRY"
            or failure.get("phase") != "SIM_k8_0"
            or failure.get("automatic_retry") is not False):
        raise HammerFailure("M1845 failure/no-retry semantics")
    exact(C.MAPPED, EXPECTED["mapped"])
    exact(DOCS359, EXPECTED["docs359"])
    C.check()


def synchronized_override(texts, name, old, new):
    if old not in texts[name]:
        raise HammerFailure("attack anchor absent: " + name + " / " + old[:60])
    changed = texts[name].replace(old, new, 1)
    contract = json.loads(texts["contract"])
    rel = C.PATHS[name].relative_to(HW).as_posix()
    contract["source_files"][rel] = text_sha(changed)
    return {name: changed, "contract": json.dumps(contract, sort_keys=True)}


def rejected(overrides):
    try:
        C.check(overrides)
    except (C.CheckFailure, SyntaxError, ValueError):
        return True
    return False


def run_attacks():
    texts = C.source_map()
    results = []

    def attack(label, name, old, new, expected_reject=True):
        result = "REJECTED" if rejected(synchronized_override(texts, name, old, new)) else "ESCAPED"
        results.append({"name": label, "expected_reject": expected_reject, "result": result})

    # Governance and namespace mutations are material because the package
    # promises authority/freshness before one unique attempt and no M1845 retry.
    attack("authority_call_removed", "runner",
           "        release_sha = verify_authority()",
           "        release_sha = \"0\" * 64")
    attack("m1856_attempt_namespace_retargeted_to_consumed_m1845", "runner",
           "results/.m1856_c2_k8_case0_mapped_fault_xz_diagnostic_attempt_consumed",
           "results/.m1845_c2_fresh_mapped_production_energy_attempt_consumed")
    attack("m1856_result_namespace_retargeted_to_m1845", "runner",
           "results/m1856_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902",
           "results/m1845_c2_fresh_mapped_production_energy_r1_20260902")
    attack("first_namespace_freshness_removed", "runner",
           "        namespaces_fresh()\n        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)",
           "        # first freshness removed\n        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)")
    attack("second_namespace_freshness_removed", "runner",
           "        collision_gate()\n        namespaces_fresh()\n        ATTEMPT.mkdir()",
           "        collision_gate()\n        # second freshness removed\n        ATTEMPT.mkdir()")
    attack("global_queue_lock_removed", "runner",
           "        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)",
           "        # global queue lock removed")
    attack("local_lock_removed", "runner",
           "        fcntl.flock(local_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
           "        # local lock removed")
    attack("run_time_collision_gate_removed", "runner",
           "    CHECK.validate_sources()\n    collision_gate()\n    env = {",
           "    CHECK.validate_sources()\n    # run-time collision gate removed\n    env = {")
    attack("result_claim_boundary_erased", "runner",
           '            "claim_boundary": CHECK.CLAIMS})',
           '            "claim_boundary": {}})')
    attack("first_stop_removed", "tb", "                $finish;", "                ;")
    attack("first_token_value_disconnected", "tb",
           "edge_name, core.protocol_error);",
           "edge_name, 1'bx);")
    # The contract path text is not execution authority, but its escape proves
    # the semantic contract checker is incomplete as written.
    contract = json.loads(texts["contract"])
    contract["exact_diagnostic_identity"]["mapped_netlist"] = "evil.v"
    result = "REJECTED" if rejected({"contract": json.dumps(contract)}) else "ESCAPED"
    results.append({"name": "contract_mapped_path_identity_drift",
                    "expected_reject": True, "result": result})

    # Controls expected to reject; these distinguish the real blind spots from
    # a completely inert checker.
    attack("second_compile_added", "runner",
           "        run(compile_command(), WORK, WORK / \"compile.log\", 7200)",
           "        run(compile_command(), WORK, WORK / \"compile.log\", 7200)\n"
           "        run(compile_command(), WORK, WORK / \"compile2.log\", 7200)")
    attack("case_changed", "runner", '"+M979_CASE=0"', '"+M979_CASE=1"')
    attack("ucli_added", "runner", '"+M979_CASE=0"', '"+M979_CASE=0", "+M979_UCLI_SAIF"')
    attack("axis_changed", "filelist", "+define+M1831_AXIS_K8", "+define+M1831_AXIS_K1X8")
    attack("case_equality_weakened", "tb",
           "(value === 1'b0) || (value === 1'b1)",
           "(value == 1'b0) || (value == 1'b1)")
    attack("internal_tap_decides", "tb",
           "            if (!is_binary(core.protocol_error)) begin",
           "            if (!is_binary(mapped_protocol_error_q_tap)) begin")

    escaped = [row for row in results if row["result"] == "ESCAPED"]
    return results, escaped


def verify_namespaces_and_collision():
    for path in (M1856_ATTEMPT, M1856_RESULT, M1856_FAILURE, M1858_RELEASE):
        if os.path.lexists(str(path)):
            raise HammerFailure("unauthorized M1856/M1858 namespace exists: " + str(path))
    # The same integer label M1857 is already used by a different exact-path
    # Formality failure review.  Exact review/release paths and authority pins
    # are distinct, so this is a readability collision, not a filesystem or
    # launch-authority collision.
    other = HW / "reviews/m1857_m1850_c2_formality_pt_failure_hammer_r1_20260902"
    if not other.is_dir():
        raise HammerFailure("expected other M1857 review absent")


def main():
    verify_sources_and_chain()
    verify_namespaces_and_collision()
    results, escaped = run_attacks()
    print(json.dumps({
        "status": "FAIL_CLOSED_M1857_M1856_DIAGNOSTIC_SOURCE_HAMMER",
        "official_static_checker": C.check()["status"],
        "attacks": len(results),
        "rejected": len(results) - len(escaped),
        "escaped": len(escaped),
        "escaped_names": [row["name"] for row in escaped],
        "results": results,
        "eda_or_license_run": False,
        "m1858_release_authorized": False,
        "milestone_label_collision_blocking": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
