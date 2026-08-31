#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh static M1440 hammer for the reachable M1432 C2 campaign.

This program never imports or invokes the M1432 executor and contains no
process-launch primitive.  It checks exact bytes/seals, independently audits
the executor's one-shot ordering, and attacks the semantic audit with mutated
source/authority candidates.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_one_shot.py"
AUTHORITY = HW / "contracts/m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_authority_r1_20260831.json"
M1361 = HW / "reviews/m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_author_r1_20260831"
M1362 = HW / "reviews/m1362_m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_blind_hammer_r1_20260831"
M1432 = HW / "reviews/m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_authority_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
UCLI = HW / "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"

SHA = {
    "runner": "314be83304d4b62cf2c4b73feb394fa2ab20e60a89afb9c3dfc07622d25a7156",
    "authority": "5b1759872a59f532cbb86a23c0922eb24125b6a1781aac20e490f486044394aa",
    "authority_sum": "ecfb2434646cb6e59ec201a1f192b89f4b66e72a8bef5c854c21aafcaf3f5c25",
    "authority_outer": "ec4b8d2265af176876783d89c77b19006ae1517012dbb966433b2c9e10176f1d",
    "m1361_review": "d4369a78849b7f3f7411cc1c21365e17450275b01ed906468c368781b140126c",
    "m1361_manifest": "e00f9cfc6222c92ecd7f6b7e0ca7d0f1c46204634f208cdac3545e707e4edaaa",
    "m1361_outer": "634258227ac5143d820fa696ed8cb572f8c622d7b4ad8e3c0db404a0b2adbdaf",
    "m1362_review": "dafe39f181c85c1b08c7aaaaee29039005ec6a6b55386f2a2755aabca3f441b5",
    "m1362_manifest": "b546b35fbed2b0a8966b66ee34c22f0f72c93db00e5248c9808c0eda40360dd5",
    "m1362_outer": "32dae68fe7bdca213619ca19e2361799873e91b87f5e1b75e2402201bc71e4bb",
    "m1432_review": "5b74e3eb71949d2a2c580cd72a47f0b535d5d0944d1b84f15ab6d137eeea1420",
    "m1432_manifest": "f52992b3efd4ecf9bb3bbacf7c881a3ecbc93731e2c20ad7c338d855fb2f4140",
    "m1432_outer": "de8c18d3572e2a83abe8d64f0e2fa8c3fe5a0a1333c5d6452c1b9eb2c51bfa0e",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "ucli": "c90153dfd58ff4e653852a54b31ad3b19cb8fabd993e15c21d9071b555cbebc1",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for key, value in items:
            assert key not in out, "duplicate key"
            out[key] = value
        return out
    assert path.is_file() and not path.is_symlink()
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AssertionError("nonfinite " + token)))
    assert type(value) is dict
    return value


def verify_dir(root: Path, review: str, manifest: str, outer: str) -> dict[str, Any]:
    assert root.is_dir() and not root.is_symlink()
    assert sha(root / "review.json") == review
    assert sha(root / "SHA256SUMS") == manifest
    assert sha(root / "SHA256SUMS.seal.sha256") == outer
    assert (root / "SHA256SUMS.seal.sha256").read_text().split() == [manifest, "SHA256SUMS"]
    listed = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        rel = Path(name)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert not rel.is_absolute() and ".." not in rel.parts and name not in listed
        member = root / rel
        assert member.is_file() and not member.is_symlink() and sha(member) == digest
        listed.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    assert actual == listed
    return strict_json(root / "review.json")


def require(text: str, token: str, count: int = 1) -> None:
    assert text.count(token) == count, (token, text.count(token), count)


def semantic_runner(text: str) -> None:
    """Audit semantics without relying on the final exact-byte pin."""
    main = text[text.index("def main() -> int:"):]
    require(text, 'M1440 = HW / "reviews/m1440_m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_hammer_r1_20260831"')
    require(text, '"M1432_EXPECTED_RUNNER_SHA256", "M1432_EXPECTED_AUTHORITY_SHA256",')
    require(text, '"M1432_EXPECTED_M1440_REVIEW_SHA256", "M1432_EXPECTED_M1440_MANIFEST_SHA256",')
    require(text, '"M1432_EXPECTED_M1440_OUTER_FILE_SHA256",')
    require(text, 'final.get("status") != "PASS_M1440_AUTHORIZE_ONE_M1432_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH"')
    require(text, 'final.get("authorization") != {"launch": True, "campaigns": 1,')
    require(text, '"automatic_retry": False}', 2)
    require(text, 'if final.get("bindings") != required_bindings or final.get("claim_boundary") != CLAIMS:')
    require(text, 'renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)')
    require(text, 'if actual != listed: raise Failure("sealed directory population drift")')
    require(text, 'if any(state[key] != value for key, value in exact_counts.items()):')
    require(text, '"partial_axis_citable": False')
    assert '"automatic_retry": True' not in text
    assert "shell=True" not in text and "os.system" not in text
    assert text.count("subprocess.run(") == 2
    require(main, 'state["vcs_compiles"] += 1')
    require(main, 'state["simv_runs"] += 1')
    require(main, 'state["saif_files"] += 1')
    require(main, 'state["ptpx_runs"] += 1')
    require(main, 'run([str(VCS)')
    require(main, 'run(["./simv"')
    require(main, 'run([str(PYTHON)')
    require(main, 'run([str(PT)')
    require(main, 'for case in range(5)', 2)
    assert 'CYCLES = {"k8": [51, 131, 486, 1231, 14],\n          "k1x8": [53, 133, 499, 1246, 14]}' in text
    order = [
        'verify_authority(); state["identity"] = identity(); namespaces_fresh()',
        "collision_gate()",
        "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "collision_gate()",
        "resource_gate(); namespaces_fresh()",
        'state["phase"] = "LICENSE_PREFLIGHT"',
        'subprocess.run([str(LMUTIL)',
        'state["phase"] = "ATTEMPT_CONSUME"',
        'ATTEMPT.mkdir(); state["attempt"] = True',
        'run([str(VCS)',
        'run(["./simv"',
        'state["saif_files"] != 10',
        'run([str(PT)',
        'seal_dir(STAGE); publish_no_replace(WORK, PRIVATE); publish_no_replace(STAGE, RESULT)',
    ]
    cursor = -1
    for token in order:
        cursor = main.index(token, cursor + 1)
    assert main.index("collision_gate()") < main.index("LMUTIL")
    assert main.index("collision_gate()", main.index("collision_gate()") + 1) < main.index("LMUTIL")
    assert main.index("ATTEMPT.mkdir()") < main.index("run([str(VCS)") < main.index("run([str(PT)")
    assert main.index('state["saif_files"] != 10') < main.index("run([str(PT)")
    assert 'exact_counts = {"vcs_compiles": 2, "simv_runs": 10,\n                        "saif_files": 10, "ptpx_runs": 10}' in main
    assert '"campaigns": 1, "automatic_retry": False' in main
    assert "while " not in main


def semantic_contract(value: dict[str, Any]) -> None:
    assert value["status"] == "AUTHORIZE_AT_MOST_ONE_C2_MAPPED_VCS_SAIF_PTPX_ATTEMPT__FRESH_M1440_REQUIRED"
    assert value["identity"]["mapped_activity_runner_sha256"] == SHA["runner"]
    assert value["m1362_blind"] == {
        "review_sha256": SHA["m1362_review"], "manifest_sha256": SHA["m1362_manifest"],
        "outer_file_sha256": SHA["m1362_outer"],
        "status": "PASS_M1361_EXACT_SOURCE__FINAL_LAUNCH_AUTHORITY_AUTHORING_ONLY",
        "score": 100, "attacks": 159, "false_negatives": 0}
    assert value["executor_reachability"]["consumes_fresh_m1440_final_hammer"] is True
    assert value["workload"]["axes"] == ["k8", "k1x8"]
    assert value["workload"]["cases"] == [0, 1, 2, 3, 4]
    assert value["one_shot"]["campaigns"] == 1
    assert value["one_shot"]["automatic_retry"] is False
    assert value["one_shot"]["replacement_allowed"] is False
    assert value["execution_budget"] == {
        "ordered_stages": ["mapped_vcs", "production_saif", "ptpx"],
        "mapped_vcs_compiles": 2, "simv_runs": 10, "production_saif_files": 10,
        "ptpx_runs": 10, "retry_attempts": 0, "partial_axis_publication": False,
        "ptpx_only_after_all_mapped_correctness_and_saif_gates": True}
    assert value["resource_fail_close"]["collision_gate_1_before_any_license_or_tool"] is True
    assert value["resource_fail_close"]["collision_gate_2_under_lease_before_any_license_or_tool"] is True
    assert value["receipt_contract"]["failure"]["partial_axis_citable"] is False
    assert value["final_gate"]["required_authorization"] == {
        "launch": True, "campaigns": 1, "automatic_retry": False}
    assert value["authorization"]["automatic_retry"] is False
    assert all(v is False for v in value["claim_boundary"].values())


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) >= 1, (old, text.count(old))
    return text.replace(old, new, 1)


def main() -> int:
    assert sha(RUNNER) == SHA["runner"] and sha(AUTHORITY) == SHA["authority"]
    authority_sum = Path(str(AUTHORITY) + ".sha256")
    authority_outer = Path(str(authority_sum) + ".seal.sha256")
    assert sha(authority_sum) == SHA["authority_sum"]
    assert sha(authority_outer) == SHA["authority_outer"]
    assert authority_sum.read_text() == f'{SHA["authority"]}  {AUTHORITY.name}\n'
    assert authority_outer.read_text() == f'{SHA["authority_sum"]}  {authority_sum.name}\n'
    verify_dir(M1361, SHA["m1361_review"], SHA["m1361_manifest"], SHA["m1361_outer"])
    m1362 = verify_dir(M1362, SHA["m1362_review"], SHA["m1362_manifest"], SHA["m1362_outer"])
    m1432 = verify_dir(M1432, SHA["m1432_review"], SHA["m1432_manifest"], SHA["m1432_outer"])
    assert m1362["replay"]["false_negatives"] == 0 and m1362["score"] == 100
    assert m1432["status"] == "PASS_M1432_RELEASE_AUTHORING__FRESH_M1440_REQUIRED__NO_EDA"
    assert sha(DOCS359) == SHA["docs359"] and sha(UCLI) == SHA["ucli"]

    text = RUNNER.read_text(encoding="utf-8")
    contract = strict_json(AUTHORITY)
    semantic_runner(text); semantic_contract(contract)

    source_attacks: list[tuple[str, str, str]] = [
        ("drop_verify_authority", 'verify_authority(); state["identity"] = identity(); namespaces_fresh()', 'state["identity"] = identity(); namespaces_fresh()'),
        ("drop_first_collision", '        collision_gate()\n        fcntl.flock', '        fcntl.flock'),
        ("drop_second_collision", '        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n        collision_gate(); resource_gate()', '        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n        resource_gate()'),
        ("attempt_after_vcs", 'ATTEMPT.mkdir(); state["attempt"] = True', 'state["attempt"] = True'),
        ("six_cases", 'for case in range(5):', 'for case in range(6):'),
        ("nine_saif_guard", 'state["saif_files"] != 10', 'state["saif_files"] != 9'),
        ("one_compile_receipt", 'exact_counts = {"vcs_compiles": 2, "simv_runs": 10,', 'exact_counts = {"vcs_compiles": 1, "simv_runs": 10,'),
        ("nine_ptpx_receipt", 'exact_counts = {"vcs_compiles": 2, "simv_runs": 10,\n                        "saif_files": 10, "ptpx_runs": 10}', 'exact_counts = {"vcs_compiles": 2, "simv_runs": 10,\n                        "saif_files": 10, "ptpx_runs": 9}'),
        ("partial_axis_citable", '"partial_axis_citable": False', '"partial_axis_citable": True'),
        ("replace_publication", 'publish_no_replace(STAGE, RESULT)', 'os.replace(STAGE, RESULT)'),
        ("weaken_recursive_population", 'if actual != listed: raise Failure("sealed directory population drift")', 'if not listed: raise Failure("sealed directory population drift")'),
        ("wrong_m1440_path", 'reviews/m1440_m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_hammer_r1_20260831', 'reviews/m1440_wrong'),
        ("drop_m1440_status", 'PASS_M1440_AUTHORIZE_ONE_M1432_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH', 'PASS_ANY'),
        ("allow_two_campaigns", 'final.get("authorization") != {"launch": True, "campaigns": 1,', 'final.get("authorization") != {"launch": True, "campaigns": 2,'),
        ("remove_claim_binding", ' or final.get("claim_boundary") != CLAIMS', ''),
        ("shell_execution", 'check=False)', 'check=False, shell=True)'),
        ("retry_loop", '        for axis in ("k8", "k1x8"):', '        while True:\n            for axis in ("k8", "k1x8"):'),
    ]
    false_negatives = 0
    for name, old, new in source_attacks:
        candidate = replace_once(text, old, new)
        try:
            semantic_runner(candidate)
        except (AssertionError, ValueError):
            continue
        false_negatives += 1
        raise AssertionError("source attack accepted: " + name)

    contract_attacks: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("runner_sha", lambda d: d["identity"].__setitem__("mapped_activity_runner_sha256", "0" * 64)),
        ("m1362_false_negative", lambda d: d["m1362_blind"].__setitem__("false_negatives", 1)),
        ("drop_m1440", lambda d: d["executor_reachability"].__setitem__("consumes_fresh_m1440_final_hammer", False)),
        ("add_k1", lambda d: d["workload"].__setitem__("axes", ["k8", "k1", "k1x8"])),
        ("drop_case", lambda d: d["workload"].__setitem__("cases", [0, 1, 2, 3])),
        ("two_campaigns", lambda d: d["one_shot"].__setitem__("campaigns", 2)),
        ("auto_retry", lambda d: d["one_shot"].__setitem__("automatic_retry", True)),
        ("replacement", lambda d: d["one_shot"].__setitem__("replacement_allowed", True)),
        ("one_compile", lambda d: d["execution_budget"].__setitem__("mapped_vcs_compiles", 1)),
        ("nine_simv", lambda d: d["execution_budget"].__setitem__("simv_runs", 9)),
        ("nine_saif", lambda d: d["execution_budget"].__setitem__("production_saif_files", 9)),
        ("nine_ptpx", lambda d: d["execution_budget"].__setitem__("ptpx_runs", 9)),
        ("retry_budget", lambda d: d["execution_budget"].__setitem__("retry_attempts", 1)),
        ("partial_publication", lambda d: d["execution_budget"].__setitem__("partial_axis_publication", True)),
        ("early_ptpx", lambda d: d["execution_budget"].__setitem__("ptpx_only_after_all_mapped_correctness_and_saif_gates", False)),
        ("gate1_off", lambda d: d["resource_fail_close"].__setitem__("collision_gate_1_before_any_license_or_tool", False)),
        ("gate2_off", lambda d: d["resource_fail_close"].__setitem__("collision_gate_2_under_lease_before_any_license_or_tool", False)),
        ("failure_citable", lambda d: d["receipt_contract"]["failure"].__setitem__("partial_axis_citable", True)),
        ("final_retry", lambda d: d["final_gate"]["required_authorization"].__setitem__("automatic_retry", True)),
        ("release_retry", lambda d: d["authorization"].__setitem__("automatic_retry", True)),
        ("headline_true", lambda d: d["claim_boundary"].__setitem__("headline", True)),
    ]
    for name, mutate in contract_attacks:
        candidate = copy.deepcopy(contract); mutate(candidate)
        try:
            semantic_contract(candidate)
        except AssertionError:
            continue
        false_negatives += 1
        raise AssertionError("contract attack accepted: " + name)
    assert false_negatives == 0
    output = {
        "schema": "m1440_m1432_c2_final_launch_blind_hammer_output_r1_v1",
        "status": "PASS_ZERO_FALSE_NEGATIVE_GATE",
        "source_attacks": len(source_attacks),
        "contract_attacks": len(contract_attacks),
        "attacks": len(source_attacks) + len(contract_attacks),
        "false_negatives": 0,
        "m1432_tests_replayed_before_creation": "15/15 PASS",
        "m1361_author_time_tests": "36/36 PASS (sealed evidence; lifecycle source-absent test is no longer applicable after M1362 publication)",
        "ordering": {"collision_gates_before_lmstat": 2,
                     "attempt_before_first_eda_tool": True,
                     "all_ten_saif_before_ptpx": True},
        "budget": {"campaigns": 1, "mapped_vcs_compiles": 2,
                   "simv_runs": 10, "saif_files": 10, "ptpx_runs": 10,
                   "automatic_retry": False, "partial_axis_publication": False},
        "protected": {"docs359_sha256": sha(DOCS359), "ucli_sha256": sha(UCLI)},
        "tool_runs_by_hammer": {"license_queries": 0, "vcs": 0, "simv": 0,
                                "saif": 0, "ptpx": 0, "eda": 0},
    }
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
