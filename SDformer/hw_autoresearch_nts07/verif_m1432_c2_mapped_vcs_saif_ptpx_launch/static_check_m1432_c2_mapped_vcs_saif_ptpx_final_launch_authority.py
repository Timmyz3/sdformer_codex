#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only exact checker for the inert M1432 C2 launch authority."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


HW = Path(__file__).resolve().parents[1]
CHECKER = Path(__file__).resolve()
TEST = HW / "verif_m1432_c2_mapped_vcs_saif_ptpx_launch/test_m1432_c2_mapped_vcs_saif_ptpx_final_launch_authority.py"
CONTRACT = HW / "contracts/m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_authority_r1_20260831.json"
CONTRACT_SUM = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT_SUM) + ".seal.sha256")
AUTHOR = HW / "reviews/m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_authority_author_r1_20260831"
FUTURE_HAMMER = HW / "reviews/m1440_m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_hammer_r1_20260831"

M1361_CHECKER = HW / "verif_m1361_c2_activity_final_launch_exact/static_check_m1361_c2_activity_final_launch_exact_source.py"
M1361_TEST = HW / "verif_m1361_c2_activity_final_launch_exact/test_m1361_c2_activity_final_launch_exact_source.py"
M1361_CONTRACT = HW / "contracts/m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_contract_r1_20260831.json"
M1361_CONTRACT_SUM = Path(str(M1361_CONTRACT) + ".sha256")
M1361_CONTRACT_OUTER = Path(str(M1361_CONTRACT_SUM) + ".seal.sha256")
M1361_AUTHOR = HW / "reviews/m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_author_r1_20260831"
M1362 = HW / "reviews/m1362_m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_blind_hammer_r1_20260831"

MAPPED_RUNNER = HW / "dc_handoff/scripts/run_m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_one_shot.py"
MAPPED_TB = HW / "dc_handoff/tb/tb_m1334_c2_headline_mapped_production_activity.sv"
UCLI = HW / "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"
PTPX_TCL = HW / "dc_handoff/scripts/run_ptpx.tcl"
M872 = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
TOP = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

M1361_SHA = {
    "checker": "13a98be09ec5e00d5f6ec7f07e53f27bc2d66c5d72d11b778c19e5a511422745",
    "test": "2938595d4192528e05b1aea22201f4086f35a5789756348e2d9034f35afdc8dd",
    "contract": "fb2e5f83a4befef0252a030402c2e18f8babc336e326d30f7d91d90969c00c9a",
    "contract_sum": "db556d08e4b69274589594e4d8c03bd0010309fc9cdded56cc6fd11a4799fead",
    "contract_outer": "448eca4c4e99daf81b064d7a2efbb7f0f8475f45a65098cbe1a8a8eb1e3f1cb0",
    "author_review": "d4369a78849b7f3f7411cc1c21365e17450275b01ed906468c368781b140126c",
    "author_manifest": "e00f9cfc6222c92ecd7f6b7e0ca7d0f1c46204634f208cdac3545e707e4edaaa",
    "author_outer": "634258227ac5143d820fa696ed8cb572f8c622d7b4ad8e3c0db404a0b2adbdaf",
}
M1362_SHA = {
    "review": "dafe39f181c85c1b08c7aaaaee29039005ec6a6b55386f2a2755aabca3f441b5",
    "manifest": "b546b35fbed2b0a8966b66ee34c22f0f72c93db00e5248c9808c0eda40360dd5",
    "outer": "32dae68fe7bdca213619ca19e2361799873e91b87f5e1b75e2402201bc71e4bb",
}
STATIC_SHA = {
    "runner": "314be83304d4b62cf2c4b73feb394fa2ab20e60a89afb9c3dfc07622d25a7156",
    "tb": "eacc165bad9eb3ef6c38e87f6f0de8cafd75e167f0ef02d340647634540982ca",
    "ucli": "c90153dfd58ff4e653852a54b31ad3b19cb8fabd993e15c21d9071b555cbebc1",
    "ptpx_tcl": "879398c8b8708589d42346af10d4825afac19c7c0622601685d1ea3f72245368",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AXIS_SHA = {
    "k8": {
        "mapped_netlist_sha256": "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
        "mapped_sdc_sha256": "70a0d0e7700188f5a80f31b06c2f3d401f56c7d1e2a29428e3837064a722a96c",
    },
    "k1x8": {
        "mapped_netlist_sha256": "65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc",
        "mapped_sdc_sha256": "24806d5c2d5c0afae2c01d518927e3ca96ec977d000287b0a6bc62fc42a7e317",
    },
}
BLOCKED = ["vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
           "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec",
           "common_shell_exe"]
CLAIMS = {key: False for key in (
    "functional_vcs_verified", "production_saif", "ptpx", "power", "energy",
    "performance", "system_speedup", "paper_ppa_ready", "headline")}
NAMESPACES = {
    "attempt": "results/.m1432_c2_mapped_vcs_saif_ptpx_attempt_consumed",
    "result": "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831",
    "failure": "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine",
    "private_build": "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite",
}
RECEIPT_KEYS = [
    "m1361_checker_sha256", "m1361_test_sha256", "m1361_contract_sha256",
    "m1361_contract_digest_file_sha256", "m1361_contract_outer_file_sha256",
    "m1361_author_review_sha256", "m1361_author_manifest_sha256",
    "m1361_author_outer_file_sha256", "m1362_review_sha256",
    "m1362_manifest_sha256", "m1362_outer_file_sha256",
    "m1432_authority_sha256", "m1440_review_sha256", "m1440_manifest_sha256",
    "m1440_outer_file_sha256", "mapped_runner_sha256", "ptpx_tcl_sha256"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            need(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    need(path.is_file() and not path.is_symlink(), "JSON absent/nonregular")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AssertionError("nonfinite JSON token: " + token)))
    need(type(value) is dict, "JSON root must be object")
    return value


def verify_dir(root: Path, review_sha: str, manifest_sha: str,
               outer_sha: str) -> dict[str, Any]:
    need(root.is_dir() and not root.is_symlink(), "sealed directory invalid")
    review = root / "review.json"; manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(sha(review) == review_sha and sha(manifest) == manifest_sha and
         sha(outer) == outer_sha, "sealed directory exact SHA drift")
    need(outer.read_text(encoding="ascii").split() == [manifest_sha, "SHA256SUMS"],
         "outer seal content drift")
    listed: set[str] = set()
    for row in manifest.read_text(encoding="utf-8").splitlines():
        fields = row.split(maxsplit=1); need(len(fields) == 2, "manifest field count")
        digest, name = fields; name = name.lstrip("*"); rel = Path(name)
        need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
             not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "manifest row invalid")
        member = root / rel
        need(member.is_file() and not member.is_symlink() and sha(member) == digest,
             "manifest member drift: " + name)
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "sealed directory exact population drift")
    return strict_json(review)


def load_m1361():
    need(sha(M1361_CHECKER) == M1361_SHA["checker"], "M1361 checker drift")
    spec = importlib.util.spec_from_file_location("m1432_sealed_m1361", M1361_CHECKER)
    need(spec is not None and spec.loader is not None, "cannot load M1361")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1361 = load_m1361()


def axis_path(axis: str, suffix: str) -> Path:
    return M872 / axis / "netlist" / f"{TOP}_{suffix}"


def expected_contract() -> dict[str, Any]:
    return {
        "schema": "m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_authority_r1_v1",
        "status": "AUTHORIZE_AT_MOST_ONE_C2_MAPPED_VCS_SAIF_PTPX_ATTEMPT__FRESH_M1440_REQUIRED",
        "date": "2026-08-31",
        "purpose": "Inert additive release authority for one fair K8 versus equal-bandwidth K1x8 mapped VCS to per-case SAIF to PTPX campaign, exact-bound to M1361 and its zero-false-negative M1362 blind review.",
        "identity": {
            "checker": str(CHECKER.relative_to(HW)), "checker_sha256": sha(CHECKER),
            "test": str(TEST.relative_to(HW)), "test_sha256": sha(TEST),
            "mapped_activity_runner": str(MAPPED_RUNNER.relative_to(HW)),
            "mapped_activity_runner_sha256": STATIC_SHA["runner"],
            "mapped_testbench": str(MAPPED_TB.relative_to(HW)),
            "mapped_testbench_sha256": STATIC_SHA["tb"],
            "ucli": str(UCLI.relative_to(HW)), "ucli_sha256": STATIC_SHA["ucli"],
            "ptpx_tcl": str(PTPX_TCL.relative_to(HW)),
            "ptpx_tcl_sha256": STATIC_SHA["ptpx_tcl"],
        },
        "m1361_source": {
            "checker_sha256": M1361_SHA["checker"], "test_sha256": M1361_SHA["test"],
            "contract_sha256": M1361_SHA["contract"],
            "contract_digest_file_sha256": M1361_SHA["contract_sum"],
            "contract_outer_file_sha256": M1361_SHA["contract_outer"],
            "author_review_sha256": M1361_SHA["author_review"],
            "author_manifest_sha256": M1361_SHA["author_manifest"],
            "author_outer_file_sha256": M1361_SHA["author_outer"],
            "status": "PASS_M1361_EXACT_SOURCE_AUTHOR__FRESH_M1362_BLIND_REQUIRED",
        },
        "m1362_blind": {
            "review_sha256": M1362_SHA["review"],
            "manifest_sha256": M1362_SHA["manifest"],
            "outer_file_sha256": M1362_SHA["outer"],
            "status": "PASS_M1361_EXACT_SOURCE__FINAL_LAUNCH_AUTHORITY_AUTHORING_ONLY",
            "score": 100, "attacks": 159, "false_negatives": 0,
        },
        "executor_reachability": {
            "additive_runner_not_frozen_m1344": True,
            "consumes_m1361_source": True, "consumes_m1362_blind": True,
            "consumes_m1432_authority": True, "consumes_fresh_m1440_final_hammer": True,
            "external_sha_environment_keys": [
                "M1432_EXPECTED_RUNNER_SHA256", "M1432_EXPECTED_AUTHORITY_SHA256",
                "M1432_EXPECTED_M1440_REVIEW_SHA256",
                "M1432_EXPECTED_M1440_MANIFEST_SHA256",
                "M1432_EXPECTED_M1440_OUTER_FILE_SHA256"],
            "old_m1345_m1346_m1347_paths_forbidden": True,
            "mapped_vcs_saif_ptpx_all_reachable_in_one_executor": True,
        },
        "workload": {
            "axes": ["k8", "k1x8"], "diagnostic_k1_excluded": True,
            "cases": [0, 1, 2, 3, 4], "clock_period_ns": 3.0,
            "k8_cycles": [51, 131, 486, 1231, 14],
            "k1x8_cycles": [53, 133, 499, 1246, 14],
            "mapped_artifacts": AXIS_SHA,
            "same_work_and_memory_service": True,
        },
        "one_shot": {
            "namespaces": dict(NAMESPACES), "all_absent_at_authoring": True,
            "attempt_publish": "O_EXCL_OR_RENAME_NOREPLACE",
            "attempt_consumed_before_first_eda_tool": True,
            "campaigns": 1, "automatic_retry": False, "replacement_allowed": False,
        },
        "execution_budget": {
            "ordered_stages": ["mapped_vcs", "production_saif", "ptpx"],
            "mapped_vcs_compiles": 2, "simv_runs": 10,
            "production_saif_files": 10, "ptpx_runs": 10,
            "retry_attempts": 0, "partial_axis_publication": False,
            "ptpx_only_after_all_mapped_correctness_and_saif_gates": True,
        },
        "resource_fail_close": {
            "same_uid_blocked_processes": list(BLOCKED),
            "collision_gate_1_before_any_license_or_tool": True,
            "collision_gate_2_under_lease_before_any_license_or_tool": True,
            "memory_and_commit_headroom_before_attempt": True,
            "namespace_residue_before_attempt": "reject",
            "any_gate_failure_consumes_no_attempt_and_authorizes_no_retry": True,
        },
        "measurement_gates": {
            "numeric_mismatches": 0, "tuple_mismatches": 0,
            "accepted_transaction_unknowns": 0, "protocol_errors": 0,
            "assertion_failures": 0, "saif_scope": "mapped_DUT_only",
            "reset_and_idle_excluded": True,
            "duration_equals_measured_cycles_times_3ns": True,
            "saif_tx_entries": 0, "check_power_before_update_power": True,
            "energy_formula": "E_case_pJ=P_case_mW*duration_ns",
            "aggregate": "sum_same_five_case_energy_per_axis",
        },
        "receipt_contract": {
            "paths": ["attempt", "failure", "success"],
            "identity_sha_keys_each": list(RECEIPT_KEYS),
            "identity_key_order_exact": True, "active_value_expressions_exact": True,
            "attempt": {"status": "M1432_ATTEMPT_CONSUMED", "campaigns": 1,
                        "automatic_retry": False, "budget": {"vcs_compiles": 2,
                        "simv_runs": 10, "saif_files": 10, "ptpx_runs": 10}},
            "failure": {"status": "FAILED_OR_INCOMPLETE", "canonical_result": False,
                        "partial_axis_citable": False, "automatic_retry": False},
            "success": {"status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
                        "attempt_consumed": True, "axes": ["k8", "k1x8"],
                        "cases_per_axis": 5, "automatic_retry": False,
                        "claim_boundary_exact": True},
        },
        "final_gate": {
            "path": str(FUTURE_HAMMER.relative_to(HW)),
            "must_be_absent_at_authoring": True, "fresh_different_author": True,
            "required_status": "PASS_M1440_AUTHORIZE_ONE_M1432_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH",
            "required_authorization": {"launch": True, "campaigns": 1,
                                       "automatic_retry": False},
            "actual_launch_ready": False,
        },
        "authorization": {
            "release_authority": True, "final_hammer_authoring": True,
            "launch_now": False, "license_query_now": False, "vcs_now": False,
            "simv_now": False, "saif_now": False, "ptpx_now": False,
            "eda_now": False, "automatic_retry": False,
        },
        "claim_boundary": dict(CLAIMS),
        "protected_files": {
            "docs359": {"path": str(DOCS359.relative_to(HW)),
                        "sha256": STATIC_SHA["docs359"]},
            "ucli": {"path": str(UCLI.relative_to(HW)),
                     "sha256": STATIC_SHA["ucli"]},
        },
    }


def verify_upstream() -> dict[str, Any]:
    fixed = [(M1361_TEST, M1361_SHA["test"]), (M1361_CONTRACT, M1361_SHA["contract"]),
             (M1361_CONTRACT_SUM, M1361_SHA["contract_sum"]),
             (M1361_CONTRACT_OUTER, M1361_SHA["contract_outer"]),
             (MAPPED_RUNNER, STATIC_SHA["runner"]), (MAPPED_TB, STATIC_SHA["tb"]),
             (UCLI, STATIC_SHA["ucli"]), (PTPX_TCL, STATIC_SHA["ptpx_tcl"]),
             (DOCS359, STATIC_SHA["docs359"])]
    for path, digest in fixed:
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed input drift: " + str(path))
    need(M1361_CONTRACT_SUM.read_text(encoding="ascii") ==
         f'{M1361_SHA["contract"]}  {M1361_CONTRACT.name}\n', "M1361 digest content")
    need(M1361_CONTRACT_OUTER.read_text(encoding="ascii") ==
         f'{M1361_SHA["contract_sum"]}  {M1361_CONTRACT_SUM.name}\n', "M1361 outer content")
    m1361_author = verify_dir(M1361_AUTHOR, M1361_SHA["author_review"],
                              M1361_SHA["author_manifest"], M1361_SHA["author_outer"])
    m1362 = verify_dir(M1362, M1362_SHA["review"], M1362_SHA["manifest"], M1362_SHA["outer"])
    need(m1361_author.get("status") == expected_contract()["m1361_source"]["status"],
         "M1361 author status drift")
    need(m1362.get("status") == expected_contract()["m1362_blind"]["status"] and
         m1362.get("score") == 100 and m1362.get("replay", {}).get("false_negatives") == 0 and
         m1362.get("authorization") == {
             "final_launch_authority_authoring": True, "launch": False,
             "license_query": False, "vcs": False, "simv": False,
             "saif": False, "ptpx": False, "eda": False,
             "automatic_retry": False} and m1362.get("claim_boundary") == CLAIMS,
         "M1362 verdict/authorization drift")
    common = M1361.validate_common(skip_author=False)
    need(common["m1357_false_negatives_repaired"] == 30 and
         common["launch_authorized"] is False, "M1361 inherited source drift")
    for axis, digests in AXIS_SHA.items():
        netlist = axis_path(axis, "mapped.v"); sdc = axis_path(axis, "mapped.sdc")
        need(sha(netlist) == digests["mapped_netlist_sha256"] and
             sha(sdc) == digests["mapped_sdc_sha256"], "mapped artifact drift: " + axis)
    return {"m1361_source": True, "m1362_zero_false_negative": True,
            "mapped_axes": 2, "protected": True}


def validate_runner_source() -> dict[str, Any]:
    text = MAPPED_RUNNER.read_text(encoding="utf-8")
    for required in ("M1361_CONTRACT", "M1362 =", "AUTHORITY =", "M1440 =",
                     "M1432_EXPECTED_RUNNER_SHA256", "M1432_EXPECTED_AUTHORITY_SHA256",
                     "M1432_EXPECTED_M1440_REVIEW_SHA256", "mapped_vcs_saif_ptpx",
                     "sub" + "process.run", "PTPX_TCL",
                     "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER"):
        need(required in text, "runner reachability token absent: " + required)
    for forbidden in ("m1345_m1344", "m1346_m1344", "m1347_m1346"):
        need(forbidden not in text, "runner retains unreachable frozen path: " + forbidden)
    body = text[text.index("def main()") :]
    first = body.index("collision_gate()")
    second = body.index("collision_gate()", first + 1)
    license_call = body.index("LMUTIL")
    vcs_call = body.index("run([str(VCS)")
    pt_call = body.index("run([str(PT)")
    need(first < second < license_call < vcs_call < pt_call,
         "runner collision/license/VCS/PTPX ordering drift")
    all_saif_guard = body.index('state["saif_files"] != 10')
    need(vcs_call < all_saif_guard < pt_call,
         "PTPX may start before the complete mapped VCS/SAIF campaign")
    need(body.count("state[\"vcs_compiles\"] += 1") == 1 and
         body.count("state[\"simv_runs\"] += 1") == 1 and
         body.count("state[\"saif_files\"] += 1") == 1 and
         body.count("state[\"ptpx_runs\"] += 1") == 1,
         "runner count-accounting cardinality drift")
    need('for axis in ("k8", "k1x8")' in body and "for case in range(5)" in body,
         "runner fair-axis/case loop drift")
    return {"exact_sha": sha(MAPPED_RUNNER), "two_collision_gates_before_license": True,
            "stages_reachable": ["mapped_vcs", "production_saif", "ptpx"],
            "frozen_m1344_future_paths": False}


def validate_contract(skip_author: bool = False) -> dict[str, Any]:
    contract = strict_json(CONTRACT)
    need(contract == expected_contract(), "M1432 contract exact-set/value drift")
    need(CONTRACT_SUM.is_file() and not CONTRACT_SUM.is_symlink() and
         CONTRACT_OUTER.is_file() and not CONTRACT_OUTER.is_symlink(),
         "M1432 authority sidecar absent/nonregular")
    need(CONTRACT_SUM.read_text(encoding="ascii") == f"{sha(CONTRACT)}  {CONTRACT.name}\n" and
         CONTRACT_OUTER.read_text(encoding="ascii") ==
         f"{sha(CONTRACT_SUM)}  {CONTRACT_SUM.name}\n", "M1432 authority sidecar drift")
    if not skip_author:
        review = verify_dir(AUTHOR, sha(AUTHOR / "review.json"),
                            sha(AUTHOR / "SHA256SUMS"),
                            sha(AUTHOR / "SHA256SUMS.seal.sha256"))
        need(review.get("status") == "PASS_M1432_RELEASE_AUTHORING__FRESH_M1440_REQUIRED__NO_EDA" and
             review.get("bindings") == {"checker_sha256": sha(CHECKER),
                                         "test_sha256": sha(TEST),
                                         "contract_sha256": sha(CONTRACT),
                                         "m1362_review_sha256": M1362_SHA["review"],
                                         "m1362_manifest_sha256": M1362_SHA["manifest"],
                                         "m1362_outer_file_sha256": M1362_SHA["outer"]} and
             review.get("authorization") == expected_contract()["authorization"] and
             review.get("claim_boundary") == CLAIMS,
             "M1432 author seal drift")
    return contract


def validate_absence(paths: dict[str, Path] | None = None) -> dict[str, Any]:
    concrete = ({key: HW / value for key, value in NAMESPACES.items()} if paths is None else paths)
    need(set(concrete) == set(NAMESPACES), "namespace key set drift")
    for key, path in concrete.items():
        need(not os.path.lexists(str(path)), "namespace residue: " + key)
    need(not os.path.lexists(str(FUTURE_HAMMER)), "future M1440 residue")
    return {"fresh_namespaces": len(concrete), "future_m1440_absent": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent",), required=True)
    parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args()
    upstream = verify_upstream(); runner = validate_runner_source()
    contract = validate_contract(args.skip_author)
    absence = validate_absence()
    print(json.dumps({
        "schema": "m1432_c2_mapped_vcs_saif_ptpx_final_launch_authority_check_r1_v1",
        "status": "PASS_M1432_RELEASE_SOURCE__FRESH_M1440_REQUIRED__NO_EDA",
        "upstream": upstream, "runner": runner, "absence": absence,
        "budget": contract["execution_budget"],
        "authorization": contract["authorization"],
        "claim_boundary": contract["claim_boundary"],
        "license_queries": 0, "vcs_runs": 0, "simv_runs": 0,
        "saif_runs": 0, "ptpx_runs": 0, "eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
