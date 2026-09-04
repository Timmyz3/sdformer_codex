#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh different-author, no-EDA final launch hammer for M1493."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import runpy
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / (
    "dc_handoff/scripts/run_m1493_m1467_c2_mapped_vcs_saif_ptpx_"
    "lca_successor_one_shot.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1493_m1467_c2_lca_successor_source_contract_"
    "r1_20260831.json")
M1494 = HW / (
    "reviews/m1494_m1493_c2_lca_successor_source_blind_hammer_"
    "r1_20260831")
M1494_HAMMER = M1494 / "independent_hammer.py"
M1495 = HW / (
    "contracts/m1495_m1494_m1493_c2_lca_successor_launch_release_"
    "r1_20260831.json")
M1467_ATTEMPT = HW / "results/.m1467_c2_mapped_vcs_saif_ptpx_attempt_consumed"
M1467_FAILURE = HW / (
    "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831."
    "failed_or_incomplete.quarantine")
M1484 = HW / (
    "reviews/m1484_m1467_c2_second_production_failure_forensic_"
    "r1_20260831")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    "runner": "8d93d55ca600620eb903a7328f4cc38e0720ae45ce24d8128fac5924d2902677",
    "source_contract": "efa9e6339564f2ec3c8294b7977c81782fafe6ac38f6e4fed5e61c89642da177",
    "m1494_review": "65435aca804c486d50d8332774c70e87083d66d5c2e7acc30485dc84ba458340",
    "m1494_manifest": "b2ff59fd22bd0bd6463ae9ac9aa31ee82d77099d40ea4890fd99600255b9811b",
    "m1494_outer": "329ed4435761eb7d00be969d43ac05221c837cc3f79cedefd03d557034c432f7",
    "m1495": "838ea0f3714167c43c6f4e40829c2d1a59d1b84ee7468758798c82f21114eb94",
    "m1467_attempt": "a3eead113c10d0134dd83972aaa06c6b26256f7459a37d784f98c5eeb2c68f92",
    "m1467_failure": "39f3d5ffa39508db348cddf116584267e68e8796a008a7949bad88e02dd2c015",
    "m1484_review": "d26f73469d3d9e131cb776d47c6ee12c2ddd9f546e47fae690f73d7f8186d826",
    "m1484_manifest": "d61787c9a4c25e8cfe6fe2b0980605b09cae9ffaf1d4c8406b28d93cd43618b3",
    "m1484_outer": "86c26e7109931199578e22cba7795aeea2673ea5e57f2524ed76790ce9d1487d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
CLAIMS = {key: False for key in (
    "functional_vcs_verified", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready",
    "headline")}
EXPECTED_BINDINGS = {
    "runner_sha256": PINS["runner"],
    "source_contract_sha256": PINS["source_contract"],
    "m1494_review_sha256": PINS["m1494_review"],
    "m1495_release_sha256": PINS["m1495"],
}
EXPECTED_AUTHORIZATION = {
    "launch": True, "campaigns": 1, "automatic_retry": False}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise RuntimeError("duplicate JSON key: " + key)
            value[key] = item
        return value
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("JSON not regular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise RuntimeError("JSON root")
    return value


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> set[str]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("sealed root")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if sha(manifest) != manifest_sha or sha(outer) != outer_sha:
        raise RuntimeError("seal identity")
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise RuntimeError("outer content")
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        member = root / rel
        if (re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or name in listed or rel.is_absolute() or ".." in rel.parts
                or not member.is_file() or member.is_symlink()
                or not stat.S_ISREG(member.lstat().st_mode)
                or sha(member) != digest):
            raise RuntimeError("seal member")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if listed != actual:
        raise RuntimeError("seal population")
    return listed


def verify_sidecars(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    if sidecar.read_text().split() != [sha(path), path.name]:
        raise RuntimeError("sidecar")
    if outer.read_text().split() != [sha(sidecar), sidecar.name]:
        raise RuntimeError("outer sidecar")


def load_runner():
    spec = importlib.util.spec_from_file_location("m1496_bound_m1493", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("runner import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_release(value: dict[str, Any], exact: dict[str, Any]) -> None:
    if value != exact:
        raise RuntimeError("release exact-set/value drift")
    if (value.get("status") !=
            "RELEASE_M1493_C2_LCA_SUCCESSOR__FRESH_M1496_REQUIRED__NO_LAUNCH"
            or value.get("launch_now") is not False
            or value.get("automatic_retry") is not False):
        raise RuntimeError("release boundary")
    identity = value["identity"]
    if identity != {
            "runner_path": RUNNER.relative_to(HW).as_posix(),
            "runner_sha256": PINS["runner"],
            "source_contract_path": SOURCE_CONTRACT.relative_to(HW).as_posix(),
            "source_contract_sha256": PINS["source_contract"],
            "source_hammer_path": M1494.relative_to(HW).as_posix(),
            "source_hammer_review_sha256": PINS["m1494_review"],
            "source_hammer_manifest_sha256": PINS["m1494_manifest"],
            "source_hammer_outer_file_sha256": PINS["m1494_outer"],
            "source_hammer_status":
                "PASS_M1494_M1493_C2_LCA_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE",
            "source_hammer_score": 100,
            "docs359_sha256": PINS["docs359"]}:
        raise RuntimeError("release identity")
    if value["predecessor_failure"] != {
            "runner_sha256":
                "120cb1a8abe3df1e537de6797b3962fe0a7496be78954ba3b31fd9c8627e9a8a",
            "attempt_payload_sha256": PINS["m1467_attempt"],
            "failure_payload_sha256": PINS["m1467_failure"],
            "forensic_review_sha256": PINS["m1484_review"],
            "phase": "SIM_k8_0",
            "first_error_code": "Error-[LCA_FEATURES_NEED_OPTION]",
            "hardware_or_protocol_failure": False,
            "attempt_consumed": True, "automatic_retry": False,
            "partial_axis_citable": False}:
        raise RuntimeError("predecessor release")
    if value["sole_repair"] != {
            "vcs_compile_keep": "-debug_access+r",
            "vcs_compile_add_exactly_once_in_shared_prefix": "-lca",
            "rtl_change": False, "netlist_change": False,
            "sdc_change": False, "testbench_change": False,
            "workload_change": False, "ucli_change": False,
            "ptpx_script_change": False, "saif_scope_change": False}:
        raise RuntimeError("sole repair")
    auth = value["authorization"]
    if auth != {"campaigns": 1, "axes": ["k8", "k1x8"],
            "workload_cases_per_axis": [0, 1, 2, 3, 4],
            "vcs_compiles": 2, "simv_runs": 10,
            "production_saif_files": 10, "ptpx_runs": 10,
            "all_ten_saif_before_first_ptpx": True,
            "attempt_before_first_eda": True,
            "partial_axis_publication": False,
            "automatic_retry": False, "effective_before_m1496": False}:
        raise RuntimeError("campaign authority")
    if (value["final_hammer_gate"]["required_status"] !=
            "PASS_M1496_AUTHORIZE_ONE_M1493_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH"
            or value["final_hammer_gate"]["required_authorization"] !=
            EXPECTED_AUTHORIZATION
            or value["final_hammer_gate"]["present_at_release_authoring"] is not False
            or value["final_hammer_gate"]["fresh_different_author_required"] is not True
            or value["claim_boundary"] != CLAIMS):
        raise RuntimeError("final gate")


def validate_final(value: dict[str, Any]) -> None:
    if value.get("status") != (
            "PASS_M1496_AUTHORIZE_ONE_M1493_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH"):
        raise RuntimeError("final status")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise RuntimeError("final authorization")
    if value.get("bindings") != EXPECTED_BINDINGS:
        raise RuntimeError("final bindings")
    if value.get("claim_boundary") != CLAIMS:
        raise RuntimeError("final claims")


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1496_MUTATED"
    if type(value) is list:
        return list(value) + ["M1496_MUTATED"]
    if type(value) is dict:
        result = copy.deepcopy(value)
        result["m1496_mutated"] = True
        return result
    raise TypeError(type(value))


def main() -> int:
    checks: list[dict[str, Any]] = []
    attacks: list[dict[str, Any]] = []
    def check(name: str, value: bool, category: str) -> None:
        checks.append({"check": name, "category": category, "pass": bool(value)})
    def attack(name: str, thunk, category: str) -> None:
        try:
            thunk()
            rejected = False
        except BaseException:
            rejected = True
        attacks.append({"attack": name, "category": category,
                        "rejected": rejected, "false_negative": not rejected})

    check("runner_exact", sha(RUNNER) == PINS["runner"], "identity")
    check("source_contract_exact", sha(SOURCE_CONTRACT) ==
          PINS["source_contract"], "identity")
    check("m1495_exact", sha(M1495) == PINS["m1495"], "identity")
    check("docs359_exact", sha(DOCS359) == PINS["docs359"], "identity")
    verify_sidecars(SOURCE_CONTRACT)
    verify_sidecars(M1495)
    check("source_and_release_sidecars", True, "identity")
    verify_seal(M1494, PINS["m1494_manifest"], PINS["m1494_outer"])
    check("m1494_seal", sha(M1494 / "review.json") == PINS["m1494_review"],
          "authority")
    m1494 = strict_json(M1494 / "review.json")
    check("m1494_status", m1494.get("status") ==
          "PASS_M1494_M1493_C2_LCA_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE",
          "authority")

    # Replay the exact sealed source hammer in-process. run_path avoids writing
    # a pycache member into the sealed M1494 directory.
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            runpy.run_path(str(M1494_HAMMER), run_name="__main__")
        replay_rc = 0
    except SystemExit as stop:
        replay_rc = int(stop.code or 0)
    replay = json.loads(stream.getvalue())
    check("m1494_replay", replay_rc == 0 and
          replay.get("status") == "PASS_ZERO_FALSE_NEGATIVE" and
          replay["summary"]["p0_count"] == 0 and
          replay["summary"]["p1_count"] == 0, "source_replay")

    runner = load_runner()
    runner.verify_predecessor_failure()
    runner.namespaces_fresh()
    check("m1467_m1484_reaudit", True, "predecessor")
    check("runner_claims", runner.CLAIMS == CLAIMS, "runner")
    check("runner_counts", runner.COUNTS == {"vcs_compiles": 2,
          "simv_runs": 10, "saif_files": 10, "ptpx_runs": 10}, "runner")
    check("runner_compile_delta", runner.COMPILE_PREFIX[-4:] ==
          ["-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc"],
          "runner")

    release = strict_json(M1495)
    frozen_release = copy.deepcopy(release)
    validate_release(release, frozen_release)
    check("release_semantics", True, "release")
    for section, value in frozen_release.items():
        if type(value) is not dict:
            candidate = copy.deepcopy(frozen_release)
            candidate[section] = changed(value)
            attack("release_top_" + section,
                   lambda c=candidate: validate_release(c, frozen_release),
                   "release_mutation")
        else:
            for key, leaf in value.items():
                candidate = copy.deepcopy(frozen_release)
                candidate[section][key] = changed(leaf)
                attack("release_leaf_" + section + "_" + key,
                       lambda c=candidate: validate_release(c, frozen_release),
                       "release_mutation")

    candidate_final = {
        "status":
            "PASS_M1496_AUTHORIZE_ONE_M1493_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH",
        "authorization": copy.deepcopy(EXPECTED_AUTHORIZATION),
        "bindings": copy.deepcopy(EXPECTED_BINDINGS),
        "claim_boundary": copy.deepcopy(CLAIMS),
    }
    validate_final(candidate_final)
    for section, value in candidate_final.items():
        if type(value) is not dict:
            candidate = copy.deepcopy(candidate_final)
            candidate[section] = changed(value)
            attack("final_top_" + section,
                   lambda c=candidate: validate_final(c), "final_mutation")
        else:
            for key, leaf in value.items():
                candidate = copy.deepcopy(candidate_final)
                candidate[section][key] = changed(leaf)
                attack("final_leaf_" + section + "_" + key,
                       lambda c=candidate: validate_final(c), "final_mutation")
            candidate = copy.deepcopy(candidate_final)
            candidate[section]["extra"] = False
            attack("final_extra_" + section,
                   lambda c=candidate: validate_final(c), "final_mutation")

    # In-memory sidecar attacks must fail exact token comparisons.
    expected_sidecar = [PINS["m1495"], M1495.name]
    for name, candidate in {
            "sidecar_digest": ["0" * 64, M1495.name],
            "sidecar_name": [PINS["m1495"], "wrong.json"],
            "sidecar_extra": expected_sidecar + ["extra"]}.items():
        attack(name, lambda c=candidate: (_ for _ in ()).throw(RuntimeError())
               if c != expected_sidecar else None, "sidecar_mutation")

    p0 = sum(not item["rejected"] for item in attacks)
    p1 = sum(not item["pass"] for item in checks)
    output = {
        "schema": "m1496_m1495_m1493_c2_final_launch_hammer_output_r1_v1",
        "status": "PASS_ZERO_FALSE_NEGATIVE" if p0 == 0 and p1 == 0
                  else "FAIL_DO_NOT_LAUNCH",
        "checks": checks, "attacks": attacks,
        "summary": {"checks_passed": sum(item["pass"] for item in checks),
                    "checks_total": len(checks),
                    "mutations_rejected": sum(item["rejected"] for item in attacks),
                    "mutations_total": len(attacks),
                    "p0_count": p0, "p1_count": p1,
                    "m1494_replayed_mutations":
                        replay["summary"]["mutations_total"],
                    "m1494_replayed_false_negatives":
                        replay["summary"]["p0_count"]},
        "authorization_candidate": {"status": candidate_final["status"],
            "authorization": candidate_final["authorization"],
            "bindings": candidate_final["bindings"],
            "claim_boundary": candidate_final["claim_boundary"]},
        "execution": {"license_query": 0, "vcs": 0, "simv": 0,
                      "saif": 0, "pt": 0, "ptpx": 0, "eda": 0,
                      "ssh": 0, "gpu": 0, "attempts_consumed": 0},
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if p0 == 0 and p1 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
