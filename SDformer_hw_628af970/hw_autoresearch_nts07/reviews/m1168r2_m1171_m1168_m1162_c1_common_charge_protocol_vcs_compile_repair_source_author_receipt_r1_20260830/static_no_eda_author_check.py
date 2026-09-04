#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author-side fail-closed check for M1168R2; never invokes VCS/EDA."""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_contract_r1_20260830.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1168r2_m1162_c1_common_charge_protocol_exact_sha_r2.sh"
STATIC = HW / "verif_m1168r2_c1_common_charge_protocol/static_check_m1168r2_m1162_vcs_source.py"
TB = HW / "verif_m1168r2_c1_common_charge_protocol/tb_m1168r2_m1162_common_charge_protocol_unit_delay_r2.sv"
SVA = HW / "verif_m1168r2_c1_common_charge_protocol/m1168r2_m1162_common_charge_protocol_assertions_r2.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
R1_QUARANTINE = HW / "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830.failed_or_incomplete.3074649.quarantine"
R1_ATTEMPT_ID = HW / "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed/identity.txt"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
R2_ATTEMPT = HW / "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed"
R2_RESULT = HW / "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830"

EXPECTED = {
    CONTRACT: "7abf99b60fce68ee0823b0e087f3276dccbc33b4d6921c5e6fe34bf3e16abe21",
    Path(str(CONTRACT) + ".sha256"): "d6f0e14eaf2a23a7369a86b9783b194b05c67cd9dbd5dfa2bb0ad5fe30e6c9f4",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "06c134e50fec169fd5609956fdc723d9ddfe9297ec132b5a4e29869bf0692d44",
    RUNNER: "4a661d50ca1929968b31258dd4950945bdd792311c090389f6a882e52aba58c3",
    STATIC: "022cf2d61d29cb22547db78de3dc8f5dbbbc8e0b03443c7469abd4f56d6beae8",
    TB: "bd5a2c3ce1ab9f03a7017756c96d5013577116583fc7d007ef3374593272ee35",
    SVA: "59ff9141175159e9043d86dd5932a4113fde88582005487f1eb65e372c6a684f",
    FILELIST: "96331eb20fb6d4e72e157d23c579841a121103053ed6246f0b76f812399f1411",
    R1_ATTEMPT_ID: "7b624fd913046f028506594e1b354bbb76c777a7c6467e1652c178fc7e05faae",
    R1_QUARANTINE / "compile.log": "39765d45f5e53de02a4c9139915253b0d0d8190f042027b70344dea08b0037ff",
    R1_QUARANTINE / "SHA256SUMS": "6f7d480bc752ea5835c3442de72f8e5e484ae41db3a5377b49e593e13838614c",
    R1_QUARANTINE / "SHA256SUMS.seal.sha256": "72ec416eb80888bb5c30a448c870b0859912097d43564662a3a88953182316c7",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks = 0
mutations = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular(path: Path, digest: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and sha(path) == digest,
            "identity drift: " + str(path))


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON " + token)))


def validate_contract(data) -> None:
    require(data["status"] == "SOURCE_READY_FOR_FRESH_M1172_HAMMER__NO_VCS_RELEASE", "status")
    require(data["identity"]["runner_sha256"] == EXPECTED[RUNNER], "runner identity")
    f = data["r1_failure_forensics"]
    require(f["attempt_reusable"] is False and f["dtinpcil_errors"] == 5 and
            f["irfpca_autovar_errors"] == 5 and f["simulation_started"] is False,
            "failure forensics")
    repair = data["repair"]
    require(repair["module_scope_stage_fields"] == 5 and
            repair["automatic_formals_on_force_rhs"] == 0 and
            repair["hierarchical_dut_force_statements"] == 10 and
            repair["all_stage_fields_assigned_before_force"] is True and
            repair["force_target_preserved"] is True and
            repair["lrm_compile_mode_mutations_rejected"] == 5,
            "repair semantics")
    preserved = data["preserved_verification"]
    require(preserved["assert_properties"] == 16 and preserved["cover_properties"] == 6 and
            preserved["protocol_attacks"] == 7 and
            preserved["service_assumption_attacks"] == 2 and
            preserved["deterministic_random_transactions"] == 24,
            "verification preservation")
    unique = data["unique_r2_attempt"]
    require(unique["attempt_path"] == "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed" and
            unique["r1_namespace_reuse"] is False and unique["automatic_retry"] is False,
            "new namespace")
    future = data["future_gates"]
    require(future["fresh_different_author_hammer_required"] is True and
            future["separate_release_required_after_hammer"] is True and
            future["direct_r2_execution_now"] is False,
            "future hammer/release gate")
    require(data["authorization"] == {
        "source_authoring": True, "vcs_compiles_now": 0, "simv_runs_now": 0,
        "all_eda_runs_now": 0,
        "fresh_hammer_may_author_separate_release_after_pass": True,
    }, "source-only authorization")
    for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                "ppa", "power", "energy", "system_speedup", "paper_citable", "headline"):
        require(data["claim_boundary"][key] is False, "claim opened: " + key)


def reject(base, mutate) -> None:
    global mutations
    trial = copy.deepcopy(base)
    mutate(trial)
    try:
        validate_contract(trial)
    except (KeyError, RuntimeError, TypeError):
        mutations += 1
        return
    raise RuntimeError("semantic mutation accepted")


def main() -> None:
    for path, digest in EXPECTED.items():
        regular(path, digest)
    require((CONTRACT.with_name(CONTRACT.name + ".sha256")).read_text().split() ==
            [EXPECTED[CONTRACT], CONTRACT.name], "contract sidecar content")
    require((CONTRACT.with_name(CONTRACT.name + ".sha256.seal.sha256")).read_text().split() ==
            [EXPECTED[Path(str(CONTRACT) + ".sha256")], CONTRACT.name + ".sha256"],
            "contract outer content")
    subprocess.check_call(["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-B", "-I", str(STATIC)],
                          stdout=subprocess.DEVNULL)
    contract = strict_json(CONTRACT)
    validate_contract(contract)
    for mutator in (
        lambda d: d.__setitem__("status", "READY"),
        lambda d: d["r1_failure_forensics"].__setitem__("attempt_reusable", True),
        lambda d: d["repair"].__setitem__("automatic_formals_on_force_rhs", 1),
        lambda d: d["repair"].__setitem__("force_target_preserved", False),
        lambda d: d["preserved_verification"].__setitem__("assert_properties", 15),
        lambda d: d["unique_r2_attempt"].__setitem__("r1_namespace_reuse", True),
        lambda d: d["future_gates"].__setitem__("fresh_different_author_hammer_required", False),
        lambda d: d["future_gates"].__setitem__("direct_r2_execution_now", True),
        lambda d: d["authorization"].__setitem__("vcs_compiles_now", 1),
        lambda d: d["claim_boundary"].__setitem__("functional_vcs_verified", True),
        lambda d: d["claim_boundary"].__setitem__("paper_citable", True),
    ):
        reject(contract, mutator)
    runner = RUNNER.read_text()
    for token in (
        "M1168R2_EXPECTED_RELEASE_SHA256", "M1168R2_EXPECTED_HAMMER_REVIEW_SHA256",
        "M1168R2_EXPECTED_HAMMER_OUTER_SHA256", "verify_recursive_seal \"${R1_QUARANTINE}\"",
        "sha_exact 7b624fd913046f028506594e1b354bbb76c777a7c6467e1652c178fc7e05faae \"${R1_ATTEMPT_ID}\"",
        ".m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed",
        "+define+UNIT_DELAY", "SOURCE_READY_FOR_FRESH_M1172_HAMMER__NO_VCS_RELEASE",
        "PASS_M1172_M1168R2_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE",
        "AUTHORIZE_EXACTLY_ONE_M1168R2_FUNCTIONAL_VCS_ATTEMPT",
    ):
        require(token in runner, "runner gate absent: " + token)
    require(runner.count('"${VCS_BIN}" -full64') == 1 and runner.count('./simv -no_save') == 1,
            "future one-shot cardinality")
    require(".m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed" in runner and
            "ATTEMPT=\"${HW_ROOT}/results/.m1168r2_" in runner,
            "r1 forensics/r2 namespace distinction")
    require(not os.path.lexists(R2_ATTEMPT) and not os.path.lexists(R2_RESULT),
            "r2 attempt/result already exists")
    require(not glob.glob(str(HW / "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_work.*")),
            "r2 work already exists")
    print(json.dumps({
        "status": "PASS_M1168R2_SOURCE_REPAIR_AUTHOR_CHECK__FRESH_M1172_HAMMER_AND_M1173_RELEASE_REQUIRED__NO_VCS_NO_EDA",
        "checks_passed": checks,
        "contract_mutations_rejected": mutations,
        "lrm_force_mutations_rejected": 5,
        "assertions": 16,
        "covers": 6,
        "r1_attempt_reused": False,
        "r2_namespace_fresh": True,
        "runner_invocations": 0,
        "vcs_compiles": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
