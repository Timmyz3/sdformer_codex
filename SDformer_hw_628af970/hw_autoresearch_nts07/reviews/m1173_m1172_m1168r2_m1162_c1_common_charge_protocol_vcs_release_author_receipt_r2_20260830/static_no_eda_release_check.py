#!/usr/bin/env python3
"""Author check for the inert M1173/M1168R2 VCS release; no EDA."""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path("contracts/m1173_m1172_m1168r2_m1162_c1_common_charge_protocol_vcs_launch_release_r2_20260830.json")
SOURCE_CONTRACT = Path("contracts/m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_contract_r1_20260830.json")
RUNNER = Path("dc_handoff/scripts/run_vcs_m1168r2_m1162_c1_common_charge_protocol_exact_sha_r2.sh")
AUTHOR = Path("reviews/m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_author_receipt_r1_20260830")
HAMMER = Path("reviews/m1172_m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_hammer_r1_20260830")
R1_ATTEMPT = Path("results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed")
R1_QUARANTINE = Path("results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830.failed_or_incomplete.3074649.quarantine")
R2_ATTEMPT = Path("results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed")
R2_RESULT = Path("results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830")
R2_WORK_GLOB = "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_work.*"
R2_QUARANTINE_GLOB = "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830.failed_or_incomplete.*.quarantine"
FUTURE_HAMMER = Path("reviews/m1174_m1173_m1172_m1168r2_m1162_c1_vcs_launch_release_hammer_r2_20260830")
DOCS359 = Path("docs/359_DATE终局冻结_20260813.md")
MIN_MEM_KIB = 67_108_864

EXPECTED = {
    RELEASE: "31302e769ac6e3938df0b37adf336f5cca20cbdfe92e2b95b16a37979d52f52f",
    Path(str(RELEASE) + ".sha256"): "c4e3a638b4b47659ee466823067b862299f6a04c85bab53f5621a7c24fa8e031",
    SOURCE_CONTRACT: "7abf99b60fce68ee0823b0e087f3276dccbc33b4d6921c5e6fe34bf3e16abe21",
    Path(str(SOURCE_CONTRACT) + ".sha256"): "d6f0e14eaf2a23a7369a86b9783b194b05c67cd9dbd5dfa2bb0ad5fe30e6c9f4",
    Path(str(SOURCE_CONTRACT) + ".sha256.seal.sha256"): "06c134e50fec169fd5609956fdc723d9ddfe9297ec132b5a4e29869bf0692d44",
    RUNNER: "4a661d50ca1929968b31258dd4950945bdd792311c090389f6a882e52aba58c3",
    AUTHOR / "review.json": "7d5b94241eb726a9287619f69816c15d6ff76feac3f64cf0829806c41520c002",
    AUTHOR / "SHA256SUMS": "86e27f8170cdeabd05fa98549f04fb15ce6700256368d0fb79013322c0e49197",
    AUTHOR / "SHA256SUMS.seal.sha256": "acae14e78699d817cf20a989e41926c42fffc222c0a189481840f1a2557ca756",
    HAMMER / "review.json": "d82bf311ec6332cd724feab4aaa3bdf9de2075da233c5bce0e9b3ff3ba450a6b",
    HAMMER / "SHA256SUMS": "6c52562bc3e5cf090ab40c131ccfdc7a27ec63439f5d31158ea86b99a305425d",
    HAMMER / "SHA256SUMS.seal.sha256": "1b8ef5ac32517bf1e95fc4bc604f8abf20485cea8f341e6898a216841248076f",
    Path("rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"): "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    Path("rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"): "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    Path("rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"): "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    Path("verif_m1168r2_c1_common_charge_protocol/m1168r2_m1162_common_charge_protocol_assertions_r2.sv"): "59ff9141175159e9043d86dd5932a4113fde88582005487f1eb65e372c6a684f",
    Path("verif_m1168r2_c1_common_charge_protocol/tb_m1168r2_m1162_common_charge_protocol_unit_delay_r2.sv"): "bd5a2c3ce1ab9f03a7017756c96d5013577116583fc7d007ef3374593272ee35",
    Path("dc_handoff/filelists/date_m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs.f"): "96331eb20fb6d4e72e157d23c579841a121103053ed6246f0b76f812399f1411",
    Path("verif_m1168r2_c1_common_charge_protocol/static_check_m1168r2_m1162_vcs_source.py"): "022cf2d61d29cb22547db78de3dc8f5dbbbc8e0b03443c7469abd4f56d6beae8",
    R1_ATTEMPT / "identity.txt": "7b624fd913046f028506594e1b354bbb76c777a7c6467e1652c178fc7e05faae",
    R1_QUARANTINE / "compile.log": "39765d45f5e53de02a4c9139915253b0d0d8190f042027b70344dea08b0037ff",
    R1_QUARANTINE / "SHA256SUMS": "6f7d480bc752ea5835c3442de72f8e5e484ae41db3a5377b49e593e13838614c",
    R1_QUARANTINE / "SHA256SUMS.seal.sha256": "72ec416eb80888bb5c30a448c870b0859912097d43564662a3a88953182316c7",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXTERNAL = {
    Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"): "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    Path("/opt/anaconda3/envs/pytorch310/bin/python3.10"): "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"): "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
}

checks = 0
mutations = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise RuntimeError(message)


def full(relative: Path) -> Path:
    return ROOT / relative


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_load(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("non-finite JSON " + token)))


def verify_sidecars(relative: Path) -> None:
    target = full(relative)
    sidecar = Path(str(target) + ".sha256")
    outer = Path(str(target) + ".sha256.seal.sha256")
    require(sidecar.read_text().split() == [sha(target), target.name], "sidecar content")
    require(outer.read_text().split() == [sha(sidecar), sidecar.name], "outer sidecar content")


def verify_recursive(relative: Path) -> None:
    directory = full(relative)
    require(directory.is_dir() and not directory.is_symlink(), "bad sealed directory")
    listed = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed, "duplicate manifest member")
        listed[name] = digest
    actual = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if not (root_path / name).is_symlink()]
        for name in files:
            path = root_path / name
            require(not path.is_symlink(), "symlink in sealed directory")
            if name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} and stat.S_ISREG(path.lstat().st_mode):
                actual.add(str(path.relative_to(directory)))
    require(set(listed) == actual, "recursive seal membership")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, "recursive member drift")
    require((directory / "SHA256SUMS.seal.sha256").read_text().split() ==
            [sha(directory / "SHA256SUMS"), "SHA256SUMS"], "recursive outer seal")


def validate_release(data, source, author, hammer) -> None:
    require(data["schema"] == "m1173_m1172_m1168r2_m1162_c1_common_charge_protocol_vcs_launch_release_r2_v1",
            "schema")
    require(data["status"] == "AUTHORIZE_EXACTLY_ONE_M1168R2_FUNCTIONAL_VCS_ATTEMPT", "status")
    require(data["release"] is True and data["launch_now"] is False
            and data["inert_authoring_only"] is True, "inert flags")
    identity = data["identity"]
    require(identity["runner_sha256"] == EXPECTED[RUNNER], "runner identity")
    require(identity["contract_sha256"] == EXPECTED[SOURCE_CONTRACT], "source contract identity")
    require(identity["m1168r2_author_review_sha256"] == EXPECTED[AUTHOR / "review.json"], "author identity")
    require(identity["hammer_review_sha256"] == EXPECTED[HAMMER / "review.json"], "hammer review")
    require(identity["hammer_manifest_sha256"] == EXPECTED[HAMMER / "SHA256SUMS"], "hammer manifest")
    require(identity["hammer_outer_seal_file_sha256"] == EXPECTED[HAMMER / "SHA256SUMS.seal.sha256"], "hammer outer")
    require(identity["hammer_status"] == "PASS_M1172_M1168R2_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE"
            and identity["hammer_verdict"] == "GO" and identity["hammer_score"] == 99
            and identity["hammer_p0"] == 0 and identity["hammer_p1"] == 0, "hammer admission")
    require(source["status"] == "SOURCE_READY_FOR_FRESH_M1172_HAMMER__NO_VCS_RELEASE", "source status")
    require(source["identity"]["runner_sha256"] == EXPECTED[RUNNER], "source runner")
    require(author["status"] == "PASS_M1168R2_COMPILE_REPAIR_SOURCE_ONLY__FRESH_M1172_HAMMER_AND_M1173_RELEASE_REQUIRED__NO_VCS_NO_EDA",
            "author status")
    require(hammer["status"] == "PASS_M1172_M1168R2_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE"
            and hammer["verdict"] == "GO" and hammer["issue_counts"]["P0"] == 0
            and hammer["issue_counts"]["P1"] == 0, "hammer status")
    require(data["authorization"] == {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0},
            "one compile/one simv authorization")
    require(data["required_environment"] == {
        "M1168R2_EXPECTED_RELEASE_SHA256": "SHA256_OF_THIS_EXACT_RELEASE_JSON",
        "M1168R2_EXPECTED_HAMMER_REVIEW_SHA256": EXPECTED[HAMMER / "review.json"],
        "M1168R2_EXPECTED_HAMMER_OUTER_SHA256": EXPECTED[HAMMER / "SHA256SUMS.seal.sha256"],
        "all_three_required_nonempty_and_exact": True,
        "runner_arguments": 0,
    }, "exact environment")
    unique = data["unique_attempt"]
    require(unique["attempt_path"] == str(R2_ATTEMPT), "attempt path")
    require(unique["result_path"] == str(R2_RESULT), "result path")
    require(unique["work_prefix"] == R2_WORK_GLOB[:-1], "work prefix")
    require(unique["quarantine_prefix"] == R2_QUARANTINE_GLOB.split("*")[0], "quarantine prefix")
    require(unique["single_attempt"] is True
            and unique["attempt_result_work_and_quarantine_absent_at_release_authoring"] is True,
            "fresh exactly-once namespace")
    require(data["r1_failure_forensics"]["old_attempt_reusable"] is False
            and data["r1_failure_forensics"]["r1_namespace_forbidden_as_r2_write_target"] is True,
            "r1 reuse forbidden")
    gate = data["fresh_release_hammer_gate"]
    require(gate["required"] is True and gate["direct_execution_before_fresh_release_hammer"] is False,
            "future release hammer gate")
    require(data["independence"]["m1173_author_did_not_perform_m1172_source_hammer"] is False,
            "same-author history must be explicit")
    for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                "ppa", "power", "energy", "full_storage_physically_integrated",
                "system_speedup", "paper_citable", "headline"):
        require(data["claim_boundary"][key] is False, "claim opened: " + key)


def reject_mutation(base, source, author, hammer, mutate) -> None:
    global mutations
    trial = copy.deepcopy(base)
    mutate(trial)
    try:
        validate_release(trial, source, author, hammer)
    except (KeyError, RuntimeError, TypeError):
        mutations += 1
        return
    raise RuntimeError("semantic release mutation accepted")


def namespace_fresh() -> None:
    for relative in (R2_ATTEMPT, R2_RESULT, FUTURE_HAMMER):
        require(not os.path.lexists(full(relative)), "stale namespace: " + str(relative))
    require(not glob.glob(str(full(Path(R2_WORK_GLOB)))), "stale r2 work")
    require(not glob.glob(str(full(Path(R2_QUARANTINE_GLOB)))), "stale r2 quarantine")


def same_uid_eda_hits():
    blocked = {"vcs", "vcs1", "simv", "dc_shell", "pt_shell", "fm_shell",
               "icc2_shell", "common_shell_exec", "common_shell_exe"}
    ancestry = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            break
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestry:
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / "comm").read_text().strip()
            argv = [x.decode(errors="replace") for x in
                    (proc / "cmdline").read_bytes().split(b"\0") if x]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked or blocked.intersection(Path(arg).name for arg in argv):
            hits.append((proc.name, comm, argv[:4]))
    return hits


def memavailable_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable absent")


def validate_runner() -> None:
    runner = full(RUNNER).read_text()
    require(runner.count('"${VCS_BIN}" -full64') == 1, "compile cardinality")
    require(runner.count('./simv -no_save') == 1, "simv cardinality")
    require('RELEASE="${HW_ROOT}/contracts/m1173_m1172_m1168r2_m1162_c1_common_charge_protocol_vcs_launch_release_r2_20260830.json"' in runner,
            "release path mismatch")
    require('ATTEMPT="${HW_ROOT}/results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed"' in runner,
            "attempt path mismatch")
    require('RESULT="${HW_ROOT}/results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830"' in runner,
            "result path mismatch")
    require('[[ "${mem_kib}" =~ ^[0-9]+$ && "${mem_kib}" -ge 67108864 ]]' in runner,
            "64GiB runner gate absent")
    require("EDA collision" in runner and "blocked={'vcs','vcs1','simv','dc_shell'" in runner,
            "same-UID runner gate absent")
    require(runner.index('verify_recursive_seal "${HAMMER_DIR}"') < runner.index('mkdir -- "${ATTEMPT}"'),
            "attempt before hammer gate")
    require(runner.index('sha_exact "${M1168R2_EXPECTED_RELEASE_SHA256}" "${RELEASE}"') <
            runner.index('mkdir -- "${ATTEMPT}"'), "attempt before release gate")


def main() -> None:
    namespace_fresh()
    hits = same_uid_eda_hits()
    require(not hits, "same-UID EDA collision: " + repr(hits))
    mem_kib = memavailable_kib()
    require(mem_kib >= MIN_MEM_KIB, "MemAvailable below 64GiB")
    for relative, expected in EXPECTED.items():
        path = full(relative)
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "identity drift: " + str(relative))
    for path, expected in EXTERNAL.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "external identity drift: " + str(path))
    verify_sidecars(RELEASE)
    verify_sidecars(SOURCE_CONTRACT)
    for directory in (AUTHOR, HAMMER, R1_QUARANTINE):
        verify_recursive(directory)
    release = strict_load(full(RELEASE))
    source = strict_load(full(SOURCE_CONTRACT))
    author = strict_load(full(AUTHOR / "review.json"))
    hammer = strict_load(full(HAMMER / "review.json"))
    validate_release(release, source, author, hammer)
    validate_runner()
    mutation_suite = (
        lambda d: d.__setitem__("launch_now", True),
        lambda d: d["authorization"].__setitem__("vcs_compiles", 2),
        lambda d: d["authorization"].__setitem__("simv_runs", 0),
        lambda d: d["authorization"].__setitem__("all_other_eda_runs", 1),
        lambda d: d["identity"].__setitem__("runner_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_review_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_manifest_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_outer_seal_file_sha256", "0" * 64),
        lambda d: d["required_environment"].__setitem__("runner_arguments", 1),
        lambda d: d["unique_attempt"].__setitem__("attempt_path", str(R1_ATTEMPT)),
        lambda d: d["unique_attempt"].__setitem__("result_path", "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830"),
        lambda d: d["unique_attempt"].__setitem__("single_attempt", False),
        lambda d: d["r1_failure_forensics"].__setitem__("old_attempt_reusable", True),
        lambda d: d["fresh_release_hammer_gate"].__setitem__("required", False),
        lambda d: d["fresh_release_hammer_gate"].__setitem__("direct_execution_before_fresh_release_hammer", True),
        lambda d: d["claim_boundary"].__setitem__("functional_vcs_verified", True),
        lambda d: d["claim_boundary"].__setitem__("timing_verified", True),
        lambda d: d["claim_boundary"].__setitem__("speedup", True),
        lambda d: d["claim_boundary"].__setitem__("paper_citable", True),
    )
    for mutation in mutation_suite:
        reject_mutation(release, source, author, hammer, mutation)
    print(json.dumps({
        "schema": "m1173_m1168r2_inert_release_author_check_r2_v1",
        "status": "PASS_M1173_INERT_RELEASE_AUTHOR_CHECK__FRESH_M1174_HAMMER_REQUIRED__NO_EDA",
        "checks_passed": checks,
        "semantic_mutations_rejected": mutations,
        "same_uid_eda_hits": len(hits),
        "memavailable_kib": mem_kib,
        "minimum_memavailable_kib": MIN_MEM_KIB,
        "r2_attempt_absent": True,
        "r2_result_absent": True,
        "r2_work_absent": True,
        "r2_quarantine_absent": True,
        "future_release_hammer_absent": True,
        "runner_compiles_authorized_after_hammer": 1,
        "runner_simv_runs_authorized_after_hammer": 1,
        "runner_invocations": 0,
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "license_queries": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
