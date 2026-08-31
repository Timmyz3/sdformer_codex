#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh different-author hammer of the inert M1173/M1168R2 VCS release.

This is a static, read-only admission check.  It performs byte/seal checks,
live namespace/resource checks, and in-memory mutation attacks only.  It must
not invoke the runner, VCS, simv, a license query, or any EDA executable.
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RELEASE = HW / "contracts/m1173_m1172_m1168r2_m1162_c1_common_charge_protocol_vcs_launch_release_r2_20260830.json"
SOURCE_CONTRACT = HW / "contracts/m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_contract_r1_20260830.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1168r2_m1162_c1_common_charge_protocol_exact_sha_r2.sh"
RELEASE_AUTHOR = HW / "reviews/m1173_m1172_m1168r2_m1162_c1_common_charge_protocol_vcs_release_author_receipt_r2_20260830"
SOURCE_AUTHOR = HW / "reviews/m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_author_receipt_r1_20260830"
SOURCE_HAMMER = HW / "reviews/m1172_m1168r2_m1171_m1168_m1162_c1_common_charge_protocol_vcs_compile_repair_source_hammer_r1_20260830"
R1_ATTEMPT_ID = HW / "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed/identity.txt"
R1_QUARANTINE = HW / "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830.failed_or_incomplete.3074649.quarantine"
R2_ATTEMPT = HW / "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed"
R2_RESULT = HW / "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830"
R2_WORK = str(HW / "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_work.*")
R2_QUARANTINE = str(HW / "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830.failed_or_incomplete.*.quarantine")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
MIN_MEM_KIB = 67_108_864

EXPECTED = {
    RELEASE: "31302e769ac6e3938df0b37adf336f5cca20cbdfe92e2b95b16a37979d52f52f",
    Path(str(RELEASE) + ".sha256"): "c4e3a638b4b47659ee466823067b862299f6a04c85bab53f5621a7c24fa8e031",
    Path(str(RELEASE) + ".sha256.seal.sha256"): "459c489ac90c2de14eff53cd03b24ecd6d9d9a64494456787efe554db81a4ecc",
    SOURCE_CONTRACT: "7abf99b60fce68ee0823b0e087f3276dccbc33b4d6921c5e6fe34bf3e16abe21",
    Path(str(SOURCE_CONTRACT) + ".sha256"): "d6f0e14eaf2a23a7369a86b9783b194b05c67cd9dbd5dfa2bb0ad5fe30e6c9f4",
    Path(str(SOURCE_CONTRACT) + ".sha256.seal.sha256"): "06c134e50fec169fd5609956fdc723d9ddfe9297ec132b5a4e29869bf0692d44",
    RUNNER: "4a661d50ca1929968b31258dd4950945bdd792311c090389f6a882e52aba58c3",
    RELEASE_AUTHOR / "review.json": "7d486682ffb2e6dace1b141d39cb87e076ad921920ac09b700ca9f1b8294c9e1",
    RELEASE_AUTHOR / "SHA256SUMS": "6306c54d4781c95382133fc56f2d1dff2666b1e0bf2b950214755ce06cf6822a",
    RELEASE_AUTHOR / "SHA256SUMS.seal.sha256": "e40d36948e7059af963260a8beb45e03ecedb26e62598ceafc9ea44bc26ce7d1",
    SOURCE_AUTHOR / "review.json": "7d5b94241eb726a9287619f69816c15d6ff76feac3f64cf0829806c41520c002",
    SOURCE_AUTHOR / "SHA256SUMS": "86e27f8170cdeabd05fa98549f04fb15ce6700256368d0fb79013322c0e49197",
    SOURCE_AUTHOR / "SHA256SUMS.seal.sha256": "acae14e78699d817cf20a989e41926c42fffc222c0a189481840f1a2557ca756",
    SOURCE_HAMMER / "review.json": "d82bf311ec6332cd724feab4aaa3bdf9de2075da233c5bce0e9b3ff3ba450a6b",
    SOURCE_HAMMER / "SHA256SUMS": "6c52562bc3e5cf090ab40c131ccfdc7a27ec63439f5d31158ea86b99a305425d",
    SOURCE_HAMMER / "SHA256SUMS.seal.sha256": "1b8ef5ac32517bf1e95fc4bc604f8abf20485cea8f341e6898a216841248076f",
    R1_ATTEMPT_ID: "7b624fd913046f028506594e1b354bbb76c777a7c6467e1652c178fc7e05faae",
    R1_QUARANTINE / "SHA256SUMS": "6f7d480bc752ea5835c3442de72f8e5e484ae41db3a5377b49e593e13838614c",
    R1_QUARANTINE / "SHA256SUMS.seal.sha256": "72ec416eb80888bb5c30a448c870b0859912097d43564662a3a88953182316c7",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXACT_SOURCES = {
    HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    HW / "verif_m1168r2_c1_common_charge_protocol/m1168r2_m1162_common_charge_protocol_assertions_r2.sv": "59ff9141175159e9043d86dd5932a4113fde88582005487f1eb65e372c6a684f",
    HW / "verif_m1168r2_c1_common_charge_protocol/tb_m1168r2_m1162_common_charge_protocol_unit_delay_r2.sv": "bd5a2c3ce1ab9f03a7017756c96d5013577116583fc7d007ef3374593272ee35",
    HW / "dc_handoff/filelists/date_m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs.f": "96331eb20fb6d4e72e157d23c579841a121103053ed6246f0b76f812399f1411",
    HW / "verif_m1168r2_c1_common_charge_protocol/static_check_m1168r2_m1162_vcs_source.py": "022cf2d61d29cb22547db78de3dc8f5dbbbc8e0b03443c7469abd4f56d6beae8",
    Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"): "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"): "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    Path("/opt/anaconda3/envs/pytorch310/bin/python3.10"): "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
}

checks = 0
attacks: dict[str, str] = {}


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise Failure(message)


def reject(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except BaseException as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise Failure("mutation accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected,
            "identity drift: " + str(path))


def strict_text(text: str) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON " + token)))


def strict(path: Path) -> Any:
    return strict_text(path.read_text(encoding="utf-8"))


def recursive(directory: Path, expected_outer: str) -> dict[str, Any] | None:
    regular(directory / "SHA256SUMS.seal.sha256", expected_outer)
    outer = (directory / "SHA256SUMS.seal.sha256").read_text().split()
    require(outer == [sha(directory / "SHA256SUMS"), "SHA256SUMS"], "outer seal")
    listed: dict[str, str] = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        require(name not in listed and not rel.is_absolute() and ".." not in rel.parts,
                "unsafe/duplicate member")
        listed[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        rel = member.relative_to(directory).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(rel)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(listed), "recursive membership")
    for name, digest in listed.items():
        regular(directory / name, digest)
    review = directory / "review.json"
    return strict(review) if review.is_file() else None


def validate_release(data: dict[str, Any]) -> None:
    require(data["schema"] == "m1173_m1172_m1168r2_m1162_c1_common_charge_protocol_vcs_launch_release_r2_v1", "schema")
    require(data["status"] == "AUTHORIZE_EXACTLY_ONE_M1168R2_FUNCTIONAL_VCS_ATTEMPT", "status")
    require(data["release"] is True and data["launch_now"] is False and data["inert_authoring_only"] is True, "inert")
    identity = data["identity"]
    require(identity["runner_sha256"] == EXPECTED[RUNNER], "runner identity")
    require(identity["contract_sha256"] == EXPECTED[SOURCE_CONTRACT], "contract identity")
    require(identity["hammer_review_sha256"] == EXPECTED[SOURCE_HAMMER / "review.json"], "hammer review")
    require(identity["hammer_manifest_sha256"] == EXPECTED[SOURCE_HAMMER / "SHA256SUMS"], "hammer manifest")
    require(identity["hammer_outer_seal_file_sha256"] == EXPECTED[SOURCE_HAMMER / "SHA256SUMS.seal.sha256"], "hammer outer")
    require(identity["hammer_status"] == "PASS_M1172_M1168R2_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE" and
            identity["hammer_verdict"] == "GO" and identity["hammer_score"] == 99 and
            identity["hammer_p0"] == 0 and identity["hammer_p1"] == 0, "hammer admission")
    require(data["required_environment"] == {
        "M1168R2_EXPECTED_RELEASE_SHA256": "SHA256_OF_THIS_EXACT_RELEASE_JSON",
        "M1168R2_EXPECTED_HAMMER_REVIEW_SHA256": EXPECTED[SOURCE_HAMMER / "review.json"],
        "M1168R2_EXPECTED_HAMMER_OUTER_SHA256": EXPECTED[SOURCE_HAMMER / "SHA256SUMS.seal.sha256"],
        "all_three_required_nonempty_and_exact": True,
        "runner_arguments": 0,
    }, "environment")
    require(data["authorization"] == {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0}, "authorization")
    unique = data["unique_attempt"]
    require(unique == {
        "attempt_path": "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed",
        "result_path": "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830",
        "work_prefix": "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_work.",
        "quarantine_prefix": "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830.failed_or_incomplete.",
        "single_attempt": True,
        "attempt_result_work_and_quarantine_absent_at_release_authoring": True,
        "reuse_restore_delete_alias_or_namespace_substitution_forbidden": True,
    }, "unique namespaces")
    old = data["r1_failure_forensics"]
    require(old["old_attempt_reusable"] is False and old["r1_namespace_forbidden_as_r2_write_target"] is True, "r1 not reusable")
    gates = data["operational_gates"]
    require(gates["same_uid_eda_collision_scan_required_at_authoring_and_runner_launch"] is True and
            gates["minimum_memavailable_kib"] == MIN_MEM_KIB and gates["simv_timeout_seconds"] == 1800 and
            gates["failure_quarantine_recursive_seal_required"] is True and
            gates["canonical_success_recursive_seal_required"] is True, "operational gates")
    fresh = data["fresh_release_hammer_gate"]
    require(fresh["required"] is True and
            fresh["future_path"] == "reviews/m1174_m1173_m1172_m1168r2_m1162_c1_vcs_launch_release_hammer_r2_20260830" and
            fresh["direct_execution_before_fresh_release_hammer"] is False and
            fresh["root_may_launch_only_after_exact_release_hammer_pass_and_live_revalidation"] is True, "fresh gate")
    for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup", "ppa", "power", "energy", "full_storage_physically_integrated", "system_speedup", "paper_citable", "headline"):
        require(data["claim_boundary"][key] is False, "claim open: " + key)


RUNNER_TOKENS = (
    "set -euo pipefail", "umask 002", "[[ $# -eq 0 ]]", "M1168R2_EXPECTED_RELEASE_SHA256",
    "M1168R2_EXPECTED_HAMMER_REVIEW_SHA256", "M1168R2_EXPECTED_HAMMER_OUTER_SHA256",
    'sha_exact "${M1168R2_EXPECTED_RELEASE_SHA256}" "${RELEASE}"',
    'sha_exact "${M1168R2_EXPECTED_HAMMER_REVIEW_SHA256}" "${HAMMER_DIR}/review.json"',
    'sha_exact "${M1168R2_EXPECTED_HAMMER_OUTER_SHA256}" "${HAMMER_DIR}/SHA256SUMS.seal.sha256"',
    'verify_recursive_seal "${HAMMER_DIR}"',
    'RESULT="${HW_ROOT}/results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830"',
    'ATTEMPT="${HW_ROOT}/results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed"',
    'WORK="${HW_ROOT}/results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_work.$$"',
    '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]]',
    "blocked={'vcs','vcs1','simv','dc_shell','pt_shell','fm_shell','icc2_shell','common_shell_exec','common_shell_exe'}",
    "-ge 67108864", 'mkdir -- "${ATTEMPT}"', 'mkdir -- "${WORK}"', "WORK_ACTIVE=1",
    "+define+UNIT_DELAY", "+vcs+lic+wait", '-f "${FILELIST}"', '-top "${TOP}"', "-o simv",
    "/usr/bin/timeout --signal=TERM --kill-after=30s 1800s ./simv -no_save",
    'seal_dir "${WORK}"', 'mv -- "${WORK}" "${RESULT}"', "WORK_ACTIVE=0",
    "FAILED_OR_INCOMPLETE", "functional_vcs_verified=false", "functional_vcs_verified':True",
    "timing_verified':False", "cycles_measured':False", "speedup':False", "system_speedup':False",
)


def validate_runner(text: str) -> None:
    for token in RUNNER_TOKENS:
        require(token in text, "runner token absent: " + token)
    require(text.count('"${VCS_BIN}" -full64') == 1, "compile cardinality")
    require(text.count('./simv -no_save') == 1, "simv cardinality")
    attempt = text.index('mkdir -- "${ATTEMPT}"')
    require(text.index("EDA collision") < attempt and text.index("MemAvailable below 64 GiB") < attempt, "live gates after attempt")
    require(attempt < text.index('"${VCS_BIN}" -full64') < text.index('./simv -no_save'), "execution order")
    require(text.index('seal_dir "${WORK}"') < text.index('mv -- "${WORK}" "${RESULT}"'), "success seal order")
    require("eval " not in text and '. "' not in text, "dynamic injection")


def release_mutation(base: dict[str, Any], label: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    trial = copy.deepcopy(base)
    mutate(trial)
    reject(label, lambda: validate_release(trial))


def runner_mutation(base: str, label: str, old: str, new: str) -> None:
    require(old in base, "mutation anchor absent: " + label)
    reject(label, lambda: validate_runner(base.replace(old, new)))


def same_uid_hits() -> list[Any]:
    blocked = {"vcs", "vcs1", "simv", "dc_shell", "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec", "common_shell_exe"}
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
            argv = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked or blocked.intersection(Path(x).name for x in argv):
            hits.append((proc.name, comm, argv[:4]))
    return hits


def memavailable() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise Failure("MemAvailable absent")


def main() -> None:
    for path, digest in EXPECTED.items():
        regular(path, digest)
    for path, digest in EXACT_SOURCES.items():
        regular(path, digest)
    require(Path(str(RELEASE) + ".sha256").read_text().split() == [EXPECTED[RELEASE], RELEASE.name], "release sidecar")
    require(Path(str(RELEASE) + ".sha256.seal.sha256").read_text().split() == [EXPECTED[Path(str(RELEASE) + ".sha256")], RELEASE.name + ".sha256"], "release outer")
    require(Path(str(SOURCE_CONTRACT) + ".sha256").read_text().split() == [EXPECTED[SOURCE_CONTRACT], SOURCE_CONTRACT.name], "source sidecar")
    require(Path(str(SOURCE_CONTRACT) + ".sha256.seal.sha256").read_text().split() == [EXPECTED[Path(str(SOURCE_CONTRACT) + ".sha256")], SOURCE_CONTRACT.name + ".sha256"], "source outer")
    release_author = recursive(RELEASE_AUTHOR, EXPECTED[RELEASE_AUTHOR / "SHA256SUMS.seal.sha256"])
    source_author = recursive(SOURCE_AUTHOR, EXPECTED[SOURCE_AUTHOR / "SHA256SUMS.seal.sha256"])
    source_hammer = recursive(SOURCE_HAMMER, EXPECTED[SOURCE_HAMMER / "SHA256SUMS.seal.sha256"])
    recursive(R1_QUARANTINE, EXPECTED[R1_QUARANTINE / "SHA256SUMS.seal.sha256"])
    require(release_author["status"].startswith("PASS_M1173_INERT_VCS_RELEASE_AUTHORING"), "release author status")
    require(source_author["status"].startswith("PASS_M1168R2_COMPILE_REPAIR_SOURCE_ONLY"), "source author status")
    require(source_hammer["status"] == "PASS_M1172_M1168R2_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE" and source_hammer["verdict"] == "GO", "source hammer status")
    release = strict(RELEASE)
    validate_release(release)
    runner = RUNNER.read_text(encoding="utf-8")
    validate_runner(runner)
    require(not os.path.lexists(R2_ATTEMPT) and not os.path.lexists(R2_RESULT), "r2 attempt/result not fresh")
    require(not glob.glob(R2_WORK) and not glob.glob(R2_QUARANTINE), "r2 work/quarantine not fresh")
    hits = same_uid_hits()
    require(not hits, "same-UID EDA collision: " + repr(hits))
    mem_kib = memavailable()
    require(mem_kib >= MIN_MEM_KIB, "memory gate")

    release_attacks = (
        ("launch_now", lambda d: d.__setitem__("launch_now", True)),
        ("release_false", lambda d: d.__setitem__("release", False)),
        ("vcs_twice", lambda d: d["authorization"].__setitem__("vcs_compiles", 2)),
        ("simv_zero", lambda d: d["authorization"].__setitem__("simv_runs", 0)),
        ("other_eda", lambda d: d["authorization"].__setitem__("all_other_eda_runs", 1)),
        ("runner_sha", lambda d: d["identity"].__setitem__("runner_sha256", "0" * 64)),
        ("contract_sha", lambda d: d["identity"].__setitem__("contract_sha256", "0" * 64)),
        ("hammer_review", lambda d: d["identity"].__setitem__("hammer_review_sha256", "0" * 64)),
        ("hammer_manifest", lambda d: d["identity"].__setitem__("hammer_manifest_sha256", "0" * 64)),
        ("hammer_outer", lambda d: d["identity"].__setitem__("hammer_outer_seal_file_sha256", "0" * 64)),
        ("env_release", lambda d: d["required_environment"].pop("M1168R2_EXPECTED_RELEASE_SHA256")),
        ("env_review", lambda d: d["required_environment"].__setitem__("M1168R2_EXPECTED_HAMMER_REVIEW_SHA256", "0" * 64)),
        ("runner_args", lambda d: d["required_environment"].__setitem__("runner_arguments", 1)),
        ("r1_attempt_alias", lambda d: d["unique_attempt"].__setitem__("attempt_path", "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed")),
        ("r1_result_alias", lambda d: d["unique_attempt"].__setitem__("result_path", "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830")),
        ("reuse", lambda d: d["unique_attempt"].__setitem__("reuse_restore_delete_alias_or_namespace_substitution_forbidden", False)),
        ("old_reusable", lambda d: d["r1_failure_forensics"].__setitem__("old_attempt_reusable", True)),
        ("low_mem", lambda d: d["operational_gates"].__setitem__("minimum_memavailable_kib", 1)),
        ("collision_off", lambda d: d["operational_gates"].__setitem__("same_uid_eda_collision_scan_required_at_authoring_and_runner_launch", False)),
        ("fresh_off", lambda d: d["fresh_release_hammer_gate"].__setitem__("required", False)),
        ("direct_launch", lambda d: d["fresh_release_hammer_gate"].__setitem__("direct_execution_before_fresh_release_hammer", True)),
        ("claim_functional", lambda d: d["claim_boundary"].__setitem__("functional_vcs_verified", True)),
        ("claim_timing", lambda d: d["claim_boundary"].__setitem__("timing_verified", True)),
        ("claim_speed", lambda d: d["claim_boundary"].__setitem__("speedup", True)),
        ("claim_paper", lambda d: d["claim_boundary"].__setitem__("paper_citable", True)),
    )
    for label, mutation in release_attacks:
        release_mutation(release, label, mutation)

    runner_attacks = (
        ("drop_strict", "set -euo pipefail", "set -e"),
        ("drop_zero_args", "[[ $# -eq 0 ]]", "true"),
        ("drop_release_env", "M1168R2_EXPECTED_RELEASE_SHA256", "M1168R2_RELEASE_ALIAS"),
        ("drop_review_env", "M1168R2_EXPECTED_HAMMER_REVIEW_SHA256", "M1168R2_REVIEW_ALIAS"),
        ("drop_outer_env", "M1168R2_EXPECTED_HAMMER_OUTER_SHA256", "M1168R2_OUTER_ALIAS"),
        ("drop_hammer_seal", 'verify_recursive_seal "${HAMMER_DIR}"', "true #"),
        ("drop_release_sha", 'sha_exact "${M1168R2_EXPECTED_RELEASE_SHA256}" "${RELEASE}"', "true #"),
        ("drop_namespace", '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]]', "true"),
        ("drop_memory", "-ge 67108864", "-ge 1"),
        ("unit_delay", "+define+UNIT_DELAY", "+define+NO_UNIT_DELAY"),
        ("second_compile", '"${VCS_BIN}" -full64', '"${VCS_BIN}" -full64\n"${VCS_BIN}" -full64'),
        ("drop_timeout", "/usr/bin/timeout --signal=TERM --kill-after=30s 1800s ./simv -no_save", "./simv -no_save"),
        ("second_simv", "./simv -no_save", "./simv -no_save\n./simv -no_save"),
        ("drop_success_seal", 'seal_dir "${WORK}"', "true #"),
        ("open_failure", "functional_vcs_verified=false", "functional_vcs_verified=true"),
        ("open_timing", "timing_verified':False", "timing_verified':True"),
    )
    for label, old, new in runner_attacks:
        runner_mutation(runner, label, old, new)
    reject("duplicate_json_key", lambda: strict_text('{"x":1,"x":2}'))
    reject("nonfinite_json", lambda: strict_text('{"x":NaN}'))

    print(json.dumps({
        "schema": "m1174_m1173_m1168r2_vcs_release_hammer_r2_v1",
        "status": "PASS_M1174_M1173_M1168R2_VCS_RELEASE_HAMMER__AUTHORIZE_EXACTLY_ONE_FUNCTIONAL_VCS_ATTEMPT__NO_EDA_RUN",
        "checks_passed": checks,
        "attacks_rejected": len(attacks),
        "release_mutations_rejected": len(release_attacks),
        "runner_mutations_rejected": len(runner_attacks),
        "strict_json_attacks_rejected": 2,
        "recursive_sealed_directories": 4,
        "same_uid_eda_hits": 0,
        "memavailable_kib": mem_kib,
        "minimum_memavailable_kib": MIN_MEM_KIB,
        "r2_attempt_absent": True,
        "r2_result_absent": True,
        "r2_work_absent": True,
        "r2_quarantine_absent": True,
        "runner_invocations": 0,
        "vcs_compiles": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "license_queries": 0,
        "attacks": attacks,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
