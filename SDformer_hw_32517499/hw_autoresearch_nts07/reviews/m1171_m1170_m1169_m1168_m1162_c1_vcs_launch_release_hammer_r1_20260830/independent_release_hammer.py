#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent fail-closed hammer for the inert M1170 VCS release.

This program performs byte/seal checks and controlled in-memory semantic
mutations only.  It never invokes the released runner, VCS, simv, a license
query, or any other EDA tool.
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RELEASE = HW / "contracts/m1170_m1169_m1168_m1162_c1_common_charge_protocol_vcs_launch_release_r1_20260830.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1168_m1162_c1_common_charge_protocol_exact_sha_r1.sh"
M1168_CONTRACT = HW / "contracts/m1168_m1166_m1162_c1_common_charge_protocol_vcs_source_contract_r1_20260830.json"
M1162_CONTRACT = HW / "contracts/m1162_m1160_m1116c_c1_common_charge_protocol_repair_source_contract_r1_20260830.json"
M1170_AUTHOR = HW / "reviews/m1170_m1169_m1168_m1162_c1_common_charge_protocol_vcs_release_author_receipt_r1_20260830"
M1169 = HW / "reviews/m1169_m1168_m1162_c1_common_charge_protocol_vcs_source_hammer_r1_20260830"
M1168_AUTHOR = HW / "reviews/m1168_m1162_c1_common_charge_protocol_vcs_source_author_receipt_r1_20260830"
M1166 = HW / "reviews/m1166_m1162_c1_common_charge_protocol_repair_independent_hammer_r1_20260830"
M1162_AUTHOR = HW / "reviews/m1162_m1160_c1_common_charge_protocol_repair_source_author_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed"
RESULT = HW / "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830"
WORK_GLOB = str(HW / "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_work.*")
QUARANTINE_GLOB = str(HW / "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830.failed_or_incomplete.*.quarantine")
MIN_MEM_KIB = 67_108_864

EXPECTED = {
    RELEASE: "a66e4b1f9beb9fcdfb2c1fe8d0b474dc1bf7e9101b1658249a59f35db4d89487",
    Path(str(RELEASE) + ".sha256"): "9f33dca246d0e533756e68415a4ac8992d279230cd1d52a56904f3aada57d3eb",
    Path(str(RELEASE) + ".sha256.seal.sha256"): "409b19832ca49625af76272226cb1a86af07fc383d8a6d9222b7a40a7c734fd6",
    RUNNER: "9ddee66afb64b9519dee9af73b1aac4961440e0f9342eb219b046e1d6305adaf",
    M1168_CONTRACT: "626b1402a6f5ce9f32128b90fa4eb4aae17e0cf79d749f3bc62e6d8f898cc288",
    Path(str(M1168_CONTRACT) + ".sha256"): "eb77f58a4a4291be39a6ec4e74e9eb1701f6d5bec968fc5c2042a5812ad76883",
    Path(str(M1168_CONTRACT) + ".sha256.seal.sha256"): "99605cc617a7a889e95aae45c7d5b053d9a5bd757265bd8319e4d9bb8dc20e31",
    M1162_CONTRACT: "5787f3302aa3308485e357c41385e69da93e6b41bfdea92410690af5a95ecbdc",
    Path(str(M1162_CONTRACT) + ".sha256"): "88c38e071ef67a62e8267c827c4ba0e55bc49099340177a16e45ce21f0ecdbc9",
    Path(str(M1162_CONTRACT) + ".sha256.seal.sha256"): "95ef450f49b64468c1a91a2de983b03320a32bca15aef95be5021c53da81eabe",
    M1170_AUTHOR / "review.json": "9e3fc07b2d200ba0bc1c9f9f00639495fe9989503f47f89a15b6a1e05c885977",
    M1170_AUTHOR / "SHA256SUMS": "1812dd5a47dad1730411adc30d4f5d632cee24dd6d8759094c18b81ca96bb1e1",
    M1170_AUTHOR / "SHA256SUMS.seal.sha256": "1ef1fa0e2ddde1b162961932841efa82ba0381400136bb2532cfa9c8dbf4b6ee",
    M1169 / "review.json": "8599a332cc0c4e2289969c5eede2fc20850a32ce2541112d2727fbba41eb6fdc",
    M1169 / "SHA256SUMS": "e0e1c124f840f79b2ef661998eb58caeade3d045316e8e32e3d554b0a4aed671",
    M1169 / "SHA256SUMS.seal.sha256": "cc37cf92b3b30a9c6b13b7625591c262539b2461961f9bdc840660fc1a338121",
    M1168_AUTHOR / "review.json": "33de8a1947035c1be4c0c773502a99e545c37be1ebfde530fa5802dbdf45fd4c",
    M1168_AUTHOR / "SHA256SUMS": "45f3dab5ba0bdd7d3ede9ede8b578fa804a9e0a2de5761800beadda9689e2f83",
    M1168_AUTHOR / "SHA256SUMS.seal.sha256": "bef0ccacd029e0320511dfe2520fbbf37a6cdc3750e0b2cefbdf33df37035397",
    M1166 / "review.json": "7f2cdf4cb1f979c0680b491c27c1088bc35624a2fd801b97c304c5b403076b4c",
    M1166 / "SHA256SUMS": "da8daaef6b6832dd2d3278fcbdf61613170f07da5bb65e311915a3c421e76363",
    M1166 / "SHA256SUMS.seal.sha256": "afc25e37fa8b3b5c5bd8e8c1b3582fecc5d2d75450df86b7c48f71e992ea02ef",
    M1162_AUTHOR / "review.json": "734ce901318bcc62951a7b479f3d42d0230fbc7a3be9c39137270858f9ad71a5",
    M1162_AUTHOR / "SHA256SUMS": "da799abfdad2dab521ba90f48b8956a5ddcd1dee95aaf675a184b281fa34f302",
    M1162_AUTHOR / "SHA256SUMS.seal.sha256": "67cb13ac317f140f4a042373a1c79640295bb861ffc25905605c65656c5fe18a",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXACT_SOURCE = {
    HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    HW / "verif_m1168_c1_common_charge_protocol/m1168_m1162_common_charge_protocol_assertions_r1.sv": "9f7d4dcc9edb4ceb66469e2095fc4ae0043d625db309fb6fb00fc8fb197e261b",
    HW / "verif_m1168_c1_common_charge_protocol/tb_m1168_m1162_common_charge_protocol_unit_delay_r1.sv": "ae04c1c9e5104e4e4272632b0aa595fa2b8f93cef7c98ef40210afa0af7d28cc",
    HW / "dc_handoff/filelists/date_m1168_m1162_c1_common_charge_protocol_unit_delay_vcs.f": "a6d0a90e0132771992dd5c5f9c3fc1e185020e724baa5eb0648632a7a0d593be",
    HW / "verif_m1168_c1_common_charge_protocol/static_check_m1168_m1162_vcs_source.py": "0f924125286c726d6d4a7ee0ceda3147da0f1e708b8d7b18ed65fbd83c32bd12",
    Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"): "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"): "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    Path("/opt/anaconda3/envs/pytorch310/bin/python3.10"): "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
}

checks = 0
attacks: dict[str, str] = {}


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except BaseException as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise HammerFailure("attack accepted: " + label)


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


def strict_json(path: Path) -> Any:
    return strict_json_text(path.read_text(encoding="utf-8"))


def strict_json_text(text: str) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON " + token)))


def verify_recursive(directory: Path, expected_outer: str) -> dict[str, Any]:
    regular(directory / "SHA256SUMS.seal.sha256", expected_outer)
    outer_words = (directory / "SHA256SUMS.seal.sha256").read_text().split()
    require(outer_words == [sha(directory / "SHA256SUMS"), "SHA256SUMS"],
            "outer seal content drift")
    listed: dict[str, str] = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        require(name not in listed and not rel.is_absolute() and ".." not in rel.parts,
                "unsafe/duplicate sealed path")
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
    require(actual == set(listed), "recursive exact-member coverage drift")
    for name, digest in listed.items():
        regular(directory / name, digest)
    return strict_json(directory / "review.json")


def validate_release(data: dict[str, Any]) -> None:
    require(data["schema"] == "m1170_m1169_m1168_m1162_c1_common_charge_protocol_vcs_launch_release_r1_v1", "schema")
    require(data["status"] == "AUTHORIZE_EXACTLY_ONE_M1168_FUNCTIONAL_VCS_ATTEMPT", "status")
    require(data["release"] is True and data["launch_now"] is False and data["inert_authoring_only"] is True, "inert flags")
    identity = data["identity"]
    require(identity["runner_sha256"] == EXPECTED[RUNNER], "runner identity")
    require(identity["contract_sha256"] == EXPECTED[M1168_CONTRACT], "contract identity")
    require(identity["hammer_review_sha256"] == EXPECTED[M1169 / "review.json"], "hammer review")
    require(identity["hammer_manifest_sha256"] == EXPECTED[M1169 / "SHA256SUMS"], "hammer manifest")
    require(identity["hammer_outer_seal_file_sha256"] == EXPECTED[M1169 / "SHA256SUMS.seal.sha256"], "hammer outer")
    require(data["required_environment"] == {
        "M1168_EXPECTED_RELEASE_SHA256": "SHA256_OF_THIS_EXACT_RELEASE_JSON",
        "M1168_EXPECTED_HAMMER_REVIEW_SHA256": EXPECTED[M1169 / "review.json"],
        "M1168_EXPECTED_HAMMER_OUTER_SHA256": EXPECTED[M1169 / "SHA256SUMS.seal.sha256"],
        "all_three_required_nonempty_and_exact": True,
        "runner_arguments": 0,
    }, "environment closure")
    require(data["authorization"] == {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0}, "authorization")
    unique = data["unique_attempt"]
    require(unique["attempt_path"] == "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed" and
            unique["result_path"] == "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830" and
            unique["work_prefix"] == "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_work." and
            unique["single_attempt"] is True and
            unique["reuse_restore_delete_alias_or_namespace_substitution_forbidden"] is True,
            "unique namespace")
    gates = data["operational_gates"]
    require(gates["same_uid_eda_collision_scan_required_at_authoring_and_runner_launch"] is True and
            gates["minimum_memavailable_kib"] == MIN_MEM_KIB and
            gates["simv_timeout_seconds"] == 1800 and
            gates["failure_quarantine_recursive_seal_required"] is True and
            gates["canonical_success_recursive_seal_required"] is True,
            "operational gates")
    fresh = data["fresh_release_hammer_gate"]
    require(fresh["required"] is True and fresh["direct_execution_before_fresh_release_hammer"] is False and
            fresh["root_may_launch_only_after_exact_release_hammer_pass_and_live_revalidation"] is True,
            "fresh hammer gate")
    for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup", "ppa", "power", "energy", "full_storage_physically_integrated", "system_speedup", "paper_citable", "headline"):
        require(data["claim_boundary"][key] is False, "claim opened: " + key)


RUNNER_REQUIRED = (
    "set -euo pipefail", "umask 002", "[[ $# -eq 0 ]]", "M1168_EXPECTED_RELEASE_SHA256",
    "M1168_EXPECTED_HAMMER_REVIEW_SHA256", "M1168_EXPECTED_HAMMER_OUTER_SHA256",
    "sha_exact \"${M1168_EXPECTED_HAMMER_REVIEW_SHA256}\"", "sha_exact \"${M1168_EXPECTED_HAMMER_OUTER_SHA256}\"",
    "sha_exact \"${M1168_EXPECTED_RELEASE_SHA256}\"", "verify_recursive_seal \"${HAMMER_DIR}\"",
    "[[ ! -e \"${RESULT}\" && ! -e \"${ATTEMPT}\" && ! -e \"${WORK}\" ]]",
    "mem_kib=\"$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)\"", "-ge 67108864",
    "mkdir -- \"${ATTEMPT}\"", "mkdir -- \"${WORK}\"", "WORK_ACTIVE=1",
    "blocked={'vcs','vcs1','simv','dc_shell','pt_shell','fm_shell','icc2_shell','common_shell_exec','common_shell_exe'}",
    "+define+UNIT_DELAY", "+vcs+lic+wait", "-f \"${FILELIST}\"", "-top \"${TOP}\"", "-o simv",
    "/usr/bin/timeout --signal=TERM --kill-after=30s 1800s ./simv -no_save",
    "seal_dir \"${WORK}\"", "mv -- \"${WORK}\" \"${RESULT}\"", "WORK_ACTIVE=0",
    "FAILED_OR_INCOMPLETE", "functional_vcs_verified=false", "functional_vcs_verified':True",
    "timing_verified':False", "cycles_measured':False", "speedup':False", "system_speedup':False",
)


def validate_runner_text(text: str) -> None:
    for token in RUNNER_REQUIRED:
        require(token in text, "runner semantic token absent: " + token)
    require(text.count('"${VCS_BIN}" -full64') == 1, "VCS invocation cardinality")
    require(text.count('./simv -no_save') == 1, "simv invocation cardinality")
    require(text.index('mkdir -- "${ATTEMPT}"') < text.index('"${VCS_BIN}" -full64'), "attempt not before VCS")
    require(text.index('mkdir -- "${ATTEMPT}"') < text.index('./simv -no_save'), "attempt not before simv")
    require(text.index("EDA collision") < text.index('mkdir -- "${ATTEMPT}"'), "collision gate ordering")
    require(text.index("MemAvailable below 64 GiB") < text.index('mkdir -- "${ATTEMPT}"'), "memory gate ordering")
    require("eval " not in text and ". \"" not in text, "dynamic shell injection")


def reject_release_mutation(base: dict[str, Any], label: str, mutator: Callable[[dict[str, Any]], None]) -> None:
    trial = copy.deepcopy(base)
    mutator(trial)
    rejected(label, lambda: validate_release(trial))


def reject_runner_mutation(base: str, label: str, old: str, new: str = "") -> None:
    require(old in base, "runner mutation anchor absent: " + label)
    rejected(label, lambda: validate_runner_text(base.replace(old, new)))


def same_uid_eda_hits() -> list[tuple[str, str, list[str]]]:
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
            argv = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked or blocked.intersection(Path(arg).name for arg in argv):
            hits.append((proc.name, comm, argv[:4]))
    return hits


def memavailable_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise HammerFailure("MemAvailable absent")


def main() -> None:
    for path, digest in EXPECTED.items():
        regular(path, digest)
    for path, digest in EXACT_SOURCE.items():
        regular(path, digest)
    release_side = RELEASE.with_name(RELEASE.name + ".sha256")
    release_outer = RELEASE.with_name(RELEASE.name + ".sha256.seal.sha256")
    require(release_side.read_text().split() == [EXPECTED[RELEASE], RELEASE.name], "release sidecar content")
    require(release_outer.read_text().split() == [EXPECTED[release_side], release_side.name], "release outer content")
    for directory, outer in (
        (M1170_AUTHOR, EXPECTED[M1170_AUTHOR / "SHA256SUMS.seal.sha256"]),
        (M1169, EXPECTED[M1169 / "SHA256SUMS.seal.sha256"]),
        (M1168_AUTHOR, EXPECTED[M1168_AUTHOR / "SHA256SUMS.seal.sha256"]),
        (M1166, EXPECTED[M1166 / "SHA256SUMS.seal.sha256"]),
        (M1162_AUTHOR, EXPECTED[M1162_AUTHOR / "SHA256SUMS.seal.sha256"]),
    ):
        verify_recursive(directory, outer)
    release = strict_json(RELEASE)
    validate_release(release)
    runner_text = RUNNER.read_text(encoding="utf-8")
    validate_runner_text(runner_text)
    require(not os.path.lexists(ATTEMPT) and not os.path.lexists(RESULT), "attempt/result namespace not fresh")
    require(not glob.glob(WORK_GLOB) and not glob.glob(QUARANTINE_GLOB), "work/quarantine namespace not fresh")
    hits = same_uid_eda_hits()
    require(not hits, "same-UID EDA collision: " + repr(hits))
    mem_kib = memavailable_kib()
    require(mem_kib >= MIN_MEM_KIB, "MemAvailable below 64 GiB")

    release_mutations = [
        ("release_status", lambda d: d.__setitem__("status", "PASS_FUNCTIONAL_VCS_ONLY")),
        ("launch_now", lambda d: d.__setitem__("launch_now", True)),
        ("release_false", lambda d: d.__setitem__("release", False)),
        ("vcs_twice", lambda d: d["authorization"].__setitem__("vcs_compiles", 2)),
        ("simv_zero", lambda d: d["authorization"].__setitem__("simv_runs", 0)),
        ("other_eda", lambda d: d["authorization"].__setitem__("all_other_eda_runs", 1)),
        ("runner_sha", lambda d: d["identity"].__setitem__("runner_sha256", "0" * 64)),
        ("contract_sha", lambda d: d["identity"].__setitem__("contract_sha256", "0" * 64)),
        ("hammer_review_sha", lambda d: d["identity"].__setitem__("hammer_review_sha256", "0" * 64)),
        ("hammer_manifest_sha", lambda d: d["identity"].__setitem__("hammer_manifest_sha256", "0" * 64)),
        ("hammer_outer_sha", lambda d: d["identity"].__setitem__("hammer_outer_seal_file_sha256", "0" * 64)),
        ("env_release_removed", lambda d: d["required_environment"].pop("M1168_EXPECTED_RELEASE_SHA256")),
        ("env_review_alias", lambda d: d["required_environment"].__setitem__("M1168_EXPECTED_HAMMER_REVIEW_SHA256", "0" * 64)),
        ("env_outer_alias", lambda d: d["required_environment"].__setitem__("M1168_EXPECTED_HAMMER_OUTER_SHA256", "0" * 64)),
        ("runner_arg", lambda d: d["required_environment"].__setitem__("runner_arguments", 1)),
        ("attempt_alias", lambda d: d["unique_attempt"].__setitem__("attempt_path", d["unique_attempt"]["attempt_path"] + ".alias")),
        ("result_alias", lambda d: d["unique_attempt"].__setitem__("result_path", d["unique_attempt"]["result_path"] + ".alias")),
        ("reuse_allowed", lambda d: d["unique_attempt"].__setitem__("reuse_restore_delete_alias_or_namespace_substitution_forbidden", False)),
        ("low_memory_gate", lambda d: d["operational_gates"].__setitem__("minimum_memavailable_kib", 1)),
        ("collision_scan_off", lambda d: d["operational_gates"].__setitem__("same_uid_eda_collision_scan_required_at_authoring_and_runner_launch", False)),
        ("timeout_removed", lambda d: d["operational_gates"].__setitem__("simv_timeout_seconds", 0)),
        ("fresh_hammer_off", lambda d: d["fresh_release_hammer_gate"].__setitem__("required", False)),
        ("direct_launch", lambda d: d["fresh_release_hammer_gate"].__setitem__("direct_execution_before_fresh_release_hammer", True)),
        ("claim_functional", lambda d: d["claim_boundary"].__setitem__("functional_vcs_verified", True)),
        ("claim_timing", lambda d: d["claim_boundary"].__setitem__("timing_verified", True)),
        ("claim_speed", lambda d: d["claim_boundary"].__setitem__("speedup", True)),
        ("claim_paper", lambda d: d["claim_boundary"].__setitem__("paper_citable", True)),
    ]
    for label, mutator in release_mutations:
        reject_release_mutation(release, label, mutator)

    runner_mutations = [
        ("drop_strict_shell", "set -euo pipefail", "set -e"),
        ("drop_zero_args", "[[ $# -eq 0 ]]", "true"),
        ("drop_env_release", "M1168_EXPECTED_RELEASE_SHA256", "M1168_RELEASE_SHA_ALIAS"),
        ("drop_env_review", "M1168_EXPECTED_HAMMER_REVIEW_SHA256", "M1168_REVIEW_SHA_ALIAS"),
        ("drop_env_outer", "M1168_EXPECTED_HAMMER_OUTER_SHA256", "M1168_OUTER_SHA_ALIAS"),
        ("drop_release_exact", "sha_exact \"${M1168_EXPECTED_RELEASE_SHA256}\"", "true #"),
        ("drop_review_exact", "sha_exact \"${M1168_EXPECTED_HAMMER_REVIEW_SHA256}\"", "true #"),
        ("drop_recursive_hammer", "verify_recursive_seal \"${HAMMER_DIR}\"", "true #"),
        ("drop_namespace", "[[ ! -e \"${RESULT}\" && ! -e \"${ATTEMPT}\" && ! -e \"${WORK}\" ]]", "true"),
        ("drop_collision", "if hits: raise SystemExit('EDA collision: %r' % hits)", "pass"),
        ("drop_memory", "-ge 67108864", "-ge 1"),
        ("attempt_after_vcs", "mkdir -- \"${ATTEMPT}\"", "# attempt disabled"),
        ("unit_delay_removed", "+define+UNIT_DELAY", "+define+NO_UNIT_DELAY"),
        ("second_vcs", '"${VCS_BIN}" -full64', '"${VCS_BIN}" -full64\n"${VCS_BIN}" -full64'),
        ("simv_timeout_removed", "/usr/bin/timeout --signal=TERM --kill-after=30s 1800s ./simv -no_save", "./simv -no_save"),
        ("second_simv", "./simv -no_save", "./simv -no_save\n./simv -no_save"),
        ("success_unsealed", "seal_dir \"${WORK}\"", "true #"),
        ("failure_boundary_open", "functional_vcs_verified=false", "functional_vcs_verified=true"),
        ("timing_boundary_open", "timing_verified':False", "timing_verified':True"),
    ]
    for label, old, new in runner_mutations:
        reject_runner_mutation(runner_text, label, old, new)

    # Strict parser must reject duplicate release keys and non-finite values.
    rejected("duplicate_json_key", lambda: strict_json_text('{"status":1,"status":2}'))
    rejected("nonfinite_json", lambda: strict_json_text('{"x":NaN}'))

    output = {
        "status": "PASS_M1171_M1170_VCS_RELEASE_HAMMER__AUTHORIZE_EXACTLY_ONE_FUNCTIONAL_VCS_ATTEMPT__NO_EDA_RUN",
        "checks_passed": checks,
        "attacks_rejected": len(attacks),
        "release_mutations_rejected": len(release_mutations),
        "runner_mutations_rejected": len(runner_mutations),
        "strict_json_attacks_rejected": 2,
        "exact_files": len(EXPECTED) + len(EXACT_SOURCE),
        "recursive_sealed_directories": 5,
        "same_uid_eda_hits": 0,
        "memavailable_kib": mem_kib,
        "minimum_memavailable_kib": MIN_MEM_KIB,
        "attempt_absent": True,
        "result_absent": True,
        "work_absent": True,
        "quarantine_absent": True,
        "runner_invocations": 0,
        "vcs_compiles": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "attacks": attacks,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
