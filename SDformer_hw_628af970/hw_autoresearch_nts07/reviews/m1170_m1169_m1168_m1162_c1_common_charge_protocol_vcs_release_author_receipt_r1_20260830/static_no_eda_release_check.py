#!/usr/bin/env python3
"""Fail-closed author check for the inert M1170 VCS launch release.

This checker never invokes the M1168 runner, VCS, simv, a license query, or any
other EDA tool.  It validates the frozen byte/seal chain and the live launch
preconditions while the canonical attempt namespace is still absent.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path("contracts/m1170_m1169_m1168_m1162_c1_common_charge_protocol_vcs_launch_release_r1_20260830.json")
M1168_CONTRACT = Path("contracts/m1168_m1166_m1162_c1_common_charge_protocol_vcs_source_contract_r1_20260830.json")
RUNNER = Path("dc_handoff/scripts/run_vcs_m1168_m1162_c1_common_charge_protocol_exact_sha_r1.sh")
M1168_AUTHOR = Path("reviews/m1168_m1162_c1_common_charge_protocol_vcs_source_author_receipt_r1_20260830")
M1169 = Path("reviews/m1169_m1168_m1162_c1_common_charge_protocol_vcs_source_hammer_r1_20260830")
M1162_CONTRACT = Path("contracts/m1162_m1160_m1116c_c1_common_charge_protocol_repair_source_contract_r1_20260830.json")
M1162_AUTHOR = Path("reviews/m1162_m1160_c1_common_charge_protocol_repair_source_author_receipt_r1_20260830")
M1166 = Path("reviews/m1166_m1162_c1_common_charge_protocol_repair_independent_hammer_r1_20260830")
DOCS359 = Path("docs/359_DATE终局冻结_20260813.md")
ATTEMPT = Path("results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_attempt_consumed")
RESULT = Path("results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830")
WORK_GLOB = "results/.m1168_m1162_c1_common_charge_protocol_vcs_r1_work.*"
QUARANTINE_GLOB = "results/m1168_m1162_c1_common_charge_protocol_unit_delay_vcs_r1_20260830.failed_or_incomplete.*.quarantine"
FUTURE_HAMMER = Path("reviews/m1171_m1170_m1169_m1168_m1162_c1_vcs_launch_release_hammer_r1_20260830")
MIN_MEM_KIB = 67_108_864

EXPECTED = {
    RELEASE: "a66e4b1f9beb9fcdfb2c1fe8d0b474dc1bf7e9101b1658249a59f35db4d89487",
    Path(str(RELEASE) + ".sha256"): "9f33dca246d0e533756e68415a4ac8992d279230cd1d52a56904f3aada57d3eb",
    Path(str(RELEASE) + ".sha256.seal.sha256"): "409b19832ca49625af76272226cb1a86af07fc383d8a6d9222b7a40a7c734fd6",
    M1168_CONTRACT: "626b1402a6f5ce9f32128b90fa4eb4aae17e0cf79d749f3bc62e6d8f898cc288",
    Path(str(M1168_CONTRACT) + ".sha256"): "eb77f58a4a4291be39a6ec4e74e9eb1701f6d5bec968fc5c2042a5812ad76883",
    Path(str(M1168_CONTRACT) + ".sha256.seal.sha256"): "99605cc617a7a889e95aae45c7d5b053d9a5bd757265bd8319e4d9bb8dc20e31",
    RUNNER: "9ddee66afb64b9519dee9af73b1aac4961440e0f9342eb219b046e1d6305adaf",
    M1168_AUTHOR / "review.json": "33de8a1947035c1be4c0c773502a99e545c37be1ebfde530fa5802dbdf45fd4c",
    M1168_AUTHOR / "SHA256SUMS": "45f3dab5ba0bdd7d3ede9ede8b578fa804a9e0a2de5761800beadda9689e2f83",
    M1168_AUTHOR / "SHA256SUMS.seal.sha256": "bef0ccacd029e0320511dfe2520fbbf37a6cdc3750e0b2cefbdf33df37035397",
    M1169 / "review.json": "8599a332cc0c4e2289969c5eede2fc20850a32ce2541112d2727fbba41eb6fdc",
    M1169 / "SHA256SUMS": "e0e1c124f840f79b2ef661998eb58caeade3d045316e8e32e3d554b0a4aed671",
    M1169 / "SHA256SUMS.seal.sha256": "cc37cf92b3b30a9c6b13b7625591c262539b2461961f9bdc840660fc1a338121",
    M1162_CONTRACT: "5787f3302aa3308485e357c41385e69da93e6b41bfdea92410690af5a95ecbdc",
    Path(str(M1162_CONTRACT) + ".sha256"): "88c38e071ef67a62e8267c827c4ba0e55bc49099340177a16e45ce21f0ecdbc9",
    Path(str(M1162_CONTRACT) + ".sha256.seal.sha256"): "95ef450f49b64468c1a91a2de983b03320a32bca15aef95be5021c53da81eabe",
    M1162_AUTHOR / "review.json": "734ce901318bcc62951a7b479f3d42d0230fbc7a3be9c39137270858f9ad71a5",
    M1162_AUTHOR / "SHA256SUMS": "da799abfdad2dab521ba90f48b8956a5ddcd1dee95aaf675a184b281fa34f302",
    M1162_AUTHOR / "SHA256SUMS.seal.sha256": "67cb13ac317f140f4a042373a1c79640295bb861ffc25905605c65656c5fe18a",
    M1166 / "review.json": "7f2cdf4cb1f979c0680b491c27c1088bc35624a2fd801b97c304c5b403076b4c",
    M1166 / "SHA256SUMS": "da8daaef6b6832dd2d3278fcbdf61613170f07da5bb65e311915a3c421e76363",
    M1166 / "SHA256SUMS.seal.sha256": "afc25e37fa8b3b5c5bd8e8c1b3582fecc5d2d75450df86b7c48f71e992ea02ef",
    Path("rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"): "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    Path("rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"): "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    Path("rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"): "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    Path("verif_m1168_c1_common_charge_protocol/m1168_m1162_common_charge_protocol_assertions_r1.sv"): "9f7d4dcc9edb4ceb66469e2095fc4ae0043d625db309fb6fb00fc8fb197e261b",
    Path("verif_m1168_c1_common_charge_protocol/tb_m1168_m1162_common_charge_protocol_unit_delay_r1.sv"): "ae04c1c9e5104e4e4272632b0aa595fa2b8f93cef7c98ef40210afa0af7d28cc",
    Path("dc_handoff/filelists/date_m1168_m1162_c1_common_charge_protocol_unit_delay_vcs.f"): "a6d0a90e0132771992dd5c5f9c3fc1e185020e724baa5eb0648632a7a0d593be",
    Path("verif_m1168_c1_common_charge_protocol/static_check_m1168_m1162_vcs_source.py"): "0f924125286c726d6d4a7ee0ceda3147da0f1e708b8d7b18ed65fbd83c32bd12",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXTERNAL_EXPECTED = {
    Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"): "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    Path("/opt/anaconda3/envs/pytorch310/bin/python3.10"): "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"): "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full(relative: Path) -> Path:
    return ROOT / relative


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def strict_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def verify_sidecars(relative: Path) -> None:
    target = full(relative)
    subprocess.check_call(["sha256sum", "-c", target.name + ".sha256"],
                          cwd=target.parent, stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", target.name + ".sha256.seal.sha256"],
                          cwd=target.parent, stdout=subprocess.DEVNULL)


def verify_recursive(relative: Path) -> None:
    directory = full(relative)
    require(directory.is_dir() and not directory.is_symlink(), "bad sealed dir: %s" % relative)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"], cwd=directory,
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=directory,
                          stdout=subprocess.DEVNULL)
    listed = set()
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            listed.add(line.split(None, 1)[1].lstrip("*"))
    actual = set()
    for base, dirs, files in os.walk(directory, followlinks=False):
        base_path = Path(base)
        dirs[:] = [name for name in dirs if not (base_path / name).is_symlink()]
        for name in files:
            path = base_path / name
            if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                continue
            require(not path.is_symlink(), "symlink in sealed dir: %s" % path)
            if stat.S_ISREG(path.lstat().st_mode):
                actual.add(str(path.relative_to(directory)))
    require(listed == actual, "recursive seal coverage mismatch: %s" % relative)


def validate_release(release, m1168, m1168_author, m1169, m1162, m1166) -> None:
    require(release["schema"] ==
            "m1170_m1169_m1168_m1162_c1_common_charge_protocol_vcs_launch_release_r1_v1",
            "release schema")
    require(release["status"] == "AUTHORIZE_EXACTLY_ONE_M1168_FUNCTIONAL_VCS_ATTEMPT",
            "release status")
    require(release["release"] is True and release["launch_now"] is False and
            release["inert_authoring_only"] is True, "inert release flags")
    identity = release["identity"]
    require(identity["runner_sha256"] == EXPECTED[RUNNER], "runner identity")
    require(identity["contract_sha256"] == EXPECTED[M1168_CONTRACT], "M1168 contract identity")
    require(identity["hammer_review_sha256"] == EXPECTED[M1169 / "review.json"],
            "M1169 review identity")
    require(identity["hammer_manifest_sha256"] == EXPECTED[M1169 / "SHA256SUMS"],
            "M1169 manifest identity")
    require(identity["hammer_outer_seal_file_sha256"] ==
            EXPECTED[M1169 / "SHA256SUMS.seal.sha256"], "M1169 outer identity")
    require(m1168["status"] == "SOURCE_READY_FOR_FRESH_M1169_HAMMER__NO_VCS_RELEASE",
            "M1168 status")
    require(m1168["identity"]["runner_sha256"] == EXPECTED[RUNNER], "M1168 runner drift")
    require(m1168_author["status"] ==
            "PASS_M1168_VCS_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_M1169_HAMMER_REQUIRED__NO_VCS_NO_EDA",
            "M1168 author status")
    require(m1169["status"] == "PASS_M1169_M1168_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE",
            "M1169 status")
    require(m1169["verdict"] == "GO" and m1169["score"] == 97 and
            m1169["issue_counts"]["P0"] == 0 and m1169["issue_counts"]["P1"] == 0,
            "M1169 admission")
    require(m1169["authorization"]["one_additive_m1170_launch_release_source_next"] is True and
            m1169["authorization"]["direct_vcs_without_m1170"] is False,
            "M1169 authorization")
    require(m1162["status"] ==
            "PASS_M1162_ADDITIVE_PROTOCOL_REPAIR_SOURCE_ONLY__FRESH_HAMMER_REQUIRED__NO_VCS_NO_EDA",
            "M1162 source status")
    require(m1166["status"] ==
            "PASS_M1166_M1162_PROTOCOL_REPAIR_SOURCE_HAMMER__AUTHORIZE_ONE_ADDITIVE_VCS_SOURCE_LAUNCH_PACKAGE__NO_VCS_NO_EDA",
            "M1166 status")
    require(release["authorization"] ==
            {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0},
            "one-shot authorization")
    env = release["required_environment"]
    require(env == {
        "M1168_EXPECTED_RELEASE_SHA256": "SHA256_OF_THIS_EXACT_RELEASE_JSON",
        "M1168_EXPECTED_HAMMER_REVIEW_SHA256": EXPECTED[M1169 / "review.json"],
        "M1168_EXPECTED_HAMMER_OUTER_SHA256": EXPECTED[M1169 / "SHA256SUMS.seal.sha256"],
        "all_three_required_nonempty_and_exact": True,
        "runner_arguments": 0,
    }, "required environment")
    unique = release["unique_attempt"]
    require(unique["attempt_path"] == str(ATTEMPT), "attempt coordinate")
    require(unique["result_path"] == str(RESULT), "result coordinate")
    require(unique["work_prefix"] == WORK_GLOB[:-1], "work prefix coordinate")
    require(unique["single_attempt"] is True, "single attempt")
    gate = release["fresh_release_hammer_gate"]
    require(gate["required"] is True and gate["direct_execution_before_fresh_release_hammer"] is False,
            "fresh release hammer gate")
    for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                "ppa", "power", "energy", "full_storage_physically_integrated",
                "system_speedup", "paper_citable", "headline"):
        require(release["claim_boundary"][key] is False, "claim opened: %s" % key)


def reject_mutation(base, validator, mutator) -> None:
    trial = copy.deepcopy(base)
    mutator(trial)
    try:
        validator(trial)
    except (KeyError, RuntimeError, TypeError):
        return
    raise RuntimeError("semantic mutation accepted")


def assert_namespace_absent() -> None:
    for relative in (ATTEMPT, RESULT, FUTURE_HAMMER):
        require(not os.path.lexists(full(relative)), "stale namespace: %s" % relative)
    require(not glob.glob(str(full(Path(WORK_GLOB)))), "stale work namespace")
    require(not glob.glob(str(full(Path(QUARANTINE_GLOB)))), "stale quarantine namespace")


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


def main() -> None:
    assert_namespace_absent()
    require(not same_uid_eda_hits(), "same-UID EDA collision")
    mem_kib = memavailable_kib()
    require(mem_kib >= MIN_MEM_KIB, "MemAvailable below 64 GiB")
    for relative, expected in EXPECTED.items():
        path = full(relative)
        require(path.is_file() and not path.is_symlink(), "missing/nonregular/symlink: %s" % relative)
        require(sha(path) == expected, "SHA drift: %s" % relative)
    for path, expected in EXTERNAL_EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), "external identity absent: %s" % path)
        require(sha(path) == expected, "external SHA drift: %s" % path)
    verify_sidecars(RELEASE)
    verify_sidecars(M1168_CONTRACT)
    verify_sidecars(M1162_CONTRACT)
    for directory in (M1168_AUTHOR, M1169, M1162_AUTHOR, M1166):
        verify_recursive(directory)
    release = strict_load(full(RELEASE))
    m1168 = strict_load(full(M1168_CONTRACT))
    m1168_author = strict_load(full(M1168_AUTHOR / "review.json"))
    m1169 = strict_load(full(M1169 / "review.json"))
    m1162 = strict_load(full(M1162_CONTRACT))
    m1166 = strict_load(full(M1166 / "review.json"))
    validator = lambda candidate: validate_release(candidate, m1168, m1168_author, m1169, m1162, m1166)
    validator(release)
    mutations = [
        lambda d: d.__setitem__("status", "PASS_FUNCTIONAL_VCS_ONLY"),
        lambda d: d.__setitem__("launch_now", True),
        lambda d: d["authorization"].__setitem__("vcs_compiles", 2),
        lambda d: d["authorization"].__setitem__("simv_runs", 0),
        lambda d: d["authorization"].__setitem__("all_other_eda_runs", 1),
        lambda d: d["identity"].__setitem__("runner_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("contract_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_review_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_manifest_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_outer_seal_file_sha256", "0" * 64),
        lambda d: d["required_environment"].__setitem__("M1168_EXPECTED_HAMMER_REVIEW_SHA256", "0" * 64),
        lambda d: d["required_environment"].__setitem__("M1168_EXPECTED_HAMMER_OUTER_SHA256", "0" * 64),
        lambda d: d["required_environment"].__setitem__("runner_arguments", 1),
        lambda d: d["unique_attempt"].__setitem__("attempt_path", str(ATTEMPT) + ".alias"),
        lambda d: d["unique_attempt"].__setitem__("result_path", str(RESULT) + ".alias"),
        lambda d: d["fresh_release_hammer_gate"].__setitem__("required", False),
        lambda d: d["fresh_release_hammer_gate"].__setitem__("direct_execution_before_fresh_release_hammer", True),
        lambda d: d["claim_boundary"].__setitem__("functional_vcs_verified", True),
        lambda d: d["claim_boundary"].__setitem__("timing_verified", True),
        lambda d: d["claim_boundary"].__setitem__("speedup", True),
        lambda d: d["claim_boundary"].__setitem__("paper_citable", True),
    ]
    for mutator in mutations:
        reject_mutation(release, validator, mutator)
    print(json.dumps({
        "status": "PASS_M1170_INERT_RELEASE_AUTHOR_CHECK__FRESH_M1171_HAMMER_REQUIRED__NO_EDA",
        "exact_files": len(EXPECTED) + len(EXTERNAL_EXPECTED),
        "recursive_sealed_directories": 4,
        "semantic_mutations_rejected": len(mutations),
        "same_uid_eda_hits": 0,
        "memavailable_kib": mem_kib,
        "attempt_absent": True,
        "result_absent": True,
        "work_absent": True,
        "quarantine_absent": True,
        "future_hammer_absent": True,
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
