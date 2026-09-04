#!/usr/bin/env python3
"""Source-only author check for the inert M1183 M1168R3 release."""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import stat
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RELEASE = HW / "contracts/m1183_m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_launch_release_r3_20260830.json"
CONTRACT = HW / "contracts/m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_author_receipt_r1_20260830"
HAMMER = HW / "reviews/m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_hammer_r1_20260830"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1168r3_m1162_c1_common_charge_protocol_exact_sha_r3.sh"
R2_ATTEMPT = HW / "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed/identity.txt"
R2_Q = HW / "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830.failed_or_incomplete.3284331.quarantine"
ATTEMPT = HW / "results/.m1168r3_m1162_c1_common_charge_protocol_vcs_r3_attempt_consumed"
RESULT = HW / "results/m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830"
WORK_GLOB = str(HW / "results/.m1168r3_m1162_c1_common_charge_protocol_vcs_r3_work.*")
Q_GLOB = str(HW / "results/m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830.failed_or_incomplete.*")
FUTURE_HAMMER = HW / "reviews/m1184_m1183_m1182_m1181_m1168r3_m1162_c1_vcs_launch_release_hammer_r1_20260830"
MIN_MEM_KIB = 67_108_864


EXPECTED = {
    RELEASE: "cc285797c98784548933f86d98f410000f0036ac9dbdfe27f19cdd1f241c3403",
    Path(str(RELEASE) + ".sha256"): "0d1922e0d0386a250ad0c021924e819dcbba3c270fcddb79d79f527367a4d231",
    Path(str(RELEASE) + ".sha256.seal.sha256"): "225a614da5e0aec48c5df2abc20cb79095c3b1a91c0b982dd57ff1629466db3f",
    CONTRACT: "64e5f2935a1c401d7151a2bad2434af2ae51e59beccf7df677de0aedb9bb4389",
    Path(str(CONTRACT) + ".sha256"): "439faae37b6025e8415ed7dfc2e4c57355010f002bf5b38a6cfb5b52cd4b515c",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "f955f228edd834f066b170f953e6176bef28b26bc48c7203f6e13af60f3b2d12",
    AUTHOR / "review.json": "662f1eca4fd2eaad9f1bcbfe0db630912f1562161688c1719ac2a17d4da51aa0",
    AUTHOR / "SHA256SUMS": "3aff95cebbe2bdcb3956fab42ddb02e281b9f78e619a21043118e5ca847bb338",
    AUTHOR / "SHA256SUMS.seal.sha256": "cb4c80b6458605e89258e2c6250f7a33ec86e68c4adadcc5a353a32dfe49c71f",
    HAMMER / "review.json": "9216102c2298966d54ddd478e42734b01c25f1d4c685762fbe579d08b07bf96e",
    HAMMER / "SHA256SUMS": "35d0b079db73282d70e284f5a417f0df743ffee3058a356ca6a8eef18ef2a67c",
    HAMMER / "SHA256SUMS.seal.sha256": "b2efc1076de8be88b420d2701f4e0b7dd065dfe449b45cc9bae3bdc84d16ac18",
    RUNNER: "d64b887e4313f83b93ee68d50d73c3702cb7aec84cfe42080e0459b2d8b51344",
    HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    HW / "verif_m1168r3_c1_common_charge_protocol/tb_m1168r3_m1162_common_charge_protocol_unit_delay_r3.sv": "b68e0f452cdd7aa87c5408b7d90222f0531faacdf0605a87dedced359b7d5a2d",
    HW / "verif_m1168r3_c1_common_charge_protocol/static_check_m1168r3_m1162_vcs_source.py": "30a67b1f4b0a12017c09077cbc730de936ee532e76df4445d2957035ee47320e",
    HW / "dc_handoff/filelists/date_m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs.f": "9030f139e20d301ef9bc558a726c7c524353bb830845a9914d7c738d6e4e50a3",
    HW / "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    R2_ATTEMPT: "dde2eca905affe76a5e5a74966fe2502bc1fb82364493ee314819264d8bd75ca",
    R2_Q / "compile.log": "1ee670031e192b1d2e894183e851cf6b65dd86319ba1224d76ea938dbf979de4",
    R2_Q / "sim.log": "fbcc88d9893be34d3aa5bbf3cb49936cc4c1f5f24d0eab1eb797e3039bd657c3",
    R2_Q / "SHA256SUMS": "f3926823e62535facb13a369d78f0d13489be90494ac2cb1ea192885e412ecb9",
    R2_Q / "SHA256SUMS.seal.sha256": "c147c7e8a6ff7d523aa96f159e715967054ce0d4d19f699f7f8ed4daef8f9989",
    Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"): "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"): "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    Path("/opt/anaconda3/envs/pytorch310/bin/python3.10"): "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, f"duplicate JSON key {key}")
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def verify_leaf(path: Path) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(side.read_text().split() == [sha(path), path.name], "leaf sidecar")
    need(outer.read_text().split() == [sha(side), side.name], "leaf outer")


def verify_recursive(directory: Path) -> None:
    need(directory.is_dir() and not directory.is_symlink(), f"sealed dir {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "recursive outer")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        p = Path(name)
        need(name not in listed and not p.is_absolute() and ".." not in p.parts, "unsafe manifest")
        listed[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        rel = member.relative_to(directory).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        # The frozen VCS quarantine contains tool-generated symlinks.  Its
        # runner seal deliberately covers every regular evidence byte and
        # ignores those links; reproduce that exact policy.
        if member.is_symlink():
            continue
        if stat.S_ISREG(member.lstat().st_mode):
            actual.add(rel)
    need(actual == set(listed), "recursive membership")
    for name, digest in listed.items():
        need(sha(directory / name) == digest, f"recursive drift {name}")


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
            argv = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
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


def validate_release(release, contract, author, hammer, runner: str) -> None:
    need(release["schema"] == "m1183_m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_launch_release_r3_v1", "schema")
    need(release["status"] == "AUTHORIZE_EXACTLY_ONE_M1168R3_FUNCTIONAL_VCS_ATTEMPT", "status")
    need(release["release"] is True and release["launch_now"] is False
         and release["inert_authoring_only"] is True, "inert flags")
    ident = release["identity"]
    need(ident["runner_sha256"] == EXPECTED[RUNNER], "runner identity")
    need(ident["source_contract_sha256"] == EXPECTED[CONTRACT], "contract identity")
    need(ident["source_author_review_sha256"] == EXPECTED[AUTHOR / "review.json"], "author identity")
    need(ident["source_author_manifest_sha256"] == EXPECTED[AUTHOR / "SHA256SUMS"], "author manifest")
    need(ident["source_author_outer_seal_file_sha256"] == EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"], "author outer")
    need(ident["hammer_review_sha256"] == EXPECTED[HAMMER / "review.json"], "hammer review")
    need(ident["hammer_manifest_sha256"] == EXPECTED[HAMMER / "SHA256SUMS"], "hammer manifest")
    need(ident["hammer_outer_seal_file_sha256"] == EXPECTED[HAMMER / "SHA256SUMS.seal.sha256"], "hammer outer")
    need(contract["status"] == "SOURCE_READY_FOR_FRESH_M1182_HAMMER__NO_VCS_RELEASE", "source status")
    need(author["status"] == "PASS_M1181_M1168R3_SOURCE_ONLY_FORENSICS_AND_NEGATIVE_TEST_ISOLATION__FRESH_M1182_HAMMER_REQUIRED__NO_VCS_NO_EDA", "author status")
    need(hammer["status"] == "PASS_M1182_M1181_M1168R3_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE"
         and hammer["verdict"] == "GO" and hammer["score"] == 99
         and hammer["issue_counts"]["P0"] == 0 and hammer["issue_counts"]["P1"] == 0,
         "hammer admission")
    need(release["authorization"] == {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0},
         "authorization cardinality")
    need(release["required_environment"] == {
        "M1168R3_EXPECTED_RELEASE_SHA256": "SHA256_OF_THIS_EXACT_RELEASE_JSON",
        "M1168R3_EXPECTED_HAMMER_REVIEW_SHA256": EXPECTED[HAMMER / "review.json"],
        "M1168R3_EXPECTED_HAMMER_OUTER_SHA256": EXPECTED[HAMMER / "SHA256SUMS.seal.sha256"],
        "all_three_required_nonempty_and_exact": True, "runner_arguments": 0}, "environment")
    u = release["unique_attempt"]
    need(HW / u["attempt_path"] == ATTEMPT and HW / u["result_path"] == RESULT, "namespace coordinate")
    need(u["work_prefix"] == "results/.m1168r3_m1162_c1_common_charge_protocol_vcs_r3_work."
         and u["quarantine_prefix"] == "results/m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830.failed_or_incomplete.", "namespace prefixes")
    need(u["single_attempt"] is True and u["r2_namespace_reuse_restore_delete_alias_or_substitution_forbidden"] is True,
         "single attempt/reuse")
    gate = release["fresh_release_hammer_gate"]
    need(gate["required"] is True and gate["direct_execution_before_fresh_release_hammer"] is False,
         "release hammer gate")
    need(gate["future_path"] == str(FUTURE_HAMMER.relative_to(HW)), "future hammer path")
    op = release["operational_gates"]
    need(op["minimum_memavailable_kib"] == MIN_MEM_KIB and op["simv_timeout_seconds"] == 1800,
         "resource gates")
    need(op["failure_quarantine_recursive_seal_required"] is True, "failure seal")
    for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                "ppa", "power", "energy", "full_storage_physically_integrated",
                "system_speedup", "paper_citable", "headline"):
        need(release["claim_boundary"][key] is False, f"claim opened {key}")
    need(runner.count('"${VCS_BIN}" -full64') == 1 and runner.count("./simv -no_save") == 1,
         "runner cardinality")
    need("verify_recursive_seal \"${R2_QUARANTINE}\"" in runner, "R2 quarantine not bound")
    need("seal_dir \"${WORK}\"" in runner and "failed_or_incomplete.$$.quarantine" in runner,
         "failure sealing absent")
    need("EDA collision" in runner and "MemAvailable below 64 GiB" in runner, "live gate absent")


def reject_mutation(base, validator, mutator):
    trial = copy.deepcopy(base)
    mutator(trial)
    try:
        validator(trial)
    except (KeyError, RuntimeError, TypeError):
        return
    raise RuntimeError("semantic mutation accepted")


def main() -> None:
    need(not ATTEMPT.exists() and not RESULT.exists(), "R3 attempt/result already exists")
    need(not glob.glob(WORK_GLOB) and not glob.glob(Q_GLOB), "R3 work/quarantine already exists")
    need(not FUTURE_HAMMER.exists(), "future release hammer pre-created")
    hits = same_uid_eda_hits()
    need(not hits, f"same-UID EDA collision {hits}")
    mem = memavailable_kib()
    need(mem >= MIN_MEM_KIB, "MemAvailable below 64 GiB")
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest, f"identity drift {path}")
    verify_leaf(RELEASE)
    verify_leaf(CONTRACT)
    for directory in (AUTHOR, HAMMER, R2_Q):
        verify_recursive(directory)
    release = strict_load(RELEASE)
    contract = strict_load(CONTRACT)
    author = strict_load(AUTHOR / "review.json")
    hammer = strict_load(HAMMER / "review.json")
    runner = RUNNER.read_text()
    validator = lambda trial: validate_release(trial, contract, author, hammer, runner)
    validator(release)
    mutations = [
        lambda d: d.__setitem__("status", "PASS_FUNCTIONAL_VCS_ONLY"),
        lambda d: d.__setitem__("launch_now", True),
        lambda d: d["authorization"].__setitem__("vcs_compiles", 2),
        lambda d: d["authorization"].__setitem__("simv_runs", 0),
        lambda d: d["authorization"].__setitem__("all_other_eda_runs", 1),
        lambda d: d["identity"].__setitem__("runner_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("source_contract_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("source_author_review_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("source_author_manifest_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("source_author_outer_seal_file_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_review_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_manifest_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_outer_seal_file_sha256", "0" * 64),
        lambda d: d["required_environment"].__setitem__("M1168R3_EXPECTED_HAMMER_REVIEW_SHA256", "0" * 64),
        lambda d: d["required_environment"].__setitem__("M1168R3_EXPECTED_HAMMER_OUTER_SHA256", "0" * 64),
        lambda d: d["required_environment"].__setitem__("runner_arguments", 1),
        lambda d: d["unique_attempt"].__setitem__("attempt_path", "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed"),
        lambda d: d["unique_attempt"].__setitem__("result_path", str(RESULT.relative_to(HW)) + ".alias"),
        lambda d: d["unique_attempt"].__setitem__("work_prefix", "results/.m1168r2_work."),
        lambda d: d["fresh_release_hammer_gate"].__setitem__("required", False),
        lambda d: d["fresh_release_hammer_gate"].__setitem__("direct_execution_before_fresh_release_hammer", True),
        lambda d: d["operational_gates"].__setitem__("minimum_memavailable_kib", 0),
        lambda d: d["operational_gates"].__setitem__("failure_quarantine_recursive_seal_required", False),
        lambda d: d["claim_boundary"].__setitem__("functional_vcs_verified", True),
        lambda d: d["claim_boundary"].__setitem__("timing_verified", True),
        lambda d: d["claim_boundary"].__setitem__("speedup", True),
        lambda d: d["claim_boundary"].__setitem__("paper_citable", True),
    ]
    for mutator in mutations:
        reject_mutation(release, validator, mutator)
    print(json.dumps({
        "schema": "m1183_m1168r3_inert_release_author_static_check_r1_v1",
        "status": "PASS_M1183_INERT_RELEASE_AUTHOR_CHECK__FRESH_M1184_HAMMER_REQUIRED__NO_EDA",
        "exact_files": len(EXPECTED), "recursive_sealed_directories": 3,
        "semantic_mutations_rejected": len(mutations), "same_uid_eda_hits": 0,
        "memavailable_kib": mem, "attempt_absent": True, "result_absent": True,
        "work_absent": True, "quarantine_absent": True, "future_hammer_absent": True,
        "vcs_runs": 0, "simv_runs": 0, "all_eda_runs": 0, "license_queries": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
