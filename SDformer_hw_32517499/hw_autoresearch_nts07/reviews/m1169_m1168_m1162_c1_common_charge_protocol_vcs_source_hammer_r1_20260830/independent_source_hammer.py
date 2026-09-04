#!/usr/bin/env python3
"""Read-only M1169 hammer for the M1168/M1162 VCS source package.

This checker performs exact-identity, recursive-seal, structural, scenario,
bounded source-mutation and launch-interlock checks.  It does not invoke VCS,
simv, synthesis, replay, GPU or remote work.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SVA = HW / "verif_m1168_c1_common_charge_protocol/m1168_m1162_common_charge_protocol_assertions_r1.sv"
TB = HW / "verif_m1168_c1_common_charge_protocol/tb_m1168_m1162_common_charge_protocol_unit_delay_r1.sv"
CHECKER = HW / "verif_m1168_c1_common_charge_protocol/static_check_m1168_m1162_vcs_source.py"
FILELIST = HW / "dc_handoff/filelists/date_m1168_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1168_m1162_c1_common_charge_protocol_exact_sha_r1.sh"
CONTRACT = HW / "contracts/m1168_m1166_m1162_c1_common_charge_protocol_vcs_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1168_m1162_c1_common_charge_protocol_vcs_source_author_receipt_r1_20260830"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SVA: "9f7d4dcc9edb4ceb66469e2095fc4ae0043d625db309fb6fb00fc8fb197e261b",
    TB: "ae04c1c9e5104e4e4272632b0aa595fa2b8f93cef7c98ef40210afa0af7d28cc",
    CHECKER: "0f924125286c726d6d4a7ee0ceda3147da0f1e708b8d7b18ed65fbd83c32bd12",
    FILELIST: "a6d0a90e0132771992dd5c5f9c3fc1e185020e724baa5eb0648632a7a0d593be",
    RUNNER: "9ddee66afb64b9519dee9af73b1aac4961440e0f9342eb219b046e1d6305adaf",
    CONTRACT: "626b1402a6f5ce9f32128b90fa4eb4aae17e0cf79d749f3bc62e6d8f898cc288",
    AUTHOR / "review.json": "33de8a1947035c1be4c0c773502a99e545c37be1ebfde530fa5802dbdf45fd4c",
    AUTHOR / "SHA256SUMS": "45f3dab5ba0bdd7d3ede9ede8b578fa804a9e0a2de5761800beadda9689e2f83",
    AUTHOR / "SHA256SUMS.seal.sha256": "bef0ccacd029e0320511dfe2520fbbf37a6cdc3750e0b2cefbdf33df37035397",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    PARENT: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str, checks: list[str]) -> None:
    if not value:
        raise RuntimeError(message)
    checks.append(message)


def members(directory: Path) -> set[str]:
    return {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and not p.is_symlink()
        and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }


def manifest(directory: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        if name in rows:
            raise RuntimeError("duplicate manifest member " + name)
        rows[name] = digest
    return rows


def structural_oracle(sva: str, tb: str, expected_asserts: list[str],
                      expected_covers: list[str], tasks: list[str]) -> bool:
    assertions = re.findall(
        r"^\s*(ap_[A-Za-z0-9_]+):\s*assert property", sva, re.M)
    covers = re.findall(
        r"^\s*(cp_[A-Za-z0-9_]+):\s*cover property", sva, re.M)
    if assertions != expected_asserts or covers != expected_covers:
        return False
    return all(len(re.findall(r"^\s*" + re.escape(task) + r"\(\);", tb,
                              re.M)) == 1 for task in tasks)


def main() -> None:
    checks: list[str] = []
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(),
                f"regular source {path.relative_to(HW)}", checks)
        require(sha(path) == expected,
                f"exact SHA {path.relative_to(HW)}", checks)

    listed = manifest(AUTHOR)
    require(set(listed) == members(AUTHOR), "author recursive exact members", checks)
    for name, expected in listed.items():
        require(sha(AUTHOR / name) == expected, "author member " + name, checks)
    outer = (AUTHOR / "SHA256SUMS.seal.sha256").read_text().split()
    require(outer == [EXPECTED[AUTHOR / "SHA256SUMS"], "SHA256SUMS"],
            "author outer seal", checks)

    contract = json.loads(CONTRACT.read_text())
    require(contract["status"] ==
            "SOURCE_READY_FOR_FRESH_M1169_HAMMER__NO_VCS_RELEASE",
            "source status is pre-release", checks)
    require(contract["one_shot_policy"]["fresh_different_author_hammer_required"] is True,
            "fresh hammer required", checks)
    require(contract["one_shot_policy"]["vcs_compiles_after_release"] == 1 and
            contract["one_shot_policy"]["simv_runs_after_release"] == 1 and
            contract["one_shot_policy"]["all_other_eda_runs"] == 0,
            "one compile one simv no other EDA", checks)

    sva = SVA.read_text()
    tb = TB.read_text()
    runner = RUNNER.read_text()
    assertions = re.findall(
        r"^\s*(ap_[A-Za-z0-9_]+):\s*assert property", sva, re.M)
    covers = re.findall(
        r"^\s*(cp_[A-Za-z0-9_]+):\s*cover property", sva, re.M)
    require(len(assertions) == len(set(assertions)) == 16,
            "sixteen unique executable assertions", checks)
    require(len(covers) == len(set(covers)) == 6,
            "six unique executable covers", checks)

    tasks = [
        "directed_weight_first", "directed_psum_first_and_backpressure",
        "directed_nonfirst", "directed_ii2", "reset_pending_cases",
        "sticky_fault_attacks", "service_assumption_attacks",
        "normal_m935_completion",
    ]
    for task in tasks:
        require(len(re.findall(r"^\s*task automatic " + re.escape(task) + r"\b",
                               tb, re.M)) == 1,
                "one task definition " + task, checks)
        require(len(re.findall(r"^\s*" + re.escape(task) + r"\(\);", tb,
                               re.M)) == 1,
                "one scenario call " + task, checks)
    for token in (
        "cov_long_request_stall < 5", "cov_long_response_backpressure < 5",
        "cov_reset_partial != 1", "cov_reset_complete != 1",
        "cov_reset_skew != 1", "cov_unsolicited_weight != 1",
        "cov_unsolicited_psum != 1", "cov_same_cycle_early != 1",
        "cov_duplicate_response != 1", "cov_cancel != 1",
        "cov_tuple_mutation != 1", "cov_nonfirst_psum != 1",
        "cov_weight_payload_mutation != 1", "cov_psum_valid_drop != 1",
        "cov_random_transactions != 24", "cov_normal_issue != 2",
        "cov_normal_row != 1", "cov_normal_task != 1",
        "second_accept_cycle - first_accept_cycle != 2",
        "weight_fire_count != w0 + 1", "psum_fire_count != p0 + first",
        "count_issue_accepts != issue0 + 2",
    ):
        require(token in tb, "scenario/gate token " + token, checks)

    mutations = 0
    for name in assertions:
        mutated = sva.replace(name + ": assert property",
                              name + ": cover property", 1)
        require(not structural_oracle(mutated, tb, assertions, covers, tasks),
                "reject assertion mutation " + name, checks)
        mutations += 1
    for name in covers:
        mutated = sva.replace(name + ": cover property",
                              name + ": assert property", 1)
        require(not structural_oracle(mutated, tb, assertions, covers, tasks),
                "reject cover mutation " + name, checks)
        mutations += 1
    for task in tasks:
        mutated = tb.replace("        " + task + "();",
                             "        // removed_" + task + "();", 1)
        require(not structural_oracle(sva, mutated, assertions, covers, tasks),
                "reject scenario-call mutation " + task, checks)
        mutations += 1

    flines = [line.strip() for line in FILELIST.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    require(len(flines) == len(set(flines)) == 6,
            "six unique filelist sources", checks)
    require(all(Path(item).is_file() and not Path(item).is_symlink()
                for item in flines), "filelist members regular", checks)

    for token in (
        '[[ -n "${M1168_EXPECTED_RELEASE_SHA256:-}"',
        'verify_recursive_seal "${HAMMER_DIR}"',
        'sha_exact "${M1168_EXPECTED_HAMMER_REVIEW_SHA256}"',
        'sha_exact "${M1168_EXPECTED_HAMMER_OUTER_SHA256}"',
        'sha_exact "${M1168_EXPECTED_RELEASE_SHA256}"',
        '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}"',
        'mkdir -- "${ATTEMPT}"', '"${VCS_BIN}" -full64', './simv -no_save',
    ):
        require(token in runner, "runner interlock " + token, checks)
    require(runner.index('verify_recursive_seal "${HAMMER_DIR}"') <
            runner.index('mkdir -- "${ATTEMPT}"') <
            runner.index('"${VCS_BIN}" -full64'),
            "hammer/release checks precede token and compile", checks)

    print(json.dumps({
        "schema": "m1169_m1168_vcs_source_independent_hammer_mechanical_v1",
        "status": "PASS_SOURCE_ONLY_NO_VCS_NO_EDA",
        "assertions": len(assertions), "covers": len(covers),
        "scenario_tasks": len(tasks),
        "bounded_source_mutations_rejected": mutations,
        "checks_passed": len(checks),
        "claim_boundary": {
            "vcs": False, "simv": False, "eda": False,
            "cycles": False, "speedup": False, "ppa": False,
            "energy": False, "system_speedup": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
