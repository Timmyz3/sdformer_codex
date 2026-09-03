#!/usr/bin/env python3
"""Independent M2026 result hammer.  Never invokes EDA or a license query."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULTS = HW / "results"
RESULT = RESULTS / "m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_r1_20260902"
ATTEMPT = RESULTS / ".m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_attempt_consumed"
M2024 = HW / "reviews/m2024_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs_source_hammer_r1_20260902"
RUNNER = HW / "dc_handoff/scripts/run_m2025_m2024_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs_one_shot.sh"
RTL = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
ADAPTER = HW / "rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs.f"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

RUN_COMPLETE = RESULT / "RUN_COMPLETE.txt"
RECEIPT = RESULT / "receipt.json"
LICENSE = RESULT / "license_preflight.log"
COMPILE = RESULT / "vcs_compile.log"
SIM = RESULT / "simv.log"
ATTEMPT_RECORD = ATTEMPT / "ATTEMPT_CONSUMED.txt"
M2024_REVIEW = M2024 / "review.json"

EXPECTED = {
    RUN_COMPLETE: "8512d17723454d0d1aaef5c9092e57e0bddbf05fe9922bed404729489844b9cd",
    RECEIPT: "229cd1460f2a6e6aa7be804d7774eed543a3c84f6b62c71d452c73ab20ec2cd2",
    LICENSE: "8a2dae0a67dc696461714fc8a2342bda5c746aeeaadd1808062cbf162a77287e",
    COMPILE: "47cfe2c55401c51ee864ef5ccdbdc7336741c7c132958ab500fb1787e4b7b257",
    SIM: "74b3247030efe3cbb23d8ae6bc4c1ead2b0fffe8c7ed2a8c34d0b24921528215",
    RESULT / "SHA256SUMS": "144d5d70cbeb670108e3b3f5cd18145b71f4b6edbb705743f29965de94320109",
    RESULT / "SHA256SUMS.seal.sha256": "430b664c60a96a97f5fd8e7b5f4b210143a91b1cda163489e8a624370b24f63d",
    ATTEMPT_RECORD: "bca3b2bfacdc6688e3ebe2d83c4bcb1c6113e4109f983296aa0c839e9f6ade6a",
    ATTEMPT / "SHA256SUMS": "37babde85c21b7a16cb8bf8bd7c66695fdf6846973d5f9666cf1c14e15e4a961",
    ATTEMPT / "SHA256SUMS.seal.sha256": "9c10cd56a222f8bdeb16b3b8ddc4ff6724f9556df2c2fd5f05d6bd4a12cde3e4",
    M2024_REVIEW: "571262fc6262e76d29007b8ea92b6b0703daf9a7a05b4def5e38c3144ae71119",
    M2024 / "SHA256SUMS": "84fc0b4dc0e1233689a7304c6ec1a205674e1bc1ef805bb2b196545e87e7b444",
    M2024 / "SHA256SUMS.seal.sha256": "4486a8e58cfe9b7bc90046a6e650cf9a4a4a75b10c941c896a932cb206f0ccfa",
    RUNNER: "652ed028610848fcf76a0b2e568c5430df73a097b06a8a5c614be6dff89d2b66",
    RTL: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    ADAPTER: "dfd24f7dbb4122140be8bdb945fe5346c60cc2431a1def7e25f1085df319293c",
    FILELIST: "759a18d9c975ed912b8c75eeeb92b527afb46185c8f8e64f50a8e83f76d86996",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_PASS = (
    "PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED "
    "rows=48 issues=576 products=9216 commits=24 bundles_base=576 "
    "bundles_tsbg=144 scalar_base=4608 scalar_tsbg=1152 stale=1 "
    "retired_replay=1 replay_accept=0 reset=2 recovery=1"
)

PHASES = (
    "reset", "full_load", "full_execute", "retired_replay",
    "replay_reset_recovery", "stale_attack", "stale_reset_recovery",
    "recovery_load", "recovery_execute", "final_checks",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_directory_seal(directory, expected_rows):
    assert directory.is_dir() and not directory.is_symlink()
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    outer_digest, outer_name = outer.read_text().strip().split(maxsplit=1)
    assert outer_name.lstrip(" *") == manifest.name
    assert outer_digest == sha(manifest)
    rows = manifest.read_text().splitlines()
    assert len(rows) == expected_rows
    names = []
    for row in rows:
        digest, relative = row.split(maxsplit=1)
        relative = relative.lstrip(" *")
        assert relative not in names
        names.append(relative)
        target = directory / relative
        assert target.is_file() and not target.is_symlink()
        assert sha(target) == digest


def verify_namespace():
    stem = "m2025_m2018_c2_tsbg_b4_divfree_directed_vcs"
    related = sorted(item.name for item in RESULTS.iterdir() if stem in item.name)
    assert related == [
        ".m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_attempt_consumed",
        "m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_r1_20260902",
    ]
    assert not (RESULTS / ".m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_launch_lock").exists()


def verify_receipts():
    assert RUN_COMPLETE.read_text() == (
        "RAW_PASS_M2025_M2018_DIVFREE_DIRECTED_VCS_PENDING_INDEPENDENT_RESULT_REVIEW\n")
    assert ATTEMPT_RECORD.read_text() == (
        "status=M2025_ATTEMPT_CONSUMED\n"
        "license_queries=1\n"
        "vcs_compiles=1\n"
        "simv_runs=1\n"
        "retry=false\n")
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["status"] == "RAW_PASS_M2025_M2018_DIVFREE_DIRECTED_VCS_PENDING_INDEPENDENT_RESULT_REVIEW"
    assert receipt["license_queries"] == 1
    assert receipt["vcs_compiles"] == 1
    assert receipt["simv_runs"] == 1
    assert receipt["automatic_retry"] is False
    assert receipt["directed_geometry"] == {
        "bundle": 4, "source_groups": 12, "sources_per_group": 16,
        "lanes": 16,
    }
    assert receipt["production_g48_dynamic"] is False
    assert receipt["m1970_phase_pairs"] == 10
    assert receipt["m1970_load_begin"] == 52
    assert receipt["m1970_load_complete"] == 52
    assert receipt["m1970_load_timeout"] == 0
    assert receipt["claim_boundary"] == {
        "behavioral_rtl_directed_only": True,
        "same_area": False,
        "exact_cycle_speedup": False,
        "system_speedup": False,
        "paper_admitted": False,
        "headline": False,
    }


def verify_compile(text):
    assert text.count("Chronologic VCS (TM)") == 1
    assert text.count("Version V-2023.12-SP1_Full64") == 1
    parsed = re.findall(r"^Parsing design file '([^']+)'$", text, re.M)
    assert parsed == [row for row in FILELIST.read_text().splitlines()]
    assert text.count("Top Level Modules:\n       tb_m1880_c2_tsbg_b4_real_channel_signed_frontend") == 1
    assert text.count("7 modules and 0 UDP read.") == 1
    assert text.count("All of 7 modules done") == 1
    forbidden = (
        r"Error-", r"Warning-\[SVAA-RNF\]",
        r"Ignoring.*global_finish_maxfail",
        r"global_finish_maxfail.*(?:ignored|unknown)",
        r"Unknown.*global_finish_maxfail",
    )
    for pattern in forbidden:
        assert not re.search(pattern, text, re.I), pattern
    # The 24 accepted KUAI warnings are confined to frozen TB task variables;
    # the M2018 DUT and M2020 adapter do not appear as their source location.
    warning_positions = [match.start() for match in re.finditer(r"Warning-\[KUAI\]", text)]
    assert len(warning_positions) == 24
    for position in warning_positions:
        block = text[position:text.find("\n\n", position)]
        assert "tb_m1984/tb_m1984_c2_tsbg_b4_parseable_pass.sv" in block
        assert "rtl_m2018" not in block and "rtl_m2020" not in block


def verify_sim(text):
    assert text.count("Chronologic VCS simulator copyright") == 1
    assert text.count("Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64") == 1
    assert len(re.findall(r"^PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED", text, re.M)) == 1
    assert len(re.findall(r"^" + re.escape(EXPECTED_PASS) + r"$", text, re.M)) == 1
    for phase in PHASES:
        assert len(re.findall(r"^M1970_PHASE " + re.escape(phase) + r"_begin\b", text, re.M)) == 1
        assert len(re.findall(r"^M1970_PHASE " + re.escape(phase) + r"_complete\b", text, re.M)) == 1
    assert len(re.findall(r"^M1970_LOAD_BEGIN\b", text, re.M)) == 52
    assert len(re.findall(r"^M1970_LOAD_COMPLETE\b", text, re.M)) == 52
    assert len(re.findall(r"^M1970_LOAD_TIMEOUT\b", text, re.M)) == 0
    forbidden = (
        r"Warning-\[SVAA-RNF\]", r": started at .* failed at",
        r"Assertion[^\n]*failed", r"Error-\[SVA", r"\$(?:error|fatal)",
        r"Fatal:", r"whole-test watchdog expired", r"directed timeout",
        r"post-reset legal-service timeout",
    )
    for pattern in forbidden:
        assert not re.search(pattern, text, re.I), pattern
    assert text.count("$finish called from file") == 1
    tsbg_covers = re.findall(
        r"sva_tsbg\.(cp_[A-Za-z0-9_]+), ([0-9]+) attempts, ([0-9]+) match",
        text)
    assert len(tsbg_covers) == 11
    assert len(set(name for name, _, _ in tsbg_covers)) == 11
    assert all(int(attempts) > 0 and int(matches) > 0
               for _, attempts, matches in tsbg_covers)


def verify_authority():
    verify_directory_seal(M2024, 5)
    review = json.loads(M2024_REVIEW.read_text())
    assert review["status"] == "PASS_M2024_M2020_M2018_C2_TSBG_B4_DIVFREE_DIRECTED_VCS_SOURCE_HAMMER"
    assert review["score_over_100"] >= 95
    assert review["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}
    assert review["authorization"] == {
        "license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
        "all_other_eda_runs": 0, "automatic_retry": False,
    }
    runner = RUNNER.read_text()
    assert runner.count('"${LMUTIL}" lmstat -a') == 1
    assert len(re.findall(r'^\s*"\$\{VCS\}"\s+-full64', runner, re.M)) == 1
    assert len(re.findall(r'180s\s+"\$\{WORK\}/simv"', runner)) == 1


def verify_mutations(compile_text, sim_text):
    compile_mutations = [
        compile_text + "\nError-[INJECTED]\n",
        compile_text.replace("All of 7 modules done", "All of 6 modules done", 1),
        compile_text.replace("tb_m1984/tb_m1984_c2_tsbg_b4_parseable_pass.sv", "rtl_m2018/injected.sv", 1),
    ]
    sim_mutations = [
        sim_text + "\n" + EXPECTED_PASS + "\n",
        sim_text.replace("M1970_PHASE final_checks_complete", "M1970_PHASE final_checks_missing", 1),
        sim_text.replace("M1970_LOAD_COMPLETE", "M1970_LOAD_TIMEOUT", 1),
        sim_text + "\nAssertion injected failed\n",
        sim_text.replace("rows=48", "rows=47", 1),
    ]
    rejected = 0
    for mutant in compile_mutations:
        try:
            verify_compile(mutant)
        except AssertionError:
            rejected += 1
    for mutant in sim_mutations:
        try:
            verify_sim(mutant)
        except AssertionError:
            rejected += 1
    assert rejected == len(compile_mutations) + len(sim_mutations)
    return rejected


def main():
    for path, digest in EXPECTED.items():
        assert path.is_file() and not path.is_symlink(), path
        assert sha(path) == digest, (path, sha(path), digest)
    verify_directory_seal(RESULT, 17)
    verify_directory_seal(ATTEMPT, 1)
    verify_namespace()
    verify_receipts()
    verify_authority()
    assert LICENSE.read_text().count("Flexible License Manager status on") == 1
    assert LICENSE.read_text().count("license server UP (MASTER)") == 1
    compile_text = COMPILE.read_text()
    sim_text = SIM.read_text()
    verify_compile(compile_text)
    verify_sim(sim_text)
    rejected = verify_mutations(compile_text, sim_text)
    print("PASS_M2026_INDEPENDENT_RESULT_HAMMER mutations={0}/{0} pass=1 phases=10 loads=52/52 tsbg_covers=11 p0=0 p1=0 p2=0".format(rejected))


if __name__ == "__main__":
    main()
