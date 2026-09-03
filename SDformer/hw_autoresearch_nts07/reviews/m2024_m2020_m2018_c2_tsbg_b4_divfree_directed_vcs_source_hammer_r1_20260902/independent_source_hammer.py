#!/usr/bin/env python3
"""Independent M2024 source hammer.  Never invokes EDA or a license query."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RTL = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
ADAPTER = HW / "rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA = HW / "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv"
TB = HW / "tb_m1984/tb_m1984_c2_tsbg_b4_parseable_pass.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs.f"
RUNNER = HW / "dc_handoff/scripts/run_m2025_m2024_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs_one_shot.sh"
TEST = HW / "system_simulator/tests/test_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs_source.py"
CONTRACT = HW / "contracts/m2020_m2018_c2_tsbg_b4_divfree_directed_vcs_source_contract_r1_20260902.json"
M2019_DIR = HW / "reviews/m2019_m2018_c2_tsbg_b4_divfree_fair_scheduler_source_hammer_r1_20260902"
M2019_REVIEW = M2019_DIR / "review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RTL: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    ADAPTER: "dfd24f7dbb4122140be8bdb945fe5346c60cc2431a1def7e25f1085df319293c",
    M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    SVA: "e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2",
    TB: "d46a47dada89e16cdc3f2593020a89e3513060a8a1a03ae3a1963d0483b96081",
    FILELIST: "759a18d9c975ed912b8c75eeeb92b527afb46185c8f8e64f50a8e83f76d86996",
    RUNNER: "652ed028610848fcf76a0b2e568c5430df73a097b06a8a5c614be6dff89d2b66",
    TEST: "88551ac9c24949196f28346271359a84570efbfc0d03ba427fcdcf9f228b8267",
    CONTRACT: "82d85453e2cd6f41ca139f950801deab4902c9154f49d0e91d6fbc659cb3aff7",
    M2019_REVIEW: "bf1cfc2d1090f5932419e19921a3cd1966adbbec5585ad446e43f9bcb266477d",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PARAMETERS = (
    "SCHEDULE_MODE", "BUNDLE", "SOURCE_GROUPS", "SOURCES_PER_GROUP",
    "OUTPUT_SLICES", "CACHE_ROWS", "TAG_BITS", "CHANNEL_BITS",
    "EPOCH_BITS", "GENERATION_BITS", "LANES",
)

EXPECTED_FILELIST = [
    "hw_autoresearch_nts07/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
    "hw_autoresearch_nts07/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv",
    "hw_autoresearch_nts07/rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv",
    "hw_autoresearch_nts07/verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv",
    "hw_autoresearch_nts07/tb_m1984/tb_m1984_c2_tsbg_b4_parseable_pass.sv",
]

EXPECTED_PASS = (
    "PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED "
    "rows=48 issues=576 products=9216 commits=24 bundles_base=576 "
    "bundles_tsbg=144 scalar_base=4608 scalar_tsbg=1152 stale=1 "
    "retired_replay=1 replay_accept=0 reset=2 recovery=1"
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


def verify_directory_seal(directory):
    assert directory.is_dir() and not directory.is_symlink()
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    outer_digest, outer_name = outer.read_text().strip().split(maxsplit=1)
    assert outer_name.lstrip(" *") == manifest.name
    assert sha(manifest) == outer_digest
    for row in manifest.read_text().splitlines():
        digest, relative = row.split(maxsplit=1)
        target = directory / relative.lstrip(" *")
        assert target.is_file() and not target.is_symlink()
        assert sha(target) == digest


def verify_file_seal(path):
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(manifest) + ".seal.sha256")
    assert path.is_file() and not path.is_symlink()
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    digest, name = manifest.read_text().strip().split(maxsplit=1)
    assert name.lstrip(" *") == path.name and digest == sha(path)
    seal_digest, seal_name = outer.read_text().strip().split(maxsplit=1)
    assert seal_name.lstrip(" *") == manifest.name
    assert seal_digest == sha(manifest)


def normalized_header(text):
    begin = text.index("#(", text.index("module "))
    end = text.index(");", begin) + 2
    header = text[begin:end]
    header = re.sub(r"/\*.*?\*/", "", header, flags=re.S)
    header = re.sub(r"//[^\n]*", "", header)
    return re.sub(r"\s+", " ", header).strip()


def active_source(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return "\n".join(row for row in text.splitlines()
                     if not row.lstrip().startswith("`timescale"))


def verify_adapter_and_rtl(adapter, rtl):
    assert normalized_header(adapter) == normalized_header(rtl)
    assert adapter.count(
        "module m1880_c2_tsbg_b4_real_channel_signed_frontend #(") == 1
    assert adapter.count(
        "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #(") == 1
    for parameter in PARAMETERS:
        assert adapter.count(".{0}({0})".format(parameter)) == 1
    body = adapter.split(");", 1)[1].split("endmodule", 1)[0]
    assert body.count(") implementation (.*);") == 1
    assert body.count(".*") == 1
    assert not re.search(r"\b(always|always_comb|always_ff|always_latch|assign|initial|final|force|release)\b", body)
    assert len(re.findall(r"\bmodule\s+\w+", adapter)) == 1

    active = active_source(rtl)
    assert not re.search(r"(?<!/)/(?![/*])", active)
    assert "%" not in active
    assert "active_q" not in active and "sign_q" not in active
    assert "scan_linear_q" not in active and "find_linear" not in active
    assert "[current_context_q][current_group_q]" not in active
    assert "logic [ORDERED_ROWS-1:0] ordered_row_live;" in active
    assert "localparam int ORDERED_ROWS = 192;" in active
    # One generate-time ordering branch and one constant-folded clear branch.
    assert active.count("if (SCHEDULE_MODE == 0)") == 2
    assert active.count("ordered_active_row[block * PRIORITY_LANES + lane]") == 1
    assert active.count("ordered_sign_row[block * PRIORITY_LANES + lane]") == 1
    assert "current_active_row_q <= find_active_row;" in active
    assert "current_sign_row_q <= find_sign_row;" in active


def verify_filelist(text):
    rows = text.splitlines()
    assert rows == EXPECTED_FILELIST
    assert len(rows) == len(set(rows)) == 5
    assert "m1995" not in text.lower()
    assert "+incdir" not in text
    for row in rows:
        path = ROOT / row
        assert path.is_file() and not path.is_symlink()


def verify_tb_and_sva(tb, sva):
    assert tb.count("module tb_m1880_c2_tsbg_b4_real_channel_signed_frontend;") == 1
    assert tb.count("localparam int BUNDLE=4, GROUPS=12, SLICES=6, LANES=16;") == 1
    assert tb.count("`CONNECT_M1880(dut_base, base, 0, load_valid_base);") == 1
    assert tb.count("`CONNECT_M1880(dut_tsbg, tsbg, 1, load_valid_tsbg);") == 1
    # One occurrence in each of the DUT and SVA connection macros; each macro
    # is expanded twice for MODE=0 and MODE=1.
    assert tb.count(".SOURCE_GROUPS(GROUPS)") == 2
    assert tb.count("PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED") == 1
    for token in (
            "full_base_done_cycle * 1.0 / full_tsbg_done_cycle < 1.15",
            "M1970 whole-test watchdog expired",
            "M1880 directed timeout",
            "M1880 post-reset legal-service timeout",
            "M1880 retired legal identity replay was accepted",
            "M1880 exact -(-128)=+128 bridge corner missing"):
        assert token in tb
    assert sva.count(
        "module m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions #(") == 1
    assert len(re.findall(r"\bassert\s+property\s*\(", sva)) >= 20
    assert len(re.findall(r"\bcover\s+property\s*\(", sva)) >= 10


def verify_runner(text):
    pins = [
        'sha_exact 96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21 "${RTL}"',
        'sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${M803}"',
        'sha_exact dfd24f7dbb4122140be8bdb945fe5346c60cc2431a1def7e25f1085df319293c "${ADAPTER}"',
        'sha_exact e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2 "${SVA}"',
        'sha_exact d46a47dada89e16cdc3f2593020a89e3513060a8a1a03ae3a1963d0483b96081 "${TB}"',
        'sha_exact 759a18d9c975ed912b8c75eeeb92b527afb46185c8f8e64f50a8e83f76d86996 "${FILELIST}"',
        'sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"',
        'sha_exact bf1cfc2d1090f5932419e19921a3cd1966adbbec5585ad446e43f9bcb266477d "${M2019_REVIEW}"',
    ]
    for token in pins:
        assert text.count(token) == 1, token
    assert text.count('verify_dir_seal "${M2019_DIR}"') == 1
    assert text.count('verify_dir_seal "${M2024_DIR}"') == 1
    assert text.count("M2025_EXPECTED_RUNNER_SHA256") == 2
    assert text.count("M2025_EXPECTED_M2024_REVIEW_SHA256") == 2
    assert text.count('"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"') == 1
    assert len(re.findall(r'^\s*"\$\{VCS\}"\s+-full64\s+-sverilog', text, re.M)) == 1
    assert len(re.findall(r'180s\s+"\$\{WORK\}/simv"', text)) == 1
    assert not re.search(r"\b(dc_shell|pt_shell|fm_shell|icc2_shell)\s+-", text)
    assert "for attempt" not in text.lower()

    ordering = [
        'verify_dir_seal "${M2024_DIR}"',
        '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}"',
        'for proc in /proc/[0-9]*; do',
        'mem_available="$(awk',
        'mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"',
        'seal_dir "${ATTEMPT}"',
        '"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"',
        '"${VCS}" -full64 -sverilog',
        '180s "${WORK}/simv"',
        'grep -Fxc "${EXPECTED_PASS}"',
        'seal_dir "${WORK}"',
        'mv -T -- "${WORK}" "${RESULT}"',
    ]
    cursor = -1
    for token in ordering:
        position = text.index(token, cursor + 1)
        assert position > cursor, token
        cursor = position

    required = (
        "set -euo pipefail", "[[ $# -eq 0 ]]", "sha_file()",
        "sha_exact()", "verify_dir_seal()", "seal_dir()", "trap on_exit",
        '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]]',
        '[[ "${real_uid}" == "${EUID}" ]]',
        "commit_limit-committed", "16777216", "WORK_ACTIVE=1",
        "FAILED_OR_INCOMPLETE_DO_NOT_CITE", "failed_or_incomplete.$$.quarantine",
        "-assert svaext", "-assert global_finish_maxfail=1",
        "--signal=TERM --kill-after=10s 180s", EXPECTED_PASS,
        "grep -Fc 'PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED'",
        "grep -Fxc \"${EXPECTED_PASS}\"", "M1970_LOAD_BEGIN",
        "M1970_LOAD_COMPLETE", "M1970_LOAD_TIMEOUT",
        '"production_g48_dynamic": false', '"same_area": false',
        '"exact_cycle_speedup": false', '"system_speedup": false',
        '"paper_admitted": false', '"headline": false',
        '"automatic_retry": false', "WORK_ACTIVE=0",
    )
    for token in required:
        assert token in text, token
    assert text.count("16777216") == 2
    phases = (
        "reset", "full_load", "full_execute", "retired_replay",
        "replay_reset_recovery", "stale_attack", "stale_reset_recovery",
        "recovery_load", "recovery_execute", "final_checks",
    )
    phase_loop = "for phase in {0}; do".format(" ".join(phases))
    assert text.count(phase_loop) == 1
    assert text.count('grep -Fc "M1970_PHASE ${phase}_begin"') == 1
    assert text.count('grep -Fc "M1970_PHASE ${phase}_complete"') == 1
    assert text.count("lmstat -a") == 1
    assert "retry=true" not in text
    assert text.count("automatic_retry': False") == 1
    assert text.count('"simv_runs": 1') == 1
    assert text.count('"license_queries": 1') == 1
    assert text.count('"vcs_compiles": 1') == 1
    assert text.count("'all_other_eda_runs': 0") == 1


def verify_contract():
    verify_file_seal(CONTRACT)
    contract = json.loads(CONTRACT.read_text())
    assert contract["status"] == "SOURCE_ONLY_M2020_M2018_DIVFREE_DIRECTED_VCS_TOOLCHAIN__NO_EDA"
    assert contract["author_execution"] == {
        "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0,
        "dc_runs": 0, "pt_runs": 0, "fm_runs": 0, "gpu_runs": 0,
        "attempts": 0, "results": 0, "reviews_m2024": 0,
    }
    assert all(value is False for value in contract["authorization"].values())
    for path, digest in EXPECTED.items():
        relative = str(path.relative_to(ROOT))
        if relative in contract["source_sha256"]:
            assert contract["source_sha256"][relative] == digest
    assert contract["future_authority"]["source_review"] == "M2024"
    assert contract["future_authority"]["one_shot_execution"] == "M2025"
    assert contract["future_authority"]["exact_m2024_review_sha_pin_required"] is True
    assert contract["claim_boundary"]["directed_g12_only"] is True
    assert contract["claim_boundary"]["production_g48_dynamic"] is False


def mutation_check(runner):
    mutations = [
        runner.replace("M2025_EXPECTED_RUNNER_SHA256", "M2025_UNPINNED_RUNNER", 2),
        runner.replace("M2025_EXPECTED_M2024_REVIEW_SHA256", "M2025_UNPINNED_REVIEW", 2),
        runner.replace('verify_dir_seal "${M2024_DIR}"', ":", 1),
        runner.replace(EXPECTED[FILELIST], "0" * 64, 1),
        runner.replace('for proc in /proc/[0-9]*; do', 'for proc in /proc/none; do', 1),
        runner.replace("16777216", "1"),
        runner.replace('"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"',
                       '"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"\n"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"', 1),
        runner.replace('"${VCS}" -full64 -sverilog',
                       '"${VCS}" -full64 -sverilog\n"${VCS}" -full64 -sverilog', 1),
        runner.replace('180s "${WORK}/simv"',
                       '180s "${WORK}/simv"\n180s "${WORK}/simv"', 1),
        runner.replace("--kill-after=10s 180s", "--kill-after=10s 181s", 1),
        runner.replace("global_finish_maxfail=1", "global_finish_maxfail=2", 1),
        runner.replace('grep -Fxc "${EXPECTED_PASS}"', 'grep -Fc "${EXPECTED_PASS}"', 1),
        runner.replace('"production_g48_dynamic": false', '"production_g48_dynamic": true', 1),
        runner.replace('"paper_admitted": false', '"paper_admitted": true', 1),
        runner.replace('"automatic_retry": false', '"automatic_retry": true', 1),
        runner.replace('mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine"', ":", 1),
    ]
    rejected = 0
    for mutant in mutations:
        try:
            verify_runner(mutant)
        except (AssertionError, ValueError):
            rejected += 1
    assert rejected == len(mutations)
    return rejected


def main():
    for path, digest in EXPECTED.items():
        assert path.is_file() and not path.is_symlink(), path
        assert sha(path) == digest, path
    verify_directory_seal(M2019_DIR)
    m2019 = json.loads(M2019_REVIEW.read_text())
    assert m2019["status"] == "PASS_M2019_M2018_C2_TSBG_B4_DIVFREE_FAIR_SCHEDULER_SOURCE_HAMMER"
    assert m2019["score_over_100"] >= 95
    assert m2019["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}
    assert m2019["authorization"]["next_vcs_source_authoring"] is True
    assert m2019["authorization"]["eda_execution"] is False

    adapter = ADAPTER.read_text()
    rtl = RTL.read_text()
    runner = RUNNER.read_text()
    verify_adapter_and_rtl(adapter, rtl)
    verify_filelist(FILELIST.read_text())
    verify_tb_and_sva(TB.read_text(), SVA.read_text())
    verify_runner(runner)
    verify_contract()
    subprocess.check_call(["/bin/bash", "-n", str(RUNNER)])
    rejected = mutation_check(runner)
    print("PASS_M2024_INDEPENDENT_SOURCE_HAMMER mutations={0}/{0} rows=5 g12_only=1 p0=0 p1=0 p2=0".format(rejected))


if __name__ == "__main__":
    main()
