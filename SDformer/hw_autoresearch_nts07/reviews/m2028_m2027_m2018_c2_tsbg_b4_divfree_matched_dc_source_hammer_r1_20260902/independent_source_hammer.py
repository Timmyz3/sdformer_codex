#!/usr/bin/env python3
"""Independent M2028 source hammer; never invokes EDA or license tools."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_m2029_m2028_m2027_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_one_shot.sh"
FILELIST = HW / "dc_handoff/filelists/iscas_m2027_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc.f"
RTL = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
M2026_DIR = HW / "reviews/m2026_m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_result_hammer_r1_20260902"
M2026 = M2026_DIR / "review.json"
M1866_DIR = HW / "reviews/m1866_tsbg_ep34_same_io_b2_b4_b8_quickkill_independent_hammer_r1_20260902"
M1866 = M1866_DIR / "review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RUNNER: "3123b5a2809f413f14eeaf060ff172bf060c833f7b64712a0dcc799256a367b4",
    FILELIST: "4850aa79e6194612cc1bf935d1a9f8714d421a18aae4b10de22fc68773410cd3",
    RTL: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    TCL: "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    SDC: "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    M2026: "6033a37048dbba8e5d4ed555da9c1e81748330c657ca5a4bc080c1924bc2ac47",
    M1866: "6560b3660d247440691d31dea7cccd0ca0294cd203c7f2d957a183116eb81830",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dir_seal(directory):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert directory.is_dir() and not directory.is_symlink()
    assert manifest.is_file() and outer.is_file()
    digest, name = outer.read_text().strip().split(maxsplit=1)
    assert name.lstrip(" *") == manifest.name and digest == sha(manifest)
    for row in manifest.read_text().splitlines():
        digest, relative = row.split(maxsplit=1)
        target = directory / relative.lstrip(" *")
        assert target.is_file() and not target.is_symlink() and sha(target) == digest


def verify_filelist(text):
    assert text.splitlines() == [
        "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv",
    ]
    assert "m1995" not in text.lower() and "m1880" not in text.lower()


def verify_rtl(text):
    required = (
        "parameter int SCHEDULE_MODE = 1",
        "parameter int BUNDLE = 4",
        "parameter int SOURCE_GROUPS = 48",
        "parameter int SOURCES_PER_GROUP = 16",
        "parameter int CACHE_ROWS = 4",
        "parameter int LANES = 16",
        "localparam int ORDERED_ROWS = 192;",
        "localparam int PRODUCTION_SOURCE_GROUPS = 48;",
        "if (SCHEDULE_MODE == 0) begin : g_token_major_order",
        "begin : g_group_major_order",
        "ordered_active_row[block * PRIORITY_LANES + lane]",
        "ordered_sign_row[block * PRIORITY_LANES + lane]",
    )
    for token in required:
        assert token in text, token
    active = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    active = re.sub(r'"(?:\\.|[^"\\])*"', '""', active)
    active = "\n".join(row for row in active.splitlines()
                       if not row.lstrip().startswith("`timescale"))
    assert not re.search(r"(?<!/)/(?![/*])", active)
    assert "%" not in active


def verify_runner(text):
    pins = (
        'sha_exact 96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21 "${RTL}"',
        'sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${M803}"',
        'sha_exact c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe "${TCL}"',
        'sha_exact 808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5 "${SDC}"',
        'sha_exact 6033a37048dbba8e5d4ed555da9c1e81748330c657ca5a4bc080c1924bc2ac47 "${M2026}"',
        'sha_exact 6560b3660d247440691d31dea7cccd0ca0294cd203c7f2d957a183116eb81830 "${M1866}"',
        'sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"',
    )
    for token in pins:
        assert text.count(token) == 1, token
    required = (
        "set -euo pipefail", "[[ $# -eq 0 ]]", "verify_dir_seal()", "seal_dir()",
        'verify_dir_seal "${M2026_DIR}"', 'verify_dir_seal "${M1866_DIR}"',
        'verify_dir_seal "${SOURCE_REVIEW_DIR}"', "M2029_EXPECTED_RUNNER_SHA256",
        "M2029_EXPECTED_SOURCE_REVIEW_SHA256", "same-UID DC collision",
        '[[ "${mem_available}" -ge 50331648 && $((commit_limit-committed)) -ge 33554432 ]]',
        'axis_names=(ordinary_lru4 tsbg_b4)', 'axis_modes=(0 1)',
        'for index in 0 1; do', 'ELAB_PARAMETERS="SCHEDULE_MODE=${mode}"',
        '--kill-after=60s 21600s', "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
        "failed_or_incomplete.$$.quarantine", 'seal_dir "${WORK}"',
        'mv -T -- "${WORK}" "${RESULT}"',
        "reports/area.rpt", "reports/timing_setup.rpt",
        "reports/timing_hold_diagnostic.rpt", "reports/port_count.txt",
        "TIM-209=0", "OPT-150=0", "This design has no violated constraints.",
        "absolute_area_delta_at_most_10pct", "both_setup_met",
        "public_ports_equal", "m1866_cpu_premodel_speedup_not_upgraded_to_rtl",
        "'hold_closed': False", "'power': False", "'energy': False",
        "'exact_rtl_cycle_speedup': False", "'same_area': False",
        "'system_speedup': False", "'paper_ppa_ready': False",
        "'production_g48_dynamically_verified': False",
        "'cpu_premodel_2p533808x_upgraded_by_dc': False",
        "'physical_schedule_ablation_not_full_conventional_baseline': True",
        "'state_arrays_synthesized_as_standard_cells': True",
    )
    for token in required:
        assert token in text, token
    assert text.count('"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f Design-Compiler') == 1
    assert len(re.findall(r'^\s*"\$\{DC_SHELL\}" -f "\$\{TCL\}"', text, re.M)) == 1
    assert text.count('ELAB_PARAMETERS="SCHEDULE_MODE=${mode}"') == 1
    assert not re.search(r'^\s*"?\$?\{?(?:VCS|PT_SHELL|FM_SHELL|ICC2_SHELL)', text, re.M | re.I)
    assert "lmstat -a" not in text and "retry=true" not in text
    order = (
        'verify_dir_seal "${SOURCE_REVIEW_DIR}"',
        '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}"',
        "same-UID DC collision",
        'mem_available="$(awk',
        'mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"',
        'seal_dir "${ATTEMPT}"',
        '"${LMUTIL}" lmstat -c',
        'for index in 0 1; do',
        '"${DC_SHELL}" -f "${TCL}"',
        "if not all(receipt['candidate_gate'].values())",
        'seal_dir "${WORK}"',
        'mv -T -- "${WORK}" "${RESULT}"',
    )
    cursor = -1
    for token in order:
        cursor2 = text.index(token, cursor + 1)
        assert cursor2 > cursor, token
        cursor = cursor2


def reject_mutations(runner, filelist):
    cases = [
        ("duplicate_mode", runner.replace("axis_modes=(0 1)", "axis_modes=(0 0)"), filelist),
        ("third_axis", runner.replace("for index in 0 1; do", "for index in 0 1 2; do"), filelist),
        ("extra_elab_axis", runner.replace('ELAB_PARAMETERS="SCHEDULE_MODE=${mode}"', 'ELAB_PARAMETERS="SCHEDULE_MODE=${mode},CACHE_ROWS=8"'), filelist),
        ("extra_license", runner.replace(
            '"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f Design-Compiler',
            '"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f Design-Compiler\n'
            '"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f Design-Compiler', 1), filelist),
        ("extra_dc", runner.replace('"${DC_SHELL}" -f "${TCL}"', '"${DC_SHELL}" -f "${TCL}"\n      "${DC_SHELL}" -f "${TCL}"', 1), filelist),
        ("weak_memory", runner.replace("50331648", "1"), filelist),
        ("weak_commit", runner.replace("33554432", "1"), filelist),
        ("drop_collision", runner.replace("same-UID DC collision", "collision disabled"), filelist),
        ("drop_quarantine", runner.replace("FAILED_OR_INCOMPLETE_DO_NOT_CITE", "FAILED"), filelist),
        ("drop_area_artifact", runner.replace("reports/area.rpt", "reports/not_area.rpt"), filelist),
        ("false_to_true", runner.replace("'system_speedup': False", "'system_speedup': True"), filelist),
        ("drop_review_seal", runner.replace('verify_dir_seal "${SOURCE_REVIEW_DIR}"', "true", 1), filelist),
        ("wrong_timeout", runner.replace("21600s", "99999s"), filelist),
        ("m1995_filelist", runner, filelist + "rtl_m1995/fake.sv\n"),
    ]
    rejected = 0
    for name, mutated_runner, mutated_filelist in cases:
        try:
            verify_filelist(mutated_filelist)
            verify_runner(mutated_runner)
        except (AssertionError, ValueError):
            rejected += 1
        else:
            raise AssertionError("mutation survived: " + name)
    return len(cases), rejected


def main():
    for path, expected in EXPECTED.items():
        assert path.is_file() and not path.is_symlink(), path
        assert sha(path) == expected, (path, sha(path), expected)
    verify_dir_seal(M2026_DIR)
    verify_dir_seal(M1866_DIR)
    m2026 = json.loads(M2026.read_text())
    assert m2026["status"].startswith("PASS_M2026")
    assert m2026["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}
    assert m2026["claim_boundary"]["production_g48_dynamic"] is False
    m1866 = json.loads(M1866.read_text())
    assert m1866["authorization"]["b4_new_fail_closed_source_contract_may_be_authored"] is True
    assert m1866["claim_boundary"]["cpu_premodel"] is True
    filelist = FILELIST.read_text()
    runner = RUNNER.read_text()
    verify_filelist(filelist)
    verify_rtl(RTL.read_text())
    verify_runner(runner)
    total, rejected = reject_mutations(runner, filelist)
    print("PASS_M2028_INDEPENDENT_SOURCE_HAMMER mutations={0}/{1} p0=0 p1=0 p2=0 no_eda=1 no_license=1".format(rejected, total))


if __name__ == "__main__":
    main()
