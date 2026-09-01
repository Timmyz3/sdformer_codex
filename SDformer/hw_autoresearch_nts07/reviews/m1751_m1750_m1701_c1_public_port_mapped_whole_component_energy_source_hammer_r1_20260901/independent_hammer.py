#!/usr/bin/env python3
"""Independent, zero-EDA M1750 source and accounting mutation hammer."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1750_m1701_c1_public_port_mapped_whole_component_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1750_m1701_c1_public_port_mapped_whole_component_energy_source_author_receipt_r1_20260901"
M1745 = HW / "reviews/m1745_m1739_m1701_c1_public_port_mapped_production_energy_source_hammer_r1_20260901"
M1743 = HW / "contracts/m1743_m1742_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_release_r1_20260901.json"
TIMING = HW / "dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901"
LEDGER = HW / "results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901/ep34_c1_support16_rows.memh"
TB = HW / "dc_handoff/tb/tb_m1739_c1_m1701_public_port_mapped_production_energy.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1739_c1_m1701_public_port_mapped_production_energy.f"
UCLI = HW / "dc_handoff/scripts/m1739_c1_m1701_public_port_mapped_production_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1750_c1_m1701_public_port_mapped_whole_component_energy.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1750_m1701_c1_public_port_mapped_whole_component_energy_one_shot.py"
CHECKER = HW / "system_simulator/scripts/check_m1750_c1_m1701_public_port_mapped_whole_component_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1750_c1_m1701_public_port_mapped_whole_component_energy_source.py"
M1752 = HW / "contracts/m1752_m1751_m1750_m1701_c1_public_port_mapped_whole_component_energy_launch_release_r1_20260901.json"
ATTEMPT = HW / "results/.m1750_c1_public_port_mapped_component_energy_attempt_consumed"
RESULT = HW / "results/m1750_c1_public_port_mapped_component_energy_r1_20260901"

EXPECTED_SOURCE = {
    "dc_handoff/tb/tb_m1739_c1_m1701_public_port_mapped_production_energy.sv": "efccfc7b8eca975958e4d13596a604ae469d711fab7b67284c9fb90982baaa9b",
    "dc_handoff/filelists/date_m1739_c1_m1701_public_port_mapped_production_energy.f": "016bbe13849909b260c2f3dad24164fa7176a1624e80508fc3d3ad8d56afbff6",
    "dc_handoff/scripts/m1739_c1_m1701_public_port_mapped_production_energy.ucli.tcl": "ec798508ed37410d2a13c40bb5c255de52583adcbc26b9acab967211b1d5f396",
    "dc_handoff/scripts/run_ptpx_m1750_c1_m1701_public_port_mapped_whole_component_energy.tcl": "1b9fdb335290e2e7dc14b3cdc1a0cbf3dbe63ed0ca691226762b037726a184c6",
    "dc_handoff/scripts/run_m1750_m1701_c1_public_port_mapped_whole_component_energy_one_shot.py": "c6ce6ce2eee2a013a835fa3336fe7e2d1c90a464c3972742efb5a4ad54d51f18",
    "system_simulator/scripts/check_m1750_c1_m1701_public_port_mapped_whole_component_energy_source.py": "a9320fa60ad898b2804d2014493ccd6b86ddb27a9eb18ea18b953a91460c38c7",
    "system_simulator/tests/test_m1750_c1_m1701_public_port_mapped_whole_component_energy_source.py": "219b3918f48652a1c8d202c01aaa449b7747686dc4caf57167b8333602ba8a99",
}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "JSON nonregular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_file_seal(path, payload_sha, sidecar_sha, outer_sha):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(sha(path) == payload_sha and sha(sidecar) == sidecar_sha
         and sha(outer) == outer_sha, "file seal identity")
    need(sidecar.read_text() == payload_sha + "  " + path.name + "\n",
         "file sidecar content")
    need(outer.read_text() == sidecar_sha + "  " + sidecar.name + "\n",
         "file outer content")


def verify_dir_seal(root, manifest_sha, outer_sha):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
         "directory seal identity " + str(root))
    need(outer.read_text() == manifest_sha + "  SHA256SUMS\n",
         "directory outer content")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest member")
        need(sha(root / rel) == fields[0], "manifest member drift " + name)
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "sealed population drift " + str(root))


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def tcl_commands(text):
    rows = []
    current = ""
    for raw in text.splitlines():
        row = raw.split("#", 1)[0].strip()
        if not row:
            continue
        if row.endswith("\\"):
            current += row[:-1].strip() + " "
        else:
            rows.append((current + row).strip())
            current = ""
    need(not current, "unterminated Tcl continuation")
    return rows


def load_checker():
    spec = importlib.util.spec_from_file_location("m1750_target", str(CHECKER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def power_report(switching, internal, leakage, total=None):
    if total is None:
        total = switching + internal + leakage
    return ("****************************************\nReport : Averaged Power\n"
            "\t-unit mW\n"
            "Net Switching Power = %.9f\nCell Internal Power = %.9f\n"
            "Cell Leakage Power = %.9f\nTotal Power = %.9f\n" %
            (switching, internal, leakage, total))


def must_fail(function):
    try:
        function()
    except Exception:
        return 1
    raise RuntimeError("negative mutation survived")


def main():
    verify_file_seal(CONTRACT,
        "f893267a0a77d18c23ec1a1f4387c67cdfdeab6ef0214fdd6fbfae97479251af",
        "fcdefe217aa84bea6059983536d3c627b80db675accd8d96d1d0c758aa00f414",
        "47a8c93b95ec94040d97a65f3c58f712f3553296534d603819e7b1a2f017d3fa")
    verify_dir_seal(AUTHOR,
        "02766a34c2513fd4cd2a9b7ebb0cf0f6fabdd62727fab5eeec5b0966516a6bfd",
        "0cab0410622e2b3eed68010a79cdc65d9c415df203b7a527e475025531e08e73")
    verify_dir_seal(M1745,
        "c5b1f83b618ab8aadff16dc9e2a8f6498a852c66559d7a55171f93831bf3595a",
        "f81c8c0166da2d2e6ce7a99aa469bad9d800193edab865c33fe64ab6753c0404")
    need(sha(AUTHOR / "receipt.json") ==
         "e35f0efd0c566ec707026cb707c4f66ddb0b59df0a0ee0c71db5518236784f5d",
         "author receipt")
    need(sha(M1745 / "review.json") ==
         "44fca21fde5163ae39f249f5a485c5f2d4953910d8ff76e911aff6a543373359",
         "M1745 review")
    verify_file_seal(M1743,
        "3c623618115c4ecf2e4bfec6efe167c90296825428ce87e16e6d52bd79216921",
        "7d481d605bffd1386b8926e709424a2c78b3f78eff340caf1727dbe7ec84cfe1",
        "7a52c2e7692b62857dfe1d2b1bd9e2825372a0fc839822abf086d4837bbcf112")
    verify_dir_seal(TIMING,
        "d3f2e14a6f6c0600abce2f5af2479402d41986736e3d9c32c6044e4225f64c75",
        "6f2b17f7016665cd663b9694a1ccbd29fa551ecf75ba29aa52c4bb56c5769b38")
    need(sha(TIMING / "receipt.json") ==
         "0b3ee22f9369a38eb83f674a4f1eb73fac39757ee85a3e1aeebe032bd0c76a1e",
         "timing receipt")

    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "receipt.json")
    failed = strict_json(M1745 / "review.json")
    timing = strict_json(TIMING / "receipt.json")
    need(contract["status"] ==
         "SOURCE_ONLY__M1745_FAIL_BOUND__M1743_TIMING_PINNED__M1751_REVIEW_AND_M1752_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(author["source_contract_sha256"] == sha(CONTRACT)
         and author["author_execution"] == {"eda_attempts_created": 0,
             "license_queries": 0, "m1752_created": False,
             "ptpx_runs": 0, "results_created": 0, "saif_files": 0,
             "simv_runs": 0, "vcs_runs": 0}, "author execution boundary")
    need(failed["status"] ==
         "FAIL_M1745_P0_DO_NOT_AUTHORIZE_M1746__ADDITIVE_PTPX_MACRO_POWER_REPAIR_REQUIRED"
         and failed["p0_count"] == 1 and failed["m1746_authorized"] is False,
         "M1745 fail disposition")
    need(timing["formality"]["passing_compare_points"] == 16549
         and [timing["formality"][key] for key in
              ("failing", "aborted", "unverified", "unmatched")] == [0, 0, 0, 0]
         and timing["prime_time"]["setup_wns_ns"] == "0.027871"
         and timing["prime_time"]["hold_wns_ns"] == "0.001827"
         and timing["prime_time"]["macro_count"] == "9"
         and timing["claim_boundary"]["power"] is False
         and timing["claim_boundary"]["energy"] is False,
         "M1743 timing boundary")
    need(not os.path.lexists(str(M1752)) and not os.path.lexists(str(ATTEMPT))
         and not os.path.lexists(str(RESULT)), "premature launch namespace")

    inventory = dict((row["path"], row["sha256"])
                     for row in contract["source_files"])
    need(inventory == EXPECTED_SOURCE, "source inventory")
    for relative, digest in EXPECTED_SOURCE.items():
        need(sha(HW / relative) == digest, "source drift " + relative)
    need(contract["claim_boundary"] == dict((key, False) for key in (
        "launch_authorized", "launch_executed", "mapped_vcs", "production_saif",
        "ptpx", "logic_power", "component_energy", "total_c1_energy",
        "energy_per_frame", "performance", "system_speedup", "paper_ppa_ready",
        "headline")), "source claim promotion")

    # Recompute the complete frozen ledger, not a sample.
    histogram = [0] * 17
    rows = 0
    with LEDGER.open("rb", buffering=1 << 20) as stream:
        for raw in stream:
            need(len(raw) == 9 and raw[8:] == b"\n" and raw[:4] == b"0000",
                 "ledger row syntax")
            histogram[bin(int(raw[4:8], 16)).count("1")] += 1
            rows += 1
    expected_histogram = [26535787, 7880233, 5335070, 3774342, 2614180,
        1861862, 1383722, 907501, 608784, 448874, 213441, 124172,
        72126, 41560, 22171, 10962, 5213]
    need(rows == 51840000 and histogram == expected_histogram,
         "full support histogram")
    active = rows - histogram[0]
    cumulative = 0
    quantiles = {}
    for support in range(1, 17):
        cumulative += histogram[support]
        for label, numerator in (("p25", 1), ("p50", 2), ("p75", 3)):
            rank = (active * numerator + 3) // 4
            if label not in quantiles and cumulative >= rank:
                quantiles[label] = support
    need(active == 25304213 and quantiles == {"p25": 1, "p50": 2, "p75": 4},
         "support quantiles")

    active_tb = strip_comments(TB.read_text()).lower()
    need("force " not in active_tb and "release " not in active_tb
         and "dut." not in active_tb, "non-public testbench action")
    need([row.split("#", 1)[0].strip() for row in UCLI.read_text().splitlines()
          if row.split("#", 1)[0].strip()] == [
             "power -gate_level all mda sv", "power " +
             "tb_m1739_c1_m1701_public_port_mapped_production_energy.dut",
             "run", "power -enable", "run", "power -disable",
             "power -report $::env(M1739_SAIF_FILE) 1e-9 " +
             "tb_m1739_c1_m1701_public_port_mapped_production_energy.dut",
             "quit"], "UCLI scope/order")

    pt = PT_TCL.read_text()
    commands = tcl_commands(pt)
    reports = [row for row in commands if row.startswith("report_power ")]
    need(len(reports) == 2, "unexpected report_power population")
    need(reports[0] == "report_power -unit mW -nosplit -significant_digits 8 > \"$output_dir/reports/ptpx_whole_mapped_c1_including_9macro_liberty.rpt\"",
         "primary report is not the whole current design")
    need(reports[1].startswith("report_power -hierarchy -area -unit mW ")
         and "$macro_cells" not in reports[1], "diagnostic report boundary")
    need(not any("report_power $macro_cells" in row or "-cell_power" in row
                 or "-net_power" in row for row in reports),
         "selected-cell power report survived")
    for token in ("macro_count != $expected_macro_count",
                  "M1750_FAIL_EXACT_NET_ANNOTATION_GATE",
                  "M1750_FAIL_EXACT_LEAF_ANNOTATION_GATE",
                  "corner_classification=mixed_corner_component_estimate",
                  "standard_cell_power_library=TT_0p9V_25C",
                  "parent_sram_macro_liberty=SSG_0p9V_125C",
                  "not_single_corner_signoff=true", "top_minus_macro=false",
                  "ptpx_plus_datasheet_sram_combined=false"):
        need(token in pt, "PTPX boundary token " + token)
    need("set_operating_conditions tt0p9v25c" in pt
         and "set macro_slow_db" in pt and "set std_tt_db" in pt,
         "mixed library source boundary")

    runner = RUNNER.read_text()
    ordered = ("verify_authority()", "CHECK.validate_sources()",
               "namespaces_fresh()", "fcntl.flock(queue_handle.fileno()",
               "resource_gate()", "probe = subprocess.run", "ATTEMPT.mkdir()",
               'state["vcs_compiles"] += 1', 'state["simv_runs"] += 1',
               'state["ptpx_runs"] += 1', "CHECK.whole_component_power(",
               "seal_dir(STAGE)", "publish_no_replace(STAGE, RESULT)")
    cursor = 0
    for token in ordered:
        position = runner.find(token, cursor)
        need(position >= 0, "runner order " + token)
        cursor = position + len(token)
    for token in ("M1750_EXPECTED_M1751_MANIFEST_SHA256",
                  "M1750_EXPECTED_M1751_OUTER_FILE_SHA256",
                  "M1750_EXPECTED_M1751_REVIEW_SHA256",
                  "M1750_EXPECTED_M1752_RELEASE_SHA256"):
        need(token in runner, "authority pin " + token)
    need("CHECK.combine_power(" not in runner
         and "known_component_workload_energy_pj" not in runner
         and "ptpx_nine_parent_macros" not in runner,
         "old combined accounting survived")

    checker = load_checker()
    source = checker.validate_sources()
    need(source["status"] == "PASS_M1750_SOURCE_ONLY_NO_EDA", "target self-check")
    attacks = {"runtime": 0, "saif": 0, "power": 0}
    good_log = ("M1739_PUBLIC_COUNTERS cycles=777 issue_accepts=145 parent_edges=20 "
                "macro_reads=13 macro_writes=9 forwards=7 dead_write_elisions=55 "
                "psum_commits=64 row_completions=64\n"
                "PASS_M1739_C1_M1701_PUBLIC_PORT_MAPPED_DIRECTED_COMPONENT_ACTIVITY\n")
    good_saif = ("(SAIFILE\n (DURATION 300)\n (INSTANCE " + checker.TOP +
                 "\n  (INSTANCE dut\n   (NET (x (T0 100) (T1 200) "
                 "(TX 0) (TC 9))))))\n")
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        log = root / "sim.log"
        log.write_text(good_log)
        checker.validate_runtime(log)
        for old, new in (("macro_reads=13", "macro_reads=12"),
                         ("macro_writes=9", "macro_writes=8"),
                         ("psum_commits=64", "psum_commits=63"),
                         ("row_completions=64", "row_completions=63"),
                         ("cycles=777", "cycles=0")):
            log.write_text(good_log.replace(old, new))
            attacks["runtime"] += must_fail(lambda: checker.validate_runtime(log))
        saif = root / "x.saif"
        saif.write_text(good_saif)
        checker.validate_saif(saif, 100)
        for changed in (good_saif.replace("DURATION 300", "DURATION 297"),
                        good_saif.replace("TX 0", "TX 1"),
                        good_saif.replace("TC 9", "TC 0"),
                        good_saif.replace("INSTANCE dut", "INSTANCE bad")):
            saif.write_text(changed)
            attacks["saif"] += must_fail(lambda: checker.validate_saif(saif, 100))
        top = root / "top.rpt"
        top.write_text(power_report(3.0, 6.0, 1.0))
        base = checker.whole_component_power(top, 100, 5, 3)
        changed_counts = checker.whole_component_power(top, 100, 500, 300)
        need(base["ptpx_whole_mapped_c1_including_9macro_liberty"] ==
             changed_counts["ptpx_whole_mapped_c1_including_9macro_liberty"],
             "datasheet counters contaminated PTPX primary")
        need(base["parent_sram_datasheet_alternative_sensitivity"] !=
             changed_counts["parent_sram_datasheet_alternative_sensitivity"],
             "datasheet sensitivity did not remain separate")
        need(base["ptpx_plus_datasheet_sram_combined"] is False
             and base["parent_sram_datasheet_alternative_sensitivity"]
                 ["added_to_ptpx_whole_component"] is False,
             "combined accounting flag")
        top.write_text(power_report(3.0, 6.0, 1.0, 9.0))
        attacks["power"] += must_fail(
            lambda: checker.whole_component_power(top, 100, 5, 3))
        top.write_text(power_report(3.0, 6.0, -1.0, 8.0))
        attacks["power"] += must_fail(
            lambda: checker.whole_component_power(top, 100, 5, 3))
        top.write_text(power_report(3.0, 6.0, 1.0) +
                       "Total Power = 10.000000000\n")
        attacks["power"] += must_fail(
            lambda: checker.whole_component_power(top, 100, 5, 3))
        top.write_text(power_report(3.0, 6.0, 1.0).replace("-unit mW", "-unit uW"))
        attacks["power"] += must_fail(
            lambda: checker.whole_component_power(top, 100, 5, 3))
    need(attacks == {"runtime": 5, "saif": 4, "power": 4},
         "mutation count")

    result = {
        "schema": "m1751_m1750_c1_energy_source_independent_hammer_r1_v1",
        "status": "PASS_M1751_M1750_C1_WHOLE_COMPONENT_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_CAMPAIGN",
        "python": __import__("sys").version.split()[0],
        "contract_and_author_double_seals": "PASS",
        "m1745_fail_exact_bound": True,
        "m1743_timing_exact_bound": True,
        "ledger_rows": rows,
        "ledger_histogram": histogram,
        "active_rows": active,
        "active_quantiles": quantiles,
        "public_port_tb_and_exact_dut_saif_gate": "PASS_STATIC_AND_MUTATION",
        "primary": "whole_mapped_c1_top_including_9macro_liberty",
        "primary_fields": ["Net Switching Power", "Cell Internal Power",
                           "Cell Leakage Power", "Total Power"],
        "component_total_conservation": "PASS_AND_MUTATION",
        "selected_macro_report": False,
        "top_minus_macro": False,
        "ptpx_plus_datasheet_sram_combined": False,
        "parent_sram_datasheet_role": "separate_alternative_sensitivity_only",
        "corner_classification": "mixed_corner_component_estimate",
        "standard_cells": "TT 0.9V 25C",
        "parent_sram_macro_liberty": "SSG 0.9V 125C",
        "single_corner_signoff": False,
        "authority_before_attempt_and_one_shot": "PASS_STATIC",
        "mutations_rejected": attacks,
        "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "eda_or_license_runs": 0,
    }
    output = HERE / ("cpython" + str(__import__("sys").version_info[0]) +
                     str(__import__("sys").version_info[1]) + "_hammer.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
