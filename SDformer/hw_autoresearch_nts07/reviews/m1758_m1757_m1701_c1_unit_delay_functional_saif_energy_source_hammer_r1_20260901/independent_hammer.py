#!/usr/bin/env python3
"""Independent zero-EDA hammer for the M1757 C1 energy successor source."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1757_m1701_c1_unit_delay_functional_saif_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1757_m1701_c1_unit_delay_functional_saif_energy_source_author_receipt_r1_20260901"
FAILURE = HW / "reviews/m1757_m1750_c1_public_port_mapped_whole_component_energy_failure_receipt_r1_20260901"
M1745 = HW / "reviews/m1745_m1739_m1701_c1_public_port_mapped_production_energy_source_hammer_r1_20260901"
M1743 = HW / "contracts/m1743_m1742_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_release_r1_20260901.json"
TIMING = HW / "dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901"
LEDGER = HW / "results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901/ep34_c1_support16_rows.memh"
TB = HW / "dc_handoff/tb/tb_m1739_c1_m1701_public_port_mapped_production_energy.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1739_c1_m1701_public_port_mapped_production_energy.f"
UCLI = HW / "dc_handoff/scripts/m1739_c1_m1701_public_port_mapped_production_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1750_c1_m1701_public_port_mapped_whole_component_energy.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1757_m1701_c1_unit_delay_functional_saif_energy_one_shot.py"
CHECKER = HW / "system_simulator/scripts/check_m1757_c1_m1701_unit_delay_functional_saif_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1757_c1_m1701_unit_delay_functional_saif_energy_source.py"
M1759 = HW / "contracts/m1759_m1758_m1757_m1701_c1_unit_delay_functional_saif_energy_launch_release_r1_20260901.json"
ATTEMPT = HW / "results/.m1757_c1_unit_delay_functional_saif_energy_attempt_consumed"
RESULT = HW / "results/m1757_c1_unit_delay_functional_saif_energy_r1_20260901"
OLD_CONFLICT = HW / "reviews/m1756_m1750_c1_public_port_mapped_whole_component_energy_failure_receipt_r1_20260901"

SOURCE_SHAS = {
    "dc_handoff/tb/tb_m1739_c1_m1701_public_port_mapped_production_energy.sv": "efccfc7b8eca975958e4d13596a604ae469d711fab7b67284c9fb90982baaa9b",
    "dc_handoff/filelists/date_m1739_c1_m1701_public_port_mapped_production_energy.f": "016bbe13849909b260c2f3dad24164fa7176a1624e80508fc3d3ad8d56afbff6",
    "dc_handoff/scripts/m1739_c1_m1701_public_port_mapped_production_energy.ucli.tcl": "ec798508ed37410d2a13c40bb5c255de52583adcbc26b9acab967211b1d5f396",
    "dc_handoff/scripts/run_ptpx_m1750_c1_m1701_public_port_mapped_whole_component_energy.tcl": "1b9fdb335290e2e7dc14b3cdc1a0cbf3dbe63ed0ca691226762b037726a184c6",
    "dc_handoff/scripts/run_m1757_m1701_c1_unit_delay_functional_saif_energy_one_shot.py": "b7df92c54d20af892264044d9882bbdf43de1cfa79f21d57d11cbb0d613876ea",
    "system_simulator/scripts/check_m1757_c1_m1701_unit_delay_functional_saif_energy_source.py": "c1b26c42896822b9903061525636aa2f36ea7a6651c1cba0e14c594808861a7b",
    "system_simulator/tests/test_m1757_c1_m1701_unit_delay_functional_saif_energy_source.py": "79bd4dfdcfba09e4b6b88f70cdb26041e510504e437d9eab57563925d81d93e2",
}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
         and sha(outer) == outer_sha, "file seal identity " + str(path))
    need(sidecar.read_text() == payload_sha + "  " + path.name + "\n",
         "file sidecar content")
    need(outer.read_text() == sidecar_sha + "  " + sidecar.name + "\n",
         "file outer content")


def verify_dir_seal(root, manifest_sha, outer_sha):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "sealed root invalid")
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
        path = root / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == fields[0],
             "manifest member drift " + name)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "symlink in sealed tree")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
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
    spec = importlib.util.spec_from_file_location("m1757_target", str(CHECKER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def power_report(switching, internal, leakage, total=None):
    if total is None:
        total = switching + internal + leakage
    return ("Report : Averaged Power\nCommand : report_power -unit mW\n"
            "Net Switching Power = %.9f\nCell Internal Power = %.9f\n"
            "Cell Leakage Power = %.9f\nTotal Power = %.9f\n" %
            (switching, internal, leakage, total))


def must_fail(function):
    try:
        function()
    except Exception:
        return 1
    raise RuntimeError("negative mutation survived")


FORBIDDEN = ("+notimingcheck", "+no_notifier", "+nospecify", "+initreg",
             "+define+no_warning", "+define+NO_INPUT_FLOATING_CHECK")


def runner_policy(text):
    need(text.count('"+define+UNIT_DELAY"') == 1,
         "UNIT_DELAY compiler definition count")
    for token in FORBIDDEN:
        need(token not in text, "forbidden compile token " + token)
    need("m1750_c1_public_port_mapped_component_energy_r1_20260901.private_build" not in text,
         "old private build reuse")
    need("M1757_FAILURE" in text and "M1758" in text and "M1759" in text,
         "authority chain missing")
    need("fresh" in text.lower() and "old_m1750_private_build_reused\": False" in text,
         "fresh namespace/reuse disclosure missing")
    return True


def main():
    verify_file_seal(CONTRACT,
        "505e3f248fee60b757dfea62516d073e01442daf2ad00e3a3b0d350e7cc09a51",
        "249443c8828b2baa9a3fe11af8a6d00ed0f9516250305167324c40825700ee90",
        "8881ac1225cc05b186a18d175e3547fab43436847e8fa24cd9fb8b05b214f6bf")
    verify_dir_seal(AUTHOR,
        "f01196bbd36119118851012a66e359baccba4b61a04d4b1fd4c175b93ac4d6b1",
        "23107601acfb6faf8f560b89b0e13f2a0726743f7e036ac28658bdfcc1524a2c")
    verify_dir_seal(FAILURE,
        "66cebf72e6c4308433fe9cd58cd6ef01b3c42be4e92cf98a0a24b99472012105",
        "6fb9d51708c366c36208e29d0c3ab07fa677ace76d393fb476f3cb4f009d9a2e")
    verify_dir_seal(M1745,
        "c5b1f83b618ab8aadff16dc9e2a8f6498a852c66559d7a55171f93831bf3595a",
        "f81c8c0166da2d2e6ce7a99aa469bad9d800193edab865c33fe64ab6753c0404")
    verify_file_seal(M1743,
        "3c623618115c4ecf2e4bfec6efe167c90296825428ce87e16e6d52bd79216921",
        "7d481d605bffd1386b8926e709424a2c78b3f78eff340caf1727dbe7ec84cfe1",
        "7a52c2e7692b62857dfe1d2b1bd9e2825372a0fc839822abf086d4837bbcf112")
    verify_dir_seal(TIMING,
        "d3f2e14a6f6c0600abce2f5af2479402d41986736e3d9c32c6044e4225f64c75",
        "6f2b17f7016665cd663b9694a1ccbd29fa551ecf75ba29aa52c4bb56c5769b38")

    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "receipt.json")
    failed = strict_json(FAILURE / "receipt.json")
    forensic = strict_json(FAILURE / "forensic_counts.json")
    timing = strict_json(TIMING / "receipt.json")
    need(sha(AUTHOR / "receipt.json") ==
         "0de5d83ebcd26f94785e732a8ca7564fe6076053b59811b9b6dbb8f114cda8ab",
         "author receipt")
    need(sha(FAILURE / "receipt.json") ==
         "6a91e9e48958890bf81c7604eacbc9f63f5abebb3246befc161a07904c7617d2",
         "failure receipt")
    need(contract["status"] ==
         "SOURCE_ONLY__M1757_CONTAINED_M1750_FAILURE_BOUND__UNIT_DELAY_FUNCTIONAL_ONLY__M1743_TIMING_PINNED__M1758_REVIEW_AND_M1759_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(author["source_contract_sha256"] == sha(CONTRACT)
         and all(value == 0 or value is False for value in
                 author["author_execution"].values()),
         "author execution boundary")
    need(failed["status"] ==
         "SEALED_M1750_FAILURE__ATTEMPT_CONSUMED__OLD_BUILD_UNSEALED_DO_NOT_REUSE__AUTHORIZE_SOURCE_ONLY_SUCCESSOR_ANALYSIS",
         "failure disposition")
    need(failed["sealed_failure"]["counts"] ==
         {"vcs_compiles": 1, "simv_runs": 1, "saif_files": 0, "ptpx_runs": 0}
         and failed["namespace_state_at_receipt"]["canonical_result_absent"] is True,
         "failure counts/result boundary")
    disclosure = failed["post_failure_operator_disclosure"]
    need(disclosure["invocation"] ==
         "old private unsealed build simv -help, once, through an early-terminated pipe"
         and disclosure["ucli_or_m1739_plusarg"] is False
         and disclosure["saif_produced"] is False
         and disclosure["ptpx_produced"] is False
         and disclosure["canonical_result_produced"] is False
         and disclosure["complete_provenance_log"] is False,
         "post-failure diagnostic invocation disclosure")
    need(forensic["macro_timing_violation_lines"] == 35662
         and forensic["macro_ceb_unknown_or_high_z_warning_lines"] == 269046
         and forensic["watchdog_timeout_lines"] == 1
         and forensic["runtime_pass_lines"] == 0
         and forensic["saif_files"] == 0 and forensic["ptpx_runs"] == 0,
         "failure forensic counters")
    need(timing["formality"]["passing_compare_points"] == 16549
         and [timing["formality"][key] for key in
              ("failing", "aborted", "unverified", "unmatched")] == [0, 0, 0, 0]
         and timing["prime_time"]["setup_wns_ns"] == "0.027871"
         and timing["prime_time"]["hold_wns_ns"] == "0.001827"
         and timing["prime_time"]["macro_count"] == "9"
         and timing["claim_boundary"]["power"] is False
         and timing["claim_boundary"]["energy"] is False,
         "independent timing boundary")
    need(not os.path.lexists(str(M1759)) and not os.path.lexists(str(ATTEMPT))
         and not os.path.lexists(str(RESULT)), "premature launch namespace")

    inventory = dict((row["path"], row["sha256"])
                     for row in contract["source_files"])
    need(inventory == SOURCE_SHAS, "source inventory")
    for relative, digest in SOURCE_SHAS.items():
        need(sha(HW / relative) == digest, "source drift " + relative)
    need(all(value is False for value in contract["claim_boundary"].values()),
         "source claim promotion")
    need(contract["root_cause_and_delta"]["unit_delay_define_count"] == 1
         and contract["root_cause_and_delta"]["fresh_compile"] is True
         and contract["root_cause_and_delta"]["old_binary_or_csrc_reuse"] is False,
         "root cause/delta boundary")
    need(contract["gate_simulation"]["mode"] == "UNIT_DELAY_functional"
         and contract["gate_simulation"]["timing_simulation"] is False
         and contract["gate_simulation"]["timing_signoff"] is False
         and contract["gate_simulation"]["independent_pt_timing"] is True,
         "functional-vs-timing boundary")

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
    filelist = [row.split("#", 1)[0].strip() for row in FILELIST.read_text().splitlines()
                if row.split("#", 1)[0].strip()]
    need(len(filelist) == 4 and filelist[-2].endswith(
         "m935_m912_three_stage_exact_parent_match_product_capture_island_m1695_fastmin_hold_closed_mapped.v")
         and filelist[-1] == str(TB), "mapped filelist/top boundary")

    pt = PT_TCL.read_text()
    reports = [row for row in tcl_commands(pt) if row.startswith("report_power ")]
    need(len(reports) == 2, "unexpected report_power population")
    need(reports[0] ==
         'report_power -unit mW -nosplit -significant_digits 8 > "$output_dir/reports/ptpx_whole_mapped_c1_including_9macro_liberty.rpt"',
         "primary report is not whole mapped top")
    need(reports[1].startswith("report_power -hierarchy -area -unit mW ")
         and "$macro_cells" not in reports[1], "diagnostic report boundary")
    for token in ("expected_macro_count 9", "M1750_FAIL_EXACT_NET_ANNOTATION_GATE",
                  "M1750_FAIL_EXACT_LEAF_ANNOTATION_GATE",
                  "corner_classification=mixed_corner_component_estimate",
                  "standard_cell_power_library=TT_0p9V_25C",
                  "parent_sram_macro_liberty=SSG_0p9V_125C",
                  "not_single_corner_signoff=true", "top_minus_macro=false",
                  "ptpx_plus_datasheet_sram_combined=false"):
        need(token in pt, "PTPX boundary token " + token)
    need("report_power $macro_cells" not in pt and "ptpx_nine_parent_macros" not in pt,
         "macro subtraction survived")

    runner = RUNNER.read_text()
    runner_policy(runner)
    ordered = ("verify_authority()", "CHECK.validate_sources()",
               "namespaces_fresh()", "fcntl.flock(queue_handle.fileno()",
               "resource_gate()", "probe = subprocess.run", "ATTEMPT.mkdir()",
               'state["vcs_compiles"] += 1', '"+define+UNIT_DELAY"',
               'state["simv_runs"] += 1', 'state["ptpx_runs"] += 1',
               "CHECK.whole_component_power(", "seal_dir(STAGE)",
               "publish_no_replace(STAGE, RESULT)")
    cursor = 0
    for token in ordered:
        position = runner.find(token, cursor)
        need(position >= 0, "runner order " + token)
        cursor = position + len(token)
    for token in ("M1757_EXPECTED_M1758_MANIFEST_SHA256",
                  "M1757_EXPECTED_M1758_OUTER_FILE_SHA256",
                  "M1757_EXPECTED_M1758_REVIEW_SHA256",
                  "M1757_EXPECTED_M1759_RELEASE_SHA256"):
        need(token in runner, "future authority pin " + token)
    live_text = "\n".join((RUNNER.read_text(), CHECKER.read_text(), TEST.read_text(),
                            CONTRACT.read_text()))
    need("reviews/m1756_m1750" not in live_text, "conflicting M1756 referenced")
    checker = load_checker()
    macro_text = checker.MACRO_V.read_text()
    need("provides UNIT_DELAY mode for the fast function" in macro_text
         and "All timing values in the specification are not checked" in macro_text
         and "`ifdef UNIT_DELAY" in macro_text and "specify" in macro_text,
         "foundry UNIT_DELAY contract drift")

    source_attacks = 0
    for mutant in (runner.replace('"+define+UNIT_DELAY"', '"+define+BROKEN"'),
                   runner.replace('"+define+UNIT_DELAY"',
                                  '"+define+UNIT_DELAY", "+define+UNIT_DELAY"'),
                   runner.replace('"+define+UNIT_DELAY"',
                                  '"+define+UNIT_DELAY", "+notimingcheck"'),
                   runner.replace('"+define+UNIT_DELAY"',
                                  '"+define+UNIT_DELAY", "+no_notifier"'),
                   runner.replace('"+define+UNIT_DELAY"',
                                  '"+define+UNIT_DELAY", "+nospecify"'),
                   runner.replace('"+define+UNIT_DELAY"',
                                  '"+define+UNIT_DELAY", "+initreg"'),
                   runner + "\nm1750_c1_public_port_mapped_component_energy_r1_20260901.private_build\n"):
        source_attacks += must_fail(lambda value=mutant: runner_policy(value))

    source = checker.validate_sources()
    need(source["status"] == "PASS_M1757_UNIT_DELAY_FUNCTIONAL_SOURCE_ONLY_NO_EDA",
         "target self-check")
    attacks = {"runtime": 0, "saif": 0, "power": 0, "source": source_attacks}
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
        for changed in (good_log.replace("macro_reads=13", "macro_reads=12"),
                        good_log.replace("macro_writes=9", "macro_writes=8"),
                        good_log.replace("psum_commits=64", "psum_commits=63"),
                        good_log.replace("row_completions=64", "row_completions=63"),
                        good_log.replace("cycles=777", "cycles=0"),
                        good_log + good_log,
                        good_log + "$fatal\n"):
            log.write_text(changed)
            attacks["runtime"] += must_fail(lambda: checker.validate_runtime(log))
        saif = root / "x.saif"
        saif.write_text(good_saif)
        checker.validate_saif(saif, 100)
        for changed in (good_saif.replace("DURATION 300", "DURATION 297"),
                        good_saif.replace("TX 0", "TX 1"),
                        good_saif.replace("TC 9", "TC 0"),
                        good_saif.replace("INSTANCE dut", "INSTANCE bad"),
                        good_saif.replace("\n  (INSTANCE dut",
                                          "\n  (INSTANCE dut)\n  (INSTANCE dut", 1)):
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
             "datasheet sensitivity not separate")
        need(base["ptpx_plus_datasheet_sram_combined"] is False
             and base["parent_sram_datasheet_alternative_sensitivity"]
                 ["added_to_ptpx_whole_component"] is False,
             "combined accounting flag")
        for changed in (power_report(3.0, 6.0, 1.0, 9.0),
                        power_report(3.0, 6.0, -1.0, 8.0),
                        power_report(3.0, 6.0, 1.0) + "Total Power = 10.000000000\n",
                        power_report(3.0, 6.0, 1.0).replace("-unit mW", "-unit uW")):
            top.write_text(changed)
            attacks["power"] += must_fail(
                lambda: checker.whole_component_power(top, 100, 5, 3))
    need(attacks == {"runtime": 7, "saif": 5, "power": 4, "source": 7},
         "mutation count")

    result = {
        "schema": "m1758_m1757_c1_unit_delay_energy_source_independent_hammer_r1_v1",
        "status": "PASS_M1758_M1757_C1_UNIT_DELAY_FUNCTIONAL_SAIF_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_CAMPAIGN",
        "python": __import__("sys").version.split()[0],
        "contract_author_failure_double_seals": "PASS",
        "m1750_failure": {"attempt_consumed": True,
            "sealed_counts": failed["sealed_failure"]["counts"],
            "post_failure_simv_help_disclosed": True,
            "post_failure_saif_or_result": False,
            "old_build_reusable": False},
        "m1743_timing_exact_bound": True,
        "ledger_rows": rows, "ledger_histogram": histogram,
        "active_rows": active, "active_quantiles": quantiles,
        "gate_simulation_mode": "UNIT_DELAY_functional",
        "unit_delay_define_count": 1,
        "timing_simulation": False,
        "independent_pt_timing": True,
        "forbidden_switches_absent": list(FORBIDDEN),
        "fresh_compile_and_no_old_build_reuse": True,
        "public_port_tb_and_exact_dut_saif_gate": "PASS_STATIC_AND_MUTATION",
        "primary": "whole_mapped_c1_top_including_9macro_liberty",
        "primary_fields": ["Net Switching Power", "Cell Internal Power",
                           "Cell Leakage Power", "Total Power"],
        "component_total_conservation": "PASS_AND_MUTATION",
        "parent_sram_datasheet_role": "separate_alternative_sensitivity_only",
        "top_minus_macro": False,
        "ptpx_plus_datasheet_sram_combined": False,
        "corner_classification": "mixed_corner_component_estimate",
        "single_corner_signoff": False,
        "conflicting_m1756_referenced": False,
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
