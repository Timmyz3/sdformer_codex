#!/usr/bin/env python3
"""Independent zero-EDA M1739 review and mutation hammer."""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1739_m1701_c1_public_port_mapped_production_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1739_m1701_c1_public_port_mapped_production_energy_source_author_receipt_r1_20260901"
M1743 = HW / "contracts/m1743_m1742_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_release_r1_20260901.json"
TIMING = HW / "dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901"
LEDGER = HW / "results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901/ep34_c1_support16_rows.memh"
TB = HW / "dc_handoff/tb/tb_m1739_c1_m1701_public_port_mapped_production_energy.sv"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1739_c1_m1701_public_port_mapped_production_energy.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1739_m1701_c1_public_port_mapped_production_energy_one_shot.py"
CHECKER = HW / "system_simulator/scripts/check_m1739_c1_m1701_public_port_mapped_production_energy_source.py"
MAN = Path("/opt/synopsys/prime/W-2024.09-SP3/doc/pt/man/cat2/report_power.2")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON " + token)))


def verify_seal(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal")
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest")
        need(sha(root / rel) == digest, "member drift " + name)
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "sealed population")


def verify_file_seal(path):
    sum_path = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(sum_path.read_text().split() == [sha(path), path.name], "file sidecar")
    need(outer.read_text().split() == [sha(sum_path), sum_path.name], "file outer")


def load_checker():
    spec = importlib.util.spec_from_file_location("m1739_target", str(CHECKER))
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


def main():
    expected_source = {
        "dc_handoff/tb/tb_m1739_c1_m1701_public_port_mapped_production_energy.sv": "efccfc7b8eca975958e4d13596a604ae469d711fab7b67284c9fb90982baaa9b",
        "dc_handoff/filelists/date_m1739_c1_m1701_public_port_mapped_production_energy.f": "016bbe13849909b260c2f3dad24164fa7176a1624e80508fc3d3ad8d56afbff6",
        "dc_handoff/scripts/m1739_c1_m1701_public_port_mapped_production_energy.ucli.tcl": "ec798508ed37410d2a13c40bb5c255de52583adcbc26b9acab967211b1d5f396",
        "dc_handoff/scripts/run_ptpx_m1739_c1_m1701_public_port_mapped_production_energy.tcl": "459b80d74a22318bd361b3394c93b2f24f07775b7be04dc1ee1a12ad21baca2b",
        "dc_handoff/scripts/run_m1739_m1701_c1_public_port_mapped_production_energy_one_shot.py": "172ff74be5db1bfae3667c00e5b03153e02a3699c6b2de2fbdda880f39364d18",
        "system_simulator/scripts/check_m1739_c1_m1701_public_port_mapped_production_energy_source.py": "8a6feb660e2120d74856706ae3bbedc387108115debf0554acd7d51b1b7e2403",
        "system_simulator/tests/test_m1739_c1_m1701_public_port_mapped_production_energy_source.py": "47eb3ce9a0de3a184b629b3e0f8bc7731fbebcbcddd7e2bcb139ddf968d57887",
    }
    verify_file_seal(CONTRACT)
    verify_seal(AUTHOR)
    verify_file_seal(M1743)
    verify_seal(TIMING)
    need(sha(CONTRACT) == "f4056267e134ceb433d722d2c5cc4e7d6fe90191f2ed50103decb65d7a2a5803", "contract")
    need(sha(AUTHOR / "receipt.json") == "d711e346be38cc5eea1be827b0b90ec5a719b15e4e9729ed889d3f1e5098529a", "author receipt")
    need(sha(M1743) == "3c623618115c4ecf2e4bfec6efe167c90296825428ce87e16e6d52bd79216921", "M1743")
    need(sha(TIMING / "receipt.json") == "0b3ee22f9369a38eb83f674a4f1eb73fac39757ee85a3e1aeebe032bd0c76a1e", "timing receipt")
    contract = strict_json(CONTRACT)
    receipt = strict_json(AUTHOR / "receipt.json")
    timing = strict_json(TIMING / "receipt.json")
    need(receipt["source_contract"]["sha256"] == sha(CONTRACT), "author contract pin")
    need(dict((row["path"], row["sha256"]) for row in contract["source_files"]) == expected_source,
         "source inventory")
    for relative, digest in expected_source.items():
        need(sha(HW / relative) == digest, "source SHA " + relative)
    need(timing["status"] == "PASS_CANONICAL_C1_FORMALITY_AND_INDEPENDENT_PT_PRELAYOUT", "timing status")
    need(timing["prime_time"]["clock_period_ns"] == "3.000"
         and timing["prime_time"]["setup_wns_ns"] == "0.027871"
         and timing["prime_time"]["hold_wns_ns"] == "0.001827"
         and timing["prime_time"]["macro_count"] == "9", "PT values")
    formal = timing["formality"]
    need(formal["passing_compare_points"] == 16549 and
         [formal[key] for key in ("failing", "aborted", "unverified", "unmatched")] == [0, 0, 0, 0], "Formality values")
    need(timing["claim_boundary"]["formality"] is True
         and timing["claim_boundary"]["independent_pt"] is True
         and timing["claim_boundary"]["power"] is False
         and timing["claim_boundary"]["energy"] is False, "timing boundary")

    # Full 51.84M-row independent support histogram.
    histogram = [0] * 17
    rows = 0
    with LEDGER.open("rb", buffering=1 << 20) as stream:
        for raw in stream:
            need(len(raw) == 9 and raw[8:] == b"\n" and raw[:4] == b"0000",
                 "ledger row format")
            value = int(raw[4:8], 16)
            histogram[bin(value).count("1")] += 1
            rows += 1
    need(rows == 51840000 and histogram == [26535787, 7880233, 5335070,
         3774342, 2614180, 1861862, 1383722, 907501, 608784, 448874,
         213441, 124172, 72126, 41560, 22171, 10962, 5213], "histogram")
    active = rows - histogram[0]
    cumulative = 0
    quantiles = {}
    for support in range(1, 17):
        cumulative += histogram[support]
        for label, numerator in (("p25", 1), ("p50", 2), ("p75", 3)):
            rank = (active * numerator + 3) // 4
            if label not in quantiles and cumulative >= rank:
                quantiles[label] = support
    need(quantiles == {"p25": 1, "p50": 2, "p75": 4}, "quantiles")

    active_tb = re.sub(r"/\*.*?\*/|//[^\n]*", "", TB.read_text(), flags=re.S).lower()
    need("force " not in active_tb and "release " not in active_tb and "dut." not in active_tb,
         "non-public TB action")
    for token in ("case (row % 3)", "0: support = 1", "1: support = 2",
                  "default: support = 4", "count_macro_reads", "count_macro_writes",
                  "psum_write_data", "issue_request_source_valid"):
        need(token in TB.read_text(), "TB token " + token)

    runner = RUNNER.read_text()
    ordered = ("verify_authority()", "CHECK.validate_sources()", "namespaces_fresh()",
               "fcntl.flock(queue_handle.fileno()", "resource_gate()",
               "probe = subprocess.run", "ATTEMPT.mkdir()", "state[\"vcs_compiles\"] += 1",
               "state[\"simv_runs\"] += 1", "state[\"ptpx_runs\"] += 1",
               "seal_dir(STAGE)", "publish_no_replace(STAGE, RESULT)")
    cursor = 0
    for token in ordered:
        position = runner.find(token, cursor)
        need(position >= 0, "runner order " + token)
        cursor = position + len(token)
    clean = runner[runner.index("def clean_env"):runner.index("def run(")]
    need('"HOME": os.environ["HOME"]' in clean and not any(
         token in clean for token in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")),
         "environment boundary")
    need('COUNTS = {"vcs_compiles": 1, "simv_runs": 1, "saif_files": 1,' in runner
         and '"ptpx_runs": 1}' in runner
         and runner.count('"automatic_retry": False') >= 3,
         "one-shot boundary")

    # Official tool documentation proves the selected-macro command is not a
    # selected-cell summary: without -cell_power/-net_power/-hierarchy it is a
    # current-design summary. This is the decisive fail-closed finding.
    need(sha(MAN) == "507ec6ebb7b65c851b8f9a7570b35c1b5bfdffa439be1e8897aeae623dc4ec40", "PrimeTime man page")
    man = MAN.read_text(errors="replace")
    need("When  none" in man and "summary power" in man.lower(), "report_power manual semantics")
    pt = PT_TCL.read_text()
    macro_line = next(row.strip() for row in pt.splitlines()
                      if row.strip().startswith("report_power $macro_cells"))
    need("-cell_power" not in macro_line and "-net_power" not in macro_line
         and "-hierarchy" not in macro_line, "P0 unexpectedly repaired")

    checker = load_checker()
    runtime_attacks = 0
    saif_attacks = 0
    power_attacks = 0
    good_log = ("M1739_PUBLIC_COUNTERS cycles=777 issue_accepts=145 parent_edges=20 "
                "macro_reads=13 macro_writes=9 forwards=7 dead_write_elisions=55 "
                "psum_commits=64 row_completions=64\n"
                "PASS_M1739_C1_M1701_PUBLIC_PORT_MAPPED_DIRECTED_COMPONENT_ACTIVITY\n")
    saif = ("(SAIFILE (DURATION 300) (INSTANCE " + checker.TOP +
            " (INSTANCE dut (NET (x (T0 100) (T1 200) (TX 0) (TC 9)))))))\n")
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        log = root / "sim.log"; log.write_text(good_log)
        checker.validate_runtime(log)
        for old, new in (("macro_reads=13", "macro_reads=12"),
                         ("macro_writes=9", "macro_writes=8"),
                         ("psum_commits=64", "psum_commits=63"),
                         ("row_completions=64", "row_completions=63"),
                         ("cycles=777", "cycles=0"),
                         ("parent_edges=20", "parent_edges=0")):
            log.write_text(good_log.replace(old, new))
            runtime_attacks += must_fail(lambda: checker.validate_runtime(log))
        path = root / "x.saif"; path.write_text(saif)
        checker.validate_saif(path, 100)
        for changed in (saif.replace("DURATION 300", "DURATION 297"),
                        saif.replace("TX 0", "TX 1"),
                        saif.replace("TC 9", "TC 0"),
                        saif.replace("INSTANCE dut", "INSTANCE bad"),
                        saif + saif):
            path.write_text(changed)
            saif_attacks += must_fail(lambda: checker.validate_saif(path, 100))
        top = root / "top.rpt"; macro = root / "macro.rpt"
        top.write_text(power_report(3.0, 6.0, 1.0))
        macro.write_text(power_report(0.5, 1.3, 0.2))
        good_power = checker.combine_power(top, macro, 100, 5, 3)
        for values in ((0.5, 6.1, 0.2), (3.1, 1.3, 0.2), (0.5, 1.3, 1.1)):
            macro.write_text(power_report(*values))
            power_attacks += must_fail(lambda: checker.combine_power(top, macro, 100, 5, 3))
        # Decisive checker survivor: identical top and alleged macro summaries
        # pass, return zero logic, and are not tied to nine selected instances.
        macro.write_text(top.read_text())
        survivor = checker.combine_power(top, macro, 100, 5, 3)
        need(survivor["logic_only_total_power_mw_top_minus_macro_liberty"] == 0.0,
             "top==macro survivor absent")
        # Independent consistency survivor: Total need not equal components.
        top.write_text(power_report(3.0, 6.0, 1.0, total=11.0))
        macro.write_text(power_report(0.5, 1.3, 0.2, total=2.0))
        inconsistent = checker.combine_power(top, macro, 100, 5, 3)
        need(inconsistent["logic_only_total_power_mw_top_minus_macro_liberty"] == 9.0,
             "inconsistent total survivor absent")

    result = {
        "schema": "m1745_m1739_c1_energy_source_independent_hammer_r1_v1",
        "status": "FAIL_P0_DO_NOT_AUTHORIZE_M1746",
        "python": __import__("sys").version.split()[0],
        "identity_and_authority": "PASS",
        "histogram": histogram,
        "rows": rows, "active_rows": active, "active_quantiles": quantiles,
        "mapped_tb_public_port_static": "PASS_NO_FORCE_RELEASE_OR_DUT_HIERARCHICAL_READ",
        "runner_one_shot_queue_proxy_home": "PASS_STATIC",
        "runtime_mutations_rejected": runtime_attacks,
        "saif_mutations_rejected": saif_attacks,
        "power_underflow_mutations_rejected": power_attacks,
        "power_top_equals_macro_false_accept": True,
        "power_inconsistent_total_false_accept": True,
        "official_report_power_manual_sha256": sha(MAN),
        "p0": "report_power macro object_list lacks -cell_power/-net_power/-hierarchy; manual defines this as current-design summary, not selected-cell aggregate",
        "p1": "cell-based switching includes nets driven by selected cells; subtracting it without adding interface-net energy undercounts the split",
        "eda_or_license_runs": 0,
    }
    output = HERE / ("cpython" + str(__import__("sys").version_info[0]) +
                     str(__import__("sys").version_info[1]) + "_hammer.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("FAIL_M1745_P0_DO_NOT_AUTHORIZE_M1746")


if __name__ == "__main__":
    main()
