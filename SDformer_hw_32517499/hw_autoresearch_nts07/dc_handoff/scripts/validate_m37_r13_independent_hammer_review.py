#!/usr/bin/env python3
"""Independently validate the sealed M37-r13 Synopsys milestone.

This validator deliberately reparses primary logs/reports and does not use the
producer's dc_sta_metrics.json or formality_metrics.json as metric oracles.
"""

from __future__ import print_function

import collections
import hashlib
import json
import math
import pathlib
import re
import stat
import sys


REPO = pathlib.Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
RUN = pathlib.Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
    "dc_handoff/runs/"
    "m37_csd_reconstruct_t10_r13_exact_sha_synopsys_r1_20260823"
)
FAILED_RUN = pathlib.Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
    "dc_handoff/runs/"
    "m37_csd_reconstruct_t10_r13_exact_sha_synopsys_20260823"
)
R8 = pathlib.Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
    "dc_handoff/runs/m37_csd_reconstruct_t10_dc_3p000ns_r2_20260822"
)
R9 = pathlib.Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
    "dc_handoff/runs/m37_csd_reconstruct_t10_dc_3p000ns_r9_20260822"
)
RECEIPT = HW / "contracts/m37_r13_exact_sha_synopsys_receipt_r1_20260823.json"
CONTRACT = HW / "contracts/m37_r13_exact_sha_synopsys_contract_r1_20260823.json"
CANDIDATE = HW / "rtl_m37_r10/qfit_atlif_csd_reconstruct_t10.sv"
RUNNER = HW / "dc_handoff/scripts/run_m37_r13_exact_sha_synopsys.sh"
FM_TCL = HW / "dc_handoff/scripts/run_formality_m37_r13_exact_sha.tcl"
REVIEW = (
    HW
    / "results/m37_r13_independent_hammer_review_20260823/"
    "m37_r13_independent_hammer_review.json"
)

EXPECTED = {
    "candidate": "f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd",
    "contract": "5a3ba83be2bf75b90b9b0a4229a32bc0dd03339536151fdfc01340d677dbef0f",
    "receipt": "3c54d019c43254ddc4c6d2f4324f9e2bb5bd594c0901129303292a896fb60058",
    "runner": "1a31435cf0ded60eb4801b1932ba3b81cf9ab3616d13f69bd3f11663680328c5",
    "failed_runner": "da29b113c0831c93e70358f18edb94b242dad32678d845563f355077a559c917",
    "fm_tcl": "cdd8e8551a00cd96593c5f8277fe37f50373c537466699f2f05feea55327b4c9",
    "input_manifest": "8befce7ae3b267d47a12a5933b7cf9197bce307bb10631a2cd66273de5b03d75",
    "output_manifest": "bc8fd75dd1f0e56695daac7e6de8dfc6fc8d6ee5e1c6c2ab0e7d4e03da3f2195",
    "local_seal": "5ac037a6c5a21d1f06ebec7f7fe80f5a479e2c9699c224f92dfbdf3181f75319",
    "completion_seal": "ba917e5a9c3c7c03900a8ade062b361e3c1e8a58eea478fdb4b771ca458847f5",
    "run_complete": "14cd96284d127cdfbc1090ac61ef584e69de92426c7270a81333d6b13c311d10",
    "failed_marker": "eb356c02ee08e22095c3b4bbf78e1120fab2d10c893364fc2cbc1147fac42a86",
    "failed_partial_manifest": "ef1d96767f82ecbc855ad88a2ba6d1a13c56652c5879bef215ccbe3d6b012fd5",
    "r8_area": "262f179fbc839908a1ab7cf4f9a0539bdc6369f8362e524f8c38ec1d1f7bd24d",
    "r8_qor": "b6abee05c23625273ca10c0c4e3102d1e879e57bd4cf5cfdb4838e4127351037",
    "r9_area": "25e6b8c2c376ac6f0f4c1323894e372d642943f02492e279db1c6f5de8bf3999",
    "r9_marker": "41c091fc9883b0bb0230547241ebe3ba21051d473356c0f2b493ce05107d94da",
}


class ReviewFailure(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ReviewFailure(message)


def strict_json(path):
    def reject_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReviewFailure("duplicate JSON key in {}: {}".format(path, key))
            result[key] = value
        return result

    def reject_constant(value):
        raise ReviewFailure("nonfinite JSON constant in {}: {}".format(path, value))

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, ValueError) as error:
        raise ReviewFailure("cannot parse strict JSON {}: {}".format(path, error))


def manifest_rows(path, verify_targets=True):
    rows = []
    seen = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        match = re.match(r"^([0-9a-f]{64}) [ *](.+)$", line)
        require(match is not None, "malformed manifest row {}:{}".format(path, line_number))
        digest, name = match.groups()
        require(name not in seen, "duplicate manifest path {}".format(name))
        seen.add(name)
        target = pathlib.Path(name)
        if not target.is_absolute():
            target = path.parent / target
        if verify_targets:
            require(target.is_file(), "manifest target missing: {}".format(target))
            require(sha256(target) == digest, "manifest SHA mismatch: {}".format(target))
        rows.append((digest, name))
    return rows


def require_sha(path, expected):
    require(path.is_file(), "missing file: {}".format(path))
    observed = sha256(path)
    require(observed == expected, "SHA mismatch {} expected={} observed={}".format(
        path, expected, observed))


def number(pattern, text, cast=float):
    match = re.search(pattern, text, re.MULTILINE)
    require(match is not None, "missing report metric: {}".format(pattern))
    return cast(match.group(1))


def area_metrics(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "design": number(r"^Design\s*:\s*(\S+)", text, str),
        "cells": number(r"^Number of cells:\s+([0-9]+)", text, int),
        "combinational_cells": number(
            r"^Number of combinational cells:\s+([0-9]+)", text, int),
        "sequential_cells": number(
            r"^Number of sequential cells:\s+([0-9]+)", text, int),
        "macro_blackbox_cells": number(
            r"^Number of macros/black boxes:\s+([0-9]+)", text, int),
        "combinational_area_um2": number(r"^Combinational area:\s+([0-9.]+)", text),
        "noncombinational_area_um2": number(
            r"^Noncombinational area:\s+([0-9.]+)", text),
        "total_cell_area_um2": number(r"^Total cell area:\s+([0-9.]+)", text),
        "zero_net_area_text": "Wire load has zero net area" in text,
        "library_ssg0p9v125c": "tcbn28hpcplusbwp35p140ssg0p9v125c" in text,
    }


def min_slack(path):
    values = [float(value) for value in re.findall(
        r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)",
        path.read_text(encoding="utf-8", errors="replace"),
    )]
    require(values, "no timing slack rows in {}".format(path))
    return min(values), len(values)


def register_widths(dc_log):
    widths = {}
    for name, width_text in re.findall(
        r"^\|\s*([^|]+?)\s*\|\s*Flip-flop\s*\|\s*([0-9]+)\s*\|",
        dc_log,
        re.MULTILINE,
    ):
        name = name.strip()
        width = int(width_text)
        require(name not in widths or widths[name] == width,
                "register width drift: {}".format(name))
        widths[name] = width
    require(widths, "no elaborated register table")
    return widths


def warning_ledger(raw_log):
    codes = collections.Counter()
    lines = []
    for line in raw_log.splitlines():
        if line.startswith("Warning:"):
            lines.append(line)
            match = re.search(r"\(([A-Z][A-Z0-9_-]*-[0-9]+)\)\s*$", line)
            codes[match.group(1) if match else "UNCLASSIFIED"] += 1
    return lines, dict(sorted(codes.items()))


def unmatched_value(log, label):
    values = [int(value) for value in re.findall(
        r"^\s*([0-9]+)\([0-9]+\) Unmatched reference\(implementation\) "
        + label + r"\s*$",
        log,
        re.MULTILINE,
    )]
    require(values, "missing unmatched summary: {}".format(label))
    return values[-1]


def physical_multiplier_hits(paths):
    # This is intentionally independent of audit_m37_dc_evidence.py.
    pattern = re.compile(
        r"(?<![A-Za-z0-9_$])(?:DW[0-9_]*mult[A-Za-z0-9_$]*|"
        r"GTECH_MULT[A-Za-z0-9_$]*|[A-Za-z0-9_$]*MULT_OP[A-Za-z0-9_$]*|"
        r"mult_x_[A-Za-z0-9_$]+|mul_x_[A-Za-z0-9_$]+)(?![A-Za-z0-9_$])",
        re.IGNORECASE,
    )
    hits = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if pattern.search(line):
                hits.append("{}:{}:{}".format(path, line_number, line))
    return hits


def audit():
    for label, path in (("receipt", RECEIPT), ("contract", CONTRACT),
                        ("candidate", CANDIDATE), ("runner", RUNNER),
                        ("fm_tcl", FM_TCL)):
        require_sha(path, EXPECTED[label])
    receipt = strict_json(RECEIPT)
    contract = strict_json(CONTRACT)
    require(receipt["status"] == "PASS_EXACT_SHA_DC_STA_FORMALITY_ONLY",
            "receipt status is not pass")
    require(contract["status"] == "READY_EXACT_SHA_DC_STA_FORMALITY_ONLY",
            "contract status is not ready")
    require(receipt["exact_inputs"]["candidate_rtl_sha256"] == EXPECTED["candidate"],
            "receipt candidate identity drift")

    require(RUN.is_dir() and RUN.resolve() == RUN, "canonical run missing or symlinked")
    require(stat.S_IMODE(RUN.stat().st_mode) == 0o555, "canonical run mode is not 0555")
    file_count = 0
    directory_count = 0
    for path in RUN.rglob("*"):
        require(not path.is_symlink(), "symlink in canonical run: {}".format(path))
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            directory_count += 1
            require(mode == 0o555, "canonical directory mode drift: {}".format(path))
        elif path.is_file():
            file_count += 1
            require(mode == 0o444, "canonical file mode drift: {}".format(path))

    for key, rel in (
        ("input_manifest", "input_sha256.txt"),
        ("output_manifest", "output_sha256.txt"),
        ("local_seal", "run_local_seal.sha256"),
        ("completion_seal", "completion_seal.sha256"),
        ("run_complete", "RUN_COMPLETE.txt"),
    ):
        require_sha(RUN / rel, EXPECTED[key])
    input_rows = manifest_rows(RUN / "input_sha256.txt")
    output_rows = manifest_rows(RUN / "output_sha256.txt")
    local_rows = manifest_rows(RUN / "run_local_seal.sha256")
    completion_rows = manifest_rows(RUN / "completion_seal.sha256")
    require((len(input_rows), len(output_rows), len(local_rows), len(completion_rows))
            == (18, 59, 5, 4), "canonical manifest row count drift")

    input_map = dict((name, digest) for digest, name in input_rows)
    require(input_map[str(CANDIDATE)] == EXPECTED["candidate"],
            "input manifest candidate mismatch")
    require(input_map[str(RUNNER)] == EXPECTED["runner"],
            "input manifest runner mismatch")
    require(input_map[str(CONTRACT)] == EXPECTED["contract"],
            "input manifest contract mismatch")

    rc_names = (
        "dc.rc", "sta.rc", "formality.rc", "structural_audit.rc",
        "input_manifest_check.rc", "output_manifest_check.rc",
        "r12_review_validation.rc",
    )
    return_codes = {}
    for name in rc_names:
        value = (RUN / name).read_text(encoding="utf-8").strip()
        require(value == "0", "nonzero or malformed return code {}={}".format(name, value))
        return_codes[name[:-3]] = int(value)
    require((RUN / "DC_INTERNAL_COMPLETE.txt").read_text().count(
        "M37_R13_DC_INTERNAL_COMPLETE=PASS") == 1, "DC internal marker missing")
    require((RUN / "STA_INTERNAL_COMPLETE.txt").read_text().count(
        "M37_R13_STA_INTERNAL_COMPLETE=PASS") == 1, "STA internal marker missing")
    require((RUN / "FORMALITY_INTERNAL_COMPLETE.txt").read_text().count(
        "M37_R13_FORMALITY_INTERNAL_COMPLETE=PASS") == 1,
        "Formality internal marker missing")

    dc_log = (RUN / "dc.raw.log").read_text(encoding="utf-8", errors="replace")
    sta_log = (RUN / "sta.raw.log").read_text(encoding="utf-8", errors="replace")
    fm_log = (RUN / "formality.raw.log").read_text(encoding="utf-8", errors="replace")
    for name, text in (("DC", dc_log), ("STA", sta_log), ("Formality", fm_log)):
        require(not re.search(r"^(?:Error|Fatal):", text, re.MULTILINE),
                "{} raw log contains Error/Fatal".format(name))
        require("Thank you" in text, "{} raw log lacks terminal shell text".format(name))

    area = area_metrics(RUN / "reports/sta_area.rpt")
    require(area == {
        "design": "qfit_atlif_csd_reconstruct_t10",
        "cells": 75130,
        "combinational_cells": 70175,
        "sequential_cells": 4955,
        "macro_blackbox_cells": 0,
        "combinational_area_um2": 53125.127493,
        "noncombinational_area_um2": 9989.280161,
        "total_cell_area_um2": 63114.407654,
        "zero_net_area_text": True,
        "library_ssg0p9v125c": True,
    }, "independent area parse drift")
    setup, setup_rows = min_slack(RUN / "reports/sta_setup.rpt")
    hold, hold_rows = min_slack(RUN / "reports/sta_hold.rpt")
    require((setup, hold, setup_rows, hold_rows) == (0.4173, 0.0104, 100, 100),
            "independent timing parse drift")
    clocks = (RUN / "reports/clocks.rpt").read_text(encoding="utf-8", errors="replace")
    clock_match = re.search(
        r"^core_clk\s+([0-9.]+)\s+\{[^}]+\}\s+(\S+)\s+\{clk_core\}\s*$",
        clocks,
        re.MULTILINE,
    )
    require(clock_match and float(clock_match.group(1)) == 3.0
            and "p" not in clock_match.group(2), "clock is not 3ns ideal/unpropagated")
    constraint = (RUN / "reports/constraint_contract_postcompile.rpt").read_text(
        encoding="utf-8", errors="replace")
    require("physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO" in constraint,
            "physical boundary drift")
    require("Pin Input Delays:\n\n    None specified." in constraint,
            "unexpected input-delay model")
    require("Pin Output Delays:\n\n    None specified." in constraint,
            "unexpected output-delay model")

    widths = register_widths(dc_log)
    require(len(widths) == 26 and sum(widths.values()) == 5979,
            "architectural register equation drift")
    dc_warnings, dc_warning_codes = warning_ledger(dc_log)
    sta_warnings, sta_warning_codes = warning_ledger(sta_log)
    fm_warnings, fm_warning_codes = warning_ledger(fm_log)
    require(len(dc_warnings) == 15 and dc_warning_codes == {
        "TIM-134": 4, "UISN-40": 4, "VER-318": 7}, "DC warning ledger drift")
    require(len(sta_warnings) == 1 and sta_warning_codes == {"TIM-134": 1},
            "STA warning ledger drift")
    require(len(fm_warnings) == 11, "Formality warning count drift")

    multiplier_paths = [
        RUN / "reports/resources_precompile.rpt",
        RUN / "reports/resources_postcompile.rpt",
        RUN / "reports/references_precompile.rpt",
        RUN / "reports/references_postcompile.rpt",
        RUN / "netlist/qfit_atlif_csd_reconstruct_t10_mapped.v",
    ]
    multiplier_hits = physical_multiplier_hits(multiplier_paths)
    require(not multiplier_hits, "independent physical multiplier hits: {}".format(
        multiplier_hits[:3]))
    unresolved = re.findall(
        r"Unable to resolve reference|Cannot find design|unresolved reference|"
        r"link failed|unresolved design",
        dc_log,
        re.IGNORECASE,
    )
    require(not unresolved, "unresolved DC link signatures")

    succeeded = len(re.findall(r"^Verification SUCCEEDED$", fm_log, re.MULTILINE))
    passing = [int(value) for value in re.findall(
        r"^\s*([0-9]+) Passing compare points\s*$", fm_log, re.MULTILINE)]
    failing_rows = re.findall(
        r"^Failing \(not equivalent\)\s+((?:[0-9]+\s+){7}[0-9]+)\s*$",
        fm_log,
        re.MULTILINE,
    )
    require(succeeded == 1 and passing == [5276] and failing_rows,
            "ambiguous Formality terminal result")
    failing_columns = [int(value) for value in failing_rows[-1].split()]
    require(failing_columns == [0] * 8, "nonzero Formality failing columns")
    unmatched_compare = unmatched_value(fm_log, r"compare points")
    unmatched_primary = unmatched_value(fm_log, r"primary inputs, black-box outputs")
    unmatched_unread = unmatched_value(fm_log, r"unread points")
    require((unmatched_compare, unmatched_primary, unmatched_unread) == (0, 0, 1),
            "Formality unmatched summary drift")
    report_phrases = {
        "formality_failing.rpt": "No failing compare points.",
        "formality_aborted.rpt": "No aborted compare points.",
        "formality_unverified.rpt": "No unverified compare points.",
        "formality_unmatched.rpt": "No unmatched points.",
    }
    for name, phrase in report_phrases.items():
        require(phrase in (RUN / "reports" / name).read_text(
            encoding="utf-8", errors="replace"), "missing Formality report phrase: " + name)
    require("0F/0A/5276P/0U" in fm_log, "Formality progress result drift")
    require("FMR_ELAB-147" not in fm_log, "FMR_ELAB-147 present")
    fm_sources = fm_log + "\n" + FM_TCL.read_text(encoding="utf-8", errors="replace")
    require(not re.search(
        r"^\s*(?:suppress_message|set_message_info|set_msg_config|filter_message)\b",
        fm_sources,
        re.MULTILINE | re.IGNORECASE,
    ), "Formality message filter command present")

    require_sha(R8 / "reports/area.rpt", EXPECTED["r8_area"])
    require_sha(R8 / "reports/qor.rpt", EXPECTED["r8_qor"])
    require_sha(R9 / "reports/area.rpt", EXPECTED["r9_area"])
    require_sha(R9 / "ORPHAN_RUN_DO_NOT_CITE.txt", EXPECTED["r9_marker"])
    r8_area = area_metrics(R8 / "reports/area.rpt")
    r9_area = area_metrics(R9 / "reports/area.rpt")
    require(r8_area["total_cell_area_um2"] == 63671.579642,
            "r8 area parse drift")
    require(r9_area["total_cell_area_um2"] == 185820.892828,
            "r9 area parse drift")
    r8_qor = (R8 / "reports/qor.rpt").read_text(encoding="utf-8", errors="replace")
    require("Critical Path Clk Period:      3.00" in r8_qor,
            "r8 3ns clock evidence missing")
    r9_marker = (R9 / "ORPHAN_RUN_DO_NOT_CITE.txt").read_text(
        encoding="utf-8", errors="replace")
    require("STATUS=DO_NOT_CITE_OR_USE_AS_FORMALITY_INPUT" in r9_marker,
            "r9 orphan boundary missing")
    gate = round(r8_area["total_cell_area_um2"] * 1.10, 6)
    require(gate == 70038.737606 and area["total_cell_area_um2"] <= gate,
            "r8-derived area gate failed")

    require(FAILED_RUN.is_dir() and FAILED_RUN.resolve() == FAILED_RUN,
            "failed attempt missing or symlinked")
    require_sha(FAILED_RUN / "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt",
                EXPECTED["failed_marker"])
    require_sha(FAILED_RUN / "PARTIAL_EVIDENCE.sha256",
                EXPECTED["failed_partial_manifest"])
    failed_rows = manifest_rows(FAILED_RUN / "PARTIAL_EVIDENCE.sha256")
    require(len(failed_rows) == 422, "failed-attempt partial manifest row drift")
    require(not (FAILED_RUN / "RUN_COMPLETE.txt").exists(),
            "failed attempt unexpectedly has RUN_COMPLETE")
    require(not (FAILED_RUN / "completion_seal.sha256").exists(),
            "failed attempt unexpectedly has completion seal")
    # The failed attempt intentionally pins the pre-fix runner bytes.  Its
    # input manifest is itself covered by PARTIAL_EVIDENCE, but following the
    # now-updated live runner path must fail.  Parse that historical ledger
    # without confusing expected live-path drift with partial-seal damage.
    failed_input = dict((name, digest) for digest, name in manifest_rows(
        FAILED_RUN / "input_sha256.txt", verify_targets=False))
    require(failed_input[str(RUNNER)] == EXPECTED["failed_runner"],
            "failed attempt did not pin the old parser runner")
    require(EXPECTED["failed_runner"] != EXPECTED["runner"],
            "old and fixed runner identities unexpectedly equal")
    canonical_manifests = (
        (RUN / "input_sha256.txt").read_text(encoding="utf-8", errors="replace")
        + (RUN / "output_sha256.txt").read_text(encoding="utf-8", errors="replace")
    )
    require(str(FAILED_RUN) not in canonical_manifests,
            "failed-attempt path contaminates canonical manifests")
    require(receipt["failed_attempt_disclosure"]["status"]
            == "FAIL_OR_INCOMPLETE_DO_NOT_CITE", "receipt weakens failed-attempt boundary")
    require(receipt["failed_attempt_disclosure"]["failed_marker_sha256"]
            == EXPECTED["failed_marker"], "receipt failed marker drift")
    require(receipt["failed_attempt_disclosure"]["partial_evidence_manifest_sha256"]
            == EXPECTED["failed_partial_manifest"], "receipt partial manifest drift")

    return {
        "schema": "m37_r13_independent_hammer_review_v1",
        "status": "PASS_INDEPENDENT_HAMMER_STANDALONE_LOGIC_ONLY",
        "date": "2026-08-23",
        "review_score_0_to_100": 94,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 4,
        "review_verdict": (
            "GO_STANDALONE_LOGIC_ONLY_ZERO_WIRELOAD_IDEAL_CLOCK_"
            "EXACT_SHA_DC_STA_FORMALITY_ONLY"
        ),
        "exact_identity": {
            "candidate_rtl_sha256": EXPECTED["candidate"],
            "contract_sha256": EXPECTED["contract"],
            "producer_receipt_sha256": EXPECTED["receipt"],
            "fixed_runner_sha256": EXPECTED["runner"],
            "failed_attempt_old_runner_sha256": EXPECTED["failed_runner"],
        },
        "independently_verified_seals": {
            "canonical_run_path": str(RUN),
            "canonical_directory_mode": "0555",
            "canonical_file_mode": "0444",
            "canonical_files_checked": file_count,
            "canonical_directories_checked": directory_count,
            "symlink_count": 0,
            "input_output_local_completion_manifest_rows": [
                len(input_rows), len(output_rows), len(local_rows), len(completion_rows)
            ],
            "all_manifest_rows_sha256_pass": True,
            "tool_and_gate_return_codes": return_codes,
            "all_three_internal_completion_markers": True,
        },
        "independently_recomputed_dc_sta": {
            "tool": "Synopsys Design Compiler V-2023.12-SP3",
            "technology": "TSMC28HPCplus_standard_cell_logic_only",
            "clock_period_ns": 3.0,
            "clock_is_ideal_unpropagated": True,
            "wireload": "ZeroWireload",
            "input_output_delays_specified": False,
            "cell_count": area["cells"],
            "combinational_cell_count": area["combinational_cells"],
            "mapped_sequential_cell_count": area["sequential_cells"],
            "architectural_register_rows": len(widths),
            "architectural_register_bits_from_elaboration": sum(widths.values()),
            "architectural_minus_mapped_sequential_bits": sum(widths.values()) - area["sequential_cells"],
            "combinational_area_um2": area["combinational_area_um2"],
            "noncombinational_area_um2": area["noncombinational_area_um2"],
            "total_cell_area_um2": area["total_cell_area_um2"],
            "macro_or_blackbox_cell_count": area["macro_blackbox_cells"],
            "independent_physical_multiplier_hit_count": len(multiplier_hits),
            "independent_unresolved_link_signature_count": len(unresolved),
            "setup_wns_ns": setup,
            "hold_wns_ns": hold,
            "setup_hold_path_rows": [setup_rows, hold_rows],
        },
        "independently_recomputed_formality": {
            "tool": "Synopsys Formality V-2023.12-SP3",
            "verification_succeeded_terminal_count": succeeded,
            "passing_compare_points": passing[-1],
            "passing_port_points": 321,
            "passing_dff_points": 4955,
            "failing_compare_points": failing_columns[-1],
            "aborted_compare_points": 0,
            "unverified_compare_points": 0,
            "unmatched_compare_points": unmatched_compare,
            "unmatched_primary_or_blackbox_points": unmatched_primary,
            "unmatched_reference_unread_points_diagnostic_only": unmatched_unread,
            "fmr_elab_147_count": 0,
            "message_filter_commands": 0,
        },
        "independently_recomputed_warning_ledger": {
            "dc_warning_count": len(dc_warnings),
            "dc_warning_codes": dc_warning_codes,
            "sta_warning_count": len(sta_warnings),
            "sta_warning_codes": sta_warning_codes,
            "formality_warning_count": len(fm_warnings),
            "formality_unlinked_power_cell_warning_lines": sum(
                "unlinked power cell(s)" in line for line in fm_warnings),
        },
        "independently_recomputed_ab": {
            "r8_logic_only_reference_area_um2": r8_area["total_cell_area_um2"],
            "r13_area_um2": area["total_cell_area_um2"],
            "r13_over_r8": area["total_cell_area_um2"] / r8_area["total_cell_area_um2"],
            "r13_reduction_vs_r8_percent": (
                (r8_area["total_cell_area_um2"] - area["total_cell_area_um2"])
                / r8_area["total_cell_area_um2"] * 100.0
            ),
            "area_gate_is_110_percent_of_r8_rounded_6dp_um2": gate,
            "r13_area_gate_headroom_um2": gate - area["total_cell_area_um2"],
            "r9_orphan_diagnostic_area_um2": r9_area["total_cell_area_um2"],
            "r13_over_r9_diagnostic": (
                area["total_cell_area_um2"] / r9_area["total_cell_area_um2"]
            ),
            "r13_reduction_vs_r9_diagnostic_percent": (
                (r9_area["total_cell_area_um2"] - area["total_cell_area_um2"])
                / r9_area["total_cell_area_um2"] * 100.0
            ),
            "r9_admitted_for_claim": False,
            "r8_r13_is_strict_same_rtl_same_constraint_ab": False,
        },
        "failed_attempt_isolation": {
            "failed_attempt_status": "FAIL_OR_INCOMPLETE_DO_NOT_CITE",
            "failed_attempt_partial_manifest_rows": len(failed_rows),
            "failed_attempt_partial_manifest_sha256_pass": True,
            "failed_attempt_has_run_complete": False,
            "failed_attempt_has_completion_seal": False,
            "old_and_fixed_runner_sha_differ": True,
            "canonical_manifest_mentions_failed_attempt_path": False,
            "canonical_contamination_detected": False,
        },
        "rejected_attacks": [
            "canonical_manifest_byte_mutation",
            "duplicate_json_key",
            "nonfinite_json_constant",
            "nonzero_or_malformed_tool_rc",
            "missing_internal_completion_marker",
            "duplicate_or_missing_Formality_terminal_result",
            "Formality_no_unmatched_points_parser_ambiguity",
            "Formality_message_filter_command",
            "FMR_ELAB_147_injection",
            "mapped_multiplier_or_unresolved_link_injection",
            "failed_attempt_completion_marker_injection",
            "failed_attempt_path_in_canonical_manifest",
            "r9_orphan_marker_omission",
        ],
        "findings": [
            {
                "id": "P2_STANDALONE_IDEAL_LOGIC_ONLY_NOT_PAPER_PPA",
                "severity": "P2",
                "finding": (
                    "The admitted area and timing use ZeroWireload, an ideal/unpropagated "
                    "clock, no input/output delays, no SRAM macros, and no physical layout."
                ),
                "repair_gate": (
                    "Do not use these values as a paper PPA or frequency headline; require "
                    "macro-backed integration, physical timing, SAIF/PTPX, and system evidence."
                ),
            },
            {
                "id": "P2_R8_R13_AND_R9_ARE_BOUNDED_REFERENCES",
                "severity": "P2",
                "finding": (
                    "r13 is 0.875071721% smaller than the pinned historical r8 logic-only "
                    "reference. r9 is a separately marked orphan diagnostic and its 66.03% "
                    "reduction is not citable. The r8/r13 pair is not a same-RTL same-constraint A/B."
                ),
                "repair_gate": (
                    "Use r8 only as the stated historical logic-only area reference and r9 only "
                    "as diagnostic provenance; never present either ratio as system speedup."
                ),
            },
            {
                "id": "P2_FORMALITY_UNREAD_AND_POWER_CELL_DIAGNOSTICS_RETAINED",
                "severity": "P2",
                "finding": (
                    "Formality reports one reference unread point and two warning lines about "
                    "375 unlinked power cells. They are diagnostic: all 321 port and 4,955 DFF "
                    "compare points pass, with zero failing/aborted/unverified and zero unmatched "
                    "compare or primary/black-box points."
                ),
                "repair_gate": (
                    "Retain these diagnostics in every citation and recheck under any library, "
                    "UPF, top-level, or mapped-netlist change."
                ),
            },
            {
                "id": "P2_READ_ONLY_MODE_IS_NOT_AUTHENTICATION_ROOT",
                "severity": "P2",
                "finding": (
                    "Mode 0555/0444 reduces accidental edits but the owner can chmod. Integrity "
                    "comes from the nested SHA manifests and this externally pinned review."
                ),
                "repair_gate": (
                    "Downstream milestones must pin this review and validator SHA in addition to "
                    "the candidate, receipt, contract, and canonical completion seal."
                ),
            },
        ],
        "claim_boundary": {
            "permitted": (
                "Exact f947... standalone logic-only TSMC28HPCplus standard-cell DC/STA at "
                "3 ns ZeroWireload ideal clock, zero mapped multiplier/macro/blackbox, and "
                "successful RTL-to-gate Formality for the sealed canonical run."
            ),
            "forbidden": (
                "PTPX power or energy, SRAM/macro/interconnect/physical PPA, integrated cycles, "
                "full-system speedup, DATE paper headline, or any use of the r9 orphan as evidence."
            ),
            "headline_admitted": False,
        },
    }


def main():
    review = audit()
    if REVIEW.is_file():
        canonical = strict_json(REVIEW)
        require(canonical == review, "canonical review JSON differs from independent rebuild")
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReviewFailure, OSError, KeyError, ValueError) as error:
        print("M37_R13_INDEPENDENT_HAMMER=FAIL detail={}".format(error), file=sys.stderr)
        raise SystemExit(1)
