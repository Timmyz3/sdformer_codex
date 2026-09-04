#!/usr/bin/env python3
"""Independent adversarial review of the sealed M35-r6/M33 Synopsys run.

Primary DC/STA/Formality reports are reparsed.  The producer receipt is checked
for identity and boundary only; it is not used as a metric oracle.
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
    "m35_r6_m33_fair_exact_sha_synopsys_3p000ns_r6_20260823"
)
FAILED_RUN = pathlib.Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
    "dc_handoff/runs/"
    "m35_r6_m33_fair_exact_sha_synopsys_3p000ns_r5_20260823"
)
CONTRACT = (
    HW / "contracts/m35_r6_m33_fair_exact_sha_synopsys_contract_r1_20260823.json"
)
LAUNCH = HW / "contracts/m35_r6_m33_fair_exact_sha_launch_manifest_r1_20260823.json"
CANDIDATE = HW / "rtl_m35_r4/qfit_complement_csd8_canonical.sv"
M33_SOURCE = HW / "rtl_m33/qfit_threshold_late_scale_uq0p24_radix20x4.sv"
M33_POOL = HW / "rtl_m31/qfit_signed_int8_mul96_pool.sv"
BUILDER = HW / "dc_handoff/scripts/build_m35_r6_m33_fair_receipt.py"
REVIEW = (
    HW / "results/m35_r6_m33_fair_independent_hammer_review_20260823/"
    "m35_r6_m33_fair_independent_hammer_review.json"
)

EXPECTED = {
    "candidate": "84b1f3cb6344863ecfdbac2af8abcfdd15b1f16571979588badbc3e2e0dd1854",
    "m33_source": "2df1c28c0d22cd5a1c38a78a5838101b23bb13beec9e3e5e60ac8f84aba16c4c",
    "m33_pool": "7872d25c01c112f07a7d8e3cfe728029eef1f68e0f7bf87bdf2a50416776ea18",
    "contract": "261d980c9a5795fd25a6acb5f6fb6028fb523ecf57f357172845ccea69da5e74",
    "launch": "a468eb69ca5d620a15db79037415db81e403b9456ffe036984ba2abbc35183de",
    "receipt": "5cce986c4efe6e521029db413d5f973c3f788ee42182c6587cd3acab36f1d568",
    "fixed_builder": "4525c438cca1ca9a2f29ae08209a4a3fb790d3ac733f6e7264893f3197b4cca5",
    "contract_old_builder": "1e2ecc527d1255314cf05e4002a1e77c6870d2a9c29be107d2fccefed4d278d9",
    "launch_seal": "f1596cb2b3185fe5d73be4b75d1f1ca3839919393c448c114d974ce281a5b867",
    "snapshot_manifest": "14eec1b51242efccfe456479e7e496e41518d54daf89598ba3aa06104dd1be5b",
    "output_manifest": "8d8006aa0285716f984989c1df6b8d5bcfcbf557467751b7d674ad6f3275ec31",
    "completion_seal": "5a3c6063d29904cf72a4f633db8846a347fb222f50888200467fa78ff2ac0280",
    "run_complete": "af388cbd5e4b68e22a4ad60f8a4df959b9918b9c1c31219e493fbf60fb6b49a7",
    "failed_marker": "dbbcca2a18cb2ba398d1964750efa00e81f75b11b712f76a30eaa2d46fb0befe",
    "failed_partial": "1539f3638f028e2b20792c2a2bf47b3dfbaa4c22b7b9a92973ca91153f885bb7",
    "failed_parser_stderr": "7ff715b728100640aae34c4071e626fe4aa8b47e9d8e32f786d497ae93f2974a",
}


class ReviewFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ReviewFailure(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path, expected):
    require(path.is_file(), "missing file: {}".format(path))
    observed = sha256(path)
    require(observed == expected, "SHA mismatch {} expected={} observed={}".format(
        path, expected, observed))


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


def manifest_rows(path, base=None, verify_targets=True):
    rows = []
    seen = set()
    base = path.parent if base is None else pathlib.Path(base)
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        match = re.match(r"^([0-9a-f]{64}) [ *](.+)$", line)
        require(match is not None, "malformed manifest row {}:{}".format(
            path, line_number))
        digest, name = match.groups()
        require(name not in seen, "duplicate manifest path {}".format(name))
        seen.add(name)
        target = pathlib.Path(name)
        if not target.is_absolute():
            target = base / target
        if verify_targets:
            require(target.is_file(), "manifest target missing: {}".format(target))
            require(sha256(target) == digest, "manifest SHA mismatch: {}".format(target))
        rows.append((digest, name))
    return rows


def normalized_manifest_map(rows):
    return dict((name[2:] if name.startswith("./") else name, digest)
                for digest, name in rows)


def require_zero_rc(path):
    require(path.read_bytes() == b"0\n", "nonzero or malformed return code: {}".format(path))
    return 0


def number(pattern, text, cast=float):
    match = re.search(pattern, text, re.MULTILINE)
    require(match is not None, "missing report metric: {}".format(pattern))
    return cast(match.group(1))


def area_metrics(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "design": number(r"^Design\s*:\s*(\S+)", text, str),
        "ports": number(r"^Number of ports:\s+([0-9]+)", text, int),
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


def warning_ledger(raw_log):
    lines = []
    codes = collections.Counter()
    for line in raw_log.splitlines():
        if re.match(r"^\s*Warning:", line):
            lines.append(line)
            match = re.search(r"\(([A-Z][A-Z0-9_-]*-[0-9]+)\)\s*$", line)
            codes[match.group(1) if match else "UNCLASSIFIED"] += 1
    return lines, dict(sorted(codes.items()))


def unmatched_value(log, label):
    values = [int(value) for value in re.findall(
        r"^\s*([0-9]+)\([0-9]+\) Unmatched reference\(implementation\) "
        + label + r"\s*$", log, re.MULTILINE)]
    require(values, "missing unmatched summary: {}".format(label))
    return values[-1]


def require_empty_formality_report(text, phrase):
    require(text.count(phrase) == 1, "missing or duplicate empty report phrase: " + phrase)
    residue = text.replace(phrase, "")
    require(not re.search(
        r"(?i)(?<!no )\b(?:failing|aborted|unverified|unmatched) compare points?\b",
        residue), "contradictory nonempty Formality report content")


def formality_metrics(side):
    log = (RUN / side / "formality.raw.log").read_text(
        encoding="utf-8", errors="replace")
    succeeded = len(re.findall(r"^Verification SUCCEEDED$", log, re.MULTILINE))
    passing = [int(value) for value in re.findall(
        r"^\s*([0-9]+) Passing compare points\s*$", log, re.MULTILINE)]
    passing_rows = re.findall(
        r"^Passing \(equivalent\)\s+((?:[0-9]+\s+){7}[0-9]+)\s*$",
        log, re.MULTILINE)
    failing_rows = re.findall(
        r"^Failing \(not equivalent\)\s+((?:[0-9]+\s+){7}[0-9]+)\s*$",
        log, re.MULTILINE)
    require(succeeded == 1 and len(passing) == 1 and len(passing_rows) == 1
            and len(failing_rows) == 1, "ambiguous Formality terminal result: " + side)
    passing_columns = [int(value) for value in passing_rows[0].split()]
    failing_columns = [int(value) for value in failing_rows[0].split()]
    require(failing_columns == [0] * 8, "nonzero Formality failing points: " + side)
    reports = {
        "formality_failing.rpt": "No failing compare points.",
        "formality_aborted.rpt": "No aborted compare points.",
        "formality_unverified.rpt": "No unverified compare points.",
        "formality_unmatched.rpt": "No unmatched points.",
    }
    for name, phrase in reports.items():
        require_empty_formality_report((RUN / side / "reports" / name).read_text(
            encoding="utf-8", errors="replace"), phrase)
    require("FMR_ELAB-147" not in log, "FMR_ELAB-147 present: " + side)
    require(not re.search(
        r"^\s*(?:suppress_message|set_message_info|set_msg_config|filter_message)\b",
        log, re.MULTILINE | re.IGNORECASE), "message filter command present: " + side)
    return {
        "verification_succeeded_terminal_count": succeeded,
        "passing_compare_points": passing[0],
        "passing_port_points": passing_columns[4],
        "passing_dff_points": passing_columns[5],
        "failing_result_columns": failing_columns,
        "aborted_compare_points": 0,
        "unverified_compare_points": 0,
        "unmatched_compare_points": unmatched_value(log, r"compare points"),
        "unmatched_primary_or_blackbox_points": unmatched_value(
            log, r"primary inputs, black-box outputs"),
        "fmr_elab_147_count": 0,
        "message_filter_commands": 0,
    }


def physical_multiplier_hits(paths):
    pattern = re.compile(
        r"(?<![A-Za-z0-9_$])(?:DW[0-9_]*mult[A-Za-z0-9_$]*|"
        r"GTECH_MULT[A-Za-z0-9_$]*|[A-Za-z0-9_$]*MULT_OP[A-Za-z0-9_$]*|"
        r"mult_x_[A-Za-z0-9_$]+|mul_x_[A-Za-z0-9_$]+)(?![A-Za-z0-9_$])",
        re.IGNORECASE)
    hits = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(
                encoding="utf-8", errors="replace").splitlines(), 1):
            if pattern.search(line):
                hits.append("{}:{}:{}".format(path, line_number, line))
    return hits


def contract_builder_identity(contract, launch):
    entries = dict((entry["snapshot"], entry["sha256"])
                   for entry in launch["entries"])
    key = "hw_autoresearch_nts07/dc_handoff/scripts/build_m35_r6_m33_fair_receipt.py"
    return {
        "contract_declared_sha256": contract["common_synopsys_flow"][
            "receipt_builder_sha256"],
        "launch_manifest_sha256": entries[key],
        "snapshot_file_sha256": sha256(RUN / "snapshot/inputs" / key),
        "live_file_sha256": sha256(BUILDER),
    }


def require_identity_mismatch(identity):
    require(identity["contract_declared_sha256"]
            != identity["snapshot_file_sha256"],
            "contract builder identity contradiction was hidden or repaired")
    require(identity["launch_manifest_sha256"] == identity["snapshot_file_sha256"]
            == identity["live_file_sha256"],
            "canonical fixed-builder identities do not agree")


def audit():
    for label, path in (
        ("candidate", CANDIDATE), ("m33_source", M33_SOURCE),
        ("m33_pool", M33_POOL), ("contract", CONTRACT), ("launch", LAUNCH),
        ("fixed_builder", BUILDER),
    ):
        require_sha(path, EXPECTED[label])
    receipt_path = RUN / "m35_r6_m33_fair_receipt.json"
    require_sha(receipt_path, EXPECTED["receipt"])
    contract = strict_json(CONTRACT)
    launch = strict_json(LAUNCH)
    receipt = strict_json(receipt_path)
    require(receipt["status"] == "PASS_EXACT_SHA_FRESH_M35_AND_M33_DC_STA_FORMALITY",
            "producer receipt status drift")
    require(contract["status"]
            == "FROZEN_EXACT_SOURCE_FRESH_SEQUENTIAL_SAME_FLOW_3NS_DC_STA_FORMALITY",
            "contract status drift")

    require(RUN.is_dir() and RUN.resolve() == RUN, "canonical run missing or symlinked")
    file_count = 0
    directory_count = 0
    for path in [RUN] + list(RUN.rglob("*")):
        require(not path.is_symlink(), "symlink in canonical run: {}".format(path))
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            directory_count += 1
            require(mode == 0o555, "canonical directory mode drift: {}".format(path))
        elif path.is_file():
            file_count += 1
            require(mode == 0o444, "canonical file mode drift: {}".format(path))
    require((file_count, directory_count) == (321, 53), "canonical tree count drift")

    for key, rel in (
        ("launch_seal", "launch_manifest.sha256"),
        ("snapshot_manifest", "snapshot_input_sha256.txt"),
        ("output_manifest", "output_sha256.txt"),
        ("completion_seal", "completion_seal.sha256"),
        ("run_complete", "RUN_COMPLETE.txt"),
    ):
        require_sha(RUN / rel, EXPECTED[key])
    require((RUN / "launch_manifest.sha256").read_text().strip() == EXPECTED["launch"],
            "launch manifest content drift")
    require_sha(RUN / "snapshot/inputs/launch_manifest.json", EXPECTED["launch"])
    snapshot_rows = manifest_rows(
        RUN / "snapshot_input_sha256.txt", RUN / "snapshot/inputs")
    output_rows = manifest_rows(RUN / "output_sha256.txt", RUN)
    completion_rows = manifest_rows(RUN / "completion_seal.sha256", RUN)
    require((len(snapshot_rows), len(output_rows), len(completion_rows)) == (22, 109, 5),
            "manifest row count drift")
    snapshot_map = normalized_manifest_map(snapshot_rows)
    require(snapshot_map[str(CONTRACT.relative_to(REPO))] == EXPECTED["contract"],
            "snapshot contract identity drift")
    require(snapshot_map[str(CANDIDATE.relative_to(REPO))] == EXPECTED["candidate"],
            "snapshot candidate identity drift")
    require(snapshot_map[str(BUILDER.relative_to(REPO))] == EXPECTED["fixed_builder"],
            "snapshot builder identity drift")
    require(not (RUN / "run_local_seal.sha256").exists(),
            "unexpected standalone local seal changes run schema")

    return_codes = {}
    for side in ("m35", "m33"):
        for tool in ("dc", "sta", "formality"):
            return_codes[side + "_" + tool] = require_zero_rc(RUN / side / (tool + ".rc"))
    for name in ("m35/structural_audit.rc", "receipt_builder.rc", "validation.rc",
                 "output_manifest_check.rc"):
        return_codes[name.replace("/", "_")[:-3]] = require_zero_rc(RUN / name)
    require((RUN / "dc.version.rc").read_bytes() == b"1\n",
            "dc version probe exit behavior drift")
    require("dc_shell version    -  V-2023.12-SP3" in (RUN / "dc.version.raw.log").read_text(),
            "DC version evidence drift")
    for side in ("m35", "m33"):
        for tool in ("DC", "STA", "FORMALITY"):
            marker = (RUN / side / (tool + "_INTERNAL_COMPLETE.txt")).read_text()
            require(marker.count("M35_R6_M33_FAIR_{}_INTERNAL_COMPLETE=PASS".format(tool)) == 1,
                    "missing internal marker: {} {}".format(side, tool))

    expected_area = {
        "m35": {
            "design": "qfit_complement_csd8_canonical", "ports": 887,
            "cells": 15089, "combinational_cells": 13304,
            "sequential_cells": 1785, "macro_blackbox_cells": 0,
            "combinational_area_um2": 10102.553879,
            "noncombinational_area_um2": 3599.064058,
            "total_cell_area_um2": 13701.617937,
            "zero_net_area_text": True, "library_ssg0p9v125c": True,
        },
        "m33": {
            "design": "qfit_threshold_late_scale_uq0p24_radix20x4", "ports": 3657,
            "cells": 14324, "combinational_cells": 14045,
            "sequential_cells": 278, "macro_blackbox_cells": 0,
            "combinational_area_um2": 11680.199890,
            "noncombinational_area_um2": 528.317997,
            "total_cell_area_um2": 12208.517887,
            "zero_net_area_text": True, "library_ssg0p9v125c": True,
        },
    }
    timing_expected = {"m35": (0.0001, 0.0100), "m33": (0.0000, 0.0111)}
    warning_expected = {
        "m35": {
            "dc": (8, {"TIM-134": 4, "UISN-40": 4}),
            "sta": (1, {"TIM-134": 1}),
            "formality": (7, {"FMR_VLOG-057": 4, "FMR_VLOG-954": 1,
                               "UNCLASSIFIED": 2}),
        },
        "m33": {
            "dc": (11, {"UISN-40": 4, "VER-318": 7}),
            "sta": (0, {}),
            "formality": (12, {"FM-399": 1, "FMR_VLOG-057": 7,
                                "FMR_VLOG-954": 2, "UNCLASSIFIED": 2}),
        },
    }
    designs = {}
    for side in ("m35", "m33"):
        area = area_metrics(RUN / side / "reports/sta_area.rpt")
        require(area == expected_area[side], "independent area parse drift: " + side)
        setup, setup_rows = min_slack(RUN / side / "reports/sta_setup.rpt")
        hold, hold_rows = min_slack(RUN / side / "reports/sta_hold.rpt")
        require((setup, hold, setup_rows, hold_rows)
                == timing_expected[side] + (100, 100), "timing parse drift: " + side)
        clocks = (RUN / side / "reports/clocks.rpt").read_text(errors="replace")
        match = re.search(r"^core_clk\s+([0-9.]+)\s+\{[^}]+\}\s+(\S+)\s+\{clk_core\}\s*$",
                          clocks, re.MULTILINE)
        require(match and float(match.group(1)) == 3.0 and "p" not in match.group(2),
                "clock is not 3ns ideal/unpropagated: " + side)
        constraint = (RUN / side / "reports/constraint_contract_postcompile.rpt").read_text(
            errors="replace")
        require("physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO" in constraint,
                "physical boundary drift: " + side)
        require("Pin Input Delays:\n\n    None specified." in constraint
                and "Pin Output Delays:\n\n    None specified." in constraint,
                "unexpected IO delay model: " + side)
        raw_logs = {}
        ledgers = {}
        for tool in ("dc", "sta", "formality"):
            raw = (RUN / side / (tool + ".raw.log")).read_text(errors="replace")
            require(not re.search(r"^(?:Error|Fatal):", raw, re.MULTILINE),
                    "Error/Fatal in {} {}".format(side, tool))
            require("Thank you" in raw, "missing terminal shell text: {} {}".format(side, tool))
            raw_logs[tool] = raw
            lines, codes = warning_ledger(raw)
            require((len(lines), codes) == warning_expected[side][tool],
                    "warning ledger drift: {} {}".format(side, tool))
            ledgers[tool] = {"count": len(lines), "codes": codes}
        formality = formality_metrics(side)
        expected_fm = {"m35": (2334, 549, 1785), "m33": (655, 377, 278)}[side]
        require((formality["passing_compare_points"], formality["passing_port_points"],
                 formality["passing_dff_points"]) == expected_fm,
                "Formality passing-point drift: " + side)
        require(formality["unmatched_compare_points"] == 0
                and formality["unmatched_primary_or_blackbox_points"] == 0,
                "Formality unmatched point drift: " + side)
        designs[side] = {
            "dc_sta": dict(area, setup_wns_ns=setup, hold_wns_ns=hold,
                           setup_path_rows=setup_rows, hold_path_rows=hold_rows,
                           clock_period_ns=3.0, clock_is_ideal_unpropagated=True,
                           input_output_delays_specified=False),
            "formality": formality,
            "warnings": ledgers,
        }

    multiplier_paths = [RUN / "m35/reports" / name for name in (
        "resources_precompile.rpt", "resources_postcompile.rpt",
        "references_precompile.rpt", "references_postcompile.rpt")]
    multiplier_paths.append(RUN / "m35/netlist/qfit_complement_csd8_canonical_mapped.v")
    multiplier_hits = physical_multiplier_hits(multiplier_paths)
    require(not multiplier_hits, "M35 independent physical multiplier hits: {}".format(
        multiplier_hits[:3]))
    structure = (RUN / "m35/reports/m35_r6_zero_multiplier_audit.rpt").read_text()
    for phrase in (
        "status=PASS_STRICT_ZERO_PHYSICAL_MULTIPLIER_AND_LINK_AUDIT",
        "physical_multiplier_hit_total=0", "postcompile_blackbox_attribute_count=0",
        "unresolved_link_signature_count=0"):
        require(phrase in structure, "M35 structural audit drift: " + phrase)
    m33_refs = (RUN / "m33/reports/references_postcompile.rpt").read_text(errors="replace")
    require("qfit_signed_int8_mul96_pool_MULTIPLIERS96_IN_W8" in m33_refs,
            "M33 96-lane multiplier hierarchy missing")

    m35_area = designs["m35"]["dc_sta"]["total_cell_area_um2"]
    m33_area = designs["m33"]["dc_sta"]["total_cell_area_um2"]
    ratios = {
        "m35_results_per_cycle": 8,
        "m33_results_per_cycle": 4,
        "m35_over_m33_peak_result_rate": 2.0,
        "m35_over_m33_area": m35_area / m33_area,
        "m35_over_m33_result_rate_per_area": 2.0 * m33_area / m35_area,
        "m35_area_per_result_um2": m35_area / 8.0,
        "m33_area_per_result_um2": m33_area / 4.0,
        "m35_area_per_result_reduction_percent": (
            (m33_area / 4.0 - m35_area / 8.0) / (m33_area / 4.0) * 100.0),
        "sustained_full_valid_and_ready_required": True,
        "descriptor_reload_or_duty_cycle_included": False,
    }
    require(math.isclose(ratios["m35_over_m33_area"], 1.1222998617702726,
                         rel_tol=0.0, abs_tol=1e-15), "area ratio drift")
    require(math.isclose(ratios["m35_over_m33_result_rate_per_area"],
                         1.7820549285689806, rel_tol=0.0, abs_tol=1e-15),
            "density ratio drift")

    identity = contract_builder_identity(contract, launch)
    require(identity["contract_declared_sha256"] == EXPECTED["contract_old_builder"],
            "contract builder declaration unexpectedly changed")
    for key in ("launch_manifest_sha256", "snapshot_file_sha256", "live_file_sha256"):
        require(identity[key] == EXPECTED["fixed_builder"], "fixed builder identity drift")
    require_identity_mismatch(identity)

    require(FAILED_RUN.is_dir() and FAILED_RUN.resolve() == FAILED_RUN,
            "failed r5 attempt missing or symlinked")
    require_sha(FAILED_RUN / "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt",
                EXPECTED["failed_marker"])
    require_sha(FAILED_RUN / "PARTIAL_EVIDENCE.sha256", EXPECTED["failed_partial"])
    require_sha(FAILED_RUN / "receipt_builder.stderr.raw.log",
                EXPECTED["failed_parser_stderr"])
    failed_rows = manifest_rows(FAILED_RUN / "PARTIAL_EVIDENCE.sha256", FAILED_RUN)
    require(len(failed_rows) == 312, "failed r5 partial manifest row drift")
    require(not (FAILED_RUN / "RUN_COMPLETE.txt").exists()
            and not (FAILED_RUN / "completion_seal.sha256").exists(),
            "failed r5 unexpectedly has completion evidence")
    failed_stderr = (FAILED_RUN / "receipt_builder.stderr.raw.log").read_text(errors="replace")
    require("ValueError: unmatched row missing: compare points" in failed_stderr,
            "failed r5 parser signature drift")
    canonical_text = "\n".join(path.read_text(errors="replace") for path in RUN.rglob("*")
                               if path.is_file() and path.stat().st_size < 2 * 1024 * 1024)
    require(str(FAILED_RUN) not in canonical_text,
            "failed r5 path contaminates canonical run")

    return {
        "schema": "m35_r6_m33_fair_independent_hammer_review_v1",
        "status": "FAIL_INDEPENDENT_HAMMER_ONE_P1_EXACT_SHA_CONTRACT_IDENTITY_DRIFT",
        "date": "2026-08-23",
        "review_score_0_to_100": 80,
        "p0_count": 0,
        "p1_count": 1,
        "p2_count": 5,
        "review_verdict": "NO_GO_EXACT_SHA_RELEASE_REPAIR_CONTRACT_AND_FRESH_RERUN",
        "exact_identity": {
            "candidate_rtl_sha256": EXPECTED["candidate"],
            "m33_source_sha256": EXPECTED["m33_source"],
            "m33_multiplier_pool_sha256": EXPECTED["m33_pool"],
            "contract_sha256": EXPECTED["contract"],
            "launch_manifest_sha256": EXPECTED["launch"],
            "producer_receipt_sha256": EXPECTED["receipt"],
        },
        "independently_verified_seals": {
            "canonical_run_path": str(RUN),
            "canonical_directory_mode": "0555",
            "canonical_file_mode": "0444",
            "canonical_files_checked": file_count,
            "canonical_directories_checked_including_root": directory_count,
            "symlink_count": 0,
            "launch_snapshot_output_completion_manifest_rows": [1, len(snapshot_rows),
                                                                   len(output_rows),
                                                                   len(completion_rows)],
            "all_manifest_rows_sha256_pass": True,
            "standalone_run_local_seal_present": False,
            "local_integrity_supplied_by_snapshot_output_completion_chain": True,
            "tool_and_gate_return_codes": return_codes,
            "six_internal_completion_markers": True,
            "dc_version_probe_rc_is_one_not_a_tool_stage_gate": True,
        },
        "independently_recomputed_designs": designs,
        "independently_recomputed_physical_multiplier_evidence": {
            "m35_physical_multiplier_hit_count": len(multiplier_hits),
            "m35_macro_blackbox_count": designs["m35"]["dc_sta"][
                "macro_blackbox_cells"],
            "m35_unresolved_link_signature_count": 0,
            "m33_declared_signed_int8_multiplier_lanes": 96,
            "m33_architecturally_active_lanes_per_full_packet": 80,
            "m33_multiplier_hierarchy_present": True,
            "m33_hard_multiplier_macro_count": 0,
            "m33_multipliers_mapped_into_standard_cells": True,
        },
        "independently_recomputed_comparison": ratios,
        "contract_builder_identity_contradiction": dict(
            identity, mismatch_detected=True,
            producer_validator_checked_this_contract_field=False),
        "failed_r5_isolation": {
            "status": "FAIL_OR_INCOMPLETE_DO_NOT_CITE",
            "failed_marker_sha256": EXPECTED["failed_marker"],
            "partial_manifest_rows": len(failed_rows),
            "partial_manifest_sha256_pass": True,
            "parser_failure": "ValueError: unmatched row missing: compare points",
            "failed_builder_sha256": EXPECTED["contract_old_builder"],
            "canonical_fixed_builder_sha256": EXPECTED["fixed_builder"],
            "has_run_complete": False,
            "has_completion_seal": False,
            "canonical_failed_path_occurrences": 0,
            "canonical_byte_contamination_detected": False,
        },
        "findings": [
            {
                "id": "P1_CONTRACT_PINS_FAILED_R5_RECEIPT_BUILDER_NOT_CANONICAL_R6_BUILDER",
                "severity": "P1",
                "finding": (
                    "The exact-SHA contract declares receipt-builder SHA 1e2ecc..., the "
                    "failed r5 parser. The canonical launch manifest, immutable snapshot, "
                    "and live r6 builder are 4525c4.... The producer validator did not "
                    "cross-check this contract field, so the sealed exact-input contract "
                    "contradicts the bytes that produced the receipt."
                ),
                "repair_gate": (
                    "Correct the contract builder pin, regenerate the launch manifest, and "
                    "run a fresh non-overwriting DC/STA/Formality flow. Do not patch only the "
                    "receipt because changing the exact input contract requires a new run."
                ),
            },
            {
                "id": "P2_SETUP_CLOSURE_HAS_REPORT_PRECISION_ONLY_MARGIN",
                "severity": "P2",
                "finding": (
                    "At 3 ns, M33 setup WNS is 0.0000 ns and M35 is +0.0001 ns. Both pass "
                    "the literal >=0 gate but provide effectively no reported setup margin."
                ),
                "repair_gate": (
                    "Do not claim frequency headroom. Reclose with explicit positive margin "
                    "and later revalidate with propagated clocks, parasitics and variation."
                ),
            },
            {
                "id": "P2_SPECIALIZED_VS_GENERIC_INTERFACE_LIMITS_FAIRNESS",
                "severity": "P2",
                "finding": (
                    "M35 accepts only ten immutable H67-ep35 descriptor IDs, has a two-stage "
                    "pipeline and config/epoch protocol, and produces eight results/cycle. "
                    "M33 accepts any runtime UQ0.24 threshold, has one output-register stage, "
                    "and produces four results/cycle. This is a specialization comparison, "
                    "not an isofunctional or latency-normalized baseline."
                ),
                "repair_gate": (
                    "Label 1.782054929x only as peak checkpoint-specialization result-rate "
                    "density under sustained valid/ready traffic; report reload/duty-cycle and "
                    "a matched-function ablation before a broader hardware claim."
                ),
            },
            {
                "id": "P2_WARNINGS_EXPOSE_ZERO_WIRELOAD_AND_SIGNEDNESS_FRAGILITY",
                "severity": "P2",
                "finding": (
                    "M35 logs TIM-134 high-fanout nets using fanout 1000 for delay. M33 logs "
                    "seven VER-318 signedness warnings and Formality logs FM-399 for 347 "
                    "implementation undriven nets. Formal equivalence passes, but these "
                    "diagnostics are not clean physical or source-intent closure."
                ),
                "repair_gate": (
                    "Retain the warning ledger, make signed casts explicit, explain/resolve the "
                    "undriven implementation nets, and remeasure high fanout physically."
                ),
            },
            {
                "id": "P2_STANDALONE_ZERO_WIRELOAD_LOGIC_ONLY_NOT_PAPER_PPA",
                "severity": "P2",
                "finding": (
                    "Area is standard-cell total cell area with zero net area; timing uses an "
                    "ideal unpropagated clock, no IO delays, no SRAM macro and no layout."
                ),
                "repair_gate": (
                    "Keep all numbers inside the standalone logic-only boundary; require "
                    "macro-backed physical timing and power before a paper PPA headline."
                ),
            },
            {
                "id": "P2_READ_ONLY_AND_NO_DISTINCT_LOCAL_SEAL_ARE_NOT_AUTHENTICATION_ROOT",
                "severity": "P2",
                "finding": (
                    "The run is mode 0555/0444 and all nested SHA chains pass, but there is no "
                    "distinct run_local_seal.sha256 and an owner can chmod the tree."
                ),
                "repair_gate": (
                    "Pin this external review and the repaired run completion seal outside the "
                    "run; treat Unix modes only as accidental-write protection."
                ),
            },
        ],
        "claim_boundary": {
            "exact_sha_release_admitted": False,
            "diagnostic_only_underlying_result": (
                "The immutable r6 tool outputs independently support the listed standalone "
                "logic-only metrics, but they are not released as an exact-SHA milestone until "
                "the P1 contract identity defect is repaired by a fresh run."
            ),
            "forbidden": (
                "P&R/PTPX/power/energy, SRAM or interconnect PPA, integrated cycles, system "
                "speedup, external accelerator comparison, DATE headline or best-paper claim."
            ),
        },
        "rejected_attacks": [
            "snapshot_output_or_completion_manifest_byte_mutation",
            "duplicate_json_key_or_nonfinite_constant",
            "nonzero_or_malformed_tool_rc",
            "missing_or_duplicate_internal_completion_marker",
            "Formality_empty_report_phrase_with_contradictory_point_injection",
            "missing_or_duplicate_Formality_terminal_result",
            "mapped_M35_multiplier_or_unresolved_link_injection",
            "whitespace_indented_Formality_warning_omission",
            "failed_r5_completion_marker_injection",
            "failed_r5_path_in_canonical_run",
            "contract_builder_identity_mismatch_omission",
        ],
    }


def main():
    observed = audit()
    expected = strict_json(REVIEW)
    require(observed == expected, "canonical independent review JSON drift")
    print(json.dumps({
        "status": "PASS_VALIDATOR_REPRODUCED_STRICT_NO_GO_REVIEW",
        "review_sha256": sha256(REVIEW),
        "review_score_0_to_100": observed["review_score_0_to_100"],
        "review_verdict": observed["review_verdict"],
        "p0_count": observed["p0_count"],
        "p1_count": observed["p1_count"],
        "p2_count": observed["p2_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ReviewFailure as error:
        print("FAIL: {}".format(error), file=sys.stderr)
        sys.exit(1)
