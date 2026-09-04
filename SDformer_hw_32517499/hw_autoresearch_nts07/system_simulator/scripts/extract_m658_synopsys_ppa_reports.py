#!/usr/bin/env python3
"""Strict extractor for M658 DC/PTPX/PrimeTime/memory-macro reports.

The extractor intentionally accepts only a small, machine-readable projection
of real report text with required Synopsys headers.  It does not run EDA.
"""

import argparse
import json
import math
import re
from pathlib import Path


class ExtractionError(ValueError):
    pass


NUMBER = r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)"


def _one(text, pattern, label):
    values = re.findall(pattern, text, flags=re.MULTILINE)
    if len(values) != 1:
        raise ExtractionError("%s must occur exactly once" % label)
    return values[0]


def _number(text, label):
    value = float(text)
    if not math.isfinite(value):
        raise ExtractionError(label + " is non-finite")
    return value


def _header(text, expected_report, expected_tool):
    if "****************************************" not in text:
        raise ExtractionError("missing Synopsys report delimiter")
    report = _one(text, r"^Report\s*:\s*(\S+)\s*$", "Report header")
    tool = _one(text, r"^Tool\s*:\s*(\S+)\s*$", "Tool header")
    version = _one(text, r"^Version\s*:\s*(\S+)\s*$", "Version header")
    library = _one(text, r"^Library\s*:\s*(\S+)\s*$", "Library header")
    corner = _one(text, r"^(?:Operating Conditions|Corner)\s*:\s*(\S+)\s*$", "corner header")
    if report != expected_report or tool != expected_tool:
        raise ExtractionError("unexpected report/tool identity")
    return {"tool": tool, "version": version, "library": library, "corner": corner}


def extract(dc_area_report, ptpx_power_report, pt_sta_report, sram_macro_report):
    dc_text = Path(dc_area_report).read_text(encoding="utf-8")
    power_text = Path(ptpx_power_report).read_text(encoding="utf-8")
    sta_text = Path(pt_sta_report).read_text(encoding="utf-8")
    macro_text = Path(sram_macro_report).read_text(encoding="utf-8")
    identities = {
        "dc_area": _header(dc_text, "area", "dc_shell"),
        "ptpx_power": _header(power_text, "power", "pt_shell"),
        "pt_sta": _header(sta_text, "timing", "pt_shell"),
        "sram_macro": _header(macro_text, "macro_characterization", "memory_compiler"),
    }
    values = {
        "logic_area_mm2": _number(_one(dc_text, r"^Total cell area \(um2\):\s*" + NUMBER + r"\s*$",
                                          "Total cell area"), "logic area") / 1e6,
        "logic_power_mw": _number(_one(power_text, r"^Logic dynamic power \(mW\):\s*" + NUMBER + r"\s*$",
                                           "logic power"), "logic power"),
        "sram_macro_power_mw": _number(_one(power_text, r"^Memory dynamic power \(mW\):\s*" + NUMBER + r"\s*$",
                                                "memory power"), "memory power"),
        "setup_wns_ns": _number(_one(sta_text, r"^Setup WNS \(ns\):\s*" + NUMBER + r"\s*$",
                                         "setup WNS"), "setup WNS"),
        "hold_wns_ns": _number(_one(sta_text, r"^Hold WNS \(ns\):\s*" + NUMBER + r"\s*$",
                                        "hold WNS"), "hold WNS"),
        "sram_macro_area_mm2": _number(_one(macro_text, r"^Macro area \(um2\):\s*" + NUMBER + r"\s*$",
                                                "macro area"), "macro area") / 1e6,
    }
    return {"identities": identities, "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dc-area-report", required=True, type=Path)
    parser.add_argument("--ptpx-power-report", required=True, type=Path)
    parser.add_argument("--pt-sta-report", required=True, type=Path)
    parser.add_argument("--sram-macro-report", required=True, type=Path)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        value = extract(args.dc_area_report, args.ptpx_power_report,
                        args.pt_sta_report, args.sram_macro_report)
    except (OSError, ExtractionError) as exc:
        print("M658_PPA_EXTRACT_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    else:
        print("M658_PPA_EXTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
