#!/usr/bin/env python3
"""Direct native-report extractor for the M663 registry-r7 methodology.

Accepted inputs are native Synopsys ``report_area``, PrimeTime
``report_timing`` (setup and hold), PrimeTime PX averaged-power output, and a
TSMC memory-compiler ``.ds`` report.  No author-authored numeric wrapper is an
accepted input grammar.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


class ExtractionError(ValueError):
    pass


NUMBER = r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)"
REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_FIELDS = {"dc_area", "ptpx_power", "pt_setup", "pt_hold", "sram_macro"}
MANIFEST_FIELDS = {
    "schema", "status", "row_id", "configuration_manifest_sha256",
    "m527_configuration_id", "operator_scope_sha256", "design_name",
    "macro_name", "run_id", "raw_reports", "tools", "libraries", "corners",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ExtractionError(label + " fields differ")


def _one(text, pattern, label, flags=re.MULTILINE):
    values = re.findall(pattern, text, flags=flags)
    if len(values) != 1:
        raise ExtractionError(label + " must occur exactly once")
    return values[0]


def _number(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ExtractionError(label + " is non-finite")
    return result


def _native_header(text, expected_report):
    if text.count("****************************************") < 2:
        raise ExtractionError("missing native Synopsys report delimiters")
    report = _one(text, r"^Report\s*:\s*(.+?)\s*$", "Report header").strip()
    design = _one(text, r"^Design\s*:\s*(\S+)\s*$", "Design header")
    version = _one(text, r"^Version\s*:\s*(\S+)\s*$", "Version header")
    _one(text, r"^Date\s*:\s*(.+?)\s*$", "Date header")
    if report.lower() != expected_report.lower():
        raise ExtractionError("unexpected native report identity: " + report)
    return {"design": design, "version": version}


def parse_dc_area(path):
    text = Path(path).read_text(encoding="utf-8")
    identity = _native_header(text, "area")
    library = _one(text, r"^\s{2,}(\S+)\s+\(File:\s*[^\n]+\)\s*$",
                   "native DC library")
    area_um2 = _number(_one(text, r"^Total cell area:\s*" + NUMBER + r"\s*$",
                            "native Total cell area"), "logic area")
    identity.update({"tool": "dc_shell", "library": library})
    return identity, area_um2 / 1e6


def parse_pt_timing(path, expected_delay):
    text = Path(path).read_text(encoding="utf-8")
    identity = _native_header(text, "timing")
    delays = re.findall(r"^\s*-(?:delay_type|delay)\s+(max|min)\s*$", text,
                        flags=re.MULTILINE)
    path_types = re.findall(r"^\s*Path Type:\s*(max|min)\s*$", text,
                            flags=re.MULTILINE)
    observed = set(delays + path_types)
    if observed != {expected_delay}:
        raise ExtractionError("native timing delay type mismatch")
    slacks = re.findall(r"^\s*slack\s+\((?:MET|VIOLATED)\)\s+" + NUMBER + r"\s*$",
                        text, flags=re.MULTILINE)
    if not slacks:
        raise ExtractionError("native timing report has no measured slack path")
    identity.update({"tool": "pt_shell", "delay_type": expected_delay})
    return identity, min(_number(value, "timing slack") for value in slacks)


def parse_ptpx_power(path):
    text = Path(path).read_text(encoding="utf-8")
    identity = _native_header(text, "Averaged Power")
    unit = _one(text, r"^\s*-unit\s+(\S+)\s*$", "PTPX power unit")
    if unit != "mW":
        raise ExtractionError("PTPX power unit must be mW")
    memory = _one(
        text,
        r"^memory\s+" + NUMBER + r"\s+" + NUMBER + r"\s+" + NUMBER +
        r"\s+" + NUMBER + r"\s+\([^\n]+\)\s*$",
        "PTPX memory power-group row",
    )
    memory_values = [_number(value, "PTPX memory power") for value in memory]
    total_internal = _number(_one(text, r"^\s*Cell Internal Power\s*=\s*" + NUMBER +
                                  r"\s+\([^\n]+\)\s*$", "PTPX internal total"),
                             "PTPX internal total")
    total_switching = _number(_one(text, r"^\s*Net Switching Power\s*=\s*" + NUMBER +
                                   r"\s+\([^\n]+\)\s*$", "PTPX switching total"),
                              "PTPX switching total")
    total_leakage = _number(_one(text, r"^\s*Cell Leakage Power\s*=\s*" + NUMBER +
                                 r"\s+\([^\n]+\)\s*$", "PTPX leakage total"),
                            "PTPX leakage total")
    total_power = _number(_one(text, r"^Total Power\s*=\s*" + NUMBER +
                               r"\s+\([^\n]+\)\s*$", "PTPX total power"),
                          "PTPX total power")
    total_components = total_internal + total_switching + total_leakage
    if not math.isclose(total_power, total_components, rel_tol=2e-6, abs_tol=1e-9):
        raise ExtractionError("PTPX total power does not equal internal+switching+leakage")
    if not math.isclose(memory_values[3], sum(memory_values[:3]), rel_tol=2e-6,
                        abs_tol=1e-9):
        raise ExtractionError("PTPX memory total does not equal its components")
    for index, value in enumerate(memory_values[:3]):
        if value > (total_internal, total_switching, total_leakage)[index] + 1e-12:
            raise ExtractionError("PTPX memory component exceeds chip total")
    identity.update({"tool": "pt_shell", "unit": unit})
    return identity, {
        "total_internal_power_mw": total_internal,
        "total_switching_power_mw": total_switching,
        "total_leakage_power_mw": total_leakage,
        "total_power_mw": total_power,
        "sram_internal_power_mw": memory_values[0],
        "sram_switching_power_mw": memory_values[1],
        "sram_leakage_power_mw": memory_values[2],
        "sram_total_power_mw": memory_values[3],
    }


def parse_sram_macro(path):
    text = Path(path).read_text(encoding="utf-8")
    software = _one(text, r"^#### Software\s*:\s*TSMC MEMORY COMPILER\s+(\S+)\s*\*/\s*$",
                    "memory compiler software")
    library_row = _one(
        text,
        r"^#### Library Name\s*:\s*(\S+)\s+\(user specify\s*:\s*(\S+)\)\s*\*/\s*$",
        "memory macro identity",
    )
    generated = _one(text, r"^#### Generated Time\s*:\s*(.+?)\s*\*/\s*$",
                     "memory generated time")
    process, voltage, temperature = _one(
        text,
        r"^\s*2\.2 SRAM timing:\((Slow|Fast|Typical),\s*" + NUMBER +
        r",\s*" + NUMBER + r"\s*deg\.\)\s*$",
        "memory PVT identity",
    )
    area_um2 = _number(_one(
        text,
        r"^\s*\|\s*" + NUMBER + r"\s*\|\s*" + NUMBER + r"\s*\|\s*" +
        NUMBER + r"\s*\|\s*$", "memory macro dimensions")[2], "macro area")
    leakage_ua = _number(_one(text, r"^\s*Leakage Current\s+" + NUMBER +
                              r"\s*\(uA\).*$", "memory leakage current"),
                         "macro leakage current")
    read_ua_per_mhz = _number(_one(text, r"^\s*Read\s+" + NUMBER +
                                   r"\s*\(uA/MHz\)\s*$", "memory read current"),
                              "macro read current")
    write_ua_per_mhz = _number(_one(text, r"^\s*Write\s+" + NUMBER +
                                    r"\s*\(uA/MHz\)\s*$", "memory write current"),
                               "macro write current")
    return {
        "tool": "memory_compiler", "version": software,
        "library": library_row[0], "macro": library_row[1],
        "generated_time": generated.strip(), "process": process,
        "voltage_v": _number(voltage, "macro voltage"),
        "temperature_c": _number(temperature, "macro temperature"),
    }, {
        "sram_macro_area_mm2": area_um2 / 1e6,
        "sram_leakage_current_ua": leakage_ua,
        "sram_read_current_ua_per_mhz": read_ua_per_mhz,
        "sram_write_current_ua_per_mhz": write_ua_per_mhz,
    }


def _load_manifest(path):
    manifest_path = Path(path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"),
                              parse_constant=lambda token: (_ for _ in ()).throw(
                                  ExtractionError("non-finite JSON token: " + token)))
    except (OSError, ValueError) as exc:
        raise ExtractionError("cannot load native run manifest: %s" % exc)
    _exact(manifest, MANIFEST_FIELDS, "native run manifest")
    if (manifest["schema"] != "m663.h67.native_synopsys_run_manifest.r1" or
            manifest["status"] != "FROZEN_NATIVE_REPORTS"):
        raise ExtractionError("native run manifest schema/status mismatch")
    if not isinstance(manifest["raw_reports"], dict) or set(manifest["raw_reports"]) != REPORT_FIELDS:
        raise ExtractionError("native run manifest report set mismatch")
    report_paths = {}
    for name, spec in manifest["raw_reports"].items():
        _exact(spec, {"path", "sha256", "media_type"}, "native report spec")
        if spec["media_type"] != "text/plain":
            raise ExtractionError("native report media type mismatch")
        candidate = (REPO_ROOT / spec["path"]).resolve()
        if not candidate.is_file() or candidate.is_symlink() or _sha256(candidate) != spec["sha256"]:
            raise ExtractionError("native report path/SHA mismatch: " + name)
        report_paths[name] = candidate
    return manifest_path, manifest, report_paths


def extract_from_manifest(path):
    manifest_path, manifest, reports = _load_manifest(path)
    dc_identity, logic_area = parse_dc_area(reports["dc_area"])
    power_identity, power = parse_ptpx_power(reports["ptpx_power"])
    setup_identity, setup_wns = parse_pt_timing(reports["pt_setup"], "max")
    hold_identity, hold_wns = parse_pt_timing(reports["pt_hold"], "min")
    macro_identity, macro = parse_sram_macro(reports["sram_macro"])
    logic_designs = {dc_identity["design"], power_identity["design"],
                     setup_identity["design"], hold_identity["design"]}
    if logic_designs != {manifest["design_name"]}:
        raise ExtractionError("native logic reports do not match the manifest design")
    if macro_identity["macro"] != manifest["macro_name"]:
        raise ExtractionError("native macro report does not match the manifest macro")
    if dc_identity["library"] != manifest["libraries"]["dc_area"]:
        raise ExtractionError("native DC library does not match run manifest")
    if macro_identity["library"] != manifest["libraries"]["sram_macro"]:
        raise ExtractionError("native SRAM library does not match run manifest")
    identities = {
        "dc_area": dc_identity, "ptpx_power": power_identity,
        "pt_setup": setup_identity, "pt_hold": hold_identity,
        "sram_macro": macro_identity,
    }
    expected_tools = {name: {"tool": identities[name]["tool"],
                             "version": identities[name]["version"]}
                      for name in sorted(identities)}
    if manifest["tools"] != expected_tools:
        raise ExtractionError("native run tool/version identity mismatch")
    logic_internal = power["total_internal_power_mw"] - power["sram_internal_power_mw"]
    logic_switching = power["total_switching_power_mw"] - power["sram_switching_power_mw"]
    logic_leakage = power["total_leakage_power_mw"] - power["sram_leakage_power_mw"]
    values = {
        "logic_area_mm2": logic_area,
        "sram_macro_area_mm2": macro["sram_macro_area_mm2"],
        "logic_internal_power_mw": logic_internal,
        "logic_switching_power_mw": logic_switching,
        "logic_dynamic_power_mw": logic_internal + logic_switching,
        "logic_leakage_power_mw": logic_leakage,
        "logic_total_power_mw": logic_internal + logic_switching + logic_leakage,
        "sram_internal_power_mw": power["sram_internal_power_mw"],
        "sram_switching_power_mw": power["sram_switching_power_mw"],
        "sram_dynamic_power_mw": (power["sram_internal_power_mw"] +
                                   power["sram_switching_power_mw"]),
        "sram_leakage_power_mw": power["sram_leakage_power_mw"],
        "sram_total_power_mw": power["sram_total_power_mw"],
        "total_internal_power_mw": power["total_internal_power_mw"],
        "total_switching_power_mw": power["total_switching_power_mw"],
        "total_dynamic_power_mw": (power["total_internal_power_mw"] +
                                    power["total_switching_power_mw"]),
        "total_leakage_power_mw": power["total_leakage_power_mw"],
        "total_power_mw": power["total_power_mw"],
        "setup_wns_ns": setup_wns,
        "hold_wns_ns": hold_wns,
    }
    return {
        "manifest_path": str(manifest_path), "run_identity": {
            "row_id": manifest["row_id"],
            "configuration_manifest_sha256": manifest["configuration_manifest_sha256"],
            "m527_configuration_id": manifest["m527_configuration_id"],
            "operator_scope_sha256": manifest["operator_scope_sha256"],
            "design_name": manifest["design_name"], "macro_name": manifest["macro_name"],
            "run_id": manifest["run_id"],
        },
        "identities": identities, "libraries": manifest["libraries"],
        "corners": manifest["corners"], "values": values,
        "macro_diagnostics": macro,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = extract_from_manifest(args.run_manifest)
    except (OSError, ExtractionError) as exc:
        print("M663_NATIVE_PPA_EXTRACT_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    else:
        print("M663_NATIVE_PPA_EXTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
