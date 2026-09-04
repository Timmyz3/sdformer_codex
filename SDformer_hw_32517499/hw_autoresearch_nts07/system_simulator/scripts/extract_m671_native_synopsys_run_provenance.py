#!/usr/bin/env python3
"""Fail-closed native Synopsys/memory-compiler extractor for M671 r8.

The extractor accepts only repository-relative, non-symlink inputs.  A report
set is inseparable from its configuration, typed library/corner inventory,
SRAM organization and rooted tool-run provenance.  Synthetic fixtures may
exercise the grammar, but no fixture is production authority.
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
REPORT_FIELDS = {
    "dc_area", "dc_environment", "pt_setup", "pt_setup_environment",
    "pt_hold", "pt_hold_environment", "ptpx_power", "ptpx_environment",
    "vcs_simulation", "formality_verification",
    "weight_sram_macro", "state_sram_macro", "parent_scratch_macro",
}
MANIFEST_FIELDS = {
    "schema", "status", "row_id", "configuration_manifest",
    "configuration_manifest_sha256", "m527_configuration_id",
    "operator_scope_sha256", "design_name", "run_id", "raw_reports",
    "target_corners", "library_dbs", "memory_inventory", "tool_run_receipt",
}
TOOL_RUN_FIELDS = {
    "schema", "status", "row_id", "configuration_manifest_sha256",
    "design_name", "run_id", "generation_argv", "command_scripts",
    "rtl_sources", "netlist", "sdc", "library_dbs", "activity", "tool_logs",
    "tool_executables", "tool_versions", "exit_status", "output_reports",
    "memory_inventory_sha256", "component_root_sha256",
}
STEPS = ("vcs", "dc", "formality", "pt_setup", "pt_hold", "ptpx",
         "memory_compiler")
TOOL_NAMES = ("vcs", "dc_shell", "fm_shell", "pt_shell", "memory_compiler")
RTL_SOURCE_ROLES = ("design_rtl", "testbench", "assertions")
LIBRARY_ROLES = (
    "logic_setup", "logic_hold", "logic_power",
    "sram_setup", "sram_hold", "sram_power",
)
TARGET_CORNERS = {
    "dc_area": {"operating_condition": "ssg0p9v125c", "process": "Slow",
                "voltage_v": 0.9, "temperature_c": 125.0},
    "pt_setup": {"operating_condition": "ssg0p9v125c", "process": "Slow",
                 "voltage_v": 0.9, "temperature_c": 125.0},
    "pt_hold": {"operating_condition": "ffg1p05vm40c", "process": "Fast",
                "voltage_v": 1.05, "temperature_c": -40.0},
    "ptpx_power": {"operating_condition": "tt0p9v25c", "process": "Typical",
                   "voltage_v": 0.9, "temperature_c": 25.0},
    "sram_macro": {"operating_condition": "ssg0p9v125c", "process": "Slow",
                   "voltage_v": 0.9, "temperature_c": 125.0},
}
EXPECTED_LIBRARY_NAMES = {
    "logic_setup": "tcbn28hpcplusbwp35p140ssg0p9v125c",
    "logic_hold": "tcbn28hpcplusbwp35p140ffg1p05vm40c",
    "logic_power": "tcbn28hpcplusbwp30p140tt0p9v25c",
    "sram_setup": "h67_sram_ssg0p9v125c",
    "sram_hold": "h67_sram_ffg1p05vm40c",
    "sram_power": "h67_sram_tt0p9v25c",
}
EXPECTED_MEMORY_MACROS = (
    {"role": "weight_sram", "report_id": "weight_sram_macro",
     "macro_name": "TS2N28HPCPHVTB1024X128M4D", "depth_words": 1024,
     "width_bits": 128, "port_type": "1R1W", "port_count": 2,
     "bank_count": 8, "instance_count": 8},
    {"role": "state_sram", "report_id": "state_sram_macro",
     "macro_name": "TS2N28HPCPHVTB512X192M4D", "depth_words": 512,
     "width_bits": 192, "port_type": "1R1W", "port_count": 2,
     "bank_count": 8, "instance_count": 8},
    {"role": "parent_scratch", "report_id": "parent_scratch_macro",
     "macro_name": "TS2N28HPCPHVTB1024X128M4D", "depth_words": 1024,
     "width_bits": 128, "port_type": "1R1W", "port_count": 2,
     "bank_count": 1, "instance_count": 1},
)
EXPECTED_TOOL_VERSIONS = {
    "vcs": "V-2023.12-SP1_Full64",
    "dc_shell": "V-2023.12-SP3",
    "fm_shell": "V-2023.12-SP3",
    "pt_shell": "W-2024.09-SP3",
    "memory_compiler": "tsn28hpcpd127spsram_2012.02.00.d.180a",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _map_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _nonnegative(value, label):
    result = _number(value, label)
    if result < 0.0:
        raise ExtractionError(label + " is negative")
    return result


def _positive(value, label):
    result = _number(value, label)
    if result <= 0.0:
        raise ExtractionError(label + " is not positive")
    return result


def _load_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              ExtractionError("non-finite JSON token: " + token)))
    except (OSError, ValueError) as exc:
        raise ExtractionError("cannot load %s: %s" % (label, exc))


def _secure_repo_file(raw_path, label, prefix=None):
    if not isinstance(raw_path, str) or not raw_path:
        raise ExtractionError(label + " path is missing")
    if (raw_path.startswith("/") or "\\" in raw_path or "//" in raw_path or
            "\x00" in raw_path):
        raise ExtractionError(label + " path is not canonical repo-relative")
    parts = raw_path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ExtractionError(label + " path contains forbidden component")
    if prefix is not None and not raw_path.startswith(prefix):
        raise ExtractionError(label + " path is outside required prefix")
    candidate = REPO_ROOT
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ExtractionError(label + " path traverses a symlink")
    try:
        resolved = candidate.resolve()
        resolved.relative_to(REPO_ROOT)
    except (OSError, ValueError):
        raise ExtractionError(label + " path escapes repository")
    if not candidate.is_file():
        raise ExtractionError(label + " is not a regular file")
    return candidate


def _file_spec(spec, label, media_types=None, prefix=None):
    _exact(spec, {"path", "sha256", "media_type"}, label)
    if media_types is not None and spec["media_type"] not in media_types:
        raise ExtractionError(label + " media type mismatch")
    path = _secure_repo_file(spec["path"], label, prefix)
    if not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"] or ""):
        raise ExtractionError(label + " SHA is malformed")
    if _sha256(path) != spec["sha256"]:
        raise ExtractionError(label + " SHA mismatch")
    return path


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
    area_um2 = _positive(_one(text, r"^Total cell area:\s*" + NUMBER + r"\s*$",
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
    if set(delays + path_types) != {expected_delay}:
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
    memory = _one(text, r"^memory\s+" + NUMBER + r"\s+" + NUMBER + r"\s+" +
                  NUMBER + r"\s+" + NUMBER + r"\s+\([^\n]+\)\s*$",
                  "PTPX memory power-group row")
    memory_values = [_nonnegative(value, "PTPX memory power") for value in memory]
    total_internal = _nonnegative(_one(text, r"^\s*Cell Internal Power\s*=\s*" + NUMBER +
                                  r"\s+\([^\n]+\)\s*$", "PTPX internal total"),
                             "PTPX internal total")
    total_switching = _nonnegative(_one(text, r"^\s*Net Switching Power\s*=\s*" + NUMBER +
                                   r"\s+\([^\n]+\)\s*$", "PTPX switching total"),
                              "PTPX switching total")
    total_leakage = _nonnegative(_one(text, r"^\s*Cell Leakage Power\s*=\s*" + NUMBER +
                                 r"\s+\([^\n]+\)\s*$", "PTPX leakage total"),
                            "PTPX leakage total")
    total_power = _positive(_one(text, r"^Total Power\s*=\s*" + NUMBER +
                               r"\s+\([^\n]+\)\s*$", "PTPX total power"),
                          "PTPX total power")
    if not math.isclose(total_power, total_internal + total_switching + total_leakage,
                        rel_tol=2e-6, abs_tol=1e-9):
        raise ExtractionError("PTPX total power does not equal internal+switching+leakage")
    if not math.isclose(memory_values[3], sum(memory_values[:3]), rel_tol=2e-6,
                        abs_tol=1e-9):
        raise ExtractionError("PTPX memory total does not equal its components")
    if memory_values[3] <= 0.0:
        raise ExtractionError("PTPX integrated SRAM power is not positive")
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


def parse_vcs_simulation(path):
    text = Path(path).read_text(encoding="utf-8")
    version = _one(
        text,
        r"^Compiler version\s+(V-[^;\s]+);\s+Runtime version\s+(V-[^;\s]+);.*$",
        "native VCS compiler/runtime identity",
    )
    if tuple(version) != (EXPECTED_TOOL_VERSIONS["vcs"],
                          EXPECTED_TOOL_VERSIONS["vcs"]):
        raise ExtractionError("native VCS compiler/runtime version mismatch")
    if len(re.findall(r"^M671_TABLE_A_VCS_PASS\s*$", text, flags=re.MULTILINE)) != 1:
        raise ExtractionError("native VCS report lacks one exact production PASS marker")
    forbidden = (r"^M671_TABLE_A_VCS_FAIL\s*$", r"\bUVM_(?:FATAL|ERROR)\b",
                 r"\bAssertion\s+failed\b", r"(?:^|\s)Fatal:", r"^Error-\[")
    if any(re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
           for pattern in forbidden):
        raise ExtractionError("native VCS report contains a failure signature")
    return {"tool": "vcs", "version": version[0], "status": "PASS"}


def parse_formality_verification(path):
    text = Path(path).read_text(encoding="utf-8")
    version = _one(text, r"^\s*Version\s+(V-\S+)\s+for\s+linux64\s+-.*$",
                   "native Formality version")
    if version != EXPECTED_TOOL_VERSIONS["fm_shell"]:
        raise ExtractionError("native Formality version mismatch")
    if len(re.findall(r"^Verification SUCCEEDED\s*$", text,
                      flags=re.MULTILINE)) != 1:
        raise ExtractionError("native Formality proof is not exactly one success")
    if re.search(r"^Verification (?:FAILED|INCONCLUSIVE)\s*$", text,
                 flags=re.MULTILINE):
        raise ExtractionError("native Formality proof contains a failure result")
    return {"tool": "fm_shell", "version": version, "status": "SUCCEEDED"}


def parse_synopsys_environment(path, expected_analysis):
    text = Path(path).read_text(encoding="utf-8")
    identity = _native_header(text, "environment")
    analysis = _one(text, r"^Analysis View\s*:\s*(\S+)\s*$", "analysis view")
    if analysis != expected_analysis:
        raise ExtractionError("environment analysis view mismatch")
    corner = {
        "operating_condition": _one(text, r"^Operating Condition\s*:\s*(\S+)\s*$",
                                     "operating condition"),
        "process": _one(text, r"^Process\s*:\s*(Slow|Fast|Typical)\s*$", "process"),
        "voltage_v": _number(_one(text, r"^Voltage\s*:\s*" + NUMBER + r"\s*V\s*$",
                                  "voltage"), "voltage"),
        "temperature_c": _number(_one(text, r"^Temperature\s*:\s*" + NUMBER +
                                       r"\s*C\s*$", "temperature"), "temperature"),
    }
    rows = re.findall(r"^Library\s*:\s*(\S+)\s+(\S+)\s+\(File:\s*(\S+)\)\s*$",
                      text, flags=re.MULTILINE)
    libraries = {}
    for role, name, file_path in rows:
        if role in libraries:
            raise ExtractionError("duplicate environment library role")
        libraries[role] = {"library_name": name, "path": file_path}
    expected_roles = {
        "dc": {"logic_setup", "sram_setup"},
        "setup": {"logic_setup", "sram_setup"},
        "hold": {"logic_hold", "sram_hold"},
        "power": {"logic_power", "sram_power"},
    }[expected_analysis]
    if set(libraries) != expected_roles:
        raise ExtractionError("environment library role set mismatch")
    identity.update({"tool": "dc_shell" if expected_analysis == "dc" else "pt_shell",
                     "analysis": expected_analysis})
    return identity, corner, libraries


def parse_sram_macro(path):
    text = Path(path).read_text(encoding="utf-8")
    software = _one(text, r"^#### Software\s*:\s*TSMC MEMORY COMPILER\s+(\S+)\s*\*/\s*$",
                    "memory compiler software")
    technology = _one(text, r"^#### Technology\s*:\s*(.+?)\s*\*/\s*$",
                      "memory technology")
    if "28nm" not in technology:
        raise ExtractionError("memory compiler technology is not 28nm")
    memory_type = _one(text, r"^####\s+Memory Type\s*:\s*(.+?)\s*\*/\s*$",
                       "memory type")
    library_row = _one(text, r"^#### Library Name\s*:\s*(\S+)\s+\(user specify\s*:\s*(\S+)\)\s*\*/\s*$",
                       "memory macro identity")
    library_version = _one(text, r"^#### Library Version\s*:\s*(\S+)\s*\*/\s*$",
                           "memory library version")
    generated = _one(text, r"^#### Generated Time\s*:\s*(.+?)\s*\*/\s*$",
                     "memory generated time")
    process, voltage, temperature = _one(
        text, r"^\s*2\.2 SRAM timing:\((Slow|Fast|Typical),\s*" + NUMBER +
        r",\s*" + NUMBER + r"\s*deg\.\)\s*$", "memory PVT identity")
    organizations = []
    for name in library_row:
        match = re.search(r"(\d+)[xX](\d+)", name)
        if match is None:
            raise ExtractionError("memory macro name lacks depth x width")
        organizations.append((int(match.group(1)), int(match.group(2))))
    if len(set(organizations)) != 1:
        raise ExtractionError("memory library and user macro organization differ")
    depth_words, width_bits = organizations[0]
    address = _one(text, r"^\s*A\[(\d+):0\]\s+" + NUMBER + r"\s*$",
                   "memory address pin")
    data = _one(text, r"^\s*D\[(\d+):0\]\s+" + NUMBER + r"\s*$",
                "memory data pin")
    if 1 << (int(address[0]) + 1) != depth_words or int(data[0]) + 1 != width_bits:
        raise ExtractionError("memory pins do not match macro depth x width")
    if "Single Port SRAM" in memory_type:
        port_type, port_count = "1RW", 1
    elif "Two Port SRAM" in memory_type:
        port_type, port_count = "1R1W", 2
    else:
        raise ExtractionError("unsupported memory port type")
    area_um2 = _positive(_one(text, r"^\s*\|\s*" + NUMBER + r"\s*\|\s*" + NUMBER +
                            r"\s*\|\s*" + NUMBER + r"\s*\|\s*$",
                            "memory macro dimensions")[2], "macro area")
    leakage_ua = _nonnegative(_one(text, r"^\s*Leakage Current\s+" + NUMBER +
                              r"\s*\(uA\).*$", "memory leakage current"),
                         "macro leakage current")
    read_ua_per_mhz = _nonnegative(_one(text, r"^\s*Read\s+" + NUMBER +
                                   r"\s*\(uA/MHz\)\s*$", "memory read current"),
                              "macro read current")
    write_ua_per_mhz = _nonnegative(_one(text, r"^\s*Write\s+" + NUMBER +
                                    r"\s*\(uA/MHz\)\s*$", "memory write current"),
                               "macro write current")
    bytes_per_instance = (depth_words * width_bits + 7) // 8
    return {
        "tool": "memory_compiler", "version": software, "technology": technology,
        "library": library_row[0], "macro": library_row[1],
        "library_version": library_version, "generated_time": generated.strip(),
        "process": process, "voltage_v": _number(voltage, "macro voltage"),
        "temperature_c": _number(temperature, "macro temperature"),
        "depth_words": depth_words, "width_bits": width_bits,
        "port_type": port_type, "port_count": port_count,
        "macro_rounded_bytes_per_instance": bytes_per_instance,
    }, {
        "sram_macro_area_mm2_per_instance": area_um2 / 1e6,
        "sram_leakage_current_ua_per_instance": leakage_ua,
        "sram_read_current_ua_per_mhz_per_instance": read_ua_per_mhz,
        "sram_write_current_ua_per_mhz_per_instance": write_ua_per_mhz,
    }


def _target_corner_equal(actual, expected, label):
    if (actual.get("operating_condition") != expected["operating_condition"] or
            actual.get("process") != expected["process"] or
            not math.isclose(float(actual.get("voltage_v", math.nan)), expected["voltage_v"],
                             rel_tol=0.0, abs_tol=1e-12) or
            not math.isclose(float(actual.get("temperature_c", math.nan)),
                             expected["temperature_c"], rel_tol=0.0, abs_tol=1e-12)):
        raise ExtractionError(label + " corner mismatch")


def _expected_generation_argv(receipt):
    executables = receipt["tool_executables"]
    scripts = receipt["command_scripts"]
    return {
        "vcs": [executables["vcs"]["file"]["path"], "-full64", "-sverilog",
                "-f", scripts["vcs"]["path"]],
        "dc": [executables["dc_shell"]["file"]["path"], "-f", scripts["dc"]["path"]],
        "formality": [executables["fm_shell"]["file"]["path"], "-f",
                      scripts["formality"]["path"]],
        "pt_setup": [executables["pt_shell"]["file"]["path"], "-f",
                     scripts["pt_setup"]["path"]],
        "pt_hold": [executables["pt_shell"]["file"]["path"], "-f",
                    scripts["pt_hold"]["path"]],
        "ptpx": [executables["pt_shell"]["file"]["path"], "-f",
                 scripts["ptpx"]["path"]],
        "memory_compiler": [executables["memory_compiler"]["file"]["path"], "-f",
                            scripts["memory_compiler"]["path"]],
    }


def _step_input_roots(step, receipt):
    rtl = {"rtl:" + role: receipt["rtl_sources"][role]["sha256"]
           for role in RTL_SOURCE_ROLES}
    if step == "vcs":
        return rtl
    if step == "dc":
        common = {"rtl:design_rtl": receipt["rtl_sources"]["design_rtl"]["sha256"],
                  "sdc": receipt["sdc"]["sha256"]}
        common.update({role: receipt["library_dbs"][role]["file"]["sha256"]
                       for role in ("logic_setup", "sram_setup")})
    elif step == "formality":
        common = {"rtl:design_rtl": receipt["rtl_sources"]["design_rtl"]["sha256"],
                  "netlist": receipt["netlist"]["sha256"]}
        common.update({role: receipt["library_dbs"][role]["file"]["sha256"]
                       for role in ("logic_setup", "sram_setup")})
    elif step == "pt_setup":
        common = {"netlist": receipt["netlist"]["sha256"],
                  "sdc": receipt["sdc"]["sha256"]}
        common.update({role: receipt["library_dbs"][role]["file"]["sha256"]
                       for role in ("logic_setup", "sram_setup")})
    elif step == "pt_hold":
        common = {"netlist": receipt["netlist"]["sha256"],
                  "sdc": receipt["sdc"]["sha256"]}
        common.update({role: receipt["library_dbs"][role]["file"]["sha256"]
                       for role in ("logic_hold", "sram_hold")})
    elif step == "ptpx":
        common = {"netlist": receipt["netlist"]["sha256"],
                  "sdc": receipt["sdc"]["sha256"]}
        common.update({role: receipt["library_dbs"][role]["file"]["sha256"]
                       for role in ("logic_power", "sram_power")})
        common["activity"] = receipt["activity"]["sha256"]
    else:
        common = {"memory_inventory": receipt["memory_inventory_sha256"]}
    return common


def _step_outputs(step, receipt):
    reports = receipt["output_reports"]
    return {
        "vcs": {"vcs_simulation": reports["vcs_simulation"]["sha256"],
                "activity": receipt["activity"]["sha256"]},
        "dc": {"dc_area": reports["dc_area"]["sha256"],
               "dc_environment": reports["dc_environment"]["sha256"],
               "mapped_netlist": receipt["netlist"]["sha256"]},
        "formality": {"formality_verification":
                      reports["formality_verification"]["sha256"]},
        "pt_setup": {name: reports[name]["sha256"] for name in
                     ("pt_setup", "pt_setup_environment")},
        "pt_hold": {name: reports[name]["sha256"] for name in
                    ("pt_hold", "pt_hold_environment")},
        "ptpx": {name: reports[name]["sha256"] for name in
                 ("ptpx_power", "ptpx_environment")},
        "memory_compiler": {name: reports[name]["sha256"] for name in
                            ("weight_sram_macro", "state_sram_macro",
                             "parent_scratch_macro")},
    }[step]


def _parse_provenance_log(path):
    text = Path(path).read_text(encoding="utf-8")
    block = _one(text, r"^M671_PROVENANCE_BEGIN\s*$([\s\S]*?)^M671_PROVENANCE_END\s*$",
                 "provenance log block")
    fields = {}
    outputs = {}
    for line in block.strip().splitlines():
        if line.startswith("OUTPUT "):
            parts = line.split()
            if len(parts) != 3 or parts[1] in outputs:
                raise ExtractionError("malformed provenance output row")
            outputs[parts[1]] = parts[2]
        else:
            parts = line.split(None, 1)
            if len(parts) != 2 or parts[0] in fields:
                raise ExtractionError("malformed provenance field")
            fields[parts[0]] = parts[1]
    required = {"STEP", "TOOL", "TOOL_EXECUTABLE_SHA256", "TOOL_VERSION",
                "ARGV_SHA256", "COMMAND_SCRIPT_SHA256", "INPUT_ROOT_SHA256",
                "EXIT_STATUS"}
    if set(fields) != required:
        raise ExtractionError("provenance log field set mismatch")
    return fields, outputs


def _parse_command_root(path):
    text = Path(path).read_text(encoding="utf-8")
    block = _one(text,
                 r"^M671_COMMAND_ROOT_BEGIN\s*$([\s\S]*?)^M671_COMMAND_ROOT_END\s*$",
                 "command root block")
    fields = {}
    outputs = {}
    for line in block.strip().splitlines():
        if line.startswith("OUTPUT "):
            parts = line.split()
            if len(parts) != 4 or parts[1] in outputs:
                raise ExtractionError("malformed command output row")
            outputs[parts[1]] = {"path": parts[2], "sha256": parts[3]}
        else:
            parts = line.split(None, 1)
            if len(parts) != 2 or parts[0] in fields:
                raise ExtractionError("malformed command root field")
            fields[parts[0]] = parts[1]
    if set(fields) != {"STEP", "DESIGN", "INPUT_ROOT_SHA256"}:
        raise ExtractionError("command root field set mismatch")
    return fields, outputs


def _validate_library_dbs(entries):
    if not isinstance(entries, dict) or set(entries) != set(LIBRARY_ROLES):
        raise ExtractionError("library DB inventory set mismatch")
    resolved = {}
    for role in LIBRARY_ROLES:
        entry = entries[role]
        _exact(entry, {"role", "library_name", "corner", "file"}, "library DB " + role)
        if (entry["role"] != role or entry["library_name"] != EXPECTED_LIBRARY_NAMES[role]):
            raise ExtractionError("library DB role/name mismatch")
        expected_corner = TARGET_CORNERS["pt_setup" if role.endswith("setup") else
                                         "pt_hold" if role.endswith("hold") else
                                         "ptpx_power"]
        if entry["corner"] != expected_corner:
            raise ExtractionError("library DB target corner mismatch")
        resolved[role] = _file_spec(entry["file"], "library DB " + role,
                                    ("application/octet-stream",),
                                    "hw_autoresearch_nts07/results/")
    return resolved


def _validate_tool_run(receipt_spec, manifest, reports):
    receipt_path = _file_spec(receipt_spec, "tool run receipt", ("application/json",),
                              "hw_autoresearch_nts07/results/")
    receipt = _load_json(receipt_path, "tool run receipt")
    _exact(receipt, TOOL_RUN_FIELDS, "tool run receipt")
    if (receipt["schema"] != "m671.h67.native_tool_run_receipt.r1" or
            receipt["status"] != "PASS_EXIT_ZERO_ROOTED" or
            receipt["row_id"] != manifest["row_id"] or
            receipt["configuration_manifest_sha256"] !=
            manifest["configuration_manifest_sha256"] or
            receipt["design_name"] != manifest["design_name"] or
            receipt["run_id"] != manifest["run_id"] or
            receipt["memory_inventory_sha256"] !=
            _map_sha(manifest["memory_inventory"]) or
            receipt["output_reports"] != manifest["raw_reports"] or
            receipt["library_dbs"] != manifest["library_dbs"]):
        raise ExtractionError("tool run receipt identity/root mismatch")
    if (not isinstance(receipt["command_scripts"], dict) or
            set(receipt["command_scripts"]) != set(STEPS)):
        raise ExtractionError("tool command script set mismatch")
    if (not isinstance(receipt["tool_logs"], dict) or
            set(receipt["tool_logs"]) != set(STEPS)):
        raise ExtractionError("tool log set mismatch")
    if (not isinstance(receipt["tool_executables"], dict) or
            set(receipt["tool_executables"]) != set(TOOL_NAMES)):
        raise ExtractionError("tool executable set mismatch")
    if receipt["tool_versions"] != EXPECTED_TOOL_VERSIONS:
        raise ExtractionError("tool version inventory mismatch")
    if receipt["exit_status"] != {step: 0 for step in STEPS}:
        raise ExtractionError("tool exit status is not all zero")
    if (not isinstance(receipt["rtl_sources"], dict) or
            set(receipt["rtl_sources"]) != set(RTL_SOURCE_ROLES)):
        raise ExtractionError("RTL/testbench/assertion source set mismatch")
    for role in RTL_SOURCE_ROLES:
        _file_spec(receipt["rtl_sources"][role], "RTL source " + role,
                   ("text/plain", "text/x-systemverilog"),
                   "hw_autoresearch_nts07/results/")
    _file_spec(receipt["netlist"], "tool netlist", ("text/plain",),
               "hw_autoresearch_nts07/results/")
    _file_spec(receipt["sdc"], "tool SDC", ("text/plain",),
               "hw_autoresearch_nts07/results/")
    _file_spec(receipt["activity"], "tool activity", ("text/plain",),
               "hw_autoresearch_nts07/results/")
    _validate_library_dbs(receipt["library_dbs"])
    for tool in TOOL_NAMES:
        entry = receipt["tool_executables"][tool]
        _exact(entry, {"file", "version"}, "tool executable " + tool)
        if entry["version"] != EXPECTED_TOOL_VERSIONS[tool]:
            raise ExtractionError("tool executable version mismatch")
        _file_spec(entry["file"], "tool executable " + tool,
                   ("application/octet-stream",), "hw_autoresearch_nts07/results/")
    if receipt["generation_argv"] != _expected_generation_argv(receipt):
        raise ExtractionError("tool generation argv mismatch")
    component_hashes = {
        "netlist": receipt["netlist"]["sha256"], "sdc": receipt["sdc"]["sha256"],
        "activity": receipt["activity"]["sha256"],
        "memory_inventory": receipt["memory_inventory_sha256"],
    }
    component_hashes.update({"rtl:" + role: receipt["rtl_sources"][role]["sha256"]
                             for role in RTL_SOURCE_ROLES})
    component_hashes.update({"report:" + name: receipt["output_reports"][name]["sha256"]
                             for name in sorted(REPORT_FIELDS)})
    component_hashes.update({"argv:" + step:
                             _map_sha(receipt["generation_argv"][step])
                             for step in STEPS})
    component_hashes.update({"library:" + role: receipt["library_dbs"][role]["file"]["sha256"]
                             for role in LIBRARY_ROLES})
    for tool in TOOL_NAMES:
        component_hashes["executable:" + tool] = receipt["tool_executables"][tool]["file"]["sha256"]
    for step in STEPS:
        script = _file_spec(receipt["command_scripts"][step], "command script " + step,
                            ("text/plain",), "hw_autoresearch_nts07/results/")
        log = _file_spec(receipt["tool_logs"][step], "tool log " + step,
                         ("text/plain",), "hw_autoresearch_nts07/results/")
        component_hashes["script:" + step] = receipt["command_scripts"][step]["sha256"]
        component_hashes["log:" + step] = receipt["tool_logs"][step]["sha256"]
        command_fields, command_outputs = _parse_command_root(script)
        expected_command_fields = {
            "STEP": step, "DESIGN": receipt["design_name"],
            "INPUT_ROOT_SHA256": _map_sha(_step_input_roots(step, receipt)),
        }
        expected_command_outputs = {
            name: {"path": receipt["output_reports"][name]["path"], "sha256": digest}
            for name, digest in _step_outputs(step, receipt).items()
            if name in receipt["output_reports"]
        }
        if "mapped_netlist" in _step_outputs(step, receipt):
            expected_command_outputs["mapped_netlist"] = {
                "path": receipt["netlist"]["path"],
                "sha256": receipt["netlist"]["sha256"],
            }
        if "activity" in _step_outputs(step, receipt):
            expected_command_outputs["activity"] = {
                "path": receipt["activity"]["path"],
                "sha256": receipt["activity"]["sha256"],
            }
        if (command_fields != expected_command_fields or
                command_outputs != expected_command_outputs):
            raise ExtractionError("tool command root mismatch: " + step)
        fields, outputs = _parse_provenance_log(log)
        expected_tool = ("vcs" if step == "vcs" else
                         "dc_shell" if step == "dc" else
                         "fm_shell" if step == "formality" else
                         "memory_compiler" if step == "memory_compiler" else "pt_shell")
        expected = {
            "STEP": step, "TOOL": expected_tool,
            "TOOL_EXECUTABLE_SHA256": receipt["tool_executables"][expected_tool]["file"]["sha256"],
            "TOOL_VERSION": EXPECTED_TOOL_VERSIONS[expected_tool],
            "ARGV_SHA256": _map_sha(receipt["generation_argv"][step]),
            "COMMAND_SCRIPT_SHA256": _sha256(script),
            "INPUT_ROOT_SHA256": _map_sha(_step_input_roots(step, receipt)),
            "EXIT_STATUS": "0",
        }
        if fields != expected or outputs != _step_outputs(step, receipt):
            raise ExtractionError("tool provenance log mismatch: " + step)
    if receipt["component_root_sha256"] != _map_sha(component_hashes):
        raise ExtractionError("tool provenance component root mismatch")
    return receipt_path, receipt, component_hashes


def _validate_memory_inventory(manifest, config, macro_identities):
    resource = config.get("resource_tuple")
    if not isinstance(resource, dict):
        raise ExtractionError("configuration lacks resource tuple")
    expected_counts = {
        "weight_sram": (resource.get("weight_sram_bank_count"),
                        resource.get("weight_sram_port_mode")),
        "state_sram": (resource.get("state_sram_bank_count"),
                       resource.get("state_sram_port_mode")),
        "parent_scratch": (resource.get("parent_scratch_bank_count"),
                           resource.get("parent_scratch_port_mode")),
    }
    inventory = manifest["memory_inventory"]
    _exact(inventory, {"target_onchip_sram_bytes_total", "macro_rounded_total_bytes",
                       "macros"}, "memory inventory")
    if (inventory["target_onchip_sram_bytes_total"] !=
            resource.get("onchip_sram_bytes_total") or
            inventory["target_onchip_sram_bytes_total"] != 245760 or
            not isinstance(inventory["macros"], list) or
            len(inventory["macros"]) != len(EXPECTED_MEMORY_MACROS)):
        raise ExtractionError("memory inventory target/config mismatch")
    total = 0
    projected = []
    for actual, frozen in zip(inventory["macros"], EXPECTED_MEMORY_MACROS):
        fields = set(frozen) | {"macro_rounded_bytes_per_instance",
                                "macro_rounded_total_bytes", "library_name"}
        _exact(actual, fields, "memory macro inventory row")
        for field, expected in frozen.items():
            if actual[field] != expected:
                raise ExtractionError("memory macro frozen identity mismatch: " + field)
        if actual["library_name"] != actual["macro_name"].lower():
            raise ExtractionError("memory macro library/name mismatch")
        expected_count, expected_port = expected_counts[actual["role"]]
        if actual["bank_count"] != expected_count or actual["port_type"] != expected_port:
            raise ExtractionError("memory macro bank/port does not project configuration")
        identity = macro_identities[actual["report_id"]]
        compare = {key: actual[key] for key in
                   ("macro_name", "library_name", "depth_words", "width_bits",
                    "port_type", "port_count", "macro_rounded_bytes_per_instance")}
        observed = {
            "macro_name": identity["macro"], "library_name": identity["library"],
            "depth_words": identity["depth_words"], "width_bits": identity["width_bits"],
            "port_type": identity["port_type"], "port_count": identity["port_count"],
            "macro_rounded_bytes_per_instance":
                identity["macro_rounded_bytes_per_instance"],
        }
        if compare != observed or actual["instance_count"] != actual["bank_count"]:
            raise ExtractionError("memory datasheet organization/instance mismatch")
        rounded = actual["macro_rounded_bytes_per_instance"] * actual["instance_count"]
        if actual["macro_rounded_total_bytes"] != rounded:
            raise ExtractionError("memory macro-rounded row mismatch")
        total += rounded
        projected.append(actual)
    if (inventory["macro_rounded_total_bytes"] != total or
            total != inventory["target_onchip_sram_bytes_total"] or total > 245760):
        raise ExtractionError("memory macro-rounded total does not equal target configuration")
    return {"macros": projected, "macro_rounded_total_bytes": total}


def _load_manifest(path):
    raw_manifest_path = str(path)
    manifest_path = _secure_repo_file(raw_manifest_path, "native run manifest",
                                      "hw_autoresearch_nts07/results/")
    manifest = _load_json(manifest_path, "native run manifest")
    _exact(manifest, MANIFEST_FIELDS, "native run manifest")
    if (manifest["schema"] != "m671.h67.native_synopsys_run_manifest.r2" or
            manifest["status"] != "FROZEN_ROOTED_NATIVE_TOOL_RUN"):
        raise ExtractionError("native run manifest schema/status mismatch")
    config_path = _file_spec(manifest["configuration_manifest"], "configuration manifest",
                             ("application/json",),
                             "hw_autoresearch_nts07/system_simulator/")
    config = _load_json(config_path, "configuration manifest")
    if (manifest["configuration_manifest_sha256"] !=
            manifest["configuration_manifest"]["sha256"] or
            config.get("configuration_id") != manifest["m527_configuration_id"]):
        raise ExtractionError("configuration manifest identity mismatch")
    if manifest["target_corners"] != TARGET_CORNERS:
        raise ExtractionError("target corner map is not the frozen typed map")
    _validate_library_dbs(manifest["library_dbs"])
    if not isinstance(manifest["raw_reports"], dict) or set(manifest["raw_reports"]) != REPORT_FIELDS:
        raise ExtractionError("native run manifest report set mismatch")
    report_paths = {}
    for name, spec in manifest["raw_reports"].items():
        report_paths[name] = _file_spec(spec, "native report " + name, ("text/plain",),
                                        "hw_autoresearch_nts07/results/")
    _, tool_run, component_hashes = _validate_tool_run(
        manifest["tool_run_receipt"], manifest, report_paths)
    report_hashes = {name: manifest["raw_reports"][name]["sha256"]
                     for name in sorted(REPORT_FIELDS)}
    expected_run = "m671_%s_%s_%s" % (
        re.sub(r"[^a-zA-Z0-9_]+", "_", manifest["row_id"]),
        manifest["configuration_manifest_sha256"][:12],
        _map_sha(report_hashes)[:12])
    if manifest["run_id"] != expected_run or manifest_path.parent.name != expected_run:
        raise ExtractionError("native run ID/path does not bind reports and provenance")
    return manifest_path, manifest, config, report_paths, tool_run, component_hashes


def extract_from_manifest(path):
    manifest_path, manifest, config, reports, tool_run, component_hashes = _load_manifest(path)
    dc_identity, logic_area = parse_dc_area(reports["dc_area"])
    vcs_identity = parse_vcs_simulation(reports["vcs_simulation"])
    formality_identity = parse_formality_verification(
        reports["formality_verification"])
    power_identity, power = parse_ptpx_power(reports["ptpx_power"])
    setup_identity, setup_wns = parse_pt_timing(reports["pt_setup"], "max")
    hold_identity, hold_wns = parse_pt_timing(reports["pt_hold"], "min")
    dc_env = parse_synopsys_environment(reports["dc_environment"], "dc")
    setup_env = parse_synopsys_environment(reports["pt_setup_environment"], "setup")
    hold_env = parse_synopsys_environment(reports["pt_hold_environment"], "hold")
    power_env = parse_synopsys_environment(reports["ptpx_environment"], "power")
    logic_identities = [dc_identity, power_identity, setup_identity, hold_identity,
                        dc_env[0], setup_env[0], hold_env[0], power_env[0]]
    if {item["design"] for item in logic_identities} != {manifest["design_name"]}:
        raise ExtractionError("native logic reports do not match manifest design")
    expected_versions = [EXPECTED_TOOL_VERSIONS["dc_shell"],
                         EXPECTED_TOOL_VERSIONS["pt_shell"],
                         EXPECTED_TOOL_VERSIONS["pt_shell"],
                         EXPECTED_TOOL_VERSIONS["pt_shell"],
                         EXPECTED_TOOL_VERSIONS["dc_shell"],
                         EXPECTED_TOOL_VERSIONS["pt_shell"],
                         EXPECTED_TOOL_VERSIONS["pt_shell"],
                         EXPECTED_TOOL_VERSIONS["pt_shell"]]
    if [item["version"] for item in logic_identities] != expected_versions:
        raise ExtractionError("native report version does not match rooted executable")
    if (vcs_identity["version"] != tool_run["tool_versions"]["vcs"] or
            formality_identity["version"] !=
            tool_run["tool_versions"]["fm_shell"]):
        raise ExtractionError("native verification report version mismatch")
    environments = {"dc_area": dc_env, "pt_setup": setup_env,
                    "pt_hold": hold_env, "ptpx_power": power_env}
    for view, (_, corner, libraries) in environments.items():
        _target_corner_equal(corner, TARGET_CORNERS[view], view)
        for role, parsed in libraries.items():
            entry = manifest["library_dbs"][role]
            if (parsed["library_name"] != entry["library_name"] or
                    parsed["path"] != entry["file"]["path"]):
                raise ExtractionError("native environment library/DB mismatch")
    if dc_identity["library"] != EXPECTED_LIBRARY_NAMES["logic_setup"]:
        raise ExtractionError("DC area library differs from setup DB")
    macro_identities = {}
    macro_diagnostics = {}
    macro_area = 0.0
    for frozen in EXPECTED_MEMORY_MACROS:
        identity, diagnostic = parse_sram_macro(reports[frozen["report_id"]])
        _target_corner_equal({"operating_condition": TARGET_CORNERS["sram_macro"]["operating_condition"],
                              "process": identity["process"],
                              "voltage_v": identity["voltage_v"],
                              "temperature_c": identity["temperature_c"]},
                             TARGET_CORNERS["sram_macro"], frozen["role"])
        if identity["version"] != EXPECTED_TOOL_VERSIONS["memory_compiler"]:
            raise ExtractionError("memory compiler version mismatch")
        macro_identities[frozen["report_id"]] = identity
        macro_diagnostics[frozen["report_id"]] = diagnostic
    memory_projection = _validate_memory_inventory(manifest, config, macro_identities)
    for item in memory_projection["macros"]:
        macro_area += (macro_diagnostics[item["report_id"]]
                       ["sram_macro_area_mm2_per_instance"] * item["instance_count"])
    logic_internal = power["total_internal_power_mw"] - power["sram_internal_power_mw"]
    logic_switching = power["total_switching_power_mw"] - power["sram_switching_power_mw"]
    logic_leakage = power["total_leakage_power_mw"] - power["sram_leakage_power_mw"]
    if min(logic_internal, logic_switching, logic_leakage) < -1e-12:
        raise ExtractionError("PTPX SRAM power exceeds a chip component")
    if setup_wns < 0.0 or hold_wns < 0.0:
        raise ExtractionError("production setup/hold timing is not met")
    values = {
        "logic_area_mm2": logic_area, "sram_macro_area_mm2": macro_area,
        "logic_internal_power_mw": logic_internal,
        "logic_switching_power_mw": logic_switching,
        "logic_dynamic_power_mw": logic_internal + logic_switching,
        "logic_leakage_power_mw": logic_leakage,
        "logic_total_power_mw": logic_internal + logic_switching + logic_leakage,
        "sram_internal_power_mw": power["sram_internal_power_mw"],
        "sram_switching_power_mw": power["sram_switching_power_mw"],
        "sram_dynamic_power_mw": power["sram_internal_power_mw"] + power["sram_switching_power_mw"],
        "sram_leakage_power_mw": power["sram_leakage_power_mw"],
        "sram_total_power_mw": power["sram_total_power_mw"],
        "total_internal_power_mw": power["total_internal_power_mw"],
        "total_switching_power_mw": power["total_switching_power_mw"],
        "total_dynamic_power_mw": power["total_internal_power_mw"] + power["total_switching_power_mw"],
        "total_leakage_power_mw": power["total_leakage_power_mw"],
        "total_power_mw": power["total_power_mw"],
        "setup_wns_ns": setup_wns, "hold_wns_ns": hold_wns,
    }
    identities = {
        "vcs_simulation": vcs_identity,
        "dc_area": dc_identity, "dc_environment": dc_env[0],
        "formality_verification": formality_identity,
        "pt_setup": setup_identity, "pt_setup_environment": setup_env[0],
        "pt_hold": hold_identity, "pt_hold_environment": hold_env[0],
        "ptpx_power": power_identity, "ptpx_environment": power_env[0],
    }
    identities.update(macro_identities)
    return {
        "manifest_path": manifest["run_id"] + "/native_run_manifest.json",
        "run_identity": {
            "row_id": manifest["row_id"],
            "configuration_manifest_sha256": manifest["configuration_manifest_sha256"],
            "m527_configuration_id": manifest["m527_configuration_id"],
            "operator_scope_sha256": manifest["operator_scope_sha256"],
            "design_name": manifest["design_name"], "run_id": manifest["run_id"],
        },
        "identities": identities, "corners": manifest["target_corners"],
        "library_dbs": manifest["library_dbs"], "memory_inventory": memory_projection,
        "tool_run_receipt_sha256": _sha256(_secure_repo_file(
            manifest["tool_run_receipt"]["path"], "tool run receipt")),
        "provenance_component_root_sha256": tool_run["component_root_sha256"],
        "provenance_component_sha256": component_hashes,
        "values": values, "macro_diagnostics": macro_diagnostics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = extract_from_manifest(args.run_manifest)
    except (OSError, ExtractionError) as exc:
        print("M671_NATIVE_PPA_EXTRACT_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False))
    else:
        print("M671_NATIVE_PPA_EXTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
