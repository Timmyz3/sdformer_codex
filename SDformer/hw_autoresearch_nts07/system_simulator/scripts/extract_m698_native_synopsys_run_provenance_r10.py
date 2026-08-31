#!/usr/bin/env python3
"""M698 r10 independent-evidence overlay for Table-A native runs.

M691/r9 remains byte-exact and is used only as the scalar/native-report grammar
layer.  This overlay rejects the five receipt-blind false positives found by
M695.  Most importantly, passing this extractor is *not* production authority:
the r10 registry has an empty, code-pinned authority allowlist.  A future
additive registry revision must pin a fresh review of an actual native run.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path


class ExtractionError(ValueError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
R9_PATH = HW_ROOT / "system_simulator/scripts/extract_m691_native_synopsys_run_provenance.py"
R9_SHA256 = "3d8ee74b58df9ecdeb1ed8fb87c7feb3cbf3a6ba81ec49b6b972557d16fec420"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_module(name, path, expected_sha):
    if _sha256(path) != expected_sha:
        raise RuntimeError("sealed dependency SHA drift: " + str(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sealed dependency")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R9 = _load_module("m698_sealed_m691_extractor", R9_PATH, R9_SHA256)
TOOL_NAMES = tuple(R9.TOOL_NAMES)
STRICT_STEPS = tuple(R9.STRICT_STEPS)
SCOPE_ANCHORS = tuple(R9.SCOPE_ANCHORS)
EXPECTED_MEMORY_MACROS = tuple(R9.EXPECTED_MEMORY_MACROS)

EXTENSION_FIELDS = {
    "schema", "status", "evidence_class", "row_id", "design_name", "run_id",
    "r9_run_manifest_sha256", "tool_identity_report", "db_native_read_report",
    "scope_and_formality_report", "pt_saif_annotation_report",
    "netlist_macro_hierarchy_report", "dc_area_split_report",
    "ptpx_macro_power_report", "component_root_sha256",
}
REPORT_MEDIA = ("text/plain",)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX_BUILD = re.compile(r"^[0-9a-f]{40,64}$")
NATIVE_TOOL_BASENAMES = {
    "vcs": {"vcs"},
    "dc_shell": {"dc_shell", "dc_shell-xg-t"},
    "fm_shell": {"fm_shell"},
    "pt_shell": {"pt_shell"},
    "memory_compiler": {"memory_compiler", "mem_compiler"},
}
STEP_TO_LOGICAL_TOOL = {
    "vcs_compile": "vcs", "vcs_run": "simv", "dc": "dc_shell",
    "formality": "fm_shell", "pt_setup": "pt_shell", "pt_hold": "pt_shell",
    "ptpx": "pt_shell", "memory_compiler": "memory_compiler",
}


def _exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ExtractionError(label + " fields differ")


def _load_json(path, label):
    try:
        return R9._load_json(path, label)
    except R9.ExtractionError as exc:
        raise ExtractionError(str(exc))


def _secure_file(raw_path, label, prefix=None):
    try:
        return R9._secure_file(raw_path, label, prefix)
    except R9.ExtractionError as exc:
        raise ExtractionError(str(exc))


def _file_spec(spec, label, media_types=REPORT_MEDIA):
    try:
        return R9._file_spec(spec, label, media_types,
                             "hw_autoresearch_nts07/results/")
    except R9.ExtractionError as exc:
        raise ExtractionError(str(exc))


def _read_block(spec, label, begin, end):
    path = _file_spec(spec, label)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != begin or lines[-1] != end:
        raise ExtractionError(label + " begin/end marker mismatch")
    if any(not line or "\x00" in line for line in lines[1:-1]):
        raise ExtractionError(label + " contains empty/NUL record")
    return path, [line.split("|") for line in lines[1:-1]]


def _validate_tool_identity(spec, loaded, evidence_class):
    _, rows = _read_block(spec, "tool identity report",
                          "M698_TOOL_IDENTITY_V1", "M698_TOOL_IDENTITY_END")
    tool_rows, step_rows = {}, {}
    for fields in rows:
        if fields[0] == "TOOL" and len(fields) == 12:
            name = fields[1]
            if name in tool_rows:
                raise ExtractionError("duplicate tool identity")
            tool_rows[name] = fields[2:]
        elif fields[0] == "STEP" and len(fields) == 6:
            name = fields[1]
            if name in step_rows:
                raise ExtractionError("duplicate step process identity")
            step_rows[name] = fields[2:]
        else:
            raise ExtractionError("tool identity report record mismatch")
    if set(tool_rows) != set(TOOL_NAMES) or set(step_rows) != set(STRICT_STEPS):
        raise ExtractionError("tool/step identity set mismatch")

    tool_run, proof = loaded[4], loaded[7]
    native_hashes, native_build_ids = [], []
    for tool in TOOL_NAMES:
        # family, installed_path, realpath, device, inode, build_id, sha,
        # version_sha, proc_exe_sha, native_read_status
        values = tool_rows[tool]
        family, installed, realpath = values[0], values[1], values[2]
        device, inode, build_id, digest = values[3], values[4], values[5], values[6]
        version_sha, proc_sha, status = values[7], values[8], values[9]
        expected_file = tool_run["tool_executables"][tool]["file"]
        expected_version_sha = _text_sha256(R9.EXPECTED_TOOL_VERSIONS[tool])
        if (family != tool or not installed.startswith("/") or
                not realpath.startswith("/") or not device.isdigit() or
                int(device) <= 0 or not inode.isdigit() or int(inode) <= 0 or
                HEX_BUILD.fullmatch(build_id) is None or
                HEX64.fullmatch(digest) is None or digest != expected_file["sha256"] or
                version_sha != expected_version_sha or proc_sha != digest):
            raise ExtractionError("tool executable identity mismatch: " + tool)
        if evidence_class == "NATIVE_SYNOPSYS_EXECUTION":
            if (Path(installed).name not in NATIVE_TOOL_BASENAMES[tool] or
                    Path(realpath).name not in NATIVE_TOOL_BASENAMES[tool] or
                    "synopsys" not in installed.lower() or
                    installed.startswith(str(REPO_ROOT)) or
                    installed.startswith("/bin/") or installed.startswith("/usr/bin/") or
                    status != "PROC_EXE_AND_VERSION_STDOUT_BOUND"):
                raise ExtractionError("unapproved native tool identity: " + tool)
            native_hashes.append(digest)
            native_build_ids.append(build_id)
        elif status != "SYNTHETIC_GRAMMAR_ONLY":
            raise ExtractionError("synthetic tool identity marker mismatch")
    if evidence_class == "NATIVE_SYNOPSYS_EXECUTION":
        if len(set(native_hashes)) != len(native_hashes) or len(set(native_build_ids)) != len(native_build_ids):
            raise ExtractionError("native tool families are not distinct executable builds")

    for step in STRICT_STEPS:
        # logical_tool, executable_snapshot_path, executable_sha, argv0
        logical, snapshot_path, digest, argv0 = step_rows[step]
        entry = proof["execution_steps"][step]
        expected_logical = STEP_TO_LOGICAL_TOOL[step]
        if (logical != expected_logical or snapshot_path != entry["executable"]["path"] or
                digest != entry["executable"]["sha256"] or argv0 != entry["argv"][0] or
                snapshot_path != argv0):
            raise ExtractionError("step executable/argv0/proc identity mismatch: " + step)
        if step == "vcs_run":
            if digest != proof["simv"]["sha256"]:
                raise ExtractionError("simv process identity mismatch")
        elif digest != tool_run["tool_executables"][expected_logical]["file"]["sha256"]:
            raise ExtractionError("step tool family identity mismatch: " + step)


def _validate_db_native_reads(spec, loaded, evidence_class):
    _, rows = _read_block(spec, "DB native-read report",
                          "M698_DB_NATIVE_READ_V1", "M698_DB_NATIVE_READ_END")
    parsed = {}
    for fields in rows:
        if len(fields) != 11 or fields[0] != "DB":
            raise ExtractionError("DB native-read record mismatch")
        role = fields[1]
        if role in parsed:
            raise ExtractionError("duplicate DB native-read role")
        parsed[role] = fields[2:]
    if set(parsed) != set(R9.LIBRARY_ROLES):
        raise ExtractionError("DB native-read role set mismatch")
    tool_run = loaded[4]
    dc_sha = tool_run["tool_executables"]["dc_shell"]["file"]["sha256"]
    for role in R9.LIBRARY_ROLES:
        # library_name, db_sha, reader_sha, status, cells, opconds,
        # voltage_mV, time_unit, native_library_fingerprint
        values = parsed[role]
        library = tool_run["library_dbs"][role]
        library_name, db_sha, reader_sha, status = values[:4]
        cells, opconds, voltage, time_unit, fingerprint = values[4:]
        wanted_status = ("NATIVE_DC_READ_OK" if evidence_class ==
                         "NATIVE_SYNOPSYS_EXECUTION" else "SYNTHETIC_PARSE_OK")
        if (library_name != library["library_name"] or
                db_sha != library["file"]["sha256"] or reader_sha != dc_sha or
                status != wanted_status or not cells.isdigit() or int(cells) < 100 or
                not opconds.isdigit() or int(opconds) < 1 or not voltage.isdigit() or
                not 500 <= int(voltage) <= 1500 or time_unit not in ("ps", "ns") or
                HEX64.fullmatch(fingerprint) is None):
            raise ExtractionError("DB native-read evidence mismatch: " + role)


def _module_body(text, module):
    matches = re.findall(r"\bmodule\s+" + re.escape(module) +
                         r"\b([\s\S]*?)\bendmodule\b", text)
    return matches[0] if len(matches) == 1 else None


def _validate_native_scope_text(tool_run):
    rtl_path = _file_spec(tool_run["rtl_sources"]["design_rtl"], "native scope RTL",
                          ("text/plain", "text/x-systemverilog"))
    netlist_path = _file_spec(tool_run["netlist"], "native scope mapped netlist")
    if _sha256(rtl_path) == _sha256(netlist_path):
        raise ExtractionError("RTL equals mapped netlist")
    rtl = rtl_path.read_text(encoding="utf-8")
    netlist = netlist_path.read_text(encoding="utf-8")
    cell_tokens = re.compile(r"\b(?:DFF|SDFF|LATCH|AND|NAND|OR|NOR|XOR|XNOR|MUX|AOI|OAI|INV|BUF)[A-Za-z0-9_]*\b")
    for _, module, _ in SCOPE_ANCHORS:
        rtl_body, net_body = _module_body(rtl, module), _module_body(netlist, module)
        if rtl_body is None or net_body is None:
            raise ExtractionError("native scope module absent: " + module)
        rtl_clean = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/|\s+", "", rtl_body)
        if rtl_clean in (";wirealive;", "wirealive;") or not re.search(
                r"\b(always|always_ff|always_comb|assign|if|case)\b|[+*<>=&|^-]", rtl_body):
            raise ExtractionError("behavioral stub operator scope: " + module)
        if len(cell_tokens.findall(net_body)) < 2:
            raise ExtractionError("operator lacks mapped standard-cell census: " + module)


def _validate_scope_report(spec, loaded, evidence_class):
    _, rows = _read_block(spec, "scope/formality report",
                          "M698_SCOPE_FORMALITY_V1", "M698_SCOPE_FORMALITY_END")
    top_rows = [row for row in rows if row[0] == "TOP"]
    op_rows = [row for row in rows if row[0] == "OP"]
    if len(top_rows) != 1 or len(op_rows) != len(SCOPE_ANCHORS):
        raise ExtractionError("scope/formality row count mismatch")
    top = top_rows[0]
    if len(top) != 16:
        raise ExtractionError("scope/formality TOP record mismatch")
    # TOP, design, rtl_sha, netlist_sha, formal_status, compare_points,
    # unmatched, top_seq, top_comb, top_leaf, total_nets, total_pins,
    # mapped_ref_count, mapped_ref_root, elaboration_root, netlist_root
    values = top[1:]
    tool_run, proof = loaded[4], loaded[7]
    rtl_sha = tool_run["rtl_sources"]["design_rtl"]["sha256"]
    net_sha = tool_run["netlist"]["sha256"]
    numeric = values[5:12]
    if (values[0] != proof["design_name"] or values[1] != rtl_sha or
            values[2] != net_sha or values[3] != "PASS" or
            any(not item.isdigit() for item in numeric) or int(values[4]) < 100 or
            int(values[5]) != 0 or any(int(item) <= 0 for item in values[6:12]) or
            any(HEX64.fullmatch(item) is None for item in values[12:15])):
        raise ExtractionError("scope/formality TOP identity mismatch")
    expected = {(op, module, instance) for op, module, instance in SCOPE_ANCHORS}
    actual, sums = set(), [0, 0, 0]
    for row in op_rows:
        if len(row) != 9:
            raise ExtractionError("operator scope record mismatch")
        key = tuple(row[1:4])
        if (key in actual or key not in expected or
                any(not item.isdigit() for item in row[4:7]) or
                any(int(item) <= 0 for item in row[4:7]) or
                int(row[6]) != int(row[4]) + int(row[5]) or
                HEX64.fullmatch(row[7]) is None or row[8] != "MAPPED_STDCELL_REFERENCES"):
            raise ExtractionError("operator semantic/cell census mismatch")
        actual.add(key)
        for index in range(3):
            sums[index] += int(row[4 + index])
    if actual != expected or sums != [int(values[6]), int(values[7]), int(values[8])]:
        raise ExtractionError("operator/top cell census reconciliation mismatch")
    if evidence_class == "NATIVE_SYNOPSYS_EXECUTION":
        if int(values[9]) < 100 or int(values[10]) < 100 or int(values[11]) < 20:
            raise ExtractionError("native mapped design census is implausibly small")
        _validate_native_scope_text(tool_run)
    return {"total_nets": int(values[9]), "total_pins": int(values[10])}


def _saif_tc_names(path):
    text = Path(path).read_text(encoding="utf-8")
    names = re.findall(
        r"\(NET\s+([^\s()]+)(?:(?!\n\s*\(NET\b)[\s\S])*?\(TC\s+[0-9]+",
        text)
    return set(names)


def _validate_saif_annotation(spec, loaded, scope_totals, evidence_class):
    _, rows = _read_block(spec, "PT SAIF annotation report",
                          "M698_PT_SAIF_ANNOTATION_V1",
                          "M698_PT_SAIF_ANNOTATION_END")
    summaries = [row for row in rows if row[0] == "SUMMARY"]
    net_rows = [row for row in rows if row[0] == "NET"]
    pin_rows = [row for row in rows if row[0] == "PIN"]
    if len(summaries) != 1 or any(len(row) != 3 for row in net_rows + pin_rows):
        raise ExtractionError("PT SAIF annotation row mismatch")
    summary = summaries[0]
    if len(summary) != 9:
        raise ExtractionError("PT SAIF summary mismatch")
    proof, tool_run = loaded[7], loaded[4]
    expected = [proof["design_name"], tool_run["activity"]["sha256"],
                tool_run["netlist"]["sha256"]]
    if (summary[1:4] != expected or
            any(not item.isdigit() for item in summary[4:8]) or
            summary[8] != "PT_REPORT_ACTIVITY_DERIVED"):
        raise ExtractionError("PT SAIF summary identity mismatch")
    annotated_nets, total_nets, annotated_pins, total_pins = map(int, summary[4:8])
    if (total_nets != scope_totals["total_nets"] or
            total_pins != scope_totals["total_pins"] or
            len(net_rows) != annotated_nets or len(pin_rows) != annotated_pins or
            len({tuple(row[1:]) for row in net_rows}) != annotated_nets or
            len({tuple(row[1:]) for row in pin_rows}) != annotated_pins or
            annotated_nets / float(total_nets) < 0.95 or
            annotated_pins / float(total_pins) < 0.95):
        raise ExtractionError("PT-derived SAIF coverage reconciliation mismatch")
    saif_path = _secure_file(tool_run["activity"]["path"], "r10 SAIF")
    tc_names = _saif_tc_names(saif_path)
    if not {row[1] for row in net_rows}.issubset(tc_names):
        raise ExtractionError("annotated SAIF net is absent from real TC census")
    if evidence_class == "NATIVE_SYNOPSYS_EXECUTION" and len(tc_names) < 100:
        raise ExtractionError("native SAIF has fewer than 100 distinct TC nets")


def _expected_macro_rows(loaded):
    reports = loaded[3]
    diagnostics = {}
    for frozen in EXPECTED_MEMORY_MACROS:
        try:
            _, diagnostic = R9.R8.parse_sram_macro(reports[frozen["report_id"]])
        except R9.R8.ExtractionError as exc:
            raise ExtractionError(str(exc))
        diagnostics[frozen["report_id"]] = diagnostic
    return R9._expected_macro_instances(diagnostics)


def _validate_macro_crosscheck(extension, loaded):
    expected = _expected_macro_rows(loaded)
    expected_keys = [(row["role"], row["instance"], row["macro_name"])
                     for row in expected]

    _, hierarchy = _read_block(extension["netlist_macro_hierarchy_report"],
                               "netlist macro hierarchy report",
                               "M698_NETLIST_MACRO_HIERARCHY_V1",
                               "M698_NETLIST_MACRO_HIERARCHY_END")
    if (len(hierarchy) != 17 or any(len(row) != 5 or row[0] != "MACRO" or
                                    row[4] != "LINKED" for row in hierarchy) or
            [tuple(row[1:4]) for row in hierarchy] != expected_keys):
        raise ExtractionError("netlist 17-macro hierarchy mismatch")

    _, area_rows = _read_block(extension["dc_area_split_report"],
                               "DC area split report", "M698_DC_AREA_SPLIT_V1",
                               "M698_DC_AREA_SPLIT_END")
    if (len(area_rows) != 1 or len(area_rows[0]) != 5 or
            area_rows[0][0] != "AREA" or
            area_rows[0][4] != "DC_REPORT_HIER_DERIVED"):
        raise ExtractionError("DC area split row mismatch")
    try:
        total, logic, macro = [float(value) for value in area_rows[0][1:4]]
    except ValueError:
        raise ExtractionError("DC area split is not numeric")
    expected_macro = math.fsum(row["area_mm2"] for row in expected)
    try:
        _, expected_total = R9.R8.parse_dc_area(loaded[3]["dc_area"])
    except R9.R8.ExtractionError as exc:
        raise ExtractionError(str(exc))
    if (not all(math.isfinite(value) and value > 0.0 for value in (total, logic, macro)) or
            not math.isclose(total, logic + macro, rel_tol=0.0, abs_tol=1e-12) or
            not math.isclose(total, expected_total, rel_tol=0.0, abs_tol=1e-12) or
            not math.isclose(macro, expected_macro, rel_tol=0.0, abs_tol=1e-12)):
        raise ExtractionError("DC logic/macro/total equation mismatch")

    _, power_rows = _read_block(extension["ptpx_macro_power_report"],
                                "PTPX macro power report", "M698_PTPX_MACRO_POWER_V1",
                                "M698_PTPX_MACRO_POWER_END")
    if len(power_rows) != 17:
        raise ExtractionError("PTPX macro power row count mismatch")
    keys, totals = [], []
    for row in power_rows:
        if len(row) != 9 or row[0] != "MEM_POWER" or row[8] != "PTPX_HIER_DERIVED":
            raise ExtractionError("PTPX macro power record mismatch")
        keys.append(tuple(row[1:4]))
        try:
            internal, switching, leakage, total_power = map(float, row[4:8])
        except ValueError:
            raise ExtractionError("PTPX macro power is not numeric")
        if (not all(math.isfinite(value) and value >= 0.0 for value in
                    (internal, switching, leakage, total_power)) or total_power <= 0.0 or
                not math.isclose(total_power, internal + switching + leakage,
                                 rel_tol=0.0, abs_tol=1e-12)):
            raise ExtractionError("PTPX per-instance power equation mismatch")
        totals.append(total_power)
    if keys != expected_keys:
        raise ExtractionError("PTPX 17-macro instance/reference mismatch")
    try:
        _, base_power = R9.R8.parse_ptpx_power(loaded[3]["ptpx_power"])
    except R9.R8.ExtractionError as exc:
        raise ExtractionError(str(exc))
    if not math.isclose(math.fsum(totals), base_power["sram_total_power_mw"],
                        rel_tol=0.0, abs_tol=1e-12):
        raise ExtractionError("PTPX macro sum/total power mismatch")
    return {"total_area_mm2": total, "logic_area_mm2": logic,
            "macro_area_mm2": macro, "sram_power_mw": math.fsum(totals)}


def _load_extension(path):
    extension_path = _secure_file(str(path), "r10 trust extension",
                                  "hw_autoresearch_nts07/results/")
    extension = _load_json(extension_path, "r10 trust extension")
    _exact(extension, EXTENSION_FIELDS, "r10 trust extension")
    if (extension["schema"] != "m698.h67.native_synopsys_trust_extension.r1" or
            extension["status"] != "STRUCTURAL_EVIDENCE_COMPLETE__NOT_AUTHORITY" or
            extension["evidence_class"] not in
            ("NATIVE_SYNOPSYS_EXECUTION", "SYNTHETIC_GRAMMAR_ONLY")):
        raise ExtractionError("r10 trust extension schema/status mismatch")
    root = dict(extension)
    del root["component_root_sha256"]
    if extension["component_root_sha256"] != R9._map_sha(root):
        raise ExtractionError("r10 trust extension component root mismatch")
    return extension_path, extension


def extract_from_bundle(run_manifest_path, trust_extension_path,
                        allow_synthetic_grammar=False):
    try:
        base_result = R9.extract_from_manifest(run_manifest_path)
        loaded = R9._load_manifest(run_manifest_path)
    except R9.ExtractionError as exc:
        raise ExtractionError(str(exc))
    extension_path, extension = _load_extension(trust_extension_path)
    manifest_path, manifest = loaded[0], loaded[1]
    if (extension["r9_run_manifest_sha256"] != _sha256(manifest_path) or
            extension["row_id"] != manifest["row_id"] or
            extension["design_name"] != manifest["design_name"] or
            extension["run_id"] != manifest["run_id"]):
        raise ExtractionError("r10 extension/r9 run identity mismatch")
    evidence_class = extension["evidence_class"]
    if evidence_class == "SYNTHETIC_GRAMMAR_ONLY" and not allow_synthetic_grammar:
        raise ExtractionError("synthetic grammar evidence is never production evidence")
    _validate_tool_identity(extension["tool_identity_report"], loaded, evidence_class)
    _validate_db_native_reads(extension["db_native_read_report"], loaded, evidence_class)
    scope_totals = _validate_scope_report(
        extension["scope_and_formality_report"], loaded, evidence_class)
    _validate_saif_annotation(
        extension["pt_saif_annotation_report"], loaded, scope_totals, evidence_class)
    physical = _validate_macro_crosscheck(extension, loaded)
    result = dict(base_result)
    result.update({
        "r10_trust_extension_sha256": _sha256(extension_path),
        "r10_evidence_class": evidence_class,
        "r10_structural_evidence_pass": True,
        "r10_production_authority_embedded": False,
        "r10_production_eligible_without_pinned_authority": False,
        "r10_independent_physical_crosscheck": physical,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--trust-extension", required=True)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = extract_from_bundle(args.run_manifest, args.trust_extension, False)
    except (OSError, RuntimeError, ExtractionError) as exc:
        print("M698_NATIVE_PPA_EXTRACT_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False))
    else:
        print("M698_NATIVE_PPA_EXTRACT_PASS__STRUCTURAL_ONLY__AUTHORITY_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
