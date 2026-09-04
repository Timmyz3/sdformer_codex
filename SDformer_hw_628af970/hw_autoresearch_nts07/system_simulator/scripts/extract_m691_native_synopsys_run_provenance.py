#!/usr/bin/env python3
"""M691 r9 fail-closed production provenance overlay.

The sealed M671 r8 parser remains the native-report grammar layer.  This r9
overlay adds execution, design-scope, activity and macro-integration evidence
that must be satisfied before any r8 scalar can be returned.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
from pathlib import Path


class ExtractionError(ValueError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[3]
R8_PATH = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/scripts/extract_m671_native_synopsys_run_provenance.py"
R8_SHA256 = "d635f5b5a63fece0c7a46fdf2f462ee566f1b54a4f0f9acf5e3623ecf063c4fb"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name, path, expected_sha):
    if _sha256(path) != expected_sha:
        raise RuntimeError("sealed dependency SHA drift: " + str(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sealed dependency")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R8 = _load_module("m691_sealed_m671_extractor", R8_PATH, R8_SHA256)
REPORT_FIELDS = set(R8.REPORT_FIELDS)
MANIFEST_FIELDS = set(R8.MANIFEST_FIELDS) | {"production_proof"}
TOOL_NAMES = tuple(R8.TOOL_NAMES)
RTL_SOURCE_ROLES = tuple(R8.RTL_SOURCE_ROLES)
LIBRARY_ROLES = tuple(R8.LIBRARY_ROLES)
TARGET_CORNERS = R8.TARGET_CORNERS
EXPECTED_MEMORY_MACROS = R8.EXPECTED_MEMORY_MACROS
EXPECTED_TOOL_VERSIONS = R8.EXPECTED_TOOL_VERSIONS

SCOPE_ANCHORS = (
    ("patch_embed", "h67_patch_embed", "u_patch_embed"),
    ("Conv2d", "h67_conv2d", "u_conv2d"),
    ("ConvTranspose2d", "h67_convtranspose2d", "u_convtranspose2d"),
    ("fc1", "h67_fc1", "u_fc1"),
    ("fc2", "h67_fc2", "u_fc2"),
    ("dynamic_BN", "h67_dynamic_bn", "u_dynamic_bn"),
    ("ATLIF", "h67_atlif", "u_atlif"),
    ("attention", "h67_attention", "u_attention"),
    ("prediction_head", "h67_prediction_head", "u_prediction_head"),
    ("all_required_preprocess_and_completion", "h67_preprocess_completion",
     "u_preprocess_completion"),
)
STRICT_STEPS = ("vcs_compile", "vcs_run", "dc", "formality", "pt_setup",
                "pt_hold", "ptpx", "memory_compiler")
PROOF_FIELDS = {
    "schema", "status", "row_id", "configuration_manifest_sha256",
    "design_name", "run_id", "legacy_tool_run_receipt_sha256", "simv",
    "scope_anchors", "execution_steps", "tool_version_runs",
    "saif_annotation", "macro_census", "ptpx_memory_census",
    "component_root_sha256",
}
STEP_FIELDS = {
    "executable", "argv", "script", "log", "exit_status", "start_time_ns",
    "end_time_ns", "input_sha256", "output_sha256",
}
VERSION_FIELDS = {
    "executable_sha256", "argv", "log", "exit_status", "reported_version",
}


def _map_sha(value):
    return R8._map_sha(value)


def _exact(value, fields, label):
    try:
        R8._exact(value, fields, label)
    except R8.ExtractionError as exc:
        raise ExtractionError(str(exc))


def _load_json(path, label):
    try:
        return R8._load_json(path, label)
    except R8.ExtractionError as exc:
        raise ExtractionError(str(exc))


def _secure_file(raw_path, label, prefix=None):
    try:
        return R8._secure_repo_file(raw_path, label, prefix)
    except R8.ExtractionError as exc:
        raise ExtractionError(str(exc))


def _file_spec(spec, label, media_types=None, prefix=None):
    try:
        return R8._file_spec(spec, label, media_types, prefix)
    except R8.ExtractionError as exc:
        raise ExtractionError(str(exc))


def _elf_spec(spec, label):
    path = _file_spec(spec, label, ("application/octet-stream",),
                      "hw_autoresearch_nts07/results/")
    data = path.read_bytes()
    mode = path.stat().st_mode
    if len(data) < 4096 or data[:4] != b"\x7fELF" or not (mode & (stat.S_IXUSR |
                                                                  stat.S_IXGRP |
                                                                  stat.S_IXOTH)):
        raise ExtractionError(label + " is not an executable ELF snapshot")
    return path


def _binary_db_spec(spec, label):
    path = _file_spec(spec, label, ("application/octet-stream",),
                      "hw_autoresearch_nts07/results/")
    data = path.read_bytes()
    if len(data) < 4096 or b"\x00" not in data:
        raise ExtractionError(label + " is not a binary DB snapshot")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return path
    raise ExtractionError(label + " is a plaintext DB snapshot")


def _parse_exec_log(path):
    text = Path(path).read_text(encoding="utf-8")
    match = re.findall(r"^M691_EXECUTION_BEGIN\s*$([\s\S]*?)^M691_EXECUTION_END\s*$",
                       text, flags=re.MULTILINE)
    if len(match) != 1:
        raise ExtractionError("execution log block must occur exactly once")
    fields, outputs = {}, {}
    for line in match[0].strip().splitlines():
        parts = line.split()
        if parts and parts[0] == "OUTPUT":
            if len(parts) != 3 or parts[1] in outputs:
                raise ExtractionError("malformed execution output")
            outputs[parts[1]] = parts[2]
        else:
            parts = line.split(None, 1)
            if len(parts) != 2 or parts[0] in fields:
                raise ExtractionError("malformed execution field")
            fields[parts[0]] = parts[1]
    required = {"STEP", "EXECUTABLE_SHA256", "ARGV_SHA256", "SCRIPT_SHA256",
                "INPUT_ROOT_SHA256", "START_TIME_NS", "END_TIME_NS", "EXIT_STATUS"}
    if set(fields) != required:
        raise ExtractionError("execution field set mismatch")
    return fields, outputs


def _module_nonempty(text, top):
    matches = re.findall(r"\bmodule\s+" + re.escape(top) + r"\b([\s\S]*?)\bendmodule\b",
                         text)
    if len(matches) != 1:
        return False
    body = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", matches[0])
    return bool(re.sub(r"[\s;()]+", "", body))


def _has_instance(text, module_name, instance_name):
    pattern = (r"\b" + re.escape(module_name) +
               r"\s*(?:#\s*\([\s\S]*?\)\s*)?" + re.escape(instance_name) + r"\b")
    return re.search(pattern, text) is not None


def _validate_scope(proof, tool_run):
    expected = [{"operator": op, "rtl_module": module, "instance": instance}
                for op, module, instance in SCOPE_ANCHORS]
    if proof["scope_anchors"] != expected:
        raise ExtractionError("ten-operator scope anchor map mismatch")
    rtl_path = _file_spec(tool_run["rtl_sources"]["design_rtl"], "design RTL",
                          ("text/plain", "text/x-systemverilog"),
                          "hw_autoresearch_nts07/results/")
    netlist_path = _file_spec(tool_run["netlist"], "mapped netlist", ("text/plain",),
                              "hw_autoresearch_nts07/results/")
    rtl, netlist = rtl_path.read_text(encoding="utf-8"), netlist_path.read_text(encoding="utf-8")
    top = proof["design_name"]
    if not _module_nonempty(rtl, top) or not _module_nonempty(netlist, top):
        raise ExtractionError("RTL/netlist top is absent or empty")
    for _, module, instance in SCOPE_ANCHORS:
        if (not re.search(r"\bmodule\s+" + re.escape(module) + r"\b", rtl) or
                not _has_instance(rtl, module, instance) or
                not _has_instance(netlist, module, instance)):
            raise ExtractionError("RTL/netlist ten-operator hierarchy mismatch")


def _parse_saif(activity_path, expected_top):
    text = Path(activity_path).read_text(encoding="utf-8")
    duration = re.findall(r"\(DURATION\s+([0-9]+(?:\.[0-9]+)?)\)", text)
    timescale = re.findall(r"\(TIMESCALE\s+([1-9][0-9]*)\s+(s|ms|us|ns|ps|fs)\)", text)
    instances = re.findall(r"\(INSTANCE\s+([^\s()]+)", text)
    toggles = [float(value) for value in re.findall(r"\(TC\s+([0-9]+(?:\.[0-9]+)?)\)", text)]
    if (len(duration) != 1 or float(duration[0]) <= 0.0 or len(timescale) != 1 or
            not instances or instances[0] != expected_top or not toggles or max(toggles) <= 0.0):
        raise ExtractionError("SAIF duration/timescale/top/toggle census mismatch")
    return float(duration[0]), "%s %s" % timescale[0]


def _expected_macro_instances(macro_diagnostics):
    rows = []
    for frozen in EXPECTED_MEMORY_MACROS:
        area = macro_diagnostics[frozen["report_id"]]["sram_macro_area_mm2_per_instance"]
        if frozen["role"] == "weight_sram":
            names = ["u_weight_sram_%d" % index for index in range(8)]
        elif frozen["role"] == "state_sram":
            names = ["u_state_sram_%d" % index for index in range(8)]
        else:
            names = ["u_parent_scratch"]
        for name in names:
            rows.append({"role": frozen["role"], "instance": name,
                         "macro_name": frozen["macro_name"], "area_mm2": area})
    return rows


def _expected_strict_argv(step, proof, tool_run):
    executables = tool_run["tool_executables"]
    if step == "vcs_compile":
        return ([executables["vcs"]["file"]["path"], "-full64", "-sverilog"] +
                [tool_run["rtl_sources"][role]["path"] for role in RTL_SOURCE_ROLES] +
                ["-o", proof["simv"]["path"]])
    if step == "vcs_run":
        return [proof["simv"]["path"]]
    tool = ("dc_shell" if step == "dc" else "fm_shell" if step == "formality" else
            "memory_compiler" if step == "memory_compiler" else "pt_shell")
    return [executables[tool]["file"]["path"], "-f",
            proof["execution_steps"][step]["script"]["path"]]


def _strict_input_output(step, proof, tool_run):
    if step == "vcs_compile":
        inputs = {"rtl:" + role: tool_run["rtl_sources"][role]["sha256"]
                  for role in RTL_SOURCE_ROLES}
        outputs = {"simv": proof["simv"]["sha256"]}
        return inputs, outputs
    if step == "vcs_run":
        inputs = {"simv": proof["simv"]["sha256"]}
        inputs.update({"rtl:" + role: tool_run["rtl_sources"][role]["sha256"]
                       for role in RTL_SOURCE_ROLES})
        outputs = R8._step_outputs("vcs", tool_run)
        return inputs, outputs
    return R8._step_input_roots(step, tool_run), R8._step_outputs(step, tool_run)


def _validate_script(step, path, proof, tool_run):
    text = Path(path).read_text(encoding="utf-8")
    required = {
        "dc": ("read_verilog", "read_sdc", "compile_ultra"),
        "formality": ("read_verilog", "set_top", "verify"),
        "pt_setup": ("read_verilog", "read_sdc", "report_timing"),
        "pt_hold": ("read_verilog", "read_sdc", "report_timing"),
        "ptpx": ("read_verilog", "read_sdc", "read_saif", "report_power"),
        "memory_compiler": ("compile_memory",),
    }[step]
    if any(token not in text for token in required):
        raise ExtractionError("native command script lacks required command: " + step)
    expected_paths = []
    if step == "dc":
        expected_paths = [tool_run["rtl_sources"]["design_rtl"]["path"],
                          tool_run["sdc"]["path"]]
    elif step == "formality":
        expected_paths = [tool_run["rtl_sources"]["design_rtl"]["path"],
                          tool_run["netlist"]["path"]]
    elif step in ("pt_setup", "pt_hold"):
        expected_paths = [tool_run["netlist"]["path"], tool_run["sdc"]["path"]]
    elif step == "ptpx":
        expected_paths = [tool_run["netlist"]["path"], tool_run["sdc"]["path"],
                          tool_run["activity"]["path"]]
    else:
        expected_paths = [row["macro_name"] for row in EXPECTED_MEMORY_MACROS]
    if any(value not in text for value in expected_paths):
        raise ExtractionError("native command script lacks exact rooted input: " + step)


def _validate_proof(spec, manifest, tool_run, reports):
    proof_path = _file_spec(spec, "r9 production proof", ("application/json",),
                            "hw_autoresearch_nts07/results/")
    proof = _load_json(proof_path, "r9 production proof")
    _exact(proof, PROOF_FIELDS, "r9 production proof")
    if (proof["schema"] != "m691.h67.production_execution_proof.r1" or
            proof["status"] != "PASS_WRAPPER_ROOTED_NATIVE_EXECUTION" or
            proof["row_id"] != manifest["row_id"] or
            proof["configuration_manifest_sha256"] != manifest["configuration_manifest_sha256"] or
            proof["design_name"] != manifest["design_name"] or
            proof["run_id"] != manifest["run_id"] or
            proof["legacy_tool_run_receipt_sha256"] !=
            _sha256(_secure_file(manifest["tool_run_receipt"]["path"], "legacy receipt"))):
        raise ExtractionError("r9 production proof identity mismatch")
    _elf_spec(proof["simv"], "simv")
    for tool in TOOL_NAMES:
        _elf_spec(tool_run["tool_executables"][tool]["file"], "tool " + tool)
    for role in LIBRARY_ROLES:
        _binary_db_spec(tool_run["library_dbs"][role]["file"], "DB " + role)
    _validate_scope(proof, tool_run)

    if not isinstance(proof["execution_steps"], dict) or set(proof["execution_steps"]) != set(STRICT_STEPS):
        raise ExtractionError("strict execution step set mismatch")
    for step in STRICT_STEPS:
        entry = proof["execution_steps"][step]
        _exact(entry, STEP_FIELDS, "strict execution step " + step)
        executable = _elf_spec(entry["executable"], "step executable " + step)
        expected_argv = _expected_strict_argv(step, proof, tool_run)
        if entry["argv"] != expected_argv or entry["exit_status"] != 0:
            raise ExtractionError("strict argv/exit mismatch: " + step)
        if (not isinstance(entry["start_time_ns"], int) or
                not isinstance(entry["end_time_ns"], int) or
                entry["start_time_ns"] < 0 or entry["end_time_ns"] <= entry["start_time_ns"]):
            raise ExtractionError("strict execution time mismatch: " + step)
        expected_inputs, expected_outputs = _strict_input_output(step, proof, tool_run)
        if entry["input_sha256"] != expected_inputs or entry["output_sha256"] != expected_outputs:
            raise ExtractionError("strict input/output root mismatch: " + step)
        if step in ("vcs_compile", "vcs_run"):
            if entry["script"] is not None:
                raise ExtractionError("VCS direct execution must not use a metadata script")
        else:
            script = _file_spec(entry["script"], "strict script " + step,
                                ("text/plain",), "hw_autoresearch_nts07/results/")
            _validate_script(step, script, proof, tool_run)
        log = _file_spec(entry["log"], "strict log " + step, ("text/plain",),
                         "hw_autoresearch_nts07/results/")
        fields, outputs = _parse_exec_log(log)
        expected_fields = {
            "STEP": step, "EXECUTABLE_SHA256": entry["executable"]["sha256"],
            "ARGV_SHA256": _map_sha(entry["argv"]),
            "SCRIPT_SHA256": "NONE" if entry["script"] is None else entry["script"]["sha256"],
            "INPUT_ROOT_SHA256": _map_sha(expected_inputs),
            "START_TIME_NS": str(entry["start_time_ns"]),
            "END_TIME_NS": str(entry["end_time_ns"]), "EXIT_STATUS": "0",
        }
        if fields != expected_fields or outputs != expected_outputs:
            raise ExtractionError("strict execution log mismatch: " + step)
        if executable != _secure_file(entry["executable"]["path"], "step executable"):
            raise ExtractionError("strict executable path mismatch")

    if not isinstance(proof["tool_version_runs"], dict) or set(proof["tool_version_runs"]) != set(TOOL_NAMES):
        raise ExtractionError("tool version run set mismatch")
    for tool in TOOL_NAMES:
        item = proof["tool_version_runs"][tool]
        _exact(item, VERSION_FIELDS, "tool version run " + tool)
        executable = tool_run["tool_executables"][tool]["file"]
        flag = "-ID" if tool == "vcs" else "-version"
        if (item["executable_sha256"] != executable["sha256"] or
                item["argv"] != [executable["path"], flag] or item["exit_status"] != 0 or
                item["reported_version"] != EXPECTED_TOOL_VERSIONS[tool]):
            raise ExtractionError("tool version execution mismatch: " + tool)
        log = _file_spec(item["log"], "tool version log " + tool, ("text/plain",),
                         "hw_autoresearch_nts07/results/")
        version_text = log.read_text(encoding="utf-8")
        expected_line = "M691_VERSION %s %s %s" % (
            tool, executable["sha256"], EXPECTED_TOOL_VERSIONS[tool])
        if version_text.strip() != expected_line:
            raise ExtractionError("tool version log mismatch: " + tool)

    duration, timescale = _parse_saif(_secure_file(tool_run["activity"]["path"], "SAIF"),
                                      proof["design_name"])
    annotation = proof["saif_annotation"]
    _exact(annotation, {"activity_sha256", "top_instance", "duration", "timescale",
                        "annotated_nets", "total_nets", "annotated_pins", "total_pins"},
           "SAIF annotation")
    if (annotation["activity_sha256"] != tool_run["activity"]["sha256"] or
            annotation["top_instance"] != proof["design_name"] or
            not math.isclose(float(annotation["duration"]), duration, rel_tol=0.0, abs_tol=0.0) or
            annotation["timescale"] != timescale):
        raise ExtractionError("SAIF annotation identity mismatch")
    for annotated, total in ((annotation["annotated_nets"], annotation["total_nets"]),
                             (annotation["annotated_pins"], annotation["total_pins"])):
        if (not isinstance(annotated, int) or not isinstance(total, int) or total <= 0 or
                annotated < 0 or annotated > total or float(annotated) / total < 0.95):
            raise ExtractionError("SAIF annotation coverage is below 95 percent")

    macro_diagnostics = {}
    for frozen in EXPECTED_MEMORY_MACROS:
        try:
            _, diagnostic = R8.parse_sram_macro(reports[frozen["report_id"]])
        except R8.ExtractionError as exc:
            raise ExtractionError(str(exc))
        macro_diagnostics[frozen["report_id"]] = diagnostic
    expected_instances = _expected_macro_instances(macro_diagnostics)
    census = proof["macro_census"]
    _exact(census, {"area_mode", "dc_total_cell_area_mm2", "logic_cell_area_mm2",
                    "macro_cell_area_mm2", "instances"}, "macro census")
    if census["area_mode"] != "DC_TOTAL_INCLUDES_MACROS" or census["instances"] != expected_instances:
        raise ExtractionError("macro instance/area mode census mismatch")
    netlist_text = _secure_file(tool_run["netlist"]["path"], "netlist").read_text(encoding="utf-8")
    for row in expected_instances:
        if not _has_instance(netlist_text, row["macro_name"], row["instance"]):
            raise ExtractionError("mapped netlist lacks exact macro instance")
    macro_area = math.fsum(row["area_mm2"] for row in expected_instances)
    try:
        _, dc_total = R8.parse_dc_area(reports["dc_area"])
    except R8.ExtractionError as exc:
        raise ExtractionError(str(exc))
    numbers = [census[name] for name in ("dc_total_cell_area_mm2", "logic_cell_area_mm2",
                                        "macro_cell_area_mm2")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or value <= 0.0 for value in numbers):
        raise ExtractionError("macro area census is not positive finite")
    if (not math.isclose(census["macro_cell_area_mm2"], macro_area, rel_tol=0.0, abs_tol=1e-12) or
            not math.isclose(census["dc_total_cell_area_mm2"], dc_total, rel_tol=0.0, abs_tol=1e-12) or
            not math.isclose(census["dc_total_cell_area_mm2"],
                             census["logic_cell_area_mm2"] + census["macro_cell_area_mm2"],
                             rel_tol=0.0, abs_tol=1e-12)):
        raise ExtractionError("DC/macro area inclusion reconciliation mismatch")

    ptpx = proof["ptpx_memory_census"]
    _exact(ptpx, {"instances", "sram_total_power_mw"}, "PTPX memory census")
    if (not isinstance(ptpx["instances"], list) or
            [(row.get("role"), row.get("instance"), row.get("macro_name"))
             for row in ptpx["instances"]] !=
            [(row["role"], row["instance"], row["macro_name"]) for row in expected_instances]):
        raise ExtractionError("PTPX memory instance census mismatch")
    try:
        _, power = R8.parse_ptpx_power(reports["ptpx_power"])
    except R8.ExtractionError as exc:
        raise ExtractionError(str(exc))
    if (not isinstance(ptpx["sram_total_power_mw"], (int, float)) or
            ptpx["sram_total_power_mw"] <= 0.0 or
            not math.isclose(ptpx["sram_total_power_mw"], power["sram_total_power_mw"],
                             rel_tol=0.0, abs_tol=1e-12)):
        raise ExtractionError("PTPX SRAM power census mismatch")

    root_value = dict(proof)
    del root_value["component_root_sha256"]
    if proof["component_root_sha256"] != _map_sha(root_value):
        raise ExtractionError("r9 production proof component root mismatch")
    return proof_path, proof, census


def _load_manifest(path):
    manifest_path = _secure_file(str(path), "r9 native run manifest",
                                 "hw_autoresearch_nts07/results/")
    manifest = _load_json(manifest_path, "r9 native run manifest")
    _exact(manifest, MANIFEST_FIELDS, "r9 native run manifest")
    if (manifest["schema"] != "m691.h67.native_synopsys_run_manifest.r3" or
            manifest["status"] != "FROZEN_ROOTED_NATIVE_TOOL_RUN_R9"):
        raise ExtractionError("r9 run manifest schema/status mismatch")
    config_path = _file_spec(manifest["configuration_manifest"], "configuration manifest",
                             ("application/json",), "hw_autoresearch_nts07/system_simulator/")
    config = _load_json(config_path, "configuration manifest")
    if (manifest["configuration_manifest_sha256"] !=
            manifest["configuration_manifest"]["sha256"] or
            config.get("configuration_id") != manifest["m527_configuration_id"] or
            manifest["target_corners"] != TARGET_CORNERS):
        raise ExtractionError("r9 configuration/corner identity mismatch")
    try:
        R8._validate_library_dbs(manifest["library_dbs"])
    except R8.ExtractionError as exc:
        raise ExtractionError(str(exc))
    if not isinstance(manifest["raw_reports"], dict) or set(manifest["raw_reports"]) != REPORT_FIELDS:
        raise ExtractionError("r9 native report set mismatch")
    reports = {name: _file_spec(spec, "native report " + name, ("text/plain",),
                                "hw_autoresearch_nts07/results/")
               for name, spec in manifest["raw_reports"].items()}
    legacy_manifest = dict(manifest)
    del legacy_manifest["production_proof"]
    legacy_manifest["schema"] = "m671.h67.native_synopsys_run_manifest.r2"
    legacy_manifest["status"] = "FROZEN_ROOTED_NATIVE_TOOL_RUN"
    try:
        _, tool_run, component_hashes = R8._validate_tool_run(
            manifest["tool_run_receipt"], legacy_manifest, reports)
    except R8.ExtractionError as exc:
        raise ExtractionError(str(exc))
    report_hashes = {name: manifest["raw_reports"][name]["sha256"]
                     for name in sorted(REPORT_FIELDS)}
    expected_run = "m691_%s_%s_%s" % (
        re.sub(r"[^a-zA-Z0-9_]+", "_", manifest["row_id"]),
        manifest["configuration_manifest_sha256"][:12], _map_sha(report_hashes)[:12])
    if manifest["run_id"] != expected_run or manifest_path.parent.name != expected_run:
        raise ExtractionError("r9 run ID/path mismatch")
    proof_path, proof, census = _validate_proof(
        manifest["production_proof"], manifest, tool_run, reports)
    return (manifest_path, manifest, config, reports, tool_run, component_hashes,
            proof_path, proof, census)


def _reject_negative_power(values):
    for key, value in values.items():
        if key.endswith("power_mw") and value < 0.0:
            raise ExtractionError("negative derived power component: " + key)


def extract_from_manifest(path):
    loaded = _load_manifest(path)
    base_tuple = loaded[:6]
    original = R8._load_manifest
    R8._load_manifest = lambda ignored: base_tuple
    try:
        result = R8.extract_from_manifest(path)
    except R8.ExtractionError as exc:
        raise ExtractionError(str(exc))
    finally:
        R8._load_manifest = original
    proof_path, proof, census = loaded[6], loaded[7], loaded[8]
    _reject_negative_power(result["values"])
    result["values"]["logic_area_mm2"] = census["logic_cell_area_mm2"]
    result["values"]["sram_macro_area_mm2"] = census["macro_cell_area_mm2"]
    result["production_proof_sha256"] = _sha256(proof_path)
    result["production_proof_component_root_sha256"] = proof["component_root_sha256"]
    result["area_mode"] = census["area_mode"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = extract_from_manifest(args.run_manifest)
    except (OSError, RuntimeError, ExtractionError) as exc:
        print("M691_NATIVE_PPA_EXTRACT_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False))
    else:
        print("M691_NATIVE_PPA_EXTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
