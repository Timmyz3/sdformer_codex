#!/usr/bin/env python3
"""M1635 different-author, read-only hammer for the M1634 DC source bundle."""
from __future__ import print_function

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1634_m1609_c2_registered_fault_three_axis_logic_only_exact_sha_r1.sh"
FILELIST = HW / "dc_handoff/filelists/date_m1634_c2_m1609_registered_fault_three_axis_logic_only_dc.f"
CONTRACT = HW / "contracts/m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_source_contract_r1_20260901.json"
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
M1609 = HW / "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
M214 = HW / "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
M1627 = HW / "reviews/m1627_m1613_c2_registered_fault_directed_vcs_result_independent_hammer_r1_20260901"
M903 = HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
AUTHOR = HW / "reviews/m1634_m1609_c2_registered_fault_three_axis_dc_source_author_receipt_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RELEASE = HW / "contracts/m1636_m1635_m1634_m1609_c2_three_axis_dc_launch_release_r1_20260901.json"
RESULT = HW / "dc_handoff/runs/m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_3p000ns_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_attempt_consumed"
LOCK = HW / "dc_handoff/runs/.m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_launch_lock"
WORK_GLOB = ".m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_work.*"
FAIL_GLOB = "m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_3p000ns_r1_20260901.failed_or_incomplete.*.quarantine"

TOP = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
COMPACTOR = "m214_fc2_raw4_to_descriptor4_terminal_hint_compactor"
PREDECESSOR = "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
EXPECTED_ROWS = (
    "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv",
    "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
    "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
    "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
    "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
    "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
    "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
    "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv",
    "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
)
EXPECTED_SOURCE_SHA = {
    "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv": "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv": "8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0",
    "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv": "529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267",
    "rtl_m218/m218_fc2_tagged_slice_service_island.sv": "f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1",
    "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv": "44f7df331af66ba62fadf5e336b9c0c00d00f809e215aa8e091e9de011c5627e",
    "rtl_m519/m519_fc2_k1_registered_release_service_island.sv": "3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871",
    "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv": "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
    "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv": "6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815",
    "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv": "11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv": "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
    "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv": "3328e52d8cf1eec6098ebb7b0525ac55cd8bd6b2fe5b5e504b337d1a678e3c4b",
}
EXPECTED = {
    RUNNER: "da9cd0d118021eb85c8b548d93f6779ec6d25b6fec7ca5894bdae988a95840b7",
    FILELIST: "03c4dcd546da19d5de231fa80032473e7c365592012661e6ed77019d7bab4f3f",
    CONTRACT: "9f5e5b1cb40da5cd403270ba48ceac9b5a7d6aecd79b7ad98cf3d644d0f8f030",
    TCL: "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    SDC: "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    M1609: "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    M214: "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    M1627 / "review.json": "ab4f2187667301a37fbd5f523687a8971282e642163d42886edcdc138edc43d4",
    M903 / "review.json": "89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a",
    AUTHOR / "review.json": "74c4ff0e300764f6a366965e741802317fa9cdadfd2e5422be06508b126bf6fd",
    AUTHOR / "SHA256SUMS": "3aa6db09416530d84fb99bcd33bb54f7eae7a6019764c8e7dd6d760b89b65bad",
    AUTHOR / "SHA256SUMS.seal.sha256": "2a9d9d95f17a6122904e150dda03b1294e322acd27c408a750585c659095ed53",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class Failure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load_text(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          Failure("non-finite JSON constant: " + value)))


def strict_json(path):
    return strict_load_text(Path(path).read_text(encoding="utf-8"))


def uncomment(text, marker="#"):
    return "\n".join(line.split(marker, 1)[0] for line in text.splitlines())


def verify_regular(path, digest):
    require(path.is_file() and not path.is_symlink() and
            stat.S_ISREG(path.lstat().st_mode), "nonregular: " + str(path))
    require(sha(path) == digest, "identity drift: " + str(path))


def verify_file_seal(payload):
    sidecar = Path(str(payload) + ".sha256")
    outer = Path(str(payload) + ".sha256.seal.sha256")
    verify_regular(payload, sha(payload))
    require(sidecar.is_file() and not sidecar.is_symlink(), "sidecar absent")
    require(outer.is_file() and not outer.is_symlink(), "outer seal absent")
    require(sidecar.read_text(encoding="ascii") ==
            sha(payload) + "  " + payload.name + "\n", "sidecar mismatch")
    require(outer.read_text(encoding="ascii") ==
            sha(sidecar) + "  " + sidecar.name + "\n", "outer seal mismatch")


def verify_dir_seal(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(root.is_dir() and not root.is_symlink(), "sealed dir absent")
    require(manifest.is_file() and not manifest.is_symlink(), "manifest absent")
    require(outer.is_file() and not outer.is_symlink(), "outer absent")
    require(outer.read_text(encoding="ascii") ==
            sha(manifest) + "  SHA256SUMS\n", "directory outer seal mismatch")
    rows = manifest.read_text(encoding="utf-8").splitlines()
    require(rows == sorted(rows, key=lambda row: row.split("  ", 1)[1]),
            "manifest ordering drift")
    listed = {}
    for row in rows:
        require(re.match(r"^[0-9a-f]{64}  (?:\./)?[^/\n][^\n]*$", row) is not None,
                "malformed manifest row")
        digest, raw_name = row.split("  ", 1)
        name = raw_name[2:] if raw_name.startswith("./") else raw_name
        require(name not in listed and not Path(name).is_absolute() and
                all(part not in ("", ".", "..") for part in Path(name).parts),
                "unsafe/duplicate manifest member")
        listed[name] = digest
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        for name in list(dirs) + list(files):
            path = Path(base) / name
            require(not path.is_symlink(), "symlink in sealed authority")
            rel = path.relative_to(root).as_posix()
            if path.is_file() and rel not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                actual.add(rel)
    require(actual == set(listed), "sealed authority topology drift")
    for name, digest in listed.items():
        verify_regular(root / name, digest)


def module_header(text, module):
    match = re.search(r"\bmodule\s+" + re.escape(module) + r"\b(.*?);", text, re.S)
    require(match is not None, "module header absent: " + module)
    return match.group(1)


def ports(text, module):
    return tuple(re.findall(
        r"\b(?:input|output)\s+logic(?:\s+signed)?(?:\s*\[[^\]]+\])?\s+([A-Za-z_]\w*)",
        module_header(text, module)))


def module_map(rows, source_texts):
    mapping = {}
    for row in rows:
        names = re.findall(r"(?m)^\s*module\s+([A-Za-z_]\w*)\b", source_texts[row])
        require(len(names) == 1, "expected one module per source: " + row)
        require(names[0] not in mapping, "duplicate module definition: " + names[0])
        mapping[names[0]] = row
    return mapping


def dependencies(mapping, source_texts):
    result = {}
    names = set(mapping)
    for module, row in mapping.items():
        text = uncomment(source_texts[row], "//")
        result[module] = set()
        for candidate in names:
            if candidate == module:
                continue
            if re.search(r"\b" + re.escape(candidate) +
                         r"\s*(?:#\s*\(|[A-Za-z_]\w*\s*\()", text):
                result[module].add(candidate)
    return result


def reachable(graph, root):
    seen = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(sorted(graph.get(node, ())))
    return seen


def validate_bundle(runner, rows, source_texts, tcl, sdc, contract,
                    old_m214_text):
    require(tuple(rows) == EXPECTED_ROWS, "filelist topology/order drift")
    require(PREDECESSOR not in rows and rows.count(EXPECTED_ROWS[0]) == 1,
            "successor-only compactor selection drift")
    require(set(source_texts) == set(rows), "source-text set drift")
    require(contract.get("exact_sources") == EXPECTED_SOURCE_SHA,
            "contract exact-source ledger drift")

    mapping = module_map(rows, source_texts)
    require(mapping.get(COMPACTOR) == EXPECTED_ROWS[0],
            "M1609 is not the unique compactor definition")
    new_text = source_texts[EXPECTED_ROWS[0]]
    require(ports(new_text, COMPACTOR) == ports(old_m214_text, COMPACTOR),
            "M1609/predecessor port contract drift")
    require(new_text.count("assign protocol_error = fault_q;") == 1 and
            "assign protocol_error = fault_q || illegal_request;" not in new_text,
            "registered-only protocol_error seam drift")
    require(re.search(r"\bif\s*\(illegal_request\)\s*fault_q\s*<=\s*1", new_text) is not None,
            "illegal_request is not sticky-latched into fault_q")

    graph = dependencies(mapping, source_texts)
    top_text = source_texts[EXPECTED_ROWS[-1]]
    branch_roots = (
        ("ARCH_MODE == 0", "g_k1", "m519_fc2_k1_registered_release_8bank_raw4_acc24"),
        ("ARCH_MODE == 1", "g_k8", "m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24"),
        ("ARCH_MODE == 2", "g_k1x8", "m519_fc2_k1x8_registered_release_raw4_acc24"),
    )
    for mode, branch, root in branch_roots:
        pattern = (re.escape(mode) + r"\)\s*begin\s*:\s*" + re.escape(branch) +
                   r".*?\b" + re.escape(root) + r"\s*#\s*\(")
        require(re.search(pattern, top_text, re.S) is not None,
                "top branch/root drift: " + branch)
        cone = reachable(graph, root)
        require(COMPACTOR in cone, "M1609 seam absent from " + branch + " cone")
    require(graph["m216_fc2_raw4_to_source_cap_frontend"] ==
            {"m214_fc2_raw4_to_descriptor4_terminal_hint_compactor",
             "m216_fc2_descriptor4_source_cap_frontend"},
            "raw frontend dependency/bypass drift")

    commands = uncomment(runner)
    require("axis_names=(k1 k8 k1x8)" in commands and
            "axis_modes=(0 1 2)" in commands and
            len(re.findall(r"(?m)^for index in 0 1 2; do$", commands)) == 1,
            "three-axis loop/mode mapping drift")
    require(commands.count('"${DC_SHELL}" -f "${TCL}"') == 1,
            "DC invocation must be one lexical call inside the three-axis loop")
    loop_start = commands.index("for index in 0 1 2; do")
    loop_end = commands.index("\ndone", loop_start)
    call = commands.index('"${DC_SHELL}" -f "${TCL}"')
    require(loop_start < call < loop_end, "DC invocation escaped axis loop")
    for token in (
            'DESIGN_NAME="${DESIGN}"', 'HW_ROOT="${HW_ROOT}"',
            'RTL_FILELIST="${FILELIST}"', 'LIB_DB="${SLOW_DB}"',
            'MIN_LIB_DB="${FAST_DB}"', 'SDC_FILE="${SDC}"',
            'OPERATING_CONDITION=ssg0p9v125c', 'CLOCK_PERIOD_NS=3.000',
            'ELAB_PARAMETERS="ARCH_MODE=${mode}"', 'env -i PATH=/usr/bin:/bin'):
        require(token in commands[loop_start:loop_end], "axis call drift: " + token)
    require(not re.search(r"(?mi)^\s*(?:cp|mv|ln)\b[^\n]*(?:M872_RESULT|\.ddc|_mapped\.v)", commands),
            "old netlist/artifact is imported into fresh flow")
    require(not re.search(r"(?i)\b(?:read_ddc|read_verilog|read_file|read_db)\b", tcl),
            "Tcl reads prebuilt netlist/database as design input")
    require(re.search(r"(?m)^analyze -format sverilog -define SYNTHESIS \$rtl_files$", tcl) is not None,
            "fresh RTL analyze command drift")
    require(re.search(r"(?m)^\s*elaborate \$design_name -parameters \$dc_parameters$", tcl) is not None,
            "parameterized fresh elaborate drift")
    require(len(re.findall(r"(?m)^\s*compile_ultra\s*$", uncomment(tcl))) == 1 and
            not re.search(r"(?m)^\s*compile(?:\s|$)", uncomment(tcl)),
            "compile count/type drift")

    require("set_app_var target_library [list $lib_db]" in tcl and
            'set_min_library $lib_db -min_version $min_lib_db' in tcl and
            "source $sdc_file" in tcl, "common library/SDC plumbing drift")
    require("set_wire_load_model -name ZeroWireload [current_design]" in tcl,
            "ZeroWireload drift")
    combined = uncomment(tcl) + "\n" + uncomment(sdc)
    for token in ("set_propagated_clock", "clock_opt", "compile_clock_tree",
                  "create_generated_clock", "set_false_path", "set_multicycle_path",
                  "set_min_delay", "set_max_delay", "set_disable_timing",
                  "set_case_analysis"):
        require(re.search(r"(?m)^\s*" + token + r"\b", combined) is None,
                "ideal/common constraint violation: " + token)
    require(re.search(r"create_clock[\s\S]*?-period\s+\$clock_period_ns", sdc) is not None and
            "set_clock_uncertainty -setup 0.200" in sdc and
            "set_clock_uncertainty -hold 0.050" in sdc,
            "3ns/uncertainty constraint drift")
    require(not re.search(r"(?i)\b(?:sram|memory)_macro\b|\.lib\b|\.lef\b", "\n".join(rows)),
            "macro source/library entered logic-only filelist")

    require(contract.get("status") ==
            "SOURCE_ONLY_M1634_M1609_C2_REGISTERED_FAULT_THREE_AXIS_LOGIC_ONLY_DC__NO_EDA_AUTHORIZED",
            "source-only contract status drift")
    require(contract.get("authorization") == {
        "dc_runs_now": 0, "future_dc_shell_runs_max": 3,
        "all_other_eda_runs": 0}, "authorization budget drift")
    fair = contract.get("fair_three_axis_definition", {})
    require(fair.get("top") == TOP and fair.get("axis_order") ==
            ["k1", "k8", "k1x8"] and
            [fair.get("axes", {}).get(axis, {}).get("arch_mode")
             for axis in ("k1", "k8", "k1x8")] == [0, 1, 2] and
            all(fair.get(key) is True for key in
                ("all_axes_same_filelist", "all_axes_same_constraints",
                 "all_axes_same_libraries")) and
            fair.get("frozen_baseline_netlist_reuse") is False,
            "fair three-axis definition drift")
    physical = contract.get("physical_flow", {})
    require(physical.get("clock_period_ns") == 3.0 and
            physical.get("setup_uncertainty_ns") == 0.2 and
            physical.get("hold_uncertainty_ns") == 0.05 and
            physical.get("clock") == "ideal pre-CTS" and
            physical.get("wireload") == "ZeroWireload" and
            physical.get("macro_count") == 0 and
            physical.get("flow") == "one compile_ultra per fresh architecture axis",
            "physical labeling drift")
    claims = contract.get("claim_boundary", {})
    require(claims.get("source_only") is True and
            claims.get("structural_source_cone_closed") is True and
            all(claims.get(key) is False for key in (
                "dc_authorized", "dc_completed", "fresh_mapped_k8",
                "fresh_mapped_k1x8", "setup_area", "hold_closed", "power",
                "energy", "formality", "paper_ppa_ready", "system_speedup",
                "paper_headline", "docs359_modified")), "claim boundary drift")
    future = contract.get("future_release_chain", {})
    require(future.get("source_hammer_status") ==
            "PASS_M1635_M1634_M1609_C2_THREE_AXIS_DC_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT" and
            future.get("release_status") ==
            "AUTHORIZE_ONE_M1634_M1609_C2_THREE_AXIS_LOGIC_ONLY_DC_ATTEMPT" and
            future.get("caller_must_pin_runner_and_release_sha") is True and
            future.get("present_at_source_authoring") is False,
            "future release chain drift")
    require(all(token in commands for token in (
                'verify_dir_seal "${HAMMER_DIR}"',
                'verify_file_seal "${RELEASE}"', 'mkdir -- "${ATTEMPT}"')),
            "future gate/attempt token absent")
    require(commands.index('verify_dir_seal "${HAMMER_DIR}"') <
            commands.index('verify_file_seal "${RELEASE}"') <
            commands.index('mkdir -- "${ATTEMPT}"') < call,
            "hammer/release/attempt/tool order drift")
    require('M1634_EXPECTED_DC_RUNNER_SHA256' in commands and
            'M1634_EXPECTED_DC_RELEASE_SHA256' in commands,
            "caller identity pins absent")
    require("rm -rf" not in commands and "automatic_retry':True" not in commands and
            "pt_shell" not in commands and "fm_shell" not in commands and
            "/opt/synopsys/vcs" not in commands,
            "destructive/retry/other-EDA token present")


def mutated_contract(contract, path, value):
    candidate = copy.deepcopy(contract)
    cursor = candidate
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return candidate


def run_mutations(baseline):
    runner, rows, sources, tcl, sdc, contract, old_m214 = baseline
    rejected = []

    def reject(label, mutate):
        candidate = [runner, list(rows), dict(sources), tcl, sdc,
                     copy.deepcopy(contract), old_m214]
        mutate(candidate)
        try:
            validate_bundle(*candidate)
        except Failure:
            rejected.append(label)
            return
        raise Failure("mutation escaped: " + label)

    reject("predecessor_selected", lambda c: c[1].__setitem__(0, PREDECESSOR))
    reject("duplicate_successor", lambda c: c[1].insert(1, EXPECTED_ROWS[0]))
    reject("source_reorder", lambda c: c[1].__setitem__(slice(0, 2), c[1][0:2][::-1]))
    reject("source_drop", lambda c: c[1].pop(8))
    reject("contract_source_hash", lambda c: c[5]["exact_sources"].__setitem__(EXPECTED_ROWS[0], "0" * 64))
    reject("second_compactor_definition", lambda c: c[2].__setitem__(EXPECTED_ROWS[1], c[2][EXPECTED_ROWS[1]] + "\nmodule " + COMPACTOR + "; endmodule\n"))
    reject("combinational_fault_pulse", lambda c: c[2].__setitem__(EXPECTED_ROWS[0], c[2][EXPECTED_ROWS[0]].replace("assign protocol_error = fault_q;", "assign protocol_error = fault_q || illegal_request;")))
    reject("fault_not_latched", lambda c: c[2].__setitem__(EXPECTED_ROWS[0], c[2][EXPECTED_ROWS[0]].replace("if (illegal_request) fault_q <= 1;", "if (illegal_request) fault_q <= 0;")))
    reject("k1_branch_bypass", lambda c: c[2].__setitem__(EXPECTED_ROWS[-1], c[2][EXPECTED_ROWS[-1]].replace("m519_fc2_k1_registered_release_8bank_raw4_acc24 #(", "m218_fc2_tagged_slice_service_island #(", 1)))
    reject("k8_branch_bypass", lambda c: c[2].__setitem__(EXPECTED_ROWS[-1], c[2][EXPECTED_ROWS[-1]].replace("m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24 #(", "m218_fc2_tagged_slice_service_island #(", 1)))
    reject("k1x8_branch_bypass", lambda c: c[2].__setitem__(EXPECTED_ROWS[-1], c[2][EXPECTED_ROWS[-1]].replace("m519_fc2_k1x8_registered_release_raw4_acc24 #(", "m218_fc2_tagged_slice_service_island #(", 1)))
    reject("raw_frontend_bypass", lambda c: c[2].__setitem__(EXPECTED_ROWS[2], c[2][EXPECTED_ROWS[2]].replace("m214_fc2_raw4_to_descriptor4_terminal_hint_compactor #(", "m218_fc2_tagged_slice_service_island #(", 1)))
    reject("axis_name_drop", lambda c: c.__setitem__(0, c[0].replace("axis_names=(k1 k8 k1x8)", "axis_names=(k1 k8)")))
    reject("axis_mode_alias", lambda c: c.__setitem__(0, c[0].replace("axis_modes=(0 1 2)", "axis_modes=(0 1 1)")))
    reject("axis_loop_drop", lambda c: c.__setitem__(0, c[0].replace("for index in 0 1 2; do", "for index in 0 1; do")))
    reject("dc_call_outside_loop", lambda c: c.__setitem__(0, c[0].replace('"${DC_SHELL}" -f "${TCL}"', 'true\ndone\n"${DC_SHELL}" -f "${TCL}"\nfor index in 0 1 2; do', 1)))
    reject("old_ddc_copy", lambda c: c.__setitem__(0, c[0].replace("for index in 0 1 2; do", 'cp "${M872_RESULT}/k8/netlist/${DESIGN}.ddc" "${WORK}/old.ddc"\nfor index in 0 1 2; do', 1)))
    reject("axis_specific_filelist", lambda c: c.__setitem__(0, c[0].replace('RTL_FILELIST="${FILELIST}"', 'RTL_FILELIST="${FILELIST}.${axis}"', 1)))
    reject("axis_specific_sdc", lambda c: c.__setitem__(0, c[0].replace('SDC_FILE="${SDC}"', 'SDC_FILE="${SDC}.${axis}"', 1)))
    reject("axis_specific_slow_lib", lambda c: c.__setitem__(0, c[0].replace('LIB_DB="${SLOW_DB}"', 'LIB_DB="${SLOW_DB}.${axis}"', 1)))
    reject("clock_2p5ns", lambda c: c.__setitem__(0, c[0].replace("CLOCK_PERIOD_NS=3.000", "CLOCK_PERIOD_NS=2.500", 1)))
    reject("environment_inheritance", lambda c: c.__setitem__(0, c[0].replace("env -i PATH=/usr/bin:/bin", "env PATH=/usr/bin:/bin", 1)))
    reject("read_old_ddc", lambda c: c.__setitem__(3, c[3].replace("analyze -format sverilog", "read_ddc old.ddc\nanalyze -format sverilog", 1)))
    reject("fresh_analyze_drop", lambda c: c.__setitem__(3, c[3].replace("analyze -format sverilog -define SYNTHESIS $rtl_files", "# analyze removed", 1)))
    reject("second_compile", lambda c: c.__setitem__(3, c[3].replace("    compile_ultra", "    compile_ultra\n    compile_ultra", 1)))
    reject("incremental_compile", lambda c: c.__setitem__(3, c[3].replace("    compile_ultra", "    compile_ultra -incremental", 1)))
    reject("propagated_clock", lambda c: c.__setitem__(4, c[4] + "\nset_propagated_clock [all_clocks]\n"))
    reject("clock_tree", lambda c: c.__setitem__(3, c[3] + "\nclock_opt\n"))
    reject("wireload_change", lambda c: c.__setitem__(3, c[3].replace("ZeroWireload", "ForQA")))
    reject("false_path", lambda c: c.__setitem__(4, c[4] + "\nset_false_path -from [all_inputs]\n"))
    reject("multicycle", lambda c: c.__setitem__(4, c[4] + "\nset_multicycle_path 2 -to [all_outputs]\n"))
    reject("contract_status", lambda c: c.__setitem__(5, mutated_contract(c[5], ["status"], "AUTHORIZED")))
    reject("dc_now", lambda c: c.__setitem__(5, mutated_contract(c[5], ["authorization", "dc_runs_now"], 1)))
    reject("future_run_four", lambda c: c.__setitem__(5, mutated_contract(c[5], ["authorization", "future_dc_shell_runs_max"], 4)))
    reject("other_eda", lambda c: c.__setitem__(5, mutated_contract(c[5], ["authorization", "all_other_eda_runs"], 1)))
    reject("reuse_old_netlist", lambda c: c.__setitem__(5, mutated_contract(c[5], ["fair_three_axis_definition", "frozen_baseline_netlist_reuse"], True)))
    reject("different_filelists", lambda c: c.__setitem__(5, mutated_contract(c[5], ["fair_three_axis_definition", "all_axes_same_filelist"], False)))
    reject("different_constraints", lambda c: c.__setitem__(5, mutated_contract(c[5], ["fair_three_axis_definition", "all_axes_same_constraints"], False)))
    reject("different_libraries", lambda c: c.__setitem__(5, mutated_contract(c[5], ["fair_three_axis_definition", "all_axes_same_libraries"], False)))
    reject("contract_macro_one", lambda c: c.__setitem__(5, mutated_contract(c[5], ["physical_flow", "macro_count"], 1)))
    reject("contract_clock_claim", lambda c: c.__setitem__(5, mutated_contract(c[5], ["physical_flow", "clock"], "propagated")))
    reject("claim_dc_complete", lambda c: c.__setitem__(5, mutated_contract(c[5], ["claim_boundary", "dc_completed"], True)))
    reject("claim_setup_area", lambda c: c.__setitem__(5, mutated_contract(c[5], ["claim_boundary", "setup_area"], True)))
    reject("claim_hold", lambda c: c.__setitem__(5, mutated_contract(c[5], ["claim_boundary", "hold_closed"], True)))
    reject("claim_power", lambda c: c.__setitem__(5, mutated_contract(c[5], ["claim_boundary", "power"], True)))
    reject("claim_system", lambda c: c.__setitem__(5, mutated_contract(c[5], ["claim_boundary", "system_speedup"], True)))
    reject("release_present", lambda c: c.__setitem__(5, mutated_contract(c[5], ["future_release_chain", "present_at_source_authoring"], True)))
    reject("release_status", lambda c: c.__setitem__(5, mutated_contract(c[5], ["future_release_chain", "release_status"], "AUTHORIZE_UNBOUNDED")))
    reject("gate_after_attempt", lambda c: c.__setitem__(0, c[0].replace('verify_dir_seal "${HAMMER_DIR}"', "true", 1)))
    reject("runner_pin_drop", lambda c: c.__setitem__(0, c[0].replace("M1634_EXPECTED_DC_RUNNER_SHA256", "M1634_UNPINNED_RUNNER", 2)))
    require(len(rejected) == 50, "mutation population drift")
    return rejected


def main():
    for path, digest in EXPECTED.items():
        verify_regular(path, digest)
    verify_file_seal(CONTRACT)
    verify_dir_seal(M1627)
    verify_dir_seal(M903)
    verify_dir_seal(AUTHOR)
    require(sha(AUTHOR / "SHA256SUMS") == EXPECTED[AUTHOR / "SHA256SUMS"] and
            sha(AUTHOR / "SHA256SUMS.seal.sha256") ==
            EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"],
            "author receipt seal identity drift")

    require(not RELEASE.exists() and not RELEASE.is_symlink(),
            "M1636 release exists before M1635 admission")
    require(not Path(str(RELEASE) + ".sha256").exists() and
            not Path(str(RELEASE) + ".sha256.seal.sha256").exists(),
            "M1636 release sidecar exists before admission")
    require(not RESULT.exists() and not RESULT.is_symlink(), "M1634 result already exists")
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink(), "M1634 attempt consumed")
    require(not LOCK.exists() and not LOCK.is_symlink(), "M1634 launch lock exists")
    run_root = HW / "dc_handoff/runs"
    require(not list(run_root.glob(WORK_GLOB)), "M1634 PID-work residue exists")
    require(not list(run_root.glob(FAIL_GLOB)), "M1634 failed/quarantine residue exists")

    runner = RUNNER.read_text(encoding="utf-8")
    rows = [row for row in FILELIST.read_text(encoding="utf-8").splitlines()
            if row.strip()]
    sources = {row: (HW / row).read_text(encoding="utf-8") for row in rows}
    for row, digest in EXPECTED_SOURCE_SHA.items():
        verify_regular(HW / row, digest)
    contract = strict_json(CONTRACT)
    tcl = TCL.read_text(encoding="utf-8")
    sdc = SDC.read_text(encoding="utf-8")
    old_m214 = M214.read_text(encoding="utf-8")
    validate_bundle(runner, rows, sources, tcl, sdc, contract, old_m214)

    author = strict_json(AUTHOR / "review.json")
    require(author.get("status") ==
            "PASS_M1634_SOURCE_AUTHOR_RECEIPT__REQUEST_DIFFERENT_AUTHOR_M1635_HAMMER__NO_EDA_AUTHORIZED" and
            author.get("score") >= 95 and author.get("p0_count") == 0 and
            author.get("p1_count") == 0 and author.get("eda_launched_by_author") is False,
            "author receipt admission drift")
    require(author.get("identity", {}).get("runner_sha256") == sha(RUNNER) and
            author.get("identity", {}).get("filelist_sha256") == sha(FILELIST) and
            author.get("identity", {}).get("source_contract_sha256") == sha(CONTRACT),
            "author receipt identity binding drift")
    completed = subprocess.run(
        ["/usr/bin/bash", "-n", str(RUNNER)], cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=10, check=False)
    require(completed.returncode == 0, "bash -n failed: " + completed.stdout)

    rejected = run_mutations((runner, rows, sources, tcl, sdc, contract, old_m214))
    result = {
        "schema": "m1635_independent_source_hammer_mechanical_stdout_v1",
        "status": "PASS",
        "static_checks": 20,
        "mutations_rejected": len(rejected),
        "mutation_labels": rejected,
        "bash_n": "PASS",
        "eda_launched": False,
        "runner_sha256": sha(RUNNER),
        "filelist_sha256": sha(FILELIST),
        "contract_sha256": sha(CONTRACT),
        "m1609_sha256": sha(M1609),
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Failure as error:
        raise SystemExit("FAIL_CLOSED_M1635: " + str(error))
