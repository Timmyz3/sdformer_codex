#!/usr/bin/env python3
"""Fail-closed M1334 source/future-result checker; never launches EDA."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
BASE = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
NET = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
SDC = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.sdc"
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
TB_OLD = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
MEM = HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv"
SVA = HW / "dc_handoff/tb/m1334_c2_production_activity_assertions.sv"
TB = HW / "dc_handoff/tb/tb_m1334_c2_headline_mapped_production_activity.sv"
UCLI = HW / "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1334_c2_headline_mapped_production_activity_source.py"
CONTRACT = HW / "contracts/m1334_c2_headline_mapped_production_activity_source_contract_r1_20260831.json"
FILELISTS = {
    "k8": HW / "dc_handoff/filelists/date_m1334_c2_k8_mapped_production_activity.f",
    "k1x8": HW / "dc_handoff/filelists/date_m1334_c2_k1x8_mapped_production_activity.f",
}
AXES = {
    "k8": {
        "define": "M979_AXIS_K8", "opposite": "M979_AXIS_K1X8",
        "net_sha": "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
        "sdc_sha": "70a0d0e7700188f5a80f31b06c2f3d401f56c7d1e2a29428e3837064a722a96c",
        "cycles": [51, 131, 486, 1231, 14],
    },
    "k1x8": {
        "define": "M979_AXIS_K1X8", "opposite": "M979_AXIS_K8",
        "net_sha": "65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc",
        "sdc_sha": "24806d5c2d5c0afae2c01d518927e3ca96ec977d000287b0a6bc62fc42a7e317",
        "cycles": [53, 133, 499, 1246, 14],
    },
}
SOURCE_KEYS = {
    "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv": MEM,
    "dc_handoff/tb/m1334_c2_production_activity_assertions.sv": SVA,
    "dc_handoff/tb/tb_m1334_c2_headline_mapped_production_activity.sv": TB,
    "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl": UCLI,
    "dc_handoff/filelists/date_m1334_c2_k8_mapped_production_activity.f": FILELISTS["k8"],
    "dc_handoff/filelists/date_m1334_c2_k1x8_mapped_production_activity.f": FILELISTS["k1x8"],
    "system_simulator/scripts/check_m1334_c2_headline_mapped_production_activity_source.py": CHECKER,
    "system_simulator/tests/test_m1334_c2_headline_mapped_production_activity_source.py": TEST,
}
FROZEN = {
    TB_OLD: "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/review.json": "89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a",
    HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/SHA256SUMS.seal.sha256": "0394ce7e485c780355dbb841797f7fa518171bb00330ae07234a1a9a4e96316f",
    HW / "reviews/m1333_m1332_c2_headline_mapped_production_activity_source_blind_hammer_r1_20260831/review.json": "a78b7b826650c490405f3c2ee003fef904779fb479f4680fc565a4e0ec617574",
    HW / "reviews/m1333_m1332_c2_headline_mapped_production_activity_source_blind_hammer_r1_20260831/SHA256SUMS.seal.sha256": "19ee7ed02f85a5e122b4f04f55c5ef884fe9f6e6cd8e0b8a04808ba625e4beba",
    HW / "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def must(condition, message):
    if not condition:
        raise RuntimeError(message)


def strip_comments(text):
    """Remove SV/C comments while preserving strings and line structure."""
    out, i, state = [], 0, "code"
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == '"':
                out.append(ch); state = "string"; i += 1
            elif ch == "/" and nxt == "/":
                out.extend("  "); state = "line"; i += 2
            elif ch == "/" and nxt == "*":
                out.extend("  "); state = "block"; i += 2
            else:
                out.append(ch); i += 1
        elif state == "string":
            out.append(ch); i += 1
            if ch == "\\" and i < len(text):
                out.append(text[i]); i += 1
            elif ch == '"':
                state = "code"
        elif state == "line":
            if ch == "\n":
                out.append("\n"); state = "code"
            else:
                out.append(" ")
            i += 1
        else:
            if ch == "*" and nxt == "/":
                out.extend("  "); state = "code"; i += 2
            else:
                out.append("\n" if ch == "\n" else " "); i += 1
    must(state not in ("block", "string"), "unterminated comment/string")
    return "".join(out)


def sv_tokens(text):
    active = strip_comments(text)
    return re.findall(r"\$?[A-Za-z_][A-Za-z0-9_$]*|\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ]+|'[01xXzZ]|===|!==|<=|>=|==|!=|&&|\|\||\+\+|--|\S", active)


def module_names(text):
    active = strip_comments(text)
    return re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", active)


def find_block(tokens, marker):
    """Return tokens inside the begin/end following an exact token marker."""
    hits = []
    n = len(marker)
    for pos in range(len(tokens) - n + 1):
        if tokens[pos:pos + n] == marker:
            hits.append(pos + n - 1)
    must(len(hits) == 1, "active structure marker count != 1: " + " ".join(marker))
    start = hits[0]
    must(tokens[start] == "begin", "marker does not end at begin")
    depth = 0
    for pos in range(start, len(tokens)):
        if tokens[pos] == "begin": depth += 1
        elif tokens[pos] == "end":
            depth -= 1
            if depth == 0:
                return tokens[start + 1:pos]
    raise RuntimeError("unclosed begin/end block")


def compact(tokens):
    return " ".join(tokens)


def exact_paths(axis):
    return [
        "+define+" + AXES[axis]["define"],
        "+define+SVA_RUNTIME_ENABLED",
        str(CELL),
        str((BASE / axis / NET).resolve()),
        str(MEM.resolve()), str(TB_OLD.resolve()), str(SVA.resolve()),
        str(TB.resolve()),
    ]


def active_filelist_lines(text):
    active = strip_comments(text)
    return [line.strip() for line in active.splitlines() if line.strip()]


def validate_filelist(text, axis):
    must(axis in AXES, "unknown axis")
    lines = active_filelist_lines(text)
    expected = exact_paths(axis)
    must(lines == expected, axis + " filelist is not the exact ordered allowlist")
    for raw in lines[2:]:
        path = Path(raw)
        must(path.is_absolute() and path.is_file() and not path.is_symlink(),
             "filelist member is missing/symlink/non-file: " + raw)
        must(path.resolve() == path, "filelist member path is not canonical: " + raw)
    net = Path(lines[3])
    must(sha(net) == AXES[axis]["net_sha"],
         axis + " active netlist path/SHA mismatch")
    must(sha(CELL) == "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
         "active cell-model path/SHA mismatch")
    providers = []
    for path in (MEM, TB_OLD, SVA, TB):
        for name in module_names(path.read_text(errors="strict")):
            if name == "m349_fc2_scalar_bank_memory_model":
                providers.append(path.resolve())
    must(providers == [MEM.resolve()],
         "memory module provider is not the unique M1334 allowlisted unit")
    return lines


def validate_memory_source(text):
    tokens = sv_tokens(text)
    active = strip_comments(text)
    must(module_names(text) == ["m349_fc2_scalar_bank_memory_model"],
         "memory compilation unit/module mismatch")
    reset = compact(find_block(tokens, ["if", "(", "rst_core", ")", "begin"]))
    for lhs in ("cycle_q", "held_valid_q", "held_slot_q",
                "endpoint_protocol_fault_q", "request_count",
                "response_count", "pending_count", "live_slot_reuse_error"):
        must(re.search(r"\b" + lhs + r"\s*<=", reset),
             "active reset omits state: " + lhs)
    for lhs in ("pending_q", "due_q", "epoch_q", "generation_q", "tag_q",
                "block_q", "slice_q", "channel_q"):
        must(re.search(r"\b" + lhs + r"\s*\[\s*slot\s*\]\s*<=", reset),
             "active reset omits payload array: " + lhs)

    request_block = compact(find_block(tokens,
        ["if", "(", "request_fire_clean", ")", "begin"]))
    response_block = compact(find_block(tokens,
        ["if", "(", "response_fire_clean", ")", "begin"]))
    for lhs in ("pending_q", "due_q", "epoch_q", "generation_q", "tag_q",
                "block_q", "slice_q", "channel_q"):
        pattern = r"\b" + lhs + r"\s*\[\s*mem_req_slot\s*\]\s*<="
        must(len(re.findall(pattern, active)) == 1
             and re.search(pattern, request_block),
             "request-indexed state write escapes clean-fire guard: " + lhs)
    must(re.search(r"pending_q\s*\[\s*selected_slot\s*\]\s*<=", response_block),
         "response state clear escapes clean-fire guard")
    definition = re.search(r"request_fire_clean\s*=([^;]+);", active, re.S)
    must(definition is not None, "request_fire_clean definition absent")
    gate = definition.group(1)
    for clause in ("mem_req_accept === 1'b1", "mem_req_valid === 1'b1",
                   "mem_req_ready === 1'b1", "request_payload_known"):
        must(clause in gate, "clean request gate omits: " + clause)
    definition = re.search(r"response_fire_clean\s*=([^;]+);", active, re.S)
    must(definition is not None, "response_fire_clean definition absent")
    gate = definition.group(1)
    for clause in ("mem_rsp_accept === 1'b1", "mem_rsp_valid === 1'b1",
                   "mem_rsp_ready === 1'b1", "selected_slot >= 0"):
        must(clause in gate, "clean response gate omits: " + clause)
    must("mem_req_accept === 1'b1 && !request_fire_clean" in active,
         "illegal request accept does not fault without state access")
    must("mem_rsp_accept === 1'b1 && !response_fire_clean" in active,
         "illegal response accept does not fault without state access")


def validate_assertion_source(text):
    active = strip_comments(text)
    must(module_names(text) == ["m1334_c2_production_activity_assertions"],
         "assertion compilation unit/module mismatch")
    required_asserts = [
        "ap_header_accept_exact", "ap_raw_accept_exact",
        "ap_request_accept_exact", "ap_response_accept_exact",
        "ap_result_accept_exact", "ap_done_accept_exact",
        "ap_raw_payload_known", "ap_request_payload_known",
        "ap_response_payload_known", "ap_result_payload_known",
        "ap_done_payload_known", "ap_result_stable_under_stall",
        "ap_done_stable_under_stall", "ap_no_endpoint_fault",
        "ap_no_protocol_fault",
    ]
    for label in required_asserts:
        starts = [m.start() for m in re.finditer(r"\b" + label + r"\s*:", active)]
        must(len(starts) == 1, "active SVA label absent/duplicated: " + label)
        end = active.find(";", starts[0])
        must(end >= 0, "unterminated active SVA: " + label)
        statement = active[starts[0]:end + 1]
        must(re.search(r"\bassert\s+property\s*\(", statement)
             and re.search(r"\)\s*else\s*\$fatal\s*\(", statement),
             "active fail-closed SVA absent: " + label)
    for label in ("cp_source", "cp_endpoint", "cp_commit", "cp_stall", "cp_done"):
        must(re.search(r"\b" + label + r"\s*:\s*cover\s+property\s*\(", active),
             "active runtime cover absent: " + label)
    for phrase in ("M1334 handshake control unknown",
                   "M1334 raw payload unknown",
                   "M1334 request payload unknown",
                   "M1334 response payload unknown",
                   "M1334 result payload unknown",
                   "M1334 done payload unknown",
                   "M1334 result stability violation",
                   "M1334 done stability violation",
                   "M1334 endpoint/DUT fault"):
        must(re.search(r"\$fatal\s*\([^;]*" + re.escape(phrase), active, re.S),
             "procedural fatal path absent: " + phrase)
    must("case_id == 4 && endpoint_count != 0" in active,
         "case4 exact-zero endpoint runtime gate absent")
    must("case_id < 4 && endpoint_count == 0" in active,
         "cases0..3 endpoint runtime gate absent")
    must("M1334 assertion absolute watchdog" in active,
         "absolute watchdog absent")
    reset = compact(find_block(sv_tokens(text),
        ["if", "(", "rst_core", ")", "begin"]))
    for state in ("header_count", "source_count", "endpoint_count",
                  "commit_count", "stall_count", "done_count",
                  "unknown_count", "fatal_count", "check_pending",
                  "result_hold_q", "done_hold_q", "result_snapshot_q",
                  "done_snapshot_q"):
        must(re.search(r"\b" + state + r"\s*<=", reset),
             "assertion reset omits state: " + state)


def active_tcl_commands(text):
    commands = []
    for raw in text.splitlines():
        line, escaped, quoted = [], False, False
        for ch in raw:
            if escaped:
                line.append(ch); escaped = False; continue
            if ch == "\\":
                line.append(ch); escaped = True; continue
            if ch == '"':
                line.append(ch); quoted = not quoted; continue
            if ch == "#" and not quoted:
                break
            line.append(ch)
        command = "".join(line).strip()
        if command:
            commands.append(re.sub(r"\s+", " ", command))
    return commands


def validate_ucli(text):
    scope = "tb_m1334_c2_headline_mapped_production_activity.core.dut"
    expected = [
        "power -gate_level all mda sv", "power " + scope, "run",
        "power -enable", "run", "power -disable",
        "power -report $::env(M1334_SAIF_FILE) 1e-9 " + scope, "quit",
    ]
    commands = active_tcl_commands(text)
    must(commands == expected,
         "active UCLI commands/scope/order differ from exact DUT-only recipe")
    return commands


def _sexpr_tokens(text):
    return re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', text)


def _parse_saif(text):
    tokens = _sexpr_tokens(text)
    must(tokens, "empty SAIF")
    pos = [0]
    def parse_one():
        must(pos[0] < len(tokens) and tokens[pos[0]] == "(",
             "malformed SAIF expression")
        pos[0] += 1
        node = []
        while pos[0] < len(tokens) and tokens[pos[0]] != ")":
            if tokens[pos[0]] == "(": node.append(parse_one())
            else:
                node.append(tokens[pos[0]]); pos[0] += 1
        must(pos[0] < len(tokens), "unterminated SAIF expression")
        pos[0] += 1
        return node
    root = parse_one()
    must(pos[0] == len(tokens) and root and root[0] == "SAIFILE",
         "SAIF must contain one SAIFILE root")
    return root


def _forms(node, tag):
    return [item for item in node[1:]
            if isinstance(item, list) and item and item[0] == tag]


def _direct_instance(node, name):
    hits = [item for item in _forms(node, "INSTANCE")
            if len(item) >= 2 and item[1].lstrip("\\") == name]
    must(len(hits) == 1, "exact SAIF instance absent/duplicated: " + name)
    return hits[0]


def _all_forms(node, tag):
    found = []
    if isinstance(node, list):
        if node and node[0] == tag: found.append(node)
        for item in node:
            if isinstance(item, list): found.extend(_all_forms(item, tag))
    return found


def _activity_under(node):
    activity = {}
    for item in _all_forms(node, "TC"):
        # TC forms are consumed by their parent below, not standalone.
        pass
    def walk(value):
        if not isinstance(value, list): return
        tc = _forms(value, "TC")
        if value and isinstance(value[0], str) and tc and len(tc) == 1:
            name = value[0].lstrip("\\")
            must(len(tc[0]) == 2, "malformed TC record")
            activity[name] = activity.get(name, 0.0) + float(tc[0][1])
        for child in value[1:]:
            if isinstance(child, list): walk(child)
    walk(node)
    return activity


def _cone(activity, prefixes):
    return sum(value for name, value in activity.items()
               if any(name == p or name.startswith(p + "[") for p in prefixes))


def validate_saif(path, axis, case_id, cycles):
    must(axis in AXES and 0 <= case_id < 5, "invalid headline axis/case")
    must(cycles == AXES[axis]["cycles"][case_id],
         "M903 cycle anchor mismatch")
    path = Path(path)
    must(path.is_file() and not path.is_symlink(), "SAIF missing/symlink")
    root = _parse_saif(path.read_text(errors="strict"))
    durations = _forms(root, "DURATION")
    must(len(durations) == 1 and len(durations[0]) == 2,
         "SAIF duration absent/duplicated")
    duration = float(durations[0][1])
    must(abs(duration - cycles * 3.0) <= 1e-6,
         "SAIF duration is not cycle anchor times 3 ns")
    tx = _all_forms(root, "TX")
    must(tx and all(len(item) == 2 and float(item[1]) == 0.0 for item in tx),
         "SAIF has absent/nonzero TX")
    top = _direct_instance(root,
        "tb_m1334_c2_headline_mapped_production_activity")
    core = _direct_instance(top, "core")
    dut = _direct_instance(core, "dut")
    all_dut = [item for item in _all_forms(root, "INSTANCE")
               if len(item) >= 2 and item[1].lstrip("\\") == "dut"]
    must(all_dut == [dut], "SAIF contains substitute/duplicate dut scope")
    activity = _activity_under(dut)
    must(activity, "core.dut subtree has no activity")
    cones = {
        "clock": _cone(activity, ("clk_core",)),
        "source": _cone(activity, ("raw_valid", "raw_accept", "raw_bitmap")),
        "endpoint": _cone(activity, ("mem_req_valid", "mem_req_accept",
                                      "mem_rsp_valid", "mem_rsp_accept")),
        "commit": _cone(activity, ("result_valid", "result_accept",
                                    "result_accumulator")),
        "done": _cone(activity, ("token_done_valid", "token_done_accept")),
    }
    for name in ("clock", "source", "commit", "done"):
        must(cones[name] > 0.0, "zero DUT production cone: " + name)
    if case_id < 4:
        must(cones["endpoint"] > 0.0,
             "nonzero case has zero DUT endpoint activity")
    else:
        must(cones["endpoint"] == 0.0,
             "case4 DUT endpoint activity must equal zero")
    reset_tc = _cone(activity, ("rst_core",))
    must(reset_tc == 0.0, "reset toggled inside DUT production window")
    return {
        "schema": "m1334_c2_production_saif_check_r1",
        "status": "PASS_M1334_HEADLINE_AXIS_DUT_ONLY_PRODUCTION_SAIF",
        "axis": axis, "case": case_id, "cycles": cycles,
        "duration_ns": duration, "saif_sha256": sha(path),
        "tx_nonzero": 0, "reset_tc": reset_tc, "major_cone_tc": cones,
    }


def validate_runtime_log(path, axis, case_id):
    path = Path(path)
    must(path.is_file() and not path.is_symlink(), "runtime log missing/symlink")
    text = path.read_text(errors="strict")
    forbidden = ("Assertion failed", "Error-", "Fatal:", "$fatal",
                 "coverage/fault gate failed", "payload unknown",
                 "stability violation", "protocol fault")
    must(not any(token in text for token in forbidden),
         "runtime log contains fatal/assertion diagnostic")
    pattern = (r"PASS M1334 coverage case=" + str(case_id)
        + r" source=([1-9][0-9]*) endpoint=([0-9]+) commit=([1-9][0-9]*)"
        + r" stall=([1-9][0-9]*) done=1 unknown=0 fatal=0")
    hits = re.findall(pattern, text)
    must(len(hits) == 1, "runtime PASS token absent/duplicated/malformed")
    endpoint = int(hits[0][1])
    must((case_id < 4 and endpoint > 0) or (case_id == 4 and endpoint == 0),
         "runtime endpoint count violates frozen case rule")
    return {"log_sha256": sha(path), "endpoint_count": endpoint}


def validate_inventory(path):
    path = Path(path)
    must(path.is_file() and not path.is_symlink(), "inventory missing/symlink")
    data = json.loads(path.read_text())
    must(data.get("schema") == "m1334_c2_production_activity_inventory_r1",
         "inventory schema mismatch")
    must(data.get("status") == "CANDIDATE_UNSEALED_DO_NOT_CITE",
         "source checker only accepts unsealed candidate inventory")
    entries = data.get("entries")
    must(isinstance(entries, list) and len(entries) == 10,
         "inventory must contain exactly ten entries")
    expected = {(axis, case_id) for axis in AXES for case_id in range(5)}
    seen, saif_paths, log_paths, inodes, results = set(), [], [], set(), []
    root = path.parent.resolve()
    for entry in entries:
        must(set(entry) == {"axis", "case", "cycles", "saif", "saif_sha256",
                           "runtime_log", "runtime_log_sha256"},
             "inventory entry key set mismatch")
        axis, case_id, cycles = entry["axis"], entry["case"], entry["cycles"]
        key = (axis, case_id)
        must(key in expected and key not in seen,
             "inventory axis/case absent or duplicated")
        seen.add(key)
        must(cycles == AXES[axis]["cycles"][case_id],
             "inventory cycle anchor mismatch")
        sp = (root / entry["saif"]).resolve()
        lp = (root / entry["runtime_log"]).resolve()
        must(root == sp.parent and root == lp.parent,
             "inventory paths must be direct children of result directory")
        for item in (sp, lp):
            must(item.is_file() and not item.is_symlink(),
                 "inventory member missing/symlink")
        inode = (os.stat(sp).st_dev, os.stat(sp).st_ino)
        must(inode not in inodes and sp not in saif_paths,
             "SAIF reused by multiple inventory coordinates")
        inodes.add(inode); saif_paths.append(sp); log_paths.append(lp)
        must(sha(sp) == entry["saif_sha256"], "inventory SAIF SHA mismatch")
        must(sha(lp) == entry["runtime_log_sha256"],
             "inventory runtime-log SHA mismatch")
        saif_result = validate_saif(sp, axis, case_id, cycles)
        log_result = validate_runtime_log(lp, axis, case_id)
        results.append({"axis": axis, "case": case_id,
                        "cycles": cycles, "saif": saif_result,
                        "runtime": log_result})
    must(seen == expected, "inventory Cartesian product incomplete")
    disk_saifs = {item.resolve() for item in root.glob("*.saif")
                  if item.is_file() and not item.is_symlink()}
    must(disk_saifs == set(saif_paths),
         "result directory SAIF inventory has missing/extra files")
    must(len(set(log_paths)) == 10,
         "runtime log reused by multiple inventory coordinates")
    return {"schema": "m1334_c2_production_activity_inventory_check_r1",
            "status": "PASS_M1334_EXACT_TEN_FILE_CANDIDATE_INVENTORY",
            "entry_count": 10, "coordinates": sorted(seen),
            "inventory_sha256": sha(path), "results": results}


def validate_static(contract=CONTRACT):
    for path, expected in FROZEN.items():
        must(path.is_file() and not path.is_symlink() and sha(path) == expected,
             "frozen identity drift: " + str(path))
    m903 = json.loads((HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/review.json").read_text())
    must(m903["status"] == "PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED",
         "M903 admission status drift")
    m1333 = json.loads((HW / "reviews/m1333_m1332_c2_headline_mapped_production_activity_source_blind_hammer_r1_20260831/review.json").read_text())
    must(m1333["status"] == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"
         and m1333["false_negative_count"] == 10,
         "M1333 fail-do-not-cite predecessor boundary drift")
    for axis, spec in AXES.items():
        net, sdc = BASE / axis / NET, BASE / axis / SDC
        must(net.is_file() and not net.is_symlink() and sha(net) == spec["net_sha"],
             axis + " mapped netlist identity drift")
        must(sdc.is_file() and not sdc.is_symlink() and sha(sdc) == spec["sdc_sha"],
             axis + " mapped SDC identity drift")
        validate_filelist(FILELISTS[axis].read_text(errors="strict"), axis)
    validate_memory_source(MEM.read_text(errors="strict"))
    validate_assertion_source(SVA.read_text(errors="strict"))
    must(module_names(TB.read_text(errors="strict"))
         == ["tb_m1334_c2_headline_mapped_production_activity"],
         "wrapper compilation unit/module mismatch")
    active_tb = strip_comments(TB.read_text(errors="strict"))
    must(re.search(r"\btb_m979_c2_three_axis_mapped_gate_case_saif\s+core\s*\(", active_tb),
         "wrapper does not instantiate exact frozen M979 core")
    must("core.g_memory[bank].memory.endpoint_protocol_fault_q" in active_tb,
         "endpoint fault tap absent")
    validate_ucli(UCLI.read_text(errors="strict"))

    data = json.loads(Path(contract).read_text())
    must(data.get("schema") == "m1334_c2_headline_mapped_production_activity_source_contract_r1"
         and data.get("status") == "PASS_M1334_SOURCE_ONLY__NO_EDA_EXECUTED",
         "M1334 contract schema/status mismatch")
    must(data.get("predecessor", {}).get("status")
         == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
         "contract does not preserve M1332 failure boundary")
    must(data.get("axes") == ["k8", "k1x8"]
         and data.get("cases_per_axis") == 5,
         "headline geometry drift")
    source_files = data.get("source_files")
    must(isinstance(source_files, list), "contract source_files missing")
    mapping = {item.get("path"): item.get("sha256") for item in source_files}
    must(len(mapping) == len(source_files) and set(mapping) == set(SOURCE_KEYS),
         "contract source_files exact key set mismatch")
    for key, source in SOURCE_KEYS.items():
        must(source.is_file() and not source.is_symlink()
             and mapping[key] == sha(source),
             "contract source path/SHA mismatch: " + key)
    boundary = data.get("claim_boundary", {})
    for key in ("vcs", "mapped_functionality", "saif", "ptpx", "power",
                "energy", "performance", "system_speedup",
                "paper_ppa_ready", "headline"):
        must(boundary.get(key) is False,
             "source contract falsely admits: " + key)
    return {"schema": "m1334_c2_headline_mapped_production_activity_static_check_r1",
            "status": "PASS_M1334_SOURCE_ONLY__NO_EDA",
            "closed_predecessor_false_negatives": 10,
            "axes": ["k8", "k1x8"], "cases": 10,
            "same_frozen_workloads": True, "headline_rtl_modified": False,
            "eda_executed": False, "contract_sha256": sha(contract)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--saif", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--axis", choices=sorted(AXES))
    parser.add_argument("--case", type=int, dest="case_id")
    parser.add_argument("--cycles", type=int)
    args = parser.parse_args()
    must(not (args.saif and args.inventory),
         "SAIF and inventory modes are mutually exclusive")
    if args.inventory:
        out = validate_inventory(args.inventory)
    elif args.saif:
        must(args.axis is not None and args.case_id is not None
             and args.cycles is not None,
             "SAIF mode requires axis/case/cycles")
        out = validate_saif(args.saif, args.axis, args.case_id, args.cycles)
    else:
        out = validate_static(args.contract)
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
