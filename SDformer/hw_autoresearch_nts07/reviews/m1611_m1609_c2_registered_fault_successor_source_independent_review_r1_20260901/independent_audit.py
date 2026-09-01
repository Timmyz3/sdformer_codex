#!/usr/bin/env python3
from __future__ import print_function

import argparse
import hashlib
import json
import os
import re
import sys


PINS = {
    "predecessor": "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    "successor": "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    "author_test": "bac962b016dcf8556d86826c322391411f685ae1de69751bb1e8c8a289e5d18c",
    "m214_wrapper": "d5caa7f3431761bacde2190412215ef84346a64b3b0559e7cff3116c63f97862",
    "m216_frontend": "529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267",
    "m519_service_top": "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
}

MODULE = "m214_fc2_raw4_to_descriptor4_terminal_hint_compactor"
INTRO = """// M1609 additive source successor of the frozen M214 compactor at
// rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv
// (frozen SHA-256 e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5).
// This file deliberately keeps the legacy module name and full port contract;
// a future filelist must select exactly one of the predecessor or this source.
// The only executable semantic delta is that public protocol_error exposes the
// compactor's synchronous sticky fault_q, not the current-cycle combinational
// illegal_request. illegal_request still gates raw/header acceptance and is
// still sampled into fault_q on a clock edge. Other C2/frontend/service error
// sources remain outside this local boundary and must not be masked upstream.
//
"""
OLD_ASSIGN = "    assign protocol_error = fault_q || illegal_request;\n"
NEW_ASSIGN = """    // M1609: only sampled, sticky compactor faults cross this public boundary.
    // A true illegal_request remains blocked by ready/legal gating and is
    // latched into fault_q by state_update at the accepting clock boundary.
    assign protocol_error = fault_q;
"""


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read(path):
    with open(path, "r") as handle:
        return handle.read()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(
                             ValueError("nonfinite JSON: " + token)))


def pinned_regular(path, expected, label):
    require(os.path.isfile(path) and not os.path.islink(path),
            label + " is absent/nonregular/symlinked")
    require(sha256(path) == expected, label + " SHA mismatch")


def build(project_root):
    hw = os.path.join(project_root, "hw_autoresearch_nts07")
    paths = {
        "predecessor": os.path.join(hw, "rtl_m214",
            "m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"),
        "successor": os.path.join(hw, "rtl_m1609",
            "m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"),
        "author_test": os.path.join(hw, "system_simulator", "tests",
            "test_m1609_c2_compactor_registered_fault_successor_source.py"),
        "m214_wrapper": os.path.join(hw, "rtl_m214",
            "m214_fc2_raw4_to_same_done_load_frontend.sv"),
        "m216_frontend": os.path.join(hw, "rtl_m216",
            "m216_fc2_raw4_to_source_cap_frontend.sv"),
        "m519_service_top": os.path.join(hw, "rtl_m519",
            "m519_fc2_registered_release_standalone_raw4_acc24.sv"),
        "docs359": os.path.join(hw, "docs", "359_DATE终局冻结_20260813.md")
    }
    for label, path in paths.items():
        pinned_regular(path, PINS[label], label)

    old = read(paths["predecessor"])
    new = read(paths["successor"])
    require(old.count(INTRO) == 0 and new.count(INTRO) == 1,
            "additive identity comment mismatch")
    require(old.count(OLD_ASSIGN) == 1 and new.count(NEW_ASSIGN) == 1,
            "protocol assignment blocks are not unique")
    normalized = new.replace(INTRO, "", 1).replace(NEW_ASSIGN, OLD_ASSIGN, 1)
    require(normalized == old,
            "normalized executable/text identity has another delta")
    require("assign protocol_error = fault_q || illegal_request;" not in new and
            new.count("assign protocol_error = fault_q;") == 1,
            "public fault output is not registered-only")
    require(len(re.findall(r"^module\s+" + re.escape(MODULE) + r"\b",
                           old, flags=re.MULTILINE)) == 1 and
            len(re.findall(r"^module\s+" + re.escape(MODULE) + r"\b",
                           new, flags=re.MULTILINE)) == 1,
            "legacy module identity changed")

    preserved = [
        "assign illegal_request = (header_valid",
        "|| (raw_valid && !raw_packet_legal);",
        "if (illegal_request) fault_q <= 1;",
        "if (rst_core) begin\n            fault_q <= 0;",
        "assign header_ready = !fault_q && !token_active_q && header_legal;",
        "assign header_accept = header_valid && header_ready;",
        "assign raw_ready = !fault_q && raw_packet_legal",
        "assign raw_accept = raw_valid && raw_ready;"
    ]
    for token in preserved:
        require(old.count(token) == 1 and new.count(token) == 1,
                "ready/legal/fault token changed: " + token)

    m214_wrapper = read(paths["m214_wrapper"])
    m216 = read(paths["m216_frontend"])
    m519 = read(paths["m519_service_top"])
    require("assign protocol_error = local_fault_q\n"
            "        || (header_valid && !header_shape_legal)\n"
            "        || m202_protocol_error || m204_protocol_error;" in m214_wrapper,
            "M214 wrapper error OR chain changed")
    require("assign protocol_error = local_fault_q\n"
            "        || (header_valid && !header_shape_legal)\n"
            "        || m202_protocol_error || m204_protocol_error;" in m216,
            "M216 error OR chain changed")
    require("assign protocol_error = adapter_fault_q\n"
            "        || (header_valid && !integration_header_legal)\n"
            "        || fe_protocol_error || svc_protocol_error;" in m519,
            "service top error OR chain changed")
    require("assign numeric_overflow = svc_numeric_overflow;" in m519 and
            "assign stale_response_seen = svc_stale_response_seen;" in m519,
            "service numeric/stale error observability changed")
    for token in ["m216_fc2_raw4_to_source_cap_frontend",
                  "m204_protocol_error", "svc_protocol_error",
                  "numeric_overflow", "stale_response_seen"]:
        require(token not in new,
                "successor illegally absorbs/masks outer error source: " + token)

    filelist_root = os.path.join(hw, "dc_handoff", "filelists")
    predecessor_filelists = 0
    successor_filelists = 0
    both_filelists = []
    old_rel = "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
    new_rel = "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
    for name in os.listdir(filelist_root):
        if not name.endswith(".f"):
            continue
        text = read(os.path.join(filelist_root, name))
        has_old = old_rel in text
        has_new = new_rel in text
        predecessor_filelists += int(has_old)
        successor_filelists += int(has_new)
        if has_old and has_new:
            both_filelists.append(name)
    require(successor_filelists == 0 and not both_filelists,
            "an unreviewed filelist already selects M1609")

    return {
        "schema": "m1611_m1609_c2_registered_fault_successor_independent_audit_r1_v1",
        "status": "PASS_SOURCE_ONLY_NORMALIZED_SINGLE_DELTA__GO_AUTHOR_NEW_EXCLUSIVE_FILELIST_AND_VCS_DC_SOURCES",
        "date": "2026-09-01",
        "identity": dict((key + "_sha256", value) for key, value in PINS.items()),
        "normalized_diff": {
            "only_executable_delta": "protocol_error: fault_q || illegal_request -> fault_q",
            "normalized_successor_equals_frozen_predecessor": True,
            "frozen_m214_unchanged": True,
            "same_legacy_module_name": MODULE,
            "same_port_and_parameter_contract": True
        },
        "preserved_local_semantics": {
            "illegal_request_expression": True,
            "illegal_header_and_raw_ready_block": True,
            "header_and_raw_accept_require_ready": True,
            "illegal_request_latches_sticky_fault_q": True,
            "reset_clears_fault_q": True,
            "public_protocol_error_is_fault_q_only": True,
            "public_illegal_fault_visibility_latency": "after the sampling rising edge, not current-cycle combinational"
        },
        "outer_error_audit": {
            "m214_wrapper_keeps_local_m202_m204_or": True,
            "m216_keeps_local_m202_m204_or": True,
            "service_top_keeps_adapter_frontend_service_or": True,
            "numeric_overflow_passthrough": True,
            "stale_response_seen_passthrough": True,
            "successor_contains_no_outer_error_source": True,
            "interpretation": "Replacing only the compactor source cannot mask outer errors; a future VCS top must still exercise each OR-chain source."
        },
        "filelist_audit": {
            "existing_filelists_selecting_predecessor": predecessor_filelists,
            "existing_filelists_selecting_successor": successor_filelists,
            "existing_filelists_selecting_both": both_filelists,
            "new_filelist_required": True,
            "selection_rule": "exactly one of predecessor or successor; same module name forbids both"
        },
        "authorization": {
            "author_new_filelist_and_runner_sources": True,
            "run_vcs_now": False,
            "run_dc_now": False,
            "run_ptpx_now": False,
            "next_gate": "independent exact-SHA filelist/runner review, then directed VCS; DC only after VCS PASS"
        },
        "claim_boundary": {
            "source_only": True,
            "static_test_only": True,
            "vcs": False,
            "dc": False,
            "ptpx": False,
            "timing": False,
            "area": False,
            "power": False,
            "performance": False,
            "rtl_behavior_proven": False,
            "frozen_m214_modified": False,
            "docs359_modified": False
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-frozen")
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    value = build(project_root)
    if args.check_frozen:
        require(strict_json(args.check_frozen) == value,
                "frozen audit differs from recomputation")
        print("PASS_M1611_FROZEN_AUDIT_MATCH")
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
