#!/usr/bin/env python3
import hashlib
import pathlib


REVIEW = pathlib.Path(__file__).resolve().parent
HW = REVIEW.parent.parent
SOURCE = HW / "rtl_m126/m126_block_phased_k4_forwarding_accumulator_island.sv"
FOLD_SOURCE = HW / "rtl_m125/m125_block_phased_k4_row_fold.sv"
OUTPUT = REVIEW / "m126_registered_fault_barrier_delta.sv"
FOLD_OUTPUT = REVIEW / "m125_registered_state_busy_delta.sv"
EXPECTED_SHA = "b75c64cfa0803461bef4690025a723df9e039e8d2eef6a0da918fc3b9c063e01"
EXPECTED_FOLD_SHA = "cc343bd514777a215ef5e00cf64f8bf00cea700a1d066bdccd5a16feedcc3d30"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha256(SOURCE) != EXPECTED_SHA:
    raise SystemExit("M126 production RTL SHA drift")
if sha256(FOLD_SOURCE) != EXPECTED_FOLD_SHA:
    raise SystemExit("M125 production RTL SHA drift")

text = SOURCE.read_text(encoding="utf-8")
replacements = [
    (
        """assign accumulator_start_valid = window_start_valid && !rst_core
                                   && !wrapper_fault_q
                                   && !wrapper_illegal_request
                                   && !fold_protocol_error;""",
        """assign accumulator_start_valid = window_start_valid && !rst_core
                                   && !wrapper_fault_q
                                   && !wrapper_illegal_request;""",
    ),
    (
        """assign fold_weight_fill_valid = weight_fill_valid && !rst_core
                                  && accumulator_window_active
                                  && !wrapper_fault_q
                                  && !wrapper_illegal_request
                                  && !accumulator_protocol_error;""",
        """assign fold_weight_fill_valid = weight_fill_valid && !rst_core
                                  && accumulator_window_active
                                  && !wrapper_fault_q
                                  && !wrapper_illegal_request;""",
    ),
    (
        """assign fold_row_valid = row_valid && !rst_core
                          && accumulator_window_active
                          && !wrapper_fault_q
                          && !wrapper_illegal_request
                          && !accumulator_protocol_error;""",
        """assign fold_row_valid = row_valid && !rst_core
                          && accumulator_window_active
                          && !wrapper_fault_q
                          && !wrapper_illegal_request;""",
    ),
    (
        """assign fold_update_ready = !rst_core && !wrapper_fault_q
                             && !fold_protocol_error
                             && !accumulator_protocol_error
                             && accumulator_update_ready;""",
        """assign fold_update_ready = !rst_core && !wrapper_fault_q
                             && accumulator_update_ready;""",
    ),
    (
        """assign accumulator_end_valid = window_end_valid && !rst_core
                                 && !fold_busy && !wrapper_fault_q
                                 && !wrapper_illegal_request
                                 && !fold_protocol_error;""",
        """assign accumulator_end_valid = window_end_valid && !rst_core
                                 && !fold_busy && !wrapper_fault_q
                                 && !wrapper_illegal_request;""",
    ),
    (
        """else if (wrapper_illegal_request)
            wrapper_fault_q <= 1'b1;""",
        """else if (wrapper_illegal_request || fold_protocol_error
                 || accumulator_protocol_error)
            wrapper_fault_q <= 1'b1;""",
    ),
    (
        """.update_valid(fold_update_valid && !rst_core
                      && !wrapper_fault_q && !fold_protocol_error),""",
        """.update_valid(fold_update_valid && !rst_core
                      && !wrapper_fault_q),""",
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit("delta replacement count is %d, expected 1: %s" %
                         (count, old.splitlines()[0]))
    text = text.replace(old, new)

banner = """// REVIEW-ONLY M126 registered fault-barrier delta.
// Generated from exact-SHA production RTL.  Raw child protocol errors are
// observed by the top-level protocol_error immediately, but sibling gating is
// driven only by wrapper_fault_q on the following cycle.  This removes the
// fold_error <-> accumulator_error combinational cone while preserving the
// admitted no-fault datapath cycle-for-cycle.  Not production RTL.
"""
OUTPUT.write_text(text.replace("`timescale 1ns/1ps\n", "`timescale 1ns/1ps\n" + banner, 1),
                  encoding="utf-8")

fold_text = FOLD_SOURCE.read_text(encoding="utf-8")
fold_old = "assign busy = fill_active_q || row_active_q || update_valid;"
fold_new = "assign busy = fill_active_q || row_active_q;"
if fold_text.count(fold_old) != 1:
    raise SystemExit("M125 busy replacement count is not 1")
fold_banner = """// REVIEW-ONLY M125 registered-state busy delta.
// update_valid implies row_active_q, so removing the redundant update_valid
// term is Boolean-equivalent in synthesizable two-state logic and keeps busy
// off the protocol_error -> update_valid combinational cone.  Not production.
"""
fold_text = fold_text.replace(fold_old, fold_new)
FOLD_OUTPUT.write_text(
    fold_text.replace("`timescale 1ns/1ps\n",
                      "`timescale 1ns/1ps\n" + fold_banner, 1),
    encoding="utf-8")
print("PASS generated M126 registered fault-barrier delta sha256=" + sha256(OUTPUT))
print("PASS generated M125 registered-state busy delta sha256=" + sha256(FOLD_OUTPUT))
