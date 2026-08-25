#!/usr/bin/env python3
"""Fail-closed source audit for the M37-r9 static-index candidate."""

import argparse
import hashlib
import importlib.util
import pathlib
import re
import sys


FROZEN_R9_RTL_SHA256 = (
    "a5f42567fc5262a99152ef04699c9062cbedc70075c0a91397ce8d00dc4397ed"
)


def load_base_auditor(path):
    spec = importlib.util.spec_from_file_location("m37_r8_source_auditor", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_base(base):
    parameter = base.PARAMETER
    static = base.ELAB_INDEX
    power2 = base.POWER2_INDEX
    base.EXPECTED_STARS = tuple(base.Star(*entry) for entry in (
        (21, 36, parameter), (22, 40, parameter),
        (23, 41, parameter), (23, 47, parameter),
        (32, 32, parameter), (33, 32, parameter),
        (34, 32, parameter), (35, 32, parameter),
        (35, 38, parameter), (36, 21, parameter),
        (46, 33, parameter), (186, 51, static),
        (191, 51, static), (191, 64, static),
        (196, 58, static), (200, 51, static),
        (200, 64, static), (205, 51, static),
        (205, 64, static), (209, 43, static),
        (209, 56, static), (212, 58, static),
        (214, 47, static), (214, 60, static),
        (220, 41, static), (221, 56, static),
        (326, 49, static), (329, 49, static),
        (332, 51, static), (332, 64, static),
        (337, 45, static), (348, 60, parameter),
        (350, 51, power2), (362, 49, static),
        (363, 59, static), (365, 50, static),
        (366, 60, static), (368, 50, static),
        (369, 60, static), (404, 50, parameter),
        (425, 73, static), (439, 60, static),
        (486, 50, static),
    ))
    base.SHIFT_ADD_RE = re.compile(
        r"selected_coefficient\s*=\s*\(\s*selected_row\s*<<\s*1\s*\)"
        r"\s*\+\s*selected_row\s*\+\s*rank_index\s*;"
    )


def require_exact_static_selects(cleaned):
    checks = {
        "bounded_bias_row_select": r"if\s*\(\s*selected_row\s*==\s*row_index\s*\)",
        "bounded_runtime_phase_select": r"if\s*\(\s*phase_cycle_q\s*==\s*phase_index\s*\)",
        "bounded_coefficient_select": r"if\s*\(\s*selected_coefficient\s*==\s*coefficient_index\s*\)",
    }
    for name, pattern in checks.items():
        count = len(re.findall(pattern, cleaned))
        if count != 1:
            raise RuntimeError("{} count is {}, not 1".format(name, count))
    forbidden = {
        "dynamic_bias_array_index": r"bias_q\s*\[\s*selected_row\s*\]",
        "dynamic_term_valid_array_index": r"term_valid_q\s*\[\s*selected_coefficient\s*\]",
        "dynamic_term_negative_array_index": r"term_negative_q\s*\[\s*selected_coefficient\s*\]",
        "dynamic_term_shift_array_index": r"term_shift_q\s*\[\s*selected_coefficient\s*\]",
        "dynamic_intermediate_array_index": r"intermediate_bank_q\s*\[[^\]]+\]\s*\[\s*selected_intermediate\s*\]",
    }
    for name, pattern in forbidden.items():
        if re.search(pattern, cleaned):
            raise RuntimeError("forbidden {} is present".format(name))


def audit_text(base, text):
    cleaned, stars = base.audit_text(text)
    require_exact_static_selects(cleaned)
    return cleaned, stars


def assert_rejected(base, name, forged):
    try:
        audit_text(base, forged)
    except (base.AuditFailure, RuntimeError) as error:
        return "counterexample={} result=REJECT detail={}".format(name, error)
    raise RuntimeError("counterexample {} was incorrectly accepted".format(name))


def run_counterexamples(base, source):
    endmodule = source.rfind("endmodule")
    if endmodule < 0:
        raise RuntimeError("endmodule not found")
    hidden_data = source[:endmodule] + "assign forged = a*b;\n" + source[endmodule:]
    hidden_control = source[:endmodule] + "assign forged = selected_row* RANK;\n" + source[endmodule:]
    shift_match = base.SHIFT_ADD_RE.search(base.blank_comments_and_strings(source))
    if shift_match is None:
        raise RuntimeError("cannot construct comment-signature forgery")
    comment_forgery = (
        source[:shift_match.start()]
        + "// selected_coefficient = (selected_row << 1) + selected_row + rank_index;\n"
        + "selected_coefficient = selected_row* RANK + rank_index;"
        + source[shift_match.end():]
    )
    dynamic_bias = source.replace("bias_q[row_index]", "bias_q[selected_row]", 1)
    dynamic_term = source.replace(
        "term_valid_q[\n                                                            coefficient_index]",
        "term_valid_q[selected_coefficient]",
        1,
    )
    static_intermediate = """[(rank_index*LANES)
                                                                + (output_index
                                                                    % LANES)]"""
    dynamic_intermediate = source.replace(
        static_intermediate, "[selected_intermediate]", 1
    )
    return (
        assert_rejected(base, "hidden_data_a_times_b_no_spaces", hidden_data),
        assert_rejected(base, "hidden_control_selected_row_times_space_rank", hidden_control),
        assert_rejected(base, "comment_shift_add_signature_real_multiply", comment_forgery),
        assert_rejected(base, "dynamic_bias_selected_row_oob", dynamic_bias),
        assert_rejected(base, "dynamic_term_selected_coefficient_oob", dynamic_term),
        assert_rejected(base, "dynamic_selected_intermediate_oob", dynamic_intermediate),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("report", type=pathlib.Path)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    observed_sha = hashlib.sha256(source_bytes).hexdigest()
    if observed_sha != FROZEN_R9_RTL_SHA256:
        raise RuntimeError("r9 RTL SHA mismatch: {}".format(observed_sha))
    base_path = pathlib.Path(__file__).with_name("audit_m37_r8_source_intent.py")
    base = load_base_auditor(base_path)
    configure_base(base)
    source = source_bytes.decode("utf-8")
    cleaned, stars = audit_text(base, source)
    counterexamples = run_counterexamples(base, source)
    lines = [
        "status=PASS_M37_R9_STATIC_INDEX_SOURCE_AUDIT",
        "rtl_sha256={}".format(observed_sha),
        "canonical_star_token_count={}".format(len(stars)),
        "data_multiplication_token_count=0",
        "runtime_non_power_of_two_control_multiplication_token_count=0",
        "rank3_shift_add_count=1",
        "bounded_bias_row_select_count=1",
        "bounded_runtime_phase_select_count=1",
        "bounded_coefficient_select_count=1",
        "dynamic_bias_array_index_count=0",
        "dynamic_term_array_index_count=0",
        "dynamic_intermediate_array_index_count=0",
        "padding_used=false",
        "formality_message_filter_used=false",
        "physical_structure_claim=false_requires_fresh_r9_dc_after_review",
        "canonical_star_ledger_begin",
    ]
    cleaned_lines = cleaned.splitlines()
    for entry in base.EXPECTED_STARS:
        lines.append("line={} column={} classification={} source={}".format(
            entry.line, entry.column, entry.classification,
            " ".join(cleaned_lines[entry.line - 1].split()),
        ))
    lines.extend(("canonical_star_ledger_end", "counterexamples_begin"))
    lines.extend(counterexamples)
    lines.append("counterexamples_end")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print("M37_R9_SOURCE_AUDIT=PASS stars={} counterexamples={}".format(len(stars), len(counterexamples)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, UnicodeError) as error:
        print("M37_R9_SOURCE_AUDIT=FAIL detail={}".format(error), file=sys.stderr)
        raise SystemExit(1)
