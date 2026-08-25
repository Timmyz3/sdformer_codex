#!/usr/bin/env python3
"""Fail-closed source-intent audit for the frozen M37-r8 RTL.

This is deliberately a lexical/contract audit, not a claim about the mapped
implementation.  Comments and strings are blanked while preserving source
coordinates, every remaining ``*`` token must match the frozen canonical
ledger, and the selected-row rank-3 index must be the exact shift-add form.
Fresh DC reports and the mapped netlist are audited separately.
"""

import argparse
import collections
import hashlib
import pathlib
import re
import sys


FROZEN_RTL_SHA256 = (
    "ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd"
)


Star = collections.namedtuple("Star", ("line", "column", "classification"))


PARAMETER = "PARAMETER_OR_PORT_WIDTH_CONSTANT"
ELAB_INDEX = "STATIC_FOR_LOOP_INDEX_TIMES_PARAMETER_CONSTANT"
POWER2_INDEX = "RUNTIME_CONTROL_INDEX_TIMES_POWER_OF_TWO"


# One entry per multiplication token after comments and strings are removed.
# Columns are one-based.  The frozen source SHA and exact location ledger make
# insertion, whitespace evasion, and replacement fail closed.
EXPECTED_STARS = (
    Star(21, 36, PARAMETER),
    Star(22, 40, PARAMETER),
    Star(23, 41, PARAMETER),
    Star(23, 47, PARAMETER),
    Star(32, 32, PARAMETER),
    Star(33, 32, PARAMETER),
    Star(34, 32, PARAMETER),
    Star(35, 32, PARAMETER),
    Star(35, 38, PARAMETER),
    Star(36, 21, PARAMETER),
    Star(46, 33, PARAMETER),
    Star(186, 51, ELAB_INDEX),
    Star(191, 51, ELAB_INDEX),
    Star(191, 64, ELAB_INDEX),
    Star(196, 58, ELAB_INDEX),
    Star(200, 51, ELAB_INDEX),
    Star(200, 64, ELAB_INDEX),
    Star(205, 51, ELAB_INDEX),
    Star(205, 64, ELAB_INDEX),
    Star(209, 43, ELAB_INDEX),
    Star(209, 56, ELAB_INDEX),
    Star(212, 58, ELAB_INDEX),
    Star(214, 47, ELAB_INDEX),
    Star(214, 60, ELAB_INDEX),
    Star(220, 41, ELAB_INDEX),
    Star(221, 56, ELAB_INDEX),
    Star(328, 49, ELAB_INDEX),
    Star(331, 49, ELAB_INDEX),
    Star(334, 51, ELAB_INDEX),
    Star(334, 64, ELAB_INDEX),
    Star(339, 45, ELAB_INDEX),
    Star(350, 60, PARAMETER),
    Star(352, 51, POWER2_INDEX),
    Star(357, 41, ELAB_INDEX),
    Star(358, 51, ELAB_INDEX),
    Star(360, 42, ELAB_INDEX),
    Star(361, 52, ELAB_INDEX),
    Star(363, 42, ELAB_INDEX),
    Star(364, 52, ELAB_INDEX),
    Star(391, 64, PARAMETER),
    Star(393, 54, POWER2_INDEX),
    Star(403, 46, ELAB_INDEX),
    Star(414, 52, ELAB_INDEX),
    Star(459, 50, ELAB_INDEX),
)


SHIFT_ADD_RE = re.compile(
    r"selected_coefficient\s*=\s*\(\s*selected_row\s*<<\s*1\s*\)"
    r"\s*\+\s*selected_row\s*\+\s*rank_index\s*;"
)


class AuditFailure(RuntimeError):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def blank_comments_and_strings(text):
    """Replace comment/string bytes with spaces while retaining newlines."""

    output = []
    index = 0
    state = "normal"
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "normal":
            if char == "/" and following == "/":
                output.extend((" ", " "))
                index += 2
                state = "line_comment"
                continue
            if char == "/" and following == "*":
                output.extend((" ", " "))
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                output.append(" ")
                index += 1
                state = "string"
                continue
            output.append(char)
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                output.append("\n")
                state = "normal"
            else:
                output.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and following == "/":
                output.extend((" ", " "))
                index += 2
                state = "normal"
                continue
            output.append("\n" if char == "\n" else " ")
            index += 1
            continue
        if state == "string":
            if char == "\\" and following:
                output.append(" ")
                output.append("\n" if following == "\n" else " ")
                index += 2
                continue
            if char == '"':
                output.append(" ")
                state = "normal"
            else:
                output.append("\n" if char == "\n" else " ")
            index += 1
            continue
        raise AssertionError(state)
    if state in {"block_comment", "string"}:
        raise AuditFailure(f"unterminated SystemVerilog {state}")
    return "".join(output)


def observed_stars(cleaned):
    found = []
    for line_number, line in enumerate(cleaned.splitlines(), 1):
        found.extend(
            (line_number, column)
            for column, char in enumerate(line, 1)
            if char == "*"
        )
    return tuple(found)


def audit_text(text):
    cleaned = blank_comments_and_strings(text)
    observed = observed_stars(cleaned)
    expected = tuple((entry.line, entry.column) for entry in EXPECTED_STARS)
    if observed != expected:
        unexpected = sorted(set(observed) - set(expected))
        missing = sorted(set(expected) - set(observed))
        raise AuditFailure(
            f"star-token ledger mismatch: unexpected={unexpected} missing={missing}"
        )
    matches = tuple(SHIFT_ADD_RE.finditer(cleaned))
    if len(matches) != 1:
        raise AuditFailure(
            f"rank-3 selected-row shift-add statement count is {len(matches)}, not 1"
        )
    selected_statement = matches[0].group(0)
    if "*" in selected_statement:
        raise AuditFailure("rank-3 selected-row statement contains multiplication")
    return cleaned, observed


def assert_rejected(name, forged):
    try:
        audit_text(forged)
    except AuditFailure as error:
        return f"counterexample={name} result=REJECT detail={error}"
    raise AuditFailure(f"counterexample {name} was incorrectly accepted")


def run_counterexamples(source):
    endmodule = source.rfind("endmodule")
    if endmodule < 0:
        raise AuditFailure("endmodule not found for counterexample construction")
    hidden_data = source[:endmodule] + "assign forged = a*b;\n" + source[endmodule:]
    hidden_control = (
        source[:endmodule]
        + "assign forged = selected_row* RANK;\n"
        + source[endmodule:]
    )
    old_statement = """selected_coefficient = (selected_row << 1)
                                + selected_row + rank_index;"""
    forged_statement = """// selected_coefficient = (selected_row << 1)
                            //     + selected_row + rank_index;
                            selected_coefficient = selected_row* RANK
                                + rank_index;"""
    if source.count(old_statement) != 1:
        raise AuditFailure("cannot construct comment-signature counterexample")
    comment_signature = source.replace(old_statement, forged_statement)
    return (
        assert_rejected("hidden_data_a_times_b_no_spaces", hidden_data),
        assert_rejected(
            "hidden_control_selected_row_times_space_rank", hidden_control
        ),
        assert_rejected("comment_shift_add_signature_real_multiply", comment_signature),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("--expected-sha256", default=FROZEN_RTL_SHA256)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha = sha256_bytes(source_bytes)
    if source_sha != args.expected_sha256:
        raise AuditFailure(
            f"frozen RTL SHA mismatch: expected={args.expected_sha256} observed={source_sha}"
        )
    source = source_bytes.decode("utf-8")
    cleaned, observed = audit_text(source)
    counterexamples = run_counterexamples(source)
    lines = [
        "status=PASS_FROZEN_R8_SOURCE_TOKEN_LEDGER",
        f"source_sha256={source_sha}",
        "comments_and_strings_removed_before_tokenization=true",
        f"canonical_star_token_count={len(observed)}",
        "data_multiplication_token_count=0",
        "runtime_non_power_of_two_control_multiplication_token_count=0",
        "rank3_selected_row_shift_add_statement_count=1",
        "dut_uses_integer_multiplier_signal_used_as_structure_proof=false",
        "canonical_star_ledger_begin",
    ]
    cleaned_lines = cleaned.splitlines()
    for entry in EXPECTED_STARS:
        source_line = " ".join(cleaned_lines[entry.line - 1].split())
        lines.append(
            f"line={entry.line} column={entry.column} "
            f"classification={entry.classification} source={source_line}"
        )
    lines.extend(("canonical_star_ledger_end", "counterexamples_begin"))
    lines.extend(counterexamples)
    lines.extend(("counterexamples_end", "physical_structure_claim=false_requires_dc"))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "M37_SOURCE_INTENT_AUDIT=PASS "
        f"stars={len(observed)} counterexamples={len(counterexamples)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditFailure, OSError, UnicodeError) as error:
        print(f"M37_SOURCE_INTENT_AUDIT=FAIL detail={error}", file=sys.stderr)
        raise SystemExit(1)
