#!/usr/bin/env python3
"""Audit fresh M37 DC reports for physical multiplier and link closure."""

import argparse
import pathlib
import re
import sys


MULTIPLIER_RE = re.compile(
    r"(?<![A-Za-z0-9_$])(?:"
    r"DW[0-9]*_*mult(?:_[A-Za-z0-9_$]+)?|"
    r"DW_mult(?:_[A-Za-z0-9_$]+)?|"
    r"GTECH_MULT[A-Za-z0-9_$]*|"
    r"\*?MULT(?:_[A-Za-z0-9_$]+)*_OP(?:_[A-Za-z0-9_$]+)*|"
    r"mult_x_[A-Za-z0-9_$]+|mul_x_[A-Za-z0-9_$]+|"
    r"mult_[0-9][A-Za-z0-9_$]*|mul_[0-9][A-Za-z0-9_$]*"
    r")(?![A-Za-z0-9_$])",
    re.IGNORECASE,
)

UNRESOLVED_RE = re.compile(
    r"Unable to resolve reference|Cannot find design|unresolved reference|"
    r"link failed|unresolved design",
    re.IGNORECASE,
)


class AuditFailure(RuntimeError):
    pass


def multiplier_hits(path):
    hits = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if MULTIPLIER_RE.search(line):
            hits.append((line_number, line))
    return hits


def reference_blackbox_rows(path):
    """Return only table rows whose actual Attributes field contains b.

    The explanatory ``b - black box`` legend occurs before any table header
    and is intentionally not parsed.
    """

    rows = []
    saw_header = False
    in_table = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if re.match(r"^Reference\s+Library\s+Unit Area\s+Count\s+Total Area\s+Attributes\s*$", line):
            saw_header = True
            in_table = False
            continue
        if saw_header and re.match(r"^-{20,}\s*$", line):
            if in_table:
                saw_header = False
                in_table = False
            else:
                in_table = True
            continue
        if not in_table:
            continue
        attribute_match = re.search(r"\s([a-z]+(?:,\s*[a-z]+)*)\s*$", line)
        if not attribute_match:
            continue
        attributes = [item.strip() for item in attribute_match.group(1).split(",")]
        if "b" in attributes:
            rows.append((line_number, line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dc-log", type=pathlib.Path, required=True)
    parser.add_argument("--resources-pre", type=pathlib.Path, required=True)
    parser.add_argument("--resources-post", type=pathlib.Path, required=True)
    parser.add_argument("--references-pre", type=pathlib.Path, required=True)
    parser.add_argument("--references-post", type=pathlib.Path, required=True)
    parser.add_argument("--mapped-netlist", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()

    searched = (
        ("resources_precompile", args.resources_pre),
        ("resources_postcompile", args.resources_post),
        ("references_precompile", args.references_pre),
        ("references_postcompile", args.references_post),
        ("mapped_netlist", args.mapped_netlist),
    )
    all_hits = []
    counts = []
    for label, path in searched:
        if not path.is_file() or path.stat().st_size == 0:
            raise AuditFailure("missing or empty structural evidence: {}".format(path))
        hits = multiplier_hits(path)
        counts.append((label, len(hits)))
        all_hits.extend((label, path, line_number, line) for line_number, line in hits)

    # Precompile synthetic DesignWare operators may intentionally carry a
    # black-box attribute until compile maps them.  Closure is therefore judged
    # only on the postcompile reference table.
    blackboxes = [
        ("references_postcompile", args.references_post, line_number, line)
        for line_number, line in reference_blackbox_rows(args.references_post)
    ]

    unresolved = []
    for line_number, line in enumerate(
        args.dc_log.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if UNRESOLVED_RE.search(line):
            unresolved.append((line_number, line))

    status = "PASS_STRICT_ZERO_PHYSICAL_MULTIPLIER_AND_LINK_AUDIT"
    if all_hits or blackboxes or unresolved:
        status = "FAIL_STRUCTURAL_OR_LINK_AUDIT_DO_NOT_CITE"
    lines = [
        "status={}".format(status),
        "scope=PRECOMPILE_POSTCOMPILE_REFERENCE_AND_MAPPED_NETLIST",
        "multiplier_pattern={}".format(MULTIPLIER_RE.pattern),
        "reference_legend_blackbox_text_ignored=true",
        "precompile_synthetic_blackbox_attribute_not_used_for_link_closure=true",
        "dut_uses_integer_multiplier_signal_used_as_structure_proof=false",
    ]
    for label, count in counts:
        lines.append("{}_physical_multiplier_hit_count={}".format(label, count))
    lines.extend(
        (
            "physical_multiplier_hit_total={}".format(len(all_hits)),
            "reference_table_blackbox_attribute_count={}".format(len(blackboxes)),
            "unresolved_link_signature_count={}".format(len(unresolved)),
            "multiplier_hits_begin",
        )
    )
    lines.extend(
        "{}:{}:{}:{}".format(label, path, line_number, line)
        for label, path, line_number, line in all_hits
    )
    lines.append("multiplier_hits_end")
    lines.append("blackbox_rows_begin")
    lines.extend(
        "{}:{}:{}:{}".format(label, path, line_number, line)
        for label, path, line_number, line in blackboxes
    )
    lines.append("blackbox_rows_end")
    lines.append("unresolved_hits_begin")
    lines.extend(
        "{}:{}:{}".format(args.dc_log, line_number, line)
        for line_number, line in unresolved
    )
    lines.append("unresolved_hits_end")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status.startswith("FAIL"):
        raise AuditFailure(
            "strict-zero audit failed: multiplier={} blackbox={} unresolved={}".format(
                len(all_hits), len(blackboxes), len(unresolved)
            )
        )
    print("M37_DC_STRUCTURE_AUDIT=PASS multiplier=0 blackbox=0 unresolved=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditFailure, OSError) as error:
        print("M37_DC_STRUCTURE_AUDIT=FAIL detail={}".format(error), file=sys.stderr)
        raise SystemExit(1)
