#!/usr/bin/env python3
"""Fail-closed M43-r3 multiplier, black-box, and link audit."""

from __future__ import print_function

import argparse
import pathlib
import re


MULTIPLIER_RE = re.compile(
    r"(?<![A-Za-z0-9_$])(?:DW[0-9]*_*mult(?:_[A-Za-z0-9_$]+)?|"
    r"DW_mult(?:_[A-Za-z0-9_$]+)?|GTECH_MULT[A-Za-z0-9_$]*|"
    r"\*?MULT(?:_[A-Za-z0-9_$]+)*_OP(?:_[A-Za-z0-9_$]+)*|"
    r"mult_x_[A-Za-z0-9_$]+|mul_x_[A-Za-z0-9_$]+|"
    r"mult_[0-9][A-Za-z0-9_$]*|mul_[0-9][A-Za-z0-9_$]*"
    r")(?![A-Za-z0-9_$])", re.IGNORECASE)
UNRESOLVED_RE = re.compile(
    r"Unable to resolve reference|Cannot find design|unresolved reference|"
    r"link failed|unresolved design", re.IGNORECASE)


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("missing or empty evidence: {}".format(path))


def lines(path):
    require_file(path)
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def multiplier_hits(path):
    return [(number, line) for number, line in enumerate(lines(path), 1)
            if MULTIPLIER_RE.search(line)]


def reference_blackbox_rows(path):
    rows = []
    saw_header = False
    in_table = False
    for number, line in enumerate(lines(path), 1):
        if re.match(
                r"^Reference\s+Library\s+Unit Area\s+Count\s+Total Area\s+Attributes\s*$",
                line):
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
        if in_table:
            match = re.search(r"\s([a-z]+(?:,\s*[a-z]+)*)\s*$", line)
            if match and "b" in [item.strip() for item in match.group(1).split(",")]:
                rows.append((number, line))
    return rows


def one_int(pattern, source, label):
    found = re.findall(pattern, source, re.MULTILINE)
    if len(found) != 1:
        raise ValueError("ambiguous {}: {} matches".format(label, len(found)))
    return int(found[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dc-log", type=pathlib.Path, required=True)
    parser.add_argument("--resources-pre", type=pathlib.Path, required=True)
    parser.add_argument("--resources-post", type=pathlib.Path, required=True)
    parser.add_argument("--references-pre", type=pathlib.Path, required=True)
    parser.add_argument("--references-post", type=pathlib.Path, required=True)
    parser.add_argument("--mapped-netlist", type=pathlib.Path, required=True)
    parser.add_argument("--area", type=pathlib.Path, required=True)
    parser.add_argument("--check-design", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()

    searched = (
        ("resources_precompile", args.resources_pre),
        ("resources_postcompile", args.resources_post),
        ("references_precompile", args.references_pre),
        ("references_postcompile", args.references_post),
        ("mapped_netlist", args.mapped_netlist),
    )
    hits = []
    counts = []
    for label, path in searched:
        found = multiplier_hits(path)
        counts.append((label, len(found)))
        hits.extend((label, path, number, line) for number, line in found)

    blackboxes = [
        ("references_postcompile", args.references_post, number, line)
        for number, line in reference_blackbox_rows(args.references_post)
    ]
    area_text = "\n".join(lines(args.area)) + "\n"
    macro_count = one_int(
        r"^Number of macros/black boxes:\s+([0-9]+)\s*$", area_text,
        "area macro/black-box count")
    dc_lines = lines(args.dc_log)
    check_lines = lines(args.check_design)
    unresolved = []
    for label, path, source_lines in (
            ("dc_log", args.dc_log, dc_lines),
            ("check_design_postcompile", args.check_design, check_lines)):
        unresolved.extend((label, path, number, line)
                          for number, line in enumerate(source_lines, 1)
                          if UNRESOLVED_RE.search(line))

    passed = not hits and not blackboxes and not unresolved and macro_count == 0
    output = [
        "status={}".format(
            "PASS_STRICT_ZERO_PHYSICAL_MULTIPLIER_BLACKBOX_AND_LINK_AUDIT"
            if passed else "FAIL_STRUCTURAL_OR_LINK_AUDIT_DO_NOT_CITE"),
        "scope=M43_R3_PRE_POST_REFERENCE_RESOURCE_MAPPED_NETLIST_AND_AREA",
        "uses_source_operator_absence_as_structure_proof=false",
    ]
    output.extend("{}_multiplier_hit_count={}".format(label, count)
                  for label, count in counts)
    output.extend([
        "physical_multiplier_hit_total={}".format(len(hits)),
        "postcompile_reference_blackbox_attribute_count={}".format(len(blackboxes)),
        "area_macro_or_blackbox_cell_count={}".format(macro_count),
        "unresolved_link_signature_count={}".format(len(unresolved)),
        "multiplier_hits_begin",
    ])
    output.extend("{}:{}:{}:{}".format(*row) for row in hits)
    output.append("blackboxes_begin")
    output.extend("{}:{}:{}:{}".format(*row) for row in blackboxes)
    output.append("unresolved_begin")
    output.extend("{}:{}:{}:{}".format(*row) for row in unresolved)
    args.report.write_text("\n".join(output) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError("M43-r3 strict structural/link audit failed")
    print("M43_R3_STRUCTURAL_AUDIT=PASS")


if __name__ == "__main__":
    main()
