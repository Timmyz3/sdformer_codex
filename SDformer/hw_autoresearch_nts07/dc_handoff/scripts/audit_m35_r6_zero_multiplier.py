#!/usr/bin/env python3
"""Fail-closed structural audit for the exact M35-r6 mapped result."""

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


def multiplier_hits(path):
    return [(number, line) for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
        if MULTIPLIER_RE.search(line)]


def blackbox_rows(path):
    rows = []
    saw_header = False
    in_table = False
    for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
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
        if in_table:
            match = re.search(r"\s([a-z]+(?:,\s*[a-z]+)*)\s*$", line)
            if match and "b" in [item.strip() for item in match.group(1).split(",")]:
                rows.append((number, line))
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
    hits = []
    counts = []
    for label, path in searched:
        require_file(path)
        found = multiplier_hits(path)
        counts.append((label, len(found)))
        hits.extend((label, path, number, line) for number, line in found)
    blackboxes = [("references_postcompile", args.references_post, number, line)
                  for number, line in blackbox_rows(args.references_post)]
    require_file(args.dc_log)
    unresolved = [(number, line) for number, line in enumerate(
        args.dc_log.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
        if UNRESOLVED_RE.search(line)]
    passed = not hits and not blackboxes and not unresolved
    lines = [
        "status={}".format(
            "PASS_STRICT_ZERO_PHYSICAL_MULTIPLIER_AND_LINK_AUDIT" if passed
            else "FAIL_STRUCTURAL_OR_LINK_AUDIT_DO_NOT_CITE"),
        "scope=M35_R6_PRE_POST_REFERENCE_AND_MAPPED_NETLIST",
        "uses_integer_multiplier_output_used_as_structure_proof=false",
    ]
    lines.extend("{}_multiplier_hit_count={}".format(label, count)
                 for label, count in counts)
    lines.extend([
        "physical_multiplier_hit_total={}".format(len(hits)),
        "postcompile_blackbox_attribute_count={}".format(len(blackboxes)),
        "unresolved_link_signature_count={}".format(len(unresolved)),
        "hits_begin",
    ])
    lines.extend("{}:{}:{}:{}".format(*row) for row in hits)
    lines.append("blackboxes_begin")
    lines.extend("{}:{}:{}:{}".format(*row) for row in blackboxes)
    lines.append("unresolved_begin")
    lines.extend("{}:{}:{}".format(args.dc_log, number, line)
                 for number, line in unresolved)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError("M35-r6 structural audit failed")
    print("M35_R6_ZERO_MULTIPLIER_AUDIT=PASS")


if __name__ == "__main__":
    main()
