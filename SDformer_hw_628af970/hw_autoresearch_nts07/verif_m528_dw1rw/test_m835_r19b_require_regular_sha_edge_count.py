#!/usr/bin/env python3
"""Continuation-aware source audit for M835/R19b require_regular_sha edges.

This is an additive source-only checker.  It never executes the audited shell
script.  Heredoc payloads are removed, shell lines ending in a backslash are
joined into one logical command, and every top-level require_regular_sha call
is enumerated as a digest/operand edge.  In particular, the frozen docs/359
call spans physical lines 1125-1126 and must count as one of 95 unique edges.
"""

import argparse
import json
import re
from pathlib import Path


HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
EDGE_RE = re.compile(r"^\s*require_regular_sha\s+([0-9a-f]{64})\s+(.+?)\s*$")
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
DOCS359_OPERAND = '"${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"'


def without_heredocs(text):
    kept = []
    terminator = None
    for line_no, line in enumerate(text.splitlines(), 1):
        if terminator is not None:
            if line == terminator or line.lstrip("\t") == terminator:
                terminator = None
            continue
        kept.append((line_no, line))
        match = HEREDOC_RE.search(line)
        if match:
            terminator = match.group(1)
    if terminator is not None:
        raise RuntimeError("unterminated heredoc")
    return kept


def logical_shell_lines(text):
    logical = []
    pieces = []
    start = end = None
    for line_no, line in without_heredocs(text):
        stripped = line.rstrip()
        continued = stripped.endswith("\\")
        piece = stripped[:-1].rstrip() if continued else line
        if not pieces:
            start = line_no
        pieces.append(piece)
        end = line_no
        if not continued:
            logical.append((start, end, " ".join(part.strip() for part in pieces)))
            pieces = []
            start = end = None
    if pieces:
        raise RuntimeError("unterminated shell line continuation")
    return logical


def audit_text(text):
    edges = []
    for start, end, line in logical_shell_lines(text):
        match = EDGE_RE.match(line)
        if not match:
            continue
        digest = match.group(1)
        operand = " ".join(match.group(2).split())
        edges.append({
            "start_line": start,
            "end_line": end,
            "physical_line_count": end - start + 1,
            "sha256": digest,
            "operand": operand,
        })
    identities = [(edge["sha256"], edge["operand"]) for edge in edges]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    multiline = [edge for edge in edges if edge["physical_line_count"] > 1]
    docs = [edge for edge in edges
            if edge["sha256"] == DOCS359_SHA and edge["operand"] == DOCS359_OPERAND]
    return {
        "edges": edges,
        "edge_count": len(edges),
        "unique_edge_count": len(set(identities)),
        "single_line_edge_count": len(edges) - len(multiline),
        "multiline_edge_count": len(multiline),
        "duplicate_edges": [{"sha256": sha, "operand": operand}
                            for sha, operand in duplicates],
        "docs359_edges": docs,
    }


def self_test():
    synthetic = """require_regular_sha {sha_a} \"${{ROOT}}/a\"
require_regular_sha {sha_b} \\
  \"${{ROOT}}/b\"
python3 -I - <<'PY'
require_regular_sha {sha_c} \"must/not/count\"
PY
""".format(sha_a="1" * 64, sha_b="2" * 64, sha_c="3" * 64)
    result = audit_text(synthetic)
    if result["edge_count"] != 2 or result["unique_edge_count"] != 2:
        raise RuntimeError("synthetic continuation count failure")
    if result["single_line_edge_count"] != 1 or result["multiline_edge_count"] != 1:
        raise RuntimeError("synthetic physical-span classification failure")
    second = result["edges"][1]
    if second["start_line"] != 2 or second["end_line"] != 3:
        raise RuntimeError("synthetic continuation line-span failure")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.runner.is_file() or args.runner.is_symlink():
        raise RuntimeError("runner must be a regular non-symlink file")
    self_test_pass = self_test() if args.self_test else None
    result = audit_text(args.runner.read_text(encoding="utf-8"))
    failures = []
    if result["edge_count"] != 95:
        failures.append("edge_count")
    if result["unique_edge_count"] != 95:
        failures.append("unique_edge_count")
    if result["single_line_edge_count"] != 94:
        failures.append("single_line_edge_count")
    if result["multiline_edge_count"] != 1:
        failures.append("multiline_edge_count")
    if result["duplicate_edges"]:
        failures.append("duplicate_edges")
    docs = result["docs359_edges"]
    if len(docs) != 1 or docs[0]["start_line"] != 1125 or docs[0]["end_line"] != 1126:
        failures.append("docs359_continuation_identity")
    output = {
        "schema": "m835_r19b_require_regular_sha_edge_count_v1",
        "status": "PASS_95_UNIQUE_EDGES__94_SINGLE_LINE_PLUS_1_DOCS359_CONTINUATION" if not failures else "FAIL_EDGE_COUNT",
        "runner": str(args.runner),
        "edge_count": result["edge_count"],
        "unique_edge_count": result["unique_edge_count"],
        "single_line_edge_count": result["single_line_edge_count"],
        "multiline_edge_count": result["multiline_edge_count"],
        "duplicate_edge_count": len(result["duplicate_edges"]),
        "docs359_continuation": docs,
        "self_test_pass": self_test_pass,
        "failures": failures,
        "runner_executed": False,
        "vcs_or_license_or_eda_executed": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
