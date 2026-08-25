#!/usr/bin/env python3
"""Create the exact relative-path reanchor ledger consumed by M31-r5."""

import argparse
import os
from pathlib import Path

import build_m31_r5_synopsys_receipt as r5


def seal(work_root, repo_root, runs_root, run_dir, output):
    work, repo, _, run = r5.validate_roots(
        work_root, repo_root, runs_root, run_dir)
    expected_output = repo / (
        "hw_autoresearch_nts07/results/"
        "m31_r5_frozen_evidence_reanchor_20260822/"
        "m31_r5_live_reanchor_relative_exact.sha256")
    output = Path(os.path.abspath(str(output)))
    if output != expected_output:
        raise ValueError("M31-r5 reanchor ledger output path drift")
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite M31-r5 reanchor ledger")
    expected = r5.expected_reanchor_files(work, repo, run)
    rows = []
    for relative in sorted(expected):
        source = r5.canonical_file(
            work / relative, work, "r5 reanchor source", relative)
        rows.append("{}  {}\n".format(r5.sha256(source), relative))
    output.parent.mkdir(parents=True, exist_ok=True)
    r5.canonical_dir(
        output.parent, repo, "r5 reanchor output parent",
        "hw_autoresearch_nts07/results/"
        "m31_r5_frozen_evidence_reanchor_20260822")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(output), flags, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("".join(rows))
    output.chmod(0o444)
    _, parsed = r5.parse_exact_relative_ledger(
        output, work, expected, "r5 relative reanchor ledger")
    if len(parsed) != 10:
        raise ValueError("M31-r5 reanchor entry population drift")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = seal(args.work_root, args.repo_root, args.runs_root,
                  args.run_dir, args.output)
    print("{} {}".format(r5.sha256(output), output))


if __name__ == "__main__":
    main()
