#!/usr/bin/env python3
"""Independent Python recount of Q==0 / identical-K vs RTL GROUP sums."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_memh(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]


def unpack_k(packed: int) -> list[int]:
    return [(packed >> (i * 32)) & ((1 << 32) - 1) for i in range(5)]


def classify(vector_dir: Path, groups: int, sources: int = 450) -> dict[str, int]:
    q = read_memh(vector_dir / "input_q.memh")
    k = read_memh(vector_dir / "input_candidate_k.memh")
    valid = read_memh(vector_dir / "input_valid.memh")
    n = groups * sources
    if len(q) < n:
        raise ValueError(f"{vector_dir} short q {len(q)} < {n}")
    q0 = ident = 0
    for i in range(n):
        if q[i] == 0:
            q0 += 1
            continue
        present = [unpack_k(k[i])[c] for c in range(5) if (valid[i] >> c) & 1]
        if present and len(set(present)) == 1:
            ident += 1
    return {"dest": n, "q0": q0, "identk": ident}


GROUP_RE = re.compile(r"qsilent_rows=(?P<q>\d+) identk_rows=(?P<i>\d+)")


def rtl_sums(log: Path) -> dict[str, int]:
    q = i = 0
    for match in GROUP_RE.finditer(log.read_text()):
        q += int(match.group("q"))
        i += int(match.group("i"))
    return {"q0": q, "identk": i}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--rtl-log", type=Path, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    py = classify(args.vector_dir, args.groups)
    rtl = rtl_sums(args.rtl_log)
    report = {
        "schema": "local5_identk_python_rtl_recount_v1",
        "python": py,
        "rtl": rtl,
        "q0_match": py["q0"] == rtl["q0"],
        "identk_match": py["identk"] == rtl["identk"],
        "status": "PASS" if py["q0"] == rtl["q0"] and py["identk"] == rtl["identk"] else "FAIL",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(
        f"# ident-K recount\n\n"
        f"- python Q==0 {py['q0']} identK {py['identk']}\n"
        f"- rtl Q==0 {rtl['q0']} identK {rtl['identk']}\n"
        f"- **{report['status']}**\n"
    )
    print(f"{report['status']} q0 {py['q0']}/{rtl['q0']} ident {py['identk']}/{rtl['identk']}")
    if report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
