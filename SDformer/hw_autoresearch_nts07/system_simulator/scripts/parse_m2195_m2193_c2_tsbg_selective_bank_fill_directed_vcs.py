#!/usr/bin/python3.12
"""Fail-closed parser for the one raw M2195 directed VCS result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def parse(log: Path, compile_log: Path, sim_rc: Path) -> dict:
    need(log.is_file() and not log.is_symlink(), "missing/symlink sim log")
    need(compile_log.is_file() and not compile_log.is_symlink(), "missing/symlink compile log")
    need(sim_rc.read_text().strip() == "0", "sim exit code")
    text = log.read_text(errors="replace")
    build = compile_log.read_text(errors="replace")
    fatal_terms = ("Error-", "Syntax error", "Compiler directive error", "$fatal",
                   "Assertion failed", "Offending", "UVM_FATAL")
    need(not any(term in build for term in fatal_terms), "compile diagnostic")
    need(not any(term in text for term in fatal_terms), "simulation diagnostic")
    cover_re = re.compile(
        r"^M2193_COVER partial_o=(\d+) partial_t=(\d+) eviction_o=(\d+) "
        r"eviction_t=(\d+) reorder_o=(\d+) reorder_t=(\d+) reqstall_o=(\d+) "
        r"reqstall_t=(\d+) bridgestall_o=(\d+) bridgestall_t=(\d+) "
        r"commitstall_o=(\d+) commitstall_t=(\d+) zero_o=(\d+) zero_t=(\d+)$",
        re.MULTILINE)
    pass_re = re.compile(
        r"^PASS_M2193_C2_TSBG_SELECTIVE_BANK_FILL_DIRECTED bundles=(\d+) "
        r"commits_o=(\d+) commits_t=(\d+) partial_o=(\d+) partial_t=(\d+) "
        r"refills_o=(\d+) refills_t=(\d+) scalar_o=(\d+) scalar_t=(\d+) "
        r"products_o=(\d+) products_t=(\d+)$", re.MULTILINE)
    covers = cover_re.findall(text)
    passes = pass_re.findall(text)
    need(len(covers) == 1 and len(passes) == 1, "unique cover/pass token")
    cover = list(map(int, covers[0]))
    values = list(map(int, passes[0]))
    need(all(value > 0 for value in cover), "all cover counters nonzero")
    need(values[0] == 3 and values[1] == values[2] == 72, "bundle/commit ledger")
    need(values[3] > 0 and values[4] > 0, "partial refill missing")
    need(values[5] == values[7] and values[6] == values[8],
         "source_count/popcount refill ledger")
    need(values[9] == values[10] and values[9] > 0, "mode product mismatch")
    return {
        "schema": "m2195_m2193_c2_tsbg_selective_bank_fill_directed_vcs_receipt_r1_v1",
        "status": "RAW_PASS_M2195_M2193_DIRECTED_VCS_PENDING_M2196_RESULT_HAMMER",
        "coverage": {
            "partial_hit_ordinary": cover[0], "partial_hit_tsbg": cover[1],
            "eviction_ordinary": cover[2], "eviction_tsbg": cover[3],
            "response_reorder_ordinary": cover[4], "response_reorder_tsbg": cover[5],
            "request_backpressure_ordinary": cover[6],
            "request_backpressure_tsbg": cover[7],
            "bridge_backpressure_ordinary": cover[8],
            "bridge_backpressure_tsbg": cover[9],
            "commit_backpressure_ordinary": cover[10],
            "commit_backpressure_tsbg": cover[11],
            "zero_descriptor_skip_ordinary": cover[12],
            "zero_descriptor_skip_tsbg": cover[13],
            "positive_and_negative_sources": True,
            "exact_acc24_commits_per_mode": 72,
            "phase0_union_mask_exact": True,
            "phase1_missing_only_refill_mask_exact": True,
        },
        "ledger": {"bundles": values[0], "commits_ordinary": values[1],
                   "commits_tsbg": values[2], "refill_banks_ordinary": values[5],
                   "refill_banks_tsbg": values[6], "scalar_requests_ordinary": values[7],
                   "scalar_requests_tsbg": values[8], "products_each": values[9]},
        "claim_boundary": {"directed_vcs_only": True, "rtl_performance": False,
                           "same_area": False, "timing": False, "energy": False,
                           "power": False, "paper_result": False,
                           "component_speedup_admitted": False,
                           "system_speedup": False, "headline": False},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-log", type=Path, required=True)
    ap.add_argument("--compile-log", type=Path, required=True)
    ap.add_argument("--sim-rc", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    need(not args.output.exists(), "fresh parser output required")
    result = parse(args.sim_log, args.compile_log, args.sim_rc)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2195_PARSE_FAIL_CLOSED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
