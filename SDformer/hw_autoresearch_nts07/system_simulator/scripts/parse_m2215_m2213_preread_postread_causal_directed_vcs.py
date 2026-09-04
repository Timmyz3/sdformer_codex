#!/opt/anaconda3/bin/python3.12
"""Fail-closed parser for the future M2215 causal-ablation VCS result."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PASS_RE = re.compile(
    r"^RAW_PASS_M2215_M2213_PREREAD_POSTREAD_CAUSAL_DIRECTED "
    r"ordinary_reads=(\d+) postread_reads=(\d+) preread_reads=(\d+) "
    r"suppressed_reads=(\d+) ordinary_cycles=(\d+) postread_cycles=(\d+) "
    r"preread_cycles=(\d+)$",
    re.MULTILINE,
)
COVER_RE = re.compile(
    r"^M2213_COVER rows=(\d+) hits_post=(\d+) hits_pre=(\d+) "
    r"real_postread_rows=(\d+) postread_bundle_req=(\d+) "
    r"postread_bundle_rsp=(\d+) postread_bank_req=(\d+) "
    r"postread_bank_rsp=(\d+) identity_rsp=(\d+) commits_each=(\d+) "
    r"products_each=(\d+) golden_mismatches=(\d+)$",
    re.MULTILINE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-log", required=True, type=Path)
    ap.add_argument("--sim-log", required=True, type=Path)
    ap.add_argument("--sim-rc", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    compile_log = args.compile_log.read_text(errors="replace")
    sim_log = args.sim_log.read_text(errors="replace")
    rc_text = args.sim_rc.read_text().strip()
    assert rc_text == "0", f"sim rc {rc_text!r}"
    assert "Chronologic VCS simulator copyright" in compile_log
    assert "Error-" not in compile_log and "Syntax error" not in compile_log
    assert "$fatal" not in sim_log and "Error:" not in sim_log
    assert "assertion failed" not in sim_log.lower()
    passes = PASS_RE.findall(sim_log)
    covers = COVER_RE.findall(sim_log)
    assert len(passes) == 1, f"pass lines={len(passes)}"
    assert len(covers) == 1, f"cover lines={len(covers)}"
    ordinary, postread, preread, suppressed, cyc_o, cyc_l, cyc_p = map(
        int, passes[0])
    (rows, hits_l, hits_p, post_rows, bundle_req, bundle_rsp, bank_req,
     bank_rsp, identity_rsp, commits, products, mismatches) = map(
        int, covers[0])
    assert (rows, hits_l, hits_p, post_rows) == (24, 18, 18, 18)
    assert (ordinary, postread, preread, suppressed) == (2304, 2304, 576, 1728)
    assert (bundle_req, bundle_rsp, bank_req, bank_rsp, identity_rsp) == (
        216, 216, 1728, 1728, 216)
    assert commits == 24 and products == 4608 and mismatches == 0
    assert suppressed == postread - preread == bank_req
    assert ordinary == postread and preread < postread
    assert min(cyc_o, cyc_l, cyc_p) > 0
    receipt = {
        "schema": "m2215_m2213_preread_postread_causal_directed_vcs_raw_receipt_r1_v1",
        "status": "RAW_PASS_M2215_M2213_PENDING_M2216_INDEPENDENT_RESULT_HAMMER",
        "axes": {
            "ordinary": "frozen M2018 token-major LRU4",
            "postread": "M2213 group-major LRU4 with real full-row read before late admission",
            "preread": "frozen M2018 group-major LRU4 with hit admission before ST_FETCH_REQ",
        },
        "ledger": {
            "rows_each": rows,
            "cache_hits_postread": hits_l,
            "cache_hits_preread": hits_p,
            "scalar_bank_reads_ordinary": ordinary,
            "scalar_bank_reads_postread": postread,
            "scalar_bank_reads_preread": preread,
            "scalar_bank_reads_suppressed_by_preread": suppressed,
            "postread_rows": post_rows,
            "postread_bundle_requests": bundle_req,
            "postread_bundle_responses": bundle_rsp,
            "postread_bank_requests": bank_req,
            "postread_bank_responses": bank_rsp,
            "postread_identity_accepts": identity_rsp,
            "commits_each": commits,
            "signed_products_each": products,
            "golden_mismatches_all_axes": mismatches,
            "cycles_ordinary": cyc_o,
            "cycles_postread": cyc_l,
            "cycles_preread": cyc_p,
        },
        "claim_boundary": {
            "raw_vcs_pending_independent_review": True,
            "rtl_function_verified": False,
            "component_speedup": False,
            "system_speedup": False,
            "area": False,
            "timing": False,
            "power": False,
            "energy": False,
            "paper_citable": False,
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt["status"])


if __name__ == "__main__":
    main()
