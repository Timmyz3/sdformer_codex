#!/usr/bin/env python3
"""Read-only integer recompute for the M722-r2 author handoff."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m722r2_lb_fuse_decoder_cpu_fastkill_r1_20260828"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise RuntimeError("duplicate key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError(token)))


def verify_seal(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    errors = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if sha256(root / name) != expected:
            errors.append(name)
    expected, name = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    if name != "SHA256SUMS" or expected != sha256(manifest):
        errors.append("outer")
    return not errors


def main():
    report = strict_json(RESULT / "report.json")
    rows = [json.loads(line) for line in
            (RESULT / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    headline = [row for row in rows if row["module_index"] in (0, 2, 3)]
    sequence_count = defaultdict(int)
    for row in headline:
        sequence_count[row["sequence"]] += 1
    a1 = sum(row["a1_cycles"]["total"] for row in headline)
    lb = sum(row["lb_cycles"]["total"] for row in headline)
    osg = sum(row["a1_osg_groups"] for row in headline)
    direct = sum(row["lb_direct_groups"] for row in headline)
    a1_rmw = sum(row["traffic"]["a1_onchip_psum_rmw_bytes"]
                 for row in headline)
    lb_rmw = sum(row["traffic"]["lb_onchip_psum_rmw_bytes"]
                 for row in headline)
    a1_commit = sum(row["traffic"]["dense_commit_bytes_a1"]
                    for row in headline)
    lb_commit = sum(row["traffic"]["dense_commit_bytes_lb"]
                    for row in headline)
    checks = {
        "result_double_seal": verify_seal(RESULT),
        "rows_1200": len(rows) == 1200,
        "headline_rows_900": len(headline) == 900,
        "three_sequences_300_headline_planes_each":
            sorted(sequence_count.values()) == [300, 300, 300],
        "a1_cycle_sum": a1 == 21590945350,
        "lb_cycle_sum": lb == 23377337337,
        "osg_group_sum": osg == 827946728,
        "direct_group_sum": direct == 1170190821,
        "a1_rmw_sum": a1_rmw == 476897315328,
        "lb_rmw_sum": lb_rmw == 549335071872,
        "commit_equal": a1_commit == lb_commit == 11612160000,
        "all_rows_zero_offchip_spill": all(
            row["traffic"]["a1_offchip_psum_spill_bytes"] == 0 and
            row["traffic"]["lb_offchip_psum_spill_bytes"] == 0
            for row in rows),
        "all_rows_zero_port_conflicts": all(
            row["port_model"]["lb_port_conflict_events"] == 0
            for row in rows),
        "numeric_mismatch_zero":
            report["numeric_exactness"]["a1_lb_acc24_mismatches"] == 0,
        "d3_acc16_order_independent_safe":
            report["numeric_exactness"]["dynamic_ranges"]["D3"]
            ["trace_all_orders_fit_acc16"] is True,
        "d0_acc16_not_order_independent_safe":
            report["numeric_exactness"]["dynamic_ranges"]["D0"]
            ["trace_all_orders_fit_acc16"] is False,
        "fair_a1_zero_spill":
            report["decision"]["fair_a1_zero_offchip_psum_spill"] is True,
        "kill_no_rtl": report["status"] ==
            "KILL_NO_RTL__FAIR_A1_ZERO_PSUM_SPILL",
        "no_rtl_authorization": report["decision"]["rtl_authorized_now"] is False,
        "docs359_frozen": sha256(DOCS359) ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    }
    payload = {
        "schema": "m722r2_lb_fuse_decoder_author_recompute_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed": {
            "headline_a1_cycles": a1,
            "headline_lb_cycles": lb,
            "headline_a1_over_lb": format(a1 / lb, ".12f"),
            "headline_a1_osg_groups": osg,
            "headline_lb_direct_groups": direct,
            "headline_lb_over_osg_groups": format(direct / osg, ".12f"),
            "headline_a1_onchip_psum_rmw_bytes": a1_rmw,
            "headline_lb_onchip_psum_rmw_bytes": lb_rmw,
            "headline_commit_bytes_each": a1_commit,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
