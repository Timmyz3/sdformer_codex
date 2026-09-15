"""Read-only audit of completed cert_transport records; never runs RTL.

Per group: 1 command + 15 requests + 15 response accepts + 1 START +
N PLANE + 1 retirement = 32 + (1+N), before the three recorded stall types.
The coefficient/PN preload and upstream Y/threshold production are excluded.
"""
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "cert_transport" / "results"


def main():
    data = json.loads((SOURCE / "summary.json").read_text())
    assert data["bus_bits"] == 64 and data["max_outstanding"] == 1
    assert data["Y_reads_per_group"] == 5 and data["tau_reads_per_group"] == 10
    rows = data["rows"]
    assert len(rows) == 64
    bykey = {(r["trace"], r["population"], r["backpressure"], r["mode"]): r for r in rows}
    assert len(bykey) == 64
    counters_checked = 0
    csv_rows_checked = 0
    for r in rows:
        n = r["groups"]
        assert n == (20000 if r["population"] == "capture20000" else 35)
        assert r["source_reads"] == 5 * n and r["tau_reads"] == 10 * n
        assert r["bus_bytes"] == 8 * (r["source_reads"] + r["tau_reads"]) == 120 * n
        assert r["cycles"] == r["header_planes"] + 32 * n + sum(
            r[k] for k in ("request_stall", "response_wait", "output_stall"))
        assert all(r[k] == 0 for k in ("dec_mismatch", "plane_mismatch", "exponent_mismatch"))
        if r["backpressure"] == 0:
            assert all(r[k] == 0 for k in ("request_stall", "response_wait", "output_stall"))
        if r["mode"] == "fx_full":
            assert r["header_planes"] == 24 * n, "stale 25-beat FX receipt"
            if r["backpressure"] == 0:
                assert r["cycles"] == 56 * n
        key = (r["trace"], r["population"], r["backpressure"])
        for name in ("fx_full", "bf_full"):
            assert math.isclose(r["total_ratio_vs_" + name],
                                r["cycles"] / bykey[key + (name,)]["cycles"], rel_tol=1e-12)
        counters_checked += 1
    for trace, pop, bp in sorted({k[:3] for k in bykey}):
        path = SOURCE / f"{trace}_{pop}_bp{bp}.csv"
        originals = list(csv.DictReader(path.open()))
        assert len(originals) == 4
        for record in originals:
            summary = bykey[(trace, pop, bp, record["mode"])]
            for k, v in record.items():
                assert summary[k] == (v if k == "mode" else int(v))
            csv_rows_checked += 1
    comparisons = []
    for r in rows:
        if r["population"] == "capture20000" and r["mode"] == "bf_cert":
            key = (r["trace"], r["population"], r["backpressure"])
            comparisons.append({
                "trace": r["trace"], "backpressure": r["backpressure"],
                "fx_full_cycles_per_group": bykey[key + ("fx_full",)]["cycles_per_group"],
                "bf_full_cycles_per_group": bykey[key + ("bf_full",)]["cycles_per_group"],
                "bf_cert_cycles_per_group": r["cycles_per_group"],
                "saved_vs_bf_full_percent": 100 * (1 - r["total_ratio_vs_bf_full"]),
                "saved_vs_fx_full_percent": 100 * (1 - r["total_ratio_vs_fx_full"]),
            })
    result = {
        "pass": True, "summary_rows_checked": counters_checked,
        "csv_rows_checked": csv_rows_checked,
        "capture_decisions": sum(r["groups"] * 10 for r in rows if r["population"] == "capture20000"),
        "boundary_decisions": sum(r["groups"] * 10 for r in rows if r["population"] == "boundary35"),
        "identity": "cycles=header_planes+32*groups+request_stall+response_wait+output_stall",
        "FX_full_header_plus_planes": 24,
        "per_group_source_bytes": 40, "per_group_threshold_bytes": 80,
        "source_byte_saving_from_certificate": 0,
        "scope": "independent accounting and saved-record audit; no RTL rerun and no BN identity proof",
        "comparisons": comparisons,
    }
    (ROOT / "cert_transport_billing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
