"""Local Verilator experiment; original Claude captures remain read-only."""
import csv
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parents[1] / "claude_fusion_trials_20260914/results/t5_rtl"
TRACES = ("s0_stage0", "s0_stage3", "s10_stage0", "s10_stage3")


def main():
    subprocess.run(["verilator", "--cc", "--exe", "-Wno-fatal", "--top-module",
                    "cert_transport", "cert_transport.sv", "tb.cpp"], cwd=ROOT, check=True)
    subprocess.run(["make", "-C", "obj_dir", "-f", "Vcert_transport.mk", "-j", "4"],
                   cwd=ROOT, check=True)
    dest = ROOT / "results"
    dest.mkdir(exist_ok=True)
    records = []
    for trace in TRACES:
        folder = INPUT / trace
        for bp in (0, 1):
            for directed in (0, 1):
                pop = "boundary35" if directed else "capture20000"
                output = dest / f"{trace}_{pop}_bp{bp}.csv"
                cmd = [str(ROOT / "obj_dir/Vcert_transport"), f"+a={folder}/a.hex",
                       f"+pn={folder}/pn.hex", f"+dir={folder}", f"+bp={bp}",
                       f"+out={output}", f"+limit={0 if directed else 20000}",
                       f"+directed={directed}"]
                subprocess.run(cmd, check=True, cwd=ROOT)
                rows = list(csv.DictReader(output.open()))
                bymode = {r["mode"]: r for r in rows}
                for r in rows:
                    for k in r:
                        if k != "mode":
                            r[k] = int(r[k])
                    for k in ("dec_mismatch", "plane_mismatch", "exponent_mismatch"):
                        assert r[k] == 0
                fx = bymode["fx_full"]["cycles"]
                bf = bymode["bf_full"]["cycles"]
                for r in rows:
                    r.update(trace=trace, population=pop, backpressure=bp,
                             cycles_per_group=r["cycles"] / r["groups"],
                             payload_per_group=r["header_planes"] / r["groups"],
                             total_ratio_vs_fx_full=r["cycles"] / fx,
                             total_ratio_vs_bf_full=r["cycles"] / bf,
                             bus_bytes=r["groups"] * 120)
                    records.append(r)
    (dest / "summary.json").write_text(json.dumps({
        "scope": "post-Y, precomputed threshold transport plus certificate gate; no BN producer",
        "bus_bits": 64, "max_outstanding": 1, "Y_reads_per_group": 5,
        "tau_reads_per_group": 10, "rows": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
