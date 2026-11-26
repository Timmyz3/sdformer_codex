#!/usr/bin/env python3
"""Frozen-inventory arithmetic bounds, optionally bound to actual E1 work."""
import argparse
import hashlib
import json
from pathlib import Path

MODES = ("Q6", "patch_Q4", "fc1_Q4", "fc2_Q4", "mlp_Q4")


def chosen(name, mode):
    return ((mode == "patch_Q4" and "residual_encoding.resblocks" in name)
            or (mode == "fc1_Q4" and name.endswith(".fc1"))
            or (mode == "fc2_Q4" and name.endswith(".fc2"))
            or (mode == "mlp_Q4" and ".mlp." in name))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--actual-result", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    inv = json.loads(a.inventory.read_text())
    rows = [x for x in inv["inventory"]["affines"] if x["scope"] in ("MLP12", "patch4")]
    assert len(rows) == 28
    summaries = {x["implementation"]: x["summary"]["PE_arithmetic_ticks"]
                 for x in inv["results"] if x["scope"] == "MLP12_patch4"}
    native, q6 = summaries["native"], summaries["Q6_serial"]
    r = {"schema": "flow_e1_arithmetic_upper_bounds_v1",
         "inventory_sha256": hashlib.sha256(a.inventory.read_bytes()).hexdigest(),
         "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "native_arithmetic_ticks": native, "Q6_arithmetic_ticks": q6,
         "scope": "Kernel work upper bounds only; not net cycle simulation, GPU latency or chip speed. New selection, banking, reordering and supply costs excluded.",
         "modes": {}}
    for mode in MODES:
        delta = sum(2 * x["row_domains"][0]["complete_rows"]
                    * ((x["K"] + 127) // 128) * ((x["Cout"] + 15) // 16)
                    for x in rows if chosen(x["weight"][:-7], mode))
        r["modes"][mode] = {"extra_kernel_ticks_cancelled_vs_Q6": delta,
                            "extra_fraction_of_native": delta / native,
                            "optimistic_total_arithmetic_ticks": q6 - delta,
                            "optimistic_fraction_below_native": 1 - (q6 - delta) / native}
    if a.actual_result:
        actual = json.loads(a.actual_result.read_text())
        assert actual["status"] == "LOCALIZATION80_AND4_REFERENCE_COMPLETE"
        checks = 0
        for mode in MODES:
            for frame in actual["evaluations"][mode]["frames"]:
                assert len(frame["work"]) == 28
                for op in rows:
                    name = op["weight"][:-7]
                    w = frame["work"][name]
                    complete = op["row_domains"][0]["complete_rows"]
                    tail = op["rows"] - complete
                    bits = 4 if chosen(name, mode) else 6
                    assert (w["rows"], w["Cin"], w["Cout"], w["complete_rows"], w["tail_rows"]) == (
                        op["rows"], op["K"], op["Cout"], complete, tail)
                    assert w["logical_row_plane_work"] == bits * complete + 8 * tail
                    assert w["full_T_and_channels"] and w["spatial_rows_retained"]
                    checks += 1
        r.update(actual_source_sha256=hashlib.sha256(a.actual_result.read_bytes()).hexdigest(),
                 actual_affine_work_checks=checks, actual_work_status="PASS")
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r["modes"]))


if __name__ == "__main__":
    main()
