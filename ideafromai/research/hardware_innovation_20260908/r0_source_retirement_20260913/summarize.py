"""Join measured RTL arms and separate diverse10 quality; no new experiment."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    old = json.loads((ROOT / "rtl/SUMMARY.json").read_text())
    new = json.loads((ROOT / "pair_parent_merge/SUMMARY.json").read_text())
    with (ROOT / "data/quality_summary.csv").open(newline="") as stream:
        quality = {r["arm"]: r for r in csv.DictReader(stream)}
    rows = []
    for arm, modes in new["formal_arms"].items():
        qarm = {"dense": "dense_q16", "mixed_retirement25": "fusion72"}.get(arm, arm)
        strong = modes["5"]["0"]
        merged = modes["6"]["0"]
        baseline = old["formal_arms"][arm]["3"]["0"]
        rebuilt = old["formal_arms"][arm]["4"]["0"]
        rows.append({
            "arm": arm,
            "quality_AEE_diverse10": float(quality[qarm]["AEE_frame_mean"]),
            "tiles": merged["tiles"],
            "mode3_core": baseline["cycles"],
            "mode4_core": rebuilt["cycles"],
            "mode5_core": strong["cycles"],
            "mode6_core": merged["cycles"],
            "mode6_fresh_source_origin_cycles": merged["cycles_with_fresh_source_origin"],
            "mode6_vs_mode5_core_reduction_pct": 100 * (1 - merged["cycles"] / strong["cycles"]),
            "mode6_vs_mode5_fresh_reduction_pct": 100 * (1 - merged["cycles_with_fresh_source_origin"] / strong["cycles_with_fresh_source_origin"]),
            "source_10bit_words": merged["source_words"],
            "weight_128bit_words": merged["weight_words"],
            "mode5_psum_256bit_reads": strong["psum_reads"],
            "mode6_psum_256bit_reads": merged["psum_reads"],
            "mode6_psum_256bit_writes": merged["psum_writes"],
            "paid_merge_issues": merged["merge_issues"],
        })
    with (ROOT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "scope": "Measured Verilator native r0 C96/N96/T10 K864 linear leaf, eight 2x2 output tiles, command0 without external backpressure. AEE is separate diverse10 floating-consumer evaluation, not full-network bittrue.",
        "fresh_source_origin_cycles_per_tile": 1537,
        "static_weight_mask_initial_cycles": 10656,
        "actual_cold_instance_configuration_cycles": 12193,
        "rows": rows,
        "RTL_command_runs_this_stage": 160 + old["runs"] + new["runs"],
        "scalar_output_comparisons_this_stage": 614400 + old["checked_outputs"] + new["checked_outputs"],
        "comparison_count_note": "Includes repeat commands, pressure waveforms and functional controls; not that many distinct network outputs.",
        "valid825": False,
        "ASIC_PPA": False,
    }
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["arm"], row["mode6_core"], row["mode6_fresh_source_origin_cycles"], f'{row["mode6_vs_mode5_core_reduction_pct"]:.4f}%', row["quality_AEE_diverse10"])


if __name__ == "__main__":
    main()
