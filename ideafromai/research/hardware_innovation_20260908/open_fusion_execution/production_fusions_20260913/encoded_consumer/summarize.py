"""Summarize the five completed H8 common-supply runs; no model execution."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent
FILES = ["ordinary_corner_final.json", "ordinary_interior_final.json",
         "lifting_raw_corner_final.json", "lifting_raw_interior_final.json",
         "ordinary_interior_final_stress.json"]


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, stages, comparisons = [], [], []
    for name in FILES:
        data = json.loads((HERE / name).read_text())
        assert data["complete"] and len(data["rows"]) == 7
        arms = {r["arm"]: r for r in data["rows"]}
        raw = arms["raw"]["service_slots"]
        for r in data["rows"]:
            c = r["counts"]
            assert r["service_slots"] == r["producer_end"] + r["consumer_service_slots"]
            assert sum(r["stages"].values()) == r["service_slots"]
            assert all(v["differences"] == 0 for v in r["consumer"]["checks"].values())
            assert r["consumer"]["actual_PED_SRAM_egress_checked"]
            if r["quantizer"]:
                assert c["observed_quantizer_values_compared"] == 15360
            if r["encoded"]:
                assert c["common_Q8_H8_source_RF_load"] == c["Q8_vector_store"] == 1920
            key = dict(axis=data["axis"], window=data["window"],
                       pressure="stress" if data["stress"] else "ready", arm=r["arm"])
            rows.append(dict(**key, service_slots=r["service_slots"],
                producer_slots=r["producer_end"], consumer_slots=r["consumer_service_slots"],
                change_vs_raw_percent=100 * (r["service_slots"] / raw - 1),
                **{f"{k}_bytes": v for k, v in r["port_bytes"].items()},
                operand_wait=c.get("operand_wait", 0),
                port_or_writeback_wait=c.get("port_or_writeback_wait", 0),
                quantizer_values_checked=c.get("observed_quantizer_values_compared", 0),
                clipped_lanes=c.get("quantizer_clipped_lanes", 0),
                CSE_spill_SW64=c.get("latent_CSE_spill_SW64", 0),
                CSE_reload_vectors=c.get("latent_CSE_reload_RF_load", 0),
                original_result_differences=0, source=name))
            stages.extend(dict(**key, stage=k, slots=v) for k, v in r["stages"].items())
        for mode in ("fixed", "affine", "full"):
            expanded, code = arms[mode+"_expanded"], arms[mode+"_code8"]
            comparisons.append(dict(axis=data["axis"], window=data["window"],
                pressure="stress" if data["stress"] else "ready", mode=mode,
                expanded_slots=expanded["service_slots"], code_slots=code["service_slots"],
                saved_slots=expanded["service_slots"]-code["service_slots"],
                saved_percent=100*(1-code["service_slots"]/expanded["service_slots"]),
                code_change_vs_raw_percent=100*(code["service_slots"]/raw-1)))
    write_csv("summary.csv", rows)
    write_csv("stages.csv", stages)
    write_csv("comparisons.csv", comparisons)
    result = dict(evidence="CPU finite-resource payload model; not RTL, PPA, layer or network",
        common_source_supply="H8_T10_RF60_69", final_cases=len(rows),
        quantized_cases=sum(r["quantizer_values_checked"] != 0 for r in rows),
        RF_live_bounds=dict(full_U=73, full_CSE_with_paid_spill=96, V=90),
        rows=rows, same_function_comparisons=comparisons)
    (HERE / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(final_cases=len(rows), all_integer_endpoints_equal=True,
                         same_function_comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()
