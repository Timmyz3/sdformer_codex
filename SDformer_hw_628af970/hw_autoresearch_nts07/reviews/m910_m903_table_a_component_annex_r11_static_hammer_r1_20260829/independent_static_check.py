#!/usr/bin/env python3
"""Receipt-blind static check of M910 against sealed M698/M706/M903 evidence."""

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
CONFIG = HW_ROOT / "system_simulator/config/m910_h67_table_a_component_annex_r11_20260829.json"
CONTRACT = HW_ROOT / "contracts/m910_h67_table_a_component_annex_r11_contract_r1_20260829.json"
M698 = HW_ROOT / "system_simulator/config/m698_h67_paper_metric_registry_r10_20260828.json"
M706 = HW_ROOT / "reviews/m706_m698_table_a_registry_r10_fresh_hammer_r1_20260828"
M903 = HW_ROOT / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
DOCS359 = HW_ROOT / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    str(CONFIG): "4e8ce01102d18c90ea9ed95544266ddafb4adf27b28653a2028d60365ea81d1b",
    str(CONTRACT): "a97fb69412268f8f16dda5ef17cc8d987a5c491e1094217e040f3c1da70386d4",
    str(M698): "6d9dedb378acfc43330a09315274e4cbe372c2abf1b9749916b606261ab2e5a3",
    str(M706 / "review.json"): "a1b109235cea7af04a63c88001290d9e785935e77aa1e65f10834c08b6eb8b16",
    str(M706 / "SHA256SUMS"): "b3aee3e711c99892d3ec13d76010c333072ff7374ebdb66dee6f0885cc0371d9",
    str(M706 / "SHA256SUMS.seal.sha256"): "960c816fa7ac3b6b47236e3457927fc42a54c1b882e390cb00c05e187524dc73",
    str(M903 / "review.json"): "89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a",
    str(M903 / "SHA256SUMS"): "e99268c516969eba1cd0ae29131146dc4b5ece2d7197b10924debab0b60d9984",
    str(M903 / "SHA256SUMS.seal.sha256"): "0394ce7e485c780355dbb841797f7fa518171bb00330ae07234a1a9a4e96316f",
    str(DOCS359): "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_seal(root, manifest_sha):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    assert outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        if rel.startswith("./"):
            rel = rel[2:]
        assert sha(root / rel) == digest


def main():
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, path
    check_seal(M706, EXPECTED[str(M706 / "SHA256SUMS")])
    check_seal(M903, EXPECTED[str(M903 / "SHA256SUMS")])
    base = load(M698)
    m706 = load(M706 / "review.json")
    source = load(M903 / "review.json")
    config = load(CONFIG)
    contract = load(CONTRACT)
    assert base["production_run_bundles"] == {}
    assert base["claim_boundary"]["table_a_admitted_rows"] == 0
    assert m706["canonical"]["validated_production_runs"] == 0
    assert m706["admission"]["table_a_ppa_row_admitted"] is False
    assert source["score_out_of_100"] == 100
    assert source["p0_count"] == source["p1_count"] == source["p2_count"] == 0
    assert source["claim_boundary"]["system_speedup"] is False
    assert source["claim_boundary"]["power"] is False
    assert source["claim_boundary"]["energy"] is False
    assert source["claim_boundary"]["paper_ppa_ready"] is False
    row = config["component_rows"]["c2_typed_signed_k8_vs_equal_bandwidth_k1x8"]
    area = row["dc_setup_area"]["axes"]
    source_area = source["dc_evidence"]["axes"]
    for axis in ("k1", "k8", "k1x8"):
        assert Decimal(area[axis]["cell_area_um2"]) == Decimal(
            str(source_area[axis]["area_um2"]))
    fair = row["directed_equal_bandwidth_metrics"]
    assert fair["k8_sum_cycles"] == 1913
    assert fair["k1x8_sum_cycles"] == 1945
    cycle = Decimal(1945) / Decimal(1913)
    throughput = cycle * Decimal(area["k1x8"]["cell_area_um2"]) / Decimal(
        area["k8"]["cell_area_um2"])
    saving = (Decimal(1) - Decimal(area["k8"]["cell_area_um2"]) /
              Decimal(area["k1x8"]["cell_area_um2"])) * Decimal(100)
    assert Decimal(fair["fair_cycle_speedup_x"]) == cycle.quantize(
        Decimal("0.00000001"), rounding=ROUND_HALF_UP)
    assert Decimal(fair["fair_throughput_per_mm2_x"]) == throughput.quantize(
        Decimal("0.000000001"), rounding=ROUND_HALF_UP)
    assert Decimal(fair["logic_cell_area_saving_percent"]) == saving.quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP)
    boundary = row["claim_boundary"]
    for key in ("system_speedup", "power", "energy", "ppa", "paper_ppa_ready",
                "paper_headline", "macro_inclusive", "full_network", "trace_weighted",
                "k8_vs_single_k1_performance_headline"):
        assert boundary[key] is False
    admission = config["admission_boundary"]
    assert admission["table_a_full_system_production_rows"] == 0
    assert admission["production_component_rows"] == 1
    assert contract["schema_decision"]["existing_m698_table_a_schema_accepts_m903_as_full_system_row"] is False
    assert contract["claim_boundary"]["system_speedup"] is False
    print("M910_STATIC_HAMMER_PASS score=100 p0=0 p1=0 p2=0 component_rows=1 full_system_table_a_rows=0")
    print("areas_um2=124620.173180,131086.241193,585479.153645 cycles=1913/1945 fair_speedup=1.01672765 throughput_per_mm2=4.541077998 area_saving_percent=77.6104")
    print("eda=0 gpu=0 remote=0 license_queries=0 docs359_sha256=" + EXPECTED[str(DOCS359)])


if __name__ == "__main__":
    main()
