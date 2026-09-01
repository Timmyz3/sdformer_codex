#!/usr/bin/env python3
"""Build the conservative C1 full-storage macro-inclusive area model."""
from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


getcontext().prec = 32
ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
M1102 = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M993 = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829/m993_recovered_dc_receipt.json"
AREA = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829/original_quarantine/reports/area_hierarchy.rpt"
M1114 = HW / "reviews/m1114_m1102_c1_work8_full_replay_result_hammer_r1_20260830/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    M1102: "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91",
    M993: "193a06e847755cca99b9dcf079cd0fee203664203e7d8b1abc8cad72c73007cc",
    AREA: "ff6683e13fe9ad8eaa0e47ff64c2f17037bfb1ee8993290331a4fc355185a94c",
    M1114: "8ced2392215b7bd70b8afcc90efab3f6078c9b3cc9b1a9d7b0c1d5e33d36b8bc",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
SCHEMA = "m1591_c1_full_storage_macro_area_model_r1_v1"
STATUS = "PASS_M1591_C1_FULL_STORAGE_CONSERVATIVE_MACRO_AREA_MODEL"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in rows:
            require(key not in output, "duplicate key: " + key)
            output[key] = value
        return output
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite token: " + token)))
    require(type(value) is dict, "JSON root is not object")
    return value


def decimal_field(pattern: str, text: str) -> Decimal:
    match = re.search(pattern, text, flags=re.M)
    require(match is not None, "area field missing: " + pattern)
    return Decimal(match.group(1))


def build() -> dict[str, Any]:
    for path, digest in PINS.items():
        require(path.is_file() and not path.is_symlink() and sha256(path) == digest,
                "identity drift: " + str(path))
    old = strict_json(M1102)
    receipt = strict_json(M993)
    capacity = old["raw_cpu_model"]["capacity"]
    require(capacity["derived_total_bytes"] == 214_912 and
            capacity["budget_bytes"] == 245_760 and
            capacity["macro_bytes"] == 2_048,
            "capacity coordinate drift")
    parent = capacity["parent_plus_other"]
    require(parent["bytes"] == 42_880 and parent["parent_scratch_bytes"] == 18_432 and
            capacity["psum"] == {"bytes": 122_880, "groups": 4,
                                 "macro_count": 60, "wide_slices_per_group": 15} and
            capacity["weight"] == {"bytes": 49_152, "macro_count": 24},
            "storage decomposition drift")
    metadata = parent["bytes"] - parent["parent_scratch_bytes"]
    metadata_macros = math.ceil(metadata / capacity["macro_bytes"])
    counts = {"parent_scratch": 9, "psum": 60, "weight": 24,
              "metadata_and_reserve_conservative": metadata_macros}
    total_macros = sum(counts.values())
    physical_bytes = total_macros * capacity["macro_bytes"]
    require(total_macros == 105 and physical_bytes == 215_040,
            "full-storage macro rounding drift")

    report = AREA.read_text(encoding="utf-8")
    total_area = decimal_field(r"^Total cell area:\s+([0-9.]+)$", report)
    macro_area_9 = decimal_field(r"^Macro/Black Box area:\s+([0-9.]+)$", report)
    require(total_area == Decimal(str(receipt["total_cell_area_um2_dc_reported"])) and
            receipt["macro_count"] == 9 and receipt["setup"]["met"] is True,
            "M993 area/setup receipt drift")
    logic_area = total_area - macro_area_9
    macro_area_each = macro_area_9 / Decimal(9)
    modeled_macro_area = macro_area_each * Decimal(total_macros)
    modeled_total_area = logic_area + modeled_macro_area
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "identity": {path.relative_to(ROOT).as_posix(): digest
                     for path, digest in PINS.items()},
        "technology": {"nm": 28, "macro_cell": receipt["macro_cell"],
                       "macro_geometry": "128x128-bit 1RW single-port",
                       "macro_capacity_bytes": 2048},
        "logical_storage": {
            "parent_scratch_bytes": 18_432,
            "psum_bytes": 122_880,
            "weight_bytes": 49_152,
            "metadata_and_reserve_bytes": metadata,
            "total_bytes": capacity["derived_total_bytes"],
            "budget_bytes": capacity["budget_bytes"],
        },
        "conservative_macro_rounding": {
            "counts": counts, "total_macro_count": total_macros,
            "represented_bytes": physical_bytes,
            "rounding_overhead_bytes": physical_bytes - capacity["derived_total_bytes"],
            "budget_margin_after_rounding_bytes": capacity["budget_bytes"] - physical_bytes,
        },
        "area_um2": {
            "dc_logic_excluding_nine_parent_macros": str(logic_area),
            "foundry_macro_area_each_from_dc": str(macro_area_each),
            "modeled_105_macro_area": str(modeled_macro_area),
            "modeled_logic_plus_full_storage": str(modeled_total_area),
            "modeled_logic_plus_full_storage_mm2": str(modeled_total_area / Decimal(1_000_000)),
        },
        "timing": {"clock_ns": 3.0, "existing_logic_plus_nine_macro_setup_met": True,
                   "existing_setup_wns_ns": receipt["setup"]["wns_ns"],
                   "extra_96_macros_integrated_in_timing_top": False},
        "claim_boundary": {
            "macro_area_model": True,
            "conservative_same_foundry_macro_scaling": True,
            "full_storage_logic_netlist": False,
            "full_storage_timing": False,
            "power": False, "energy": False, "throughput": False,
            "throughput_per_area": False, "system_speedup": False,
            "paper_citable_after_independent_review_with_model_label": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    value = build()
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        require(not args.out.exists(), "refuse overwrite")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
