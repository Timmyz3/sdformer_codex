#!/usr/bin/env python3
"""M528 r3 schema-safe wrapper around the frozen same-ledger recompute.

The r1/r2 cycle, traffic, capacity, worker, and aggregation implementation is
kept byte-for-byte in the frozen legacy analyzer.  This wrapper fixes exactly
one integration defect: the generated SRAM area is read from the explicitly
contracted slow-corner JSON pointer before the legacy computation is entered.

``--schema-smoke-only`` validates every non-streaming identity/key dependency
without importing the row worker, creating a process pool, replaying a row, or
creating a production result directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
LEGACY_ANALYZER = (
    ROOT
    / "system_simulator/scripts/analyze_m528_h67_single_port_same_ledger_recompute.py"
)
EXPECTED_LEGACY_ANALYZER_SHA256 = (
    "c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a"
)
DEFAULT_EXECUTION_CONTRACT = (
    ROOT
    / "contracts/m528_h67_single_port_same_ledger_execution_contract_r3_20260827.json"
)
DEFAULT_OUT = (
    ROOT
    / "results/m528_h67_single_port_same_ledger_recompute_r3_20260827"
)

EXPECTED_MAPPING_SCHEMA = "tsmc28_sram_macro_mapping_audit_v1"
EXPECTED_AREA_POINTER = "generated_view_inventory.slow.area_um2"
EXPECTED_CORNER = "ssg0p9v125c"
EXPECTED_CELL = "TS1N28HPCPHVTB128X128M4S"
EXPECTED_LOGICAL_SHAPE = "128x128b 1RW SP"
EXPECTED_MACRO_COUNT = 9
EXPECTED_PER_MACRO_AREA_UM2 = 8758.3606
EXPECTED_TOTAL_AREA_UM2 = 78825.2454
SMOKE_PASS_TOKEN = "PASS_M528_R3_SCHEMA_SMOKE_ONLY"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise RuntimeError(f"non-standard JSON token: {token}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )


def json_pointer(document: dict[str, Any], pointer: str) -> Any:
    require(pointer == EXPECTED_AREA_POINTER, "schema-smoke expected area pointer mismatch")
    value: Any = document
    for component in pointer.split("."):
        require(isinstance(value, dict), f"non-object before JSON pointer component: {component}")
        require(component in value, f"missing JSON pointer component: {component}")
        value = value[component]
    return value


def verify_manifest(directory: Path, expected_manifest_sha: str | None = None) -> dict[str, str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), f"missing double seal: {directory}")
    actual_manifest_sha = sha256_file(manifest)
    if expected_manifest_sha is not None:
        require(actual_manifest_sha == expected_manifest_sha, f"manifest SHA drift: {directory}")
    outer_parts = outer.read_text(encoding="utf-8").strip().split()
    require(
        len(outer_parts) == 2
        and outer_parts[0] == actual_manifest_sha
        and outer_parts[1] == "SHA256SUMS",
        f"outer seal mismatch: {directory}",
    )
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in entries, f"duplicate manifest entry: {name}")
        path = directory / name
        require(path.is_file(), f"missing sealed file: {path}")
        require(sha256_file(path) == digest, f"sealed file SHA drift: {path}")
        entries[name] = digest
    return entries


def require_keys(document: dict[str, Any], paths: tuple[str, ...]) -> None:
    for path in paths:
        value: Any = document
        for component in path.split("."):
            require(isinstance(value, dict), f"non-object before required key: {path}")
            require(component in value, f"missing required key path: {path}")
            value = value[component]


def validate_schema(
    execution_path: Path,
    out_path: Path,
    expected_pointer: str,
    expected_corner: str,
) -> dict[str, Any]:
    """Validate all pre-row dependencies and return the explicit area adapter."""

    require(not out_path.exists(), "refuse to overwrite M528 r3 output")
    require(expected_pointer == EXPECTED_AREA_POINTER, "schema-smoke expected area pointer mismatch")
    require(expected_corner == EXPECTED_CORNER, "schema-smoke expected SRAM corner mismatch")
    require(LEGACY_ANALYZER.is_file(), "missing frozen M528 r1/r2 analyzer")
    require(
        sha256_file(LEGACY_ANALYZER) == EXPECTED_LEGACY_ANALYZER_SHA256,
        "frozen M528 r1/r2 analyzer SHA drift",
    )

    execution = strict_json(execution_path)
    require(
        execution["schema"] == "m528_h67_single_port_same_ledger_execution_contract_v1",
        "execution schema drift",
    )
    recovery = execution["r3_schema_recovery"]
    require(recovery["revision"] == "r3", "r3 recovery identity drift")
    require(recovery["area_json_pointer"] == EXPECTED_AREA_POINTER, "contract area pointer drift")
    require(recovery["corner"] == EXPECTED_CORNER, "contract SRAM corner drift")
    require(recovery["mapping_schema"] == EXPECTED_MAPPING_SCHEMA, "contract mapping schema drift")
    require(recovery["cell"] == EXPECTED_CELL, "contract generated-cell drift")
    require(recovery["logical_shape"] == EXPECTED_LOGICAL_SHAPE, "contract logical-shape drift")
    require(int(recovery["macro_count"]) == EXPECTED_MACRO_COUNT, "contract macro-count drift")
    require(
        math.isclose(float(recovery["per_macro_area_um2"]), EXPECTED_PER_MACRO_AREA_UM2, rel_tol=0.0, abs_tol=1e-12),
        "contract per-macro area drift",
    )
    require(
        math.isclose(float(recovery["total_area_um2"]), EXPECTED_TOTAL_AREA_UM2, rel_tol=0.0, abs_tol=1e-9),
        "contract total area drift",
    )
    require(
        math.isclose(
            float(recovery["per_macro_area_um2"]) * int(recovery["macro_count"]),
            float(recovery["total_area_um2"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "contract nine-macro area multiplication drift",
    )

    governing_path = ROOT / execution["governing_contract"]["path"]
    require(governing_path.is_file(), "missing governing contract")
    require(
        sha256_file(governing_path) == execution["governing_contract"]["sha256"],
        "governing contract SHA drift",
    )
    governing = strict_json(governing_path)
    require(governing["schema"] == "m528_single_port_same_ledger_recompute_contract_v1", "governing schema drift")
    require(int(governing["authorization"]["cpu_runs"]) == 1, "CPU authorization drift")
    require(
        int(governing["authorization"]["eda_runs"]) == 0
        and int(governing["authorization"]["gpu_runs"]) == 0,
        "forbidden authorization drift",
    )

    for item in governing["frozen_inputs"].values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing frozen input: {path}")
        require(sha256_file(path) == item["sha256"], f"frozen SHA drift: {path}")
    for item in execution["additional_frozen_inputs"].values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing additional input: {path}")
        require(sha256_file(path) == item["sha256"], f"additional SHA drift: {path}")

    m468_path = ROOT / governing["frozen_inputs"]["m468_r6_result"]["path"]
    m473_path = ROOT / governing["frozen_inputs"]["m473_r3_result"]["path"]
    m505_path = ROOT / governing["frozen_inputs"]["m505_result"]["path"]
    m468 = strict_json(m468_path)
    m473 = strict_json(m473_path)
    m505 = strict_json(m505_path)
    require_keys(
        m468,
        (
            "points",
        ),
    )
    require(isinstance(m468["points"], list) and m468["points"], "M468 points population missing")
    for point in m468["points"]:
        require_keys(
            point,
            (
                "mode",
                "fits_both_240k_gates",
                "resident_block_banks",
                "bandwidth_bytes_per_cycle",
                "cycles",
                "weight_dram_bytes",
                "source_sram_bytes",
                "dma_commands",
                "commit_cycles",
                "capacity.logical_items",
                "capacity.macro_rounded_items",
            ),
        )
    require_keys(
        m473,
        (
            "output_files",
            "best_128Bps_feasible_point.bit_cycles",
            "best_128Bps_feasible_point.product_cycles",
            "best_128Bps_feasible_point.weight_dram_bytes",
            "best_128Bps_feasible_point.source_sram_bytes",
            "best_128Bps_feasible_point.weight_dma_commands",
            "best_128Bps_feasible_point.commit_cycles",
            "best_128Bps_feasible_point.capacity.logical_items",
            "best_128Bps_feasible_point.capacity.macro_rounded_items",
        ),
    )
    for output in m473["output_files"].values():
        require_keys(output, ("path", "sha256"))
    require_keys(
        m505,
        (
            "identity.m410r2_rows.path",
            "identity.m410r2_rows.sha256",
            "identity.m505_analyzer.sha256",
            "identity.m504_analyzer.sha256",
            "cycle_comparison.m504_single_port_cycles",
        ),
    )

    m505_entries = verify_manifest(
        m505_path.parent,
        governing["frozen_inputs"]["m505_manifest"]["sha256"],
    )
    require(
        m505_entries[m505_path.name] == governing["frozen_inputs"]["m505_result"]["sha256"],
        "M505 result not sealed",
    )
    prior_csv = m505_path.parent / "m505_operator_sample_summary_r1.csv"
    require(prior_csv.name in m505_entries, "M505 recurrence CSV not sealed")

    m473_entries = verify_manifest(m473_path.parent)
    require(
        m473_entries[m473_path.name] == governing["frozen_inputs"]["m473_r3_result"]["sha256"],
        "M473 result not sealed",
    )
    for output in m473["output_files"].values():
        require(m473_entries[output["path"]] == output["sha256"], "M473 output identity drift")

    raw_item = m505["identity"]["m410r2_rows"]
    rows_path = ROOT / raw_item["path"]
    require(rows_path.is_file(), "missing transitively frozen row ledger")
    require(sha256_file(rows_path) == raw_item["sha256"], "transitively frozen row ledger drift")
    require(raw_item == execution["additional_frozen_inputs"]["m410r2_rows"], "execution/raw row identity mismatch")

    mapping_path = ROOT / governing["frozen_inputs"]["sram_mapping"]["path"]
    mapping = strict_json(mapping_path)
    require(mapping["schema"] == EXPECTED_MAPPING_SCHEMA, "mapping schema drift")
    inventory = mapping["generated_view_inventory"]
    require(inventory["cell"] == EXPECTED_CELL, "generated SRAM cell drift")
    require(inventory["logical_shape"] == EXPECTED_LOGICAL_SHAPE, "generated macro shape drift")
    require(inventory["slow"]["corner"] == EXPECTED_CORNER, "generated SRAM slow corner drift")
    area = float(json_pointer(mapping, recovery["area_json_pointer"]))
    require(math.isfinite(area) and area > 0.0, "generated SRAM slow area must be finite and positive")
    require(
        math.isclose(area, EXPECTED_PER_MACRO_AREA_UM2, rel_tol=0.0, abs_tol=1e-12),
        "generated SRAM per-macro area drift",
    )
    total = area * EXPECTED_MACRO_COUNT
    require(
        math.isclose(total, EXPECTED_TOTAL_AREA_UM2, rel_tol=0.0, abs_tol=1e-9),
        "generated SRAM nine-macro area drift",
    )
    require(
        math.isclose(
            total,
            float(governing["capacity_recompute"]["generated_1rw_scratch_area_um2"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "generated SRAM/governing area drift",
    )

    require_keys(
        governing,
        (
            "frozen_coordinate.trace_rows",
            "frozen_coordinate.task_order",
            "frozen_coordinate.row_tile",
            "frozen_coordinate.resident_output_block_banks",
            "frozen_coordinate.external_bandwidth_bytes_per_cycle",
            "frozen_coordinate.weight_dram_bytes",
            "frozen_coordinate.source_sram_bytes",
            "frozen_coordinate.commit_cycles",
            "frozen_aggregate_anchors.m468_strong_zero_cycles",
            "frozen_aggregate_anchors.m473_same_coordinate_bit_cycles",
            "frozen_aggregate_anchors.m473_fused_concurrent_1r1w_ceiling_cycles",
            "frozen_aggregate_anchors.m505_dead_write_only_cycles",
            "frozen_aggregate_anchors.m505_combined_pvrf_cycles",
            "capacity_recompute.budget_bytes",
            "capacity_recompute.replace_nominal_scratch_bytes",
            "capacity_recompute.generated_1rw_scratch_bytes",
            "capacity_recompute.conservative_extra_live_bitmap_macro_rounded_bytes",
            "capacity_recompute.m505_conservative_macro_rounded_bytes",
            "capacity_recompute.m505_budget_margin_bytes",
            "cpu_decision_gates.minimum_speedup_vs_m468_strong_zero",
            "cpu_decision_gates.minimum_speedup_vs_same_coordinate_bit",
            "cpu_decision_gates.maximum_cycle_regression_from_frozen_m505_dead_write_only",
            "claim_boundary",
        ),
    )
    require(int(governing["frozen_coordinate"]["row_tile"]) == 64, "row64 coordinate drift")
    require(int(governing["frozen_coordinate"]["resident_output_block_banks"]) == 8, "B8 coordinate drift")
    require(int(governing["frozen_coordinate"]["external_bandwidth_bytes_per_cycle"]) == 128, "128-byte/cycle coordinate drift")
    require(int(execution["frozen_algorithm_constants"]["cam_compare_lanes"]) == 64, "CAM64 coordinate drift")

    return {
        "mapping_path": mapping_path,
        "mapping_area_um2": area,
        "mapping_document": mapping,
    }


def load_legacy() -> Any:
    spec = importlib.util.spec_from_file_location("m528_frozen_r1r2_for_r3", LEGACY_ANALYZER)
    require(spec is not None and spec.loader is not None, "cannot load frozen M528 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract", type=Path, default=DEFAULT_EXECUTION_CONTRACT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--schema-smoke-only", action="store_true")
    parser.add_argument("--smoke-expected-pointer", default=EXPECTED_AREA_POINTER)
    parser.add_argument("--smoke-expected-corner", default=EXPECTED_CORNER)
    args = parser.parse_args()
    execution_path = args.execution_contract.resolve()
    out_path = args.out.resolve()
    adapter = validate_schema(
        execution_path,
        out_path,
        args.smoke_expected_pointer,
        args.smoke_expected_corner,
    )
    if args.schema_smoke_only:
        print(SMOKE_PASS_TOKEN)
        return 0

    legacy = load_legacy()
    legacy_strict_json = legacy.strict_json
    mapping_path = adapter["mapping_path"]
    mapping_area = adapter["mapping_area_um2"]

    def explicit_slow_corner_adapter(path: Path) -> dict[str, Any]:
        document = legacy_strict_json(path)
        if path.resolve() == mapping_path.resolve():
            # The compatibility key is derived only from the contracted slow
            # pointer after validate_schema proved schema/corner/cell/geometry.
            document = copy.deepcopy(document)
            document["generated_view_inventory"]["area_um2"] = mapping_area
        return document

    legacy.strict_json = explicit_slow_corner_adapter
    legacy.__file__ = str(Path(__file__).resolve())
    sys.argv = [
        str(Path(__file__).resolve()),
        "--execution-contract",
        str(execution_path),
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--out",
        str(out_path),
    ]
    return int(legacy.main())


if __name__ == "__main__":
    raise SystemExit(main())
