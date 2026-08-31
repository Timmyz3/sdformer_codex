#!/usr/bin/env python3
"""M528 r4 schema-safe, spawn-import-safe same-ledger recompute wrapper.

R4 imports the byte-frozen r1/r2 analyzer under its real, stable module name.
This is essential because ProcessPoolExecutor(spawn) must be able to import and
unpickle legacy.worker_init/legacy.worker_phase in child processes.  The only
data repair remains the explicit slow-corner SRAM area JSON pointer.

The preflight-only mode validates sealed inputs and can execute one lightweight
spawn round trip using exact legacy.worker_init only to open the frozen ledger,
then legacy.sha256_file on docs/359. It never calls worker_phase, never replays
one row, and never creates a production result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Stable normal import: do not replace this with spec_from_file_location or a
# virtual module name. Spawn children inherit SCRIPT_DIR in sys.path and import
# this exact module name while unpickling functions.
import analyze_m528_h67_single_port_same_ledger_recompute as LEGACY  # noqa: E402


LEGACY_MODULE_NAME = "analyze_m528_h67_single_port_same_ledger_recompute"
LEGACY_ANALYZER = SCRIPT_DIR / f"{LEGACY_MODULE_NAME}.py"
EXPECTED_LEGACY_SHA256 = "c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a"
DEFAULT_EXECUTION_CONTRACT = ROOT / "contracts/m528_h67_single_port_same_ledger_execution_contract_r4_20260827.json"
DEFAULT_OUT = ROOT / "results/m528_h67_single_port_same_ledger_recompute_r4_20260827"

EXPECTED_MAPPING_SCHEMA = "tsmc28_sram_macro_mapping_audit_v1"
EXPECTED_AREA_POINTER = "generated_view_inventory.slow.area_um2"
EXPECTED_CORNER = "ssg0p9v125c"
EXPECTED_CELL = "TS1N28HPCPHVTB128X128M4S"
EXPECTED_LOGICAL_SHAPE = "128x128b 1RW SP"
EXPECTED_MACRO_COUNT = 9
EXPECTED_PER_MACRO_AREA_UM2 = 8758.3606
EXPECTED_TOTAL_AREA_UM2 = 78825.2454
EXPECTED_DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SMOKE_PASS_TOKEN = "PASS_M528_R4_PREFLIGHT_SCHEMA_AND_SPAWN_IMPORT_SELF_TEST"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_keys(document: dict[str, Any], paths: tuple[str, ...]) -> None:
    for path in paths:
        value: Any = document
        for component in path.split("."):
            require(isinstance(value, dict), f"non-object before required key: {path}")
            require(component in value, f"missing required key path: {path}")
            value = value[component]


def json_pointer(document: dict[str, Any], pointer: str) -> Any:
    require(pointer == EXPECTED_AREA_POINTER, "schema-smoke expected area pointer mismatch")
    value: Any = document
    for component in pointer.split("."):
        require(isinstance(value, dict), f"non-object before JSON pointer component: {component}")
        require(component in value, f"missing JSON pointer component: {component}")
        value = value[component]
    return value


def validate_schema(
    execution_path: Path,
    out_path: Path,
    expected_pointer: str,
    expected_corner: str,
) -> dict[str, Any]:
    require(not out_path.exists(), "refuse to overwrite M528 r4 output")
    require(expected_pointer == EXPECTED_AREA_POINTER, "schema-smoke expected area pointer mismatch")
    require(expected_corner == EXPECTED_CORNER, "schema-smoke expected SRAM corner mismatch")
    require(LEGACY.__name__ == LEGACY_MODULE_NAME, "legacy analyzer module-name drift")
    require(Path(LEGACY.__file__).resolve() == LEGACY_ANALYZER.resolve(), "legacy analyzer import-path drift")
    require(LEGACY.sha256_file(LEGACY_ANALYZER) == EXPECTED_LEGACY_SHA256, "legacy analyzer SHA drift")

    execution = LEGACY.strict_json(execution_path)
    require(execution["schema"] == "m528_h67_single_port_same_ledger_execution_contract_v1", "execution schema drift")
    recovery = execution["r4_schema_spawn_recovery"]
    require(recovery["revision"] == "r4", "r4 recovery identity drift")
    require(recovery["legacy_module_name"] == LEGACY_MODULE_NAME, "contract legacy module-name drift")
    require(recovery["legacy_import_mode"] == "normal_import_from_script_directory", "contract legacy import-mode drift")
    require(recovery["area_json_pointer"] == EXPECTED_AREA_POINTER, "contract area pointer drift")
    require(recovery["corner"] == EXPECTED_CORNER, "contract SRAM corner drift")
    require(recovery["mapping_schema"] == EXPECTED_MAPPING_SCHEMA, "contract mapping-schema drift")
    require(recovery["cell"] == EXPECTED_CELL, "contract generated-cell drift")
    require(recovery["logical_shape"] == EXPECTED_LOGICAL_SHAPE, "contract logical-shape drift")
    require(int(recovery["macro_count"]) == EXPECTED_MACRO_COUNT, "contract macro-count drift")
    require(math.isclose(float(recovery["per_macro_area_um2"]), EXPECTED_PER_MACRO_AREA_UM2, rel_tol=0.0, abs_tol=1e-12), "contract per-macro area drift")
    require(math.isclose(float(recovery["total_area_um2"]), EXPECTED_TOTAL_AREA_UM2, rel_tol=0.0, abs_tol=1e-9), "contract total area drift")
    require(math.isclose(float(recovery["per_macro_area_um2"]) * int(recovery["macro_count"]), float(recovery["total_area_um2"]), rel_tol=0.0, abs_tol=1e-9), "contract nine-macro multiplication drift")

    governing_path = ROOT / execution["governing_contract"]["path"]
    require(LEGACY.sha256_file(governing_path) == execution["governing_contract"]["sha256"], "governing contract SHA drift")
    governing = LEGACY.strict_json(governing_path)
    require(governing["schema"] == "m528_single_port_same_ledger_recompute_contract_v1", "governing schema drift")
    require(int(governing["authorization"]["cpu_runs"]) == 1, "CPU authorization drift")
    require(int(governing["authorization"]["eda_runs"]) == 0 and int(governing["authorization"]["gpu_runs"]) == 0, "forbidden authorization drift")

    for item in governing["frozen_inputs"].values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing frozen input: {path}")
        require(LEGACY.sha256_file(path) == item["sha256"], f"frozen SHA drift: {path}")
    for item in execution["additional_frozen_inputs"].values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing additional input: {path}")
        require(LEGACY.sha256_file(path) == item["sha256"], f"additional SHA drift: {path}")

    m468_path = ROOT / governing["frozen_inputs"]["m468_r6_result"]["path"]
    m473_path = ROOT / governing["frozen_inputs"]["m473_r3_result"]["path"]
    m505_path = ROOT / governing["frozen_inputs"]["m505_result"]["path"]
    m468 = LEGACY.strict_json(m468_path)
    m473 = LEGACY.strict_json(m473_path)
    m505 = LEGACY.strict_json(m505_path)
    require(isinstance(m468["points"], list) and m468["points"], "M468 points population missing")
    for point in m468["points"]:
        require_keys(point, (
            "mode", "fits_both_240k_gates", "resident_block_banks",
            "bandwidth_bytes_per_cycle", "cycles", "weight_dram_bytes",
            "source_sram_bytes", "dma_commands", "commit_cycles",
            "capacity.logical_items", "capacity.macro_rounded_items",
        ))
    require_keys(m473, (
        "output_files", "best_128Bps_feasible_point.bit_cycles",
        "best_128Bps_feasible_point.product_cycles",
        "best_128Bps_feasible_point.weight_dram_bytes",
        "best_128Bps_feasible_point.source_sram_bytes",
        "best_128Bps_feasible_point.weight_dma_commands",
        "best_128Bps_feasible_point.commit_cycles",
        "best_128Bps_feasible_point.capacity.logical_items",
        "best_128Bps_feasible_point.capacity.macro_rounded_items",
    ))
    require_keys(m505, (
        "identity.m410r2_rows.path", "identity.m410r2_rows.sha256",
        "identity.m505_analyzer.sha256", "identity.m504_analyzer.sha256",
        "cycle_comparison.m504_single_port_cycles",
    ))

    m505_entries = LEGACY.verify_manifest(m505_path.parent, governing["frozen_inputs"]["m505_manifest"]["sha256"])
    require(m505_entries[m505_path.name] == governing["frozen_inputs"]["m505_result"]["sha256"], "M505 result not sealed")
    require("m505_operator_sample_summary_r1.csv" in m505_entries, "M505 recurrence CSV not sealed")
    m473_entries = LEGACY.verify_manifest(m473_path.parent)
    require(m473_entries[m473_path.name] == governing["frozen_inputs"]["m473_r3_result"]["sha256"], "M473 result not sealed")
    for output in m473["output_files"].values():
        require(m473_entries[output["path"]] == output["sha256"], "M473 output identity drift")

    raw_item = m505["identity"]["m410r2_rows"]
    rows_path = ROOT / raw_item["path"]
    require(LEGACY.sha256_file(rows_path) == raw_item["sha256"], "transitively frozen row-ledger drift")
    require(raw_item == execution["additional_frozen_inputs"]["m410r2_rows"], "execution/raw row identity mismatch")

    mapping_path = ROOT / governing["frozen_inputs"]["sram_mapping"]["path"]
    mapping = LEGACY.strict_json(mapping_path)
    require(mapping["schema"] == EXPECTED_MAPPING_SCHEMA, "mapping schema drift")
    inventory = mapping["generated_view_inventory"]
    require(inventory["cell"] == EXPECTED_CELL, "generated SRAM cell drift")
    require(inventory["logical_shape"] == EXPECTED_LOGICAL_SHAPE, "generated macro shape drift")
    require(inventory["slow"]["corner"] == EXPECTED_CORNER, "generated SRAM slow-corner drift")
    area = float(json_pointer(mapping, recovery["area_json_pointer"]))
    require(math.isfinite(area) and area > 0.0, "generated SRAM slow area must be finite positive")
    require(math.isclose(area, EXPECTED_PER_MACRO_AREA_UM2, rel_tol=0.0, abs_tol=1e-12), "generated SRAM per-macro area drift")
    total = area * EXPECTED_MACRO_COUNT
    require(math.isclose(total, EXPECTED_TOTAL_AREA_UM2, rel_tol=0.0, abs_tol=1e-9), "generated SRAM nine-macro area drift")
    require(math.isclose(total, float(governing["capacity_recompute"]["generated_1rw_scratch_area_um2"]), rel_tol=0.0, abs_tol=1e-9), "generated SRAM/governing area drift")

    require_keys(governing, (
        "frozen_coordinate.trace_rows", "frozen_coordinate.task_order",
        "frozen_coordinate.row_tile", "frozen_coordinate.resident_output_block_banks",
        "frozen_coordinate.external_bandwidth_bytes_per_cycle",
        "frozen_coordinate.weight_dram_bytes", "frozen_coordinate.source_sram_bytes",
        "frozen_coordinate.commit_cycles",
        "frozen_aggregate_anchors.m468_strong_zero_cycles",
        "frozen_aggregate_anchors.m473_same_coordinate_bit_cycles",
        "frozen_aggregate_anchors.m473_fused_concurrent_1r1w_ceiling_cycles",
        "frozen_aggregate_anchors.m505_dead_write_only_cycles",
        "frozen_aggregate_anchors.m505_combined_pvrf_cycles",
        "capacity_recompute.budget_bytes", "capacity_recompute.replace_nominal_scratch_bytes",
        "capacity_recompute.generated_1rw_scratch_bytes",
        "capacity_recompute.conservative_extra_live_bitmap_macro_rounded_bytes",
        "capacity_recompute.m505_conservative_macro_rounded_bytes",
        "capacity_recompute.m505_budget_margin_bytes",
        "cpu_decision_gates.minimum_speedup_vs_m468_strong_zero",
        "cpu_decision_gates.minimum_speedup_vs_same_coordinate_bit",
        "cpu_decision_gates.maximum_cycle_regression_from_frozen_m505_dead_write_only",
        "claim_boundary",
    ))
    require(int(governing["frozen_coordinate"]["row_tile"]) == 64, "row64 coordinate drift")
    require(int(governing["frozen_coordinate"]["resident_output_block_banks"]) == 8, "B8 coordinate drift")
    require(int(governing["frozen_coordinate"]["external_bandwidth_bytes_per_cycle"]) == 128, "128-byte/cycle coordinate drift")
    require(int(execution["frozen_algorithm_constants"]["cam_compare_lanes"]) == 64, "CAM64 coordinate drift")

    return {
        "mapping_path": mapping_path,
        "mapping_area_um2": area,
        "rows_path": rows_path,
    }


def run_spawn_import_self_test(rows_path: Path) -> dict[str, Any]:
    """Exercise exact worker-init pickling/import without replaying one row."""

    target = ROOT / "docs/359_DATE终局冻结_20260813.md"
    require(LEGACY.sha256_file(target) == EXPECTED_DOCS359_SHA256, "spawn target SHA drift before submit")
    require(pickle.loads(pickle.dumps(LEGACY.worker_init)) is LEGACY.worker_init, "worker_init pickle identity drift")
    require(pickle.loads(pickle.dumps(LEGACY.worker_phase)) is LEGACY.worker_phase, "worker_phase pickle identity drift")
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=LEGACY.worker_init,
        initargs=(str(rows_path),),
    ) as pool:
        returned = pool.submit(LEGACY.sha256_file, target).result(timeout=60)
    require(returned == EXPECTED_DOCS359_SHA256, "spawn-import self-test return SHA mismatch")
    return {
        "module_name": LEGACY.__name__,
        "function": "sha256_file",
        "initializer": "worker_init",
        "worker_phase_pickle_checked_not_called": True,
        "target": str(target.relative_to(ROOT)),
        "returned_sha256": returned,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract", type=Path, default=DEFAULT_EXECUTION_CONTRACT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--schema-smoke-only", action="store_true")
    parser.add_argument("--spawn-import-self-test", action="store_true")
    parser.add_argument("--smoke-expected-pointer", default=EXPECTED_AREA_POINTER)
    parser.add_argument("--smoke-expected-corner", default=EXPECTED_CORNER)
    args = parser.parse_args()
    execution_path = args.execution_contract.resolve()
    out_path = args.out.resolve()
    adapter = validate_schema(execution_path, out_path, args.smoke_expected_pointer, args.smoke_expected_corner)

    if args.schema_smoke_only:
        require(args.spawn_import_self_test, "positive r4 schema smoke requires spawn-import self-test")
        run_spawn_import_self_test(adapter["rows_path"])
        print(SMOKE_PASS_TOKEN)
        return 0
    require(not args.spawn_import_self_test, "spawn-import self-test is preflight-only")

    legacy_strict_json = LEGACY.strict_json
    mapping_path = adapter["mapping_path"]
    mapping_area = adapter["mapping_area_um2"]

    def explicit_slow_corner_adapter(path: Path) -> dict[str, Any]:
        document = legacy_strict_json(path)
        if path.resolve() == mapping_path.resolve():
            document = copy.deepcopy(document)
            document["generated_view_inventory"]["area_um2"] = mapping_area
        return document

    LEGACY.strict_json = explicit_slow_corner_adapter
    LEGACY.__file__ = str(Path(__file__).resolve())
    sys.argv = [
        str(Path(__file__).resolve()),
        "--execution-contract", str(execution_path),
        "--workers", str(args.workers),
        "--chunksize", str(args.chunksize),
        "--out", str(out_path),
    ]
    return int(LEGACY.main())


if __name__ == "__main__":
    raise SystemExit(main())
