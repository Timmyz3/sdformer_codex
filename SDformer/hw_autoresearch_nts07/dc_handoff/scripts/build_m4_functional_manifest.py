#!/usr/bin/env python3
"""Fail-closed manifest for the M4 real-trace wall-cycle/VCS milestone."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing evidence: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def assertion_matches(path: Path) -> list[int]:
    text = require_file(path).read_text(encoding="utf-8", errors="replace")
    values = [int(value) for value in re.findall(r"\b(\d+) match\b", text)]
    if len(values) != 5:
        raise ValueError(f"expected five M4 functional covers: {path}")
    return values


REAL_PASS_RE = re.compile(
    r"PASS_M4_DESCRIPTOR_RESIDENT_REAL batches=(\d+) descriptors=(\d+) "
    r"outputs=(\d+) request_beats=(\d+) bank_reads=(\d+) "
    r"output_stalls=(\d+) request_stalls=(\d+) source_checks=(\d+) "
    r"wall_cycles=(\d+) ideal=(\d+)"
)


def real_pass(path: Path) -> tuple[int, ...]:
    text = require_file(path).read_text(encoding="utf-8", errors="replace")
    matches = REAL_PASS_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one M4 real PASS record: {path}")
    return tuple(map(int, matches[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    materialized_dir = run_root / "m4_descriptor_resident_wall_cycles_20260821"
    temporal_dir = run_root / "m4_reducer_dse_temporal_fenced_20260821" / "r4"
    reducer_dir = run_root / "m4_reducer_dse_20260821"
    temporal_reducer_dir = run_root / "m4_reducer_dse_temporal_fenced_20260821"
    vector_dir = run_root / "m4_descriptor_resident_real_vectors_b400_20260821"
    vcs_dir = run_root / "m4_descriptor_resident_vcs_sva_b400_20260821"
    temporal_vector_dir = (
        run_root / "m4_descriptor_resident_real_vectors_temporal_b400_20260821"
    )
    temporal_vcs_dir = (
        run_root / "m4_descriptor_resident_vcs_sva_temporal_b400_20260821"
    )

    materialized = load_json(materialized_dir / "m4_wall_cycles.json")
    temporal = load_json(temporal_dir / "m4_wall_cycles.json")
    vectors = load_json(vector_dir / "manifest.json")
    temporal_vectors = load_json(temporal_vector_dir / "manifest.json")
    reducer = load_json(reducer_dir / "reducer_knee.json")
    temporal_reducer = load_json(temporal_reducer_dir / "reducer_knee.json")
    for label, dse in (("layer_materialized_greedy", materialized), ("temporal_fenced", temporal)):
        if dse.get("status") != "PASS_M4_EXECUTABLE_SINGLE_BUFFER_WALL_CYCLE_MODEL":
            raise ValueError(f"M4 {label} wall-cycle DSE is not admitted")
        if dse.get("architecture", {}).get("availability_mode") != label:
            raise ValueError(f"M4 {label} availability contract is missing")
        if dse.get("architecture", {}).get("output_lanes") != 96:
            raise ValueError(f"M4 {label} functional evidence is not L96")
    if vectors.get("status") != "PASS_CHECKPOINT_BOUND_REAL_BITMAP_DESCRIPTOR_BATCHES":
        raise ValueError("M4 B400 vectors are not admitted")
    if vectors.get("availability_mode") != "layer_materialized_greedy" \
            or not vectors.get("requires_upstream_materialized_activation_rows"):
        raise ValueError("M4 B400 vectors lost the materialized-activation boundary")
    if reducer.get("status") != "PASS_M4_R4_REDUCER_KNEE_SELECTED" \
            or reducer.get("selected_reduce_slots") != 4:
        raise ValueError("M4 R4 functional knee is not admitted")
    if temporal_reducer.get("selected_reduce_slots") not in (2, 4):
        raise ValueError("M4 temporal-fenced reducer sensitivity is invalid")
    population = vectors["population"]
    exact_population = {
        "batches": 400,
        "descriptors": 4880,
        "outputs": 9360,
        "compact_issue_cycles": 92376,
        "chunk_control_cycles": 13280,
        "m4_wall_cycles": 119896,
        "lane_expanded_selected_sources": 653062,
    }
    for field, expected in exact_population.items():
        if population.get(field) != expected:
            raise ValueError(f"M4 B400 {field} changed")
    if len(vectors.get("sample_batches", {})) != 40:
        raise ValueError("M4 B400 lost identity/line/sample coverage")
    if population.get("negative_sources", 0) <= 0 or len(vectors.get("object_tags", {})) < 10:
        raise ValueError("M4 B400 lost signed-Motion or weight-object coverage")
    trace = require_file(vector_dir / "real_descriptors.txt")
    if sha256(trace) != vectors["sha256"]["real_descriptors.txt"]:
        raise ValueError("M4 real descriptor SHA mismatch")
    if temporal_vectors.get("status") != "PASS_CHECKPOINT_BOUND_REAL_BITMAP_DESCRIPTOR_BATCHES" \
            or temporal_vectors.get("availability_mode") != "temporal_fenced" \
            or not temporal_vectors.get("requires_spatial_c4_row_buffer"):
        raise ValueError("M4 temporal-fenced B400 vectors are not admitted")
    temporal_population = temporal_vectors["population"]
    temporal_exact_population = {
        "batches": 400,
        "descriptors": 4880,
        "outputs": 9360,
        "compact_issue_cycles": 111373,
        "chunk_control_cycles": 13280,
        "m4_wall_cycles": 138893,
        "lane_expanded_selected_sources": 1024946,
    }
    for field, expected in temporal_exact_population.items():
        if temporal_population.get(field) != expected:
            raise ValueError(f"M4 temporal B400 {field} changed")
    if len(temporal_vectors.get("sample_batches", {})) != 40 \
            or temporal_population.get("negative_sources", 0) <= 0:
        raise ValueError("M4 temporal B400 coverage is incomplete")
    temporal_trace = require_file(temporal_vector_dir / "real_descriptors.txt")
    if sha256(temporal_trace) != temporal_vectors["sha256"]["real_descriptors.txt"]:
        raise ValueError("M4 temporal real descriptor SHA mismatch")

    expected_ranges = {
        "local": {
            "H67": (5.33, 3.84, 4.68, 3.49),
            "Local5": (5.76, 3.50, 5.48, 3.37),
        },
        "hybrid": {
            "H67": (5.23, 4.00, 4.60, 3.66),
            "Local5": (5.70, 3.63, 5.43, 3.49),
        },
    }
    materialized_performance: dict[str, Any] = {}
    temporal_performance: dict[str, Any] = {}
    for line, identities in expected_ranges.items():
        materialized_performance[line] = {}
        temporal_performance[line] = {}
        for label, lower in identities.items():
            item = materialized["variants"][line]["per_identity"][label]
            values = (
                item["speedup_vs_p1_sparse_wall"],
                item["speedup_vs_same_width_dense_wall"],
                item["p1_sparse_sample_speedup_min"],
                item["same_width_dense_sample_speedup_min"],
            )
            if any(value < threshold for value, threshold in zip(values, lower)):
                raise ValueError(f"M4 wall-cycle regression: {line}/{label} {values}")
            materialized_performance[line][label] = {
                "m4_wall_cycles": item["m4_wall_cycles"],
                "speedup_vs_p1_sparse_wall": values[0],
                "speedup_vs_same_width_dense_wall": values[1],
                "p1_sparse_sample_speedup_min": values[2],
                "same_width_dense_sample_speedup_min": values[3],
                "cross_temporal_batches": item["cross_temporal_batches"],
                "cross_spatial_row_batches": item["cross_spatial_row_batches"],
            }
            fenced_item = temporal["variants"][line]["per_identity"][label]
            if fenced_item.get("availability_mode") != "temporal_fenced" \
                    or fenced_item.get("cross_temporal_batches") != 0 \
                    or fenced_item.get("cross_operator_call_batches") != 0 \
                    or fenced_item.get("cross_sequence_batches") != 0:
                raise ValueError(f"M4 temporal fence failed: {line}/{label}")
            temporal_performance[line][label] = {
                key: fenced_item[key]
                for key in (
                    "m4_wall_cycles",
                    "speedup_vs_p1_sparse_wall",
                    "speedup_vs_same_width_dense_wall",
                    "p1_sparse_sample_speedup_min",
                    "same_width_dense_sample_speedup_min",
                    "cross_temporal_batches",
                    "cross_spatial_row_batches",
                    "cross_operator_call_batches",
                    "cross_sequence_batches",
                    "partial_context_batches",
                    "resident_context_utilization",
                )
            }

    static_log = require_file(vcs_dir / "static" / "simulation.log").read_text(
        encoding="utf-8", errors="replace"
    )
    real_log = require_file(vcs_dir / "real" / "simulation.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "PASS_M4_DESCRIPTOR_RESIDENT outputs=4" not in static_log:
        raise ValueError("M4 directed VCS PASS is absent")
    if (
        "PASS_M4_DESCRIPTOR_RESIDENT_REAL batches=400 descriptors=4880 outputs=9360"
        not in real_log
        or "request_beats=92376 bank_reads=653062" not in real_log
    ):
        raise ValueError("M4 B400 VCS cycle/data PASS is absent")
    greedy_random = real_pass(vcs_dir / "real" / "simulation.log")
    if greedy_random[:5] != (400, 4880, 9360, 92376, 653062) \
            or greedy_random[5] <= 0 or greedy_random[6] <= 0 \
            or greedy_random[7] != greedy_random[4] or greedy_random[9] != 0:
        raise ValueError("M4 greedy random-backpressure/source conservation failed")
    forbidden = re.compile(r"Assertion failed|failed at|Fatal:|^Error:", re.MULTILINE)
    if forbidden.search(static_log) or forbidden.search(real_log):
        raise ValueError("M4 VCS log contains a failure marker")
    static_cover = assertion_matches(vcs_dir / "static" / "assertion_report.txt")
    real_cover = assertion_matches(vcs_dir / "real" / "assertion_report.txt")
    if any(value <= 0 for value in static_cover):
        raise ValueError("M4 directed run did not hit every cover")
    if any(real_cover[index] <= 0 for index in (0, 1, 2, 3)):
        raise ValueError("M4 real run lost wide/negative/lane/output-stall cover")
    ideal_log = require_file(vcs_dir / "ideal" / "simulation.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "wall_cycles=119896 ideal=1" not in ideal_log or forbidden.search(ideal_log):
        raise ValueError("M4 greedy ideal VCS wall-cycle miter failed")
    greedy_ideal = real_pass(vcs_dir / "ideal" / "simulation.log")
    if greedy_ideal[5:10] != (0, 0, 653062, 119896, 1):
        raise ValueError("M4 greedy ideal source/wall conservation failed")
    ideal_cover = assertion_matches(vcs_dir / "ideal" / "assertion_report.txt")
    temporal_static_log = require_file(
        temporal_vcs_dir / "static" / "simulation.log"
    ).read_text(encoding="utf-8", errors="replace")
    temporal_real_log = require_file(
        temporal_vcs_dir / "real" / "simulation.log"
    ).read_text(encoding="utf-8", errors="replace")
    if "PASS_M4_DESCRIPTOR_RESIDENT outputs=4" not in temporal_static_log:
        raise ValueError("M4 temporal directed VCS PASS is absent")
    if (
        "PASS_M4_DESCRIPTOR_RESIDENT_REAL batches=400 descriptors=4880 outputs=9360"
        not in temporal_real_log
        or "request_beats=111373 bank_reads=1024946" not in temporal_real_log
    ):
        raise ValueError("M4 temporal B400 VCS/model PASS is absent")
    temporal_random = real_pass(temporal_vcs_dir / "real" / "simulation.log")
    if temporal_random[:5] != (400, 4880, 9360, 111373, 1024946) \
            or temporal_random[5] <= 0 or temporal_random[6] <= 0 \
            or temporal_random[7] != temporal_random[4] \
            or temporal_random[9] != 0:
        raise ValueError("M4 temporal random-backpressure/source conservation failed")
    if forbidden.search(temporal_static_log) or forbidden.search(temporal_real_log):
        raise ValueError("M4 temporal VCS log contains a failure marker")
    temporal_static_cover = assertion_matches(
        temporal_vcs_dir / "static" / "assertion_report.txt"
    )
    temporal_real_cover = assertion_matches(
        temporal_vcs_dir / "real" / "assertion_report.txt"
    )
    if any(value <= 0 for value in temporal_static_cover) \
            or any(temporal_real_cover[index] <= 0 for index in (0, 1, 2, 3)):
        raise ValueError("M4 temporal VCS lost functional coverage")
    temporal_ideal_log = require_file(
        temporal_vcs_dir / "ideal" / "simulation.log"
    ).read_text(encoding="utf-8", errors="replace")
    if "wall_cycles=138893 ideal=1" not in temporal_ideal_log \
            or forbidden.search(temporal_ideal_log):
        raise ValueError("M4 temporal ideal VCS wall-cycle miter failed")
    temporal_ideal = real_pass(temporal_vcs_dir / "ideal" / "simulation.log")
    if temporal_ideal[5:10] != (0, 0, 1024946, 138893, 1):
        raise ValueError("M4 temporal ideal source/wall conservation failed")
    temporal_ideal_cover = assertion_matches(
        temporal_vcs_dir / "ideal" / "assertion_report.txt"
    )

    source_relpaths = [
        "rtl_qfit/qfit_dual_line_descriptor_resident_engine.sv",
        "verif_qfit/qfit_dual_line_descriptor_resident_engine_assertions.sv",
        "tb_qfit/tb_qfit_dual_line_descriptor_resident_engine.sv",
        "tb_qfit/tb_qfit_dual_line_descriptor_resident_real.sv",
        "system_simulator/scripts/analyze_m4_descriptor_resident_wall_cycles.py",
        "system_simulator/scripts/build_m4_descriptor_resident_real_vectors.py",
        "system_simulator/scripts/summarize_m4_reducer_dse.py",
        "system_simulator/tests/test_m4_descriptor_resident_wall_cycles.py",
        "system_simulator/tests/test_m4_descriptor_resident_real_vectors.py",
        "system_simulator/tests/test_m4_reducer_dse.py",
        "dc_handoff/scripts/build_m4_functional_manifest.py",
        "dc_handoff/scripts/run_vcs_dual_line_descriptor_resident_sva.sh",
        "dc_handoff/filelists/date_dual_line_descriptor_resident.f",
        "dc_handoff/constraints/date_m4_descriptor_resident_pt.sdc",
    ]
    sources = []
    for relative in source_relpaths:
        path = require_file(repo / "hw_autoresearch_nts07" / relative)
        sources.append({"path": str(path), "sha256": sha256(path)})

    evidence_paths = [
        materialized_dir / "m4_wall_cycles.json",
        materialized_dir / "m4_wall_cycles.md",
        reducer_dir / "r2" / "m4_wall_cycles.json",
        reducer_dir / "r8" / "m4_wall_cycles.json",
        reducer_dir / "reducer_knee.json",
        reducer_dir / "reducer_knee.md",
        temporal_reducer_dir / "r2" / "m4_wall_cycles.json",
        temporal_reducer_dir / "r4" / "m4_wall_cycles.json",
        temporal_reducer_dir / "r8" / "m4_wall_cycles.json",
        temporal_reducer_dir / "reducer_knee.json",
        temporal_reducer_dir / "reducer_knee.md",
        vector_dir / "manifest.json",
        vector_dir / "real_descriptors.txt",
        vcs_dir / "evidence.sha256",
        vcs_dir / "static" / "compile_inputs.sha256",
        vcs_dir / "static" / "compile.log",
        vcs_dir / "static" / "simulation.log",
        vcs_dir / "static" / "assertion_report.txt",
        vcs_dir / "real" / "compile_inputs.sha256",
        vcs_dir / "real" / "compile.log",
        vcs_dir / "real" / "simulation.log",
        vcs_dir / "real" / "assertion_report.txt",
        vcs_dir / "ideal" / "simulation.log",
        vcs_dir / "ideal" / "assertion_report.txt",
        temporal_vector_dir / "manifest.json",
        temporal_vector_dir / "real_descriptors.txt",
        temporal_vcs_dir / "evidence.sha256",
        temporal_vcs_dir / "static" / "compile_inputs.sha256",
        temporal_vcs_dir / "static" / "compile.log",
        temporal_vcs_dir / "static" / "simulation.log",
        temporal_vcs_dir / "static" / "assertion_report.txt",
        temporal_vcs_dir / "real" / "compile_inputs.sha256",
        temporal_vcs_dir / "real" / "compile.log",
        temporal_vcs_dir / "real" / "simulation.log",
        temporal_vcs_dir / "real" / "assertion_report.txt",
        temporal_vcs_dir / "ideal" / "simulation.log",
        temporal_vcs_dir / "ideal" / "assertion_report.txt",
    ]
    evidence = [
        {"path": str(require_file(path)), "sha256": sha256(path)}
        for path in evidence_paths
    ]
    payload = {
        "schema": "m4_descriptor_resident_functional_manifest_v1",
        "status": "PASS_M4_REAL_BITMAP_VCS_WALL_CYCLE_PRE_PPA",
        "claim_boundary": (
            "VCS proves functional accumulation and request-beat identity on bounded "
            "temporal-fenced and layer-materialized-greedy cohorts. Ideal-interface VCS "
            "wall cycles equal the executable model on both B400 cohorts; full-trace "
            "results remain modeled source-kernel cycles, not end-to-end network "
            "performance. temporal_fenced uses "
            "only spatial-row coalescing inside one dynamic call/timestep; "
            "layer_materialized_greedy is a non-causal legacy-order sensitivity point, "
            "not an optimized upper bound, and requires a larger upstream activation store."
        ),
        "architecture": {
            key: value
            for key, value in materialized["architecture"].items()
            if key != "availability_mode"
        },
        "performance": {
            "temporal_fenced": temporal_performance,
            "layer_materialized_greedy_sensitivity": materialized_performance,
        },
        "reducer_knee": {
            "layer_materialized_greedy": reducer,
            "temporal_fenced": temporal_reducer,
        },
        "vcs": {
            "b400_population": population,
            "static_cover_matches": static_cover,
            "real_cover_matches": real_cover,
            "ideal_cover_matches": ideal_cover,
            "rtl_request_beats_equal_executable_scheduler": True,
            "rtl_wall_cycles_equal_modeled_wall_cycles": True,
            "random_weight_request_backpressure_covered": True,
            "source_bitmap_conservation_reads": greedy_random[7],
            "availability_mode": vectors["availability_mode"],
            "temporal_fenced_b400_population": temporal_population,
            "temporal_static_cover_matches": temporal_static_cover,
            "temporal_real_cover_matches": temporal_real_cover,
            "temporal_ideal_cover_matches": temporal_ideal_cover,
            "temporal_rtl_request_beats_equal_executable_scheduler": True,
            "temporal_rtl_wall_cycles_equal_modeled_wall_cycles": True,
            "temporal_random_weight_request_backpressure_covered": True,
            "temporal_source_bitmap_conservation_reads": temporal_random[7],
        },
        "remaining_gate": (
            "premacro DC/Formality/PrimeTime is running; SRAM macro, SAIF/PTPX, "
            "DRAMsim3, checkpoint INT8 weights, Motion destination-state integration, "
            "activation release/buffer accounting, and full-network FPS remain outside this PASS"
        ),
        "sources": sources,
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"PASS: wrote {args.output} with {len(sources)} sources and "
        f"{len(evidence)} evidence files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
