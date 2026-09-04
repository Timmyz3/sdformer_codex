#!/usr/bin/env python3
"""Hand-computable admission and fail-close negatives for the M17 analyzer."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints"))
sys.path.insert(0, str(REPO / "hw_autoresearch_nts07/system_simulator/scripts"))

from analyze_m17_full_spatial_c4_oracle import analyze  # noqa: E402
from h67_full_spatial_c4_oracle import H67FullSpatialC4OracleWriter  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(root: Path) -> tuple[Path, Path]:
    oracle = root / "oracle"
    contracts = [{
        "sample_id": 0, "sample_key": "sample-A", "sequence_key": "seq-A",
        "name": f"op{index}", "operator_call_index": 0,
        "atlif_name": f"atlif{index}", "atlif_module_call_index": 0,
    } for index in range(13)]
    writer = H67FullSpatialC4OracleWriter(oracle, allowed_calls=contracts)
    writer.bind_run_context({
        "artifact_identity": {
            "config_sha256": "1" * 64, "checkpoint_sha256": "2" * 64,
            "checkpoint_size": 123,
        },
        "eval_protocol": {"synthetic": "hand-computable"},
        "checkpoint_load_audit": {
            "missing_count": 0, "unexpected_count": 0,
            "overlay_missing_count": 0, "overlay_unexpected_count": 0,
        },
        "dependency_audit_sha256": "3" * 64,
        "dependency_manifest_sha256": "4" * 64,
        "dependency_events_sha256": "5" * 64,
        "source_sha256": {
            "profiler": file_sha256(REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py"),
            "full_spatial_c4_writer": file_sha256(REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints/h67_full_spatial_c4_oracle.py"),
        },
    })
    value = torch.tensor([[[1.0]], [[0.0]]])
    for index in range(13):
        writer.record_operator(
            torch.nn.Linear(1, 1, bias=False), value, name=f"op{index}",
            sample_id=0, sample_key="sample-A", sequence_key="seq-A",
            operator_call_index=0, temporal_steps=2,
        )
    writer.close()
    manifest = json.loads((oracle / "manifest.json").read_text(encoding="utf-8"))
    ledger = root / "ledger.csv"
    fields = [
        "status", "sample_id", "sequence_key", "name", "operator_call_index",
        "temporal_step", "local_work", "selected_work", "selector_rows",
        "motion_selected_rows",
    ]
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        output = csv.DictWriter(handle, fieldnames=fields)
        output.writeheader()
        for operator in manifest["operators"]:
            for timestep in range(2):
                output.writerow({
                    "status": "PASS_EXACT_SOURCE_WORK", "sample_id": 0,
                    "sequence_key": "seq-A", "name": operator["name"],
                    "operator_call_index": 0, "temporal_step": timestep,
                    "local_work": operator["lines"]["local"]["selected_product_terms_by_t"][timestep],
                    "selected_work": operator["lines"]["hybrid"]["selected_product_terms_by_t"][timestep],
                    "selector_rows": 1,
                    "motion_selected_rows": operator["motion_selected_rows_by_t"][timestep],
                })
    return oracle, ledger


def expect_failure(oracle: Path, ledger: Path, phrase: str) -> None:
    try:
        analyze(oracle, ledger)
    except ValueError as exc:
        assert phrase in str(exc), str(exc)
    else:
        raise AssertionError("corrupt M17 fixture was admitted")


def refresh_file_identity(oracle: Path, name: str) -> None:
    manifest_path = oracle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = oracle / name
    manifest["files"][name] = {"sha256": file_sha256(artifact), "bytes": artifact.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_hand_computable_success() -> None:
    with tempfile.TemporaryDirectory() as directory:
        oracle, ledger = build_fixture(Path(directory))
        result = analyze(oracle, ledger)
        # Per call: two descriptors + (3 + 2) compute + two outputs = 9 cycles.
        assert result["summary"]["local_m4_wall_cycles"] == 13 * 9
        assert result["summary"]["hybrid_m4_wall_cycles"] == 13 * 9
        assert result["summary"]["same_width_dense_wall_cycles"] == 13 * 10
        assert not any(result["summary"]["mismatches"].values())


def test_temporal_truncation_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        oracle, ledger = build_fixture(Path(directory))
        manifest_path = oracle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["operators"][0]["lines"]["local"]["selected_product_terms_by_t"].pop()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        expect_failure(oracle, ledger, "exactly 2 temporal")


def test_prototype_cost_source_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        oracle, ledger = build_fixture(Path(directory))
        path = oracle / "prototypes.json"
        prototypes = json.loads(path.read_text(encoding="utf-8"))
        prototypes[0]["local_lane_compute_cycles_by_t"][0] += 1
        path.write_text(json.dumps(prototypes, indent=2) + "\n", encoding="utf-8")
        refresh_file_identity(oracle, "prototypes.json")
        expect_failure(oracle, ledger, "cost-source identity")


def test_stream_population_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        oracle, ledger = build_fixture(Path(directory))
        path = oracle / "ordered_stream.npz"
        with np.load(path, allow_pickle=False) as stream:
            arrays = {name: stream[name].copy() for name in stream.files}
        arrays["population_cluster_id"][0] = 1
        np.savez_compressed(path, **arrays)
        refresh_file_identity(oracle, "ordered_stream.npz")
        expect_failure(oracle, ledger, "population cluster identities")


if __name__ == "__main__":
    test_hand_computable_success()
    test_temporal_truncation_rejected()
    test_prototype_cost_source_rejected()
    test_stream_population_rejected()
    print("PASS m17-full-spatial-analyzer 4/4")
