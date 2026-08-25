#!/usr/bin/env python3
"""Small exact oracles for the full-spatial adjacent-C4 writer."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[3]
        / "neuron_experiments/H9_bipolar_self_attention/entrypoints"
    ),
)

from h67_full_spatial_c4_oracle import (
    H67FullSpatialC4OracleWriter,
    compact_issue_cycles_batched,
    validate_dependency_contract,
)


def allowed_call(name: str) -> list[dict[str, object]]:
    return [{
        "sample_id": 0, "sample_key": "s", "sequence_key": "q", "name": name,
        "operator_call_index": 0, "atlif_name": "synthetic_atlif",
        "atlif_module_call_index": 0,
    }]


def scalar_scheduler(counts: np.ndarray, reduce_slots: int = 4) -> int:
    remaining = counts.astype(np.int64, copy=True)
    cycles = 0
    while np.any(remaining):
        used = np.zeros(remaining.shape[0], dtype=np.int64)
        issued = 0
        for bank in range(remaining.shape[1]):
            for context in range(remaining.shape[0]):
                if remaining[context, bank] > 0 and used[context] < reduce_slots:
                    remaining[context, bank] -= 1
                    used[context] += 1
                    issued += 1
                    break
        assert issued
        cycles += 1
    return cycles


def test_scheduler() -> None:
    rng = np.random.default_rng(7)
    states = rng.integers(0, 8, size=(97, 4, 16), dtype=np.int16)
    expected = np.asarray([scalar_scheduler(state) for state in states])
    actual = compact_issue_cycles_batched(states)
    assert np.array_equal(actual, expected)
    tails = states.copy()
    tails[:, 2:, :] = 0
    expected_tail = np.asarray([scalar_scheduler(state) for state in tails])
    assert np.array_equal(compact_issue_cycles_batched(tails), expected_tail)


def test_linear_writer() -> None:
    module = torch.nn.Linear(17, 100, bias=False)
    value = torch.zeros((2, 5, 17), dtype=torch.float32)
    value[0, 0, [0, 1, 16]] = 1
    value[0, 1, [0, 2, 4, 6]] = 1
    value[0, 3, :] = 1
    value[1] = value[0]
    value[1, 0, 1] = 0
    value[1, 2, [3, 5]] = 1
    with tempfile.TemporaryDirectory() as directory:
        writer = H67FullSpatialC4OracleWriter(Path(directory), allowed_calls=allowed_call("linear"))
        writer.bind_run_context({"synthetic": True})
        writer.record_operator(
            module, value, name="linear", sample_id=0, sample_key="s",
            sequence_key="q", operator_call_index=0, temporal_steps=2,
        )
        writer.close()
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        operator = manifest["operators"][0]
        current = value.eq(1)
        transition = torch.logical_xor(current[1], current[0])
        choose = transition.sum(1) < current[1].sum(1)
        hybrid_t1 = torch.where(choose[:, None], transition, current[1])
        expected_local = [int(current[t].sum()) * 100 for t in range(2)]
        expected_hybrid = [expected_local[0], int(hybrid_t1.sum()) * 100]
        assert operator["lines"]["local"]["selected_product_terms_by_t"] == expected_local
        assert operator["lines"]["hybrid"]["selected_product_terms_by_t"] == expected_hybrid
        assert operator["row_count"] == 5 and operator["population_clusters"] == 2
        stream = np.load(Path(directory) / "ordered_stream.npz")
        assert stream["population_cluster_id"].tolist() == [0, 1]
        assert len(set(stream["prototype_id"].tolist())) == 2


def test_grouped_conv_rows() -> None:
    module = torch.nn.Conv2d(4, 6, kernel_size=3, padding=1, groups=2, bias=False)
    value = torch.zeros((2, 1, 4, 3, 4), dtype=torch.float32)
    value[0, 0, 0, 0, 0] = 1
    value[0, 0, 1, 1, 1] = 1
    value[0, 0, 2, 2, 3] = 1
    value[1] = value[0]
    value[1, 0, 3, 1, 2] = 1
    with tempfile.TemporaryDirectory() as directory:
        writer = H67FullSpatialC4OracleWriter(Path(directory), allowed_calls=allowed_call("conv"))
        rows = writer._conv_rows(value, module, 0)
        assert rows.shape == (24, 18)
        # Ordering is batch, group, output-y, output-x.  Independently rebuild
        # the first output row for each group with explicit padding checks.
        for group, row_index in ((0, 0), (1, 12)):
            expected = []
            for channel in range(group * 2, group * 2 + 2):
                for kernel_y in range(3):
                    for kernel_x in range(3):
                        input_y = -1 + kernel_y
                        input_x = -1 + kernel_x
                        expected.append(
                            0 <= input_y < 3 and 0 <= input_x < 4
                            and bool(value[0, 0, channel, input_y, input_x])
                        )
            assert rows[row_index].tolist() == expected
        writer.bind_run_context({"synthetic": True})
        writer.record_operator(
            module, value, name="conv", sample_id=0, sample_key="s",
            sequence_key="q", operator_call_index=0, temporal_steps=2,
        )
        writer.close()
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        operator = manifest["operators"][0]
        assert operator["row_count"] == 24 and operator["population_clusters"] == 6
        assert operator["fanout"] == 3 and operator["source_width"] == 18


def test_block_invariance_and_conv_tail() -> None:
    module = torch.nn.Conv2d(2, 4, kernel_size=1, groups=1, bias=False)
    value = torch.zeros((2, 1, 2, 2, 5), dtype=torch.float32)
    value[0, 0, 0, :, ::2] = 1
    value[1, 0, 1, :, 1::2] = 1
    results = []
    for block in (1, 17):
        with tempfile.TemporaryDirectory() as directory:
            writer = H67FullSpatialC4OracleWriter(
                Path(directory), allowed_calls=allowed_call("conv_tail"), cluster_block=block,
            )
            clusters, contexts = writer._clusters_for_timestep(value, module, 0)
            # Two y rows, each split as C4+C1; clusters may not cross y.
            assert clusters.shape == (4, 4, 2)
            assert contexts.tolist() == [4, 1, 4, 1]
            writer.bind_run_context({"synthetic": True})
            writer.record_operator(
                module, value, name="conv_tail", sample_id=0, sample_key="s",
                sequence_key="q", operator_call_index=0, temporal_steps=2,
            )
            writer.close()
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            results.append(manifest["operators"][0])
    assert (
        results[0]["ordered_scheduler_sufficient_statistics_sha256"]
        == results[1]["ordered_scheduler_sufficient_statistics_sha256"]
    )
    assert results[0]["lines"] == results[1]["lines"]


def write_dependency_contract_fixture(root: Path) -> tuple[Path, Path, Path, dict, dict, dict]:
    artifact = {
        "config_path": "/old/config.yml", "config_sha256": "1" * 64,
        "checkpoint_path": "/old/checkpoint.pth", "checkpoint_size": 123,
        "checkpoint_mtime_ns": 7, "checkpoint_sha256": "2" * 64,
    }
    evaluation = {"bn_policy": "no_running", "eval_batch_size": 1, "num_workers": 0}
    load_audit = {
        "checkpoint": "/old/checkpoint.pth", "missing_count": 0, "unexpected_count": 0,
        "overlay_missing_count": 0, "overlay_unexpected_count": 0, "remap": "v1",
    }
    events_path = root / "events.jsonl"
    events = [{
        "kind": "leaf_module_enter", "sample_id": 0, "sample_key": "sample-A",
        "sequence_key": "seq-A", "name": f"producer{index}", "module_call_index": 0,
    } for index in range(13)]
    events_path.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")
    manifest_path = root / "dependency_manifest.json"
    manifest = {
        "schema": "h67_tensor_dependency_trace_v2",
        "status": "PASS_PRE_POST_VERSION_MUTATION_DAG_METADATA_ONLY",
        "samples": 1, "sample_limit": 1,
        "dependency_events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
        "run_context": {
            "artifact_identity": artifact, "eval_protocol": evaluation,
            "checkpoint_load_audit": load_audit,
            "source_sha256": {"dependency_writer": "4" * 64},
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audit_path = root / "dependency_audit.json"
    rows = [{
        "sample_id": 0, "sequence_key": "seq-A", "name": f"atlif{index}",
        "module_call_index": 0, "category": "direct_m4", "live": True,
        "admitted_for_overlap": True, "producers": [f"producer{index}"],
    } for index in range(13)]
    audit = {
        "schema": "h67_atlif_dependency_audit_v2",
        "status": "PASS_CAUSAL_DEPENDENCY_CLASSIFICATION",
        "summary": {"admitted_direct_m4_calls": 13}, "rows": rows,
        "identities": {"dependency_manifest": {
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "schema": manifest["schema"], "samples": 1,
            "artifact_identity": artifact, "eval_protocol": evaluation,
            "writer_source_sha256": "4" * 64,
        }},
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return manifest_path, audit_path, events_path, artifact, evaluation, load_audit


def test_dependency_contract_fail_close() -> None:
    with tempfile.TemporaryDirectory() as directory:
        values = write_dependency_contract_fixture(Path(directory))
        manifest, audit, events, artifact, evaluation, load_audit = values
        calls = validate_dependency_contract(
            manifest, audit, events, artifact_identity=artifact,
            eval_protocol=evaluation, checkpoint_load_audit=load_audit,
        )
        assert len(calls) == 13
        assert len({(item["name"], item["operator_call_index"]) for item in calls}) == 13
        wrong_artifact = dict(artifact)
        wrong_artifact["checkpoint_sha256"] = "3" * 64
        try:
            validate_dependency_contract(
                manifest, audit, events, artifact_identity=wrong_artifact,
                eval_protocol=evaluation, checkpoint_load_audit=load_audit,
            )
        except ValueError as exc:
            assert "checkpoint or config identity" in str(exc)
        else:
            raise AssertionError("checkpoint identity drift was admitted")


def test_uncontracted_call_rejected() -> None:
    module = torch.nn.Linear(1, 1, bias=False)
    value = torch.ones((1, 1, 1))
    with tempfile.TemporaryDirectory() as directory:
        writer = H67FullSpatialC4OracleWriter(
            Path(directory), allowed_calls=allowed_call("linear"),
        )
        try:
            writer.record_operator(
                module, value, name="linear", sample_id=0, sample_key="s",
                sequence_key="q", operator_call_index=1, temporal_steps=1,
            )
        except RuntimeError as exc:
            assert "uncontracted call" in str(exc)
        else:
            raise AssertionError("uncontracted producer call was captured")


if __name__ == "__main__":
    test_scheduler()
    test_linear_writer()
    test_grouped_conv_rows()
    test_block_invariance_and_conv_tail()
    test_dependency_contract_fail_close()
    test_uncontracted_call_rejected()
    print("PASS h67-full-spatial-c4-oracle 6/6")
