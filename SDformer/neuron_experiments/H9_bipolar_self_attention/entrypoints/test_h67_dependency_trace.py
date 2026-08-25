#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from h67_dependency_trace import TensorDependencyTraceWriter


class ATLIFTernaryPSN(torch.nn.Module):
    """Synthetic logical ATLIF with a child, matching the deployment topology."""

    def __init__(self) -> None:
        super().__init__()
        self.surrogate = torch.nn.ReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.surrogate(value)


class TinyResidual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Linear(4, 4, bias=False)
        self.right = torch.nn.Linear(4, 4, bias=False)
        self.sink = ATLIFTernaryPSN()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scratch = value.clone()
        scratch.add_(1.0)
        joined = self.left(scratch) + self.right(value)
        return self.sink(joined.reshape(1, 2, 4))


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        output = Path(raw)
        writer = TensorDependencyTraceWriter(output, sample_limit=1)
        model = TinyResidual().eval()
        writer.attach(model)
        writer.begin_sample(0, sample_key="tiny", sequence_key="tiny")
        with writer.capture(), torch.no_grad():
            result = model(torch.ones(2, 4))
        writer.end_sample()
        writer.close()
        assert list(result.shape) == [1, 2, 4]
        records = [
            json.loads(line)
            for line in (output / "dependency_events.jsonl").read_text().splitlines()
        ]
        assert any(row["kind"] == "functional_op" and row["name"].startswith("aten.add.") for row in records)
        assert any(row["kind"] == "functional_op" and row["name"].startswith("aten.view.") for row in records)
        assert any(row["kind"] == "persistent_tensor" and row["tensor_kind"] == "parameter" for row in records)
        assert any(
            row["kind"] == "functional_op"
            and row["name"].startswith("aten.add_.")
            and row["mutations"]
            and row["mutations"][0]["version_after"] > row["mutations"][0]["version_before"]
            for row in records
        )
        assert {row["name"] for row in records if row["kind"] == "leaf_module_exit"} == {
            "left", "right", "sink", "sink.surrogate"
        }
        enters = [row for row in records if row["kind"] == "leaf_module_enter"]
        exits = [row for row in records if row["kind"] == "leaf_module_exit"]
        assert len(enters) == len(exits) == 4
        assert any(
            row["name"] == "sink" and row["module_type"] == "ATLIFTernaryPSN"
            for row in enters
        )
        assert min(row["event_index"] for row in records if row["name"] == "sink") < max(
            row["event_index"] for row in records if row["name"] == "sink"
        )
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["schema"] == "h67_tensor_dependency_trace_v2"
        assert manifest["status"] == "PASS_PRE_POST_VERSION_MUTATION_DAG_METADATA_ONLY"
        assert "sink" in manifest["logical_atlif_boundaries"]
        assert manifest["mutation_records"] >= 1
        assert manifest["samples"] == 1
    print("PASS_H67_DEPENDENCY_TRACE_STORAGE_VERSION_FUNCTIONAL_DAG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
