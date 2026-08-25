#!/usr/bin/env python3
"""Record storage/version lineage across module and functional-op boundaries.

The trace is intentionally metadata-only: no tensor values are copied.  It is
used to distinguish a real producer->ATLIF edge from two adjacent hooks with
coincidentally compatible shapes, and to expose functional residual/reshape/
concat nodes that ordinary module hooks omit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _LineageDispatchMode(TorchDispatchMode):
    def __init__(self, writer: "TensorDependencyTraceWriter") -> None:
        super().__init__()
        self.writer = writer

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        actual_kwargs = kwargs or {}
        self.writer.finalize_pending_mutation()
        input_tensors = list(_iter_tensors((args, actual_kwargs)))
        inputs_before = self.writer.snapshot_refs((args, actual_kwargs))
        output = func(*args, **actual_kwargs)
        output_tensors = list(_iter_tensors(output))
        self.writer.record_function(
            str(func), inputs_before, input_tensors,
            self.writer.snapshot_refs(output), output_tensors,
        )
        return output


class TensorDependencyTraceWriter:
    """Collect a bounded metadata DAG for selected samples."""

    FUNCTION_PREFIXES = (
        "aten.add.", "aten.add_.", "aten.sub.", "aten.sub_.",
        "aten.mul.", "aten.mul_.", "aten.cat.", "aten.stack.",
        "aten.view.", "aten.reshape.", "aten._unsafe_view.",
        "aten.permute.", "aten.transpose.", "aten.flatten.",
        "aten.contiguous.", "aten.clone.", "aten.squeeze.",
        "aten.unsqueeze.", "aten.slice.", "aten.select.",
        "aten.alias.", "aten.detach.", "aten.as_strided.", "aten.narrow.",
        "aten.expand.", "aten.repeat.", "aten.repeat_interleave.",
        "aten.split.", "aten.split_with_sizes.", "aten.chunk.",
        "aten.index.", "aten.index_put.", "aten.index_put_.",
        "aten.gather.", "aten.scatter.", "aten.scatter_.",
        "aten.scatter_add.", "aten.where.", "aten.masked_fill.",
        "aten.masked_fill_.", "aten.copy_.", "aten.roll.", "aten.flip.",
        "aten.constant_pad_nd.", "aten.reflection_pad", "aten.replication_pad",
        "aten.upsample_", "aten.pixel_shuffle.",
        "aten.native_batch_norm.", "aten.batch_norm.",
        "aten._native_batch_norm_legit.", "aten.convolution.",
        "aten.mm.", "aten.addmm.", "aten.bmm.", "aten.matmul.",
    )

    def __init__(self, output_dir: Path, *, sample_limit: int = 1) -> None:
        if sample_limit <= 0:
            raise ValueError("dependency trace sample_limit must be positive")
        self.output_dir = Path(output_dir)
        self.sample_limit = int(sample_limit)
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.run_context: dict[str, Any] = {}
        self.current_sample = -1
        self.current_sample_key = ""
        self.current_sequence_key = ""
        self.enabled = False
        self._event_index = 0
        self._module_calls: defaultdict[tuple[int, str], int] = defaultdict(int)
        self._active_module_calls: defaultdict[tuple[int, str], list[int]] = defaultdict(list)
        self.module_inventory: list[dict[str, Any]] = []
        self._pending_mutation: (
            tuple[int, list[torch.Tensor], list[torch.Tensor]] | None
        ) = None
        self._persistent_tensors: list[tuple[str, str, torch.Tensor]] = []

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def bind_run_context(self, value: dict[str, Any]) -> None:
        self.run_context = dict(value)

    def begin_sample(
        self, sample_id: int, *, sample_key: str, sequence_key: str
    ) -> None:
        self.current_sample = int(sample_id)
        self.current_sample_key = sample_key
        self.current_sequence_key = sequence_key
        self.enabled = self.current_sample < self.sample_limit
        if self.enabled:
            for name, tensor_kind, tensor in self._persistent_tensors:
                self._append({
                    "kind": "persistent_tensor",
                    "name": name,
                    "tensor_kind": tensor_kind,
                    "inputs": [],
                    "outputs": [self.tensor_ref(tensor)],
                })

    def end_sample(self) -> None:
        self.finalize_pending_mutation()
        self.enabled = False

    def capture(self) -> contextlib.AbstractContextManager[Any]:
        if not self.enabled:
            return contextlib.nullcontext()
        return _LineageDispatchMode(self)

    @staticmethod
    def tensor_ref(tensor: torch.Tensor) -> dict[str, Any]:
        storage = tensor.untyped_storage()
        return {
            "python_id": int(id(tensor)),
            "storage_cdata": int(storage._cdata),
            "storage_data_ptr": int(storage.data_ptr()),
            "storage_nbytes": int(storage.nbytes()),
            "storage_offset": int(tensor.storage_offset()),
            "shape": [int(value) for value in tensor.shape],
            "stride": [int(value) for value in tensor.stride()],
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "version": int(tensor._version),
        }

    def _refs(self, value: Any) -> list[dict[str, Any]]:
        return [self.tensor_ref(tensor) for tensor in _iter_tensors(value)]

    def snapshot_refs(self, value: Any) -> list[dict[str, Any]]:
        """Snapshot refs before an op can mutate its input version counters."""
        if not self.enabled:
            return []
        return self._refs(value)

    def _append(self, record: dict[str, Any]) -> None:
        record.update({
            "event_index": self._event_index,
            "sample_id": self.current_sample,
            "sample_key": self.current_sample_key,
            "sequence_key": self.current_sequence_key,
        })
        self.records.append(record)
        self._event_index += 1

    def record_function(
        self,
        name: str,
        inputs_before: list[dict[str, Any]],
        input_tensors: list[torch.Tensor],
        outputs: list[dict[str, Any]],
        output_tensors: list[torch.Tensor],
    ) -> None:
        if not self.enabled:
            return
        # Record every tensor-producing dispatcher op.  A prefix whitelist
        # misses source constructors and model-specific indexing/padding paths,
        # leaving live ATLIF operands without a producer.  Metadata volume is
        # bounded by sample_limit; tensor values are never copied.
        if not outputs:
            return
        self._append({
            "kind": "functional_op",
            "name": name,
            "inputs": inputs_before,
            # Version increments for in-place ops are applied by the dispatcher
            # after __torch_dispatch__ returns.  The next dispatch (or
            # end_sample) replaces this provisional snapshot.
            "inputs_after": inputs_before,
            "outputs": outputs,
            "mutations": [],
        })
        self._pending_mutation = (
            len(self.records) - 1, input_tensors, output_tensors
        )

    def finalize_pending_mutation(self) -> None:
        if self._pending_mutation is None:
            return
        record_index, tensors, output_tensors = self._pending_mutation
        after_refs = [self.tensor_ref(tensor) for tensor in tensors]
        record = self.records[record_index]
        record["inputs_after"] = after_refs
        record["outputs"] = [self.tensor_ref(tensor) for tensor in output_tensors]
        mutations = []
        for index, (before, after) in enumerate(zip(record["inputs"], after_refs)):
            if (
                before["storage_cdata"] == after["storage_cdata"]
                and before["version"] != after["version"]
            ):
                mutations.append({
                    "input_index": index,
                    "storage_cdata": before["storage_cdata"],
                    "version_before": before["version"],
                    "version_after": after["version"],
                })
        record["mutations"] = mutations
        self._pending_mutation = None

    def _module_pre_hook(self, name: str, module_type: str):
        def hook(_module: torch.nn.Module, inputs: Any) -> None:
            if not self.enabled:
                return
            key = (self.current_sample, name)
            call_index = self._module_calls[key]
            self._module_calls[key] += 1
            self._active_module_calls[key].append(call_index)
            self._append({
                "kind": "leaf_module_enter",
                "name": name,
                "module_type": module_type,
                "module_call_index": call_index,
                "inputs": self._refs(inputs),
                "outputs": [],
            })

        return hook

    def _module_hook(self, name: str, module_type: str):
        def hook(_module: torch.nn.Module, inputs: Any, output: Any) -> None:
            if not self.enabled:
                return
            key = (self.current_sample, name)
            if not self._active_module_calls[key]:
                raise RuntimeError(f"dependency module exit without enter: {name}")
            call_index = self._active_module_calls[key].pop()
            self._append({
                "kind": "leaf_module_exit",
                "name": name,
                "module_type": module_type,
                "module_call_index": call_index,
                "inputs": self._refs(inputs),
                "outputs": self._refs(output),
            })

        return hook

    def attach(self, model: torch.nn.Module) -> None:
        if self.handles:
            raise RuntimeError("dependency trace writer already attached")
        for name, module in model.named_modules():
            # ATLIFTernaryPSN is a logical hardware boundary even when the
            # training implementation owns a surrogate-function child.  A
            # physical-leaf-only walk silently drops all operator->ATLIF
            # inputs and makes residual/reshape ancestry impossible to prove.
            is_logical_atlif = module.__class__.__name__ == "ATLIFTernaryPSN"
            children = list(module.children())
            is_physical_leaf = not children
            if name:
                self.module_inventory.append({
                    "name": name,
                    "module_type": module.__class__.__name__,
                    "child_count": len(children),
                    "logical_atlif_boundary": is_logical_atlif,
                    "physical_leaf": is_physical_leaf,
                    "hooked": is_logical_atlif or is_physical_leaf,
                })
            if name and (is_logical_atlif or is_physical_leaf):
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._module_pre_hook(name, module.__class__.__name__)
                    )
                )
                self.handles.append(
                    module.register_forward_hook(
                        self._module_hook(name, module.__class__.__name__)
                    )
                )
        persistent_seen: set[int] = set()
        for tensor_kind, iterator in (
            ("parameter", model.named_parameters()),
            ("buffer", model.named_buffers()),
        ):
            for name, tensor in iterator:
                if id(tensor) in persistent_seen:
                    continue
                persistent_seen.add(id(tensor))
                self._persistent_tensors.append((name, tensor_kind, tensor))

    def close(self) -> None:
        self.finalize_pending_mutation()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if any(self._active_module_calls.values()):
            raise RuntimeError("dependency trace closed with active leaf module calls")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.output_dir / "dependency_events.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        by_kind: defaultdict[str, int] = defaultdict(int)
        function_names: defaultdict[str, int] = defaultdict(int)
        mutation_events = 0
        for record in self.records:
            by_kind[record["kind"]] += 1
            if record["kind"] == "functional_op":
                function_names[record["name"]] += 1
                mutation_events += len(record.get("mutations", []))
        manifest = {
            "schema": "h67_tensor_dependency_trace_v2",
            "status": "PASS_PRE_POST_VERSION_MUTATION_DAG_METADATA_ONLY",
            "samples": len({record["sample_id"] for record in self.records}),
            "sample_limit": self.sample_limit,
            "events": len(self.records),
            "events_by_kind": dict(sorted(by_kind.items())),
            "functional_ops": dict(sorted(function_names.items())),
            "functional_capture_policy": "ALL_TENSOR_OUTPUT_DISPATCH_OPS",
            "mutation_records": mutation_events,
            "module_inventory": self.module_inventory,
            "logical_atlif_boundaries": [
                row["name"] for row in self.module_inventory
                if row["logical_atlif_boundary"]
            ],
            "persistent_tensors": len(self._persistent_tensors),
            "dependency_events_sha256": _sha256(trace_path),
            "run_context": self.run_context,
            "claim_boundary": (
                "Pre/post storage-version metadata, mutation records, logical ATLIF inventory, "
                "and selected functional-op edges only; no tensor "
                "values, tile-ready timestamps, causality proof, cycle accuracy, or PPA. "
                "Allocator reuse must be disambiguated with event order during analysis."
            ),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
