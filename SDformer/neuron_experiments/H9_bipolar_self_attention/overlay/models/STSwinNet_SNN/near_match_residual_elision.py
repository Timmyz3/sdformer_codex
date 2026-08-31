"""Evaluation-only near-pattern residual elision for the four PAFT Conv layers.

The production network is left untouched unless this installer is called.
For a 16-source im2col partition with population >= 2, the input is snapped
to its nearest frozen train-only pattern when the Hamming distance is at most
``distance_threshold``.  The original Conv is evaluated first and the exact
floating-point convolution of ``snapped_input - original_input`` is added, so
unsnapped work and bias behavior remain on the production path.
"""

import hashlib
import json
from pathlib import Path
import types
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


_STATE_ATTR = "_m284_near_match_residual_elision_state"
_CATALOG_SCHEMA = "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1"
_OPERATORS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _patterns(operator: Dict[str, Any]) -> torch.Tensor:
    partitions = operator.get("partitions", [])
    _require(len(partitions) == 432, "M284 catalog partition extent drift")
    rows = []
    for partition, row in enumerate(partitions):
        _require(int(row.get("partition", -1)) == partition,
                 "M284 catalog partition order drift")
        values = sorted(int(item["value_hex"], 16)
                        for item in row.get("patterns", []))
        _require(len(values) == 16 and len(set(values)) == 16 and
                 all(0 < value < (1 << 16) for value in values),
                 "M284 catalog value-domain drift")
        rows.append(values)
    packed = torch.tensor(rows, dtype=torch.int64)
    shifts = torch.arange(16, dtype=torch.int64)
    return ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.bool)


def _output_hw(module: torch.nn.Module, height: int, width: int) -> Tuple[int, int]:
    def pair(value):
        return value if isinstance(value, tuple) else (value, value)
    kernel = pair(module.kernel_size)
    stride = pair(module.stride)
    padding = pair(module.padding)
    dilation = pair(module.dilation)
    out_h = ((height + 2 * padding[0] - dilation[0] *
              (kernel[0] - 1) - 1) // stride[0]) + 1
    out_w = ((width + 2 * padding[1] - dilation[1] *
              (kernel[1] - 1) - 1) // stride[1]) + 1
    return int(out_h), int(out_w)


def install_near_match_residual_elision(
        model: torch.nn.Module, spec: Dict[str, Any]) -> List[str]:
    """Install the bounded lossy Conv path and return its four module names."""
    _require(not hasattr(model, _STATE_ATTR),
             "M284 refuses a stale near-match installation")
    threshold = int(spec.get("distance_threshold", -1))
    _require(0 <= threshold <= 4, "M284 threshold outside frozen DSE")
    catalog_path = Path(str(spec.get("catalog_path", ""))).resolve()
    expected_sha = str(spec.get("catalog_sha256", ""))
    _require(catalog_path.is_file() and len(expected_sha) == 64 and
             _sha256(catalog_path) == expected_sha,
             "M284 catalog missing or SHA drift")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    _require(catalog.get("schema") == _CATALOG_SCHEMA,
             "M284 catalog schema drift")
    split = catalog.get("split", {})
    _require(split.get("role") == "DSEC_TRAIN_ONLY_PAFT_CALIBRATION" and
             split.get("train_catalog_eligible") is True and
             split.get("test_or_validation_data_used") is False and
             int(split.get("train_valid825_key_overlap", -1)) == 0,
             "M284 catalog split is not train-only")
    operators = list(catalog.get("operators", []))
    names = tuple(str(row.get("operator")) for row in operators)
    _require(names == _OPERATORS, "M284 operator identity/order drift")
    modules = dict(model.named_modules())
    alpha_delta = tuple(int(value) for value in
                        spec.get("source_alpha_delta_u24", []))
    _require(alpha_delta == (121, 144, 97, 588),
             "M284 source-alpha identity drift")
    partition_chunk = int(spec.get("partition_chunk", 16))
    _require(1 <= partition_chunk <= 54 and 432 % partition_chunk == 0,
             "M284 partition chunk must divide 432")

    state: Dict[str, Any] = {
        "schema": "m284_near_match_residual_elision_runtime_state_v1",
        "distance_threshold": threshold,
        "catalog_path": str(catalog_path),
        "catalog_sha256": expected_sha,
        "operator_names": list(names),
        "partition_chunk": partition_chunk,
        "calls": dict((name, 0) for name in names),
        "partition_vectors": dict((name, 0) for name in names),
        "snapped_partition_vectors": dict((name, 0) for name in names),
        "exact_hit_snapped_partition_vectors": dict((name, 0) for name in names),
        "positive_distance_snapped_partition_vectors": dict(
            (name, 0) for name in names),
        "nearest_distance_histogram": dict((name, [0] * 17) for name in names),
        "handles": [],
    }
    setattr(model, _STATE_ATTR, state)

    for operator_index, (name, operator) in enumerate(zip(names, operators)):
        module = modules.get(name)
        _require(module is not None, "M284 missing module " + name)
        _require(int(module.in_channels) == 768 and
                 int(module.out_channels) == 768 and
                 tuple(module.kernel_size) == (3, 3) and
                 tuple(module.stride) == (1, 1) and
                 tuple(module.padding) == (1, 1) and
                 tuple(module.dilation) == (1, 1) and
                 int(module.groups) == 1,
                 "M284 Conv geometry drift for " + name)
        pattern_cpu = _patterns(operator)
        alpha = float(1.0 - alpha_delta[operator_index] / float(1 << 24))
        original_forward = module.forward

        def approximate_forward(
                this: torch.nn.Module, x: torch.Tensor,
                *, operator_name: str = name,
                frozen_patterns: torch.Tensor = pattern_cpu,
                source_alpha: float = alpha,
                production_forward=original_forward) -> torch.Tensor:
            original = production_forward(x)
            state["calls"][operator_name] += 1
            if threshold == 0:
                return original
            _require(x.dim() in (4, 5),
                     "M284 expected a four/five-dimensional Conv input")
            leading = tuple(int(value) for value in x.shape[:-3])
            channels, height, width = (int(value) for value in x.shape[-3:])
            _require(channels == 768, "M284 input-channel drift")
            x4 = x.reshape(-1, channels, height, width)
            expected_alpha = torch.as_tensor(
                source_alpha, device=x.device, dtype=x.dtype)
            valid_source = torch.logical_or(x4 == 0, x4 == expected_alpha)
            _require(bool(valid_source.all().item()),
                     "M284 input is not exact zero/source-alpha support")
            patches = F.unfold(
                x4, kernel_size=this.kernel_size,
                dilation=this.dilation, padding=this.padding,
                stride=this.stride)
            locations = int(patches.shape[-1])
            grouped = patches.transpose(1, 2).reshape(-1, 432, 16).contiguous()
            patterns = frozen_patterns.to(device=x.device)
            snapped_total = torch.zeros((), device=x.device, dtype=torch.int64)
            exact_hit_total = torch.zeros((), device=x.device,
                                          dtype=torch.int64)
            positive_distance_total = torch.zeros((), device=x.device,
                                                   dtype=torch.int64)
            distance_histogram = torch.zeros(17, device=x.device,
                                             dtype=torch.int64)
            for start in range(0, 432, partition_chunk):
                stop = start + partition_chunk
                active = grouped[:, start:stop, :]
                support = active != 0
                catalog_chunk = patterns[start:stop, :]
                hamming = torch.logical_xor(
                    support.unsqueeze(2), catalog_chunk.unsqueeze(0)).sum(-1)
                distance, nearest = torch.min(hamming, dim=2)
                population = support.sum(-1)
                snap = torch.logical_and(population >= 2,
                                         distance <= threshold)
                flat_catalog = catalog_chunk.reshape(-1, 16)
                offsets = (torch.arange(stop - start, device=x.device)
                           .reshape(1, -1) * 16)
                nearest_bits = flat_catalog[(nearest + offsets).reshape(-1)]
                nearest_bits = nearest_bits.reshape(
                    grouped.shape[0], stop - start, 16)
                active.copy_(torch.where(
                    snap.unsqueeze(-1),
                    nearest_bits.to(dtype=x.dtype) * expected_alpha,
                    active))
                snapped_total += snap.sum()
                exact_hit_total += torch.logical_and(snap, distance == 0).sum()
                positive_distance_total += torch.logical_and(
                    snap, distance > 0).sum()
                distance_histogram += torch.bincount(
                    distance.reshape(-1), minlength=17)
            partition_vectors = int(grouped.shape[0] * 432)
            snapped = int(snapped_total.item())
            exact_hit_snapped = int(exact_hit_total.item())
            positive_distance_snapped = int(positive_distance_total.item())
            _require(snapped == exact_hit_snapped + positive_distance_snapped,
                     "M284 snapped-count decomposition mismatch")
            state["partition_vectors"][operator_name] += partition_vectors
            state["snapped_partition_vectors"][operator_name] += snapped
            state["exact_hit_snapped_partition_vectors"][operator_name] += (
                exact_hit_snapped)
            state["positive_distance_snapped_partition_vectors"][operator_name] += (
                positive_distance_snapped)
            observed_histogram = [int(value) for value in
                                  distance_histogram.cpu().tolist()]
            state_histogram = state["nearest_distance_histogram"][operator_name]
            for index, count in enumerate(observed_histogram):
                state_histogram[index] += count
            delta = grouped.reshape(x4.shape[0], locations, 6912)
            delta = delta.transpose(1, 2).contiguous() - patches
            delta_rows = delta.transpose(1, 2).reshape(-1, 6912)
            correction = F.linear(
                delta_rows, this.weight.reshape(this.out_channels, -1), None)
            out_h, out_w = _output_hw(this, height, width)
            correction = correction.reshape(
                x4.shape[0], out_h * out_w, this.out_channels)
            correction = correction.transpose(1, 2).reshape(
                x4.shape[0], this.out_channels, out_h, out_w)
            correction = correction.reshape(
                *leading, this.out_channels, out_h, out_w)
            _require(tuple(correction.shape) == tuple(original.shape),
                     "M284 correction/output shape drift")
            return original + correction

        module.forward = types.MethodType(approximate_forward, module)
    return list(names)


def near_match_residual_elision_summary(model: torch.nn.Module) -> Dict[str, Any]:
    state = getattr(model, _STATE_ATTR, None)
    _require(state is not None, "M284 summary requested before installation")
    total = sum(int(value) for value in state["partition_vectors"].values())
    snapped = sum(int(value) for value in
                  state["snapped_partition_vectors"].values())
    exact_hit_snapped = sum(int(value) for value in
                            state["exact_hit_snapped_partition_vectors"].values())
    positive_distance_snapped = sum(
        int(value) for value in
        state["positive_distance_snapped_partition_vectors"].values())
    _require(snapped == exact_hit_snapped + positive_distance_snapped,
             "M284 aggregate snapped-count decomposition mismatch")
    return {
        "schema": "m284_near_match_residual_elision_runtime_summary_v1",
        "distance_threshold": state["distance_threshold"],
        "catalog_path": state["catalog_path"],
        "catalog_sha256": state["catalog_sha256"],
        "partition_chunk": state["partition_chunk"],
        "operator_names": list(state["operator_names"]),
        "calls": dict(state["calls"]),
        "partition_vectors": dict(state["partition_vectors"]),
        "snapped_partition_vectors": dict(state["snapped_partition_vectors"]),
        "exact_hit_snapped_partition_vectors": dict(
            state["exact_hit_snapped_partition_vectors"]),
        "positive_distance_snapped_partition_vectors": dict(
            state["positive_distance_snapped_partition_vectors"]),
        "nearest_distance_histogram": dict(state["nearest_distance_histogram"]),
        "aggregate_partition_vectors": total,
        "aggregate_snapped_partition_vectors": snapped,
        "aggregate_exact_hit_snapped_partition_vectors": exact_hit_snapped,
        "aggregate_positive_distance_snapped_partition_vectors":
            positive_distance_snapped,
        "aggregate_snapped_fraction": snapped / float(total) if total else 0.0,
    }
