#!/usr/bin/env python3
"""采集 Local5 的拓扑差分、边项重数与投影 term 工作负载。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


HW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HW_ROOT.parent
EXP_ROOT = REPO_ROOT / "neuron_experiments" / "H9_bipolar_self_attention"
ENTRYPOINT_ROOT = EXP_ROOT / "entrypoints"
sys.path.insert(0, str(ENTRYPOINT_ROOT))

import profile_nts11_hardware_p0 as base_profile  # noqa: E402
from h67_bit_trace import quantize_projection_weight_dyadic  # noqa: E402
from et3_ordered_trace_replay import (  # noqa: E402
    canonical_item_hash,
    deployment_contract_from_config,
)


DIRECTIONS = ("self", "up", "down", "left", "right")
GATE_CODES = 257
POST_G0_SAMPLES = 100
POST_G0_BLOCKS = 12
POST_G0_TOKENS = 450
POST_G0_LANES = 32
POST_G0_WINDOW = [2, 15, 15]
POST_G0_SAMPLING_ID = "coprime_rotating_flat_window_head_v1"
POST_G0_DATASET_SAMPLING_ID = "sequence_proportional_temporal_midpoint_v1"
POST_G0_BLOCK_PAIRS = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 0),
    (3, 1),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def string_list_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(values) + "\n").encode("utf-8")
    ).hexdigest()


def int_list_sha256(values: list[int]) -> str:
    return hashlib.sha256(
        ("\n".join(str(value) for value in values) + "\n").encode("utf-8")
    ).hexdigest()


def file_row_values(file_row: Any) -> list[str]:
    if isinstance(file_row, (list, tuple)):
        return [str(item) for item in file_row]
    return [str(file_row)]


def sequence_key_from_file_row(file_row: Any) -> str:
    return "|".join(
        "_".join(Path(item).stem.split("_")[:-1])
        for item in file_row_values(file_row)
    )


def stratified_dataset_indices(files: Any, samples: int) -> list[int]:
    """按sequence比例分配，并在每个sequence内取等间隔中点。"""

    total = len(files)
    if samples <= 0 or samples > total:
        raise ValueError("数据集分层抽样数量越界")
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for index in range(total):
        key = sequence_key_from_file_row(files[index])
        if not key:
            raise ValueError(f"dataset index {index}缺少sequence key")
        by_sequence[key].append(index)
    keys = sorted(by_sequence)
    quotas = {
        key: samples * len(by_sequence[key]) / total for key in keys
    }
    allocations = {
        key: min(len(by_sequence[key]), math.floor(quotas[key]))
        for key in keys
    }
    remaining = samples - sum(allocations.values())
    remainder_order = sorted(
        keys,
        key=lambda key: (-(quotas[key] - math.floor(quotas[key])), key),
    )
    while remaining:
        progressed = False
        for key in remainder_order:
            if allocations[key] < len(by_sequence[key]):
                allocations[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("数据集分层抽样无法完成配额")
    if samples >= len(keys):
        for missing in [key for key in keys if allocations[key] == 0]:
            donors = [key for key in keys if allocations[key] > 1]
            if not donors:
                raise ValueError("无法保证每个sequence至少一个样本")
            donor = max(donors, key=lambda key: (allocations[key], key))
            allocations[donor] -= 1
            allocations[missing] = 1
    selected: list[int] = []
    for key in keys:
        rows = by_sequence[key]
        count = allocations[key]
        selected.extend(
            rows[((2 * slot + 1) * len(rows)) // (2 * count)]
            for slot in range(count)
        )
    selected.sort()
    if len(selected) != samples or len(set(selected)) != samples:
        raise ValueError("数据集分层抽样索引不守恒")
    return selected


def expected_source_binding_paths() -> dict[str, Path]:
    baseline = REPO_ROOT / "third_party/SDformerFlow"
    overlay = EXP_ROOT / "overlay/models/STSwinNet_SNN"
    return {
        "watcher": (
            HW_ROOT / "scripts/run_local5_qfsa_profile_after_fullres.py"
        ),
        "profiler": Path(__file__).resolve(),
        "base_profiler": (
            EXP_ROOT / "entrypoints/profile_nts11_hardware_p0.py"
        ),
        "attention_impl": overlay / "bsa_attention.py",
        "checkpoint_loader": overlay / "h9_load_audit.py",
        "model_impl": (
            baseline
            / "models/STSwinNet_SNN/Spiking_STSwinNet.py"
        ),
        "dataset_impl": (
            baseline / "DSEC_dataloader/DSEC_dataset_lite.py"
        ),
        "trace_loader": HW_ROOT / "scripts/et3_ordered_trace_replay.py",
        "replay": HW_ROOT / "scripts/replay_local5_frontier_trace.py",
        "descriptor_analyzer": (
            HW_ROOT / "scripts/analyze_ds_flm_descriptor_manifest.py"
        ),
        "acceptance": (
            HW_ROOT / "scripts/validate_local5_postg0_acceptance.py"
        ),
        "relation_transpose_rtl": (
            HW_ROOT / "rtl_qfit/qfit_relation_transpose_leaf.sv"
        ),
        "retirement_scheduler_rtl": (
            HW_ROOT / "rtl_qfit/qfit_retirement_scheduler.sv"
        ),
        "relation_sync_bank_rtl": (
            HW_ROOT / "rtl_qfit/qfit_sync_1r1w_bank.sv"
        ),
        "relation_assertions": (
            HW_ROOT / "verif_qfit/qfit_relation_transpose_assertions.sv"
        ),
        "relation_sync_bank_assertions": (
            HW_ROOT / "verif_qfit/qfit_sync_bank_assertions.sv"
        ),
        "relation_vector_generator": (
            HW_ROOT / "scripts/generate_local5_relation_transpose_vectors.py"
        ),
        "relation_miter_tb": (
            HW_ROOT / "tb_qfit/tb_qfit_relation_transpose_python_miter.sv"
        ),
        "relation_miter_script": (
            HW_ROOT / "sim_qfit/run_qfit_relation_transpose_python_miter.sh"
        ),
        "score_trace_generator": (
            HW_ROOT / "scripts/generate_local5_checkpoint_score_vectors.py"
        ),
        "score_trace_reporter": (
            HW_ROOT / "scripts/report_local5_checkpoint_score_rtl.py"
        ),
        "score_trace_tb": (
            HW_ROOT / "tb_local5/tb_local5_score_shiftmax_vectors.sv"
        ),
        "score_trace_script": (
            HW_ROOT / "sim_local5/run_local5_checkpoint_score_trace_checks.sh"
        ),
        "projection_quantizer": ENTRYPOINT_ROOT / "h67_bit_trace.py",
        # Must stay aligned with run_local5_qfsa_profile_after_fullres.source_binding_paths().
        "projection_contract_verifier": (
            HW_ROOT / "scripts/verify_local5_theta_folded_projection_contract.py"
        ),
        "projection_trace_generator": (
            HW_ROOT / "scripts/generate_local5_active_projection_postg0_vectors.py"
        ),
        "projection_trace_reporter": (
            HW_ROOT / "scripts/summarize_local5_gasr2c_fivebank_rtl.py"
        ),
        "projection_trace_tb": (
            HW_ROOT / "tb_qfit/tb_qfit_local5_active_projection_postg0.sv"
        ),
        "projection_trace_script": (
            HW_ROOT / "sim_new_arch/run_local5_qgasr2c_fivebank_checks.sh"
        ),
    }


def write_checkpoint_projection_contract(
    model: torch.nn.Module,
    *,
    output_dir: Path,
    checkpoint: Path,
    bn_policy: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Export theta-folded Local5 projection weights for binary K events."""

    arrays: dict[str, np.ndarray] = {}
    blocks: list[dict[str, Any]] = []
    observed: set[tuple[int, int]] = set()
    total_weight_entries = 0
    total_weight_mismatch = 0
    total_scale_entries = 0
    total_scale_mismatch = 0
    for full_name, module in model.named_modules():
        cfg = getattr(module, "_h9_shiftmax_cfg", None)
        mode = str(getattr(cfg, "mode", ""))
        if "local5" not in mode and mode not in {"lr_ttx", "h66_lr"}:
            continue
        match = re.search(
            r"layers\.(\d+)\.swin_blocks\.(\d+)\.attn$", full_name
        )
        if match is None:
            continue
        stage, block = map(int, match.groups())
        pair = (stage, block)
        if pair not in set(POST_G0_BLOCK_PAIRS):
            raise ValueError(f"projection contract出现非目标block: {pair}")
        if pair in observed:
            raise ValueError(f"projection contract重复block: {pair}")
        projection = getattr(module, "proj", None)
        weight = getattr(projection, "weight", None)
        if weight is None or weight.ndim != 2:
            raise ValueError(f"{full_name}缺少二维proj.weight")
        spiking_neuron = getattr(getattr(module, "sn_k", None), "spiking_neuron", None)
        theta_tensor = getattr(spiking_neuron, "thresh", None)
        if theta_tensor is None:
            raise ValueError(f"{full_name}缺少K-ATLIF theta")
        theta_flat = theta_tensor.detach().float().cpu().reshape(-1)
        if theta_flat.numel() != 1:
            raise ValueError(f"{full_name}的K-ATLIF theta不是标量")
        theta = float(theta_flat.item())
        if not np.isfinite(theta) or theta <= 0.0:
            raise ValueError(f"{full_name}的K-ATLIF theta非法: {theta}")

        raw_weight = weight.detach().float().cpu()
        effective_weight = raw_weight * theta
        raw_int8, raw_scale_exp2 = quantize_projection_weight_dyadic(raw_weight)
        weight_int8, scale_exp2 = quantize_projection_weight_dyadic(effective_weight)
        weight_mismatch = int(np.count_nonzero(raw_int8 != weight_int8))
        scale_mismatch = int(np.count_nonzero(raw_scale_exp2 != scale_exp2))
        total_weight_entries += int(raw_int8.size)
        total_weight_mismatch += weight_mismatch
        total_scale_entries += int(raw_scale_exp2.size)
        total_scale_mismatch += scale_mismatch
        bias = getattr(projection, "bias", None)
        bias_float = (
            np.zeros(weight.shape[0], dtype=np.float32)
            if bias is None
            else bias.detach().float().cpu().numpy()
        )
        prefix = f"s{stage}_b{block}"
        arrays[f"{prefix}_theta_float32"] = np.asarray([theta], dtype=np.float32)
        arrays[f"{prefix}_weight_float32"] = raw_weight.numpy()
        arrays[f"{prefix}_effective_weight_float32"] = effective_weight.numpy()
        arrays[f"{prefix}_weight_int8"] = weight_int8
        arrays[f"{prefix}_weight_scale_exp2"] = scale_exp2
        arrays[f"{prefix}_bias_float32"] = bias_float
        blocks.append(
            {
                "stage": stage,
                "block": block,
                "module": full_name,
                "prefix": prefix,
                "weight_name": f"{full_name}.proj.weight",
                "theta_name": f"{full_name}.sn_k.spiking_neuron.thresh",
                "bias_name": f"{full_name}.proj.bias",
                "theta": theta,
                "weight_shape": list(raw_weight.shape),
                "heads": int(getattr(module, "num_heads")),
                "head_dim": int(weight.shape[1] // getattr(module, "num_heads")),
                "bias_present": bias is not None,
                "weight_scale_exp2_min": int(scale_exp2.min()),
                "weight_scale_exp2_max": int(scale_exp2.max()),
                "raw_vs_folded_weight_int8_mismatch": weight_mismatch,
                "raw_vs_folded_scale_exp2_mismatch": scale_mismatch,
            }
        )
        observed.add(pair)
    if observed != set(POST_G0_BLOCK_PAIRS):
        raise ValueError(
            "projection contract block集合不完整: "
            f"missing={sorted(set(POST_G0_BLOCK_PAIRS) - observed)}"
        )
    blocks.sort(key=lambda row: (row["stage"], row["block"]))
    if any(row["head_dim"] != POST_G0_LANES for row in blocks):
        raise ValueError("projection contract要求所有block head_dim=32")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "checkpoint_projection_contract.npz"
    np.savez_compressed(payload_path, **arrays)
    manifest = {
        "schema": "local5_checkpoint_projection_contract_v2",
        "status": "THETA_FOLDED_WEIGHT_CONTRACT",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "payload_file": payload_path.name,
        "payload_sha256": file_sha256(payload_path),
        "topology_contract": (
            "local5_swin_2_2_6_2_c96_192_384_768_h3_6_12_24_v1"
        ),
        "blocks": blocks,
        "value_contract": (
            "V=K_binary_event*theta_K(block); theta_K is folded into "
            "projection W before dyadic INT8 quantization"
        ),
        "quantization_order": "W_eff=theta_K*W_float; quantize_dyadic_int8(W_eff)",
        "quantization": (
            "per-output-channel symmetric INT8; scale=2^e; "
            "e=ceil(log2(max_abs/127)); RNE; clamp[-127,127]"
        ),
        "numeric_scope": (
            "theta-folded checkpoint Linear projection weight/bias; RTL replay "
            "uses per-head INT8 partial accumulators before cross-head reduction, "
            "bias, BatchNorm, requantization, residual, or decoder"
        ),
        "runtime_datapath": (
            "K remains a 1-bit event; no runtime theta multiplier or event-width increase"
        ),
        "raw_vs_folded": {
            "weight_int8_mismatch": total_weight_mismatch,
            "weight_int8_entries": total_weight_entries,
            "scale_exp2_mismatch": total_scale_mismatch,
            "scale_exp2_entries": total_scale_entries,
        },
        "bn_policy": bn_policy,
        "bn_folding": (
            "not_permitted: no_running evaluation uses input-dependent batch "
            "statistics"
            if bn_policy == "no_running"
            else "not_included"
        ),
        "source_sha256": {
            "profiler": file_sha256(Path(__file__).resolve()),
            "projection_quantizer": file_sha256(
                ENTRYPOINT_ROOT / "h67_bit_trace.py"
            ),
        },
    }
    manifest_path = output_dir / "checkpoint_projection_contract.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path, payload_path, manifest


def validate_release_receipt(
    receipt_path: Path,
    expected_hash: str,
) -> dict[str, Any]:
    if (
        not receipt_path.is_file()
        or file_sha256(receipt_path) != expected_hash
    ):
        raise ValueError("post_g0 release receipt SHA256绑定失效")
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    status_path = Path(str(value.get("status_path", ""))).resolve()
    status = status_path.read_bytes()
    prefix_bytes = int(value.get("status_prefix_bytes", -1))
    marker_start = int(value.get("marker_start_offset", -1))
    marker_end = int(value.get("marker_end_offset", -1))
    marker_line = str(value.get("marker_line", ""))
    if (
        value.get("schema") != "local5_release_receipt_v2"
        or value.get("release_marker")
        != "ALL COMPLETE fullres deploy followup"
        or not 0 <= prefix_bytes <= marker_start < marker_end <= len(status)
        or hashlib.sha256(status[:prefix_bytes]).hexdigest()
        != value.get("status_prefix_sha256")
        or status[marker_start:marker_end].decode(
            "utf-8", errors="strict"
        ).rstrip("\n") != marker_line
        or "ALL COMPLETE fullres deploy followup" not in marker_line
        or "H67" not in marker_line
        or "H66d" not in marker_line
    ):
        raise ValueError("post_g0 release receipt内容无效")
    return value


def rotating_flat_indices(
    *,
    total_groups: int,
    selected_groups: int,
    sample_id: int,
    stage: int,
    block: int,
) -> list[int]:
    """以互素步长轮转抽样，避免每个sample固定抽同一head/window。"""

    if total_groups <= 0 or selected_groups <= 0:
        return []
    selected = min(total_groups, selected_groups)
    step = max(1, total_groups // 2)
    while math.gcd(step, total_groups) != 1:
        step += 1
    offset = (stage * 131 + block * 17) % total_groups
    return [
        (
            offset
            + (sample_id * selected + slot) * step
        )
        % total_groups
        for slot in range(selected)
    ]


def histogram(values: torch.Tensor, *, minlength: int = 0) -> list[int]:
    if values.numel() == 0:
        return [0] * minlength
    return (
        torch.bincount(values.reshape(-1).to(dtype=torch.long), minlength=minlength)
        .cpu()
        .tolist()
    )


def merge_histograms(rows: list[dict[str, Any]], key: str) -> list[int]:
    size = max((len(row.get(key, [])) for row in rows), default=0)
    result = [0] * size
    for row in rows:
        for index, value in enumerate(row.get(key, [])):
            result[index] += int(value)
    return result


def hist_quantile(hist: list[int], probability: float) -> int:
    total = sum(hist)
    if total == 0:
        return 0
    target = max(1, int((total * probability) + 0.999999999))
    running = 0
    for value, count in enumerate(hist):
        running += int(count)
        if running >= target:
            return value
    return max(0, len(hist) - 1)


def hist_mean(hist: list[int]) -> float:
    total = sum(hist)
    return (
        sum(value * int(count) for value, count in enumerate(hist)) / total
        if total
        else 0.0
    )


def bitcount_integer(value: int) -> int:
    return int(value).bit_count()


def mfep_destination_stream_stats(
    term_keys: torch.Tensor,
    destination_ids: torch.Tensor,
    *,
    tokens: int,
) -> dict[str, Any]:
    """统计精确 term destination stream，不从 fanout 聚合值反推顺序。"""

    keys = term_keys.reshape(-1).to(dtype=torch.long)
    destinations = destination_ids.reshape(-1).to(dtype=torch.long)
    if keys.numel() != destinations.numel():
        raise ValueError("term_keys 与 destination_ids 数量不一致")
    if destinations.numel() and (
        int(destinations.min().item()) < 0
        or int(destinations.max().item()) >= tokens
    ):
        raise ValueError("destination 超出 token 范围")
    if keys.numel() == 0:
        return {
            "mfep_scalar_delivery": 0,
            "mfep_ppdi_delivery_exact": 0,
            "mfep_ppdi_command_reduction": 0.0,
            "mfep_destination_continuations": 0,
            "mfep_destination_delta_histogram": [0] * tokens,
            **{
                f"mfep_destination_delta_escape_b{bits}": 0
                for bits in (4, 6, 10)
            },
            **{
                f"mfep_destination_delta_escape_ratio_b{bits}": 0.0
                for bits in (4, 6, 10)
            },
        }

    # 组合键排序后，同一 term 的 destination 单调递增。
    combined = keys * tokens + destinations
    combined = torch.unique(combined, sorted=True)
    sorted_terms = torch.div(combined, tokens, rounding_mode="floor")
    sorted_destinations = torch.remainder(combined, tokens)
    _, compact_term = torch.unique_consecutive(
        sorted_terms,
        return_inverse=True,
    )
    term_count = int(compact_term.max().item()) + 1
    even = torch.zeros(term_count, dtype=torch.long, device=keys.device)
    odd = torch.zeros_like(even)
    even.scatter_add_(
        0,
        compact_term,
        sorted_destinations.bitwise_and(1).eq(0).to(torch.long),
    )
    odd.scatter_add_(
        0,
        compact_term,
        sorted_destinations.bitwise_and(1).eq(1).to(torch.long),
    )
    ppdi_delivery = int(torch.maximum(even, odd).sum().item())

    same_term = sorted_terms[1:].eq(sorted_terms[:-1])
    deltas = (
        sorted_destinations[1:] - sorted_destinations[:-1]
    )[same_term]
    delta_histogram = histogram(deltas, minlength=tokens)
    continuations = int(deltas.numel())
    result: dict[str, Any] = {
        "mfep_scalar_delivery": int(combined.numel()),
        "mfep_ppdi_delivery_exact": ppdi_delivery,
        "mfep_ppdi_command_reduction": (
            1.0 - ppdi_delivery / int(combined.numel())
        ),
        "mfep_destination_continuations": continuations,
        "mfep_destination_delta_histogram": delta_histogram,
    }
    for bits in (4, 6, 10):
        escapes = int(deltas.gt((1 << bits) - 1).sum().item())
        result[f"mfep_destination_delta_escape_b{bits}"] = escapes
        result[f"mfep_destination_delta_escape_ratio_b{bits}"] = (
            escapes / continuations if continuations else 0.0
        )
    return result


def source_gate_lane_stats(
    active: torch.Tensor,
    gate_code: torch.Tensor,
    neighbor_index: torch.Tensor,
) -> dict[str, Any]:
    """按 source token 重排 edge，统计 DiSEP source-gate-lane term。"""

    batch_windows, heads, tokens, candidates, lanes = active.shape
    if tuple(gate_code.shape) != (
        batch_windows,
        heads,
        tokens,
        candidates,
    ):
        raise ValueError("gate_code shape 与 active 不一致")
    if tuple(neighbor_index.shape) != (tokens, candidates):
        raise ValueError("neighbor_index shape 与 Local5 edge 不一致")

    bh = (
        torch.arange(
            batch_windows * heads,
            device=active.device,
            dtype=torch.long,
        )
        .reshape(batch_windows, heads, 1, 1, 1)
        .expand_as(active)
    )
    source = (
        neighbor_index.to(device=active.device, dtype=torch.long)
        .view(1, 1, tokens, candidates, 1)
        .expand_as(active)
    )
    gate = gate_code.unsqueeze(-1).expand_as(active).to(dtype=torch.long)
    lane = (
        torch.arange(lanes, device=active.device, dtype=torch.long)
        .view(1, 1, 1, 1, lanes)
        .expand_as(active)
    )
    key = (((bh * tokens + source) * GATE_CODES + gate) * lanes) + lane
    selected_keys = key[active]
    unique_terms, fanout = torch.unique(
        selected_keys,
        sorted=True,
        return_counts=True,
    )
    keyspace_per_group = tokens * GATE_CODES * lanes
    terms_per_group = torch.bincount(
        torch.div(unique_terms, keyspace_per_group, rounding_mode="floor"),
        minlength=batch_windows * heads,
    )

    active_edge = active.any(dim=-1)
    bh_edge = bh[..., 0]
    source_edge = source[..., 0]
    source_gate_key = (
        (bh_edge * tokens + source_edge) * GATE_CODES
        + gate_code.to(dtype=torch.long)
    )
    unique_source_gates = torch.unique(source_gate_key[active_edge], sorted=True)
    source_instance = torch.div(
        unique_source_gates,
        GATE_CODES,
        rounding_mode="floor",
    )
    gate_count_per_source = torch.bincount(
        source_instance,
        minlength=batch_windows * heads * tokens,
    )
    active_source_gate_count = gate_count_per_source[
        gate_count_per_source.gt(0)
    ]
    result = {
        "source_gate_lane_terms": int(unique_terms.numel()),
        "source_gate_lane_delivery": int(selected_keys.numel()),
        "source_instances": int(gate_count_per_source.numel()),
        "source_active_instances": int(active_source_gate_count.numel()),
        "source_gate_lane_max_fanout": (
            int(fanout.max().item()) if fanout.numel() else 0
        ),
        "source_gate_lane_fanout_histogram": histogram(fanout, minlength=1),
        "source_gate_lane_terms_per_window_head_histogram": histogram(
            terms_per_group
        ),
        "source_gate_cardinality_histogram": histogram(
            active_source_gate_count,
            minlength=6,
        ),
        "source_gate_cardinality_all_histogram": histogram(
            gate_count_per_source,
            minlength=6,
        ),
    }

    plane_tokens = tokens
    time_planes = 1
    side = math.isqrt(tokens)
    if side * side != tokens and tokens % 2 == 0:
        candidate_plane_tokens = tokens // 2
        candidate_side = math.isqrt(candidate_plane_tokens)
        if candidate_side * candidate_side == candidate_plane_tokens:
            plane_tokens = candidate_plane_tokens
            time_planes = 2
            side = candidate_side
    if side * side != plane_tokens:
        result["dqfs_layout_supported"] = 0
        return result

    term_lane = unique_terms % lanes
    term_without_lane = torch.div(
        unique_terms, lanes, rounding_mode="floor"
    )
    term_gate = term_without_lane % GATE_CODES
    term_source_group = torch.div(
        term_without_lane, GATE_CODES, rounding_mode="floor"
    )
    term_source = term_source_group % tokens
    term_bh = torch.div(
        term_source_group, tokens, rounding_mode="floor"
    )
    term_plane = torch.div(
        term_source, plane_tokens, rounding_mode="floor"
    )
    term_spatial = term_source % plane_tokens
    term_row = torch.div(term_spatial, side, rounding_mode="floor")
    row_groups = batch_windows * heads * time_planes * side
    row_group = (
        (term_bh * time_planes + term_plane) * side + term_row
    )
    row_value_key = (
        (row_group * lanes + term_lane) * GATE_CODES + term_gate
    )
    unique_row_values, chain_lengths = torch.unique(
        row_value_key,
        sorted=True,
        return_counts=True,
    )
    row_lane = torch.div(
        unique_row_values, GATE_CODES, rounding_mode="floor"
    )
    value_row_group = torch.div(
        row_lane, lanes, rounding_mode="floor"
    )
    row_value_count = torch.bincount(
        value_row_group,
        minlength=row_groups,
    )
    row_term_count = torch.bincount(
        row_group,
        minlength=row_groups,
    )
    row_lane_gate_count = torch.bincount(
        row_lane,
        minlength=row_groups * lanes,
    )
    row_lane_term_count = torch.bincount(
        row_group * lanes + term_lane,
        minlength=row_groups * lanes,
    )
    active_row_lane_gate_count = row_lane_gate_count[
        row_lane_gate_count.gt(0)
    ]
    result.update(
        {
            "dqfs_layout_supported": 1,
            "dqfs_row_groups": row_groups,
            "dqfs_row_value_product_computes": int(
                unique_row_values.numel()
            ),
            "dqfs_row_value_key_histogram": histogram(
                row_value_count
            ),
            "dqfs_row_term_histogram": histogram(row_term_count),
            "dqfs_row_lane_gate_cardinality_histogram": histogram(
                active_row_lane_gate_count,
                minlength=1,
            ),
            "dqfs_value_chain_length_histogram": histogram(
                chain_lengths,
                minlength=1,
            ),
        }
    )
    for ways in (2, 4, 6, 8):
        overflow = row_lane_gate_count.gt(ways)
        result[f"dqfs_lane_way_overflow_groups_w{ways}"] = int(
            overflow.sum().item()
        )
        result[f"dqfs_lane_way_overflow_terms_w{ways}"] = int(
            row_lane_term_count[overflow].sum().item()
        )
    return result


def joint_stencil_route(
    counts: torch.Tensor,
    residual_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回最小周期、direct mask、direct周期和residual wave。"""

    if counts.shape[-1] != 4:
        raise ValueError("joint stencil counts最后一维必须为4")
    if residual_width <= 0:
        raise ValueError("residual_width必须为正")
    active = counts.gt(0)
    choices = []
    direct_choices = []
    residual_choices = []
    for direct_mask in range(16):
        selected = torch.tensor(
            [
                bool((direct_mask >> direction) & 1)
                for direction in range(4)
            ],
            device=counts.device,
            dtype=torch.bool,
        )
        direct_cycles = (
            active & selected.view(*([1] * (counts.ndim - 1)), 4)
        ).sum(dim=-1)
        residual_events = (
            counts
            * (~selected)
            .view(*([1] * (counts.ndim - 1)), 4)
            .to(dtype=torch.long)
        ).sum(dim=-1)
        residual_cycles = torch.div(
            residual_events + residual_width - 1,
            residual_width,
            rounding_mode="floor",
        )
        # direct fallback与tagged residual backend是独立资源；anchor结束后并行。
        choices.append(torch.maximum(direct_cycles, residual_cycles))
        direct_choices.append(direct_cycles)
        residual_choices.append(residual_cycles)
    stacked = torch.stack(choices, dim=-1)
    cycles, direct_mask = stacked.min(dim=-1)
    direct_stack = torch.stack(direct_choices, dim=-1)
    residual_stack = torch.stack(residual_choices, dim=-1)
    selected_direct = torch.gather(
        direct_stack,
        -1,
        direct_mask.unsqueeze(-1),
    ).squeeze(-1)
    selected_residual = torch.gather(
        residual_stack,
        -1,
        direct_mask.unsqueeze(-1),
    ).squeeze(-1)
    return cycles, direct_mask, selected_direct, selected_residual


def joint_stencil_min_extra_cycles(
    counts: torch.Tensor,
    residual_width: int,
) -> torch.Tensor:
    return joint_stencil_route(counts, residual_width)[0]


def independent_direction_min_extra_cycles(
    counts: torch.Tensor,
) -> torch.Tensor:
    """同总W4的4xW1方向独立residual与共享direct engine基线。"""

    if counts.shape[-1] != 4:
        raise ValueError("independent direction counts最后一维必须为4")
    active = counts.gt(0)
    choices = []
    for direct_mask in range(16):
        selected = torch.tensor(
            [
                bool((direct_mask >> direction) & 1)
                for direction in range(4)
            ],
            device=counts.device,
            dtype=torch.bool,
        )
        direct_cycles = (
            active & selected.view(*([1] * (counts.ndim - 1)), 4)
        ).sum(dim=-1)
        residual_counts = (
            counts
            * (~selected)
            .view(*([1] * (counts.ndim - 1)), 4)
            .to(dtype=torch.long)
        )
        # 四个W1方向lane并行，各自不跨方向借用空闲lane。
        residual_cycles = residual_counts.max(dim=-1).values
        # 4xW1基线使用相同的独立direct/residual资源合同。
        choices.append(torch.maximum(direct_cycles, residual_cycles))
    return torch.stack(choices, dim=-1).min(dim=-1).values


def xorbank_stencil_route(
    k_candidates: torch.Tensor,
    valid: torch.Tensor,
    *,
    threshold: int | None = None,
    bank_pressure_threshold: int | None = None,
    pipeline_drain: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """四bank正交散列：bank=lane[1:0] XOR direction。"""

    if k_candidates.ndim != 5 or k_candidates.shape[-2] != 5:
        raise ValueError("xorbank route要求[B,H,T,5,L]")
    tokens = int(k_candidates.shape[-3])
    if tuple(valid.shape) != (tokens, 5):
        raise ValueError("valid shape 与xorbank route不一致")
    changed = (
        k_candidates[..., 1:, :].to(dtype=torch.bool)
        ^ k_candidates[..., 0:1, :].to(dtype=torch.bool)
    )
    changed &= valid[:, 1:].view(
        1,
        1,
        tokens,
        4,
        1,
    )
    bank_counts = []
    lanes = int(k_candidates.shape[-1])
    lane_ids = torch.arange(
        lanes,
        device=k_candidates.device,
        dtype=torch.long,
    )
    for direction in range(4):
        direction_banks = []
        for bank in range(4):
            lane_mask = (
                (lane_ids.remainder(4) ^ direction) == bank
            )
            direction_banks.append(
                changed[..., direction, lane_mask].sum(dim=-1)
            )
        bank_counts.append(torch.stack(direction_banks, dim=-1))
    counts = torch.stack(bank_counts, dim=-2).to(dtype=torch.long)
    active = counts.sum(dim=-1).gt(0)
    if threshold is not None:
        if threshold < 0:
            raise ValueError("xorbank threshold不能为负")
        direction_counts = counts.sum(dim=-1)
        direct_selected = direction_counts.gt(threshold) & active
        if bank_pressure_threshold is not None:
            if bank_pressure_threshold < 0:
                raise ValueError("xorbank bank threshold不能为负")
            direct_selected |= (
                counts.max(dim=-1).values.gt(bank_pressure_threshold)
                & active
            )
        direct_mask = torch.zeros(
            direct_selected.shape[:-1],
            device=k_candidates.device,
            dtype=torch.long,
        )
        for direction in range(4):
            direct_mask |= (
                direct_selected[..., direction].to(torch.long)
                << direction
            )
        direct_cycles = direct_selected.sum(dim=-1).to(torch.long)
        residual_bank_counts = (
            counts
            * (~direct_selected).unsqueeze(-1).to(torch.long)
        ).sum(dim=-2)
        residual_cycles = residual_bank_counts.max(dim=-1).values
        residual_effective = residual_cycles + (
            residual_cycles.gt(0).to(torch.long)
            if pipeline_drain
            else 0
        )
        cycles = torch.maximum(direct_cycles, residual_effective)
        return cycles, direct_mask, direct_cycles, residual_cycles

    choices = []
    direct_choices = []
    residual_choices = []
    for direct_mask in range(16):
        selected = torch.tensor(
            [
                bool((direct_mask >> direction) & 1)
                for direction in range(4)
            ],
            device=k_candidates.device,
            dtype=torch.bool,
        )
        direct_cycles = (
            active & selected.view(1, 1, 1, 4)
        ).sum(dim=-1)
        residual_bank_counts = (
            counts
            * (~selected).view(1, 1, 1, 4, 1).to(torch.long)
        ).sum(dim=-2)
        residual_cycles = residual_bank_counts.max(dim=-1).values
        residual_effective = residual_cycles + (
            residual_cycles.gt(0).to(torch.long)
            if pipeline_drain
            else 0
        )
        choices.append(torch.maximum(direct_cycles, residual_effective))
        direct_choices.append(direct_cycles)
        residual_choices.append(residual_cycles)
    stacked = torch.stack(choices, dim=-1)
    cycles, direct_mask = stacked.min(dim=-1)
    direct_stack = torch.stack(direct_choices, dim=-1)
    residual_stack = torch.stack(residual_choices, dim=-1)
    direct_cycles = torch.gather(
        direct_stack,
        -1,
        direct_mask.unsqueeze(-1),
    ).squeeze(-1)
    residual_cycles = torch.gather(
        residual_stack,
        -1,
        direct_mask.unsqueeze(-1),
    ).squeeze(-1)
    return cycles, direct_mask, direct_cycles, residual_cycles


def xorbank_stencil_residual_bank_loads(
    k_candidates: torch.Tensor,
    valid: torch.Tensor,
    direct_mask: torch.Tensor,
) -> torch.Tensor:
    """返回路由后四个XOR bank的逐destination residual事件数。"""

    if k_candidates.ndim != 5 or k_candidates.shape[-2] != 5:
        raise ValueError("xorbank bank-load要求[B,H,T,5,L]")
    tokens = int(k_candidates.shape[-3])
    if tuple(valid.shape) != (tokens, 5):
        raise ValueError("valid shape 与xorbank bank-load不一致")
    if tuple(direct_mask.shape) != tuple(k_candidates.shape[:-2]):
        raise ValueError("direct mask shape 与xorbank bank-load不一致")

    changed = (
        k_candidates[..., 1:, :].to(dtype=torch.bool)
        ^ k_candidates[..., 0:1, :].to(dtype=torch.bool)
    )
    changed &= valid[:, 1:].view(1, 1, tokens, 4, 1)
    lanes = int(k_candidates.shape[-1])
    lane_ids = torch.arange(
        lanes,
        device=k_candidates.device,
        dtype=torch.long,
    )
    bank_loads = []
    for bank in range(4):
        load = torch.zeros(
            k_candidates.shape[:-2],
            device=k_candidates.device,
            dtype=torch.long,
        )
        for direction in range(4):
            lane_mask = (lane_ids.remainder(4) ^ direction) == bank
            residual_direction = ((direct_mask >> direction) & 1) == 0
            load += (
                changed[..., direction, lane_mask].sum(dim=-1)
                * residual_direction.to(dtype=torch.long)
            )
        bank_loads.append(load)
    return torch.stack(bank_loads, dim=-1)


def joint_stencil_delta_counts(
    k_candidates: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """返回[B,H,T,4]四方向changed-lane计数。"""

    if k_candidates.ndim != 5 or k_candidates.shape[-2] != 5:
        raise ValueError("joint stencil delta要求[B,H,T,5,L]")
    tokens = int(k_candidates.shape[-3])
    if tuple(valid.shape) != (tokens, 5):
        raise ValueError("valid shape 与joint stencil delta不一致")
    self_k = k_candidates[..., 0, :].to(dtype=torch.bool)
    valid_bh = valid.view(
        1,
        1,
        tokens,
        5,
    ).expand(*k_candidates.shape[:-1])
    direction_counts = []
    for direction in range(1, 5):
        changed = (
            self_k ^ k_candidates[..., direction, :].to(dtype=torch.bool)
        )
        changed &= valid_bh[..., direction].unsqueeze(-1)
        direction_counts.append(changed.sum(dim=-1).to(dtype=torch.long))
    return torch.stack(direction_counts, dim=-1)


def joint_stencil_delta_stats(
    k_candidates: torch.Tensor,
    valid: torch.Tensor,
    *,
    residual_widths: tuple[int, ...] = (2, 4, 8),
) -> dict[str, Any]:
    """统计四方向联合delta，并计算exact residual/direct最小周期。"""

    counts = joint_stencil_delta_counts(k_candidates, valid)
    tokens = int(k_candidates.shape[-3])
    valid_bh = valid.view(
        1,
        1,
        tokens,
        5,
    ).expand(*k_candidates.shape[:-1])
    total = counts.sum(dim=-1)
    nonzero_directions = counts.gt(0).sum(dim=-1)
    degree = valid_bh.sum(dim=-1).to(dtype=torch.long)
    result: dict[str, Any] = {
        "joint_delta_event_sum": int(total.sum().item()),
        "joint_delta_total_histogram": histogram(total, minlength=129),
        "joint_delta_active_direction_histogram": histogram(
            nonzero_directions,
            minlength=5,
        ),
        "direct_serial_score_cycle_sum": int(degree.sum().item()),
        "direct_serial_score_cycle_histogram": histogram(
            degree,
            minlength=6,
        ),
    }
    independent_cycles = (
        valid_bh[..., 0].to(dtype=torch.long)
        + independent_direction_min_extra_cycles(counts)
    )
    result["independent_w1x4_score_cycle_sum"] = int(
        independent_cycles.sum().item()
    )
    result["independent_w1x4_score_cycle_histogram"] = histogram(
        independent_cycles,
        minlength=6,
    )
    (
        xorbank_extra,
        xorbank_direct_mask,
        xorbank_direct_cycles,
        xorbank_residual_cycles,
    ) = xorbank_stencil_route(
        k_candidates,
        valid,
        pipeline_drain=True,
    )
    xorbank_cycles = valid_bh[..., 0].to(torch.long) + xorbank_extra
    result["qfsa_xb4_score_cycle_sum"] = int(
        xorbank_cycles.sum().item()
    )
    result["qfsa_xb4_score_cycle_histogram"] = histogram(
        xorbank_cycles,
        minlength=6,
    )
    result["qfsa_xb4_direct_mask_histogram"] = histogram(
        xorbank_direct_mask,
        minlength=16,
    )
    result["qfsa_xb4_direct_cycle_sum"] = int(
        xorbank_direct_cycles.sum().item()
    )
    result["qfsa_xb4_residual_wave_sum"] = int(
        xorbank_residual_cycles.sum().item()
    )
    for threshold in (4, 8, 12):
        (
            threshold_extra,
            threshold_mask,
            threshold_direct,
            threshold_residual,
        ) = xorbank_stencil_route(
            k_candidates,
            valid,
            threshold=threshold,
            pipeline_drain=True,
        )
        threshold_cycles = (
            valid_bh[..., 0].to(torch.long) + threshold_extra
        )
        result[f"qfsa_xb4_t{threshold}_score_cycle_sum"] = int(
            threshold_cycles.sum().item()
        )
        result[
            f"qfsa_xb4_t{threshold}_score_cycle_histogram"
        ] = histogram(threshold_cycles, minlength=6)
        result[
            f"qfsa_xb4_t{threshold}_direct_mask_histogram"
        ] = histogram(threshold_mask, minlength=16)
        result[f"qfsa_xb4_t{threshold}_direct_cycle_sum"] = int(
            threshold_direct.sum().item()
        )
        result[f"qfsa_xb4_t{threshold}_residual_wave_sum"] = int(
            threshold_residual.sum().item()
        )
    for bank_threshold in (1, 2, 3):
        (
            pressure_extra,
            pressure_mask,
            pressure_direct,
            pressure_residual,
        ) = xorbank_stencil_route(
            k_candidates,
            valid,
            threshold=8,
            bank_pressure_threshold=bank_threshold,
            pipeline_drain=True,
        )
        pressure_cycles = valid_bh[..., 0].to(torch.long) + pressure_extra
        prefix = f"qfsa_xb4_t8b{bank_threshold}"
        result[f"{prefix}_score_cycle_sum"] = int(
            pressure_cycles.sum().item()
        )
        result[f"{prefix}_score_cycle_histogram"] = histogram(
            pressure_cycles,
            minlength=6,
        )
        result[f"{prefix}_direct_mask_histogram"] = histogram(
            pressure_mask,
            minlength=16,
        )
        result[f"{prefix}_direct_cycle_sum"] = int(
            pressure_direct.sum().item()
        )
        result[f"{prefix}_residual_wave_sum"] = int(
            pressure_residual.sum().item()
        )

    for width in residual_widths:
        (
            extra_cycles,
            direct_mask,
            direct_cycles,
            residual_waves,
        ) = joint_stencil_route(counts, width)
        qfsa_cycles = valid_bh[..., 0].to(dtype=torch.long) + extra_cycles
        result[f"qfsa_w{width}_score_cycle_sum"] = int(
            qfsa_cycles.sum().item()
        )
        result[f"qfsa_w{width}_score_cycle_histogram"] = histogram(
            qfsa_cycles,
            minlength=6,
        )
        result[f"qfsa_w{width}_direct_mask_histogram"] = histogram(
            direct_mask,
            minlength=16,
        )
        result[f"qfsa_w{width}_direct_cycle_sum"] = int(
            direct_cycles.sum().item()
        )
        result[f"qfsa_w{width}_residual_wave_sum"] = int(
            residual_waves.sum().item()
        )
    return result


def source_frontier_work(
    k_candidates: torch.Tensor,
    gate_code: torch.Tensor,
    valid: torch.Tensor,
    neighbor_index: torch.Tensor,
) -> dict[str, list[int]]:
    """导出一个window-head内按source组织的真实term与退休位置。"""

    tokens, candidates, lanes = k_candidates.shape
    if candidates != 5:
        raise ValueError("source frontier trace requires five candidates")
    if tuple(gate_code.shape) != (tokens, candidates):
        raise ValueError("gate_code shape 与 source frontier 不一致")
    if tuple(valid.shape) != (tokens, candidates):
        raise ValueError("valid shape 与 source frontier 不一致")
    if tuple(neighbor_index.shape) != (tokens, candidates):
        raise ValueError("neighbor_index shape 与 source frontier 不一致")
    expected_self = torch.arange(
        tokens,
        device=neighbor_index.device,
        dtype=neighbor_index.dtype,
    )
    if not torch.equal(neighbor_index[:, 0], expected_self):
        raise ValueError("Local5 source frontier要求candidate0为self")
    if not bool(valid[:, 0].all().item()):
        raise ValueError("Local5 source frontier要求self candidate恒合法")

    valid_lane = valid.unsqueeze(-1)
    active = (
        k_candidates.to(dtype=torch.bool)
        & valid_lane
        & gate_code.gt(0).unsqueeze(-1)
    )
    source = (
        neighbor_index.to(
            device=k_candidates.device,
            dtype=torch.long,
        )
        .unsqueeze(-1)
        .expand(tokens, candidates, lanes)
    )
    gate = gate_code.to(dtype=torch.long).unsqueeze(-1).expand_as(source)
    lane = (
        torch.arange(
            lanes,
            device=k_candidates.device,
            dtype=torch.long,
        )
        .view(1, 1, lanes)
        .expand_as(source)
    )
    term_key = ((source * GATE_CODES + gate) * lanes) + lane
    unique_terms = torch.unique(term_key[active], sorted=True)
    term_source = torch.div(
        unique_terms,
        GATE_CODES * lanes,
        rounding_mode="floor",
    )
    term_count = torch.bincount(term_source, minlength=tokens)

    active_edge = active.any(dim=-1)
    source_edge = source[..., 0]
    source_gate_key = source_edge * GATE_CODES + gate_code.to(dtype=torch.long)
    unique_source_gates = torch.unique(
        source_gate_key[active_edge],
        sorted=True,
    )
    gate_source = torch.div(
        unique_source_gates,
        GATE_CODES,
        rounding_mode="floor",
    )
    gate_count = torch.bincount(gate_source, minlength=tokens)
    delivery_count = torch.bincount(
        source[active],
        minlength=tokens,
    )

    retire = torch.full(
        (tokens,),
        -1,
        dtype=torch.long,
        device=k_candidates.device,
    )
    for destination in range(tokens):
        for candidate in range(candidates):
            if bool(valid[destination, candidate].item()):
                source_id = int(neighbor_index[destination, candidate].item())
                retire[source_id] = max(
                    int(retire[source_id].item()),
                    destination,
                )
    if bool(retire.lt(0).any().item()):
        raise ValueError("存在没有合法消费者的Local5 source")

    joint_counts = joint_stencil_delta_counts(
        k_candidates.view(1, 1, tokens, candidates, lanes),
        valid,
    )[0, 0]
    result = {
        "source_term_count": term_count.cpu().tolist(),
        "source_gate_count": gate_count.cpu().tolist(),
        "source_k_popcount": (
            k_candidates[:, 0, :].sum(dim=-1).to(dtype=torch.long).cpu().tolist()
        ),
        "source_retire_destination": retire.cpu().tolist(),
        "source_delivery_count": delivery_count.cpu().tolist(),
        "source_service_cycles_pipelined": torch.maximum(
            term_count,
            delivery_count,
        ).cpu().tolist(),
        "destination_delta_total": (
            joint_counts.sum(dim=-1).cpu().tolist()
        ),
        "destination_direction_delta_counts": (
            joint_counts.cpu().tolist()
        ),
        "destination_direct_score_cycles": (
            valid.sum(dim=-1).to(dtype=torch.long).cpu().tolist()
        ),
        "destination_independent_w1x4_score_cycles": (
            valid[:, 0].to(dtype=torch.long)
            + independent_direction_min_extra_cycles(joint_counts)
        ).cpu().tolist(),
    }
    self_valid = valid[:, 0].to(dtype=torch.long)
    (
        xorbank_extra,
        xorbank_direct_mask,
        xorbank_direct_cycles,
        xorbank_residual_waves,
    ) = xorbank_stencil_route(
        k_candidates.view(1, 1, tokens, candidates, lanes),
        valid,
        pipeline_drain=True,
    )
    result["destination_qfsa_xb4_score_cycles"] = (
        self_valid + xorbank_extra[0, 0]
    ).cpu().tolist()
    result["destination_qfsa_xb4_direct_mask"] = (
        xorbank_direct_mask[0, 0].cpu().tolist()
    )
    result["destination_qfsa_xb4_direct_cycles"] = (
        xorbank_direct_cycles[0, 0].cpu().tolist()
    )
    result["destination_qfsa_xb4_residual_waves"] = (
        xorbank_residual_waves[0, 0].cpu().tolist()
    )
    for threshold in (4, 8, 12):
        (
            threshold_extra,
            threshold_mask,
            threshold_direct,
            threshold_residual,
        ) = xorbank_stencil_route(
            k_candidates.view(1, 1, tokens, candidates, lanes),
            valid,
            threshold=threshold,
            pipeline_drain=True,
        )
        result[f"destination_qfsa_xb4_t{threshold}_score_cycles"] = (
            self_valid + threshold_extra[0, 0]
        ).cpu().tolist()
        result[f"destination_qfsa_xb4_t{threshold}_direct_mask"] = (
            threshold_mask[0, 0].cpu().tolist()
        )
        result[f"destination_qfsa_xb4_t{threshold}_direct_cycles"] = (
            threshold_direct[0, 0].cpu().tolist()
        )
        result[f"destination_qfsa_xb4_t{threshold}_residual_waves"] = (
            threshold_residual[0, 0].cpu().tolist()
        )
        if threshold == 8:
            threshold_bank_loads = xorbank_stencil_residual_bank_loads(
                k_candidates.view(1, 1, tokens, candidates, lanes),
                valid,
                threshold_mask,
            )[0, 0]
            result["destination_qfsa_xb4_t8_bank_loads"] = (
                threshold_bank_loads.cpu().tolist()
            )
            result["destination_qfsa_xb4_t8_bank_imbalance"] = (
                threshold_bank_loads.max(dim=-1).values
                - threshold_bank_loads.min(dim=-1).values
            ).cpu().tolist()
    for bank_threshold in (1, 2, 3):
        (
            pressure_extra,
            pressure_mask,
            pressure_direct,
            pressure_residual,
        ) = xorbank_stencil_route(
            k_candidates.view(1, 1, tokens, candidates, lanes),
            valid,
            threshold=8,
            bank_pressure_threshold=bank_threshold,
            pipeline_drain=True,
        )
        prefix = f"destination_qfsa_xb4_t8b{bank_threshold}"
        result[f"{prefix}_score_cycles"] = (
            self_valid + pressure_extra[0, 0]
        ).cpu().tolist()
        result[f"{prefix}_direct_mask"] = (
            pressure_mask[0, 0].cpu().tolist()
        )
        result[f"{prefix}_direct_cycles"] = (
            pressure_direct[0, 0].cpu().tolist()
        )
        result[f"{prefix}_residual_waves"] = (
            pressure_residual[0, 0].cpu().tolist()
        )
    for width in (2, 4, 8):
        (
            extra_cycles,
            direct_mask,
            direct_cycles,
            residual_waves,
        ) = joint_stencil_route(
            joint_counts,
            width,
        )
        qfsa_cycles = self_valid + extra_cycles
        result[f"destination_qfsa_w{width}_score_cycles"] = (
            qfsa_cycles.cpu().tolist()
        )
        result[f"destination_qfsa_w{width}_direct_mask"] = (
            direct_mask.cpu().tolist()
        )
        result[f"destination_qfsa_w{width}_direct_cycles"] = (
            direct_cycles.cpu().tolist()
        )
        result[f"destination_qfsa_w{width}_residual_waves"] = (
            residual_waves.cpu().tolist()
        )
    return result


def source_descriptor_trace(
    k_candidates: torch.Tensor,
    gate_code: torch.Tensor,
    valid: torch.Tensor,
    neighbor_index: torch.Tensor,
    *,
    strict_local5_geometry: bool = False,
) -> dict[str, list[int] | list[list[int]]]:
    """按relation-transpose合同导出source-major DS-FLM descriptor。"""

    tokens, candidates, lanes = k_candidates.shape
    if candidates != 5:
        raise ValueError("source descriptor trace requires five candidates")
    if lanes > 63:
        raise ValueError("source descriptor bitmap当前只支持最多63 lanes")
    if tuple(gate_code.shape) != (tokens, candidates):
        raise ValueError("gate_code shape 与 source descriptor 不一致")
    if tuple(valid.shape) != (tokens, candidates):
        raise ValueError("valid shape 与 source descriptor 不一致")
    if tuple(neighbor_index.shape) != (tokens, candidates):
        raise ValueError("neighbor_index shape 与 source descriptor 不一致")
    if bool(neighbor_index.lt(0).any().item()) or bool(
        neighbor_index.ge(tokens).any().item()
    ):
        raise ValueError("neighbor_index越界")
    expected_self = torch.arange(
        tokens, dtype=torch.long, device=neighbor_index.device
    )
    if not torch.equal(
        neighbor_index[:, 0].to(dtype=torch.long),
        expected_self,
    ):
        raise ValueError("source descriptor self relation不是恒等映射")
    if strict_local5_geometry:
        plane_tokens = tokens // 2
        side = math.isqrt(plane_tokens)
        if 2 * side * side != tokens or side != 15:
            raise ValueError("正式Local5几何必须为T2×15×15")
        grid = torch.arange(
            tokens, dtype=torch.long, device=neighbor_index.device
        ).reshape(2, side, side)
        expected_indices = [grid]
        expected_valid = [torch.ones_like(grid, dtype=torch.bool)]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy = (
                torch.arange(side, device=neighbor_index.device)
                .view(1, side, 1)
                + dy
            )
            xx = (
                torch.arange(side, device=neighbor_index.device)
                .view(1, 1, side)
                + dx
            )
            role_valid = (
                (yy >= 0)
                & (yy < side)
                & (xx >= 0)
                & (xx < side)
            )
            yy = yy.clamp(0, side - 1).expand(2, side, side)
            xx = xx.clamp(0, side - 1).expand(2, side, side)
            tt = (
                torch.arange(2, device=neighbor_index.device)
                .view(2, 1, 1)
                .expand_as(yy)
            )
            expected_indices.append(grid[tt, yy, xx])
            expected_valid.append(role_valid.expand(2, side, side))
        expected_neighbor = torch.stack(
            expected_indices, dim=-1
        ).reshape(tokens, candidates)
        expected_valid_tensor = torch.stack(
            expected_valid, dim=-1
        ).reshape(tokens, candidates)
        if not torch.equal(
            neighbor_index.to(dtype=torch.long), expected_neighbor
        ):
            raise ValueError("真实callback邻接不是精确W15 Local5 N/S/E/W几何")
        if not torch.equal(
            valid.to(dtype=torch.bool), expected_valid_tensor
        ):
            raise ValueError("真实callback valid mask不符合W15 Local5边界")

    device = k_candidates.device
    incoming_gate = torch.zeros(
        (tokens, candidates), dtype=torch.long, device=device
    )
    incoming_valid = torch.zeros(
        (tokens, candidates), dtype=torch.bool, device=device
    )
    for role in range(candidates):
        role_valid = valid[:, role].to(device=device, dtype=torch.bool)
        sources = neighbor_index[:, role].to(device=device, dtype=torch.long)
        selected_sources = sources[role_valid]
        if selected_sources.numel():
            counts = torch.bincount(selected_sources, minlength=tokens)
            if bool(counts.gt(1).any().item()):
                raise ValueError(
                    f"同一source的role{role}存在多个incoming relation"
                )
            incoming_gate[selected_sources, role] = gate_code[
                role_valid, role
            ].to(dtype=torch.long)
            incoming_valid[selected_sources, role] = True

    source_k = k_candidates[:, 0, :].to(dtype=torch.bool)
    gathered_source_k = source_k[
        neighbor_index.to(device=device, dtype=torch.long)
    ]
    valid_on_device = valid.to(device=device, dtype=torch.bool)
    if not torch.equal(
        k_candidates.to(dtype=torch.bool)[valid_on_device],
        gathered_source_k[valid_on_device],
    ):
        raise ValueError("candidate K与source self-K关系不一致")
    bit_weights = (
        torch.ones(lanes, dtype=torch.long, device=device)
        << torch.arange(lanes, dtype=torch.long, device=device)
    )
    source_k_bitmap = (
        source_k.to(dtype=torch.long) * bit_weights.view(1, lanes)
    ).sum(dim=-1)
    valid_weights = (
        torch.ones(candidates, dtype=torch.long, device=device)
        << torch.arange(candidates, dtype=torch.long, device=device)
    )
    incoming_valid_mask = (
        incoming_valid.to(dtype=torch.long)
        * valid_weights.view(1, candidates)
    ).sum(dim=-1)

    plane_tokens = tokens
    side = math.isqrt(tokens)
    if side * side != tokens and tokens % 2 == 0:
        candidate_plane_tokens = tokens // 2
        candidate_side = math.isqrt(candidate_plane_tokens)
        if candidate_side * candidate_side == candidate_plane_tokens:
            plane_tokens = candidate_plane_tokens
            side = candidate_side
    source_ids = torch.arange(tokens, dtype=torch.long, device=device)
    if side * side == plane_tokens:
        spatial = source_ids.remainder(plane_tokens)
        source_plane = torch.div(
            source_ids, plane_tokens, rounding_mode="floor"
        )
        source_y = torch.div(spatial, side, rounding_mode="floor")
        source_x = spatial.remainder(side)
    else:
        # 仅用于1D合成单测；真实Local5窗口必须走上面的方形合同。
        source_plane = torch.zeros_like(source_ids)
        source_y = torch.zeros_like(source_ids)
        source_x = source_ids
    return {
        "source_id": source_ids.cpu().tolist(),
        "source_plane": source_plane.cpu().tolist(),
        "source_y": source_y.cpu().tolist(),
        "source_x": source_x.cpu().tolist(),
        "source_k_bitmap": source_k_bitmap.cpu().tolist(),
        "incoming_gates": incoming_gate.cpu().tolist(),
        "incoming_valid_mask": incoming_valid_mask.cpu().tolist(),
    }


def validate_post_g0_export_contract(
    contract: dict[str, Any],
) -> None:
    if "local5" not in contract["attention_mode"]:
        raise ValueError("post_g0 ordered trace requires Local5 mode")
    if not contract["hardware_quant_enabled"]:
        raise ValueError("post_g0 ordered trace requires hardware quant")
    if not contract["hardware_rtl_shiftmax_enabled"]:
        raise ValueError("post_g0 ordered trace requires RTL Shiftmax")
    if not contract["hardware_mask_invalid_candidates"]:
        raise ValueError(
            "post_g0 ordered trace requires true invalid-candidate mask"
        )
    if abs(contract["hardware_score_step"] - (1.0 / 128.0)) > 1e-12:
        raise ValueError("post_g0 ordered trace requires Q7 score step")
    if abs(contract["hardware_gate_step"] - (1.0 / 128.0)) > 1e-12:
        raise ValueError("post_g0 ordered trace requires Q1.7 gate step")
    if contract["crop"] is not None:
        raise ValueError("post_g0 ordered trace requires crop=null")
    if contract["scale_factor"] != 1.0:
        raise ValueError("post_g0 ordered trace requires scale_factor=1")
    if contract["resolution"] != [480, 640]:
        raise ValueError("post_g0 ordered trace requires 480x640 resolution")
    if contract["window_size"] != POST_G0_WINDOW:
        raise ValueError("post_g0 ordered trace requires window=2x15x15")


def load_post_g0_run_identity(
    path: Path,
    *,
    config: Path,
    checkpoint: Path,
    samples: int,
    groups_per_block_sample: int,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "local5_post_g0_run_identity_v3":
        raise ValueError("post_g0 run identity schema错误")
    expected = {
        "config_sha256": file_sha256(config),
        "checkpoint_sha256": file_sha256(checkpoint),
        "relation_rtl_sha256": file_sha256(
            HW_ROOT / "rtl_qfit/qfit_relation_transpose_leaf.sv"
        ),
        "samples": samples,
        "groups_per_block_sample": groups_per_block_sample,
        "sampling_id": POST_G0_SAMPLING_ID,
        "dataset_sampling_id": POST_G0_DATASET_SAMPLING_ID,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"post_g0 run identity字段不匹配: {name}")
    if Path(str(value.get("config", ""))).resolve() != config.resolve():
        raise ValueError("post_g0 run identity config路径不匹配")
    if (
        Path(str(value.get("checkpoint", ""))).resolve()
        != checkpoint.resolve()
    ):
        raise ValueError("post_g0 run identity checkpoint路径不匹配")
    ranking_path = Path(str(value.get("ranking", "")))
    if not ranking_path.is_file() or value.get(
        "ranking_sha256"
    ) != file_sha256(ranking_path):
        raise ValueError("post_g0 ranking绑定失效")
    receipt_path = Path(str(value.get("release_receipt", ""))).resolve()
    receipt = validate_release_receipt(
        receipt_path,
        str(value.get("release_receipt_sha256", "")),
    )
    if (
        value.get("watcher_session_uuid")
        != receipt.get("watcher_session_uuid")
        or receipt.get("ranking_path") != value.get("ranking")
        or receipt.get("ranking_sha256") != value.get("ranking_sha256")
        or receipt.get("checkpoint_path") != value.get("checkpoint")
        or receipt.get("checkpoint_sha256")
        != value.get("checkpoint_sha256")
        or receipt.get("config_path") != value.get("config")
        or receipt.get("config_sha256") != value.get("config_sha256")
        or receipt.get("best_epoch") != value.get("best_epoch")
    ):
        raise ValueError("post_g0 run identity与release receipt不一致")
    bindings = value.get("source_bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError("post_g0生产软件绑定集合不完整")
    expected_paths = expected_source_binding_paths()
    # Required repo software bindings must all be present. Production runners
    # (e.g. bb1e4) may also bind run-scoped artifacts such as
    # training_config_identity; those extras are validated by path+sha only.
    missing = set(expected_paths) - set(bindings)
    if missing:
        raise ValueError(
            "post_g0生产软件绑定集合不完整: missing="
            + ",".join(sorted(missing))
        )
    for name, expected_path in expected_paths.items():
        binding = bindings[name]
        if not isinstance(binding, dict):
            raise ValueError(f"post_g0生产软件绑定失效: {name}")
        if (
            Path(str(binding.get("path", ""))).resolve()
            != expected_path.resolve()
            or not expected_path.is_file()
            or binding.get("sha256") != file_sha256(expected_path)
        ):
            raise ValueError(f"post_g0生产软件绑定失效: {name}")
    for name, binding in bindings.items():
        if name in expected_paths:
            continue
        if not isinstance(binding, dict):
            raise ValueError(f"post_g0生产软件绑定失效: {name}")
        path = Path(str(binding.get("path", "")))
        if (
            not path.is_file()
            or binding.get("sha256") != file_sha256(path)
        ):
            raise ValueError(f"post_g0生产软件绑定失效: {name}")
    return value


def post_g0_qualification(
    groups: list[dict[str, Any]],
    *,
    processed_samples: int,
    attached_blocks: int,
    groups_per_block_sample: int,
    run_identity_bound: bool,
) -> dict[str, Any]:
    """生成可机读的正式post-G0覆盖验收，不满足任一项即不晋级。"""

    modules = sorted({str(group["module"]) for group in groups})
    observed_block_pairs = {
        (int(group["stage"]), int(group["block"])) for group in groups
    }
    pair_counts: dict[tuple[str, int], int] = defaultdict(int)
    groups_by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        module = str(group["module"])
        sample = int(group["sample"])
        pair_counts[(module, sample)] += 1
        groups_by_module[module].append(group)

    pair_coverage_ok = (
        set(pair_counts)
        == {
            (module, sample)
            for module in modules
            for sample in range(processed_samples)
        }
        and all(
            count == groups_per_block_sample
            for count in pair_counts.values()
        )
    )
    head_coverage_ok = True
    rotating_group_coverage_ok = True
    exact_sampling_ok = True
    module_coverage: dict[str, Any] = {}
    for module, rows in sorted(groups_by_module.items()):
        head_counts = {int(row["heads"]) for row in rows}
        total_counts = {
            int(row["heads"]) * int(row["batch_windows"])
            for row in rows
        }
        if len(head_counts) != 1 or len(total_counts) != 1:
            head_coverage_ok = False
            rotating_group_coverage_ok = False
            continue
        heads = next(iter(head_counts))
        total_groups = next(iter(total_counts))
        observed_heads = {int(row["head"]) for row in rows}
        observed_flat = {int(row["flat_group"]) for row in rows}
        expected_unique_flat = min(
            total_groups,
            processed_samples * min(
                groups_per_block_sample,
                total_groups,
            ),
        )
        module_head_ok = observed_heads == set(range(heads))
        module_rotating_ok = len(observed_flat) == expected_unique_flat
        module_exact_sampling_ok = True
        pair_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            flat = int(row["flat_group"])
            window = int(row["window"])
            head = int(row["head"])
            row_heads = int(row["heads"])
            row_total = int(row["batch_windows"]) * row_heads
            if (
                flat != window * row_heads + head
                or not 0 <= flat < row_total
            ):
                module_exact_sampling_ok = False
            pair_rows[int(row["sample"])].append(row)
        stage = int(rows[0]["stage"])
        block = int(rows[0]["block"])
        if any(
            int(row["stage"]) != stage or int(row["block"]) != block
            for row in rows
        ):
            module_exact_sampling_ok = False
        for sample, sample_rows in pair_rows.items():
            sample_total_counts = {
                int(row["heads"]) * int(row["batch_windows"])
                for row in sample_rows
            }
            if len(sample_total_counts) != 1:
                module_exact_sampling_ok = False
                continue
            sample_total = next(iter(sample_total_counts))
            expected_indices = rotating_flat_indices(
                total_groups=sample_total,
                selected_groups=min(
                    groups_per_block_sample, sample_total
                ),
                sample_id=sample,
                stage=stage,
                block=block,
            )
            actual_indices = [
                int(row["flat_group"]) for row in sample_rows
            ]
            if actual_indices != expected_indices:
                module_exact_sampling_ok = False
        head_coverage_ok &= module_head_ok
        rotating_group_coverage_ok &= module_rotating_ok
        exact_sampling_ok &= module_exact_sampling_ok
        module_coverage[module] = {
            "heads": heads,
            "observed_heads": len(observed_heads),
            "total_flat_groups": total_groups,
            "observed_unique_flat_groups": len(observed_flat),
            "expected_unique_flat_groups": expected_unique_flat,
            "head_coverage_ok": module_head_ok,
            "rotating_group_coverage_ok": module_rotating_ok,
            "exact_sampling_ok": module_exact_sampling_ok,
        }

    shape_ok = bool(groups) and all(
        int(group["tokens"]) == POST_G0_TOKENS
        and int(group["lanes"]) == POST_G0_LANES
        and int(group["time_planes"]) == 2
        and int(group["plane_tokens"]) == 225
        and int(group["spatial_side"]) == 15
        for group in groups
    )
    checks = {
        "run_identity_bound": run_identity_bound,
        "processed_samples_100": processed_samples == POST_G0_SAMPLES,
        "attached_blocks_12": attached_blocks == POST_G0_BLOCKS,
        "captured_modules_12": len(modules) == POST_G0_BLOCKS,
        "exact_target_block_set": (
            observed_block_pairs == set(POST_G0_BLOCK_PAIRS)
        ),
        "module_sample_pair_coverage": pair_coverage_ok,
        "all_head_coverage": head_coverage_ok,
        "rotating_flat_group_coverage": rotating_group_coverage_ok,
        "exact_rotating_indices": exact_sampling_ok,
        "shape_t450_l32": shape_ok,
        "sampling_contract": all(
            group.get("selection") == POST_G0_SAMPLING_ID
            for group in groups
        ),
    }
    return {
        "schema": "local5_post_g0_qualification_v1",
        "qualified": all(checks.values()),
        "checks": checks,
        "processed_samples": processed_samples,
        "attached_blocks": attached_blocks,
        "captured_modules": len(modules),
        "captured_groups": len(groups),
        "expected_groups": (
            processed_samples
            * POST_G0_BLOCKS
            * groups_per_block_sample
        ),
        "module_coverage": module_coverage,
    }


class OrderedTermTraceSink:
    def __init__(
        self,
        *,
        groups_per_block_sample: int,
        evidence_level: str,
    ) -> None:
        self.groups_per_block_sample = groups_per_block_sample
        self.evidence_level = evidence_level
        self.group_offsets = [0]
        self.group_tags: list[int] = []
        self.item_mode: list[int] = []
        self.item_gate: list[int] = []
        self.item_lane: list[int] = []
        self.item_mult: list[int] = []
        self.item_dest: list[int] = []
        self.source_group_offsets = [0]
        self.descriptor_group_offsets = [0]
        self.descriptor_source_id: list[int] = []
        self.descriptor_source_plane: list[int] = []
        self.descriptor_source_y: list[int] = []
        self.descriptor_source_x: list[int] = []
        self.descriptor_q_bitmap: list[int] = []
        self.descriptor_k_bitmap: list[int] = []
        self.descriptor_incoming_gates: list[list[int]] = []
        self.descriptor_valid_mask: list[int] = []
        self.source_term_count: list[int] = []
        self.source_gate_count: list[int] = []
        self.source_k_popcount: list[int] = []
        self.source_retire_destination: list[int] = []
        self.source_delivery_count: list[int] = []
        self.source_service_cycles_pipelined: list[int] = []
        self.destination_delta_total: list[int] = []
        self.destination_direction_delta_counts: list[list[int]] = []
        self.destination_direct_score_cycles: list[int] = []
        self.destination_independent_w1x4_score_cycles: list[int] = []
        self.destination_qfsa_w2_score_cycles: list[int] = []
        self.destination_qfsa_w4_score_cycles: list[int] = []
        self.destination_qfsa_w8_score_cycles: list[int] = []
        self.destination_qfsa_w2_direct_mask: list[int] = []
        self.destination_qfsa_w4_direct_mask: list[int] = []
        self.destination_qfsa_w8_direct_mask: list[int] = []
        self.destination_qfsa_w2_direct_cycles: list[int] = []
        self.destination_qfsa_w4_direct_cycles: list[int] = []
        self.destination_qfsa_w8_direct_cycles: list[int] = []
        self.destination_qfsa_w2_residual_waves: list[int] = []
        self.destination_qfsa_w4_residual_waves: list[int] = []
        self.destination_qfsa_w8_residual_waves: list[int] = []
        self.destination_qfsa_xb4_score_cycles: list[int] = []
        self.destination_qfsa_xb4_direct_mask: list[int] = []
        self.destination_qfsa_xb4_direct_cycles: list[int] = []
        self.destination_qfsa_xb4_residual_waves: list[int] = []
        self.destination_qfsa_xb4_t8_bank_loads: list[list[int]] = []
        self.destination_qfsa_xb4_t8_bank_imbalance: list[int] = []
        for threshold in (4, 8, 12):
            for suffix in (
                "score_cycles",
                "direct_mask",
                "direct_cycles",
                "residual_waves",
            ):
                setattr(
                    self,
                    f"destination_qfsa_xb4_t{threshold}_{suffix}",
                    [],
                )
        for bank_threshold in (1, 2, 3):
            for suffix in (
                "score_cycles",
                "direct_mask",
                "direct_cycles",
                "residual_waves",
            ):
                setattr(
                    self,
                    (
                        "destination_qfsa_xb4_t8b"
                        f"{bank_threshold}_{suffix}"
                    ),
                    [],
                )
        self.groups: list[dict[str, Any]] = []

    def capture(
        self,
        *,
        name: str,
        stage: int,
        block: int,
        sample_id: int,
        k_candidates: torch.Tensor,
        valid: torch.Tensor,
        gate_code: torch.Tensor,
        neighbor_index: torch.Tensor,
        q_event: torch.Tensor | None = None,
    ) -> None:
        batch_windows, heads, tokens, candidates, lanes = k_candidates.shape
        if candidates != 5:
            raise ValueError("Local5 ordered producer requires five candidates")
        if self.evidence_level == "post_g0" and (
            tokens != POST_G0_TOKENS or lanes != POST_G0_LANES
        ):
            raise ValueError(
                "post_g0 descriptor shape必须为T450×32 lanes"
            )
        if self.evidence_level == "post_g0" and q_event is None:
            raise ValueError("正式post-G0 Q/K score trace缺少q_event")
        total_groups = batch_windows * heads
        selected_groups = min(
            self.groups_per_block_sample, total_groups
        )
        if selected_groups <= 0:
            return
        flat_indices = rotating_flat_indices(
            total_groups=total_groups,
            selected_groups=selected_groups,
            sample_id=sample_id,
            stage=stage,
            block=block,
        )
        for flat_index in flat_indices:
            window = int(flat_index // heads)
            head = int(flat_index % heads)
            group_k = k_candidates[window, head].to(dtype=torch.bool)
            group_q = (
                q_event[window, head].to(dtype=torch.bool)
                if q_event is not None
                else None
            )
            group_gate = gate_code[window, head].to(dtype=torch.long)
            group_valid = valid.to(device=group_k.device, dtype=torch.bool)
            frontier = source_frontier_work(
                group_k,
                group_gate,
                group_valid,
                neighbor_index.to(device=group_k.device),
            )
            descriptor = source_descriptor_trace(
                group_k,
                group_gate,
                group_valid,
                neighbor_index.to(device=group_k.device),
                strict_local5_geometry=(
                    self.evidence_level == "post_g0"
                ),
            )
            plane_tokens = tokens // 2
            side = math.isqrt(plane_tokens)
            plane_serial = 2 * side * side == tokens
            if not plane_serial:
                plane_tokens = tokens
                side = math.isqrt(tokens)
            for destination in range(tokens):
                unique_gates: list[int] = []
                for candidate in range(candidates):
                    gate = int(group_gate[destination, candidate].item())
                    if (
                        bool(group_valid[destination, candidate])
                        and gate > 0
                        and gate not in unique_gates
                    ):
                        unique_gates.append(gate)
                for lane in range(lanes):
                    for gate in unique_gates:
                        multiplicity = 0
                        for candidate in range(candidates):
                            if (
                                bool(group_valid[destination, candidate])
                                and int(
                                    group_gate[destination, candidate].item()
                                ) == gate
                                and bool(
                                    group_k[
                                        destination, candidate, lane
                                    ]
                                )
                            ):
                                multiplicity += 1
                        if multiplicity == 0:
                            continue
                        self.item_mode.append(1)
                        self.item_gate.append(gate)
                        self.item_lane.append(lane)
                        self.item_mult.append(multiplicity)
                        self.item_dest.append(destination)
            tag = len(self.group_tags)
            self.group_tags.append(tag)
            self.group_offsets.append(len(self.item_gate))
            self.source_term_count.extend(frontier["source_term_count"])
            self.source_gate_count.extend(frontier["source_gate_count"])
            self.source_k_popcount.extend(frontier["source_k_popcount"])
            self.source_retire_destination.extend(
                frontier["source_retire_destination"]
            )
            self.source_delivery_count.extend(
                frontier["source_delivery_count"]
            )
            self.source_service_cycles_pipelined.extend(
                frontier["source_service_cycles_pipelined"]
            )
            self.destination_delta_total.extend(
                frontier["destination_delta_total"]
            )
            self.destination_direction_delta_counts.extend(
                frontier["destination_direction_delta_counts"]
            )
            self.destination_direct_score_cycles.extend(
                frontier["destination_direct_score_cycles"]
            )
            self.destination_independent_w1x4_score_cycles.extend(
                frontier["destination_independent_w1x4_score_cycles"]
            )
            for width in (2, 4, 8):
                getattr(
                    self,
                    f"destination_qfsa_w{width}_score_cycles",
                ).extend(
                    frontier[
                        f"destination_qfsa_w{width}_score_cycles"
                    ]
                )
                getattr(
                    self,
                    f"destination_qfsa_w{width}_direct_mask",
                ).extend(
                    frontier[
                        f"destination_qfsa_w{width}_direct_mask"
                    ]
                )
                getattr(
                    self,
                    f"destination_qfsa_w{width}_direct_cycles",
                ).extend(
                    frontier[
                        f"destination_qfsa_w{width}_direct_cycles"
                    ]
                )
                getattr(
                    self,
                    f"destination_qfsa_w{width}_residual_waves",
                ).extend(
                    frontier[
                        f"destination_qfsa_w{width}_residual_waves"
                    ]
                )
            for suffix in (
                "score_cycles",
                "direct_mask",
                "direct_cycles",
                "residual_waves",
            ):
                getattr(
                    self,
                    f"destination_qfsa_xb4_{suffix}",
                ).extend(
                    frontier[f"destination_qfsa_xb4_{suffix}"]
                )
            for threshold in (4, 8, 12):
                for suffix in (
                    "score_cycles",
                    "direct_mask",
                    "direct_cycles",
                    "residual_waves",
                ):
                    getattr(
                        self,
                        f"destination_qfsa_xb4_t{threshold}_{suffix}",
                    ).extend(
                        frontier[
                            f"destination_qfsa_xb4_t{threshold}_{suffix}"
                        ]
                    )
            self.destination_qfsa_xb4_t8_bank_loads.extend(
                frontier["destination_qfsa_xb4_t8_bank_loads"]
            )
            self.destination_qfsa_xb4_t8_bank_imbalance.extend(
                frontier["destination_qfsa_xb4_t8_bank_imbalance"]
            )
            for bank_threshold in (1, 2, 3):
                for suffix in (
                    "score_cycles",
                    "direct_mask",
                    "direct_cycles",
                    "residual_waves",
                ):
                    key = (
                        "destination_qfsa_xb4_t8b"
                        f"{bank_threshold}_{suffix}"
                    )
                    getattr(self, key).extend(frontier[key])
            self.source_group_offsets.append(len(self.source_term_count))
            self.descriptor_source_id.extend(descriptor["source_id"])
            self.descriptor_source_plane.extend(
                descriptor["source_plane"]
            )
            self.descriptor_source_y.extend(descriptor["source_y"])
            self.descriptor_source_x.extend(descriptor["source_x"])
            if group_q is not None:
                q_bit_weights = (
                    torch.ones(
                        lanes, dtype=torch.long, device=group_q.device
                    )
                    << torch.arange(
                        lanes, dtype=torch.long, device=group_q.device
                    )
                )
                self.descriptor_q_bitmap.extend(
                    (
                        group_q.to(dtype=torch.long)
                        * q_bit_weights.view(1, lanes)
                    )
                    .sum(dim=-1)
                    .cpu()
                    .tolist()
                )
            self.descriptor_k_bitmap.extend(
                descriptor["source_k_bitmap"]
            )
            self.descriptor_incoming_gates.extend(
                descriptor["incoming_gates"]
            )
            self.descriptor_valid_mask.extend(
                descriptor["incoming_valid_mask"]
            )
            self.descriptor_group_offsets.append(
                len(self.descriptor_source_id)
            )
            self.groups.append(
                {
                    "tag": tag,
                    "empty": self.group_offsets[-1] ==
                    self.group_offsets[-2],
                    "sample": sample_id,
                    "stage": stage,
                    "block": block,
                    "window": window,
                    "head": head,
                    "flat_group": int(flat_index),
                    "batch_windows": batch_windows,
                    "heads": heads,
                    "lanes": lanes,
                    "tokens": tokens,
                    "time_planes": 2 if plane_serial else 1,
                    "plane_tokens": plane_tokens,
                    "spatial_side": side,
                    "plane_execution": (
                        "plane_serial_drain"
                        if plane_serial
                        else "synthetic_unverified"
                    ),
                    "module": name,
                    "selection": POST_G0_SAMPLING_ID,
                }
            )

    def write(
        self,
        *,
        output_dir: Path,
        config: Path,
        checkpoint: Path,
        cohort: dict[str, Any],
        sample_keys: list[str],
        sequence_keys: list[str],
        dataset_indices: list[int] | None = None,
        dataset_size: int | None = None,
        full_resolution: bool,
        software_contract: dict[str, Any],
        threshold_semantics: dict[str, Any],
        projection_contract_manifest: Path | None = None,
        projection_contract_payload: Path | None = None,
        formal_context: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.evidence_level == "post_g0":
            if (
                projection_contract_manifest is None
                or projection_contract_payload is None
            ):
                raise ValueError("正式post-G0必须绑定checkpoint projection contract")
            if (
                len(sample_keys) != POST_G0_SAMPLES
                or len(sequence_keys) != POST_G0_SAMPLES
                or len(set(sample_keys)) != POST_G0_SAMPLES
                or any(not value for value in sample_keys)
                or any(not value for value in sequence_keys)
            ):
                raise ValueError(
                    "正式post-G0 cohort必须含100个唯一sample key和"
                    "100个非空sequence key"
                )
            if (
                dataset_indices is None
                or len(dataset_indices) != POST_G0_SAMPLES
                or len(set(dataset_indices)) != POST_G0_SAMPLES
                or dataset_size is None
                or any(index < 0 or index >= dataset_size for index in dataset_indices)
            ):
                raise ValueError("正式post-G0 cohort缺少唯一有效的分层dataset索引")
            expected_descriptors = len(self.groups) * POST_G0_TOKENS
            if (
                len(self.descriptor_q_bitmap) != expected_descriptors
                or len(self.descriptor_k_bitmap) != expected_descriptors
            ):
                raise ValueError(
                    "正式post-G0 Q/K score trace数量不守恒: "
                    f"q={len(self.descriptor_q_bitmap)} "
                    f"k={len(self.descriptor_k_bitmap)} "
                    f"expected={expected_descriptors}"
                )
        cohort_path = output_dir / "ordered_cohort.json"
        cohort_artifact = {
            "schema": "ordered_trace_cohort_v2",
            "count": len(sample_keys),
            "sample_keys": sample_keys,
            "sequence_keys": sequence_keys,
            "sample_key_sha256": string_list_sha256(sample_keys),
            "sequence_key_sha256": string_list_sha256(sequence_keys),
            "dataset_sampling_id": POST_G0_DATASET_SAMPLING_ID,
            "dataset_size": dataset_size,
            "dataset_indices": dataset_indices,
            "dataset_indices_sha256": (
                int_list_sha256(dataset_indices)
                if dataset_indices is not None
                else None
            ),
            "sequence_counts": dict(
                sorted(
                    (key, sequence_keys.count(key))
                    for key in set(sequence_keys)
                )
            ),
        }
        cohort_path.write_text(
            json.dumps(cohort_artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        arrays = {
            "group_offsets": np.asarray(
                self.group_offsets, dtype=np.int64
            ),
            "group_tags": np.asarray(
                self.group_tags, dtype=np.uint64
            ),
            "item_mode_multiset": np.asarray(
                self.item_mode, dtype=np.uint8
            ),
            "item_gate_code": np.asarray(
                self.item_gate, dtype=np.uint16
            ),
            "item_lane_id": np.asarray(
                self.item_lane, dtype=np.uint16
            ),
            "item_multiplicity": np.asarray(
                self.item_mult, dtype=np.uint8
            ),
            "item_destination": np.asarray(
                self.item_dest, dtype=np.uint16
            ),
            "source_group_offsets": np.asarray(
                self.source_group_offsets, dtype=np.int64
            ),
            "descriptor_group_offsets": np.asarray(
                self.descriptor_group_offsets, dtype=np.int64
            ),
            "descriptor_source_id": np.asarray(
                self.descriptor_source_id, dtype=np.uint16
            ),
            "descriptor_source_plane": np.asarray(
                self.descriptor_source_plane, dtype=np.uint8
            ),
            "descriptor_source_y": np.asarray(
                self.descriptor_source_y, dtype=np.uint16
            ),
            "descriptor_source_x": np.asarray(
                self.descriptor_source_x, dtype=np.uint16
            ),
            "descriptor_q_bitmap": np.asarray(
                self.descriptor_q_bitmap, dtype=np.uint64
            ),
            "descriptor_k_bitmap": np.asarray(
                self.descriptor_k_bitmap, dtype=np.uint64
            ),
            "descriptor_incoming_gates": np.asarray(
                self.descriptor_incoming_gates, dtype=np.uint16
            ),
            "descriptor_valid_mask": np.asarray(
                self.descriptor_valid_mask, dtype=np.uint8
            ),
            "source_term_count": np.asarray(
                self.source_term_count, dtype=np.uint16
            ),
            "source_gate_count": np.asarray(
                self.source_gate_count, dtype=np.uint8
            ),
            "source_k_popcount": np.asarray(
                self.source_k_popcount, dtype=np.uint8
            ),
            "source_retire_destination": np.asarray(
                self.source_retire_destination, dtype=np.uint16
            ),
            "source_delivery_count": np.asarray(
                self.source_delivery_count, dtype=np.uint16
            ),
            "source_service_cycles_pipelined": np.asarray(
                self.source_service_cycles_pipelined, dtype=np.uint16
            ),
            "destination_delta_total": np.asarray(
                self.destination_delta_total, dtype=np.uint8
            ),
            "destination_direction_delta_counts": np.asarray(
                self.destination_direction_delta_counts, dtype=np.uint8
            ),
            "destination_direct_score_cycles": np.asarray(
                self.destination_direct_score_cycles, dtype=np.uint8
            ),
            "destination_independent_w1x4_score_cycles": np.asarray(
                self.destination_independent_w1x4_score_cycles,
                dtype=np.uint8,
            ),
            "destination_qfsa_w2_score_cycles": np.asarray(
                self.destination_qfsa_w2_score_cycles, dtype=np.uint8
            ),
            "destination_qfsa_w4_score_cycles": np.asarray(
                self.destination_qfsa_w4_score_cycles, dtype=np.uint8
            ),
            "destination_qfsa_w8_score_cycles": np.asarray(
                self.destination_qfsa_w8_score_cycles, dtype=np.uint8
            ),
            "destination_qfsa_w2_direct_mask": np.asarray(
                self.destination_qfsa_w2_direct_mask, dtype=np.uint8
            ),
            "destination_qfsa_w4_direct_mask": np.asarray(
                self.destination_qfsa_w4_direct_mask, dtype=np.uint8
            ),
            "destination_qfsa_w8_direct_mask": np.asarray(
                self.destination_qfsa_w8_direct_mask, dtype=np.uint8
            ),
            "destination_qfsa_w2_direct_cycles": np.asarray(
                self.destination_qfsa_w2_direct_cycles, dtype=np.uint8
            ),
            "destination_qfsa_w4_direct_cycles": np.asarray(
                self.destination_qfsa_w4_direct_cycles, dtype=np.uint8
            ),
            "destination_qfsa_w8_direct_cycles": np.asarray(
                self.destination_qfsa_w8_direct_cycles, dtype=np.uint8
            ),
            "destination_qfsa_w2_residual_waves": np.asarray(
                self.destination_qfsa_w2_residual_waves, dtype=np.uint8
            ),
            "destination_qfsa_w4_residual_waves": np.asarray(
                self.destination_qfsa_w4_residual_waves, dtype=np.uint8
            ),
            "destination_qfsa_w8_residual_waves": np.asarray(
                self.destination_qfsa_w8_residual_waves, dtype=np.uint8
            ),
            "destination_qfsa_xb4_score_cycles": np.asarray(
                self.destination_qfsa_xb4_score_cycles, dtype=np.uint8
            ),
            "destination_qfsa_xb4_direct_mask": np.asarray(
                self.destination_qfsa_xb4_direct_mask, dtype=np.uint8
            ),
            "destination_qfsa_xb4_direct_cycles": np.asarray(
                self.destination_qfsa_xb4_direct_cycles, dtype=np.uint8
            ),
            "destination_qfsa_xb4_residual_waves": np.asarray(
                self.destination_qfsa_xb4_residual_waves, dtype=np.uint8
            ),
            "destination_qfsa_xb4_t8_bank_loads": np.asarray(
                self.destination_qfsa_xb4_t8_bank_loads, dtype=np.uint8
            ),
            "destination_qfsa_xb4_t8_bank_imbalance": np.asarray(
                self.destination_qfsa_xb4_t8_bank_imbalance,
                dtype=np.uint8,
            ),
        }
        for threshold in (4, 8, 12):
            for suffix in (
                "score_cycles",
                "direct_mask",
                "direct_cycles",
                "residual_waves",
            ):
                arrays[
                    f"destination_qfsa_xb4_t{threshold}_{suffix}"
                ] = np.asarray(
                    getattr(
                        self,
                        f"destination_qfsa_xb4_t{threshold}_{suffix}",
                    ),
                    dtype=np.uint8,
                )
        for bank_threshold in (1, 2, 3):
            for suffix in (
                "score_cycles",
                "direct_mask",
                "direct_cycles",
                "residual_waves",
            ):
                key = (
                    "destination_qfsa_xb4_t8b"
                    f"{bank_threshold}_{suffix}"
                )
                arrays[key] = np.asarray(
                    getattr(self, key),
                    dtype=np.uint8,
                )
        payload_path = output_dir / "ordered_term_items.npz"
        np.savez_compressed(payload_path, **arrays)
        for index, group in enumerate(self.groups):
            group["ordered_item_sha256"] = canonical_item_hash(
                arrays,
                self.group_offsets[index],
                self.group_offsets[index + 1],
            )
        run_identity_path = (
            Path(str(formal_context["run_identity"])).resolve()
            if formal_context is not None
            else None
        )
        qualification = post_g0_qualification(
            self.groups,
            processed_samples=(
                int(formal_context["processed_samples"])
                if formal_context is not None
                else len(sample_keys)
            ),
            attached_blocks=(
                int(formal_context["attached_blocks"])
                if formal_context is not None
                else 0
            ),
            groups_per_block_sample=self.groups_per_block_sample,
            run_identity_bound=(
                run_identity_path is not None
                and run_identity_path.is_file()
            ),
        )
        manifest = {
            "schema": "et3_ordered_term_trace_v2",
            "evidence_level": self.evidence_level,
            "payload_file": payload_path.name,
            "payload_sha256": file_sha256(payload_path),
            "config": str(config.resolve()),
            "config_sha256": file_sha256(config),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "cohort_file": cohort_path.name,
            "cohort_file_sha256": file_sha256(cohort_path),
            "cohort_sha256": cohort["sample_key_sha256"],
            "run_identity_file": (
                str(run_identity_path) if run_identity_path else None
            ),
            "run_identity_file_sha256": (
                file_sha256(run_identity_path)
                if run_identity_path is not None
                else None
            ),
            "qualification": qualification,
            "resolution": {
                "full_resolution": full_resolution,
                "tokens_by_stage": sorted(
                    {int(group["tokens"]) for group in self.groups}
                ),
            },
            "software_contract": software_contract,
            "threshold_training_semantics": threshold_semantics,
            "producer_order_contract": {
                "id": "local5_mfep_lane_major_first_gate_v1",
                "term_order": [
                    "destination_ascending",
                    "lane_ascending",
                    "gate_first_valid_candidate",
                ],
                "rtl_source": str(
                    (
                        HW_ROOT
                        / "rtl_local5"
                        / "local5_mfep_term_builder.sv"
                    ).resolve()
                ),
                "rtl_source_sha256": file_sha256(
                    HW_ROOT
                    / "rtl_local5"
                    / "local5_mfep_term_builder.sv"
                ),
            },
            "source_frontier_contract": {
                "id": "local5_source_frontier_work_v1",
                "arrays": [
                    "source_term_count",
                    "source_gate_count",
                    "source_k_popcount",
                    "source_retire_destination",
                    "source_delivery_count",
                    "source_service_cycles_pipelined",
                    "destination_delta_total",
                    "destination_direction_delta_counts",
                    "destination_direct_score_cycles",
                    "destination_independent_w1x4_score_cycles",
                    "destination_qfsa_w2_score_cycles",
                    "destination_qfsa_w4_score_cycles",
                    "destination_qfsa_w8_score_cycles",
                    "destination_qfsa_w2_direct_mask",
                    "destination_qfsa_w4_direct_mask",
                    "destination_qfsa_w8_direct_mask",
                    "destination_qfsa_w2_direct_cycles",
                    "destination_qfsa_w4_direct_cycles",
                    "destination_qfsa_w8_direct_cycles",
                    "destination_qfsa_w2_residual_waves",
                    "destination_qfsa_w4_residual_waves",
                    "destination_qfsa_w8_residual_waves",
                    "destination_qfsa_xb4_score_cycles",
                    "destination_qfsa_xb4_direct_mask",
                    "destination_qfsa_xb4_direct_cycles",
                    "destination_qfsa_xb4_residual_waves",
                    "destination_qfsa_xb4_t4_score_cycles",
                    "destination_qfsa_xb4_t4_direct_mask",
                    "destination_qfsa_xb4_t4_direct_cycles",
                    "destination_qfsa_xb4_t4_residual_waves",
                    "destination_qfsa_xb4_t8_score_cycles",
                    "destination_qfsa_xb4_t8_direct_mask",
                    "destination_qfsa_xb4_t8_direct_cycles",
                    "destination_qfsa_xb4_t8_residual_waves",
                    "destination_qfsa_xb4_t8_bank_loads",
                    "destination_qfsa_xb4_t8_bank_imbalance",
                    "destination_qfsa_xb4_t8b1_score_cycles",
                    "destination_qfsa_xb4_t8b1_direct_mask",
                    "destination_qfsa_xb4_t8b1_direct_cycles",
                    "destination_qfsa_xb4_t8b1_residual_waves",
                    "destination_qfsa_xb4_t8b2_score_cycles",
                    "destination_qfsa_xb4_t8b2_direct_mask",
                    "destination_qfsa_xb4_t8b2_direct_cycles",
                    "destination_qfsa_xb4_t8b2_residual_waves",
                    "destination_qfsa_xb4_t8b3_score_cycles",
                    "destination_qfsa_xb4_t8b3_direct_mask",
                    "destination_qfsa_xb4_t8b3_direct_cycles",
                    "destination_qfsa_xb4_t8b3_residual_waves",
                    "destination_qfsa_xb4_t12_score_cycles",
                    "destination_qfsa_xb4_t12_direct_mask",
                    "destination_qfsa_xb4_t12_direct_cycles",
                    "destination_qfsa_xb4_t12_residual_waves",
                ],
                "retirement": (
                    "max destination index among valid Local5 consumers"
                ),
                "term_key": "(source,final_gate,lane)",
                "service_lower_bound": (
                    "max(product_term_count,destination_delivery_count)"
                ),
                "plane_contract": {
                    "time_planes": 2,
                    "execution": "plane_serial_drain",
                    "cross_plane_neighbor": False,
                },
            },
            "source_descriptor_contract": {
                "id": "qfit_relation_transpose_source_descriptor_v3",
                "arrays": [
                    "descriptor_group_offsets",
                    "descriptor_source_id",
                    "descriptor_source_plane",
                    "descriptor_source_y",
                    "descriptor_source_x",
                    "descriptor_k_bitmap",
                    "descriptor_incoming_gates",
                    "descriptor_valid_mask",
                ],
                "candidate_role_order": [
                    "self",
                    "up",
                    "down",
                    "left",
                    "right",
                ],
                "source_consumer_relation": [
                    "self_destination",
                    "down_destination_uses_source_as_up_candidate",
                    "up_destination_uses_source_as_down_candidate",
                    "right_destination_uses_source_as_left_candidate",
                    "left_destination_uses_source_as_right_candidate",
                ],
                "k_contract": "source self-K bitmap",
                "gate_contract": (
                    "gate from the valid destination that consumes this "
                    "source in the corresponding candidate role"
                ),
                "rtl_reference": str(
                    (
                        HW_ROOT
                        / "rtl_qfit"
                        / "qfit_relation_transpose_leaf.sv"
                    ).resolve()
                ),
                "rtl_reference_sha256": file_sha256(
                    HW_ROOT
                    / "rtl_qfit"
                    / "qfit_relation_transpose_leaf.sv"
                ),
                "rtl_dependency_bindings": {
                    name: {
                        "path": str(expected_source_binding_paths()[name].resolve()),
                        "sha256": file_sha256(
                            expected_source_binding_paths()[name]
                        ),
                    }
                    for name in (
                        "relation_transpose_rtl",
                        "retirement_scheduler_rtl",
                        "relation_sync_bank_rtl",
                        "relation_assertions",
                        "relation_sync_bank_assertions",
                        "relation_vector_generator",
                        "relation_miter_tb",
                        "relation_miter_script",
                    )
                },
            },
            "attention_score_trace_contract": {
                "id": "local5_qk_score_shiftmax_trace_v1",
                "query_array": "descriptor_q_bitmap",
                "key_array": "descriptor_k_bitmap",
                "relation": "T2x15x15_self_up_down_left_right",
                "score": "alpha_XNOR_Q7_alpha0_1_over_64_RNE",
                "shiftmax": "masked_Q8_LUT_ceil_pow2_Q1p7_RNE",
            },
            "sampling": {
                "groups_per_block_sample": (
                    self.groups_per_block_sample
                ),
                "method": POST_G0_SAMPLING_ID,
                "performance_scope": (
                    "sampled ordered groups, not full workload totals"
                ),
            },
            "groups": self.groups,
        }
        if (
            projection_contract_manifest is not None
            and projection_contract_payload is not None
        ):
            manifest.update(
                {
                    "projection_contract_file": (
                        projection_contract_manifest.name
                    ),
                    "projection_contract_file_sha256": file_sha256(
                        projection_contract_manifest
                    ),
                    "projection_contract_payload": (
                        projection_contract_payload.name
                    ),
                    "projection_contract_payload_sha256": file_sha256(
                        projection_contract_payload
                    ),
                }
            )
        manifest_path = output_dir / "ordered_term_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path, payload_path


class Local5Collector:
    def __init__(
        self,
        ordered_sink: OrderedTermTraceSink | None = None,
    ) -> None:
        self.sample_id = -1
        self.records: list[dict[str, Any]] = []
        self.modules: list[torch.nn.Module] = []
        self.ordered_sink = ordered_sink

    def attach(self, model: torch.nn.Module) -> int:
        attached = 0
        attached_pairs: set[tuple[int, int]] = set()
        for full_name, module in model.named_modules():
            cfg = getattr(module, "_h9_shiftmax_cfg", None)
            mode = str(getattr(cfg, "mode", ""))
            if "local5" not in mode and mode not in {"lr_ttx", "h66_lr"}:
                continue
            match = re.search(r"layers\.(\d+)\.swin_blocks\.(\d+)\.attn$", full_name)
            if match is None:
                if (
                    self.ordered_sink is not None
                    and self.ordered_sink.evidence_level == "post_g0"
                ):
                    raise RuntimeError(
                        "正式Local5模块名称无法解析stage/block: "
                        + full_name
                    )
                continue
            stage = int(match.group(1))
            block = int(match.group(2))
            block_pair = (stage, block)
            if (
                self.ordered_sink is not None
                and self.ordered_sink.evidence_level == "post_g0"
                and block_pair not in set(POST_G0_BLOCK_PAIRS)
            ):
                raise RuntimeError(
                    f"正式Local5出现非目标attention block: {block_pair}"
                )
            if block_pair in attached_pairs:
                raise RuntimeError(
                    f"Local5 attention block重复绑定: {block_pair}"
                )
            attached_pairs.add(block_pair)

            def callback(
                *,
                module: torch.nn.Module,
                q_event: torch.Tensor,
                k_event: torch.Tensor,
                k_orig: torch.Tensor,
                neighbor_index: torch.Tensor,
                valid: torch.Tensor,
                score_q7: torch.Tensor,
                gate: torch.Tensor,
                _name: str = full_name,
                _stage: int = stage,
                _block: int = block,
            ) -> None:
                self.records.append(
                    analyze_call(
                        name=_name,
                        stage=_stage,
                        block=_block,
                        sample_id=self.sample_id,
                        q_event=q_event,
                        k_event=k_event,
                        k_orig=k_orig,
                        neighbor_index=neighbor_index,
                        valid=valid,
                        score_q7=score_q7,
                        gate=gate,
                        ordered_sink=self.ordered_sink,
                    )
                )

            module._h9_local5_trace_collector = callback
            self.modules.append(module)
            attached += 1
        if (
            self.ordered_sink is not None
            and self.ordered_sink.evidence_level == "post_g0"
            and attached_pairs != set(POST_G0_BLOCK_PAIRS)
        ):
            missing = sorted(set(POST_G0_BLOCK_PAIRS) - attached_pairs)
            extra = sorted(attached_pairs - set(POST_G0_BLOCK_PAIRS))
            raise RuntimeError(
                f"正式Local5 block集合不匹配: missing={missing}, extra={extra}"
            )
        return attached

    def close(self) -> None:
        for module in self.modules:
            if hasattr(module, "_h9_local5_trace_collector"):
                delattr(module, "_h9_local5_trace_collector")
        self.modules.clear()


def validate_threshold_k_contract(
    k_orig: torch.Tensor | None,
    k_event: torch.Tensor,
) -> float | None:
    if k_orig is None:
        raise ValueError("post_g0 Local5 callback缺少k_orig")
    k_orig_detached = k_orig.detach()
    if not bool(torch.isfinite(k_orig_detached).all().item()):
        raise ValueError("post_g0 Local5 k_orig含非有限值")
    if bool(k_orig_detached.lt(0).any().item()):
        raise ValueError("post_g0 Local5 k_orig含负值")
    if not torch.equal(
        k_orig_detached.gt(0),
        k_event.detach().to(dtype=torch.bool),
    ):
        raise ValueError("post_g0 Local5 k_orig与k_event支持集不等价")
    positive = k_orig_detached[k_orig_detached.gt(0)]
    if positive.numel() == 0:
        return None
    amplitude = positive[0]
    if not torch.equal(positive, amplitude.expand_as(positive)):
        raise ValueError("post_g0 Local5 k_orig含多个非零幅值")
    return float(amplitude.item())


def validate_binary_k_contract(
    k_orig: torch.Tensor | None,
    k_event: torch.Tensor,
) -> float | None:
    """兼容旧调用名；正式语义为单阈值幅度事件合同。"""
    return validate_threshold_k_contract(k_orig, k_event)


def analyze_call(
    *,
    name: str,
    stage: int,
    block: int,
    sample_id: int,
    q_event: torch.Tensor,
    k_event: torch.Tensor,
    k_orig: torch.Tensor | None = None,
    neighbor_index: torch.Tensor,
    valid: torch.Tensor,
    score_q7: torch.Tensor,
    gate: torch.Tensor,
    ordered_sink: OrderedTermTraceSink | None = None,
) -> dict[str, Any]:
    if (
        ordered_sink is not None
        and ordered_sink.evidence_level == "post_g0"
    ):
        k_value_amplitude = validate_threshold_k_contract(k_orig, k_event)
    else:
        k_value_amplitude = None
    q = q_event.to(dtype=torch.bool)
    k = k_event.to(dtype=torch.bool)
    valid = valid.to(dtype=torch.bool)
    batch_windows, heads, tokens, lanes = q.shape
    candidates = int(neighbor_index.shape[-1])
    if candidates != 5:
        raise ValueError(f"Local5 profiler expects five candidates, got {candidates}")
    if tuple(gate.shape[-2:]) != (tokens, candidates):
        raise ValueError(f"unexpected Local5 gate shape: {tuple(gate.shape)}")

    k_candidates = k[:, :, neighbor_index, :]
    valid_bh = valid.view(1, 1, tokens, candidates).expand(
        batch_windows, heads, tokens, candidates
    )
    valid_lane = valid_bh.unsqueeze(-1)
    gate_code = torch.round(gate.float() * 128.0).to(dtype=torch.long).clamp(0, 256)
    score_code = torch.round(score_q7.float() * 128.0).to(dtype=torch.long)

    record: dict[str, Any] = {
        "name": name,
        "stage": stage,
        "block": block,
        "sample_id": sample_id,
        "batch_windows": batch_windows,
        "heads": heads,
        "tokens": tokens,
        "lanes": lanes,
        "valid_edges": int(valid_bh.sum().item()),
        "candidate_edges": int(valid_bh.numel()),
        "query_major_k_lane_reads": int(valid_lane.expand_as(k_candidates).sum().item()),
        "source_resident_k_lane_reads": int(k.numel()),
        "query_major_active_k_lane_reads": int((k_candidates & valid_lane).sum().item()),
        "source_resident_active_k_lanes": int(k.sum().item()),
        "k_value_positive_count": int(k.sum().item()),
        "k_value_positive_amplitude": k_value_amplitude,
        "q_zero_tokens": int((~q.any(dim=-1)).sum().item()),
        "k_zero_tokens": int((~k.any(dim=-1)).sum().item()),
        "token_heads": int(batch_windows * heads * tokens),
        "q_count_histogram": histogram(q.sum(dim=-1), minlength=lanes + 1),
        "k_count_histogram": histogram(k.sum(dim=-1), minlength=lanes + 1),
        "degree_histogram": histogram(
            valid.sum(dim=-1)
            .view(1, 1, tokens)
            .expand(batch_windows, heads, tokens),
            minlength=6,
        ),
    }

    self_k = k_candidates[..., 0, :]
    directional_valid_total = 0
    directional_delta_lanes = 0
    for direction_index, direction in enumerate(DIRECTIONS[1:], start=1):
        neighbor = k_candidates[..., direction_index, :]
        edge_valid = valid_bh[..., direction_index]
        xor = self_k ^ neighbor
        up = (~self_k) & neighbor
        down = self_k & (~neighbor)
        q_up = q & up
        q_down = q & down
        delta_count = xor.sum(dim=-1)[edge_valid]
        up_count = up.sum(dim=-1)[edge_valid]
        down_count = down.sum(dim=-1)[edge_valid]
        q_up_count = q_up.sum(dim=-1)[edge_valid]
        q_down_count = q_down.sum(dim=-1)[edge_valid]
        valid_count = int(edge_valid.sum().item())
        directional_valid_total += valid_count
        directional_delta_lanes += int(delta_count.sum().item())
        record[f"{direction}_valid_edges"] = valid_count
        record[f"{direction}_delta_lane_sum"] = int(delta_count.sum().item())
        record[f"{direction}_delta_histogram"] = histogram(
            delta_count, minlength=lanes + 1
        )
        record[f"{direction}_up_histogram"] = histogram(up_count, minlength=lanes + 1)
        record[f"{direction}_down_histogram"] = histogram(
            down_count, minlength=lanes + 1
        )
        record[f"{direction}_q_up_histogram"] = histogram(
            q_up_count, minlength=lanes + 1
        )
        record[f"{direction}_q_down_histogram"] = histogram(
            q_down_count, minlength=lanes + 1
        )
        record[f"{direction}_exact_k"] = int((delta_count == 0).sum().item())
        record[f"{direction}_neighbor_subset_self"] = int(
            (up_count == 0).sum().item()
        )
        record[f"{direction}_self_subset_neighbor"] = int(
            (down_count == 0).sum().item()
        )
    record["directional_valid_edges"] = directional_valid_total
    record["directional_delta_lane_sum"] = directional_delta_lanes
    record["direct_neighbor_lane_work"] = directional_valid_total * lanes
    record.update(
        joint_stencil_delta_stats(
            k_candidates,
            valid,
        )
    )

    valid_gate_codes = gate_code[valid_bh]
    valid_score_codes = score_code[valid_bh]
    record["valid_gate_entries"] = int(valid_gate_codes.numel())
    record["zero_gate_entries"] = int((valid_gate_codes == 0).sum().item())
    record["gate_code_histogram"] = histogram(valid_gate_codes, minlength=GATE_CODES)
    record["valid_score_histogram_offset256"] = histogram(
        (valid_score_codes + 256).clamp(0, 512), minlength=513
    )

    active = (
        k_candidates
        & valid_lane
        & gate_code.gt(0).unsqueeze(-1)
    )
    naive_edge_products = int(active.sum().item())
    record["naive_active_edge_products"] = naive_edge_products
    source_stats = source_gate_lane_stats(
        active,
        gate_code,
        neighbor_index,
    )
    if source_stats["source_gate_lane_delivery"] != naive_edge_products:
        raise RuntimeError(
            "DiSEP source-major delivery 与原始 active edge-lane work 不守恒"
        )
    if source_stats["source_gate_lane_terms"] > naive_edge_products:
        raise RuntimeError("DiSEP term 数量不能超过原始 delivery")
    record.update(source_stats)

    first_groups: list[torch.Tensor] = []
    multiplicities: list[torch.Tensor] = []
    for candidate in range(candidates):
        current_active = active[..., candidate, :]
        prior_same = torch.zeros_like(current_active)
        for prior in range(candidate):
            same_gate = gate_code[..., prior].eq(gate_code[..., candidate]).unsqueeze(-1)
            prior_same |= active[..., prior, :] & same_gate
        first = current_active & (~prior_same)
        multiplicity = torch.zeros_like(current_active, dtype=torch.long)
        for other in range(candidates):
            same_gate = gate_code[..., other].eq(gate_code[..., candidate]).unsqueeze(-1)
            multiplicity += (active[..., other, :] & same_gate).to(dtype=torch.long)
        first_groups.append(first)
        multiplicities.append(multiplicity)

    first_stack = torch.stack(first_groups, dim=-2)
    multiplicity_stack = torch.stack(multiplicities, dim=-2)
    record["destination_gate_lane_groups"] = int(first_stack.sum().item())
    multiplicity_values = multiplicity_stack[first_stack]
    record["multiplicity_histogram"] = histogram(multiplicity_values, minlength=6)

    bh = (
        torch.arange(batch_windows * heads, device=q.device, dtype=torch.long)
        .reshape(batch_windows, heads, 1, 1)
        .expand(batch_windows, heads, tokens, lanes)
    )
    lane_id = (
        torch.arange(lanes, device=q.device, dtype=torch.long)
        .view(1, 1, 1, lanes)
        .expand(batch_windows, heads, tokens, lanes)
    )
    destination_id = (
        torch.arange(tokens, device=q.device, dtype=torch.long)
        .view(1, 1, tokens, 1)
        .expand(batch_windows, heads, tokens, lanes)
    )

    offset_keys = []
    for candidate in range(candidates):
        selected = active[..., candidate, :]
        gate_id = gate_code[..., candidate].unsqueeze(-1).expand_as(lane_id)
        key = (((bh * candidates + candidate) * GATE_CODES + gate_id) * lanes) + lane_id
        offset_keys.append(key[selected])
    offset_keys_tensor = torch.cat(offset_keys) if offset_keys else torch.empty(
        0, device=q.device, dtype=torch.long
    )
    offset_unique, offset_fanout = torch.unique(
        offset_keys_tensor, return_counts=True
    )
    record["offset_multicast_terms"] = int(offset_unique.numel())
    record["offset_max_fanout"] = (
        int(offset_fanout.max().item()) if offset_fanout.numel() else 0
    )
    record["offset_fanout_histogram"] = histogram(offset_fanout, minlength=1)
    offset_keyspace = candidates * GATE_CODES * lanes
    offset_terms_per_bh = torch.bincount(
        torch.div(offset_unique, offset_keyspace, rounding_mode="floor"),
        minlength=batch_windows * heads,
    )
    record["offset_terms_per_window_head_histogram"] = histogram(offset_terms_per_bh)

    mfep_keys = []
    mfep_destinations = []
    unsafe_set_keys = []
    for candidate in range(candidates):
        selected = first_stack[..., candidate, :]
        gate_id = gate_code[..., candidate].unsqueeze(-1).expand_as(lane_id)
        mult = multiplicity_stack[..., candidate, :]
        mfep_key = (
            (((bh * GATE_CODES + gate_id) * lanes + lane_id) * 6)
            + mult
        )
        unsafe_key = ((bh * GATE_CODES + gate_id) * lanes) + lane_id
        mfep_keys.append(mfep_key[selected])
        mfep_destinations.append(destination_id[selected])
        unsafe_set_keys.append(unsafe_key[selected])
    mfep_keys_tensor = torch.cat(mfep_keys) if mfep_keys else torch.empty(
        0, device=q.device, dtype=torch.long
    )
    mfep_destinations_tensor = (
        torch.cat(mfep_destinations)
        if mfep_destinations
        else torch.empty(0, device=q.device, dtype=torch.long)
    )
    unsafe_keys_tensor = torch.cat(unsafe_set_keys) if unsafe_set_keys else torch.empty(
        0, device=q.device, dtype=torch.long
    )
    mfep_unique, mfep_fanout = torch.unique(mfep_keys_tensor, return_counts=True)
    unsafe_unique = torch.unique(unsafe_keys_tensor)
    record["mfep_multicast_terms"] = int(mfep_unique.numel())
    record["unsafe_set_multicast_terms"] = int(unsafe_unique.numel())
    record["mfep_max_fanout"] = (
        int(mfep_fanout.max().item()) if mfep_fanout.numel() else 0
    )
    record["mfep_fanout_histogram"] = histogram(mfep_fanout, minlength=1)
    record.update(
        mfep_destination_stream_stats(
            mfep_keys_tensor,
            mfep_destinations_tensor,
            tokens=tokens,
        )
    )
    if record["mfep_scalar_delivery"] != record["destination_gate_lane_groups"]:
        raise RuntimeError("MFEP destination stream 与 gate-lane group 数不守恒")
    mfep_keyspace = GATE_CODES * lanes * 6
    mfep_terms_per_bh = torch.bincount(
        torch.div(mfep_unique, mfep_keyspace, rounding_mode="floor"),
        minlength=batch_windows * heads,
    )
    record["mfep_terms_per_window_head_histogram"] = histogram(mfep_terms_per_bh)

    unique_gate_count = torch.zeros(
        (batch_windows, heads, tokens), device=q.device, dtype=torch.long
    )
    for candidate in range(candidates):
        candidate_valid = valid_bh[..., candidate] & gate_code[..., candidate].gt(0)
        prior_same = torch.zeros_like(candidate_valid)
        for prior in range(candidate):
            prior_same |= (
                valid_bh[..., prior]
                & gate_code[..., prior].gt(0)
                & gate_code[..., prior].eq(gate_code[..., candidate])
            )
        unique_gate_count += (candidate_valid & (~prior_same)).to(dtype=torch.long)
    record["active_gate_cardinality_histogram"] = histogram(
        unique_gate_count, minlength=6
    )
    if ordered_sink is not None:
        ordered_sink.capture(
            name=name,
            stage=stage,
            block=block,
            sample_id=sample_id,
            q_event=q,
            k_candidates=k_candidates,
            valid=valid,
            gate_code=gate_code,
            neighbor_index=neighbor_index,
        )
    return record


def aggregate_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count_keys = (
        "batch_windows",
        "valid_edges",
        "candidate_edges",
        "query_major_k_lane_reads",
        "source_resident_k_lane_reads",
        "query_major_active_k_lane_reads",
        "source_resident_active_k_lanes",
        "k_value_positive_count",
        "q_zero_tokens",
        "k_zero_tokens",
        "token_heads",
        "directional_valid_edges",
        "directional_delta_lane_sum",
        "direct_neighbor_lane_work",
        "valid_gate_entries",
        "zero_gate_entries",
        "naive_active_edge_products",
        "destination_gate_lane_groups",
        "offset_multicast_terms",
        "mfep_multicast_terms",
        "mfep_scalar_delivery",
        "mfep_ppdi_delivery_exact",
        "mfep_destination_continuations",
        "mfep_destination_delta_escape_b4",
        "mfep_destination_delta_escape_b6",
        "mfep_destination_delta_escape_b10",
        "unsafe_set_multicast_terms",
        "source_gate_lane_terms",
        "source_gate_lane_delivery",
        "source_instances",
        "source_active_instances",
        "dqfs_layout_supported",
        "dqfs_row_groups",
        "dqfs_row_value_product_computes",
        "joint_delta_event_sum",
        "direct_serial_score_cycle_sum",
        "qfsa_w2_score_cycle_sum",
        "qfsa_w4_score_cycle_sum",
        "qfsa_w8_score_cycle_sum",
        "qfsa_xb4_score_cycle_sum",
        "independent_w1x4_score_cycle_sum",
    )
    count_keys += tuple(
        f"qfsa_xb4_t{threshold}_score_cycle_sum"
        for threshold in (4, 8, 12)
    )
    count_keys += tuple(
        f"dqfs_lane_way_overflow_{suffix}_w{ways}"
        for ways in (2, 4, 6, 8)
        for suffix in ("groups", "terms")
    )
    result = {key: sum(int(row.get(key, 0)) for row in rows) for key in count_keys}
    result["k_value_positive_amplitudes"] = sorted(
        {
            float(row["k_value_positive_amplitude"])
            for row in rows
            if row.get("k_value_positive_amplitude") is not None
        }
    )
    for direction in DIRECTIONS[1:]:
        for suffix in (
            "valid_edges",
            "delta_lane_sum",
            "exact_k",
            "neighbor_subset_self",
            "self_subset_neighbor",
        ):
            key = f"{direction}_{suffix}"
            result[key] = sum(int(row.get(key, 0)) for row in rows)
        for suffix in (
            "delta_histogram",
            "up_histogram",
            "down_histogram",
            "q_up_histogram",
            "q_down_histogram",
        ):
            key = f"{direction}_{suffix}"
            result[key] = merge_histograms(rows, key)
    for key in (
        "q_count_histogram",
        "k_count_histogram",
        "degree_histogram",
        "gate_code_histogram",
        "valid_score_histogram_offset256",
        "multiplicity_histogram",
        "offset_fanout_histogram",
        "mfep_fanout_histogram",
        "offset_terms_per_window_head_histogram",
        "mfep_terms_per_window_head_histogram",
        "mfep_destination_delta_histogram",
        "active_gate_cardinality_histogram",
        "source_gate_lane_fanout_histogram",
        "source_gate_lane_terms_per_window_head_histogram",
        "source_gate_cardinality_histogram",
        "source_gate_cardinality_all_histogram",
        "dqfs_row_value_key_histogram",
        "dqfs_row_term_histogram",
        "dqfs_row_lane_gate_cardinality_histogram",
        "dqfs_value_chain_length_histogram",
        "joint_delta_total_histogram",
        "joint_delta_active_direction_histogram",
        "direct_serial_score_cycle_histogram",
        "qfsa_w2_score_cycle_histogram",
        "qfsa_w4_score_cycle_histogram",
        "qfsa_w8_score_cycle_histogram",
        "qfsa_xb4_score_cycle_histogram",
        "independent_w1x4_score_cycle_histogram",
    ):
        result[key] = merge_histograms(rows, key)
    for threshold in (4, 8, 12):
        key = f"qfsa_xb4_t{threshold}_score_cycle_histogram"
        result[key] = merge_histograms(rows, key)

    directional_edges = result["directional_valid_edges"]
    direct_work = result["direct_neighbor_lane_work"]
    naive_products = result["naive_active_edge_products"]
    result.update(
        {
            "records": len(rows),
            "delta_lane_density": (
                result["directional_delta_lane_sum"] / direct_work
                if direct_work
                else 0.0
            ),
            "delta_zero_edge_ratio": (
                sum(result[f"{direction}_exact_k"] for direction in DIRECTIONS[1:])
                / directional_edges
                if directional_edges
                else 0.0
            ),
            "topology_k_read_reduction": (
                1.0
                - result["source_resident_k_lane_reads"]
                / result["query_major_k_lane_reads"]
                if result["query_major_k_lane_reads"]
                else 0.0
            ),
            "active_k_read_reduction": (
                1.0
                - result["source_resident_active_k_lanes"]
                / result["query_major_active_k_lane_reads"]
                if result["query_major_active_k_lane_reads"]
                else 0.0
            ),
            "zero_gate_ratio": (
                result["zero_gate_entries"] / result["valid_gate_entries"]
                if result["valid_gate_entries"]
                else 0.0
            ),
            "multiplicity_fold_ratio": (
                result["destination_gate_lane_groups"] / naive_products
                if naive_products
                else 0.0
            ),
            "offset_term_ratio": (
                result["offset_multicast_terms"] / naive_products
                if naive_products
                else 0.0
            ),
            "mfep_term_ratio": (
                result["mfep_multicast_terms"] / naive_products
                if naive_products
                else 0.0
            ),
            "mfep_ppdi_command_reduction": (
                1.0
                - result["mfep_ppdi_delivery_exact"]
                / result["mfep_scalar_delivery"]
                if result["mfep_scalar_delivery"]
                else 0.0
            ),
            **{
                f"mfep_destination_delta_escape_ratio_b{bits}": (
                    result[f"mfep_destination_delta_escape_b{bits}"]
                    / result["mfep_destination_continuations"]
                    if result["mfep_destination_continuations"]
                    else 0.0
                )
                for bits in (4, 6, 10)
            },
            "unsafe_set_term_ratio": (
                result["unsafe_set_multicast_terms"] / naive_products
                if naive_products
                else 0.0
            ),
            "source_gate_lane_term_ratio": (
                result["source_gate_lane_terms"] / naive_products
                if naive_products
                else 0.0
            ),
            "source_active_instance_ratio": (
                result["source_active_instances"] / result["source_instances"]
                if result["source_instances"]
                else 0.0
            ),
            "dqfs_row_value_product_reduction": (
                1.0
                - result["dqfs_row_value_product_computes"]
                / result["source_gate_lane_terms"]
                if result["source_gate_lane_terms"]
                else 0.0
            ),
            **{
                f"qfsa_w{width}_score_cycle_reduction": (
                    1.0
                    - result[f"qfsa_w{width}_score_cycle_sum"]
                    / result["direct_serial_score_cycle_sum"]
                    if result["direct_serial_score_cycle_sum"]
                    else 0.0
                )
                for width in (2, 4, 8)
            },
            "qfsa_w4_vs_independent_w1x4_cycle_reduction": (
                1.0
                - result["qfsa_w4_score_cycle_sum"]
                / result["independent_w1x4_score_cycle_sum"]
                if result["independent_w1x4_score_cycle_sum"]
                else 0.0
            ),
            "qfsa_xb4_vs_independent_w1x4_cycle_reduction": (
                1.0
                - result["qfsa_xb4_score_cycle_sum"]
                / result["independent_w1x4_score_cycle_sum"]
                if result["independent_w1x4_score_cycle_sum"]
                else 0.0
            ),
            **{
                (
                    f"qfsa_xb4_t{threshold}_vs_"
                    "independent_w1x4_cycle_reduction"
                ): (
                    1.0
                    - result[
                        f"qfsa_xb4_t{threshold}_score_cycle_sum"
                    ]
                    / result["independent_w1x4_score_cycle_sum"]
                    if result["independent_w1x4_score_cycle_sum"]
                    else 0.0
                )
                for threshold in (4, 8, 12)
            },
            "delta_p50": hist_quantile(
                merge_direction_histograms(result, "delta_histogram"), 0.50
            ),
            "delta_p95": hist_quantile(
                merge_direction_histograms(result, "delta_histogram"), 0.95
            ),
            "delta_p99": hist_quantile(
                merge_direction_histograms(result, "delta_histogram"), 0.99
            ),
            "multiplicity_mean": hist_mean(result["multiplicity_histogram"]),
            "multiplicity_p95": hist_quantile(
                result["multiplicity_histogram"], 0.95
            ),
            "gate_cardinality_mean": hist_mean(
                result["active_gate_cardinality_histogram"]
            ),
            "gate_cardinality_p95": hist_quantile(
                result["active_gate_cardinality_histogram"], 0.95
            ),
            "offset_terms_per_window_head_p95": hist_quantile(
                result["offset_terms_per_window_head_histogram"], 0.95
            ),
            "mfep_terms_per_window_head_p95": hist_quantile(
                result["mfep_terms_per_window_head_histogram"], 0.95
            ),
            "source_gate_lane_terms_per_window_head_p95": hist_quantile(
                result["source_gate_lane_terms_per_window_head_histogram"],
                0.95,
            ),
            "source_gate_cardinality_mean": hist_mean(
                result["source_gate_cardinality_histogram"]
            ),
            "source_gate_cardinality_p95": hist_quantile(
                result["source_gate_cardinality_histogram"],
                0.95,
            ),
            "source_gate_cardinality_all_mean": hist_mean(
                result["source_gate_cardinality_all_histogram"]
            ),
            "dqfs_row_value_keys_mean": hist_mean(
                result["dqfs_row_value_key_histogram"]
            ),
            "dqfs_row_value_keys_p95": hist_quantile(
                result["dqfs_row_value_key_histogram"],
                0.95,
            ),
            "dqfs_row_terms_p95": hist_quantile(
                result["dqfs_row_term_histogram"],
                0.95,
            ),
            "dqfs_row_terms_max": max(
                0, len(result["dqfs_row_term_histogram"]) - 1
            ),
            "dqfs_lane_gate_cardinality_p95": hist_quantile(
                result["dqfs_row_lane_gate_cardinality_histogram"],
                0.95,
            ),
            "dqfs_lane_gate_cardinality_max": max(
                0,
                len(
                    result[
                        "dqfs_row_lane_gate_cardinality_histogram"
                    ]
                )
                - 1,
            ),
            "dqfs_value_chain_length_p95": hist_quantile(
                result["dqfs_value_chain_length_histogram"],
                0.95,
            ),
            "dqfs_value_chain_length_max": max(
                0, len(result["dqfs_value_chain_length_histogram"]) - 1
            ),
        }
    )
    gate_hist = result["gate_code_histogram"]
    gate_total = sum(gate_hist)
    result["gate_binary_nonzero_digits_mean"] = (
        sum(bitcount_integer(code) * count for code, count in enumerate(gate_hist))
        / gate_total
        if gate_total
        else 0.0
    )
    return result


def merge_direction_histograms(summary: dict[str, Any], suffix: str) -> list[int]:
    keys = [f"{direction}_{suffix}" for direction in DIRECTIONS[1:]]
    size = max((len(summary.get(key, [])) for key in keys), default=0)
    result = [0] * size
    for key in keys:
        for index, value in enumerate(summary.get(key, [])):
            result[index] += int(value)
    return result


def write_markdown(
    path: Path,
    *,
    config: Path,
    checkpoint: Path,
    samples: int,
    total: dict[str, Any],
    by_stage: dict[str, dict[str, Any]],
    cohort: dict[str, Any],
    evidence_level: str,
    eval_protocol: dict[str, Any],
    module_counts: dict[str, int],
    checkpoint_load_audit: dict[str, Any] | None,
    threshold_semantics: dict[str, Any],
) -> None:
    if evidence_level == "post_g0":
        boundary_lines = [
            "- 数值边界：这是绑定最终 config/checkpoint 身份的 post-G0 profile；",
            "  ordered trace 仍属于 workload/transaction 证据，不自动等价于 RTL、PPA",
            "  或 full-encoder speedup。",
        ]
    else:
        boundary_lines = [
            f"- 数值边界：这是 `{evidence_level}` 探索 profile。边界 mask 修复会改变前层输出，",
            "  因而后续 block 的 Q/K、K-XOR、exact/subset、gate 和 term 均须在",
            "  G0/G1 后复跑。只有 738-edge 固定拓扑和由拓扑推出的理论读取比例",
            "  不依赖数值 P0。",
        ]
    lines = [
        "# Local5 硬件特征 Profile",
        "",
        f"- 配置：`{config}`",
        f"- checkpoint：`{checkpoint}`",
        f"- samples：`{samples}`",
        f"- evidence level：`{evidence_level}`",
        f"- 评估协议：`{eval_protocol}`",
        f"- 模块数量：`{module_counts}`",
        f"- 权重加载：`{checkpoint_load_audit}`",
        f"- ATLIF 阈值训练/部署语义：`{threshold_semantics}`",
        *boundary_lines,
        "",
        "## 总结",
        "",
        "| 指标 | 数值 | 证据用途 |",
        "|---|---:|---|",
        f"| 四方向 K XOR lane density | {total['delta_lane_density']:.6%} | RCSD |",
        f"| 四方向 exact-K edge | {total['delta_zero_edge_ratio']:.6%} | Prosperity exact reuse |",
        f"| delta count p50/p95/p99 | {total['delta_p50']}/{total['delta_p95']}/{total['delta_p99']} | direct/delta 模式 |",
        f"| QFSA-W2 score cycle reduction | {total['qfsa_w2_score_cycle_reduction']:.6%} | joint direction residual 模型 |",
        f"| QFSA-W4 score cycle reduction | {total['qfsa_w4_score_cycle_reduction']:.6%} | joint direction residual 模型 |",
        f"| QFSA-W4 vs 4xW1 cycle reduction | {total['qfsa_w4_vs_independent_w1x4_cycle_reduction']:.6%} | 同总residual宽度强基线 |",
        f"| XBF-QFSA vs 4xW1 cycle reduction | {total['qfsa_xb4_vs_independent_w1x4_cycle_reduction']:.6%} | XOR-bank蝶形分配强候选 |",
        f"| XBF-QFSA-T8 vs 4xW1 cycle reduction | {total['qfsa_xb4_t8_vs_independent_w1x4_cycle_reduction']:.6%} | 可综合threshold router |",
        f"| QFSA-W8 score cycle reduction | {total['qfsa_w8_score_cycle_reduction']:.6%} | joint direction residual 模型 |",
        f"| source-resident 理论 K-bit read 减少 | {total['topology_k_read_reduction']:.6%} | line buffer |",
        f"| source-resident 活动 K-lane read 减少 | {total['active_k_read_reduction']:.6%} | source multicast |",
        f"| 有效 gate=0 | {total['zero_gate_ratio']:.6%} | 预修复 gate 指标 |",
        f"| gate cardinality mean/p95 | {total['gate_cardinality_mean']:.4f}/{total['gate_cardinality_p95']} | Shiftmax5/term |",
        f"| multiplicity mean/p95 | {total['multiplicity_mean']:.4f}/{total['multiplicity_p95']} | MFEP |",
        f"| offset term / active edge product | {total['offset_term_ratio']:.6%} | 低风险 DCTF 基线 |",
        f"| MFEP term / active edge product | {total['mfep_term_ratio']:.6%} | 多重集折叠 |",
        f"| DiSEP source-gate-lane term / active edge product | {total['source_gate_lane_term_ratio']:.6%} | source-major projection |",
        f"| active source 比例 | {total['source_active_instance_ratio']:.6%} | source 调度占用 |",
        f"| active source gate cardinality mean/p95 | {total['source_gate_cardinality_mean']:.4f}/{total['source_gate_cardinality_p95']} | DiSEP product reuse |",
        f"| all-source gate cardinality mean | {total['source_gate_cardinality_all_mean']:.4f} | 含空 source，不能与上一行混用 |",
        f"| DQFS row value product reduction | {total['dqfs_row_value_product_reduction']:.6%} | `(lane,gate,weight_epoch)`跨source精确复用 |",
        f"| DQFS row value keys mean/p95 | {total['dqfs_row_value_keys_mean']:.4f}/{total['dqfs_row_value_keys_p95']} | lane-local目录+term SRAM |",
        f"| DQFS row terms p95/max | {total['dqfs_row_terms_p95']}/{total['dqfs_row_terms_max']} | 双context容量 |",
        f"| DQFS lane gate cardinality p95/max | {total['dqfs_lane_gate_cardinality_p95']}/{total['dqfs_lane_gate_cardinality_max']} | 目录way数 |",
        f"| DQFS value chain length p95/max | {total['dqfs_value_chain_length_p95']}/{total['dqfs_value_chain_length_max']} | product驻留长度 |",
        f"| DQFS 6-way overflow groups/terms | {total['dqfs_lane_way_overflow_groups_w6']}/{total['dqfs_lane_way_overflow_terms_w6']} | exact RAW fallback压力 |",
        f"| unsafe set term / active edge product | {total['unsafe_set_term_ratio']:.6%} | 仅显示错误去重上界 |",
        f"| gate 二进制非零位均值 | {total['gate_binary_nonzero_digits_mean']:.4f} | shift-add/CSD 前筛 |",
        "",
        "## 分 Stage",
        "",
        "| Stage | XOR density | exact-K | QFSA-W4 vs serial-direct | QFSA-W4 vs 4xW1 | XBF-oracle vs 4xW1 | XBF-T8 vs 4xW1 | active K read reduction | MFEP term ratio | multiplicity p95 | MFEP term/window-head p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in sorted(by_stage, key=int):
        row = by_stage[stage]
        lines.append(
            f"| S{stage} | {row['delta_lane_density']:.6%} | "
            f"{row['delta_zero_edge_ratio']:.6%} | "
            f"{row['qfsa_w4_score_cycle_reduction']:.6%} | "
            f"{row['qfsa_w4_vs_independent_w1x4_cycle_reduction']:.6%} | "
            f"{row['qfsa_xb4_vs_independent_w1x4_cycle_reduction']:.6%} | "
            f"{row['qfsa_xb4_t8_vs_independent_w1x4_cycle_reduction']:.6%} | "
            f"{row['active_k_read_reduction']:.6%} | "
            f"{row['mfep_term_ratio']:.6%} | {row['multiplicity_p95']} | "
            f"{row['mfep_terms_per_window_head_p95']} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- `topology_k_read_reduction` 比较 query-major 五邻域重复取 K 与每个 source K",
        "  在行缓冲中读取一次；尚未加入 SRAM 端口、halo 和控制能量。",
        "- `offset term` 按 self/N/S/E/W 分开，目的 bitmap 无重复，最容易复用现有 DCTF。",
        "- `MFEP term` 使用 `(gate,lane,multiplicity)`，保持 Local5 多重集语义。",
        "- `source_gate_lane_terms` 使用 `(source token, final gate, lane)`，",
        "  delivery 必须与 active edge-lane product 守恒；用于 DiSEP 强基线。",
        "- `source_gate_cardinality` 默认只在 active source 上统计；报告另列",
        "  all-source 均值与 active-source 比例，禁止通过排除空 source 夸大收益。",
        "- `DQFS row value` 使用 `(lane,gate)`值键，默认同一profile记录内",
        "  `weight_epoch`不变；它不包含source，source只决定destination链。",
        "- DQFS reduction只统计product生成机会，不等于周期或能耗降低；",
        "  目录、term SRAM、重排反压和fallback必须由RTL/PPA计入。",
        "- `QFSA-W* score cycle reduction` 枚举四方向 direct/residual 选择，",
        "  口径为一个共享32-lane anchor/direct popcount加W-lane带方向残差后端；",
        "  不包含 compactor、SRAM、Shiftmax、projection 和控制周期。",
        "- `unsafe set term` 丢弃 multiplicity，只是错误 OR 去重能达到的乐观下界，",
        "  不允许作为可实现结果。",
        "- 本报告是 workload profile，不是 cycle、PPA 或端到端加速结果。",
        "- JSON 已记录 ordered dataset sample-key manifest 的 SHA256："
        f"`{cohort['sample_key_sha256']}`；跨模型比较时必须核对该 hash。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--ordered-groups-per-block-sample",
        type=int,
        default=0,
        help="uniformly sample this many ordered window-head groups per block/sample",
    )
    parser.add_argument(
        "--ordered-evidence-level",
        choices=("synthetic", "pre_g0", "post_g0"),
        default="pre_g0",
    )
    parser.add_argument(
        "--run-identity",
        type=Path,
        help="正式post-G0运行身份文件；post_g0时必需",
    )
    args = parser.parse_args()

    config, device = base_profile.load_config(args.config)
    deployment_contract = deployment_contract_from_config(config)
    run_identity: dict[str, Any] | None = None
    if (
        args.ordered_groups_per_block_sample > 0
        and args.ordered_evidence_level == "post_g0"
    ):
        validate_post_g0_export_contract(deployment_contract)
        if args.samples != POST_G0_SAMPLES:
            raise ValueError("正式post_g0 profile必须samples=100")
        if args.ordered_groups_per_block_sample < 4:
            raise ValueError("正式post_g0 profile每block-sample至少抽4组")
        if args.run_identity is None:
            raise ValueError("正式post_g0 profile必须绑定run identity")
        run_identity = load_post_g0_run_identity(
            args.run_identity,
            config=args.config,
            checkpoint=args.checkpoint,
            samples=args.samples,
            groups_per_block_sample=(
                args.ordered_groups_per_block_sample
            ),
        )
    dataset = base_profile.DSECDatasetLite(
        config,
        file_list="valid",
        stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1),
    )
    dataset_indices = (
        stratified_dataset_indices(dataset.files, args.samples)
        if args.ordered_evidence_level == "post_g0"
        else list(range(min(args.samples, len(dataset))))
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=dataset_indices,
        drop_last=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )
    transform_valid = None
    if config["loader"].get("crop") is not None:
        transform_valid = base_profile.Compose(
            [
                base_profile.CenterCrop(
                    (config["loader"]["crop"][0], config["loader"]["crop"][1])
                )
            ]
        )
    model = base_profile.build_model(config, args.checkpoint, device)
    checkpoint_load_audit = base_profile.validate_h9_load_audit(model, config)
    module_counts = base_profile.h9_module_counts(model)
    threshold_semantics = base_profile.threshold_training_semantics(config)
    bn_policy = config.get("test", {}).get("bn_policy", "running")
    bn_modules_changed = base_profile.configure_batch_norm_evaluation(
        model, bn_policy
    )
    (
        projection_contract_manifest,
        projection_contract_payload,
        projection_contract,
    ) = write_checkpoint_projection_contract(
        model,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        bn_policy=bn_policy,
    )
    ordered_sink = (
        OrderedTermTraceSink(
            groups_per_block_sample=(
                args.ordered_groups_per_block_sample
            ),
            evidence_level=args.ordered_evidence_level,
        )
        if args.ordered_groups_per_block_sample > 0
        else None
    )
    collector = Local5Collector(ordered_sink=ordered_sink)
    attached = collector.attach(model)
    if attached != 12:
        raise RuntimeError(f"expected 12 Local5 attention blocks, attached {attached}")
    print(f"[local5-profile] attached {attached} blocks on {device}", flush=True)

    processed = 0
    sample_keys: list[str] = []
    sequence_keys: list[str] = []
    try:
        with torch.no_grad():
            for chunk, mask, label in loader:
                if processed >= args.samples:
                    break
                base_profile.functional.reset_net(model)
                collector.sample_id = processed
                dataset_index = dataset_indices[processed]
                file_row = dataset.files[dataset_index]
                file_names = file_row_values(file_row)
                sample_keys.append("|".join(str(item) for item in file_names))
                sequence_keys.append(sequence_key_from_file_row(file_row))
                x, _, _ = base_profile.preprocess_chunk(
                    config, chunk, label, mask, transform_valid, device
                )
                model(x)
                processed += 1
                print(
                    f"[local5-profile] processed {processed}/{args.samples}",
                    flush=True,
                )
    finally:
        collector.close()

    formal_context: dict[str, Any] | None = None
    if args.ordered_evidence_level == "post_g0":
        if processed != POST_G0_SAMPLES:
            raise RuntimeError(
                f"post_g0样本不足: expected 100, processed {processed}"
            )
        if ordered_sink is None or args.run_identity is None:
            raise RuntimeError("post_g0 ordered sink/run identity缺失")
        qualification = post_g0_qualification(
            ordered_sink.groups,
            processed_samples=processed,
            attached_blocks=attached,
            groups_per_block_sample=(
                args.ordered_groups_per_block_sample
            ),
            run_identity_bound=run_identity is not None,
        )
        if not qualification["qualified"]:
            failed = [
                name
                for name, passed in qualification["checks"].items()
                if not passed
            ]
            raise RuntimeError(
                "post_g0 qualification失败: " + ", ".join(failed)
            )
        formal_context = {
            "run_identity": str(args.run_identity.resolve()),
            "processed_samples": processed,
            "attached_blocks": attached,
        }

    total = aggregate_records(collector.records)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in collector.records:
        grouped[int(row["stage"])].append(row)
    by_stage = {
        str(stage): aggregate_records(rows)
        for stage, rows in sorted(grouped.items())
    }
    cohort = {
        "count": len(sample_keys),
        "sample_key_sha256": string_list_sha256(sample_keys),
        "sequence_key_sha256": string_list_sha256(sequence_keys),
        "first_sample_key": sample_keys[0] if sample_keys else "",
        "last_sample_key": sample_keys[-1] if sample_keys else "",
        "dataset_sampling_id": POST_G0_DATASET_SAMPLING_ID,
        "dataset_size": len(dataset),
        "dataset_indices": dataset_indices,
        "dataset_indices_sha256": int_list_sha256(dataset_indices),
        "sequence_counts": dict(
            sorted(
                (key, sequence_keys.count(key))
                for key in set(sequence_keys)
            )
        ),
    }
    eval_protocol = {
        "resolution": list(config["loader"]["resolution"]),
        "crop": config["loader"].get("crop"),
        "window_size": list(config["swin_transformer"]["window_size"]),
        "pretrained_window_size": config["swin_transformer"].get(
            "pretrained_window_size"
        ),
        "tokens_per_window": math.prod(
            int(value)
            for value in config["swin_transformer"]["window_size"]
        ),
        "remap": config["loader"].get("remap"),
        "bn_policy": bn_policy,
        "bn_modules_changed": bn_modules_changed,
        "eval_batch_size": 1,
        "num_workers": args.num_workers,
    }
    result = {
        "schema": "local5_hardware_features_v1",
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "samples": processed,
        "module_counts": module_counts,
        "checkpoint_load_audit": checkpoint_load_audit,
        "threshold_training_semantics": threshold_semantics,
        "eval_protocol": eval_protocol,
        "projection_contract": {
            "manifest": str(projection_contract_manifest.resolve()),
            "manifest_sha256": file_sha256(projection_contract_manifest),
            "payload": str(projection_contract_payload.resolve()),
            "payload_sha256": file_sha256(projection_contract_payload),
            "numeric_scope": projection_contract["numeric_scope"],
        },
        "cohort": cohort,
        "profile_features": [
            "exact_mfep_term_destination_conservation",
            "parity_aware_ppdi_delivery",
            "destination_delta_histogram",
            "destination_delta_escape_b4_b6_b10",
            "source_gate_lane_terms",
            "source_gate_cardinality",
        ],
        "records": collector.records,
        "summary": total,
        "by_stage": by_stage,
        "evidence_boundary": {
            "structural_p0_independent": [
                "fixed Local5 edge topology",
                "query-major versus source-resident all-lane read-count model",
            ],
            "pre_g0_exploratory_reprofile_required": [
                "Q/K density",
                "directional K XOR",
                "exact/subset relation",
                "active-K read model",
                "gate histogram",
                "gate cardinality",
                "multiplicity",
                "offset/MFEP term counts",
                "MFEP destination delta/escape",
                "parity-aware PPDI delivery",
            ],
            "cohort_alignment": "ordered dataset sample-key manifest recorded",
            "not_claimed": ["cycles", "area", "power", "EDP", "full-encoder speedup"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "local5_hardware_features.json"
    md_path = args.output_dir / "local5_hardware_features.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        md_path,
        config=args.config,
        checkpoint=args.checkpoint,
        samples=processed,
        total=total,
        by_stage=by_stage,
        cohort=cohort,
        evidence_level=args.ordered_evidence_level,
        eval_protocol=eval_protocol,
        module_counts=module_counts,
        checkpoint_load_audit=checkpoint_load_audit,
        threshold_semantics=threshold_semantics,
    )
    print(f"[local5-profile] wrote {json_path}", flush=True)
    print(f"[local5-profile] wrote {md_path}", flush=True)
    if ordered_sink is not None:
        manifest_path, payload_path = ordered_sink.write(
            output_dir=args.output_dir,
            config=args.config,
            checkpoint=args.checkpoint,
            cohort=cohort,
            sample_keys=sample_keys,
            sequence_keys=sequence_keys,
            dataset_indices=dataset_indices,
            dataset_size=len(dataset),
            full_resolution=config["loader"].get("crop") is None,
            software_contract=deployment_contract,
            threshold_semantics=threshold_semantics,
            projection_contract_manifest=projection_contract_manifest,
            projection_contract_payload=projection_contract_payload,
            formal_context=formal_context,
        )
        print(
            f"[local5-profile] wrote ordered trace {manifest_path}",
            flush=True,
        )
        print(
            f"[local5-profile] wrote ordered payload {payload_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
