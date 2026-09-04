"""导出H67真实Q/K、Q1.7 gate与投影权重的位级trace。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1


def quantize_projection_weight_dyadic(
    weight: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """逐输出通道生成带2幂scale的对称INT8候选编码。"""

    weight_cpu = weight.detach().float().cpu()
    if weight_cpu.ndim != 2:
        raise ValueError("projection weight必须是二维[out,in]")
    absmax = weight_cpu.abs().amax(dim=1)
    exponent = torch.zeros_like(absmax, dtype=torch.int16)
    nonzero = absmax > 0
    if nonzero.any():
        exponent[nonzero] = torch.ceil(
            torch.log2(absmax[nonzero] / 127.0)
        ).to(torch.int16)
    scale = torch.pow(2.0, exponent.to(torch.float32))
    code = torch.round(weight_cpu / scale.unsqueeze(1)).clamp(-127, 127)
    return (
        code.to(torch.int8).numpy(),
        exponent.numpy(),
    )


class AttentionBitTraceWriter:
    """将选定样本和窗口直接写成紧凑NPZ，避免在JSON中展开位图。"""

    def __init__(
        self,
        output_dir: Path,
        *,
        sample_limit: int = 1,
        windows_per_call: int = 1,
        first_block_only: bool = True,
    ) -> None:
        if sample_limit <= 0 or windows_per_call <= 0:
            raise ValueError("sample_limit和windows_per_call必须为正")
        self.output_dir = Path(output_dir)
        self.sample_limit = int(sample_limit)
        self.windows_per_call = int(windows_per_call)
        self.first_block_only = bool(first_block_only)
        self.records: list[dict[str, Any]] = []
        self._written_keys: set[tuple[int, str]] = set()
        self.run_context: dict[str, Any] = {}

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def bind_run_context(self, context: dict[str, Any]) -> None:
        """Bind the trace to the exact evaluation/checkpoint provenance."""

        self.run_context = dict(context)
        if self.records:
            self.write_manifest()

    def capture(
        self,
        *,
        name: str,
        sample_id: int,
        sample_key: str,
        module: torch.nn.Module,
        q_orig: torch.Tensor,
        k_orig: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        if sample_id < 0 or sample_id >= self.sample_limit:
            return
        if self.first_block_only and ".B0.attn" not in name:
            return
        key = (int(sample_id), str(name))
        if key in self._written_keys:
            raise RuntimeError(f"同一样本attention trace重复写入: {key}")
        if q_orig.ndim != 5 or q_orig.shape[0] != 2:
            raise ValueError("q_orig必须是[2,B,H,N,D]")
        if k_orig.ndim != 4:
            raise ValueError("k_orig必须是[B,H,2N,D]")
        batch_windows, heads, total_tokens, lanes = map(int, k_orig.shape)
        spatial_tokens = int(q_orig.shape[3])
        if tuple(q_orig.shape[1:]) != (
            batch_windows,
            heads,
            spatial_tokens,
            lanes,
        ):
            raise ValueError("q_orig与k_orig的窗口/head/lane布局不一致")
        if total_tokens != 2 * spatial_tokens:
            raise ValueError("k_orig token维必须是q_orig空间token的两倍")
        if gate.ndim != 4 or tuple(gate.shape) != (
            batch_windows,
            heads,
            total_tokens,
            1,
        ):
            raise ValueError("gate必须是[B,H,2N,1]")

        selected_windows = min(self.windows_per_call, batch_windows)
        q_bits = q_orig[:, :selected_windows].detach().gt(0).cpu().numpy()
        k_bits = (
            k_orig[:selected_windows]
            .detach()
            .gt(0)
            .reshape(selected_windows, heads, 2, spatial_tokens, lanes)
            .permute(2, 0, 1, 3, 4)
            .cpu()
            .numpy()
        )
        gate_q17 = torch.round(
            gate[:selected_windows].detach().float().squeeze(-1) * 128.0
        ).to(torch.int64)
        if gate_q17.lt(0).any() or gate_q17.gt(256).any():
            raise ValueError("gate Q1.7码越界[0,256]")
        gate_q17_np = gate_q17.to(torch.int32).cpu().numpy().astype(np.uint16)

        projection = getattr(module, "proj", None)
        weight = getattr(projection, "weight", None)
        if weight is None:
            raise ValueError(f"{name}缺少proj.weight")
        weight_float = weight.detach().float().cpu().numpy()
        weight_int8, weight_scale_exp2 = quantize_projection_weight_dyadic(weight)
        bias = getattr(projection, "bias", None)
        if bias is None:
            bias_float = np.zeros(weight.shape[0], dtype=np.float32)
        else:
            bias_float = bias.detach().float().cpu().numpy()
        accumulator_scale = np.exp2(weight_scale_exp2.astype(np.float32)) / 128.0
        bias_acc = np.rint(bias_float / accumulator_scale).astype(np.int64)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(".", "_").replace("/", "_")
        path = self.output_dir / f"sample{sample_id}_{safe_name}.npz"
        np.savez_compressed(
            path,
            q_shape=np.asarray(q_bits.shape, dtype=np.int32),
            q_bits_packed=np.packbits(q_bits.reshape(-1), bitorder="little"),
            k_shape=np.asarray(k_bits.shape, dtype=np.int32),
            k_bits_packed=np.packbits(k_bits.reshape(-1), bitorder="little"),
            gate_q17=gate_q17_np,
            projection_weight_float32=weight_float,
            projection_weight_int8=weight_int8,
            projection_weight_scale_exp2=weight_scale_exp2,
            projection_bias_float32=bias_float,
            projection_bias_acc_int64=bias_acc,
        )
        with path.open("rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        record = {
            "sample_id": int(sample_id),
            "sample_key": str(sample_key),
            "name": str(name),
            "file": str(path),
            "sha256": digest,
            "windows_total": batch_windows,
            "windows_captured": selected_windows,
            "heads": heads,
            "spatial_tokens": spatial_tokens,
            "temporal_tokens": total_tokens,
            "lanes": lanes,
            "q_active_bits": int(q_bits.sum()),
            "k_active_bits": int(k_bits.sum()),
            "gate_nonzero": int(np.count_nonzero(gate_q17_np)),
            "gate_min": int(gate_q17_np.min()),
            "gate_max": int(gate_q17_np.max()),
            "weight_shape": list(weight_float.shape),
            "weight_scale_exp2_min": int(weight_scale_exp2.min()),
            "weight_scale_exp2_max": int(weight_scale_exp2.max()),
            "quantization_contract": (
                "候选：逐输出通道对称INT8，scale=2^e，"
                "e=ceil(log2(max_abs/127))；需valid825部署验证后方可冻结"
            ),
        }
        self.records.append(record)
        self._written_keys.add(key)
        self.write_manifest()

    def write_manifest(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stages = sorted(
            {
                int(record["name"].split(".")[0][1:])
                for record in self.records
                if record["name"].startswith("S")
            }
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "evidence": "[真实网络bit trace]+[候选dyadic INT8权重量化]",
            "sample_limit": self.sample_limit,
            "windows_per_call": self.windows_per_call,
            "first_block_only": self.first_block_only,
            "run_context": self.run_context,
            "records": self.records,
            "coverage": {
                "stages": stages,
                "stage_count": len(stages),
                "record_count": len(self.records),
                "four_stage_complete": stages == [0, 1, 2, 3],
            },
            "limits": [
                "Q/K与gate来自真实推理张量，不是统计塑形构造",
                "projection浮点权重来自checkpoint",
                "INT8权重编码尚未经过valid825精度验证，不能标为冻结部署合同",
            ],
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def estimate_record_bytes(
    *, heads: int, spatial_tokens: int, lanes: int, dim: int, windows: int = 1
) -> int:
    """用于采集前预算，不包含NPZ压缩收益。"""

    bit_bytes = math.ceil(2 * windows * heads * spatial_tokens * lanes / 8)
    gate_bytes = windows * heads * 2 * spatial_tokens * 2
    weight_bytes = dim * dim * (4 + 1)
    scale_bias_bytes = dim * (2 + 4 + 8)
    return 2 * bit_bytes + gate_bytes + weight_bytes + scale_bias_bytes
