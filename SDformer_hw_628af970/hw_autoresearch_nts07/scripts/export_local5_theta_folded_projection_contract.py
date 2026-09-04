#!/usr/bin/env python3
"""导出先折叠 K-ATLIF theta、再量化的 Local5 投影部署合同。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from audit_local5_k_threshold_checkpoint import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
sys.path.insert(0, str(EXP / "entrypoints"))

from h67_bit_trace import quantize_projection_weight_dyadic  # noqa: E402


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
DEFAULT_CHECKPOINT = (
    EXP
    / "results/dsec_fullres_paper_w15_h66d_local5_ep29_ft30_bs2_20260728"
    / "checkpoint_epoch29.pth"
)
DEFAULT_OUTPUT = ROOT / "results/local5_theta_folded_projection_contract_20260805"
WEIGHT_PATTERN = re.compile(
    r"^(?P<base>.*\.layers\.(?P<stage>\d+)\.swin_blocks\."
    r"(?P<block>\d+))\.attn\.proj\.weight$"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_theta_folded_contract(
    state: Mapping[str, torch.Tensor],
    *,
    output_dir: Path,
    checkpoint: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """从 state_dict 构造 checkpoint 绑定的 theta-folded v2 合同。"""

    arrays: dict[str, np.ndarray] = {}
    blocks: list[dict[str, Any]] = []
    observed: set[tuple[int, int]] = set()
    total_weight_entries = 0
    total_weight_mismatch = 0
    total_scale_entries = 0
    total_scale_mismatch = 0

    for weight_name, raw_weight in state.items():
        match = WEIGHT_PATTERN.match(weight_name)
        if match is None:
            continue
        stage = int(match.group("stage"))
        block = int(match.group("block"))
        pair = (stage, block)
        if pair not in set(POST_G0_BLOCK_PAIRS):
            continue
        if pair in observed:
            raise ValueError(f"投影合同重复 block: {pair}")

        base = match.group("base")
        threshold_name = f"{base}.attn.sn_k.spiking_neuron.thresh"
        if threshold_name not in state:
            raise ValueError(f"缺少 K-ATLIF theta: {threshold_name}")
        threshold = state[threshold_name].detach().float().cpu().reshape(-1)
        if threshold.numel() != 1:
            raise ValueError(f"K-ATLIF theta 不是标量: {threshold_name}")
        theta = float(threshold.item())
        if not np.isfinite(theta) or theta <= 0.0:
            raise ValueError(f"K-ATLIF theta 非法: {threshold_name}={theta}")

        weight = raw_weight.detach().float().cpu()
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError(f"projection weight 不是方阵: {weight_name}")
        if weight.shape[1] % 32:
            raise ValueError(f"projection 输入维度不能被 head_dim=32 整除: {weight_name}")
        effective_weight = weight * theta
        raw_int8, raw_scale = quantize_projection_weight_dyadic(weight)
        folded_int8, folded_scale = quantize_projection_weight_dyadic(
            effective_weight
        )
        weight_mismatch = int(np.count_nonzero(raw_int8 != folded_int8))
        scale_mismatch = int(np.count_nonzero(raw_scale != folded_scale))
        total_weight_entries += int(raw_int8.size)
        total_weight_mismatch += weight_mismatch
        total_scale_entries += int(raw_scale.size)
        total_scale_mismatch += scale_mismatch

        bias_name = f"{base}.attn.proj.bias"
        bias = (
            torch.zeros(weight.shape[0], dtype=torch.float32)
            if bias_name not in state
            else state[bias_name].detach().float().cpu()
        )
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"projection bias shape 不匹配: {bias_name}")

        prefix = f"s{stage}_b{block}"
        arrays[f"{prefix}_theta_float32"] = np.asarray([theta], dtype=np.float32)
        arrays[f"{prefix}_weight_float32"] = weight.numpy()
        arrays[f"{prefix}_effective_weight_float32"] = effective_weight.numpy()
        arrays[f"{prefix}_weight_int8"] = folded_int8
        arrays[f"{prefix}_weight_scale_exp2"] = folded_scale
        arrays[f"{prefix}_bias_float32"] = bias.numpy()
        blocks.append(
            {
                "stage": stage,
                "block": block,
                "module": f"{base}.attn",
                "prefix": prefix,
                "weight_name": weight_name,
                "theta_name": threshold_name,
                "theta": theta,
                "weight_shape": list(weight.shape),
                "heads": int(weight.shape[1] // 32),
                "head_dim": 32,
                "bias_present": bias_name in state,
                "weight_scale_exp2_min": int(folded_scale.min()),
                "weight_scale_exp2_max": int(folded_scale.max()),
                "raw_vs_folded_weight_int8_mismatch": weight_mismatch,
                "raw_vs_folded_scale_exp2_mismatch": scale_mismatch,
            }
        )
        observed.add(pair)

    expected = set(POST_G0_BLOCK_PAIRS)
    if observed != expected:
        raise ValueError(
            "theta-folded 投影合同 block 集合不完整: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    blocks.sort(key=lambda row: (row["stage"], row["block"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "checkpoint_projection_contract_theta_folded.npz"
    np.savez_compressed(payload_path, **arrays)
    manifest = {
        "schema": "local5_checkpoint_projection_contract_v2",
        "status": "THETA_FOLDED_WEIGHT_CONTRACT",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "payload_file": payload_path.name,
        "payload_sha256": file_sha256(payload_path),
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
        "runtime_datapath": (
            "K remains a 1-bit event; no runtime theta multiplier or event-width increase"
        ),
        "numeric_scope": (
            "checkpoint-bound per-head INT8 partial accumulators before cross-head "
            "reduction, bias, no-running BatchNorm, requantization, residual, or decoder"
        ),
        "raw_vs_folded": {
            "weight_int8_mismatch": total_weight_mismatch,
            "weight_int8_entries": total_weight_entries,
            "scale_exp2_mismatch": total_scale_mismatch,
            "scale_exp2_entries": total_scale_entries,
        },
        "provenance": {
            "exporter": str(Path(__file__).resolve()),
            "exporter_sha256": file_sha256(Path(__file__).resolve()),
            "quantizer": str(
                (EXP / "entrypoints/h67_bit_trace.py").resolve()
            ),
            "quantizer_sha256": file_sha256(
                EXP / "entrypoints/h67_bit_trace.py"
            ),
        },
    }
    manifest_path = output_dir / "checkpoint_projection_contract_theta_folded.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path, payload_path, manifest


def write_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    mismatch = manifest["raw_vs_folded"]
    rows = manifest["blocks"]
    lines = [
        "# Local5 theta 折叠投影部署合同导出",
        "",
        "## 结论",
        "",
        f"- checkpoint SHA256：`{manifest['checkpoint_sha256']}`；",
        f"- block：{len(rows)}；",
        "- 正式顺序：`W_eff = theta_K × W_float`，再做逐输出通道 dyadic INT8 量化；",
        "- 运行时 K 保持 1-bit，不增加 theta 乘法器和事件带宽；",
        f"- 相对错误的 `quantize(W)`，INT8 码变化：{mismatch['weight_int8_mismatch']}/{mismatch['weight_int8_entries']}；",
        f"- scale exponent 变化：{mismatch['scale_exp2_mismatch']}/{mismatch['scale_exp2_entries']}。",
        "",
        "## 逐 Block 绑定",
        "",
        "| Stage/Block | theta | INT8 mismatch | scale mismatch | payload prefix |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| S{row['stage']}B{row['block']} | {row['theta']:.10f} | "
            f"{row['raw_vs_folded_weight_int8_mismatch']} | "
            f"{row['raw_vs_folded_scale_exp2_mismatch']} | `{row['prefix']}` |"
        )
    lines += [
        "",
        "## 证据边界",
        "",
        "该产物证明离线部署权重合同和 provenance 可实现，证据等级为 `[模型]`。",
        "它尚未接入现有生产 profile/vector producer，也尚未用新 fullres rank-1",
        "重跑，因此不能宣称正式 Local5 attention-to-projection RTL exact。",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint)
    manifest_path, payload_path, manifest = build_theta_folded_contract(
        model.state_dict(),
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
    )
    write_report(args.output_dir, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "payload": str(payload_path),
                "blocks": len(manifest["blocks"]),
                **manifest["raw_vs_folded"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
