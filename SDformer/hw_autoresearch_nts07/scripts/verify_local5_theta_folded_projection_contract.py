#!/usr/bin/env python3
"""独立从checkpoint重算Local5 theta折叠投影合同。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from audit_local5_k_threshold_checkpoint import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ENTRYPOINTS = REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

from h67_bit_trace import quantize_projection_weight_dyadic  # noqa: E402


EXPECTED_STAGE_DEPTHS = (2, 2, 6, 2)
EXPECTED_STAGE_CHANNELS = (96, 192, 384, 768)
EXPECTED_STAGE_HEADS = (3, 6, 12, 24)
EXPECTED_HEAD_DIM = 32
TOPOLOGY_CONTRACT = "local5_swin_2_2_6_2_c96_192_384_768_h3_6_12_24_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_block_contracts() -> list[dict[str, Any]]:
    rows = []
    for stage, depth in enumerate(EXPECTED_STAGE_DEPTHS):
        channels = EXPECTED_STAGE_CHANNELS[stage]
        heads = EXPECTED_STAGE_HEADS[stage]
        for block in range(depth):
            module = (
                "sttmultires_unet.encoders.swin3d.layers."
                f"{stage}.swin_blocks.{block}.attn"
            )
            rows.append(
                {
                    "stage": stage,
                    "block": block,
                    "module": module,
                    "prefix": f"s{stage}_b{block}",
                    "weight_name": f"{module}.proj.weight",
                    "theta_name": f"{module}.sn_k.spiking_neuron.thresh",
                    "bias_name": f"{module}.proj.bias",
                    "weight_shape": [channels, channels],
                    "heads": heads,
                    "head_dim": EXPECTED_HEAD_DIM,
                    "bias_present": True,
                }
            )
    return rows


def validate_manifest_topology(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("blocks")
    expected = expected_block_contracts()
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Local5投影合同不是固定12-block拓扑")
    if manifest.get("topology_contract") not in (None, TOPOLOGY_CONTRACT):
        raise ValueError("Local5投影合同topology_contract错误")

    for index, (row, topology) in enumerate(zip(rows, expected)):
        if not isinstance(row, dict):
            raise ValueError(f"Local5投影合同block[{index}]不是object")
        for field in (
            "stage",
            "block",
            "module",
            "prefix",
            "weight_name",
            "theta_name",
            "weight_shape",
            "heads",
            "head_dim",
            "bias_present",
        ):
            if row.get(field) != topology[field]:
                raise ValueError(
                    f"Local5投影合同拓扑映射错误: block[{index}].{field}="
                    f"{row.get(field)!r}, expected={topology[field]!r}"
                )
        if "bias_name" in row and row["bias_name"] != topology["bias_name"]:
            raise ValueError(
                f"Local5投影合同拓扑映射错误: block[{index}].bias_name"
            )
    return expected


def verify_contract(
    manifest_path: Path,
    payload_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "local5_checkpoint_projection_contract_v2"
        or manifest.get("status") != "THETA_FOLDED_WEIGHT_CONTRACT"
        or manifest.get("payload_file") != payload_path.name
        or manifest.get("payload_sha256") != file_sha256(payload_path)
    ):
        raise ValueError("theta-folded projection contract元数据或SHA错误")
    topology = validate_manifest_topology(manifest)

    checkpoint = Path(str(manifest.get("checkpoint", ""))).resolve()
    if (
        not checkpoint.is_file()
        or manifest.get("checkpoint_sha256") != file_sha256(checkpoint)
    ):
        raise ValueError("theta-folded projection contract的checkpoint绑定失效")
    state = load_checkpoint(checkpoint).state_dict()

    expected_keys: set[str] = set()
    verified_entries = 0
    verified_scales = 0
    verified_biases = 0
    with np.load(payload_path) as payload:
        for row, topology_row in zip(manifest["blocks"], topology):
            prefix = str(row["prefix"])
            weight_name = str(row.get("weight_name", ""))
            theta_name = str(row.get("theta_name", ""))
            bias_name = str(topology_row["bias_name"])
            if weight_name not in state or theta_name not in state:
                raise ValueError(f"checkpoint缺少合同参数: {weight_name}/{theta_name}")

            raw_weight = state[weight_name].detach().float().cpu()
            if list(raw_weight.shape) != topology_row["weight_shape"]:
                raise ValueError(f"checkpoint投影权重形状不符合固定拓扑: {weight_name}")
            theta_flat = state[theta_name].detach().float().cpu().reshape(-1)
            if theta_flat.numel() != 1:
                raise ValueError(f"K-ATLIF theta不是标量: {theta_name}")
            theta = float(theta_flat.item())
            effective_weight = raw_weight * theta
            expected_int8, expected_scale = quantize_projection_weight_dyadic(
                effective_weight
            )
            expected_bias = (
                np.zeros(raw_weight.shape[0], dtype=np.float32)
                if bias_name not in state
                else state[bias_name].detach().float().cpu().numpy()
            )
            expected = {
                f"{prefix}_theta_float32": np.asarray([theta], dtype=np.float32),
                f"{prefix}_weight_float32": raw_weight.numpy(),
                f"{prefix}_effective_weight_float32": effective_weight.numpy(),
                f"{prefix}_weight_int8": expected_int8,
                f"{prefix}_weight_scale_exp2": expected_scale,
                f"{prefix}_bias_float32": expected_bias,
            }
            expected_keys.update(expected)
            for key, value in expected.items():
                if key not in payload.files or not np.array_equal(payload[key], value):
                    raise ValueError(f"checkpoint独立重算与投影payload不一致: {key}")
            if float(row.get("theta", float("nan"))) != theta:
                raise ValueError(f"manifest theta与checkpoint不一致: {theta_name}")
            verified_entries += int(expected_int8.size)
            verified_scales += int(expected_scale.size)
            verified_biases += int(expected_bias.size)

        unexpected = set(payload.files) - expected_keys
        missing = expected_keys - set(payload.files)
        if unexpected or missing:
            raise ValueError(
                f"projection payload数组集错误: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

    return {
        "schema": "local5_theta_folded_projection_independent_recompute_v1",
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "payload": str(payload_path.resolve()),
        "payload_sha256": file_sha256(payload_path),
        "blocks": len(manifest["blocks"]),
        "topology_contract": TOPOLOGY_CONTRACT,
        "topology_mapping": "PASS_FIXED_12_BLOCK_ABI",
        "arrays": len(expected_keys),
        "verified_weight_int8_entries": verified_entries,
        "verified_scale_entries": verified_scales,
        "verified_bias_entries": verified_biases,
        "quantizer": str((ENTRYPOINTS / "h67_bit_trace.py").resolve()),
        "quantizer_sha256": file_sha256(ENTRYPOINTS / "h67_bit_trace.py"),
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": file_sha256(Path(__file__).resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_contract(args.manifest, args.payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
