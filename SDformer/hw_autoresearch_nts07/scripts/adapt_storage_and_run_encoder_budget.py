#!/usr/bin/env python3
"""Storage schema 适配器 + 全 encoder budget 入口（新文件，不改 GPT budget 脚本）。

问题（docs/76）：
  误把 `h67_h68_storage_ablation.json`（键：状态/结果）喂给
  `model_hit_flow_full_encoder_budget.py`，后者需要
  `storage["models"]["H67"]["atlif_execution_graph"]` → KeyError。

本适配器：
1) 识别 contract / ablation / 未知 schema；
2) ablation 自动重定向到 encoder_storage_contract；
3) 校验 profile / sops / runtime-profile 路径；
4) 调用既有 `build_model` / `write_markdown` 产出 JSON+MD。

不修改 GPT 的 model_hit_flow_full_encoder_budget.py。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "results/h67_h68_encoder_storage_contract.json"
DEFAULT_ABLATION = ROOT / "results/h67_h68_storage_ablation.json"
DEFAULT_PROFILE = ROOT / "results/h67_h68_profile100_arch_features.json"
DEFAULT_JSON = ROOT / "results/hit_flow_full_encoder_budget_adapted_20260715.json"
DEFAULT_MD = ROOT / "results/hit_flow_full_encoder_budget_adapted_20260715.md"

# 可选 sops / runtime 候选（按存在性挑选）
SOPS_CANDIDATES = [
    ROOT.parent
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h68_castling_ttx_aux050_s360_retry_20260711_202914/profile_deploy_valid825"
    / "sops_summary.json",
    ROOT / "results/sops_summary.json",
]
RUNTIME_CANDIDATES = [
    ROOT.parent
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713/nts11_hardware_p0_profile.json",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_storage_schema(data: dict[str, Any]) -> str:
    models = data.get("models")
    if isinstance(models, dict) and "H67" in models:
        h67 = models["H67"]
        if isinstance(h67, dict) and "atlif_execution_graph" in h67:
            return "encoder_storage_contract"
        return "models_present_but_incomplete"
    if "状态" in data or "结果" in data:
        return "storage_ablation_yosys"
    return "unknown"


def resolve_storage(
    storage_path: Path,
    *,
    contract_fallback: Path = DEFAULT_CONTRACT,
) -> tuple[dict[str, Any], Path, str, list[str]]:
    """返回 (storage_dict, resolved_path, schema_label, notes)."""
    notes: list[str] = []
    if not storage_path.is_file():
        raise FileNotFoundError(storage_path)
    data = load_json(storage_path)
    schema = classify_storage_schema(data)
    if schema == "encoder_storage_contract":
        notes.append(f"直接使用 contract：{storage_path}")
        return data, storage_path, schema, notes
    if schema == "storage_ablation_yosys":
        notes.append(
            f"输入为 Yosys 存储消融表（无 models 键）：{storage_path}；"
            f"自动重定向到 {contract_fallback}"
        )
        if not contract_fallback.is_file():
            raise FileNotFoundError(
                f"ablation 无法适配为 budget 输入，且 fallback 不存在：{contract_fallback}"
            )
        contract = load_json(contract_fallback)
        cschema = classify_storage_schema(contract)
        if cschema != "encoder_storage_contract":
            raise ValueError(f"fallback 仍不是 contract schema: {cschema}")
        return contract, contract_fallback, "redirected_ablation_to_contract", notes
    if schema == "models_present_but_incomplete":
        raise ValueError(
            f"storage 有 models 但缺 atlif_execution_graph：{storage_path}"
        )
    raise ValueError(f"无法识别 storage schema：{storage_path} keys={list(data.keys())}")


def pick_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def validate_profile(profile: dict[str, Any]) -> None:
    if "results" not in profile:
        raise ValueError("profile 缺 results[]（需要 port_aware_pipeline_dse）")
    found = False
    for row in profile["results"]:
        if row.get("model") == "H67":
            dse = row.get("whole", {}).get("port_aware_pipeline_dse")
            if not isinstance(dse, dict) or not dse:
                raise ValueError("H67 whole.port_aware_pipeline_dse 缺失")
            found = True
            break
    if not found:
        raise ValueError("profile.results 中无 H67")


def validate_sops(sops: dict[str, Any]) -> None:
    if "estimated_total_sops" not in sops or "dense_ops" not in sops:
        raise ValueError("sops 需含 estimated_total_sops 与 dense_ops")


def main() -> int:
    # Allow running as script without installing package
    sys.path.insert(0, str(ROOT / "scripts"))
    from model_hit_flow_full_encoder_budget import build_model, write_markdown

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="可为 contract 或 ablation；ablation 自动重定向",
    )
    parser.add_argument("--contract-fallback", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--sops", type=Path, default=None)
    parser.add_argument("--runtime-profile", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--try-ablation-first",
        action="store_true",
        help="演示：先读 ablation 再适配（回归 docs/76 失败路径）",
    )
    args = parser.parse_args()

    storage_arg = DEFAULT_ABLATION if args.try_ablation_first else args.storage
    storage, resolved_storage, schema_label, notes = resolve_storage(
        storage_arg, contract_fallback=args.contract_fallback
    )

    if not args.profile.is_file():
        raise FileNotFoundError(args.profile)
    profile = load_json(args.profile)
    validate_profile(profile)

    sops_path = args.sops or pick_existing(SOPS_CANDIDATES)
    if sops_path is None:
        # Minimal synthetic fallback so CPU-only path still runs with clear label
        notes.append("未找到 sops_summary.json，使用 contract 衍生代理 sops（低证据）")
        h67 = storage["models"]["H67"]
        live_macs = int(h67["atlif_execution_graph"]["live_temporal_macs_per_frame"])
        sops = {
            "estimated_total_sops": live_macs,
            "dense_ops": live_macs * 4,
            "source": "synthetic_from_storage_contract_live_temporal_macs",
        }
        sops_path = None
    else:
        sops = load_json(sops_path)
        validate_sops(sops)
        notes.append(f"sops：{sops_path}")

    runtime_path = args.runtime_profile or pick_existing(RUNTIME_CANDIDATES)
    runtime = load_json(runtime_path) if runtime_path and runtime_path.is_file() else None
    if runtime is not None:
        notes.append(f"runtime-profile：{runtime_path}")
    else:
        notes.append("无 runtime-profile：使用 legacy sops 空间代理")

    result = build_model(storage, profile, sops, runtime)
    result["adapter"] = {
        "schema_resolved": schema_label,
        "storage_input_arg": str(storage_arg),
        "storage_resolved": str(resolved_storage),
        "profile": str(args.profile),
        "sops": str(sops_path) if sops_path else "synthetic",
        "runtime_profile": str(runtime_path) if runtime_path and runtime else None,
        "notes": notes,
        "gpu_required": False,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.md, result)
    # Append adapter footer to md
    footer = [
        "",
        "## 适配器说明（本轮新增）",
        "",
        f"- storage 输入参数：`{storage_arg}`",
        f"- 解析 schema：`{schema_label}`",
        f"- 实际使用 storage：`{resolved_storage}`",
        f"- profile：`{args.profile}`",
        f"- sops：`{sops_path if sops_path else 'synthetic'}`",
        f"- runtime：`{runtime_path if runtime else None}`",
        f"- GPU：不需要",
        "",
        "### 笔记",
        "",
    ]
    for note in notes:
        footer.append(f"- {note}")
    footer.append("")
    with args.md.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(footer))

    # Also emit a tiny schema map doc snippet path
    map_path = ROOT / "results/storage_schema_adapter_map_20260715.md"
    map_path.write_text(
        "\n".join(
            [
                "# Storage schema 适配映射",
                "",
                "| 输入文件 | 识别 schema | budget 是否可直接用 | 适配动作 |",
                "|---|---|---|---|",
                f"| `h67_h68_encoder_storage_contract.json` | encoder_storage_contract | 是 | 直通 |",
                f"| `h67_h68_storage_ablation.json` | storage_ablation_yosys | **否**（无 models） | 重定向到 contract |",
                "",
                "budget 必需字段：`storage.models.H67.atlif_execution_graph`、"
                "`activation_evidence.long_skip_elements_s0_s2`；"
                "`profile.results[].whole.port_aware_pipeline_dse`；"
                "`sops.estimated_total_sops` / `dense_ops`。",
                "",
                f"本轮运行：schema=`{schema_label}` → `{args.json.name}` / `{args.md.name}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    n_cfg = len(result.get("configurations", []))
    n_pass = sum(1 for r in result["configurations"] if r.get("passes_30fps_guarded_serial"))
    print(args.json)
    print(args.md)
    print(map_path)
    print(f"schema={schema_label} configs={n_cfg} pass30fps={n_pass} gpu_required=False")
    for note in notes:
        print(" note:", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
