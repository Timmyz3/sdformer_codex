#!/usr/bin/env python3
"""审计 H67/H68/TTX profile100 的来源、样本顺序和 compact 一致性。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_hit_flow_ordered_profiles import decode_count_trace


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_COMPACT = ROOT / "results/profile100_compact_arch_stats_20260714.json"
DEFAULT_JSON = ROOT / "results/profile100_provenance_audit_20260715.json"
DEFAULT_MD = ROOT / "results/profile100_provenance_audit_20260715.md"
PROFILE_DIRS = {
    "H67": "h67_ep19_ttb_delta_cycle_v2_profile100_20260713",
    "H68": "h68_ep19_ttb_delta_cycle_v2_profile100_20260713",
    "TTX": "ttx_ep2_ttb_delta_cycle_v2_profile100_20260713",
}
PAIR_FIELDS = (
    "pair_empty_ratio",
    "pair_motion_zero_ratio",
    "pair_update_zero_ratio",
    "pair_kzero_both_ratio",
    "pair_both_active_ratio",
    "token_kzero_ratio",
    "projection_baseline_active_lanes",
    "projection_class_channel_terms_h67",
    "projection_gate_class_channel_terms_deploy",
    "row_active_projection_gate_classes_mean_deploy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def stage_projection(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_stage = (2640, 1440, 2160, 480)
    out = []
    for index, stage in enumerate(raw["summary"]["h60_by_stage"]):
        out.append(
            {
                "stage": int(stage["group"]),
                "rows_per_frame": rows_by_stage[index],
                "zaf_kzero_token_ratio": float(stage["zaf_kzero_token_ratio"]),
                "zaf_active_entries_mean": float(stage["zaf_active_entries_mean"]),
                "zaf_fold_classes_mean": float(stage["zaf_fold_classes_mean"]),
                "ttb2_empty_ratio": float(stage["ttb2_empty_ratio"]),
                "q_active_density": float(stage["q_active_density"]),
                "k_active_density": float(stage["k_active_density"]),
            }
        )
    return out


def equal_number(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= 1e-12
    return left == right


def audit_model(name: str, compact_model: dict[str, Any]) -> dict[str, Any]:
    profile = (
        REPO
        / "neuron_experiments/H9_bipolar_self_attention/results"
        / PROFILE_DIRS[name]
        / "nts11_hardware_p0_profile.json"
    )
    raw = json.loads(profile.read_text(encoding="utf-8"))
    raw_pair = raw["summary"]["binary_temporal_pairs"]
    compact_pair = compact_model["binary_temporal_pairs"]
    mismatches = []
    for field in PAIR_FIELDS:
        if field not in compact_pair:
            continue
        if not equal_number(raw_pair[field], compact_pair[field]):
            mismatches.append(
                {"field": field, "raw": raw_pair[field], "compact": compact_pair[field]}
            )
    raw_stages = stage_projection(raw)
    for raw_stage, compact_stage in zip(raw_stages, compact_model["stages"]):
        for field, raw_value in raw_stage.items():
            if not equal_number(raw_value, compact_stage[field]):
                mismatches.append(
                    {
                        "field": f"stage{raw_stage['stage']}.{field}",
                        "raw": raw_value,
                        "compact": compact_stage[field],
                    }
                )

    config = resolve_repo_path(raw["config"])
    checkpoint = resolve_repo_path(raw["checkpoint"])
    samples = raw["summary"]["sample_records"]
    sample_order = [
        {
            "sample_id": row.get("sample_id"),
            "sample_key": row.get("sample_key"),
            "sequence_key": row.get("sequence_key"),
        }
        for row in samples
    ]
    trace_rows = sum(
        len(decode_count_trace(record["projection_baseline_active_lanes_ordered_trace"]))
        for record in raw["summary"]["h60_records"]
    )
    return {
        "experiment": raw["experiment"],
        "samples": int(raw["samples"]),
        "ordered_trace": bool(raw["ordered_trace"]),
        "trace_rows": trace_rows,
        "expected_rows": int(raw["samples"]) * 6720,
        "profile": str(profile.relative_to(REPO)),
        "profile_bytes": profile.stat().st_size,
        "profile_sha256": sha256_file(profile),
        "config": str(config.relative_to(REPO)),
        "config_sha256": sha256_file(config),
        "checkpoint": str(checkpoint.relative_to(REPO)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_order_sha256": sha256_json(sample_order),
        "sample_first": sample_order[0],
        "sample_last": sample_order[-1],
        "compact_mismatches": mismatches,
        "compact_exact": not mismatches,
        "key_metrics": {field: raw_pair[field] for field in PAIR_FIELDS},
    }


def render_md(result: dict[str, Any]) -> str:
    lines = [
        "# Profile100 来源与一致性审计",
        "",
        "本报告只做 CPU 文件审计，不重新运行网络推理。SHA-256 用于冻结论文统计来源。",
        "",
        "| 模型 | 样本 | ordered | compact一致 | 原始JSON | checkpoint | 样本顺序摘要 |",
        "|---|---:|:---:|:---:|---|---|---|",
    ]
    for name, model in result["models"].items():
        lines.append(
            f"| {name} | {model['samples']} | {'是' if model['ordered_trace'] else '否'} | "
            f"{'是' if model['compact_exact'] else '否'} | `{model['profile_sha256'][:16]}` | "
            f"`{model['checkpoint_sha256'][:16]}` | `{model['sample_order_sha256'][:16]}` |"
        )
    lines += [
        "",
        "## H67 关键统计复核",
        "",
        "| 指标 | 原始值 |",
        "|---|---:|",
    ]
    h67 = result["models"]["H67"]["key_metrics"]
    for field in PAIR_FIELDS:
        value = h67[field]
        if isinstance(value, float):
            text = f"{value:.8f}"
        else:
            text = str(value)
        lines.append(f"| `{field}` | {text} |")
    lines += [
        "",
        "## 结论与边界",
        "",
        f"- 三模型 compact 逐字段结果：`{'PASS' if result['all_compact_exact'] else 'FAIL'}`；",
        f"- 三模型 ordered trace 行数检查：`{'PASS' if result['all_trace_rows_exact'] else 'FAIL'}`；",
        "- 本报告验证文件来源和汇总一致性，不证明周期模型、RTL 或 DC 结果；",
        "- 完整 64 位 SHA-256 保存在同名 JSON，Markdown 仅显示前 16 位。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    models = {
        name: audit_model(name, compact["models"][name])
        for name in ("H67", "H68", "TTX")
    }
    result = {
        "schema_version": 1,
        "compact": str(args.compact),
        "compact_sha256": sha256_file(args.compact),
        "all_compact_exact": all(model["compact_exact"] for model in models.values()),
        "all_trace_rows_exact": all(
            model["trace_rows"] == model["expected_rows"] for model in models.values()
        ),
        "models": models,
        "evidence": "[文件审计]+[prof汇总复核]",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md.write_text(render_md(result), encoding="utf-8")
    print(args.json)
    print(args.md)
    passed = result["all_compact_exact"] and result["all_trace_rows_exact"]
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
