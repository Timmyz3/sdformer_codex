#!/usr/bin/env python3
"""由 H3/H12 页回收实测推导 Local5 H24 phase store 的资源准入预算。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOKENS = 450
LANES = 32
OUT_DIM = 32
RSS_LIMIT_KB = 512 * 1024
GUARD = 1.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case(path: Path, heads: int) -> tuple[dict[str, Any], dict[str, Any]]:
    complete_path = path / "complete.json"
    manifest_path = path / "store/manifest.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        complete.get("status") != "PASS_SEALED_STREAMING_MMAP_CANARY_NOT_G0"
        or complete.get("formal_g0") != "DENY"
        or complete.get("identity", {}).get("heads") != heads
        or manifest.get("identity", {}).get("heads") != heads
        or complete.get("verified_metrics", {}).get("expanded_rows") != manifest.get("expanded_rows")
        or complete.get("verified_metrics", {}).get("negative_cases_passed") != 10
        or complete.get("verified_metrics", {}).get("source_trace_only_pass") is not True
    ):
        raise ValueError(f"H{heads} sealed case contract differs")
    return complete, manifest


def instance_counts(heads: int) -> dict[str, int]:
    return {
        "prefix": 1,
        "head_seed": heads,
        "inter_head_gap": heads * (heads - 1),
        "head_accumulate": heads * (heads - 1),
        "tile_tail": heads,
        "tile_transition": heads - 1,
        "suffix": 1,
    }


def structural_counts(heads: int, templates: dict[str, int]) -> dict[str, int]:
    counts = instance_counts(heads)
    rows = sum(counts[name] * templates[name] for name in templates)
    payloads = (
        1
        + heads * heads * TOKENS
        + heads * heads * LANES * OUT_DIM
        + heads * TOKENS * OUT_DIM
    )
    return {"heads": heads, "rows": rows, "instances": sum(counts.values()), "payloads": payloads}


def linear_predict(x0: float, y0: float, x1: float, y1: float, target: float) -> dict[str, float]:
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    value = intercept + slope * target
    return {"slope": slope, "intercept": intercept, "predicted": value, "guarded": value * GUARD}


def projected_array_bytes(
    manifest: dict[str, Any], target_rows: int, target_instances: int,
    target_payloads: int,
) -> int:
    total = 0
    for name, entry in manifest["arrays"].items():
        dtype = np.dtype(entry["dtype"])
        if name == "payload_dictionary":
            length = target_payloads
        elif name in {"instance_class_code", "instance_tile", "instance_head"}:
            length = target_instances
        elif name == "patch_offsets":
            length = target_instances + 1
        elif name.startswith("patch_"):
            length = target_rows
        else:
            length = entry["shape"][0]
        header_bytes = entry["file_bytes"] - entry["nbytes"]
        if header_bytes < 0:
            raise ValueError(f"negative npy header bytes for {name}")
        total += length * dtype.itemsize + header_bytes
    return total


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    h24 = report["h24_model"]
    rss = report["rss_model"]
    cpu = report["cpu_model"]
    lines = [
        "# Local5 H24 Phase Array Store 资源准入预算",
        "",
        "> 证据：`[模型]+[资源实测]`；仅用于验证基础设施资源准入，不是 RTL 性能、片上存储或 ASIC PPA。",
        "",
        "## 1. 结构规模",
        "",
        "| 指标 | H24 预算 |",
        "|---|---:|",
        f"| heads | {h24['heads']:,} |",
        f"| phase rows | {h24['rows']:,} |",
        f"| phase instances | {h24['instances']:,} |",
        f"| 唯一 payload | {h24['payloads']:,} |",
        f"| array store 文件字节 | {h24['array_store_file_bytes']:,} |",
        f"| 保守 source trace 字节（64 B/row） | {h24['conservative_trace_bytes']:,} |",
        "",
        "结构行数由 H12 已验证的七类模板长度和 H24 的实例公式精确推导；payload 数由 relation、weight、final 三类身份空间加 `-` 常量推导，并已在 H3/H12 反证一致。",
        "",
        "## 2. RSS 与时间",
        "",
        "| 项 | 线性预测 | 20% 保护 | 512 MiB 门槛 |",
        "|---|---:|---:|---|",
        f"| generator RSS | {rss['generator']['predicted']/1024:.1f} MiB | {rss['generator']['guarded']/1024:.1f} MiB | {'PASS' if rss['generator']['guarded'] <= RSS_LIMIT_KB else 'FAIL'} |",
        f"| source-only verifier RSS | {rss['source_only_verifier']['predicted']/1024:.1f} MiB | {rss['source_only_verifier']['guarded']/1024:.1f} MiB | {'PASS' if rss['source_only_verifier']['guarded'] <= RSS_LIMIT_KB else 'FAIL'} |",
        f"| generator user CPU | {cpu['generator_user_seconds']['predicted']/60:.1f} min | {cpu['generator_user_seconds']['guarded']/60:.1f} min | 非 wall-time |",
        f"| source-only verifier user CPU | {cpu['source_only_verifier_user_seconds']['predicted']/60:.1f} min | {cpu['source_only_verifier_user_seconds']['guarded']/60:.1f} min | 非 wall-time |",
        "",
        "## 3. 裁决",
        "",
        f"- 资源准入：**{report['resource_admission']}**。",
        "- H24 仍必须生成真实 identity-service RTL trace，并重新通过逐行 SHA、完整 identity、10 类负例和 RSS 门槛。",
        "- 该预算不能把 H24、formal G0、full encoder 或 DATE 架构性能写成已完成。",
        "- RSS 是 H3/H12 两点线性模型；payload 字典是主要剩余常驻项，若 H24 实测超限必须改为外排/排序字典，不能放宽门槛。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3", type=Path, default=ROOT / "results/local5_h3_phase_array_store_v2_smoke_20260812")
    parser.add_argument("--h12", type=Path, default=ROOT / "results/local5_h12_phase_array_store_v2_20260812")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/local5_phase_array_store_h24_budget_v1_20260812")
    args = parser.parse_args()
    h3_path = args.h3.resolve()
    h12_path = args.h12.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    h3_complete, h3_manifest = load_case(h3_path, 3)
    h12_complete, h12_manifest = load_case(h12_path, 12)
    templates = {
        name: value["template_rows"]
        for name, value in h12_manifest["class_stats"].items()
    }
    if any(
        h3_manifest["class_stats"][name]["template_rows"] != length
        for name, length in templates.items()
    ):
        raise ValueError("H3/H12 template lengths differ")
    h3 = structural_counts(3, templates)
    h12 = structural_counts(12, templates)
    h24 = structural_counts(24, templates)
    for expected, manifest in ((h3, h3_manifest), (h12, h12_manifest)):
        if (
            expected["rows"] != manifest["expanded_rows"]
            or expected["instances"] != manifest["instances"]
            or expected["payloads"] != manifest["payload_dictionary_entries"]
        ):
            raise ValueError("structural model does not reproduce sealed case")
    h24["array_store_file_bytes"] = projected_array_bytes(
        h12_manifest, h24["rows"], h24["instances"], h24["payloads"]
    )
    h24["conservative_trace_bytes"] = h24["rows"] * 64 + len(",".join([
        "cycle", "event", "tile", "head", "source", "lane", "out", "delay",
        "index", "origin", "payload",
    ])) + 1

    h3_metrics = h3_complete["verified_metrics"]
    h12_metrics = h12_complete["verified_metrics"]
    rss = {
        "generator": linear_predict(
            h3["payloads"], h3_metrics["generator_max_rss_kb"],
            h12["payloads"], h12_metrics["generator_max_rss_kb"], h24["payloads"],
        ),
        "source_only_verifier": linear_predict(
            h3["payloads"], h3_metrics["source_only_verifier_max_rss_kb"],
            h12["payloads"], h12_metrics["source_only_verifier_max_rss_kb"], h24["payloads"],
        ),
    }
    cpu = {
        "generator_user_seconds": linear_predict(
            h3["rows"], h3_complete["resources"]["generator"]["user_seconds"],
            h12["rows"], h12_complete["resources"]["generator"]["user_seconds"], h24["rows"],
        ),
        "source_only_verifier_user_seconds": linear_predict(
            h3["rows"], h3_complete["resources"]["verifier_source_only"]["user_seconds"],
            h12["rows"], h12_complete["resources"]["verifier_source_only"]["user_seconds"], h24["rows"],
        ),
    }
    admission = (
        "CONDITIONAL_ADMIT_H24_RESOURCE_ONLY"
        if all(value["guarded"] <= RSS_LIMIT_KB for value in rss.values())
        else "DENY_H24_RESOURCE_BUDGET"
    )
    output.mkdir(parents=True)
    report = {
        "schema": "local5_phase_array_store_h24_budget_v1",
        "status": "PASS_MODEL_NOT_H24_EXECUTION" if admission.startswith("CONDITIONAL") else "FAIL_MODEL",
        "evidence": "[模型]+[H3/H12资源实测]",
        "formal_g0": "DENY",
        "h3_reproduced": h3,
        "h12_reproduced": h12,
        "h24_model": h24,
        "rss_limit_kb": RSS_LIMIT_KB,
        "guard_factor": GUARD,
        "rss_model": rss,
        "cpu_model": cpu,
        "resource_admission": admission,
        "bindings": {
            "h3_complete_sha256": sha256(h3_path / "complete.json"),
            "h3_manifest_sha256": sha256(h3_path / "store/manifest.json"),
            "h12_complete_sha256": sha256(h12_path / "complete.json"),
            "h12_manifest_sha256": sha256(h12_path / "store/manifest.json"),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "boundary": [
            "H24 结构规模是精确模型，RSS/CPU 是 H3/H12 两点外推",
            "资源条件准入不等于 H24 RTL、formal G0、full encoder 或 ASIC PPA",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(output / "report.md", report)
    print(json.dumps({"status": report["status"], "resource_admission": admission}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
