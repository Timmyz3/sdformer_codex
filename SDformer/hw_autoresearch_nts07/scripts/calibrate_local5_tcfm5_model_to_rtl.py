#!/usr/bin/env python3
"""在同一 Local5 group cohort 上分账 exact-port 模型与 RTL 周期。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from analyze_local5_active_tcfm5_postg0 import analyze_descriptor_chunk


GROUP_RE = re.compile(
    r"^GROUP .* group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"active=(?P<active>\d+) .* memory_wait=(?P<memory_wait>\d+) "
    r"terms=(?P<terms>\d+) updates=(?P<updates>\d+)"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_group_log(path: Path, expected_groups: int) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = GROUP_RE.match(line)
        if match:
            rows.append({key: int(value) for key, value in match.groupdict().items()})
    if len(rows) != expected_groups:
        raise ValueError(f"{path} group数为{len(rows)}，期望{expected_groups}")
    if [row["group"] for row in rows] != list(range(expected_groups)):
        raise ValueError(f"{path} group顺序不连续")
    return rows


def exact_port_for_selection(
    vector_manifest: dict[str, object], payload: np.lib.npyio.NpzFile
) -> dict[str, np.ndarray]:
    offsets = payload["descriptor_group_offsets"]
    names = ("active_sources", "linear5_cycles", "tcfm5_cycles")
    totals = {name: [] for name in names}
    for row in vector_manifest["selection"]["rows"]:
        index = int(row["input_group_index"])
        start, stop = int(offsets[index]), int(offsets[index + 1])
        result = analyze_descriptor_chunk(
            np.asarray(payload["descriptor_incoming_gates"][start:stop]),
            np.asarray(payload["descriptor_valid_mask"][start:stop]),
            np.asarray(payload["descriptor_k_bitmap"][start:stop]),
            np.asarray(payload["descriptor_source_plane"][start:stop]),
            np.asarray(payload["descriptor_source_y"][start:stop]),
            np.asarray(payload["descriptor_source_x"][start:stop]),
            height=15,
            width=15,
        )
        for name in names:
            totals[name].append(int(result[name].sum()))
    return {name: np.asarray(values, dtype=np.int64) for name, values in totals.items()}


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("ratio denominator必须为正")
    return numerator / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rtl_report_path = args.rtl_dir / "report.json"
    rtl_complete_path = args.rtl_dir / "complete.json"
    vector_manifest_path = args.vector_dir / "manifest.json"
    rtl_report = json.loads(rtl_report_path.read_text(encoding="utf-8"))
    rtl_complete = json.loads(rtl_complete_path.read_text(encoding="utf-8"))
    vector_manifest = json.loads(vector_manifest_path.read_text(encoding="utf-8"))
    if rtl_complete.get("status") != "SEALED":
        raise ValueError("RTL包未封存")
    if rtl_report.get("vector_manifest_sha256") != file_sha256(vector_manifest_path):
        raise ValueError("RTL报告与vector manifest SHA不一致")
    group_count = len(vector_manifest["selection"]["rows"])

    source_manifest_path = Path(vector_manifest["source_manifest"])
    source_payload_path = Path(vector_manifest["source_payload"])
    if file_sha256(source_manifest_path) != vector_manifest["source_manifest_sha256"]:
        raise ValueError("source manifest SHA失配")
    if file_sha256(source_payload_path) != vector_manifest["source_payload_sha256"]:
        raise ValueError("source payload SHA失配")
    payload = np.load(source_payload_path, mmap_mode="r")
    model = exact_port_for_selection(vector_manifest, payload)

    logs = {
        "tcfm5_l1": args.rtl_dir / "tcfm5_l1_verilator.log",
        "linear5_l1": args.rtl_dir / "linear5_l1_verilator.log",
        "tcfm5_l2": args.rtl_dir / "tcfm5_l2_verilator.log",
        "linear5_l2": args.rtl_dir / "linear5_l2_verilator.log",
    }
    rtl = {
        name: parse_group_log(path, group_count) for name, path in logs.items()
    }
    for index in range(group_count):
        expected = {
            "active": int(model["active_sources"][index]),
            "terms": int(model["tcfm5_cycles"][index]),
        }
        for name, rows in rtl.items():
            if rows[index]["active"] != expected["active"]:
                raise ValueError(f"{name} group{index} active与模型不一致")
            if rows[index]["terms"] != expected["terms"]:
                raise ValueError(f"{name} group{index} term与模型不一致")

    active = model["active_sources"]
    tcfm_model = 15 + np.maximum(active, model["tcfm5_cycles"])
    linear_model = 15 + np.maximum(active, model["linear5_cycles"])
    tcfm_l1 = np.asarray([row["cycles"] for row in rtl["tcfm5_l1"]], dtype=np.int64)
    linear_l1 = np.asarray([row["cycles"] for row in rtl["linear5_l1"]], dtype=np.int64)
    tcfm_l2 = np.asarray([row["cycles"] for row in rtl["tcfm5_l2"]], dtype=np.int64)
    linear_l2 = np.asarray([row["cycles"] for row in rtl["linear5_l2"]], dtype=np.int64)

    result = {
        "schema": "local5_tcfm5_model_to_rtl_calibration_v1",
        "status": "CALIBRATION_COMPLETE",
        "evidence": "[rtl]+[exact-port-model]",
        "scope": "same selected 100-group post-score cohort",
        "identity": {
            "rtl_report": str(rtl_report_path.resolve()),
            "rtl_report_sha256": file_sha256(rtl_report_path),
            "rtl_complete": str(rtl_complete_path.resolve()),
            "rtl_complete_sha256": file_sha256(rtl_complete_path),
            "vector_manifest": str(vector_manifest_path.resolve()),
            "vector_manifest_sha256": file_sha256(vector_manifest_path),
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": file_sha256(source_manifest_path),
            "source_payload": str(source_payload_path.resolve()),
            "source_payload_sha256": file_sha256(source_payload_path),
            "logs": {
                name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for name, path in logs.items()
            },
        },
        "groups": group_count,
        "model": {
            "active_sources": int(active.sum()),
            "product_terms": int(model["tcfm5_cycles"].sum()),
            "linear5_delivery_cycles": int(model["linear5_cycles"].sum()),
            "tcfm5_delivery_cycles": int(model["tcfm5_cycles"].sum()),
            "active_linear5_cycles": int(linear_model.sum()),
            "active_tcfm5_cycles": int(tcfm_model.sum()),
            "active_speedup": ratio(int(linear_model.sum()), int(tcfm_model.sum())),
        },
        "rtl": {
            "linear5_l1_cycles": int(linear_l1.sum()),
            "tcfm5_l1_cycles": int(tcfm_l1.sum()),
            "l1_speedup": ratio(int(linear_l1.sum()), int(tcfm_l1.sum())),
            "linear5_l2_cycles": int(linear_l2.sum()),
            "tcfm5_l2_cycles": int(tcfm_l2.sum()),
            "l2_speedup": ratio(int(linear_l2.sum()), int(tcfm_l2.sum())),
        },
        "fixed_overhead": {
            "linear5_l1_minus_model": int((linear_l1 - linear_model).sum()),
            "tcfm5_l1_minus_model": int((tcfm_l1 - tcfm_model).sum()),
            "linear5_l2_minus_model": int((linear_l2 - linear_model).sum()),
            "tcfm5_l2_minus_model": int((tcfm_l2 - tcfm_model).sum()),
            "l1_speedup_retention": ratio(
                ratio(int(linear_l1.sum()), int(tcfm_l1.sum())),
                ratio(int(linear_model.sum()), int(tcfm_model.sum())),
            ),
            "l2_speedup_retention": ratio(
                ratio(int(linear_l2.sum()), int(tcfm_l2.sum())),
                ratio(int(linear_model.sum()), int(tcfm_model.sum())),
            ),
        },
        "limits": [
            "模型只计15拍bitmap scan与active-source/product delivery；RTL差额是未进一步拆开的关系构建、同步读、clear、frontier和控制固定开销。",
            "该分账绑定同一100-group cohort，不是完整13,800 group RTL或full encoder。",
            "周期分账不是能量、面积或PPA。",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Local5 TCFM5 exact-port 模型到 RTL 校准",
        "",
        "## 结论",
        "",
        f"在同一 `{group_count}` 组 cohort 上，exact-port active 模型为 "
        f"`{result['model']['active_speedup']:.3f}x`；真实 RTL L1/L2 为 "
        f"`{result['rtl']['l1_speedup']:.3f}x/{result['rtl']['l2_speedup']:.3f}x`。",
        "模型到 RTL 的收益保留率分别为 "
        f"`{result['fixed_overhead']['l1_speedup_retention']:.2%}` 和 "
        f"`{result['fixed_overhead']['l2_speedup_retention']:.2%}`。",
        "",
        "| 层级 | Linear5 | TCFM5 | speedup |",
        "|---|---:|---:|---:|",
        f"| exact-port active模型 | {result['model']['active_linear5_cycles']:,} | {result['model']['active_tcfm5_cycles']:,} | {result['model']['active_speedup']:.3f}x |",
        f"| RTL L1 | {result['rtl']['linear5_l1_cycles']:,} | {result['rtl']['tcfm5_l1_cycles']:,} | {result['rtl']['l1_speedup']:.3f}x |",
        f"| RTL L2 | {result['rtl']['linear5_l2_cycles']:,} | {result['rtl']['tcfm5_l2_cycles']:,} | {result['rtl']['l2_speedup']:.3f}x |",
        "",
        "## RTL 相对模型的固定开销",
        "",
        f"- Linear5 L1/L2：`{result['fixed_overhead']['linear5_l1_minus_model']:,}` / `{result['fixed_overhead']['linear5_l2_minus_model']:,}` cycle。",
        f"- TCFM5 L1/L2：`{result['fixed_overhead']['tcfm5_l1_minus_model']:,}` / `{result['fixed_overhead']['tcfm5_l2_minus_model']:,}` cycle。",
        "- 两边都加入固定开销后，候选的理想 delivery 优势被稀释；不能把 full-profile model 直接当 RTL。",
        "",
        "## 边界",
        "",
    ]
    lines.extend(f"- {item}" for item in result["limits"])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    complete = {
        "schema": "local5_tcfm5_model_to_rtl_calibration_package_v1",
        "status": "SEALED",
        "evidence": result["evidence"],
        "package_files": {
            "report.json": file_sha256(report_path),
            "report.md": file_sha256(args.output_dir / "report.md"),
            "source/calibrate_local5_tcfm5_model_to_rtl.py": file_sha256(Path(__file__)),
        },
    }
    source_dir = args.output_dir / "source"
    source_dir.mkdir(exist_ok=True)
    (source_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
