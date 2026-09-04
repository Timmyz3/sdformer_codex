#!/usr/bin/env python3
"""从H67真实Q/K NPZ生成TARE-4位级回放向量与中文报告。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results/h67_real_bit_trace_20260717/manifest.json"
DEFAULT_OUTPUT = ROOT / "results/tare4_h67_real_trace_20260726"
DEFAULT_PROFILE = (
    ROOT
    / "results/h67_real_bit_trace_profile_20260717/"
    "nts11_hardware_p0_profile.json"
)


def load_reference():
    path = ROOT / "scripts/dual_line_delta_reference.py"
    spec = importlib.util.spec_from_file_location("dual_line_delta_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载dual_line_delta_reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = load_reference()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_bits(payload: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    shape = tuple(int(value) for value in payload[f"{prefix}_shape"])
    count = int(np.prod(shape))
    bits = np.unpackbits(
        payload[f"{prefix}_bits_packed"],
        bitorder="little",
    )[:count]
    return bits.reshape(shape).astype(np.bool_, copy=False)


def pack_lanes(bits: np.ndarray) -> int:
    if bits.shape != (32,):
        raise ValueError(f"lane shape必须是(32,)，实际为{bits.shape}")
    result = 0
    for lane, active in enumerate(bits):
        if bool(active):
            result |= 1 << lane
    return result


def signed10(value: int) -> int:
    if value < -512 or value > 511:
        raise ValueError(f"delta超出signed10: {value}")
    return value & 0x3FF


def process_record(record: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    path = Path(record["file"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"源trace SHA256不一致: {path}")
    payload = np.load(path)
    q = unpack_bits(payload, "q")
    k = unpack_bits(payload, "k")
    if q.shape != k.shape or q.ndim != 5 or q.shape[0] != 2:
        raise ValueError(f"Q/K布局非法: q={q.shape}, k={k.shape}")
    if q.shape[-1] != 32:
        raise ValueError("TARE-4当前只支持32 lane")

    payload_lines: list[str] = []
    expected_lines: list[str] = []
    kind_hist = [0, 0, 0]
    update_hist = [0] * 33
    raw_mismatches = 0
    q7_mismatches = 0
    delta_min = 1 << 30
    delta_max = -(1 << 30)

    _, windows, heads, tokens, _ = q.shape
    for window in range(windows):
        for head in range(heads):
            for token in range(tokens):
                q0 = pack_lanes(q[0, window, head, token])
                k0 = pack_lanes(k[0, window, head, token])
                q1 = pack_lanes(q[1, window, head, token])
                k1 = pack_lanes(k[1, window, head, token])
                score, meta = REF.h67_motion_delta_q7(q0, k0, q1, k1)
                direct_raw = (
                    REF.axnor_raw16(q1, k1)
                    + 16 * REF.popcount(k0 ^ k1)
                )
                direct_q7 = REF.rne_div16(direct_raw)
                raw_mismatches += int(meta["final_raw16"] != direct_raw)
                q7_mismatches += int(score != direct_q7)

                count = REF.popcount((q0 ^ q1) | (k0 ^ k1))
                kind = 0 if count == 0 else (1 if count <= 4 else 2)
                delta = int(meta["delta_raw16"]) if kind == 1 else 0
                kind_hist[kind] += 1
                update_hist[count] += 1
                delta_min = min(delta_min, int(meta["delta_raw16"]))
                delta_max = max(delta_max, int(meta["delta_raw16"]))

                packed_payload = (
                    (q0 << 96)
                    | (k0 << 64)
                    | (q1 << 32)
                    | k1
                )
                packed_expected = (
                    (kind << 16)
                    | (count << 10)
                    | signed10(delta)
                )
                payload_lines.append(f"{packed_payload:032x}")
                expected_lines.append(f"{packed_expected:05x}")

    pairs = len(payload_lines)
    return payload_lines, expected_lines, {
        "name": record["name"],
        "source_file": str(path),
        "source_sha256": record["sha256"],
        "windows": windows,
        "heads": heads,
        "tokens": tokens,
        "pairs": pairs,
        "kind_zero": kind_hist[0],
        "kind_sparse": kind_hist[1],
        "kind_dense": kind_hist[2],
        "fallback_ratio": kind_hist[2] / pairs if pairs else 0.0,
        "update_histogram": update_hist,
        "delta_raw16_min": delta_min,
        "delta_raw16_max": delta_max,
        "raw_mismatches": raw_mismatches,
        "q7_mismatches": q7_mismatches,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TARE-4 H67 真实位级 Trace 生成与整数审计",
        "",
        "## 证据边界",
        "",
        "- 来源是真实 H67 推理张量，不是统计塑形随机向量；",
        "- 覆盖 `sample0/zurich_city_09_a_0001`、四个 stage 的 B0、"
        "每 stage 一个窗口；",
        "- 不是 profile100、不是全部 12 个 block、不是端到端 PPA；",
        "- payload 顺序为 `{Q0,K0,Q1,K1}`，每项 128 bit；",
        "- expected 为 `{kind[1:0],count[5:0],delta_raw16[9:0]}`；",
        "",
        "## 结果",
        "",
        "| Stage/Block | heads | tokens | pairs | ZERO | SPARSE<=4 | DENSE>4 | fallback | raw/Q7 mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["records"]:
        lines.append(
            f"| {row['name']} | {row['heads']} | {row['tokens']} | "
            f"{row['pairs']} | {row['kind_zero']} | {row['kind_sparse']} | "
            f"{row['kind_dense']} | {row['fallback_ratio']:.4%} | "
            f"{row['raw_mismatches']}/{row['q7_mismatches']} |"
        )
    total = result["total"]
    lines += [
        f"| **总计** | - | - | **{total['pairs']}** | "
        f"**{total['kind_zero']}** | **{total['kind_sparse']}** | "
        f"**{total['kind_dense']}** | **{total['fallback_ratio']:.4%}** | "
        f"**{total['raw_mismatches']}/{total['q7_mismatches']}** |",
        "",
        "## 文件",
        "",
        f"- payload：`{result['payload_file']}`；",
        f"- expected：`{result['expected_file']}`；",
        f"- payload SHA256：`{result['payload_sha256']}`；",
        f"- expected SHA256：`{result['expected_sha256']}`。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if not source.get("coverage", {}).get("four_stage_complete"):
        raise ValueError("源trace未覆盖四个stage")

    all_payload: list[str] = []
    all_expected: list[str] = []
    rows: list[dict[str, Any]] = []
    for record in source["records"]:
        payload, expected, row = process_record(record)
        all_payload.extend(payload)
        all_expected.extend(expected)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = args.output_dir / "payload128.mem"
    expected_path = args.output_dir / "expected18.mem"
    payload_path.write_text("\n".join(all_payload) + "\n", encoding="ascii")
    expected_path.write_text("\n".join(all_expected) + "\n", encoding="ascii")
    total_pairs = sum(row["pairs"] for row in rows)
    total_dense = sum(row["kind_dense"] for row in rows)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    config_path = Path(profile["config"])
    checkpoint_path = Path(profile["checkpoint"])
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema": "tare4_h67_real_trace_v1",
        "source_manifest": str(args.source),
        "source_manifest_sha256": sha256(args.source),
        "source_profile": str(args.profile),
        "source_profile_sha256": sha256(args.profile),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "repository_head_at_regeneration": git_head,
        "code_provenance_limit": (
            "repository_head是本次向量再生成时的HEAD；原20260717采集未保存"
            "独立代码归档，不能将该HEAD冒充原采集代码快照。"
        ),
        "payload_file": str(payload_path),
        "expected_file": str(expected_path),
        "payload_sha256": sha256(payload_path),
        "expected_sha256": sha256(expected_path),
        "records": rows,
        "total": {
            "pairs": total_pairs,
            "kind_zero": sum(row["kind_zero"] for row in rows),
            "kind_sparse": sum(row["kind_sparse"] for row in rows),
            "kind_dense": total_dense,
            "fallback_ratio": total_dense / total_pairs if total_pairs else 0.0,
            "raw_mismatches": sum(row["raw_mismatches"] for row in rows),
            "q7_mismatches": sum(row["q7_mismatches"] for row in rows),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(manifest_path)
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
