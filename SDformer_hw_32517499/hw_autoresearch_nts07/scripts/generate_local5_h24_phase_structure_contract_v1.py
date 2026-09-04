#!/usr/bin/env python3
"""由密封 H3/H6/H12 证据预注册 H24 identity/phase 事件结构合同。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOKENS = 450
LANES = 32
OUT_DIM = 32
STATE_EVENTS = ("tx_state", "acc_state", "head_state")
STATE_POLYNOMIALS = {
    "tx_state": (3, 43202, 1),
    "acc_state": (28800, -28800, 1),
    "head_state": (46157, 0, 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须包含 JSON object")
    return value


def state_count(name: str, heads: int) -> int:
    quadratic, linear, constant = STATE_POLYNOMIALS[name]
    return quadratic * heads * heads + linear * heads + constant


def expected_event_counts(heads: int, hold_cycles: int) -> dict[str, int]:
    relation = heads * heads * TOKENS
    weight = heads * heads * LANES * OUT_DIM
    final = heads * TOKENS * OUT_DIM
    counts = {
        "manifest_binding": 1,
        "receipt_binding": 1,
        "group_start": 1,
        "group_done": 1,
        "tile_start": heads,
        "tile_done": heads,
        "head_start": heads * heads,
        "head_done": heads * heads,
        "relation_accept": relation,
        "relation_response_available": relation,
        "relation_response_accept": relation,
        "weight_accept": weight,
        "weight_response_available": weight,
        "weight_response_accept": weight,
        "final_request": final,
        "final_accept": final,
        **{name: state_count(name, heads) for name in STATE_EVENTS},
    }
    if hold_cycles:
        counts["weight_response_stall"] = weight * hold_cycles
    return dict(sorted(counts.items()))


def trace_event_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "event" not in reader.fieldnames:
            raise ValueError("H3 trace 缺少 event 列")
        for row in reader:
            event = row.get("event")
            if not event:
                raise ValueError("H3 trace event 为空")
            counts[event] += 1
    return dict(sorted(counts.items()))


def load_h3(path: Path) -> dict[str, Any]:
    complete_path = path / "complete.json"
    complete = read_json(complete_path)
    trace_binding = complete.get("external_bindings", {}).get("source_trace", {})
    trace = Path(str(trace_binding.get("path", ""))).resolve()
    if (
        complete.get("status") != "PASS_SEALED_STREAMING_MMAP_CANARY_NOT_G0"
        or complete.get("formal_g0") != "DENY"
        or complete.get("identity", {}).get("heads") != 3
        or trace_binding.get("sha256") != sha256(trace)
    ):
        raise ValueError("H3 密封证据合同不一致")
    return {
        "heads": 3,
        "identity": complete["identity"],
        "event_counts": trace_event_counts(trace),
        "bindings": {
            "complete": {"path": str(complete_path), "sha256": sha256(complete_path)},
            "trace": {"path": str(trace), "sha256": sha256(trace)},
        },
    }


def load_parameterized(path: Path, heads: int) -> dict[str, Any]:
    complete_path = path / "complete.json"
    report_path = path / "candidate_trace_verification.json"
    complete = read_json(complete_path)
    report = read_json(report_path)
    if (
        complete.get("status")
        != "PASS_SEALED_PARAMETERIZED_IDENTITY_PHASE_CANARY_NOT_G0"
        or complete.get("formal_g0") != "DENY"
        or complete.get("identity", {}).get("heads") != heads
        or report.get("identity") != complete.get("identity")
        or report.get("formal_g0") != "DENY"
        or report.get("payload_stability", {}).get("weight_hold_cycles_per_response") != 2
        or complete.get("internal_bindings", {}).get(report_path.name)
        != sha256(report_path)
    ):
        raise ValueError(f"H{heads} 参数化证据合同不一致")
    return {
        "heads": heads,
        "identity": report["identity"],
        "event_counts": report["event_counts"],
        "bindings": {
            "complete": {"path": str(complete_path), "sha256": sha256(complete_path)},
            "report": {"path": str(report_path), "sha256": sha256(report_path)},
        },
    }


def chmod_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    mode = root.stat().st_mode
    root.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h3", type=Path,
        default=ROOT / "results/local5_h3_phase_array_store_v2_smoke_20260812",
    )
    parser.add_argument(
        "--h6", type=Path,
        default=ROOT / "results/local5_h6_nonzero_identity_phase_canary_v2_20260811",
    )
    parser.add_argument(
        "--h12", type=Path,
        default=ROOT / "results/local5_h12_nonzero_identity_phase_canary_v2_20260811",
    )
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")

    calibration = [
        load_h3(args.h3.resolve()),
        load_parameterized(args.h6.resolve(), 6),
        load_parameterized(args.h12.resolve(), 12),
    ]
    for row in calibration:
        expected = expected_event_counts(row["heads"], 2)
        if row["event_counts"] != expected:
            raise ValueError(f"H{row['heads']} 不满足冻结解析/状态计数公式")

    identity = {
        "sample": args.sample,
        "stage": args.stage,
        "block": args.block,
        "window": args.window,
        "heads": 24,
    }
    expected = {
        "baseline_hold0": {
            "hold_cycles": 0,
            "event_counts": expected_event_counts(24, 0),
        },
        "candidate_hold2": {
            "hold_cycles": 2,
            "event_counts": expected_event_counts(24, 2),
        },
    }
    for row in expected.values():
        row["trace_rows"] = sum(row["event_counts"].values())
    if expected["candidate_hold2"]["trace_rows"] != 47_941_735:
        raise ValueError("H24 candidate 精确结构总行数与预注册常量不一致")

    staging = output.with_name(f"{output.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)
    contract = {
        "schema": "local5_h24_phase_structure_contract_v1",
        "status": "FROZEN_H24_EVENT_COUNTS_FROM_H3_H6_H12_NOT_G0",
        "evidence": "[rtl校准结构合同]+[解析事务计数]",
        "formal_g0": "DENY",
        "identity": identity,
        "state_count_polynomials": {
            name: {"quadratic": value[0], "linear": value[1], "constant": value[2]}
            for name, value in STATE_POLYNOMIALS.items()
        },
        "expected": expected,
        "calibration": calibration,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "boundary": [
            "在 H24 trace 产生前冻结；用于检出事件类缺失或重复",
            "状态计数由密封 H3/H6/H12 RTL 校准，不是独立算法 oracle 或 formal proof",
            "未冻结 cycle-sensitive ordered digest；完整握手顺序仍由独立 trace verifier 检查",
            "validation trace rows 不是架构性能、吞吐或 ASIC PPA",
        ],
    }
    contract_path = staging / "contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    chmod_read_only(output)
    print(json.dumps({
        "status": contract["status"],
        "candidate_trace_rows": expected["candidate_hold2"]["trace_rows"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
