#!/usr/bin/env python3
"""从已接受的 Local5 v7 H3 trace 冻结内部状态事件参考。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


STATE_EVENTS = ("tx_state", "acc_state", "head_state")
REFERENCE_FIELDS = (
    "cycle", "event", "tile", "head", "source", "lane", "out", "delay",
    "index", "origin",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 必须是 JSON object")
    return value


def canonical_state_bytes(row: dict[str, str]) -> bytes:
    values = [row[name] for name in REFERENCE_FIELDS]
    return ("\x1f".join(values) + "\n").encode("ascii")


def freeze_reference(
    trace_path: Path,
    release_manifest_path: Path,
    canary_complete_path: Path,
) -> dict[str, Any]:
    complete = read_json(canary_complete_path)
    trace_sha = sha256(trace_path)
    release_sha = sha256(release_manifest_path)
    if complete.get("status") != "PASS_SEALED_H3_IDENTITY_SERVICE_NOT_G0":
        raise ValueError("源 canary 不是已接受的 v7 H3 状态")
    if complete.get("formal_g0") != "DENY":
        raise ValueError("源 canary 的 formal G0 边界不同")
    if complete.get("release_manifest_sha256") != release_sha:
        raise ValueError("源 canary 未绑定给定 release manifest")
    if complete.get("artifacts", {}).get("identity_trace.csv") != trace_sha:
        raise ValueError("源 canary 未绑定给定 trace")

    per_event = {name: hashlib.sha256() for name in STATE_EVENTS}
    all_states = hashlib.sha256()
    counts: Counter[str] = Counter()
    total_rows = 0
    with trace_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(REFERENCE_FIELDS):
            raise ValueError("源 v7 trace 表头不是冻结的 10 列合同")
        for row in reader:
            total_rows += 1
            event = row["event"]
            if event not in STATE_EVENTS:
                continue
            if row["origin"] != "rtl_internal_state":
                raise ValueError(f"{event} origin 不符合 RTL 内部状态合同")
            payload = canonical_state_bytes(row)
            per_event[event].update(payload)
            all_states.update(payload)
            counts[event] += 1
    if any(counts[name] == 0 for name in STATE_EVENTS):
        raise ValueError("源 trace 缺少内部状态事件")

    return {
        "schema": "local5_identity_service_h3_state_reference_v1",
        "status": "FROZEN_FROM_ACCEPTED_V7_H3_NOT_G0",
        "evidence": "[rtl]",
        "formal_g0": "DENY",
        "identity": {
            "sample": 2,
            "stage": 0,
            "block": 0,
            "window": 249,
            "heads": 3,
        },
        "canonical_fields": list(REFERENCE_FIELDS),
        "source": {
            "trace_sha256": trace_sha,
            "trace_rows": total_rows,
            "release_manifest_sha256": release_sha,
            "canary_complete_sha256": sha256(canary_complete_path),
        },
        "states": {
            name: {
                "count": counts[name],
                "ordered_sha256": per_event[name].hexdigest(),
            }
            for name in STATE_EVENTS
        },
        "all_states": {
            "count": sum(counts.values()),
            "ordered_sha256": all_states.hexdigest(),
        },
        "boundary": [
            "仅冻结 sample2/stage0/block0/window249/H3 的内部状态序列",
            "用于检出状态事件缺失、重排或字段篡改；不是 formal G0",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--canary-complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"冻结参考已存在，拒绝覆盖：{output}")
    value = freeze_reference(
        args.trace.resolve(),
        args.release_manifest.resolve(),
        args.canary_complete.resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"status": value["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
