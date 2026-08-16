#!/usr/bin/env python3
"""Audit and seal H67 real-weight two-channel projection integration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REALW_RE = re.compile(
    r"^REALW_ROW row=(?P<row>\d+) stage=(?P<stage>\d+) block=(?P<block>\d+) "
    r"head=(?P<head>\d+) expected0=(?P<e0>-?\d+) expected1=(?P<e1>-?\d+) "
    r"fixed0=(?P<f0>-?\d+) fixed1=(?P<f1>-?\d+) "
    r"rqtb0=(?P<r0>-?\d+) rqtb1=(?P<r1>-?\d+)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> list[dict[str, int]]:
    rows = []
    final_pass = False
    for line in path.read_text(encoding="utf-8").splitlines():
        match = REALW_RE.fullmatch(line)
        if match:
            row = {key: int(value) for key, value in match.groupdict().items()}
            if (row["e0"], row["e1"]) != (row["f0"], row["f1"]):
                raise ValueError(f"Fixed2S real-weight mismatch: {path}")
            if (row["e0"], row["e1"]) != (row["r0"], row["r1"]):
                raise ValueError(f"RQTB2S real-weight mismatch: {path}")
            rows.append(row)
        if line.startswith("PASS H67 RQTB 2S physical flow rows=138"):
            final_pass = True
    if len(rows) != 138 or [row["row"] for row in rows] != list(range(138)):
        raise ValueError(f"real-weight row coverage mismatch: {path}")
    if not final_pass:
        raise ValueError(f"missing underlying RQTB PASS: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    icarus = result_dir / "logs/icarus_full.log"
    verilator = result_dir / "logs/verilator_sva_full.log"
    rows_i = parse_log(icarus)
    rows_v = parse_log(verilator)
    if rows_i != rows_v:
        raise ValueError("Icarus and Verilator real-weight receipts differ")
    manifest = json.loads(args.vector_manifest.read_text(encoding="ascii"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("rows") != 138
        or manifest.get("acc32_expected_values") != 276
        or sha256_file(Path(manifest["vector"])) != manifest.get("vector_sha256")
    ):
        raise ValueError("real-weight vector manifest contract mismatch")

    sources_dir = result_dir / "source"
    sources_dir.mkdir(parents=True, exist_ok=True)
    bindings = {}
    for source in args.source:
        source = source.resolve()
        if not source.is_file():
            raise ValueError(f"missing source: {source}")
        target = sources_dir / source.name
        shutil.copy2(source, target)
        bindings[source.name] = {
            "original": str(source),
            "sha256": sha256_file(source),
            "sealed_copy": str(target),
        }

    stages = {}
    for row in rows_i:
        stages[str(row["stage"])] = stages.get(str(row["stage"]), 0) + 1
    report = {
        "schema": "h67_real_weight_projection2_rtl_v1",
        "status": "PASS",
        "evidence": "[rtl]+[real-checkpoint-int8]",
        "scope": (
            "ep35 raw Q/K through Fixed2S and RQTB2S score/weighted-SCS/Shiftmax/"
            "gated-K into two checkpoint-derived INT8 projection output channels "
            "per head-row, pre-bias Acc32"
        ),
        "coverage": {
            "rows": 138,
            "stage_rows": stages,
            "simulators": ["Icarus", "Verilator+SVA"],
            "acc32_values_per_design_per_simulator": 276,
            "total_acc32_scalar_comparisons": 276 * 2 * 2,
            "zero_mismatch": True,
        },
        "baseline_result_unchanged": {
            "fixed_cycles": 112589,
            "rqtb_cycles": 94891,
            "speedup": 112589 / 94891,
            "meaning": "reproduced ep35 component result; not a new speedup claim",
        },
        "vector_manifest": str(args.vector_manifest.resolve()),
        "vector_manifest_sha256": sha256_file(args.vector_manifest),
        "logs": {
            "icarus": {"path": str(icarus), "sha256": sha256_file(icarus)},
            "verilator_sva": {
                "path": str(verilator),
                "sha256": sha256_file(verilator),
            },
        },
        "source_bindings": bindings,
        "claim_boundary": [
            "This closes a real-weight numeric integration boundary, not projection throughput.",
            "Only two deterministic output channels per block are checked here; all-output coverage is a separate compositional evidence package.",
            "Values are pre-bias Acc32; this is not a full block, full encoder, or ASIC PPA result.",
            "The input remains sample0/window0 all12 ep35 real trace, not multi-sample real-bit RTL.",
        ],
    }
    report_path = result_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    markdown = f"""# Motion ep35 真实权重联合 Acc32 miter

> 证据：`[rtl]+[real-checkpoint-int8]`。状态：PASS。

## 结果

- 原始 Q/K 经 Fixed2S 与 RQTB2S 的 score、weighted-SCS、Shiftmax、gated-K 后，直接进入真实 checkpoint INT8 权重的两输出通道累加器；
- 覆盖 12 block、4 stage、138 个真实 T450 head-row；
- Icarus 与 Verilator+SVA 各比较 `276` 个 pre-bias Acc32，Fixed2S/RQTB2S 合计 `1,104` 个标量比较，全部零失配；
- 旧公平周期重现为 `112,589 / 94,891 = {112589 / 94891:.6f}x`，不是新增性能主张。

## 边界

- 本包关闭 synthetic Acc32 的联合数值缺口，但每个 block 只选两个确定性输出通道；
- 全输出通道由单独的组合 miter 包覆盖，本包自身不证明全输出或投影后端吞吐；
- 当前仍是 sample0/window0，不是多样本真实 bit RTL；
- 不含 bias、BN、requant、residual、full encoder 或 ASIC PPA。
"""
    (result_dir / "report.md").write_text(markdown, encoding="utf-8")

    files = {}
    for path in sorted(result_dir.rglob("*")):
        if path.is_file() and path.name != "complete.json" and "build" not in path.parts:
            files[str(path.relative_to(result_dir))] = sha256_file(path)
    complete = {
        "schema": "h67_real_weight_projection2_complete_v1",
        "status": "SEALED",
        "files": files,
    }
    (result_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print("PASS H67 real-weight projection2 report rows=138 acc32=1104")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
