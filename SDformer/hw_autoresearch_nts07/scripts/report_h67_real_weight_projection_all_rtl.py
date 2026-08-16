#!/usr/bin/env python3
"""Audit and seal H67 all-output real-weight numeric sidecar evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


ROW_RE = re.compile(
    r"^REALWALL_ROW batch=(?P<batch>\d+) row=(?P<row>\d+) "
    r"stage=(?P<stage>\d+) block=(?P<block>\d+) "
    r"head=(?P<head>\d+) valid=(?P<valid>\d+)$"
)
PASS_RE = re.compile(
    r"^PASS H67 real-weight all-output sidecar batch=(?P<batch>\d+) "
    r"rows=(?P<rows>\d+) valid=(?P<valid>\d+) mismatch=0$"
)
BAD_RE = re.compile(r"\b(?:ERROR|FATAL):")
EXPECTED_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
EXPECTED_BLOCKS = {0: 2, 1: 2, 2: 6, 3: 2}
CHANNELS = 16
BATCHES = 48


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_rows(batch: int) -> list[dict[str, int]]:
    rows = []
    row = 0
    channel_base = batch * CHANNELS
    for stage in range(4):
        heads = EXPECTED_HEADS[stage]
        dim = heads * 32
        valid = max(0, min(CHANNELS, dim - channel_base))
        for block in range(EXPECTED_BLOCKS[stage]):
            for head in range(heads):
                rows.append(
                    {
                        "batch": batch,
                        "row": row,
                        "stage": stage,
                        "block": block,
                        "head": head,
                        "valid": valid,
                    }
                )
                row += 1
    return rows


def parse_log(path: Path, batch: int) -> tuple[list[dict[str, int]], dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    if BAD_RE.search(text):
        raise ValueError(f"error marker in log: {path}")
    rows = []
    pass_receipt = None
    for line in text.splitlines():
        row_match = ROW_RE.fullmatch(line)
        if row_match:
            rows.append({key: int(value) for key, value in row_match.groupdict().items()})
        pass_match = PASS_RE.fullmatch(line)
        if pass_match:
            pass_receipt = {
                key: int(value) for key, value in pass_match.groupdict().items()
            }
    if rows != expected_rows(batch):
        raise ValueError(f"all-output row coverage mismatch: {path}")
    expected_pass = {
        "batch": batch,
        "rows": 138,
        "valid": sum(row["valid"] for row in expected_rows(batch)),
    }
    if pass_receipt != expected_pass:
        raise ValueError(f"underlying fair receipt mismatch: {path}")
    return rows, pass_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--fair-report", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    manifest = json.loads(args.vector_manifest.read_text(encoding="ascii"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("rows") != 138
        or manifest.get("batch_channels") != CHANNELS
        or manifest.get("batches") != BATCHES
        or manifest.get("valid_acc32_values_per_design") != 67392
        or len(manifest.get("vectors", [])) != BATCHES
    ):
        raise ValueError("all-output vector manifest contract mismatch")
    for batch, vector in enumerate(manifest["vectors"]):
        path = Path(vector["path"])
        if (
            vector.get("batch") != batch
            or not path.is_file()
            or sha256_file(path) != vector.get("sha256")
        ):
            raise ValueError(f"vector identity mismatch for batch {batch}")

    fair_report = json.loads(args.fair_report.read_text(encoding="utf-8"))
    joint_report = json.loads(args.joint_report.read_text(encoding="utf-8"))
    if (
        fair_report.get("status") != "PASS"
        or fair_report.get("rows") != 138
        or joint_report.get("status") != "PASS"
        or joint_report.get("coverage", {}).get("rows") != 138
        or not joint_report.get("coverage", {}).get("zero_mismatch")
    ):
        raise ValueError("compositional input evidence is not admitted")

    logs = {"icarus": [], "verilator_sva": []}
    valid_total = 0
    for batch in range(BATCHES):
        expected = expected_rows(batch)
        batch_valid = sum(row["valid"] for row in expected)
        valid_total += batch_valid
        parsed = []
        for simulator, stem in (
            ("icarus", "icarus"),
            ("verilator_sva", "verilator_sva"),
        ):
            path = result_dir / "logs" / f"{stem}_batch_{batch:02d}.log"
            rows, _ = parse_log(path, batch)
            parsed.append(rows)
            logs[simulator].append(
                {
                    "batch": batch,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "valid_acc32_values_per_design": batch_valid,
                }
            )
        if parsed[0] != parsed[1]:
            raise ValueError(f"simulator receipts differ for batch {batch}")
    if valid_total != 67392:
        raise ValueError(f"valid Acc32 coverage mismatch: {valid_total}")

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

    report = {
        "schema": "h67_real_weight_projection_all_rtl_v1",
        "status": "PASS",
        "evidence": "[rtl]+[real-checkpoint-int8]",
        "scope": (
            "compositional ep35 sample0/window0 all12 proof: admitted Fixed2S/"
            "RQTB2S gated-K stream plus every checkpoint-derived INT8 projection "
            "output channel, pre-bias per-head partial Acc32"
        ),
        "coverage": {
            "rows_per_batch": 138,
            "batches": BATCHES,
            "channels_per_batch": CHANNELS,
            "simulators": ["Icarus", "Verilator+SVA"],
            "valid_acc32_values_per_design_per_simulator": valid_total,
            "total_sidecar_acc32_scalar_comparisons": valid_total * 2,
            "zero_mismatch": True,
        },
        "baseline_result_unchanged": {
            "fixed_cycles": 112589,
            "rqtb_cycles": 94891,
            "fixed_slots": 62100,
            "rqtb_slots": 34099,
            "meaning": "replayed per output batch; not a new performance result",
        },
        "vector_manifest": str(args.vector_manifest.resolve()),
        "vector_manifest_sha256": sha256_file(args.vector_manifest),
        "compositional_inputs": {
            "fair_row_descriptor_report": {
                "path": str(args.fair_report.resolve()),
                "sha256": sha256_file(args.fair_report),
            },
            "direct_two_channel_joint_report": {
                "path": str(args.joint_report.resolve()),
                "sha256": sha256_file(args.joint_report),
            },
        },
        "logs": logs,
        "source_bindings": bindings,
        "claim_boundary": [
            "This is a compositional miter, not a single monolithic all-output simulation.",
            "The admitted fair flow proves both Fixed2S and RQTB2S gated-K streams; the direct two-channel package bridges the integration boundary.",
            "This closes all output channels only for sample0/window0 ep35 all12.",
            "RTL values are per-head pre-bias partial Acc32; the generator independently checks cross-head sums.",
            "The 16-channel checker is a numeric sidecar, not projection throughput hardware.",
            "This is not multi-sample RTL, a full block/encoder result, or ASIC PPA.",
        ],
    }
    report_path = result_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    markdown = f"""# Motion ep35 真实权重全输出通道 Acc32 miter

> 证据：`[rtl]+[real-checkpoint-int8]`。状态：PASS。

## 结果

- 覆盖 sample0/window0、12 block、4 stage、138 个真实 T450 head-row；
- 按 16 通道分 48 个 batch，覆盖各 stage 的全部 `96/192/384/768` 输出通道；
- 每个设计、每个仿真器比较 `{valid_total:,}` 个有效 pre-bias partial Acc32；
- 16 通道 sidecar 在 Icarus/Verilator 各比较 `{valid_total:,}` 个有效标量，共 `{valid_total * 2:,}` 个，零失配；
- 组合输入绑定公平逐行 descriptor 报告和直接双通道联合包；主周期仍只取冻结的 `112589/94891`，本包不重报性能。

## 边界

- 这是组合 miter，不是单个 monolithic 全输出 RTL；公平包证明两条 gated-K 流，双通道包提供直接集成桥；
- sidecar 比较边界是逐 head 的 pre-bias partial Acc32；生成器另用完整张量等式检查跨 head 汇总；
- 16 通道检查器是数值 sidecar，不是吞吐型 projection backend；
- 当前仍是 sample0/window0，不是多样本真实 bit RTL；
- 不含 bias、BN、requant、residual、full encoder 或 ASIC PPA。
"""
    (result_dir / "report.md").write_text(markdown, encoding="utf-8")

    files = {}
    for path in sorted(result_dir.rglob("*")):
        if path.is_file() and path.name != "complete.json" and "build" not in path.parts:
            files[str(path.relative_to(result_dir))] = sha256_file(path)
    complete = {
        "schema": "h67_real_weight_projection_all_complete_v1",
        "status": "SEALED",
        "files": files,
    }
    (result_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        "PASS H67 real-weight all-output report "
        f"batches={BATCHES} acc32={valid_total * 2}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
