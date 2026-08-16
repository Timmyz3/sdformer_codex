#!/usr/bin/env python3
"""Seal multisample H67 Fixed2S/RQTB2S RTL evidence without changing main anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_MAIN_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
EXPECTED_SAMPLE0 = {
    "fixed_cycles": 112589,
    "rqtb_cycles": 94891,
    "fixed_slots": 62100,
    "rqtb_slots": 34099,
    "equal_pairs": 28001,
}
BAD_LOG_RE = re.compile(r"\b(?:ERROR|FATAL)\b|Assertion failed|%Error|mismatch=[1-9]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def verify_log(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"log identity mismatch: {path}")
    text = path.read_text(encoding="utf-8", errors="strict")
    if BAD_LOG_RE.search(text):
        raise ValueError(f"log contains an error or nonzero mismatch: {path}")
    if "PASS H67 RQTB 2S physical flow rows=1380" not in text:
        raise ValueError(f"log lacks the 1380-row final PASS: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--frozen-main", type=Path, required=True)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    summary_path = result_dir / "summary.json"
    summary = load_json(summary_path)
    if summary.get("status") != "PASS" or summary.get("evidence_level") != "[rtl]":
        raise ValueError("multisample summary is not admitted RTL evidence")
    coverage = summary.get("coverage", {})
    if coverage != {
        "acc32_mismatch": 0,
        "cross_simulator_exact": True,
        "gated_k_outputs_checked": coverage.get("gated_k_outputs_checked"),
        "rows": 1380,
        "rows_per_sample": 138,
        "samples": 10,
        "tokens_per_row": 450,
    } or coverage["gated_k_outputs_checked"] <= 0:
        raise ValueError("multisample coverage contract mismatch")
    samples = summary.get("samples", [])
    if len(samples) != 10:
        raise ValueError("summary does not contain ten complete samples")
    for key, expected in EXPECTED_SAMPLE0.items():
        if samples[0].get(key) != expected:
            raise ValueError(f"sample0 no longer reproduces the sealed anchor: {key}")
    if any(sample.get("speedup", 0.0) <= 1.0 for sample in samples):
        raise ValueError("at least one admitted sample does not improve cycles")
    stages = summary.get("stages", [])
    if len(stages) != 4 or any(stage.get("speedup", 0.0) <= 1.0 for stage in stages):
        raise ValueError("stage coverage or speedup contract mismatch")
    provenance = summary.get("provenance", {})
    if provenance.get("implementation_source_mode") != "frozen_flat_directory":
        raise ValueError("replay did not bind the frozen RTL source directory")
    verify_log(
        Path(provenance["icarus_log"]), provenance["icarus_log_sha256"]
    )
    verify_log(
        Path(provenance["verilator_sva_log"]),
        provenance["verilator_sva_log_sha256"],
    )
    frozen_main_sha256 = sha256_file(args.frozen_main)
    if frozen_main_sha256 != EXPECTED_MAIN_SHA256:
        raise ValueError("frozen DATE main document changed")

    cycles = summary["cycles"]
    work = summary["work"]
    report = {
        "schema": "h67_ep35_multisample10_real_rtl_seal_v1",
        "status": "PASS_SEALED_COMPONENT_RTL_NOT_ENCODER",
        "evidence": "[rtl]",
        "coverage": coverage,
        "cycles": cycles,
        "work": work,
        "samples": samples,
        "stages": stages,
        "frozen_main": {
            "path": str(args.frozen_main.resolve()),
            "sha256": frozen_main_sha256,
            "unchanged_anchor": EXPECTED_SAMPLE0,
        },
        "summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "claim_boundary": [
            "Ten preregistered ep35 samples are covered, with one selected all12 window per sample.",
            "This is real Q/K/gate RTL with a synthetic lane-weight Acc32 checksum, not multi-sample real-weight projection.",
            "The aggregate 10-sample speedup does not replace the sealed sample0/window0 112589/94891 main anchor.",
            "This is not all spatial windows, a full attention block, a full encoder, or ASIC PPA.",
        ],
    }
    report_json = result_dir / "report.json"
    report_json.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    sample_speedup = cycles["sample_distribution"]["speedup"]
    markdown = f"""# Motion ep35 十样本真实 Q/K 组件级 RTL

> 证据：`[rtl]`。状态：`PASS_SEALED_COMPONENT_RTL_NOT_ENCODER`。

## 结果

- 覆盖 10 个预注册样本，每样本 12 block、138 个 T450 head-row，共 1,380 行；
- Icarus 与 Verilator+SVA 逐行账本完全一致，共检查 {coverage['gated_k_outputs_checked']:,} 个 gated-K 输出，Acc32 checksum 零失配；
- 总周期 Fixed2S/RQTB2S 为 `{cycles['fixed_total']:,}/{cycles['rqtb_total']:,}`，组件级总加速 `{cycles['global_speedup']:.6f}x`；
- 十样本加速范围 `{sample_speedup['min']:.6f}x` 到 `{sample_speedup['max']:.6f}x`，10/10 样本均改善；
- slot 减少 `{work['slot_reduction_ratio']:.2%}`，exp 事务减少 `{work['exp_reduction_ratio']:.2%}`；
- sample0 精确复现冻结主锚点 `112589/94891`、slot `62100/34099`、equal `28001`。

## 边界

- 每个样本只选一个 all12 窗口，不是所有空间窗口或 full encoder；
- Q/K/gate 来自真实 ep35 checkpoint，Acc32 边界仍是 synthetic lane-weight checksum；
- 多样本真实 INT8 全输出投影、bias/BN/requant/residual、DC/STA/SAIF/PTPX 仍未闭合；
- `{cycles['global_speedup']:.6f}x` 是十样本组件结果，不覆盖冻结主列的 `1.1865x`。
"""
    (result_dir / "report.md").write_text(markdown, encoding="utf-8")

    files = {}
    for path in sorted(result_dir.rglob("*")):
        if path.is_file() and path.name != "complete.json" and "build" not in path.parts:
            files[str(path.relative_to(result_dir))] = sha256_file(path)
    complete = {
        "schema": "h67_ep35_multisample10_real_rtl_complete_v1",
        "status": "SEALED",
        "files": files,
    }
    (result_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        "PASS sealed H67 multisample RTL "
        f"samples=10 rows=1380 speedup={cycles['global_speedup']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
