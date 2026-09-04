#!/usr/bin/env python3
"""为 Local5 正式 numeric shard 的单窗生成独立软件 Acc32。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .local5_erep_formal_canary_expected import build_expected, sha256
else:
    from local5_erep_formal_canary_expected import build_expected, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.sample < 100:
        raise ValueError("formal sample 必须位于 0..99")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan, expected = build_expected(
        args.profile.resolve(), args.sample, args.stage, args.block
    )
    plan["scope"] = "formal_numeric_sample_shard_not_g0"
    plan_path = out / "task_plan.json"
    expected_path = out / "software_expected.npz"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez(
        expected_path,
        schema_version=np.asarray([1], dtype=np.uint16),
        expected_acc32=expected.reshape(-1),
    )
    sources = [Path(__file__).resolve(), Path(build_expected.__code__.co_filename).resolve()]
    receipt = {
        "schema": "local5_erep_numeric_window_expected_v1",
        "status": "PASS_NUMERIC_WINDOW_EXPECTED_NOT_G0",
        "evidence": "[软件整数金参考]",
        "formal_g0": "DENY",
        "identity": {
            key: int(plan[key])
            for key in ("sample", "stage", "block", "window", "heads")
        },
        "task_plan_sha256": sha256(plan_path),
        "software_expected_sha256": sha256(expected_path),
        "expected_shape": list(expected.shape),
        "expected_scalar_count": int(expected.size),
        "numpy_version": np.__version__,
        "source_bindings": [
            {"file": str(path), "sha256": sha256(path)} for path in sources
        ],
        "oracle_path": (
            "producer destination-major item_*直接整数聚合；"
            "不使用descriptor方向映射"
        ),
    }
    (out / "software_expected_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
