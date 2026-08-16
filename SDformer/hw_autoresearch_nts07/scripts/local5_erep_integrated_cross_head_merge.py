#!/usr/bin/env python3
"""只读合并 Local5 集成跨头 DUT actual 与独立软件 Acc32。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .local5_erep_integrated_cross_head_actual import (
        parse_acc32,
        sha256,
        validate_exact_run_argv,
    )
    from .local5_erep_numeric_release import SCHEMA as RELEASE_SCHEMA
    from .local5_erep_numeric_release import verify_release
else:
    from local5_erep_integrated_cross_head_actual import (
        parse_acc32,
        sha256,
        validate_exact_run_argv,
    )
    from local5_erep_numeric_release import SCHEMA as RELEASE_SCHEMA
    from local5_erep_numeric_release import verify_release


def _load_argv(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError("actual argv 文件不合法")
    return value


def receipt_matches(
    receipt: dict[str, object],
    actual: Path,
    task_plan: Path,
) -> None:
    if (
        receipt.get("schema")
        != "local5_erep_integrated_cross_head_actual_v1"
        or receipt.get("status") != "PASS_ACTUAL_NOT_G0"
        or receipt.get("actual_acc32_sha256") != sha256(actual)
        or receipt.get("task_plan_sha256") != sha256(task_plan)
    ):
        raise ValueError("actual receipt来源绑定失效")


def validate_execution_binding(
    receipt: dict[str, object], simulator: str, use_memo: int = 0,
    vector_result_mode: int | None = None,
) -> None:
    if use_memo not in (0, 1):
        raise ValueError("use_memo must be 0 or 1")
    if vector_result_mode not in (None, 0, 1):
        raise ValueError("vector_result_mode must be 0, 1, or None")
    executable = Path(str(receipt.get("executable", "")))
    tool_versions = Path(str(receipt.get("tool_versions", "")))
    if (
        not executable.is_file()
        or receipt.get("executable_sha256") != sha256(executable)
        or not tool_versions.is_file()
        or receipt.get("tool_versions_sha256") != sha256(tool_versions)
    ):
        raise ValueError("actual executable/tool来源绑定失效")
    if receipt.get("provenance_level") == "exact_argv_sealed_release":
        run_path = Path(str(receipt.get("run_argv_file", "")))
        compile_path = Path(str(receipt.get("compile_argv_file", "")))
        release_path = Path(str(receipt.get("release_manifest", "")))
        if (
            not run_path.is_file()
            or receipt.get("run_argv_file_sha256") != sha256(run_path)
            or not compile_path.is_file()
            or receipt.get("compile_argv_file_sha256") != sha256(compile_path)
            or not release_path.is_file()
            or receipt.get("release_manifest_sha256") != sha256(release_path)
        ):
            raise ValueError("actual argv/release来源绑定失效")
        run_argv = _load_argv(run_path)
        compile_argv = _load_argv(compile_path)
        if run_argv != receipt.get("run_argv") or compile_argv != receipt.get(
            "compile_argv"
        ):
            raise ValueError("actual receipt 与精确 argv 文件不一致")
        release = json.loads(release_path.read_text(encoding="utf-8"))
        if (
            not isinstance(release, dict)
            or release.get("schema") != RELEASE_SCHEMA
            or release.get("status") != "SEALED_RTL_RELEASE_NOT_G0"
        ):
            raise ValueError("actual release schema/status 失效")
        verify_release(release_path.parent)
        heads = int(receipt.get("identity", {}).get("heads", -1))
        build = release.get("builds", {}).get(str(heads), {})
        if (
            compile_argv != build.get("compile_argv")
            or receipt.get("executable_sha256") != build.get("executable_sha256")
            or executable.resolve()
            != (release_path.parent / str(build.get("executable_path", ""))).resolve()
        ):
            raise ValueError("actual receipt 未绑定对应 H-class release build")
        identity = receipt.get("identity")
        vector_bindings = receipt.get("vector_file_bindings")
        if not isinstance(identity, dict) or not isinstance(vector_bindings, list):
            raise ValueError("actual 精确 run argv 缺少 identity/vector binding")
        service_seed = validate_exact_run_argv(
            run_argv,
            executable,
            Path(str(receipt.get("actual_acc32", ""))),
            vector_bindings,
            {key: int(identity[key]) for key in ("sample", "stage", "block", "window", "heads")},
        )
        if receipt.get("service_seed") != service_seed:
            raise ValueError("actual service seed 与精确 run argv 不一致")
        return
    run_command = str(receipt.get("run_command", ""))
    compile_command = str(receipt.get("compile_command", ""))
    required_run = (
        executable.name,
        "+NO_ACC_CHECK",
        "+WEIGHTS=",
        "+ACTUAL_ACC_FILE=",
        "+STAGE_ID=",
        "+BLOCK_ID=",
        "+WINDOW_ID=",
    )
    required_compile = [
        "tb_qfit_local5_memo_multitile_cross_head",
        f"USE_MEMO={use_memo}",
        "USE_INPLACE=0",
        "TRANSACTION_INDEXED_SERVICE=1",
        "STAGE_ID=",
        "BLOCK_ID=",
        "WINDOW_ID=",
    ]
    if vector_result_mode is not None:
        required_compile.append(f"VECTOR_RESULT_MODE={vector_result_mode}")
    if (
        any(value not in run_command for value in required_run)
        or any(value not in compile_command for value in required_compile)
        or (simulator == "icarus" and not run_command.startswith("vvp "))
        or (simulator == "verilator" and run_command.startswith("vvp "))
    ):
        raise ValueError("actual compile/run命令合同失效")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--expected-receipt", type=Path, required=True)
    parser.add_argument("--actual", action="append", type=Path, required=True)
    parser.add_argument(
        "--actual-receipt", action="append", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-memo", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--vector-result-mode", type=int, choices=(0, 1), default=None
    )
    args = parser.parse_args()
    if len(args.actual) != 2 or len(args.actual_receipt) != 2:
        raise ValueError("必须恰好提供Icarus与Verilator两份actual/receipt")

    plan = json.loads(args.task_plan.read_text(encoding="utf-8"))
    expected_receipt = json.loads(
        args.expected_receipt.read_text(encoding="utf-8")
    )
    if (
        expected_receipt.get("schema")
        != "local5_erep_formal_canary_software_expected_v1"
        or expected_receipt.get("status") != "PASS_CANARY_NOT_G0"
        or expected_receipt.get("task_plan_sha256") != sha256(args.task_plan)
        or expected_receipt.get("software_expected_sha256")
        != sha256(args.expected)
    ):
        raise ValueError("软件expected receipt来源绑定失效")
    with np.load(args.expected, allow_pickle=False) as archive:
        expected = np.asarray(archive["expected_acc32"], dtype=np.int64)
    expected_count = int(plan["heads"]) * 450 * int(plan["out_dim"])
    if expected.size != expected_count:
        raise ValueError("软件expected数量不符合HxT450xOUT32")

    rows = []
    reference_actual: np.ndarray | None = None
    observed_simulators: set[str] = set()
    for actual_path, receipt_path in zip(args.actual, args.actual_receipt):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_matches(receipt, actual_path, args.task_plan)
        simulator = str(receipt.get("simulator", ""))
        if simulator in observed_simulators:
            raise ValueError("simulator重复")
        observed_simulators.add(simulator)
        if (
            receipt.get("actual_scalar_count") != expected_count
            or not receipt.get("filelist")
            or not receipt.get("vector_file_bindings")
        ):
            raise ValueError("actual receipt计数或来源闭包不完整")
        raw_log = Path(str(receipt.get("raw_log", "")))
        if not raw_log.is_file() or receipt.get("raw_log_sha256") != sha256(raw_log):
            raise ValueError("actual raw log来源绑定失效")
        validate_execution_binding(
            receipt, simulator, args.use_memo, args.vector_result_mode
        )
        for binding in receipt["filelist"]:
            source = Path(str(binding["file"]))
            if not source.is_file() or binding["sha256"] != sha256(source):
                raise ValueError("actual RTL/TB filelist来源绑定失效")
        for binding in receipt["vector_file_bindings"]:
            vector = Path(str(binding["path"]))
            if (
                not vector.is_file()
                or binding["sha256"] != sha256(vector)
                or int(binding["entries"])
                != len(vector.read_text(encoding="ascii").splitlines())
            ):
                raise ValueError("actual vector file来源绑定失效")
        actual = np.asarray(parse_acc32(actual_path), dtype=np.int64)
        if actual.size != expected.size:
            raise ValueError("DUT actual数量不匹配")
        if reference_actual is None:
            reference_actual = actual
        elif not np.array_equal(reference_actual, actual):
            raise ValueError("跨simulator原始Acc32不一致")
        delta = actual - expected
        mismatch = int(np.count_nonzero(delta))
        rows.append(
            {
                "simulator": simulator,
                "cycles": int(receipt["cycles"]),
                "actual_acc32_sha256": sha256(actual_path),
                "scalar_count": int(actual.size),
                "mismatch_count": mismatch,
                "max_abs_error": int(np.max(np.abs(delta), initial=0)),
            }
        )
        if mismatch:
            first = int(np.flatnonzero(delta)[0])
            raise ValueError(
                f"集成cross-head Acc32不一致 index={first} "
                f"actual={int(actual[first])} expected={int(expected[first])}"
            )
    if observed_simulators != {"icarus", "verilator"}:
        raise ValueError("simulator集合必须唯一等于{icarus, verilator}")
    report = {
        "schema": "local5_erep_integrated_cross_head_merge_v1",
        "status": "PASS_INTEGRATED_CROSS_HEAD_CANARY_NOT_G0",
        "evidence": "[rtl]+[软件整数金参考]",
        "formal_g0": "DENY",
        "use_relation_memo": bool(args.use_memo),
        "vector_result_mode": (
            None if args.vector_result_mode is None
            else bool(args.vector_result_mode)
        ),
        "task_plan_sha256": sha256(args.task_plan),
        "software_expected_sha256": sha256(args.expected),
        "scalar_count": int(expected.size),
        "identity": {
            key: int(plan[key])
            for key in ("sample", "stage", "block", "window", "heads", "out_dim")
        },
        "simulators": rows,
        "source_isolation": {
            "software_expected": (
                "producer destination-major item_* + checkpoint INT8 Acc32；"
                "不使用descriptor方向映射"
            ),
            "rtl_actual": "NO_ACC_CHECK集成cross-head DUT原始导出",
            "merge": "只读task plan/expected/actual/receipts，不读取原profile",
        },
        "boundary": [
            (
                f"仅sample{int(plan['sample'])}/stage{int(plan['stage'])}/"
                f"block{int(plan['block'])}/window{int(plan['window'])}"
            ),
            (
                "采用relation memo live/replay/fallback路径验证集成跨头数值"
                if args.use_memo
                else "采用recompute path验证集成跨头数值，relation memo另有RTL回归"
            ),
            "不是1200-window formal archive/admission",
            "cycles不是EREP或full-encoder性能",
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
