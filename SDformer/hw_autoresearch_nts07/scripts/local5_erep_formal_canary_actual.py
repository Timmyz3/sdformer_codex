#!/usr/bin/env python3
"""校验 Local5 canary 的 DUT 原始输出并生成 actual provenance receipt。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


GROUP_RE = re.compile(r"GROUP .* group=(\d+) cycles=(\d+) .* terms=(\d+) updates=(\d+) .*")
PASS_RE = re.compile(
    r"PASS post-G0 active projection backend=0 latency=1 groups=(\d+) "
    r"total_cycles=(\d+) descriptors=(\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))


def validate_vector_artifacts(
    manifest_path: Path, vectors: dict[str, object]
) -> list[dict[str, object]]:
    artifacts = vectors.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("vector manifest缺少artifacts")
    bindings: list[dict[str, object]] = []
    for name, metadata in sorted(artifacts.items()):
        if not isinstance(metadata, dict):
            raise ValueError(f"vector artifact {name}元数据不合法")
        path = manifest_path.parent / str(metadata.get("file", ""))
        entries = int(metadata.get("entries", -1))
        width = int(metadata.get("width", 0))
        digits = (width + 3) // 4
        if (
            not path.is_file()
            or entries < 0
            or width <= 0
            or line_count(path) != entries
            or sha256(path) != metadata.get("sha256")
        ):
            raise ValueError(f"vector artifact {name} SHA/entries失配")
        for line_number, line in enumerate(
            path.read_text(encoding="ascii").splitlines(), start=1
        ):
            if len(line) != digits or any(
                character not in "0123456789abcdefABCDEF" for character in line
            ):
                raise ValueError(f"vector artifact {name}第{line_number}行编码错误")
        bindings.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "entries": entries,
                "width": width,
                "sha256": sha256(path),
            }
        )
    return bindings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", choices=("icarus", "verilator"), required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--filelist", type=Path, nargs="+", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--compile-command", required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--tool-versions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.task_plan.read_text(encoding="utf-8"))
    vectors = json.loads(args.vector_manifest.read_text(encoding="utf-8"))
    tasks = plan.get("tasks") or []
    if (
        plan.get("schema") != "local5_projection_task_plan_v1"
        or vectors.get("task_plan_binding", {}).get("sha256") != sha256(args.task_plan)
        or vectors.get("shape", {}).get("out_dim") != 32
        or len(vectors.get("selection", {}).get("rows", [])) != len(tasks)
    ):
        raise ValueError("actual输入task/vector合同不一致")
    artifact_bindings = validate_vector_artifacts(args.vector_manifest, vectors)
    rows = []
    terminal = None
    for line in args.log.read_text(encoding="utf-8").splitlines():
        match = GROUP_RE.fullmatch(line)
        if match:
            rows.append(tuple(int(value) for value in match.groups()))
        match = PASS_RE.fullmatch(line)
        if match:
            terminal = tuple(int(value) for value in match.groups())
    expected_lines = len(tasks) * 450 * 32
    if (
        [row[0] for row in rows] != list(range(len(tasks)))
        or terminal is None
        or terminal[0] != len(tasks)
        or terminal[1] != sum(row[1] for row in rows)
        or line_count(args.actual) != expected_lines
    ):
        raise ValueError("actual DUT日志/输出数量合同失败")
    bindings = []
    for path in args.filelist:
        if not path.is_file():
            raise ValueError(f"DUT filelist文件不存在: {path}")
        bindings.append({"path": str(path.resolve()), "sha256": sha256(path)})
    if not args.executable.is_file() or not args.tool_versions.is_file():
        raise ValueError("仿真executable/tool versions不存在")
    receipt = {
        "schema": "local5_erep_formal_canary_rtl_actual_v1",
        "status": "PASS_CANARY_NOT_G0",
        "evidence": "[rtl]",
        "simulator": args.simulator,
        "run_command": args.command,
        "compile_command": args.compile_command,
        "executable": str(args.executable.resolve()),
        "executable_sha256": sha256(args.executable),
        "tool_versions": str(args.tool_versions.resolve()),
        "tool_versions_sha256": sha256(args.tool_versions),
        "task_plan_sha256": sha256(args.task_plan),
        "vector_manifest_sha256": sha256(args.vector_manifest),
        "raw_log": str(args.log.resolve()),
        "raw_log_sha256": sha256(args.log),
        "actual_acc32": str(args.actual.resolve()),
        "actual_acc32_sha256": sha256(args.actual),
        "actual_scalar_count": expected_lines,
        "group_count": len(tasks),
        "total_cycles": terminal[1],
        "dut_file_bindings": bindings,
        "vector_artifact_bindings": artifact_bindings,
        "adapter": str(Path(__file__).resolve()),
        "adapter_sha256": sha256(Path(__file__).resolve()),
        "formal_g0": "DENY",
    }
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
