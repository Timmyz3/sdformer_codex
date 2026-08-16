#!/usr/bin/env python3
"""封存 Local5 集成跨头 DUT 的原始 Acc32 与仿真来源。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

if __package__:
    from .local5_erep_numeric_release import SCHEMA as RELEASE_SCHEMA
    from .local5_erep_numeric_release import verify_release
else:
    from local5_erep_numeric_release import SCHEMA as RELEASE_SCHEMA
    from local5_erep_numeric_release import verify_release


TOKENS = 450
OUT_DIM = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_acc32(path: Path) -> list[int]:
    values: list[int] = []
    for line_number, text in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1
    ):
        value = int(text.strip(), 16)
        if value >= 1 << 31:
            value -= 1 << 32
        if not -(1 << 31) <= value < 1 << 31:
            raise ValueError(f"Acc32第{line_number}行越界")
        values.append(value)
    return values


def parse_unique_terminal(log_text: str) -> tuple[int, int]:
    matches = re.findall(
        r"^PASS Local5 multi-tile .*?cycles=(\d+).*?final=(\d+).*$",
        log_text,
        flags=re.MULTILINE,
    )
    if len(matches) != 1 or "fatal" in log_text.lower() or "%Error" in log_text:
        raise ValueError("集成跨头仿真日志未出现唯一PASS或含错误")
    return tuple(int(value) for value in matches[0])


def parse_unique_identity(log_text: str) -> tuple[int, int, int]:
    matches = re.findall(
        r"^PASS Local5 multi-tile .*?stage=(\d+) block=(\d+) "
        r"window=(\d+).*?$",
        log_text,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError("集成跨头仿真日志缺少唯一stage/block/window")
    return tuple(int(value) for value in matches[0])


def validate_vector_files(
    manifest_path: Path, vectors: dict[str, object]
) -> list[dict[str, object]]:
    generator = vectors.get("generator_binding")
    if (
        not isinstance(generator, dict)
        or generator.get("numpy_version") != np.__version__
        or not isinstance(generator.get("helpers"), list)
        or len(generator["helpers"]) != 2
    ):
        raise ValueError("vector generator/helper绑定不完整")
    for binding in [generator, *generator["helpers"]]:
        source = Path(str(binding.get("file", "")))
        if not source.is_file() or binding.get("sha256") != sha256(source):
            raise ValueError("vector generator/helper SHA失配")
    files = vectors.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("集成vector manifest缺少files")
    bindings: list[dict[str, object]] = []
    for name, metadata in sorted(files.items()):
        if not isinstance(metadata, dict):
            raise ValueError(f"vector file {name}元数据不合法")
        path = manifest_path.parent / str(metadata.get("file", ""))
        entries = int(metadata.get("entries", -1))
        if (
            not path.is_file()
            or entries != len(path.read_text(encoding="ascii").splitlines())
            or metadata.get("sha256") != sha256(path)
        ):
            raise ValueError(f"vector file {name} SHA/entries失配")
        bindings.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "entries": entries,
                "sha256": sha256(path),
            }
        )
    return bindings


def load_argv(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"argv 文件不是非空字符串数组: {path}")
    return value


def validate_exact_run_argv(
    argv: list[str],
    executable: Path,
    actual: Path,
    vector_bindings: list[dict[str, object]],
    identity: dict[str, int],
) -> int:
    vectors = {str(binding["name"]): Path(str(binding["path"])).resolve() for binding in vector_bindings}
    if set(vectors) < {"combined_head_inputs", "projection_weights"}:
        raise ValueError("run argv 缺少输入/权重 vector binding")
    service_seed = (
        17717
        + identity["sample"] * 37
        + identity["stage"] * 7
        + identity["block"]
    )
    expected = [
        str(executable.resolve()),
        f'+INPUTS={vectors["combined_head_inputs"]}',
        f'+WEIGHTS={vectors["projection_weights"]}',
        f'+STAGE_ID={identity["stage"]}',
        f'+BLOCK_ID={identity["block"]}',
        f'+WINDOW_ID={identity["window"]}',
        "+NO_ACC_CHECK",
        f"+SERVICE_SEED={service_seed}",
        f"+ACTUAL_ACC_FILE={actual.resolve()}",
    ]
    if argv != expected:
        raise ValueError("actual 精确 run argv 与 task/vector/output 合同不一致")
    return service_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", choices=("icarus", "verilator"), required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--filelist", nargs="+", type=Path, required=True)
    parser.add_argument("--command")
    parser.add_argument("--compile-command")
    parser.add_argument("--run-argv", type=Path)
    parser.add_argument("--compile-argv", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--tool-versions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    log_text = args.log.read_text(encoding="utf-8")
    cycles, final_count = parse_unique_terminal(log_text)
    stage, block, window = parse_unique_identity(log_text)
    plan = json.loads(args.task_plan.read_text(encoding="utf-8"))
    vectors = json.loads(args.vector_manifest.read_text(encoding="utf-8"))
    heads = int(plan.get("heads", 0))
    if (stage, block, window) != tuple(
        int(plan.get(key, -1)) for key in ("stage", "block", "window")
    ):
        raise ValueError("DUT 日志 stage/block/window 与 task plan 不一致")
    expected_count = heads * TOKENS * OUT_DIM
    values = parse_acc32(args.actual)
    vector_bindings = validate_vector_files(args.vector_manifest, vectors)
    if (
        vectors.get("schema")
        != "local5_erep_integrated_cross_head_vectors_v1"
        or vectors.get("task_plan_sha256") != sha256(args.task_plan)
        or len(values) != expected_count
        or final_count != expected_count
    ):
        raise ValueError("集成跨头actual数量或来源绑定不一致")
    if not args.executable.is_file() or not args.tool_versions.is_file():
        raise ValueError("仿真executable/tool versions不存在")
    exact_argv = args.run_argv is not None or args.compile_argv is not None
    if exact_argv != (args.run_argv is not None and args.compile_argv is not None):
        raise ValueError("run/compile argv 必须同时提供")
    if exact_argv and args.release_manifest is None:
        raise ValueError("精确 argv receipt 必须绑定 release manifest")
    if not exact_argv and (not args.command or not args.compile_command):
        raise ValueError("旧 receipt 路径必须提供 command/compile-command")
    run_argv = load_argv(args.run_argv) if args.run_argv else None
    compile_argv = load_argv(args.compile_argv) if args.compile_argv else None
    release_manifest = None
    release_root = None
    if args.release_manifest is not None:
        release_manifest = json.loads(
            args.release_manifest.read_text(encoding="utf-8")
        )
        if (
            not isinstance(release_manifest, dict)
            or release_manifest.get("schema") != RELEASE_SCHEMA
            or release_manifest.get("status") != "SEALED_RTL_RELEASE_NOT_G0"
        ):
            raise ValueError("release manifest schema/status 不合法")
        release_root = args.release_manifest.resolve().parent
        release_manifest = verify_release(release_root)
    identity = {
        "sample": int(plan["sample"]),
        "stage": stage,
        "block": block,
        "window": window,
        "heads": heads,
    }
    service_seed = None
    if exact_argv:
        build = release_manifest["builds"].get(str(heads), {})
        if (
            compile_argv != build.get("compile_argv")
            or args.executable.resolve()
            != (release_root / str(build.get("executable_path", ""))).resolve()
            or sha256(args.executable) != build.get("executable_sha256")
            or args.tool_versions.resolve()
            != (release_root / str(release_manifest["tool_versions_path"])).resolve()
        ):
            raise ValueError("actual 未绑定对应 H-class release build/tool")
        service_seed = validate_exact_run_argv(
            run_argv, args.executable, args.actual, vector_bindings, identity
        )
        filelist = [
            {
                "file": str((release_root / "source" / binding["path"]).resolve()),
                "sha256": binding["sha256"],
            }
            for binding in release_manifest["source_bindings"]
            if binding["path"].endswith(".sv")
        ]
    else:
        filelist = []
        for source in args.filelist:
            path = source.resolve()
            filelist.append({"file": str(path), "sha256": sha256(path)})
    receipt = {
        "schema": "local5_erep_integrated_cross_head_actual_v1",
        "status": "PASS_ACTUAL_NOT_G0",
        "evidence": "[rtl]",
        "formal_g0": "DENY",
        "simulator": args.simulator,
        "run_command": args.command,
        "compile_command": args.compile_command,
        "executable": str(args.executable.resolve()),
        "executable_sha256": sha256(args.executable),
        "tool_versions": str(args.tool_versions.resolve()),
        "tool_versions_sha256": sha256(args.tool_versions),
        "cycles": cycles,
        "identity": identity,
        "actual_scalar_count": len(values),
        "actual_acc32": str(args.actual.resolve()),
        "actual_acc32_sha256": sha256(args.actual),
        "raw_log": str(args.log.resolve()),
        "raw_log_sha256": sha256(args.log),
        "task_plan_sha256": sha256(args.task_plan),
        "vector_manifest_sha256": sha256(args.vector_manifest),
        "filelist": filelist,
        "vector_file_bindings": vector_bindings,
        "boundary": (
            "actual由集成cross-head DUT在NO_ACC_CHECK下导出；"
            "DUT/TB未读取软件expected；输出为"
            "pre-bias/pre-BN/pre-requant/pre-residual Acc32"
        ),
    }
    if exact_argv:
        receipt.update(
            {
                "provenance_level": "exact_argv_sealed_release",
                "run_argv": run_argv,
                "run_argv_file": str(args.run_argv.resolve()),
                "run_argv_file_sha256": sha256(args.run_argv),
                "compile_argv": compile_argv,
                "compile_argv_file": str(args.compile_argv.resolve()),
                "compile_argv_file_sha256": sha256(args.compile_argv),
                "release_manifest": str(args.release_manifest.resolve()),
                "release_manifest_sha256": sha256(args.release_manifest),
                "service_seed": service_seed,
            }
        )
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
