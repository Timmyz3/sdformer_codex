#!/usr/bin/env python3
"""Local5 跨 Acc 协议高速 oracle 及 main/lower 摘要对拍封装。"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import verify_local5_phase_summary_contract_v2 as contract


ORACLE_SCHEMA = "local5_cross_protocol_fast_oracle_v1"
HEX64 = re.compile(r"[0-9a-f]{16}", re.ASCII)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_oracle(c_source: Path, binary: Path) -> dict[str, Any]:
    if not c_source.is_file() or c_source.is_symlink():
        raise contract.ContractError("C oracle 源码缺失或为符号链接")
    binary.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "cc",
        "-O3",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(c_source),
        "-o",
        str(binary),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        cwd=Path("/tmp"),
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0 or not binary.is_file() or binary.is_symlink():
        raise contract.ContractError(
            "C oracle 编译失败: " + completed.stderr.strip()
        )
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "source_sha256": sha256_file(c_source),
        "binary_sha256": sha256_file(binary),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_oracle(
    binary: Path,
    target_instance: str,
    *,
    heads: int,
    output_tiles: int | None = None,
    addresses_per_tile: int = 14_400,
    address_order_path: Path | None = None,
) -> tuple[contract.CrossProtocolLedger, dict[str, Any]]:
    if not binary.is_file() or binary.is_symlink():
        raise contract.ContractError("C oracle 可执行文件缺失或为符号链接")
    if not isinstance(target_instance, str):
        raise contract.ContractError("target instance 不是字符串")
    try:
        target_bytes = target_instance.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise contract.ContractError("target instance 不是 UTF-8") from exc
    if not target_bytes or len(target_bytes) > 0xFFFF or b"\0" in target_bytes:
        raise contract.ContractError("target instance 长度或 NUL 不合法")
    heads = contract._require_int(heads, "cross protocol heads")
    output_tiles = heads if output_tiles is None else contract._require_int(
        output_tiles, "output tiles"
    )
    addresses_per_tile = contract._require_int(
        addresses_per_tile, "addresses per tile"
    )
    if heads <= 0 or output_tiles <= 0 or addresses_per_tile <= 0:
        raise contract.ContractError("cross protocol dimensions must be positive")
    argv = [
        str(binary),
        target_instance,
        str(heads),
        str(output_tiles),
        str(addresses_per_tile),
    ]
    if address_order_path is not None:
        if not address_order_path.is_file() or address_order_path.is_symlink():
            raise contract.ContractError("地址顺序文件缺失或为符号链接")
        argv.append(str(address_order_path))
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        cwd=Path("/tmp"),
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise contract.ContractError(
            "C oracle 运行失败: " + completed.stderr.strip()
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise contract.ContractError("C oracle 输出不是单一 JSON") from exc
    required = {
        "schema",
        "count",
        "read_count",
        "write_count",
        "digest0",
        "digest1",
    }
    if set(payload) != required or payload.get("schema") != ORACLE_SCHEMA:
        raise contract.ContractError("C oracle JSON schema/字段集合不匹配")
    for name in ("count", "read_count", "write_count"):
        if not isinstance(payload[name], int) or isinstance(payload[name], bool):
            raise contract.ContractError(f"C oracle {name} 不是整数")
    for name in ("digest0", "digest1"):
        if not isinstance(payload[name], str) or HEX64.fullmatch(payload[name]) is None:
            raise contract.ContractError(f"C oracle {name} 不是 16 位小写十六进制")
    expected_half = heads * output_tiles * addresses_per_tile
    if (
        payload["count"] != 2 * expected_half
        or payload["read_count"] != expected_half
        or payload["write_count"] != expected_half
    ):
        raise contract.ContractError("C oracle 违反 closed-form 事件计数")
    ledger = contract.CrossProtocolLedger(
        payload["count"],
        payload["read_count"],
        payload["write_count"],
        int(payload["digest0"], 16),
        int(payload["digest1"], 16),
    )
    report = {
        "argv": argv,
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "binary_sha256": sha256_file(binary),
        "result": payload,
        "stderr": completed.stderr,
    }
    return ledger, report


def verify_cross_summary_pair_fast(
    main: contract.OrderedSummary,
    lower: contract.OrderedSummary,
    *,
    binary: Path,
    heads: int,
    output_tiles: int | None = None,
    addresses_per_tile: int = 14_400,
    address_order_path: Path | None = None,
) -> tuple[contract.CrossProtocolLedger, dict[str, Any]]:
    if main.schema != contract.SUMMARY_SCHEMA:
        raise contract.ContractError("cross summary pair 缺少 main ordered summary")
    contract.validate_observer_summary_binding(
        lower,
        expected_schema=contract.CROSS_SUMMARY_SCHEMA,
        expected_target_instance=main.resources["CROSS_ACC_COMMAND"].instance_path,
    )
    contract.compare_summary_resources(main, lower, ["CROSS_ACC_COMMAND"])
    main_ledger = main.cross_protocol_ledger
    lower_ledger = lower.cross_protocol_ledger
    if main_ledger is None or lower_ledger is None:
        raise contract.ContractError("cross summary pair 缺少协议 ledger")
    if (
        main_ledger.count,
        main_ledger.read_count,
        main_ledger.write_count,
    ) != (
        lower_ledger.count,
        lower_ledger.read_count,
        lower_ledger.write_count,
    ):
        raise contract.ContractError("main/lower cross protocol ledger 计数不一致")
    expected, report = run_oracle(
        binary,
        main.resources["CROSS_ACC_COMMAND"].instance_path,
        heads=heads,
        output_tiles=output_tiles,
        addresses_per_tile=addresses_per_tile,
        address_order_path=address_order_path,
    )
    if main_ledger != expected:
        raise contract.ContractError("main cross protocol 摘要与 C oracle 不一致")
    return expected, report
