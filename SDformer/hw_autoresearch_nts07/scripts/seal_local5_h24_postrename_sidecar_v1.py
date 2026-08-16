#!/usr/bin/env python3
"""为不可变 H24 主包生成 rename 后可重放的外层封存 sidecar。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_STATUS = "PASS_SEALED_H24_PHASE_SUMMARY_PILOT_NOT_G0"
EXPECTED_TRACE_SHA = "096d4e0c6f6154cb80433d088a6355af941046749ed55f6c33da591e8ae56e9c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层不是 JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(argv: list[str], cwd: Path | None = None) -> dict[str, Any]:
    start = time.monotonic()
    completed = subprocess.run(
        argv, cwd=cwd, text=True, capture_output=True, check=False,
    )
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "wall_seconds": time.monotonic() - start,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def make_tree_manifest(primary: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(primary.rglob("*")):
        if path.is_symlink() or (path.exists() and not path.is_file() and not path.is_dir()):
            raise ValueError(f"主包含不允许的文件类型: {path}")
        if path.is_file():
            rows[path.relative_to(primary).as_posix()] = sha256_file(path)
    return rows


def validate_primary(primary: Path) -> tuple[dict[str, Any], dict[str, str]]:
    complete = read_json(primary / "complete.json")
    verification = read_json(primary / "verification.json")
    admission = read_json(primary / "admission_receipt.json")
    evidence_manifest = read_json(primary / "evidence_manifest.json")
    if (
        complete.get("status") != PRIMARY_STATUS
        or complete.get("formal_g0") != "DENY"
        or verification.get("status") != "PASS_H24_COMPACT_TELEMETRY_NOT_G0"
        or verification.get("formal_g0") != "DENY"
        or admission.get("status") != "ADMIT_EVIDENCE_NOT_G0"
        or admission.get("package_digest")
        != sha256_file(primary / "evidence_manifest.json")
        or complete.get("internal_bindings", {}).get("evidence_manifest.json")
        != sha256_file(primary / "evidence_manifest.json")
        or evidence_manifest.get("files") is None
    ):
        raise ValueError("H24 primary package positive admission differs")
    tree = make_tree_manifest(primary)
    for relative, digest in complete.get("internal_bindings", {}).items():
        if tree.get(relative) != digest:
            raise ValueError(f"primary complete binding differs: {relative}")
    if len(tree) != 117:
        raise ValueError(f"primary tree cardinality differs: {len(tree)}")
    return complete, tree


def relocate_staging_path(raw: str, final_root: Path) -> Path:
    marker = ".recovery_staging.553905"
    if marker not in raw:
        raise ValueError(f"不是冻结 recovery-staging 路径: {raw}")
    suffix = raw.split(marker, 1)[1]
    if suffix and not suffix.startswith("/"):
        raise ValueError(f"staging marker 后不是路径边界: {raw}")
    relocated = final_root / suffix.lstrip("/")
    if not relocated.exists():
        raise FileNotFoundError(f"rename 后目标不存在: {relocated}")
    return relocated


def strings_with_staging_path(value: Any) -> list[str]:
    """递归提取 JSON 中的冻结 staging 路径，包含无尾斜杠的根路径。"""
    if isinstance(value, str):
        return [value] if ".recovery_staging.553905" in value else []
    if isinstance(value, list):
        return [item for child in value for item in strings_with_staging_path(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in strings_with_staging_path(child)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    primary = args.primary.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    if output.exists() or staging.exists():
        raise FileExistsError(f"输出已存在: {output} 或 {staging}")
    staging.mkdir(parents=True)
    source = staging / "source"
    source.mkdir()
    script_snapshot = source / Path(__file__).name
    test_live = Path(__file__).with_name(
        "test_seal_local5_h24_postrename_sidecar_v1.py"
    ).resolve()
    test_snapshot = source / test_live.name
    shutil.copyfile(Path(__file__).resolve(), script_snapshot)
    shutil.copyfile(test_live, test_snapshot)
    try:
        complete, tree = validate_primary(primary)
        write_json(staging / "primary_tree_manifest.json", {
            "schema": "local5_h24_primary_tree_manifest_v1",
            "primary": str(primary),
            "primary_complete_sha256": sha256_file(primary / "complete.json"),
            "file_count": len(tree),
            "total_file_bytes": sum(
                (primary / relative).stat().st_size for relative in tree
            ),
            "files": tree,
        })

        stale_records: list[dict[str, Any]] = []
        for name in (
            "payload_audit.json",
            "fast_cross_oracle_recovery_compile.json",
            "fast_cross_oracle_run.json",
            "recovery_test_receipt.json",
        ):
            path = primary / name
            occurrences = strings_with_staging_path(read_json(path))
            for raw in occurrences:
                relocated = relocate_staging_path(raw, primary)
                stale_records.append({
                    "record": name,
                    "historical_path": raw,
                    "relocated_path": str(relocated),
                    "relocated_sha256": (
                        sha256_file(relocated) if relocated.is_file() else None
                    ),
                })
        stale_record_names = {row["record"] for row in stale_records}
        if len(stale_records) != 6 or len(stale_record_names) != 4:
            raise ValueError(
                "冻结主包 stale path 基数不一致: "
                f"records={len(stale_record_names)}, occurrences={len(stale_records)}"
            )

        build_dir = staging / "build"
        build_dir.mkdir()
        oracle_source = primary / "source/local5_cross_protocol_oracle_fast_v1.c"
        oracle_binary = build_dir / "local5_cross_protocol_oracle_fast_v1"
        compile_result = run([
            "cc", "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror",
            str(oracle_source), "-o", str(oracle_binary),
        ])
        if compile_result["returncode"] != 0:
            raise RuntimeError("rename 后 C oracle 编译失败")
        oracle_result = run([
            str(oracle_binary),
            "TOP.tb_qfit_local5_memo_multitile_cross_head.u_executor."
            "g_scalar_cross_head_acc.u_cross_head_accumulator",
            "24", "24", "14400",
        ])
        if oracle_result["returncode"] != 0:
            raise RuntimeError("rename 后 C oracle 运行失败")
        oracle_payload = json.loads(oracle_result["stdout"])
        expected = complete["verified_metrics"]
        if (
            oracle_payload.get("count") != expected["cross_commands"]
            or oracle_payload.get("read_count") != expected["cross_reads"]
            or oracle_payload.get("write_count") != expected["cross_writes"]
            or oracle_payload.get("digest0") != "b62d67328ef9c0d9"
            or oracle_payload.get("digest1") != "579ef539bac6bf11"
        ):
            raise ValueError("rename 后 C oracle 结果与主包不一致")
        write_json(staging / "postrename_oracle_replay.json", {
            "compile": compile_result,
            "run": oracle_result | {"stdout_json": oracle_payload},
            "source_sha256": sha256_file(oracle_source),
            "binary_sha256": sha256_file(oracle_binary),
        })
        oracle_binary_sha256 = sha256_file(oracle_binary)

        test_rows = []
        for test_name in (
            "test_verify_local5_phase_summary_contract_v2.py",
            "test_local5_cross_protocol_oracle_fast_v1.py",
        ):
            snapshot_test = primary / "source" / test_name
            live_test = ROOT / "scripts" / test_name
            if (
                not snapshot_test.is_file()
                or not live_test.is_file()
                or sha256_file(snapshot_test) != sha256_file(live_test)
            ):
                raise ValueError(f"live/snapshot 测试 SHA 不一致: {test_name}")
            result = run(
                ["/usr/bin/python3", str(live_test)],
                cwd=ROOT,
            )
            if result["returncode"] != 0:
                raise RuntimeError(f"rename 后冻结测试失败: {test_name}")
            test_rows.append(result | {
                "test": test_name,
                "execution_mode": "LIVE_REPOSITORY_LAYOUT_SHA_EQUAL_TO_SNAPSHOT",
                "live_path": str(live_test),
                "snapshot_path": str(snapshot_test),
                "live_sha256": sha256_file(live_test),
                "snapshot_sha256": sha256_file(snapshot_test),
            })
        write_json(staging / "postrename_test_replay.json", {
            "schema": "local5_h24_postrename_test_replay_v1",
            "status": "PASS",
            "rows": test_rows,
        })

        reference_trace = Path(
            complete["external_bindings"]["reference_trace"]["path"]
        )
        if (
            not reference_trace.is_file()
            or complete["external_bindings"]["reference_trace"]["sha256"]
            != EXPECTED_TRACE_SHA
            or sha256_file(reference_trace) != EXPECTED_TRACE_SHA
        ):
            raise ValueError("外部 reference trace SHA 不一致")
        write_json(staging / "postrename_path_audit.json", {
            "schema": "local5_h24_postrename_path_audit_v1",
            "status": "PASS_ALL_HISTORICAL_PATHS_RELOCATED",
            "historical_reference_policy": (
                "主包 JSON 保留历史 argv；sidecar 显式绑定最终路径并实跑，不篡改历史记录"
            ),
            "stale_path_occurrences": len(stale_records),
            "stale_path_records": len(stale_record_names),
            "relocations": stale_records,
            "reference_trace": str(reference_trace),
            "reference_trace_sha256": EXPECTED_TRACE_SHA,
        })
        payload = read_json(primary / "payload_audit.json")
        write_json(staging / "size_accounting.json", {
            "schema": "local5_h24_size_accounting_v1",
            "compact_payload_bytes": payload["bytes"],
            "compact_payload_excluded_top_dirs": payload["excluded_top_dirs"],
            "complete_primary_file_bytes": sum(
                (primary / relative).stat().st_size for relative in tree
            ),
            "complete_primary_file_count": len(tree),
            "limit_bytes": payload["limit_bytes"],
            "complete_primary_under_limit": sum(
                (primary / relative).stat().st_size for relative in tree
            ) <= payload["limit_bytes"],
            "boundary": "3.64 MB 仅指排除 build/source 后的 compact payload，不是完整包大小。",
        })
        os.replace(staging, output)
        final_binary = output / "build/local5_cross_protocol_oracle_fast_v1"
        final_replay = run([
            str(final_binary),
            "TOP.tb_qfit_local5_memo_multitile_cross_head.u_executor."
            "g_scalar_cross_head_acc.u_cross_head_accumulator",
            "24", "24", "14400",
        ])
        final_payload = json.loads(final_replay["stdout"])
        if (
            final_replay["returncode"] != 0
            or final_payload != oracle_payload
            or sha256_file(final_binary) != oracle_binary_sha256
        ):
            raise ValueError("最终 sidecar 路径 oracle replay 不一致")
        write_json(output / "final_path_oracle_replay.json", {
            "schema": "local5_h24_final_path_oracle_replay_v1",
            "status": "PASS_FINAL_PATH_REPLAY",
            "run": final_replay | {"stdout_json": final_payload},
            "binary_sha256": sha256_file(final_binary),
        })
        sidecar_files = {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in output.rglob("*")
            if path.is_file()
            and path.name not in {"sidecar_complete.json", "sidecar_manifest.json"}
        }
        write_json(output / "sidecar_manifest.json", {
            "schema": "local5_h24_postrename_sidecar_manifest_v1",
            "files": sidecar_files,
        })
        write_json(output / "sidecar_complete.json", {
            "schema": "local5_h24_postrename_sidecar_complete_v1",
            "status": "PASS_SEALED_POSTRENAME_REPLAY_NOT_G0",
            "formal_g0": "DENY",
            "primary_root": str(primary),
            "primary_complete_sha256": sha256_file(primary / "complete.json"),
            "primary_tree_manifest_sha256": sha256_file(
                output / "primary_tree_manifest.json"
            ),
            "sidecar_manifest_sha256": sha256_file(
                output / "sidecar_manifest.json"
            ),
            "source_sha256": sha256_file(output / "source" / script_snapshot.name),
            "test_source_sha256": sha256_file(
                output / "source" / test_snapshot.name
            ),
            "boundary": [
                "sidecar 不修改主包历史记录，只从最终目录重定位并实跑。",
                "该结果仍是单 H24 窗口验证，不是 formal G0、性能或 PPA。",
            ],
        })
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode & ~(
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                ))
    except BaseException as error:
        failure_root = staging if staging.exists() else output
        write_json(failure_root / "failure_receipt.json", {
            "schema": "local5_h24_postrename_sidecar_failure_v1",
            "status": "FAIL_CLOSED_NOT_G0",
            "formal_g0": "DENY",
            "exception_type": type(error).__name__,
            "exception": str(error),
        })
        raise
    print(json.dumps({
        "status": "PASS_SEALED_POSTRENAME_REPLAY_NOT_G0",
        "primary_files": len(tree),
        "primary_complete_sha256": sha256_file(primary / "complete.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
