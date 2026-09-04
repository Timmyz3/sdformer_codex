#!/usr/bin/env python3
"""从已验封 Local5 numeric batch 构造只读 compact RTL telemetry pre-ledger。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOKENS = 450
OUT_DIM = 32
EXPECTED_KEYS = {
    "memo", "inplace", "acc_backend", "tx_service", "seed", "stage",
    "block", "window", "cycles", "token", "token_delay_sum",
    "weight_delay_sum", "result_service", "hits", "fallback",
    "replay_records", "partial", "final", "child_results",
    "weight_cycles", "frontend_cycles", "readout_cycles",
    "release_cycles", "rmw_cycles", "drain_cycles", "scheduler_cycles",
    "vector", "token_service_hash", "weight_service_hash",
    "result_service_hash",
}
HASH_KEYS = {
    "token_service_hash", "weight_service_hash", "result_service_hash"
}
EXPECTED_TOPOLOGY = (
    (0, 0, 3), (0, 1, 3), (1, 0, 6), (1, 1, 6),
    (2, 0, 12), (2, 1, 12), (2, 2, 12), (2, 3, 12),
    (2, 4, 12), (2, 5, 12), (3, 0, 24), (3, 1, 24),
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
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_telemetry_log(path: Path) -> dict[str, int | str]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("PASS Local5 multi-tile ")
    ]
    if len(lines) != 1:
        raise ValueError(f"{path} must contain exactly one PASS telemetry row")
    fields: dict[str, int | str] = {}
    for token in lines[0].split()[3:]:
        key, separator, value = token.partition("=")
        if not separator or key in fields:
            raise ValueError(f"invalid or duplicate telemetry token: {token}")
        if key in HASH_KEYS:
            if not re.fullmatch(r"[0-9a-f]{16}", value):
                raise ValueError(f"invalid telemetry hash: {token}")
            fields[key] = value
        else:
            if not re.fullmatch(r"[0-9]+", value):
                raise ValueError(f"invalid telemetry integer: {token}")
            fields[key] = int(value)
    if set(fields) != EXPECTED_KEYS:
        raise ValueError("telemetry field set differs")
    return fields


def transaction_delay_sum(seed: int, stream: int, count: int) -> int:
    """独立重算 sealed TB 的 uint32 transaction_delay 总和。"""
    total = 0
    stream_mix = np.uint32(((stream + 1) * 0x7F4A7C15) & 0xFFFFFFFF)
    for start in range(0, count, 1 << 18):
        stop = min(start + (1 << 18), count)
        transaction = np.arange(start + 1, stop + 1, dtype=np.uint32)
        mixed = (
            np.uint32(seed)
            ^ transaction * np.uint32(0x9E3779B9)
            ^ stream_mix
        )
        mixed ^= mixed >> np.uint32(16)
        mixed *= np.uint32(0x045D9F3B)
        mixed ^= mixed >> np.uint32(16)
        total += int(np.sum(1 + (mixed & np.uint32(3)), dtype=np.uint64))
    return total


def load_memh_int32(path: Path) -> np.ndarray:
    values = np.fromiter(
        (int(line.strip(), 16) for line in path.read_text().splitlines()),
        dtype=np.uint32,
    )
    return values.view(np.int32)


def miter_digest(
    identity: dict[str, int], expected: np.ndarray, actual: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(
        [identity[key] for key in ("sample", "stage", "block", "window", "heads")],
        separators=(",", ":"),
    ).encode("ascii"))
    digest.update(expected.tobytes(order="C"))
    digest.update(actual.tobytes(order="C"))
    return digest.hexdigest()


def validate_telemetry_formula(
    telemetry: dict[str, int | str], heads: int
) -> None:
    partial = heads * heads * TOKENS * OUT_DIM
    final = heads * TOKENS * OUT_DIM
    expected = {
        "memo": 0,
        "inplace": 0,
        "acc_backend": 0,
        "tx_service": 1,
        "token": heads * heads * TOKENS,
        "result_service": final,
        "hits": 0,
        "fallback": 0,
        "replay_records": 0,
        "partial": partial,
        "final": final,
        "child_results": partial,
        "readout_cycles": 3 * partial,
        "release_cycles": 2 * heads * heads,
        "rmw_cycles": heads * (heads - 1) * TOKENS * OUT_DIM,
        "scheduler_cycles": 2 * heads * heads,
        "vector": 0,
    }
    if any(telemetry.get(key) != value for key, value in expected.items()):
        raise ValueError("telemetry closed-form contract differs")
    seed = int(telemetry["seed"])
    token_count = heads * heads * TOKENS
    weight_count = heads * heads * OUT_DIM * OUT_DIM
    token_delay = transaction_delay_sum(seed, 0, token_count)
    weight_delay = transaction_delay_sum(seed, 1, weight_count)
    result_delay = transaction_delay_sum(seed, 2, final)
    service_expected = {
        "token_delay_sum": token_delay,
        "weight_delay_sum": weight_delay,
        "weight_cycles": 2 * weight_count + weight_delay,
        "drain_cycles": 3 * final + result_delay,
    }
    if any(telemetry.get(key) != value for key, value in service_expected.items()):
        raise ValueError("telemetry service-delay contract differs")
    for key in ("cycles", "frontend_cycles"):
        if not isinstance(telemetry.get(key), int) or telemetry[key] <= 0:
            raise ValueError(f"telemetry {key} must be positive")


def validate_parent_batch_header(
    batch: dict[str, Any], batch_path: Path
) -> dict[str, Any]:
    source = batch.get("source")
    schema = batch.get("schema")
    if (
        schema not in {
            "local5_numeric_sample_batch_complete_v1",
            "local5_numeric_sample_batch_complete_v2",
        }
        or batch.get("status") != "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0"
        or batch.get("formal_g0") != "DENY"
        or batch.get("mismatch") != 0
        or not isinstance(batch.get("rows"), list)
        or not isinstance(source, dict)
    ):
        raise ValueError("parent numeric batch is not admissible")
    if schema == "local5_numeric_sample_batch_complete_v2":
        runtime = batch.get("runtime_environment")
        snapshots = batch.get("source_snapshots")
        if (
            not isinstance(runtime, dict)
            or runtime.get("status") != "FROZEN_EXACT_RUNTIME"
            or not isinstance(batch.get("runtime_environment_sha256"), str)
            or not isinstance(snapshots, list)
            or len(snapshots) < 3
            or batch.get("origin_policy") not in {
                "SELF_FIRST_EXECUTION", "RECOVERY_WITH_PARENT"
            }
        ):
            raise ValueError("v2 parent batch runtime/source seal is incomplete")
        runtime_path = batch_path.parent / "runtime_environment.json"
        runtime_sha = batch.get("runtime_environment_sha256")
        if (
            not runtime_path.is_file()
            or sha256(runtime_path) != runtime_sha
            or read_json(runtime_path) != runtime
        ):
            raise ValueError("v2 parent batch runtime sidecar binding differs")
        required_runtime = {
            "schema": "local5_numeric_batch_runtime_environment_v1",
            "status": "FROZEN_EXACT_RUNTIME",
        }
        if any(runtime.get(key) != value for key, value in required_runtime.items()):
            raise ValueError("v2 parent batch runtime schema differs")
        for path_key, sha_key in (
            ("resolved_executable", "executable_sha256"),
            ("numpy_file", "numpy_file_sha256"),
        ):
            artifact = Path(str(runtime.get(path_key, ""))).resolve()
            digest = runtime.get(sha_key)
            if (
                not artifact.is_file()
                or not isinstance(digest, str)
                or sha256(artifact) != digest
            ):
                raise ValueError(f"v2 parent runtime artifact differs: {path_key}")
        observed_snapshot_paths: set[Path] = set()
        for index, snapshot in enumerate(snapshots):
            if not isinstance(snapshot, dict):
                raise ValueError(f"v2 parent source snapshot {index} is not an object")
            artifact = Path(str(snapshot.get("path", ""))).resolve()
            digest = snapshot.get("sha256")
            if (
                not artifact.is_file()
                or not artifact.is_relative_to(batch_path.parent)
                or artifact in observed_snapshot_paths
                or not isinstance(digest, str)
                or sha256(artifact) != digest
            ):
                raise ValueError(f"v2 parent source snapshot {index} differs")
            observed_snapshot_paths.add(artifact)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-complete", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    batch_path = args.batch_complete.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    batch = read_json(batch_path)
    source = validate_parent_batch_header(batch, batch_path)
    parent_source = Path(str(source.get("path", ""))).resolve()
    if not parent_source.is_file() or sha256(parent_source) != source.get("sha256"):
        raise ValueError("parent batch validator source binding differs")

    rows = []
    for sample_row in batch["rows"]:
        shard = Path(str(sample_row.get("output", ""))).resolve()
        complete_path = shard / "complete.json"
        report_path = shard / "shard/numeric_shard_report.json"
        if (
            sha256(complete_path) != sample_row.get("complete_sha256")
            or sha256(report_path) != sample_row.get("report_sha256")
        ):
            raise ValueError("parent shard binding differs")
        report = read_json(report_path)
        sample = int(sample_row["sample"])
        windows = report.get("windows")
        if (
            report.get("mismatch_count") != 0
            or report.get("sample") != sample
            or not isinstance(windows, list)
            or tuple(
                (row.get("stage"), row.get("block"), row.get("heads"))
                for row in windows
            ) != EXPECTED_TOPOLOGY
            or any(row.get("sample") != sample for row in windows)
        ):
            raise ValueError("parent shard has a numeric mismatch")
        archive_path = Path(str(report.get("archive", ""))).resolve()
        if (
            not archive_path.is_relative_to(shard)
            or sha256(archive_path) != sample_row.get("acc32_archive_sha256")
        ):
            raise ValueError("parent Acc32 archive binding differs")
        with np.load(archive_path, allow_pickle=False) as archive:
            offsets = archive["window_value_offsets"]
            expected_acc32 = archive["expected_acc32"]
            actual_acc32 = archive["actual_acc32"]
            archive_metadata = tuple(zip(
                archive["window_sample"].tolist(),
                archive["window_stage"].tolist(),
                archive["window_block"].tolist(),
                archive["window_token"].tolist(),
                archive["window_heads"].tolist(),
            ))
        for window_index, window in enumerate(windows):
            stage = int(window["stage"])
            block = int(window["block"])
            window_dir = shard / f"windows/s{stage}_b{block}"
            receipt_path = window_dir / "actual_receipt.json"
            window_complete_path = window_dir / "window_complete.json"
            receipt = read_json(receipt_path)
            window_complete = read_json(window_complete_path)
            identity = receipt.get("identity")
            log_path = Path(str(receipt.get("raw_log", ""))).resolve()
            actual_path = Path(str(receipt.get("actual_acc32", ""))).resolve()
            identity_expected = {
                key: int(window[key])
                for key in ("sample", "stage", "block", "window", "heads")
            }
            artifact_sha = window_complete.get("artifact_sha256")
            start, end = int(offsets[window_index]), int(offsets[window_index + 1])
            archive_expected = expected_acc32[start:end]
            archive_actual = actual_acc32[start:end]
            memh_actual = load_memh_int32(actual_path)
            if (
                receipt.get("schema") != "local5_erep_integrated_cross_head_actual_v1"
                or receipt.get("status") != "PASS_ACTUAL_NOT_G0"
                or receipt.get("formal_g0") != "DENY"
                or receipt.get("simulator") != "verilator"
                or receipt.get("provenance_level") != "exact_argv_sealed_release"
                or not isinstance(identity, dict)
                or identity != identity_expected
                or window_complete.get("schema") != "local5_erep_numeric_window_complete_v1"
                or window_complete.get("status") != "SEALED_READY_FOR_MITER_NOT_G0"
                or window_complete.get("formal_g0") != "DENY"
                or window_complete.get("identity") != identity_expected
                or sha256(window_complete_path) != window["window_complete_sha256"]
                or not isinstance(artifact_sha, dict)
                or artifact_sha.get("actual_receipt") != sha256(receipt_path)
                or artifact_sha.get("raw_log") != sha256(log_path)
                or artifact_sha.get("actual") != sha256(actual_path)
                or not log_path.is_relative_to(shard)
                or sha256(log_path) != receipt.get("raw_log_sha256")
                or not actual_path.is_relative_to(shard)
                or sha256(actual_path) != receipt.get("actual_acc32_sha256")
                or receipt.get("actual_scalar_count")
                != int(window["heads"]) * TOKENS * OUT_DIM
                or archive_metadata[window_index] != (
                    sample, stage, block, int(window["window"]), int(window["heads"])
                )
                or not np.array_equal(archive_expected, archive_actual)
                or not np.array_equal(archive_actual, memh_actual)
                or miter_digest(identity_expected, archive_expected, archive_actual)
                != window["acc32_miter_sha256"]
            ):
                raise ValueError("actual RTL receipt binding differs")
            telemetry = parse_telemetry_log(log_path)
            validate_telemetry_formula(telemetry, int(window["heads"]))
            if (
                telemetry["seed"] != receipt.get("service_seed")
                or telemetry["stage"] != stage
                or telemetry["block"] != block
                or telemetry["window"] != int(window["window"])
                or telemetry["cycles"] != receipt.get("cycles")
                or telemetry["cycles"] != int(window["cycles"])
            ):
                raise ValueError("telemetry identity/cycle differs from receipt")
            rows.append({
                "identity": identity,
                "telemetry": telemetry,
                "actual_receipt_sha256": sha256(receipt_path),
                "window_complete_sha256": sha256(window_complete_path),
                "raw_log_sha256": sha256(log_path),
                "actual_acc32_sha256": sha256(actual_path),
                "acc32_miter_sha256": window["acc32_miter_sha256"],
                "evidence": "[rtl汇总遥测]+[父级数值证据引用]",
            })
    rows.sort(key=lambda row: tuple(
        row["identity"][key] for key in ("sample", "stage", "block")
    ))
    if len(rows) != len(batch["rows"]) * 12 or len({
        tuple(row["identity"][key] for key in ("sample", "stage", "block"))
        for row in rows
    }) != len(rows):
        raise ValueError("compact telemetry coverage is incomplete or duplicate")
    ledger = {
        "schema": "local5_compact_telemetry_preledger_v1",
        "status": "PASS_COMPACT_TELEMETRY_PRELEDGER_NOT_G0",
        "evidence": "[rtl汇总遥测]+[父级数值证据引用]",
        "formal_g0": "DENY",
        "parent_batch": str(batch_path),
        "parent_batch_sha256": sha256(batch_path),
        "sample_count": len(batch["rows"]),
        "window_count": len(rows),
        "acc32_scalars": sum(int(row["final"]) for row in (
            item["telemetry"] for item in rows
        )),
        "verification_cycles": sum(int(item["telemetry"]["cycles"]) for item in rows),
        "parent_runtime_environment_sha256": batch.get("runtime_environment_sha256"),
        "rows": rows,
        "boundary": [
            "这是逐窗口compact RTL汇总遥测健康检查，不是462600-phase逐事件ledger",
            "cycle/frontend是同源验证回归遥测，不是独立相序或性能证据",
            "不替代100/100 numeric、formal admission、full encoder或ASIC PPA",
        ],
    }
    source_dir = output_dir / "source"
    source_dir.mkdir()
    test_path = Path(__file__).with_name(
        "test_build_local5_compact_telemetry_preledger_v1.py"
    ).resolve()
    source_snapshots = []
    for artifact in (Path(__file__).resolve(), test_path):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        snapshot = source_dir / artifact.name
        shutil.copy2(artifact, snapshot)
        source_snapshots.append({
            "name": artifact.name,
            "path": str(snapshot),
            "live_path": str(artifact),
            "sha256": sha256(snapshot),
        })
    ledger["source"] = source_snapshots[0]
    ledger["source_snapshots"] = source_snapshots
    ledger_path = output_dir / "compact_telemetry_preledger.json"
    write_json(ledger_path, ledger)
    complete = {
        "schema": "local5_compact_telemetry_preledger_complete_v1",
        "status": ledger["status"],
        "formal_g0": "DENY",
        "sample_count": ledger["sample_count"],
        "window_count": ledger["window_count"],
        "acc32_scalars": ledger["acc32_scalars"],
        "verification_cycles": ledger["verification_cycles"],
        "ledger_sha256": sha256(ledger_path),
        "source_sha256": ledger["source"]["sha256"],
        "test_source_sha256": source_snapshots[1]["sha256"],
        "source_snapshots": source_snapshots,
        "parent_batch_sha256": ledger["parent_batch_sha256"],
        "parent_runtime_environment_sha256": ledger[
            "parent_runtime_environment_sha256"
        ],
        "boundary": ledger["boundary"],
    }
    write_json(output_dir / "complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
