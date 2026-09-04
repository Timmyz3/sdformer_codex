#!/usr/bin/env python3
"""限并发运行并验封 Local5 numeric sample shard 批次。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCALARS_PER_SAMPLE = 1_987_200
WINDOWS_PER_SAMPLE = 12
EXPECTED_TOPOLOGY = (
    (0, 0, 3),
    (0, 1, 3),
    (1, 0, 6),
    (1, 1, 6),
    (2, 0, 12),
    (2, 1, 12),
    (2, 2, 12),
    (2, 3, 12),
    (2, 4, 12),
    (2, 5, 12),
    (3, 0, 24),
    (3, 1, 24),
)
EXPECTED_PYTHON = (3, 12)
EXPECTED_NUMPY = "1.26.4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_source_sha(path: Path, expected_sha256: str) -> None:
    """Fail closed if a live launcher changes after the batch plan is frozen."""
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError(f"live source changed after freeze: {path}")


def freeze_runtime_environment(output: Path) -> dict[str, Any]:
    """封存实际执行解释器；证据不能只依赖目录名中的 py312。"""
    import numpy as np

    executable = Path(sys.executable)
    resolved_executable = executable.resolve()
    numpy_file = Path(np.__file__).resolve()
    if (
        sys.version_info[:2] != EXPECTED_PYTHON
        or np.__version__ != EXPECTED_NUMPY
        or resolved_executable != Path("/usr/bin/python3.12")
        or not resolved_executable.is_file()
        or not numpy_file.is_file()
    ):
        raise RuntimeError(
            "runtime contract differs: require /usr/bin/python3.12, "
            f"Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]} and "
            f"NumPy {EXPECTED_NUMPY}; got {sys.executable}, "
            f"{platform.python_version()}, {np.__version__}"
        )
    row = {
        "schema": "local5_numeric_batch_runtime_environment_v1",
        "status": "FROZEN_EXACT_RUNTIME",
        "sys_executable": str(executable),
        "resolved_executable": str(resolved_executable),
        "executable_sha256": sha256(resolved_executable),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numpy_file": str(numpy_file),
        "numpy_file_sha256": sha256(numpy_file),
        "path": os.environ.get("PATH", ""),
        "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
    }
    write_json(output, row)
    return row


def bind_parent_batch(
    parent: Path,
    samples: list[int],
    outputs: dict[int, Path],
) -> dict[str, Any]:
    """绑定首次执行批次，保留 RUN/RESUME/SKIP 的原始语义。"""
    parent = parent.resolve()
    plan_path = parent / "plan.json"
    complete_path = parent / "complete.json"
    plan = read_json(plan_path)
    complete = read_json(complete_path)
    rows = complete.get("rows")
    if (
        plan.get("schema") not in {
            "local5_numeric_sample_batch_plan_v1",
            "local5_numeric_sample_batch_plan_v2",
        }
        or complete.get("schema") not in {
            "local5_numeric_sample_batch_complete_v1",
            "local5_numeric_sample_batch_complete_v2",
        }
        or complete.get("status") != "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0"
        or complete.get("formal_g0") != "DENY"
        or plan.get("samples") != samples
        or complete.get("samples_requested") != samples
        or complete.get("samples_passed") != samples
        or complete.get("samples_failed") != []
        or complete.get("plan_sha256") != sha256(plan_path)
        or not isinstance(rows, list)
        or len(rows) != len(samples)
    ):
        raise ValueError("parent batch contract differs")
    origin_rows: list[dict[str, Any]] = []
    for sample in samples:
        row = next(
            (value for value in rows if value.get("sample") == sample), None
        )
        receipt_path = parent / f"sample{sample}.receipt.json"
        if (
            not isinstance(row, dict)
            or row.get("execution") not in {
                "RUN", "RESUME_INCOMPLETE_SHARD", "SKIP_ALREADY_SEALED"
            }
            or Path(str(row.get("output", ""))).resolve() != outputs[sample].resolve()
            or read_json(receipt_path) != row
        ):
            raise ValueError(f"parent batch sample{sample} provenance differs")
        origin_rows.append({
            "sample": sample,
            "execution": row["execution"],
            "sample_receipt_sha256": sha256(receipt_path),
        })
    return {
        "schema": "local5_numeric_parent_batch_binding_v1",
        "root": str(parent),
        "plan_sha256": sha256(plan_path),
        "complete_sha256": sha256(complete_path),
        "rows": origin_rows,
    }


def validate_sha256_list(path: Path, allowed_root: Path) -> None:
    """验证 sha256sum 清单，且拒绝跳出当前 shard 的绝对路径。"""
    allowed_root = allowed_root.resolve()
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError(f"empty SHA256 list: {path}")
    for row in rows:
        digest, separator, target_text = row.partition("  ")
        target = Path(target_text).resolve()
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not target.is_relative_to(allowed_root)
            or not target.is_file()
            or sha256(target) != digest
        ):
            raise ValueError(f"invalid SHA256 binding in {path}: {row}")


def validate_acc32_archive(
    path: Path,
    sample: int,
    report_windows: list[dict[str, Any]],
) -> None:
    """直接检查归档中的 expected/actual，避免依赖 merge 报告自证。"""
    import numpy as np

    required = {
        "schema_version",
        "window_sample",
        "window_stage",
        "window_block",
        "window_token",
        "window_weight",
        "window_heads",
        "window_value_offsets",
        "expected_acc32",
        "actual_acc32",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"Acc32 archive schema differs: {path}")
        schema = archive["schema_version"]
        expected = archive["expected_acc32"]
        actual = archive["actual_acc32"]
        samples = archive["window_sample"]
        stages = archive["window_stage"]
        blocks = archive["window_block"]
        tokens = archive["window_token"]
        weights = archive["window_weight"]
        heads = archive["window_heads"]
        offsets = archive["window_value_offsets"]
        topology = tuple(zip(stages.tolist(), blocks.tolist(), heads.tolist()))
        report_sequence = tuple(
            (
                row.get("stage"),
                row.get("block"),
                row.get("heads"),
                row.get("window"),
                row.get("weight"),
            )
            for row in report_windows
        )
        archive_sequence = tuple(zip(
            stages.tolist(),
            blocks.tolist(),
            heads.tolist(),
            tokens.tolist(),
            weights.tolist(),
        ))
        expected_offsets = np.zeros(WINDOWS_PER_SAMPLE + 1, dtype=np.int64)
        expected_offsets[1:] = np.cumsum(heads.astype(np.int64) * 450 * 32)
        if (
            schema.dtype != np.uint16
            or schema.shape != (1,)
            or int(schema[0]) != 4
            or samples.dtype != np.uint16
            or samples.shape != (WINDOWS_PER_SAMPLE,)
            or stages.dtype != np.uint8
            or stages.shape != (WINDOWS_PER_SAMPLE,)
            or blocks.dtype != np.uint8
            or blocks.shape != (WINDOWS_PER_SAMPLE,)
            or tokens.dtype != np.uint16
            or tokens.shape != (WINDOWS_PER_SAMPLE,)
            or weights.dtype != np.uint16
            or weights.shape != (WINDOWS_PER_SAMPLE,)
            or heads.dtype != np.uint8
            or heads.shape != (WINDOWS_PER_SAMPLE,)
            or offsets.dtype != np.int64
            or offsets.shape != (WINDOWS_PER_SAMPLE + 1,)
            or expected.dtype != np.int32
            or actual.dtype != np.int32
            or expected.shape != (SCALARS_PER_SAMPLE,)
            or actual.shape != expected.shape
            or not np.array_equal(expected, actual)
            or not np.all(samples == sample)
            or topology != EXPECTED_TOPOLOGY
            or archive_sequence != report_sequence
            or not np.array_equal(offsets, expected_offsets)
            or int(expected_offsets[-1]) != SCALARS_PER_SAMPLE
        ):
            raise ValueError(f"Acc32 archive content differs: {path}")


def validate_sample(
    sample: int,
    output: Path,
    release_manifest_sha: str,
) -> dict[str, Any]:
    complete_path = output / "complete.json"
    report_path = output / "shard/numeric_shard_report.json"
    binding_path = output / "release_binding.json"
    result_list_path = output / "result_sha256.txt"
    receipt_list_path = output / "receipt_sha256.txt"
    window_list_path = output / "window_receipt_sha256.txt"
    complete = read_json(complete_path)
    report = read_json(report_path)
    binding = read_json(binding_path)
    windows = report.get("windows")
    topology = (
        tuple((row.get("stage"), row.get("block"), row.get("heads")) for row in windows)
        if isinstance(windows, list)
        else set()
    )
    archive = Path(str(report.get("archive", ""))).resolve()
    manifest = Path(str(binding.get("release_manifest", ""))).resolve()
    release_complete = Path(str(binding.get("release_complete", ""))).resolve()
    if (
        complete.get("schema") != "local5_erep_numeric_sample_shard_complete_v1"
        or complete.get("status") != "PASS_NUMERIC_SAMPLE_SHARD_NOT_G0"
        or complete.get("formal_g0") != "DENY"
        or complete.get("sample") != sample
        or complete.get("release_binding_sha256") != sha256(binding_path)
        or complete.get("result_sha256_file_sha256") != sha256(result_list_path)
        or binding.get("schema") != "local5_erep_numeric_sample_release_binding_v1"
        or binding.get("status") != "PASS_RELEASE_BOUND_NOT_G0"
        or binding.get("formal_g0") != "DENY"
        or binding.get("release_manifest_sha256") != release_manifest_sha
        or not manifest.is_file()
        or sha256(manifest) != release_manifest_sha
        or not release_complete.is_file()
        or binding.get("release_complete_sha256") != sha256(release_complete)
        or report.get("schema") != "local5_erep_numeric_sample_shard_v1"
        or report.get("status") != "PASS_NUMERIC_SAMPLE_SHARD_NOT_G0"
        or report.get("formal_g0") != "DENY"
        or report.get("sample") != sample
        or report.get("window_count") != WINDOWS_PER_SAMPLE
        or report.get("final_acc32_scalar_count") != SCALARS_PER_SAMPLE
        or report.get("mismatch_count") != 0
        or report.get("max_abs_error") != 0
        or report.get("release_manifest_sha256") != release_manifest_sha
        or not isinstance(windows, list)
        or len(windows) != WINDOWS_PER_SAMPLE
        or topology != EXPECTED_TOPOLOGY
        or any(
            row.get("sample") != sample
            or row.get("mismatch_count") != 0
            or row.get("max_abs_error") != 0
            for row in windows
        )
        or not archive.is_file()
        or not archive.is_relative_to(output.resolve())
        or report.get("archive_sha256") != sha256(archive)
    ):
        raise ValueError(f"sample {sample} sealed shard contract differs")
    validate_sha256_list(window_list_path, output)
    validate_sha256_list(result_list_path, output)
    validate_sha256_list(receipt_list_path, output)
    validate_acc32_archive(archive, sample, windows)
    return {
        "sample": sample,
        "status": "PASS",
        "output": str(output),
        "complete_sha256": sha256(complete_path),
        "report_sha256": sha256(report_path),
        "acc32_archive_sha256": report["archive_sha256"],
        "windows": report["window_count"],
        "acc32_scalars": report["final_acc32_scalar_count"],
        "mismatch": report["mismatch_count"],
        "verification_cycles": report["total_regression_cycles"],
    }


def sample_execution_mode(output: Path) -> str:
    if (output / "complete.json").is_file() and (
        output / "shard/numeric_shard_report.json"
    ).is_file():
        return "SKIP_ALREADY_SEALED"
    if output.exists() and not output.is_dir():
        raise ValueError(f"sample output is not a directory: {output}")
    return "RESUME_INCOMPLETE_SHARD" if output.exists() else "RUN"


def run_sample(
    sample: int,
    profile: Path,
    release: Path,
    output: Path,
    batch_dir: Path,
    release_manifest_sha: str,
    launcher: Path,
    launcher_sha256: str,
) -> dict[str, Any]:
    execution = sample_execution_mode(output)
    if execution == "SKIP_ALREADY_SEALED":
        return validate_sample(sample, output, release_manifest_sha) | {
            "execution": execution
        }
    log = batch_dir / f"sample{sample}.log"
    timing = batch_dir / f"sample{sample}.time.json"
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "SAMPLE": str(sample),
        "PROFILE": str(profile),
        "OUT_DIR": str(output),
        "RELEASE_DIR": str(release),
        "RELEASE_SERVICE_MODE": "identity",
        "RELEASE_WEIGHT_HOLD_CYCLES": "0",
    })
    start = time.monotonic()
    require_source_sha(launcher, launcher_sha256)
    with log.open("wb") as handle:
        completed = subprocess.run(
            ["bash", str(launcher)],
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - start
    require_source_sha(launcher, launcher_sha256)
    write_json(timing, {
        "schema": "local5_numeric_sample_batch_timing_v1",
        "sample": sample,
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "log_sha256": sha256(log),
    })
    if completed.returncode != 0:
        raise RuntimeError(f"sample {sample} failed; see {log}")
    return validate_sample(sample, output, release_manifest_sha) | {
        "execution": execution,
        "wall_seconds": elapsed,
        "log_sha256": sha256(log),
        "timing_sha256": sha256(timing),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809",
    )
    parser.add_argument(
        "--release", type=Path,
        default=ROOT / "results/local5_erep_numeric_rtl_release_v5_20260811",
    )
    parser.add_argument(
        "--output-prefix", type=str,
        default="local5_erep_numeric_sample{sample}_shard_v5_batch_20260812",
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument(
        "--parent-batch", type=Path,
        help="可选首次执行/恢复批次；提供时当前批次仅复验并绑定原始执行语义",
    )
    args = parser.parse_args()
    samples = sorted(set(args.samples))
    if (
        not samples or any(sample < 0 or sample >= 100 for sample in samples)
        or args.jobs < 1 or args.jobs > 8
        or "{sample}" not in args.output_prefix
    ):
        raise ValueError("invalid samples/jobs/output-prefix")
    profile = args.profile.resolve()
    release = args.release.resolve()
    batch_dir = args.batch_dir.resolve()
    if batch_dir.exists():
        raise FileExistsError(f"batch directory exists: {batch_dir}")
    batch_dir.mkdir(parents=True)
    source_dir = batch_dir / "source"
    source_dir.mkdir()
    runtime_environment = freeze_runtime_environment(
        batch_dir / "runtime_environment.json"
    )
    runtime_environment_sha256 = sha256(
        batch_dir / "runtime_environment.json"
    )
    source_files = [
        Path(__file__).resolve(),
        Path(__file__).with_name("test_run_local5_numeric_sample_batch_v2.py").resolve(),
        (ROOT / "sim_qfit/run_local5_erep_numeric_sample_shard.sh").resolve(),
    ]
    source_rows = []
    for source_path in source_files:
        snapshot = source_dir / source_path.name
        shutil.copyfile(source_path, snapshot)
        snapshot.chmod(0o444)
        source_rows.append({
            "name": source_path.name,
            "path": str(snapshot),
            "live_path": str(source_path),
            "sha256": sha256(snapshot),
        })
    release_manifest = release / "release_manifest.json"
    release_manifest_sha = sha256(release_manifest)
    release_complete = read_json(release / "release_complete.json")
    if (
        release_complete.get("schema")
        != "local5_erep_numeric_rtl_release_complete_v2"
        or release_complete.get("status") != "PASS_RELEASE_SEALED_NOT_G0"
        or release_complete.get("formal_g0") != "DENY"
    ):
        raise ValueError("release is not the sealed v5 non-G0 release")
    outputs = {
        sample: ROOT / "results" / args.output_prefix.format(sample=sample)
        for sample in samples
    }
    parent_batch = (
        bind_parent_batch(args.parent_batch, samples, outputs)
        if args.parent_batch is not None else None
    )
    launcher = source_files[2]
    launcher_sha256 = source_rows[2]["sha256"]
    require_source_sha(launcher, launcher_sha256)
    plan = {
        "schema": "local5_numeric_sample_batch_plan_v2",
        "status": "FROZEN_BEFORE_RUN_NOT_G0",
        "formal_g0": "DENY",
        "samples": samples,
        "jobs": args.jobs,
        "profile": str(profile),
        "release": str(release),
        "release_manifest_sha256": release_manifest_sha,
        "outputs": {str(sample): str(path) for sample, path in outputs.items()},
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha256,
        "parent_batch": parent_batch,
        "origin_policy": (
            "BOUND_PARENT_BATCH" if parent_batch is not None
            else "SELF_FIRST_EXECUTION"
        ),
        "source_snapshots": source_rows,
        "invocation_launcher": {
            "live_path": str(launcher),
            "snapshot_path": source_rows[2]["path"],
            "sha256": launcher_sha256,
            "policy": "每个 sample 执行前后均 fail-closed 校验 live SHA",
        },
        "boundary": [
            "每个 sample 为 12-window numeric Acc32 shard，不生成正式 phase ledger",
            "批次完成仍不是 formal G0、性能或 PPA 证据",
        ],
    }
    write_json(batch_dir / "plan.json", plan)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_sample, sample, profile, release, outputs[sample], batch_dir,
                release_manifest_sha, launcher, launcher_sha256,
            ): sample
            for sample in samples
        }
        for future in concurrent.futures.as_completed(futures):
            sample = futures[future]
            try:
                row = future.result()
                rows.append(row)
                write_json(batch_dir / f"sample{sample}.receipt.json", row)
            except BaseException as error:
                failure = {
                    "sample": sample,
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                }
                failures.append(failure)
                write_json(batch_dir / f"sample{sample}.failure.json", failure)
    rows.sort(key=lambda row: row["sample"])
    failures.sort(key=lambda row: row["sample"])
    require_source_sha(launcher, launcher_sha256)
    result = {
        "schema": "local5_numeric_sample_batch_complete_v2",
        "status": (
            "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0" if not failures
            else "FAIL_NUMERIC_SAMPLE_BATCH_NOT_G0"
        ),
        "evidence": "[rtl]+[软件整数金参考]+[rtl-build-provenance]",
        "formal_g0": "DENY",
        "samples_requested": samples,
        "samples_passed": [row["sample"] for row in rows],
        "samples_failed": [row["sample"] for row in failures],
        "windows_passed": sum(row["windows"] for row in rows),
        "acc32_scalars": sum(row["acc32_scalars"] for row in rows),
        "mismatch": sum(row["mismatch"] for row in rows),
        "verification_regression_cycles": sum(
            row["verification_cycles"] for row in rows
        ),
        "release_manifest_sha256": release_manifest_sha,
        "plan_sha256": sha256(batch_dir / "plan.json"),
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha256,
        "parent_batch": parent_batch,
        "origin_policy": plan["origin_policy"],
        "rows": rows,
        "failures": failures,
        "source": source_rows[0],
        "source_snapshots": source_rows,
        "boundary": [
            "批次仅提高 numeric sample coverage；正式 phase ledger 与 admission 未生成",
            "回归 cycle 是验证环境记账，不是部署 latency、吞吐或架构性能",
            "formal G0、full encoder、DC/STA/SAIF 与 ASIC PPA 均未完成",
        ],
    }
    write_json(batch_dir / "complete.json", result)
    receipt_targets = [
        batch_dir / "plan.json",
        batch_dir / "complete.json",
        batch_dir / "runtime_environment.json",
    ]
    receipt_targets.extend(batch_dir / f"sample{sample}.receipt.json" for sample in samples)
    receipt_targets.extend(Path(row["path"]) for row in source_rows)
    (batch_dir / "receipt_sha256.txt").write_text("".join(
        f"{sha256(path)}  {path}\n" for path in receipt_targets if path.is_file()
    ), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "samples_passed": result["samples_passed"],
        "samples_failed": result["samples_failed"],
        "windows_passed": result["windows_passed"],
        "acc32_scalars": result["acc32_scalars"],
    }, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
