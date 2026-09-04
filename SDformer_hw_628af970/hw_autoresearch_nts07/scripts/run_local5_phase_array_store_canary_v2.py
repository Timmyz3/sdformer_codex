#!/usr/bin/env python3
"""运行 H12 Phase Array Store v2 的生成、验证、资源封账与负例回归。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "results/local5_h12_nonzero_identity_phase_canary_v2_20260811"
DEFAULT_IDENTITY = ROOT / "results/local5_identity_service_tables_sample1_h12b2_v4_20260811"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_timed(name: str, argv: list[str], output: Path) -> dict[str, object]:
    stdout = output / f"{name}.stdout"
    stderr = output / f"{name}.stderr"
    timing = output / f"{name}.time"
    command = output / f"{name}.argv.json"
    write_json(command, argv)
    timed = [
        "/usr/bin/time", "-f",
        "wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x",
        "-o", str(timing), *argv,
    ]
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(timed, stdout=out, stderr=err, check=False)
    raw = timing.read_text(encoding="utf-8")
    fields = dict(
        line.split("=", 1) for line in raw.splitlines() if "=" in line
    )
    result = {
        "argv_sha256": sha256(command),
        "stdout_sha256": sha256(stdout),
        "stderr_sha256": sha256(stderr),
        "time_sha256": sha256(timing),
        "wall_seconds": float(fields["wall_seconds"]),
        "user_seconds": float(fields["user_seconds"]),
        "system_seconds": float(fields["system_seconds"]),
        "max_rss_kb": int(fields["max_rss_kb"]),
        "exit_status": int(fields["exit_status"]),
    }
    if completed.returncode != 0 or result["exit_status"] != 0:
        raise RuntimeError(f"{name} 失败，见 {stderr}")
    return result


def chmod_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def file_inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    ]


def environment_snapshot(path: Path) -> dict[str, object]:
    meminfo: dict[str, int] = {}
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        for line in meminfo_path.read_text(encoding="ascii").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if fields and fields[0].isdigit():
                meminfo[key] = int(fields[0])
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    findmnt = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE,SOURCE,TARGET", "--target", str(path)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    disk = shutil.disk_usage(path)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "logical_cpus": os.cpu_count(),
        "cpu_max": cpu_max.read_text(encoding="ascii").strip() if cpu_max.is_file() else None,
        "load_average": list(os.getloadavg()),
        "mem_total_kb": meminfo.get("MemTotal"),
        "mem_available_kb": meminfo.get("MemAvailable"),
        "swap_total_kb": meminfo.get("SwapTotal"),
        "swap_free_kb": meminfo.get("SwapFree"),
        "filesystem": findmnt.stdout.strip() if findmnt.returncode == 0 else None,
        "filesystem_free_bytes": disk.free,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_CASE / "candidate_trace.csv")
    parser.add_argument(
        "--legacy-archive", type=Path,
        default=DEFAULT_CASE / "template_patch/phase_template_patch.npz",
    )
    parser.add_argument(
        "--source-trace-only", action="store_true",
        help="不依赖旧 NPZ；只用原始 trace 逐行和字节 SHA 验证",
    )
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument(
        "--expected-identity-manifest", type=Path,
        default=DEFAULT_IDENTITY / "manifest.json",
    )
    parser.add_argument(
        "--expected-identity-receipt", type=Path,
        default=DEFAULT_IDENTITY / "verification_receipt.json",
    )
    parser.add_argument("--max-rss-kb", type=int, default=512 * 1024)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/local5_h12_phase_array_store_v2_20260812",
    )
    args = parser.parse_args()
    trace = args.trace.resolve()
    legacy = None if args.source_trace_only else args.legacy_archive.resolve()
    output = args.output_dir.resolve()
    expected_identity_manifest = args.expected_identity_manifest.resolve()
    expected_identity_receipt = args.expected_identity_receipt.resolve()
    requested_identity = {
        "sample": args.sample, "stage": args.stage, "block": args.block,
        "window": args.window, "heads": args.heads,
    }
    frozen_manifest = json.loads(expected_identity_manifest.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    staging = output.with_name(f"{output.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    environment_before = environment_snapshot(trace)
    live_sources = [
        (ROOT / "scripts/generate_local5_phase_array_store_v2.py").resolve(),
        (ROOT / "scripts/verify_local5_phase_array_store_v2.py").resolve(),
        (ROOT / "scripts/run_local5_phase_array_store_tamper_v2.py").resolve(),
        Path(__file__).resolve(),
    ]
    source_dir = staging / "source"
    source_dir.mkdir()
    for source in live_sources:
        shutil.copy2(source, source_dir / source.name)
    generator = source_dir / "generate_local5_phase_array_store_v2.py"
    verifier = source_dir / "verify_local5_phase_array_store_v2.py"
    tamper = source_dir / "run_local5_phase_array_store_tamper_v2.py"
    store = staging / "store"
    try:
        generate_resource = run_timed("generate", [
            "python3", str(generator), "--trace", str(trace),
            "--sample", str(args.sample), "--stage", str(args.stage),
            "--block", str(args.block), "--window", str(args.window),
            "--heads", str(args.heads), "--verifier-source", str(verifier),
            "--output-dir", str(store),
        ], staging)
        verify_argv = [
            "python3", str(verifier), "--store-dir", str(store),
            "--source-trace", str(trace),
            "--expected-identity-manifest", str(expected_identity_manifest),
            "--expected-identity-receipt", str(expected_identity_receipt),
            "--output", str(staging / "verification.json"),
        ]
        if legacy is not None:
            verify_argv.extend(["--legacy-archive", str(legacy)])
        verify_resource = run_timed("verify", verify_argv, staging)
        if legacy is not None:
            verify_source_only_resource = run_timed("verify_source_only", [
                "python3", str(verifier), "--store-dir", str(store),
                "--source-trace", str(trace),
                "--expected-identity-manifest", str(expected_identity_manifest),
                "--expected-identity-receipt", str(expected_identity_receipt),
                "--output", str(staging / "verification_source_only.json"),
            ], staging)
        else:
            shutil.copy2(
                staging / "verification.json", staging / "verification_source_only.json"
            )
            verify_source_only_resource = dict(verify_resource)
        tamper_argv = [
            "python3", str(tamper), "--store-dir", str(store),
            "--source-trace", str(trace),
            "--expected-identity-manifest", str(expected_identity_manifest),
            "--expected-identity-receipt", str(expected_identity_receipt),
            "--verifier", str(verifier),
            "--evidence-dir", str(staging / "tamper_evidence"),
            "--output", str(staging / "tamper_regression.json"),
        ]
        tamper_resource = run_timed("tamper", tamper_argv, staging)
        manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
        verification = json.loads((staging / "verification.json").read_text(encoding="utf-8"))
        verification_source_only = json.loads(
            (staging / "verification_source_only.json").read_text(encoding="utf-8")
        )
        tamper_report = json.loads((staging / "tamper_regression.json").read_text(encoding="utf-8"))
        expected_verify_status = (
            "PASS_STREAMING_MMAP_LEGACY_EQUIVALENT_NOT_G0"
            if legacy is not None
            else "PASS_STREAMING_MMAP_SOURCE_TRACE_EQUIVALENT_NOT_G0"
        )
        expected_rows = manifest["expanded_rows"]
        expected_source_sha = manifest["source_trace_sha256"]
        expected_legacy_arrays = 23 if legacy is not None else 0
        expected_legacy_mismatch = 0 if legacy is not None else None
        expected_negative_cases = 10
        if (
            verification.get("status") != expected_verify_status
            or tamper_report.get("status") != "PASS_ALL_TAMPERS_REJECTED_NOT_G0"
            or verification.get("formal_g0") != "DENY"
            or verification_source_only.get("status") != "PASS_STREAMING_MMAP_SOURCE_TRACE_EQUIVALENT_NOT_G0"
            or verification_source_only.get("formal_g0") != "DENY"
            or tamper_report.get("formal_g0") != "DENY"
            or len(manifest.get("arrays", {})) != 27
            or manifest.get("identity") != requested_identity
            or frozen_manifest.get("identity") != requested_identity
            or verification.get("identity") != requested_identity
            or verification.get("frozen_expected_identity") != requested_identity
            or verification_source_only.get("identity") != requested_identity
            or verification_source_only.get("frozen_expected_identity") != requested_identity
            or manifest.get("mmap_page_drop_rows") != 1 << 20
            or verification.get("derived", {}).get("store_arrays") != 27
            or verification.get("expansion", {}).get("rows") != expected_rows
            or verification.get("expansion", {}).get("expanded_trace_sha256") != expected_source_sha
            or verification.get("bindings", {}).get("source_trace_sha256") != expected_source_sha
            or verification.get("expansion", {}).get("mmap_page_drop_rows") != 1 << 18
            or verification.get("expansion", {}).get("mmap_page_drop", {}).get("calls", 0) < 2
            or verification_source_only.get("expansion", {}).get("rows") != expected_rows
            or verification_source_only.get("expansion", {}).get("expanded_trace_sha256") != expected_source_sha
            or verification_source_only.get("expansion", {}).get("mmap_page_drop_rows") != 1 << 18
            or verification_source_only.get("expansion", {}).get("mmap_page_drop", {}).get("calls", 0) < 2
            or verification_source_only.get("legacy_equivalence", {}).get("arrays_compared") != 0
            or verification_source_only.get("legacy_equivalence", {}).get("mismatch") is not None
            or verification.get("legacy_equivalence", {}).get("arrays_compared") != expected_legacy_arrays
            or verification.get("legacy_equivalence", {}).get("mismatch") != expected_legacy_mismatch
            or tamper_report.get("passed") != expected_negative_cases
            or tamper_report.get("total") != expected_negative_cases
            or len(tamper_report.get("cases", [])) != expected_negative_cases
            or generate_resource["max_rss_kb"] > args.max_rss_kb
            or verify_resource["max_rss_kb"] > args.max_rss_kb
            or verify_source_only_resource["max_rss_kb"] > args.max_rss_kb
        ):
            raise ValueError("positive/negative/resource explicit acceptance contract differs")
        old_file_bytes = legacy.stat().st_size if legacy is not None else None
        new_file_bytes = manifest["array_file_bytes_total"]
        tamper_inventory_path = staging / "tamper_evidence_manifest.json"
        write_json(tamper_inventory_path, {
            "schema": "local5_phase_array_store_tamper_evidence_inventory_v1",
            "files": file_inventory(staging / "tamper_evidence"),
        })
        complete = {
            "schema": "local5_phase_array_store_canary_complete_v2",
            "status": "PASS_SEALED_STREAMING_MMAP_CANARY_NOT_G0",
            "evidence": "[rtl-trace-derived]+[独立软件逐行展开验证]+[资源实测]",
            "formal_g0": "DENY",
            "identity": manifest["identity"],
            "verified_metrics": {
                "expanded_rows": verification["expansion"]["rows"],
                "array_mismatch": verification["legacy_equivalence"]["mismatch"],
                "arrays_compared": verification["legacy_equivalence"]["arrays_compared"],
                "store_arrays": verification["derived"]["store_arrays"],
                "negative_cases_passed": tamper_report["passed"],
                "source_trace_only_pass": True,
                "generator_wall_seconds": generate_resource["wall_seconds"],
                "generator_max_rss_kb": generate_resource["max_rss_kb"],
                "verifier_wall_seconds": verify_resource["wall_seconds"],
                "verifier_max_rss_kb": verify_resource["max_rss_kb"],
                "source_only_verifier_max_rss_kb": verify_source_only_resource["max_rss_kb"],
                "legacy_npz_file_bytes": old_file_bytes,
                "array_store_file_bytes": new_file_bytes,
                "array_store_to_legacy_file_ratio": (
                    new_file_bytes / old_file_bytes if old_file_bytes is not None else None
                ),
            },
            "resources": {
                "generator": generate_resource,
                "verifier": verify_resource,
                "verifier_source_only": verify_source_only_resource,
                "tamper": tamper_resource,
            },
            "external_bindings": {
                "source_trace": {"path": str(trace), "sha256": sha256(trace)},
                "expected_identity_manifest": {
                    "path": str(expected_identity_manifest),
                    "sha256": sha256(expected_identity_manifest),
                },
                "expected_identity_receipt": {
                    "path": str(expected_identity_receipt),
                    "sha256": sha256(expected_identity_receipt),
                },
            },
            "source_bindings": {
                source.name: {
                    "path": f"source/{source.name}",
                    "sha256": sha256(source_dir / source.name),
                }
                for source in live_sources
            },
            "internal_bindings": {
                "store_manifest": sha256(store / "manifest.json"),
                "verification": sha256(staging / "verification.json"),
                "verification_source_only": sha256(staging / "verification_source_only.json"),
                "tamper_regression": sha256(staging / "tamper_regression.json"),
                "tamper_evidence_manifest": sha256(tamper_inventory_path),
            },
            "environment": {
                "repeat_count": 1,
                "before": environment_before,
                "after": environment_snapshot(trace),
                "max_rss_acceptance_kb": args.max_rss_kb,
            },
            "boundary": [
                "本包评价单个参数化窗口的验证归档扩展性与 trace 等价性，不是新架构 RTL",
                "wall/RSS 是验证脚本资源，不是硬件周期、功耗、片上存储或 ASIC PPA",
                "单个参数化窗口；formal G0、H24、full encoder 与 DC/STA/SAIF 均未通过",
                "chmod 与同目录 SHA 是内容绑定，不是外部不可篡改信任根",
            ],
        }
        if legacy is not None:
            complete["external_bindings"]["legacy_archive"] = {
                "path": str(legacy), "sha256": sha256(legacy)
            }
        write_json(staging / "complete.json", complete)
        os.replace(staging, output)
        chmod_read_only(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": complete["status"],
        "generator_max_rss_kb": generate_resource["max_rss_kb"],
        "verifier_max_rss_kb": verify_resource["max_rss_kb"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
