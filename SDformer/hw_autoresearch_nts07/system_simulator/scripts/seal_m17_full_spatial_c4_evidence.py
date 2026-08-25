#!/usr/bin/env python3
"""Create a self-contained, hash-complete M17 evidence tar without overwriting evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


SOURCE_PATHS = (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py",
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/h67_full_spatial_c4_oracle.py",
    "hw_autoresearch_nts07/system_simulator/scripts/analyze_m17_full_spatial_c4_oracle.py",
    "hw_autoresearch_nts07/system_simulator/scripts/seal_m17_full_spatial_c4_evidence.py",
    "hw_autoresearch_nts07/system_simulator/tests/test_h67_full_spatial_c4_oracle.py",
    "hw_autoresearch_nts07/system_simulator/tests/test_m17_full_spatial_c4_analyzer.py",
    "hw_autoresearch_nts07/system_simulator/tests/test_m17_evidence_sealer.py",
    "hw_autoresearch_nts07/system_handoff/scripts/run_h67_full_spatial_c4_oracle.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_output(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def git_stream_identity(repo: Path, *args: str) -> tuple[str, int]:
    process = subprocess.Popen(
        ["git", "-C", str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("failed to capture Git identity stream")
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    stderr = process.stderr.read() if process.stderr is not None else b""
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(
            returncode, ["git", "-C", str(repo), *args], stderr=stderr,
        )
    return digest.hexdigest(), size


def collect_files(root: Path) -> list[Path]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"evidence tree contains a symlink: {path}")
        if path.is_file():
            result.append(path)
    if not result:
        raise ValueError("evidence run directory is empty")
    return result


def accelerator_identity() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cuda_available": available,
            "cuda_device_count": torch.cuda.device_count() if available else 0,
            "cuda_devices": [
                {
                    "index": index, "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count() if available else 0)
            ],
        }
    except Exception as exc:
        return {"capture_error": f"{type(exc).__name__}: {exc}"}


def add_file(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = archive.gettarinfo(str(source), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input", action="append", nargs=2, metavar=("LABEL", "PATH"), default=[],
        help="identity-critical external input copied into the sealed evidence",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    checksum_path = output.with_name(output.name + ".sha256")
    if not run_dir.is_dir():
        raise ValueError("invalid M17 run directory")
    try:
        git_output(repo, "rev-parse", "--show-toplevel")
    except subprocess.CalledProcessError as exc:
        raise ValueError("source root is not inside a Git worktree") from exc
    if output.exists() or checksum_path.exists():
        raise FileExistsError("refusing to overwrite sealed M17 evidence")
    if run_dir == output or run_dir in output.parents:
        raise ValueError("M17 archive must be outside the result directory")
    required_results = (
        run_dir / "full_spatial_c4/manifest.json",
        run_dir / "full_spatial_c4/prototypes.json",
        run_dir / "full_spatial_c4/ordered_stream.npz",
        run_dir / "dual_line_operator_trace.csv",
        run_dir / "m17_reconciliation.json",
        run_dir / "profile_cmdline.txt",
        run_dir / "analyzer_cmdline.txt",
        run_dir / "profile_environment.txt",
        run_dir / "console.log",
    )
    if any(not path.is_file() for path in required_results):
        raise ValueError("M17 result directory is incomplete")

    sources = [(relative, repo / relative) for relative in SOURCE_PATHS]
    if any(not path.is_file() for _, path in sources):
        raise ValueError("M17 source snapshot population is incomplete")
    labels: set[str] = set()
    external_inputs = []
    for label, raw_path in args.input:
        if not label or label in labels or "/" in label or ".." in label:
            raise ValueError("M17 evidence input labels must be unique safe names")
        labels.add(label)
        path = Path(raw_path).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"M17 evidence input is not a regular file: {label}")
        external_inputs.append((label, path))

    status = git_output(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    unstaged_diff_sha, unstaged_diff_bytes = git_stream_identity(
        repo, "diff", "--binary", "--no-ext-diff",
    )
    staged_diff_sha, staged_diff_bytes = git_stream_identity(
        repo, "diff", "--cached", "--binary", "--no-ext-diff",
    )
    result_files = collect_files(run_dir)
    manifest: dict[str, Any] = {
        "schema": "m17_self_contained_evidence_seal_v1",
        "status": "SEALED_HASH_COMPLETE_SOURCE_KERNEL_EVIDENCE_NOT_SYSTEM_SPEEDUP",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "invocation": list(sys.argv),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "hostname": platform.node(), "cwd": os.getcwd(),
            "accelerator": accelerator_identity(),
            "curated_variables": {
                key: os.environ.get(key) for key in (
                    "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LC_ALL", "LANG",
                )
            },
        },
        "git_identity": {
            "head": git_output(repo, "rev-parse", "HEAD").decode().strip(),
            "branch": git_output(repo, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
            "status_porcelain_v1_z_sha256": bytes_sha256(status),
            "status_entries": len([item for item in status.split(b"\0") if item]),
            "unstaged_binary_diff_sha256": unstaged_diff_sha,
            "unstaged_binary_diff_bytes": unstaged_diff_bytes,
            "staged_binary_diff_sha256": staged_diff_sha,
            "staged_binary_diff_bytes": staged_diff_bytes,
        },
        "sources": {
            relative: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for relative, path in sources
        },
        "results": {
            str(path.relative_to(run_dir)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in result_files
        },
        "external_inputs": {
            label: {
                "original_name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
            }
            for label, path in external_inputs
        },
        "claim_boundary": (
            "Immutable source, dependency-input, invocation/environment, git-dirty, and result "
            "identity for exact same-sample M17 source-kernel evidence. The checkpoint is content-"
            "identified but not copied. Not dynamic-BN readiness, ATLIF overlap, memory-system "
            "timing, full-system speedup, VCS equivalence, energy, or PPA."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m17_seal_", dir=str(output.parent)) as directory:
        manifest_path = Path(directory) / "evidence_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
            add_file(archive, manifest_path, "evidence_manifest.json")
            for relative, path in sources:
                add_file(archive, path, "sources/" + relative)
            for path in result_files:
                add_file(archive, path, "results/" + str(path.relative_to(run_dir)))
            for label, path in sorted(external_inputs):
                add_file(archive, path, f"inputs/{label}/{path.name}")
    archive_sha = sha256(output)
    checksum_path.write_text(f"{archive_sha}  {output.name}\n", encoding="utf-8")
    print(f"PASS_M17_EVIDENCE_SEALED sha256={archive_sha} bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
