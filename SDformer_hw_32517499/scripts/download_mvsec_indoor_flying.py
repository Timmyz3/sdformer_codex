#!/usr/bin/env python3
"""Download MVSEC indoor_flying bags with wget resume + md5 verification."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from tonic.datasets import MVSEC


REPO_ROOT = Path(__file__).resolve().parents[1]
MVSEC_ROOT = REPO_ROOT / "third_party" / "SDformerFlow" / "data" / "Datasets" / "MVSEC"
SCENE = "indoor_flying"
DEFAULT_SEQUENCES = ["indoor_flying1", "indoor_flying2", "indoor_flying3"]


def sequence_files(sequence: str) -> list[tuple[str, str]]:
    resources = MVSEC.resources[SCENE]
    wanted = {f"{sequence}_data.bag", f"{sequence}_gt.bag"}
    return [(name, md5) for name, md5 in resources if name in wanted]


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_partial_candidates(filename: str, seq_dir: Path) -> list[Path]:
    candidates = [
        seq_dir / filename,
        MVSEC_ROOT / SCENE / filename,
        MVSEC_ROOT / filename.rsplit("_", 1)[0] / filename,
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path.exists() and path.stat().st_size > 0 and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def merge_partial_into(dst: Path, filename: str, seq_dir: Path) -> None:
    candidates = collect_partial_candidates(filename, seq_dir)
    if not candidates:
        return
    best = max(candidates, key=lambda p: p.stat().st_size)
    if dst.exists() and dst.stat().st_size >= best.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if best.resolve() == dst.resolve():
        return
    if dst.exists():
        dst.unlink()
    best.replace(dst)
    print(f"[resume] merged partial {best} -> {dst}")


def wget_download(url: str, dst: Path, retries: int = 5) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "--timeout=60", "--tries=5", "-O", str(dst), url]
    for attempt in range(1, retries + 1):
        print(f"[wget] attempt {attempt}/{retries}: {' '.join(cmd)}")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0:
            return
        print(f"[warn] wget failed with exit={proc.returncode}; retrying after 10s", file=sys.stderr)
        subprocess.run(["sleep", "10"], check=True)
    raise RuntimeError(f"wget failed after {retries} attempts: {dst}")


def download_file(sequence: str, filename: str, md5_hash: str, force: bool = False) -> Path:
    seq_dir = MVSEC_ROOT / sequence
    seq_dir.mkdir(parents=True, exist_ok=True)
    dst = seq_dir / filename
    merge_partial_into(dst, filename, seq_dir)

    if dst.exists() and not force and file_md5(dst) == md5_hash:
        print(f"[skip] verified {dst}")
        return dst

    url = os.path.join(MVSEC.base_url, SCENE, filename)
    print(f"[download] {url} -> {dst}")
    wget_download(url, dst)
    actual = file_md5(dst)
    if actual != md5_hash:
        raise RuntimeError(f"md5 mismatch for {dst}: expected {md5_hash}, got {actual}")
    print(f"[ok] {dst}")
    return dst


def download_sequence(sequence: str, force: bool = False) -> None:
    for filename, md5_hash in sequence_files(sequence):
        download_file(sequence, filename, md5_hash, force=force)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sequences = args.sequence or DEFAULT_SEQUENCES
    for sequence in sequences:
        download_sequence(sequence, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())