#!/usr/bin/env python3
"""Extract and organize MDR batch archives into the SDformerFlow dataset tree."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_DIR = Path("/root/private_data/mdr/train")
DEFAULT_MDR_ROOT = REPO_ROOT / "third_party" / "SDformerFlow" / "data" / "Datasets" / "MDR"
DATATYPES = ("events1", "events2", "best_density_events1", "best_density_events2", "flow")


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob(pattern))


def extract_archives(archive_dir: Path, train_dir: Path) -> None:
    if is_organized(train_dir):
        print("[prepare-mdr] organized MDR tree already exists; skip archive extraction", flush=True)
        return

    archives = sorted(archive_dir.glob("batch_*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"no batch_*.tar.gz archives under {archive_dir}")

    train_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        batch_name = archive.name.removesuffix(".tar.gz")
        batch_dir = train_dir / batch_name
        done_marker = batch_dir / ".extract_done"
        if batch_dir.exists():
            if done_marker.exists():
                print(f"[prepare-mdr] skip existing complete {batch_dir}", flush=True)
                continue
            print(f"[prepare-mdr] remove incomplete {batch_dir}", flush=True)
            shutil.rmtree(batch_dir)
        print(f"[prepare-mdr] extracting {archive} -> {train_dir}", flush=True)
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(train_dir)], check=True)
        done_marker.touch()


def merge_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def organize_batches(train_dir: Path) -> None:
    batch_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir() and p.name.startswith("batch_"))
    if not batch_dirs:
        print("[prepare-mdr] no batch_* directories to organize")
        return

    for batch_dir in batch_dirs:
        print(f"[prepare-mdr] organizing {batch_dir}", flush=True)
        for datatype in DATATYPES:
            merge_tree(batch_dir / datatype, train_dir / datatype)
        shutil.rmtree(batch_dir)


def summarize(train_dir: Path) -> None:
    print("[prepare-mdr] final counts:")
    for datatype in DATATYPES:
        suffix = "*.flo" if datatype == "flow" else "*.npz"
        print(f"  {datatype}: {count_files(train_dir / datatype, suffix)}")


def is_organized(train_dir: Path) -> bool:
    for datatype in DATATYPES:
        suffix = "*.flo" if datatype == "flow" else "*.npz"
        marker_dir = train_dir / datatype
        if not marker_dir.exists() or next(marker_dir.rglob(suffix), None) is None:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--mdr-root", type=Path, default=DEFAULT_MDR_ROOT)
    parser.add_argument("--dt", type=int, default=1)
    parser.add_argument("--organize-only", action="store_true")
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    train_dir = (args.mdr_root.resolve() / f"dt{args.dt}" / "train")
    print(f"[prepare-mdr] archive_dir={archive_dir}")
    print(f"[prepare-mdr] train_dir={train_dir}")

    if is_organized(train_dir):
        print("[prepare-mdr] organized MDR tree already exists; skip archive extraction", flush=True)
        return 0

    if not args.organize_only:
        extract_archives(archive_dir, train_dir)
    organize_batches(train_dir)
    summarize(train_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
