#!/usr/bin/env python3
"""Prepare SDformerFlow MDR training tree and print official download instructions."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MDR_ROOT = REPO_ROOT / "third_party" / "SDformerFlow" / "data" / "Datasets" / "MDR"

BAIDU_URL = "https://pan.baidu.com/s/1iSgGCjDask-M_QqPRtaLhA?pwd=z52j"
BAIDU_CODE = "z52j"
GDRIVE_CHECKPOINTS = (
    "https://drive.google.com/drive/folders/15uwhrmUzg3kK3UB6z0Qnht-sGs7Nq23o?usp=sharing"
)

REQUIRED_TRAIN_MARKERS = [
    MDR_ROOT / "dt1" / "train" / "events1",
    MDR_ROOT / "dt1" / "train" / "events2",
    MDR_ROOT / "dt1" / "train" / "best_density_events1",
    MDR_ROOT / "dt1" / "train" / "best_density_events2",
    MDR_ROOT / "dt1" / "train" / "flow",
]


def ensure_tree() -> None:
    for marker in REQUIRED_TRAIN_MARKERS:
        marker.mkdir(parents=True, exist_ok=True)


def is_ready() -> bool:
    events1 = MDR_ROOT / "dt1" / "train" / "events1"
    if not events1.exists():
        return False
    return any(events1.rglob("*.npz"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    ensure_tree()
    ready = is_ready()
    print(f"MDR root: {MDR_ROOT}")
    print(f"ready: {ready}")
    if ready:
        npz_count = len(list((MDR_ROOT / 'dt1' / 'train' / 'events1').rglob('*.npz')))
        print(f"train samples (events1 npz files): {npz_count}")
        return 0

    print("\nMDR training set is not on disk yet.")
    print("Official source (ADMFlow / ICCV 2023):")
    print(f"  URL : {BAIDU_URL}")
    print(f"  Code: {BAIDU_CODE}")
    print("\nGoogle Drive mirror (checkpoints only, not training data):")
    print(f"  {GDRIVE_CHECKPOINTS}")
    print("\nAfter download, unpack so the tree matches:")
    print("  third_party/SDformerFlow/data/Datasets/MDR/dt1/train/{events1,events2,best_density_events1,best_density_events2,flow}")
    print("\nThen organize batches with:")
    print("  cd third_party/SDformerFlow")
    print("  python MDR_dataloader/MDR_menage.py -dt 1")
    if args.check_only:
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())