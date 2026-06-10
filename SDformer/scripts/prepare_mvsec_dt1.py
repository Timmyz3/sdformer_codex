#!/usr/bin/env python3
"""Download raw MVSEC bags and encode MVSEC_test dt1 tensors for SDformerFlow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "third_party" / "SDformerFlow"
MVSEC_ROOT = UPSTREAM / "data" / "Datasets" / "MVSEC"
BASE_URL = "http://visiondata.cis.upenn.edu/mvsec"


def download_sequence(sequence: str, force: bool = False) -> None:
    scene = "indoor_flying" if sequence.startswith("indoor_flying") else sequence.rsplit("_", 1)[0]
    seq_dir = MVSEC_ROOT / sequence
    seq_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("data", "gt"):
        bag = seq_dir / f"{sequence}_{suffix}.bag"
        if bag.exists() and not force:
            print(f"[skip] {bag}")
            continue
        url = f"{BASE_URL}/{scene}/{sequence}_{suffix}.bag"
        print(f"[download] {url} -> {bag}")
        subprocess.run(["wget", "-c", "-O", str(bag), url], check=True)


def convert_bags_to_hdf5(sequence: str) -> None:
    converter = REPO_ROOT / "scripts" / "mvsec_bag_to_hdf5.py"
    cmd = [
        sys.executable,
        str(converter),
        "--sequence",
        sequence,
        "--mvsec-root",
        str(MVSEC_ROOT),
    ]
    print("[convert] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def encode_sequence(sequence: str, sparse_print: bool = True) -> None:
    encoder = UPSTREAM / "MDR_dataloader" / "MVSEC_encoder.py"
    cmd = [
        sys.executable,
        str(encoder),
        "--save-dir",
        str(MVSEC_ROOT),
        "--out-dir",
        "data/Datasets/MVSEC/MVSEC_test",
        "--save-env",
        sequence,
        "--dt",
        "1",
    ]
    if sparse_print:
        cmd.append("--sparse_print")
    print("[encode] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(UPSTREAM), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", default=["indoor_flying3"])
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    if not args.encode_only:
        for sequence in args.sequence:
            download_sequence(sequence, force=args.force_download)
    if not args.download_only:
        for sequence in args.sequence:
            data_h5 = MVSEC_ROOT / sequence / f"{sequence}_data.hdf5"
            gt_h5 = MVSEC_ROOT / sequence / f"{sequence}_gt.hdf5"
            data_ready = data_h5.exists() and data_h5.stat().st_size > 0
            gt_ready = gt_h5.exists() and gt_h5.stat().st_size > 0
            if not data_ready or not gt_ready:
                if not data_ready:
                    converter = REPO_ROOT / "scripts" / "mvsec_bag_to_hdf5.py"
                    cmd = [
                        sys.executable,
                        str(converter),
                        "--sequence",
                        sequence,
                        "--mvsec-root",
                        str(MVSEC_ROOT),
                        "--data-only",
                    ]
                    print("[convert-data] " + " ".join(cmd))
                    subprocess.run(cmd, check=True)
                if not gt_ready:
                    npz_gt = MVSEC_ROOT / sequence / f"{sequence}_gt_flow_dist.npz"
                    if npz_gt.exists():
                        npz_script = REPO_ROOT / "scripts" / "mvsec_npz_to_gt_hdf5.py"
                        cmd = [
                            sys.executable,
                            str(npz_script),
                            "--npz",
                            str(npz_gt),
                            "--output",
                            str(gt_h5),
                        ]
                        print("[convert-gt-npz] " + " ".join(cmd))
                        subprocess.run(cmd, check=True)
                    else:
                        convert_bags_to_hdf5(sequence)
            encode_sequence(sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())