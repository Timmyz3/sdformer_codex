#!/usr/bin/env python3
"""Download raw MVSEC bags and encode MVSEC_test dt1 tensors for SDformerFlow."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "third_party" / "SDformerFlow"
MVSEC_ROOT = UPSTREAM / "data" / "Datasets" / "MVSEC"
BASE_URL = "http://visiondata.cis.upenn.edu/mvsec"


def download_sequence(sequence: str, force: bool = False) -> None:
    if sequence.startswith("indoor_flying"):
        scene = "indoor_flying"
    elif sequence.startswith("outdoor_day"):
        scene = "outdoor_day"
    else:
        scene = sequence.rsplit("_", 1)[0]
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


def encode_sequence(
    sequence: str,
    sparse_print: bool = True,
    index_start: int = 0,
    index_end: int | None = None,
    fast_flowgt: bool = False,
) -> None:
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
    if index_start:
        cmd.extend(["--index-start", str(index_start)])
    if index_end is not None:
        cmd.extend(["--index-end", str(index_end)])
    if fast_flowgt:
        cmd.append("--fast-flowgt")
    print("[encode] " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")
    subprocess.run(cmd, cwd=str(UPSTREAM), env=env, check=True)


def calibration_zip_for(sequence: str) -> Path:
    if sequence.startswith("indoor_flying"):
        return MVSEC_ROOT / "indoor_flying_calib.zip"
    if sequence.startswith("outdoor_day"):
        return MVSEC_ROOT / "outdoor_day_calib.zip"
    raise ValueError(f"No calibration zip mapping for MVSEC sequence: {sequence}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--index-end", type=int)
    parser.add_argument("--fast-flowgt", action="store_true")
    args = parser.parse_args()

    sequences = args.sequence or ["indoor_flying3"]

    if not args.encode_only:
        for sequence in sequences:
            download_sequence(sequence, force=args.force_download)
    if not args.download_only:
        for sequence in sequences:
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
                        converter = REPO_ROOT / "scripts" / "mvsec_bag_to_hdf5.py"
                        cmd = [
                            sys.executable,
                            str(converter),
                            "--sequence",
                            sequence,
                            "--mvsec-root",
                            str(MVSEC_ROOT),
                            "--gt-only",
                        ]
                        print("[convert-gt] " + " ".join(cmd))
                        try:
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError:
                            flow_script = REPO_ROOT / "scripts" / "mvsec_gt_flow_from_bag.py"
                            cmd = [
                                sys.executable,
                                str(flow_script),
                                "--sequence",
                                sequence,
                                "--mvsec-root",
                                str(MVSEC_ROOT),
                                "--calib-zip",
                                str(calibration_zip_for(sequence)),
                                "--output",
                                str(gt_h5),
                            ]
                            print("[generate-gt-flow] " + " ".join(cmd))
                            subprocess.run(cmd, check=True)
            encode_sequence(
                sequence,
                index_start=args.index_start,
                index_end=args.index_end,
                fast_flowgt=args.fast_flowgt,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
