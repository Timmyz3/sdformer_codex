#!/usr/bin/env python3
"""Convert MVSEC rosbag files into the HDF5 layout expected by MVSEC_encoder.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from importRosbag.importRosbag import importRosbag


def _events_to_hdf5(events: dict) -> np.ndarray:
    """MVSEC_encoder indexes events as a 2D [N, 4] array: x, y, ts, p."""
    x = np.asarray(events["x"], dtype=np.float32)
    y = np.asarray(events["y"], dtype=np.float32)
    ts = np.asarray(events["ts"], dtype=np.float64)
    pol = np.asarray(events["pol"], dtype=np.float32)
    return np.stack([x, y, ts, pol], axis=1)


def _image_event_inds(event_ts: np.ndarray, image_ts: np.ndarray) -> np.ndarray:
    """Map each image timestamp to the exclusive end index in the event stream."""
    event_ts = np.asarray(event_ts, dtype=np.float64)
    image_ts = np.asarray(image_ts, dtype=np.float64)
    return np.searchsorted(event_ts, image_ts, side="right").astype(np.int64)


def convert_data_bag(bag_path: Path, hdf5_path: Path) -> None:
    topics = importRosbag(str(bag_path), log="ERROR")
    events = topics["/davis/left/events"]
    images = topics["/davis/left/image_raw"]
    frames = np.stack(images["frames"]).astype(np.uint8)
    image_ts = np.asarray(images["ts"], dtype=np.float64)
    if "event_inds" in images:
        event_inds = np.asarray(images["event_inds"], dtype=np.int64)
    else:
        event_inds = _image_event_inds(events["ts"], image_ts)
        print(f"[data] derived image_raw_event_inds from timestamps ({len(event_inds)} frames)")

    with h5py.File(hdf5_path, "w") as handle:
        grp = handle.create_group("davis").create_group("left")
        grp.create_dataset("events", data=_events_to_hdf5(events), compression="gzip")
        grp.create_dataset("image_raw", data=frames, compression="gzip")
        grp.create_dataset("image_raw_ts", data=image_ts)
        grp.create_dataset("image_raw_event_inds", data=event_inds)
    print(f"[data] wrote {hdf5_path}")


def convert_gt_bag(bag_path: Path, hdf5_path: Path) -> None:
    topics = importRosbag(str(bag_path), log="ERROR")
    if "/davis/left/flow_dist" in topics:
        flow = topics["/davis/left/flow_dist"]
        flow_frames = np.stack(flow["frames"]).astype(np.float32)
        flow_ts = np.asarray(flow["ts"], dtype=np.float64)
    else:
        raise KeyError(
            "GT bag does not expose /davis/left/flow_dist; use official MVSEC gt.hdf5 instead."
        )

    with h5py.File(hdf5_path, "w") as handle:
        grp = handle.create_group("davis").create_group("left")
        grp.create_dataset("flow_dist", data=flow_frames, compression="gzip")
        grp.create_dataset("flow_dist_ts", data=flow_ts)
    print(f"[gt] wrote {hdf5_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--mvsec-root", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--gt-only", action="store_true")
    args = parser.parse_args()

    seq_dir = args.mvsec_root / args.sequence
    data_bag = seq_dir / f"{args.sequence}_data.bag"
    gt_bag = seq_dir / f"{args.sequence}_gt.bag"
    data_h5 = seq_dir / f"{args.sequence}_data.hdf5"
    gt_h5 = seq_dir / f"{args.sequence}_gt.hdf5"

    if not data_bag.exists():
        raise FileNotFoundError(data_bag)
    if not gt_bag.exists():
        raise FileNotFoundError(gt_bag)

    if not args.gt_only:
        convert_data_bag(data_bag, data_h5)
    if not args.data_only:
        convert_gt_bag(gt_bag, gt_h5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())