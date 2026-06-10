#!/usr/bin/env python3
"""Convert Spike-FlowNet gt_flow_dist.npz into MVSEC_encoder-compatible gt.hdf5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = np.load(args.npz)
    keys = list(payload.files)
    print(f"[npz] keys: {keys}")

    if "flow_dist" in keys and "flow_dist_ts" in keys:
        flow = np.asarray(payload["flow_dist"], dtype=np.float32)
        flow_ts = np.asarray(payload["flow_dist_ts"], dtype=np.float64)
    elif "gt_flow_dist" in keys and "gt_flow_dist_ts" in keys:
        flow = np.asarray(payload["gt_flow_dist"], dtype=np.float32)
        flow_ts = np.asarray(payload["gt_flow_dist_ts"], dtype=np.float64)
    else:
        raise KeyError(f"Unsupported npz schema: {keys}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        grp = handle.create_group("davis").create_group("left")
        grp.create_dataset("flow_dist", data=flow, compression="gzip")
        grp.create_dataset("flow_dist_ts", data=flow_ts)
    print(f"[gt] wrote {args.output} flow_dist={flow.shape} ts={flow_ts.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())