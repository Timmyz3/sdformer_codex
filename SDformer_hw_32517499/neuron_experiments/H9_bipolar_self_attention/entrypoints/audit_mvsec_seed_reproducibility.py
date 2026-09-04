#!/usr/bin/env python3
"""Audit seeded direct-MVSEC shuffle and augmentation across worker pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = REPO_ROOT / "third_party/SDformerFlow"
sys.path.insert(0, str(UPSTREAM))

from MDR_dataloader.MVSEC import MvsecEventFlow  # noqa: E402


def capture_first_batch(config: dict, seed: int, workers: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = MvsecEventFlow(
        config,
        train=True,
        aug=True,
        manifest_role=config["data"].get("mvsec_train_split", "train"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["loader"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=False,
    )
    batch = next(iter(loader))
    digest = hashlib.sha256()
    for key in ("idx", "d_event_volume_new", "flow", "valid"):
        digest.update(batch[key].detach().cpu().numpy().tobytes())
    return {"indices": batch["idx"].tolist(), "batch_sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    seed = int(config["runtime"]["seed"])
    first = capture_first_batch(config, seed, args.workers)
    second = capture_first_batch(config, seed, args.workers)
    payload = {
        "schema": "mvsec_seed_reproducibility_audit_v1",
        "config": str(args.config.resolve()),
        "seed": seed,
        "workers": args.workers,
        "first": first,
        "second": second,
        "status": "PASS" if first == second else "FAIL",
        "scope": "data_order_and_augmentation_not_full_training_bit_exact",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
