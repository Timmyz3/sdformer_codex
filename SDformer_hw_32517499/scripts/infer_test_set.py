"""DSEC test set inference — generate flow predictions, no GT needed.

Usage:
    python scripts/infer_test_set.py \
        --checkpoint experiments/baseline_stride_upstream/MS_SpikingformerFlowNet_en4_best.pth \
        --output results_inference/test_baseline/

Creates dummy mask/label placeholders so DSECDatasetLite doesn't crash.
Saves flow predictions as .npy files + compiles a submission-ready zip.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "third_party/SDformerFlow"))
sys.path.insert(0, str(_REPO / "third_party/SDformerFlow/DSEC_dataloader"))


def create_dummy_test_files(data_root: str, sequences: list[str]) -> None:
    """Create zero-filled GT and mask placeholders for test sequences."""
    import h5py
    from DSEC_dataset_preprocess import EventSlicer

    for seq in sequences:
        # Count test windows from timestamps
        ts_path = os.path.join(
            data_root, "test_forward_optical_flow_timestamps", f"{seq}.csv"
        )
        if not os.path.exists(ts_path):
            continue
        timestamps = np.loadtxt(ts_path, delimiter=",", dtype="int64", skiprows=1)
        if timestamps.ndim == 1:
            timestamps = timestamps.reshape(1, -1)
        N = timestamps.shape[0]

        # Create dummy flow GT (zeros)
        gt_dir = os.path.join(data_root, "gt_tensors")
        os.makedirs(gt_dir, exist_ok=True)
        mask_dir = os.path.join(data_root, "mask_tensors")
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(N):
            fname = f"{seq}_{i:04d}.npy"
            gt_path = os.path.join(gt_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            if not os.path.exists(gt_path):
                np.save(gt_path, np.zeros((2, 480, 640), dtype=np.float32))
            if not os.path.exists(mask_path):
                np.save(mask_path, np.zeros((480, 640), dtype=np.bool_))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", help="Upstream YAML config path")
    parser.add_argument("--output", default="results_inference/test/", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # ── Config ──
    from configs.parser import YAMLParser

    config_path = args.config or str(
        _REPO / "configs/generated/upstream_baseline_stride.yml"
    )
    config_parser = YAMLParser(config_path)
    config = config_parser.config
    config = config_parser.combine_entries(config)
    device = torch.device(args.device)

    # ── Prepare test data ──
    data_root = config["data"]["path"]
    test_sequences = [
        "interlaken_00_b",
        "interlaken_01_a",
        "thun_01_a",
        "thun_01_b",
        "zurich_city_12_a",
        "zurich_city_14_c",
        "zurich_city_15_a",
    ]
    print("Creating dummy GT/mask placeholders for test set...")
    create_dummy_test_files(data_root, test_sequences)

    # Create test split CSV if not exists
    split_dir = os.path.join(data_root, "sequence_lists")
    test_csv = os.path.join(split_dir, "test_split_seq.csv")
    if not os.path.exists(test_csv):
        print(f"test_split_seq.csv not found at {test_csv}")
        sys.exit(1)

    # ── Model ──
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4 as ModelClass
    from spikingjelly.activation_based import functional, neuron
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    from utils.utils import load_model

    print("Loading model...")
    if config["swin_transformer"]["use_arc"][0]:
        model = ModelClass(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = ModelClass(config["model"].copy())
    model.to(device)
    model.init_weights()

    remap = config["loader"].get("remap")
    model = load_model(args.checkpoint, model, device, remap=remap, test=True)
    model.eval()

    neurontype = PSN
    configure_snn_backend(model, device, config, neurontype)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── DataLoader ──
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from DSEC_dataloader.data_augmentation import Compose, CenterCrop

    crop = config["loader"].get("crop")
    if crop:
        transform_valid = Compose([CenterCrop((crop[0], crop[1]))])
        config["swin_transformer"]["input_size"] = [crop[0], crop[1]]
    else:
        transform_valid = None

    # Patch config to use test split
    config["data"]["sequence_list_overrides"] = {"test": test_csv}

    test_dataset = DSECDatasetLite(config, file_list="test", stereo=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    # ── Inference ──
    os.makedirs(args.output, exist_ok=True)
    sample_idx = 0
    all_predictions = {}

    print("Running inference...")
    with torch.no_grad():
        for chunk, mask, label in tqdm(test_loader):
            functional.reset_net(model)
            functional.set_step_mode(model, config["data"]["step_mode"])

            chunk = chunk.to(device, dtype=torch.float32)

            if transform_valid is not None:
                # Transform expects (chunk, label, mask) but we only have chunk
                dummy_label = torch.zeros(
                    chunk.shape[0], 2, chunk.shape[-2], chunk.shape[-1],
                    device=device, dtype=torch.float32,
                )
                dummy_mask = torch.zeros(
                    chunk.shape[0], 1, chunk.shape[-2], chunk.shape[-1],
                    device=device, dtype=torch.float32,
                )
                chunk, _, _ = transform_valid((chunk, dummy_label, dummy_mask))

            # Polarity split
            if config["loader"].get("polarity", True):
                pos = torch.relu(chunk)
                neg = torch.relu(-chunk)
                chunk = torch.cat((pos.unsqueeze(2), neg.unsqueeze(2)), dim=2)

            # MinMax normalize
            if config["model"]["norm_input"] == "minmax":
                nonzero = chunk != 0
                if nonzero.any():
                    min_val, max_val = chunk[nonzero].min(), chunk[nonzero].max()
                    if min_val != max_val:
                        chunk[nonzero] = (chunk[nonzero] - min_val) / (max_val - min_val)

            pred_list = model(chunk.to(device))
            flow = pred_list["flow"][-1]  # [B, 2, H, W]

            for b in range(flow.shape[0]):
                flow_np = flow[b].cpu().numpy()
                fname = f"{sample_idx:06d}.npy"
                np.save(os.path.join(args.output, fname), flow_np)
                all_predictions[sample_idx] = fname
                sample_idx += 1

    print(f"\nDone. {sample_idx} predictions saved to {args.output}")
    print(f"To submit to DSEC benchmark, zip the .npy files and upload to:")
    print(f"  https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/")


if __name__ == "__main__":
    main()
