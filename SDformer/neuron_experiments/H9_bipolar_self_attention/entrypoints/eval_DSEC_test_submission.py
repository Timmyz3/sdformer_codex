#!/usr/bin/env python3
"""DSEC official test-set inference + submission writer (mode=test).

Closes gap P2-6 (CLAUDE_DATE_EXPERIMENT_GAPS_20260818.md): the only remaining
path to close the official AEE gap (local three-way aggregation cannot close it,
NB0_AAE_GAP_CLOSURE_20260812). Verified on 2026-08-18 that the official test
data is FULLY local (no registration download needed):
  - sequence_lists/test_split_seq.csv (416 samples, 7 sequences)
  - event_tensors/10bins/left/<seq>/<seq>_<idx>.npy (all 416, preprocessed)
  - test_forward_optical_flow_timestamps/<seq>.csv (official file_index per row)

Protocol: identical to the valid deploy path (eval_DSEC_flow_SNN.py --mode
valid, batch=1, bn_policy=no_running) except no GT. Model/config are the DATE
mainline deploy artifacts (default: H67 ep35 hardware_order q7q17 deploy).

Output (DSEC submission format, mirroring the official PNG-FI encoding used by
the train GT: v = round(flow*128 + 2^15), 3-channel uint16 x/y/mask):
  <output>/<seq>/<file_index:06d>.png

Does NOT modify eval_DSEC_flow_SNN.py, the overlay, or any frozen artifact.
Evidence tier: [模型] (no GT locally; official server evaluation is external).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision  # noqa: F401
import yaml

SD_FORMER_FLOW = Path("/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow")
sys.path.insert(0, str(SD_FORMER_FLOW))
sys.path.insert(0, str(SD_FORMER_FLOW.parents[1]))

# Overlay root: must be the frozen 66d0a339 bsa_attention.py. Since the disk
# overlay was modified by the D1 agent (2026-08-18 18:31Z, SHA a8e94f56...),
# the default is the shadow tree; pass --overlay-root to override.
FROZEN_OPERATOR_SHA = "66d0a339fec374537ef21f81ee0689d000ec1a4340a7821e49116604510fb483"
_OVERLAY_ROOT = Path(
    os.environ.get(
        "SDFORMER_OVERLAY_ROOT",
        "/tmp/p1_4_shadow/neuron_experiments/H9_bipolar_self_attention/overlay",
    )
)
_operator_path = _OVERLAY_ROOT / "models/STSwinNet_SNN/bsa_attention.py"
if not _operator_path.exists():
    raise SystemExit(f"[FATAL] overlay root not found: {_OVERLAY_ROOT}")
_sha = hashlib.sha256(_operator_path.read_bytes()).hexdigest()
if _sha != FROZEN_OPERATOR_SHA:
    raise SystemExit(f"[FATAL] overlay bsa_attention.py SHA mismatch: {_sha[:16]}... (frozen {FROZEN_OPERATOR_SHA[:16]}...)")
sys.path.insert(0, str(_OVERLAY_ROOT))

os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")

from configs.parser import YAMLParser  # noqa: E402
from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite  # noqa: E402
from DSEC_dataloader.data_augmentation import CenterCrop, Compose  # noqa: E402
from eval_DSEC_flow_SNN import (  # noqa: E402
    _configure_batch_norm_evaluation,
    _install_h9_modules,
    _install_h9_overlay,
)
from models.STSwinNet_SNN.Spiking_STSwinNet import (  # noqa: E402
    MS_SpikingformerFlowNet,
    MS_SpikingformerFlowNet_en4,
    SpikingformerFlowNet,
)
from models.STSwinNet_SNN.h9_load_audit import load_checkpoint_with_h9_audit  # noqa: E402
from models.STSwinNet_SNN.Spiking_submodules import GatedLIFNode, PSN, SLTTLIFNode  # noqa: E402
from utils.runtime_backend import configure_snn_backend  # noqa: E402
from spikingjelly.activation_based import functional, neuron  # noqa: E402


class DSECTestDatasetLite(DSECDatasetLite):
    """Test split loader: same file list / event loading as valid, no GT."""

    def __init__(self, config, file_list="test", stereo=False, scale_factor=1):
        # avoid triggering GT-dependent init; reuse parent for file list only
        self.config = config
        self.stereo = stereo
        self.scale_factor = scale_factor
        self.input = self.config["model"]["encoding"]
        self.num_frames_per_ts = config["data"]["num_frames"]
        self.num_chunks = config["data"]["num_chunks"]
        self.num_bins = self.num_frames_per_ts * self.num_chunks
        self.new_sequence = True
        self.events_path = os.path.join(
            self.config["data"]["path"], "event_tensors",
            "{}bins".format(str(self.num_frames_per_ts).zfill(2)), "left",
        )
        split_overrides = self.config["data"].get("sequence_list_overrides", {})
        sequence_file = None
        override_name = split_overrides.get(file_list)
        if override_name:
            sequence_file = os.path.join(self.config["data"]["path"], "sequence_lists", override_name)
        if sequence_file is None:
            file_list = file_list + "_split_seq.csv"
            sequence_file = os.path.join(self.config["data"]["path"], "sequence_lists", file_list)
        self.sequence_file = os.path.abspath(sequence_file)
        rows = []
        with open(self.sequence_file, newline="") as fh:
            for row in csv.reader(fh):
                if row:
                    rows.append(row)
        self.files = rows

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        target_file_1 = self.files[idx][0]
        seq_folder = "_".join(target_file_1.split("_")[:-1])
        chunk = torch.from_numpy(
            np.load(os.path.join(self.events_path, seq_folder, target_file_1), allow_pickle=True)
        )
        return chunk, target_file_1


def _build_model(config, checkpoint_path, device):
    if config["swin_transformer"]["use_arc"][0]:
        model = eval(config["model"]["name"])(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = eval(config["model"]["name"])(config["model"].copy())
    model.to(device)
    model.init_weights()
    _install_h9_modules(model, config)
    remap = config["loader"].get("remap")
    model = load_checkpoint_with_h9_audit(
        str(checkpoint_path), model, device, config=config, remap=remap, test=True,
    )
    return model


def _load_config(config_path):
    parser = YAMLParser(str(config_path))
    config = YAMLParser.combine_entries(parser.config)
    if not os.path.isabs(config["data"]["path"]):
        baseline_path = os.path.normpath(os.path.join(str(SD_FORMER_FLOW), config["data"]["path"]))
        if os.path.exists(baseline_path):
            config["data"]["path"] = baseline_path
    config["loader"]["batch_size"] = 1
    config["loader"]["shuffle"] = False
    config["loader"]["pin_memory"] = False
    config["loader"]["num_workers"] = 0
    if config["loader"].get("crop") is not None:
        config["swin_transformer"]["input_size"] = [
            int(config["loader"]["crop"][0]), int(config["loader"]["crop"][1]),
        ]
    else:
        config["swin_transformer"]["input_size"] = [
            config["loader"]["resolution"][0], config["loader"]["resolution"][1],
        ]
    return config


def _timestamps_by_seq(data_root):
    ts_root = os.path.join(data_root, "test_forward_optical_flow_timestamps")
    out = {}
    for path in sorted(Path(ts_root).glob("*.csv")):
        seq = path.stem
        with path.open() as fh:
            rows = [r for r in csv.reader(fh) if r and not r[0].startswith("#")]
        out[seq] = [int(r[2]) for r in rows]
    return out


def _save_png_fi(path: Path, flow_x: np.ndarray, flow_y: np.ndarray):
    img = np.stack([
        np.clip(np.round(flow_x * 128.0 + 32768.0), 0, 65535),
        np.clip(np.round(flow_y * 128.0 + 32768.0), 0, 65535),
        np.full(flow_x.shape, 65535, dtype=np.uint16),
    ], axis=2).astype(np.uint16)
    import cv2
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"cv2 failed to write {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bn-policy", default=None)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "status.log"

    def record(msg):
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    config_path = Path(args.config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    record(f"config sha: {hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]}...")
    record(f"checkpoint sha: {hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16]}...")

    config = _load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DSECTestDatasetLite(config, file_list="test")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        pin_memory=False, num_workers=args.num_workers,
    )
    record(f"test samples: {len(dataset)}")

    transform_valid = None
    if config["loader"].get("crop") is not None:
        transform_valid = Compose([CenterCrop((config["loader"]["crop"][0], config["loader"]["crop"][1]))])

    model = _build_model(config, ckpt_path, device)
    if args.bn_policy is not None:
        bn_policy = args.bn_policy
    else:
        bn_policy = config.get("test", {}).get("bn_policy", "no_running")
    _configure_batch_norm_evaluation(model, bn_policy)
    model.eval()

    functional.reset_net(model)
    functional.set_step_mode(model, config["data"]["step_mode"])
    ntype = config["model"]["spiking_neuron"]["neuron_type"]
    if ntype == "if":
        neurontype = getattr(neuron, "IFNode")
    elif ntype == "lif":
        neurontype = getattr(neuron, "LIFNode")
    elif ntype == "plif":
        neurontype = getattr(neuron, "ParametricLIFNode")
    elif ntype == "glif":
        neurontype = GatedLIFNode
    elif ntype == "psn":
        neurontype = PSN
    elif ntype == "SLTTlif":
        neurontype = SLTTLIFNode
    else:
        raise RuntimeError(f"neuron type not implemented: {ntype}")
    configure_snn_backend(model, device, config, neurontype)

    ts_by_seq = _timestamps_by_seq(config["data"]["path"])
    polarity = bool(config["loader"].get("polarity", False))
    sample_count = 0
    written = 0
    manifest = []
    for chunk, target_file in loader:
        if args.max_samples > 0 and sample_count >= args.max_samples:
            break
        sample_count += 1
        seq = "_".join(target_file[0].split("_")[:-1])
        functional.reset_net(model)
        chunk = chunk.to(device=device, dtype=torch.float32)
        if transform_valid is not None:
            chunk, _ = transform_valid((chunk, torch.zeros_like(chunk), torch.zeros_like(chunk[:1, 0:1])))
            chunk = chunk[0]
        if config["model"]["encoding"] == "voxel":
            if polarity:
                neg = torch.nn.functional.relu(-chunk)
                pos = torch.nn.functional.relu(chunk)
                chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)), dim=2)
        if config["model"].get("norm_input") == "minmax":
            lo, hi = torch.min(chunk[chunk != 0]), torch.max(chunk[chunk != 0])
            if not torch.equal(lo, hi):
                chunk[chunk != 0] = (chunk[chunk != 0] - lo) / (hi - lo)
        elif config["model"].get("norm_input") == "std":
            mean, stddev = chunk[chunk != 0].mean(), chunk[chunk != 0].std()
            if stddev > 0:
                chunk[chunk != 0] = (chunk[chunk != 0] - mean) / stddev
        if config["data"].get("spike_th") is not None:
            chunk[chunk > config["data"]["spike_th"]] = 1
            chunk[chunk < config["data"]["spike_th"]] = 0
        with torch.no_grad():
            pred_list = model(chunk)
            pred = pred_list["flow"][-1][0]  # [2, H, W]
        flow_x = pred[0].detach().cpu().numpy()
        flow_y = pred[1].detach().cpu().numpy()
        ts_list = ts_by_seq.get(seq, [])
        if sample_count - 1 < len(ts_list):
            file_index = ts_list[sample_count - 1]
        else:
            file_index = sample_count - 1
        seq_dir = out / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        png_path = seq_dir / f"{file_index:06d}.png"
        _save_png_fi(png_path, flow_x, flow_y)
        written += 1
        manifest.append({"seq": seq, "sample": target_file[0], "file_index": file_index,
                         "png": str(png_path)})
        if sample_count % 50 == 0:
            record(f"processed {sample_count}/{min(len(dataset), args.max_samples) if args.max_samples else len(dataset)}")

    summary = {
        "schema": "dsec_test_submission_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "samples_processed": sample_count,
        "png_written": written,
        "bn_policy": bn_policy,
        "format": "PNG-FI 3ch uint16, v=round(flow*128+2^15), name=<file_index:06d>.png per seq dir",
        "output_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    record(f"[DONE] samples={sample_count} png={written}; summary={out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
