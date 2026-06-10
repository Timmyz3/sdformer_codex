"""Correct per-layer SOPs profiler for SDFormerFlow sparsity metrics.

Computes:
  - total_spikes: raw spike count per layer (directly measured)
  - firing_rate: per-layer spike fraction (directly measured)
  - total_dense_flops: architecture-level dense FLOPs (analytical, per config)
  - effective_flops_per_layer: dense_flops_share × layer_firing_rate
  - total_effective_flops: sum of per-layer effective FLOPs
  - sparsity_ratio: 1 - effective/dense
  - SynOps: energy-equivalent synaptic operations (per SNN community standard)
  - Energy estimate: E_AC × SynOps_MAC + E_logic × SynOps_attn

Key metric: total_spikes is the most objective and cross-paper-comparable metric.

Usage:
  python -m tools.profile_sparsity \\
    --config <config>.yml --checkpoint <ckpt>.pth \\
    --split valid --num-samples 825
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# ── Architecture constants ─────────────────────────────────────────
ARCH = {
    "swin_depths": [2, 2, 6, 2],
    "swin_num_heads": [3, 6, 12, 24],
    "base_num_channels": 96,
    "window_size": (2, 9, 9),
    "mlp_ratio": 4,
    "num_steps": 10,
    "crop": (288, 384),
    "num_bins": 10,
    "num_resblocks": 2,
}

# ── 45nm CMOS energy constants ─────────────────────────────────────
E_MAC = 4.6e-12   # 4.6 pJ per multiply-accumulate
E_AC = 0.9e-12    # 0.9 pJ per accumulate (spike-driven, no multiply)
E_LOGIC = 0.1e-12  # ~0.1 pJ per bit operation (ternary attention popcount)


def _conv_flops(c_in, c_out, k, h, w, t=ARCH["num_steps"]):
    """2 × K² × C_in × C_out × H × W × T"""
    return float(2 * k * k * c_in * c_out * h * w * t)


def _linear_flops(c_in, c_out, tokens):
    """2 × C_in × C_out × N"""
    return float(2 * c_in * c_out * tokens)


def compute_total_dense_flops() -> float:
    """Compute total dense FLOPs for MS_SpikingformerFlowNet_en4 analytically.

    This is a single deterministic number based on the architecture config.
    Returns FLOPs for a single forward pass (all 10 timesteps).
    """
    total = 0.0
    depths = ARCH["swin_depths"]
    heads = ARCH["swin_num_heads"]
    C_base = ARCH["base_num_channels"]
    win = ARCH["window_size"]
    T = ARCH["num_steps"]
    H0, W0 = ARCH["crop"]

    # Patch Embed: stride-2 conv + residual blocks
    C, H, W = C_base, H0 // 2, W0 // 2
    total += _conv_flops(2, C, 1, H0, W0) * 2  # approximate as 2 convs

    for si in range(4):
        C = C_base * (2 ** si)
        H = H0 // (2 ** (si + 1))
        W = W0 // (2 ** (si + 1))
        tokens = T * H * W
        d_head = C // heads[si]
        tokens_win = win[0] * win[1] * win[2]
        nW = (T // win[0]) * (H // win[1]) * (W // win[2])

        for _ in range(depths[si]):
            # QKV projection: 3 × Linear(C, C)
            total += 3 * _linear_flops(C, C, tokens)
            # Attention matmul: Q@K^T + score@V
            total += 2 * (2 * tokens_win * tokens_win * C) * nW
            # Output projection: Linear(C, C)
            total += _linear_flops(C, C, tokens)
            # MLP: expand + project
            hidden = int(C * ARCH["mlp_ratio"])
            total += _linear_flops(C, hidden, tokens) + _linear_flops(hidden, C, tokens)

        # Downsample between stages (except last)
        if si < 3:
            C_next = C_base * (2 ** (si + 1))
            H_next, W_next = H0 // (2 ** (si + 2)), W0 // (2 ** (si + 2))
            total += _conv_flops(C, C_next, 2, H_next, W_next)

    # Resblocks at bottleneck
    C3, H3, W3 = C_base * 8, H0 // 16, W0 // 16
    total += _conv_flops(C3, C3, 3, H3, W3) * ARCH["num_resblocks"] * 2

    # Decoder: 4 stages of upsample + conv + concat skip
    for di in range(4):
        si = 3 - di
        C = C_base * (2 ** si)
        H_dec = H0 // (2 ** (si + 1))
        W_dec = W0 // (2 ** (si + 1))
        skip_C = C_base * (2 ** (si - 1)) if si > 0 else C_base
        in_ch = C + skip_C
        out_ch = skip_C if si > 0 else C_base
        total += _conv_flops(in_ch, out_ch, 3, H_dec, W_dec)

    # Prediction heads: conv to 2-channel flow
    for pi in range(4):
        si = 3 - pi
        C = C_base * (2 ** (si - 1)) if si > 0 else C_base
        H_pred = H0 // (2 ** (si + 1))
        W_pred = W0 // (2 ** (si + 1))
        total += _conv_flops(C, 2, 3, H_pred, W_pred)

    return total


# ── Spike profiler ─────────────────────────────────────────────────

def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_tensors(v)


class SpikeActivityProfiler:
    def __init__(self, model, patterns=("Spiking_neuron",)):
        self.model = model
        self.patterns = patterns
        self.handles = []
        self.records: Dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "spikes": 0, "elements": 0, "rate_sum": 0.0}
        )

    def attach(self):
        for name, module in self.model.named_modules():
            cls = module.__class__.__name__
            if any(p in cls or p in name for p in self.patterns):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _hook(self, name: str):
        def fn(_m, _in, out):
            tensors = list(_iter_tensors(out))
            if not tensors:
                return
            spikes = sum(int((t.detach() != 0).sum().item()) for t in tensors)
            elems = sum(int(t.detach().numel()) for t in tensors)
            if elems == 0:
                return
            r = self.records[name]
            r["calls"] += 1
            r["spikes"] += spikes
            r["elements"] += elems
            r["rate_sum"] += spikes / elems
        return fn


# ── Helper: determine op type from layer name ──────────────────────

def _layer_op_type(name: str) -> str:
    """Classify a spiking neuron layer as 'mac' or 'logic'."""
    if any(k in name for k in ("sn_q", "sn2_q", "sn_k", "attn_sn")):
        return "logic"
    return "mac"


# ── Main ───────────────────────────────────────────────────────────

def format_human(value: float) -> str:
    for suffix, scale in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.4f}{suffix}"
    return f"{value:.4f}"


class _MetricAccumulator:
    def __init__(self, metrics):
        self.metrics = metrics
        self.values = defaultdict(float)
        self.counts = defaultdict(int)

    def update_aee(self, values):
        for k, v in zip(["AEE", "AEE_PE1", "AEE_PE2", "AEE_PE3", "AEE_outliers"], values):
            detached = v.detach().cpu().float().reshape(-1)
            self.values[k] += float(detached.sum().item())
            self.counts[k] += int(detached.numel())

    def update_scalar(self, name, value, count=1):
        scalar = float(value.detach().cpu().item() if torch.is_tensor(value) else value)
        self.values[name] += scalar
        self.counts[name] += int(count)

    def summary(self):
        return {k: self.values[k] / self.counts[k] for k in self.counts if self.counts[k] > 0}


def _resolve_neuron_type(config):
    from spikingjelly.activation_based import neuron
    from models.STSwinNet_SNN.Spiking_submodules import GatedLIFNode, PSN, SLTTLIFNode
    nt = config["model"]["spiking_neuron"]["neuron_type"]
    mapping = {"if": neuron.IFNode, "lif": neuron.LIFNode, "plif": neuron.ParametricLIFNode,
               "glif": GatedLIFNode, "psn": PSN, "SLTTlif": SLTTLIFNode}
    return mapping.get(nt)


def run_profile(args) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))

    # H9 overlay: add overlay directory to sys.path for ATLIF/BSA attention modules
    config_path = Path(str(args.config))
    experiments_root = repo_root / "neuron_experiments"
    overlay = None
    try:
        rel = config_path.resolve().relative_to(experiments_root)
        overlay = experiments_root / rel.parts[0] / "overlay"
        if overlay.exists():
            sys.path.insert(0, str(overlay))
    except (ValueError, IndexError):
        pass  # config not under neuron_experiments
    if overlay is None:
        default_overlay = experiments_root / "H9_bipolar_self_attention" / "overlay"
        if default_overlay.exists():
            sys.path.insert(0, str(default_overlay))

    from configs.parser import YAMLParser
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from DSEC_dataloader.data_augmentation import CenterCrop, Compose
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from spikingjelly.activation_based import functional
    from torch.utils.data import DataLoader
    from utils.runtime_backend import configure_snn_backend
    from utils.utils import load_model
    from loss.flow_supervised import AAE, AEE

    old_cwd = os.getcwd()
    os.chdir(str(repo_root))
    parser = YAMLParser(str(args.config))
    config = YAMLParser.combine_entries(parser.config)
    os.chdir(old_cwd)

    # H9 configs use ../../data/... relative to the working directory (third_party/SDformerFlow).
    # Resolve to absolute before we chdir.
    data_path = config["data"]["path"]
    if not os.path.isabs(data_path):
        # Try resolving from baseline_root (the effective CWD for H9 training)
        resolved = os.path.normpath(os.path.join(str(baseline_root), data_path))
        if os.path.isdir(resolved):
            config["data"]["path"] = resolved
        else:
            # Fallback: resolve from repo_root
            resolved2 = os.path.normpath(os.path.join(str(repo_root), data_path))
            if os.path.isdir(resolved2):
                config["data"]["path"] = resolved2

    config["loader"]["batch_size"] = args.batch_size
    config["loader"]["n_workers"] = args.num_workers
    config.setdefault("runtime", {})
    config["runtime"]["snn_backend"] = args.snn_backend
    config["runtime"]["allow_tf32"] = True
    config["runtime"]["cudnn_benchmark"] = True

    if config["loader"]["crop"] is not None:
        crop = config["loader"]["crop"]
        transform = Compose([CenterCrop((crop[0], crop[1]))])
        config["swin_transformer"]["input_size"] = [crop[0], crop[1]]
    else:
        transform = None
        config["swin_transformer"]["input_size"] = [config["loader"]["resolution"][0], config["loader"]["resolution"][1]]

    device = parser.device if args.device is None else torch.device(args.device)

    os.chdir(str(baseline_root))
    dataset = DSECDatasetLite(config, file_list=args.split, stereo=False)
    os.chdir(old_cwd)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, pin_memory=True, num_workers=args.num_workers)

    model = MS_SpikingformerFlowNet_en4(config["model"].copy(), config["swin_transformer"].copy())
    model.to(device)
    model.init_weights()

    # H9 overlay: install ATLIF+BSA attention if the config has them
    if config.get("atlif_ternary_psn", {}).get("enabled") or config.get("bsa_attention", {}).get("enabled"):
        try:
            from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat
            register_shiftmax_pickle_compat()
        except Exception:
            pass
        try:
            from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn
            installed = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
            print(f"[H9] profile installed ATLIFTernaryPSN: {len(installed)} modules")
        except Exception as e:
            print(f"  [WARN] ATLIF install failed: {e}")
        try:
            from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention
            installed = install_shiftmax_attention(model, config.get("bsa_attention"))
            print(f"[H9] profile installed Shiftmax attention: {len(installed)} modules")
        except Exception as e:
            print(f"  [WARN] BSA attention install failed: {e}")

    if args.checkpoint:
        if Path(args.checkpoint).is_file():
            from models.STSwinNet_SNN.h9_load_audit import load_checkpoint_with_h9_audit

            model = load_checkpoint_with_h9_audit(
                str(args.checkpoint),
                model,
                device,
                config=config,
                test=True,
            )
        else:
            model = load_model(str(args.checkpoint), model, device, test=True)
    functional.reset_net(model)
    functional.set_step_mode(model, config["data"]["step_mode"])
    neuron_type = _resolve_neuron_type(config)
    configure_snn_backend(model, device, config, neuron_type)
    model.eval()

    profiler = SpikeActivityProfiler(model)
    profiler.attach()

    metric_accum = _MetricAccumulator(["AEE", "AAE"])
    num_seen = 0
    total_dense_flops = compute_total_dense_flops()

    with torch.no_grad():
        for chunk, mask, label in loader:
            functional.reset_net(model)
            chunk = chunk.to(device=device, dtype=torch.float32, non_blocking=True)
            label = label.to(device=device, dtype=torch.float32, non_blocking=True)
            mask = torch.unsqueeze(mask.to(device=device, non_blocking=True), dim=1)

            if transform is not None:
                chunk, label, mask = transform((chunk, label, mask.float()))

            if config["model"]["encoding"] == "voxel" and config["loader"]["polarity"]:
                pos = torch.nn.functional.relu(chunk)
                neg = torch.nn.functional.relu(-chunk)
                chunk = torch.cat((pos.unsqueeze(2), neg.unsqueeze(2)), dim=2)

            if config["model"]["norm_input"] == "minmax":
                non_zero = chunk != 0
                if non_zero.any():
                    mn, mx = chunk[non_zero].min(), chunk[non_zero].max()
                    if mn != mx:
                        chunk[non_zero] = (chunk[non_zero] - mn) / (mx - mn)

            pred_list = model(chunk.to(device))
            pred = pred_list["flow"][-1]
            metric_accum.update_aee(AEE(pred, label, mask, config["metrics"]["flow_scaling"])())
            metric_accum.update_scalar("AAE", AAE(pred, label, mask, config["metrics"]["flow_scaling"])()[0])
            num_seen += chunk.shape[0]
            if args.num_samples and num_seen >= args.num_samples:
                break

    profiler.close()

    # ── Build results ──────────────────────────────────────────────
    layers = []
    total_spikes = 0
    total_synops_mac = 0
    total_synops_logic = 0
    total_energy = 0.0
    total_effective_flops = 0.0

    for name, rec in sorted(profiler.records.items()):
        spikes = int(rec["spikes"])
        elements = int(rec["elements"])
        fr = spikes / elements if elements else 0.0
        total_spikes += spikes

        # Distribute dense FLOPs proportionally to element counts
        # (larger layers get more FLOPs share)
        flops_share = total_dense_flops * (elements / sum(r["elements"] for r in profiler.records.values()))
        effective = flops_share * fr
        total_effective_flops += effective

        op_type = _layer_op_type(name)
        if op_type == "logic":
            synops = spikes
            energy = synops * E_LOGIC
            total_synops_logic += synops
        else:
            synops = spikes
            energy = synops * E_AC
            total_synops_mac += synops
        total_energy += energy

        layers.append({
            "layer": name, "spikes": spikes, "elements": elements,
            "firing_rate": fr, "dense_flops_share": flops_share,
            "effective_flops": effective, "synops": synops,
            "energy_uj": energy * 1e6, "op_type": op_type,
        })

    global_fr = total_spikes / sum(r["elements"] for r in profiler.records.values()) if profiler.records else 0.0
    total_synops = total_synops_mac + total_synops_logic

    return {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "split": args.split, "samples": num_seen,
        "total_spikes": total_spikes,
        "total_spikes_human": format_human(total_spikes),
        "global_firing_rate": global_fr,
        "total_dense_flops": total_dense_flops,
        "total_dense_flops_human": format_human(total_dense_flops),
        "total_effective_flops": total_effective_flops,
        "total_effective_flops_human": format_human(total_effective_flops),
        "sparsity_ratio": 1.0 - (total_effective_flops / total_dense_flops if total_dense_flops else 1),
        "total_synops_mac": total_synops_mac,
        "total_synops_mac_human": format_human(total_synops_mac),
        "total_synops_logic": total_synops_logic,
        "total_synops_logic_human": format_human(total_synops_logic),
        "total_synops": total_synops,
        "total_synops_human": format_human(total_synops),
        "estimated_energy_uj": total_energy * 1e6,
        "profiled_layers": len(layers),
        "metrics": metric_accum.summary(),
        "layers": layers,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--snn-backend", default="torch")
    args = parser.parse_args(argv)

    args.config = Path(args.config).resolve()
    if args.checkpoint:
        args.checkpoint = Path(args.checkpoint).resolve()

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("experiments") / "sparsity_profiles" / stamp
    args.output_dir = Path(args.output_dir)

    result = run_profile(args)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in result.items() if k != "layers"}
    (out / "sparsity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

    with (out / "layer_sparsity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "spikes", "elements", "firing_rate",
                                           "dense_flops_share", "effective_flops",
                                           "synops", "energy_uj", "op_type"])
        w.writeheader()
        w.writerows(result["layers"])

    print(f"samples: {result['samples']}")
    print(f"total_spikes: {result['total_spikes_human']}")
    print(f"global_firing_rate: {result['global_firing_rate']:.4%}")
    print(f"total_dense_flops: {result['total_dense_flops_human']}")
    print(f"total_effective_flops: {result['total_effective_flops_human']}")
    print(f"sparsity_ratio: {result['sparsity_ratio']:.2%}")
    print(f"total_synops (MAC): {result['total_synops_mac_human']}")
    print(f"total_synops (logic): {result['total_synops_logic_human']}")
    print(f"estimated_energy: {result['estimated_energy_uj']:.2f} uJ")
    print(f"profiled_layers: {result['profiled_layers']}")
    if result.get("metrics"):
        print("metrics:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.6f}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
