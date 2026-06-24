import os
import sys
import torch
import torchvision  # must import before cupy to avoid circular import
import argparse
import json
from collections import defaultdict
from pathlib import Path
# mlflow stub: prevent ImportError when SDFORMER_USE_MLFLOW=0
_mlflow_disabled = os.getenv("SDFORMER_USE_MLFLOW", "1").lower() in {"0", "false", "no"}
if _mlflow_disabled:
    import types
    sys.modules["mlflow"] = types.ModuleType("mlflow")
import mlflow
from configs.parser import YAMLParser
from loss.flow_supervised import *

# ── H9 overlay support ─────────────────────────────────────────────
def _install_h9_overlay(config_path):
    """Add H9 overlay dir to sys.path so ATLIF/BSA modules can be imported."""
    try:
        # Search from config path and repo root
        for base in [Path(config_path).resolve().parent, Path(__file__).resolve().parents[3]]:
            for d in base.parents:
                if (d / "neuron_experiments").exists():
                    for h9_dir in sorted((d / "neuron_experiments").iterdir(), reverse=True):
                        overlay = h9_dir / "overlay"
                        if overlay.is_dir() and (overlay / "models").is_dir():
                            atlif = overlay / "models" / "STSwinNet_SNN" / "atlif_ternary_psn"
                            if atlif.is_dir():
                                sys.path.insert(0, str(overlay))
                                return str(overlay)
                    break
    except Exception:
        pass
    return None

# H9 configs auto-detect: if a config path contains overlay sections, install overlay BEFORE model imports
def _auto_install_h9_overlay():
    """Called at module import time to pre-install overlay from env or args."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            _install_h9_overlay(sys.argv[i + 1])
            break
_auto_install_h9_overlay()

from models.STSwinNet_SNN.Spiking_STSwinNet import SpikingformerFlowNet,MS_SpikingformerFlowNet, MS_SpikingformerFlowNet_en4

def _install_h9_modules(model, config):
    """Install ATLIF + BSA attention if the config has them enabled."""
    if config.get("atlif_ternary_psn", {}).get("enabled"):
        try:
            from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn
            installed = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
            print(f"[H9] eval installed ATLIFTernaryPSN: {len(installed)} modules")
        except Exception as e:
            print(f"  [WARN] ATLIF install failed: {e}")
    if config.get("bsa_attention", {}).get("enabled"):
        try:
            from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat
            register_shiftmax_pickle_compat()
        except Exception:
            pass
        try:
            from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention
            installed = install_shiftmax_attention(model, config.get("bsa_attention"))
            print(f"[H9] eval installed Shiftmax attention: {len(installed)} modules")
        except Exception as e:
            print(f"  [WARN] BSA attention install failed: {e}")
from tqdm import tqdm
from utils.mlflow import log_config, log_results
from utils.runtime_backend import configure_snn_backend
from utils.utils import load_model,  create_model_dir,save_csv, save_model, count_parameters,print_parameters
from utils.visualization import Visualization_DSEC
from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
from DSEC_dataloader.data_augmentation import downsample_data,Compose,CenterCrop,RandomCrop,RandomRotationFlip,Random_event_drop
import math
from spikingjelly.activation_based import functional,neuron
# from spikingjelly.activation_based import monitor
from models.STSwinNet_SNN.Spiking_submodules import *

import random

# ── Architecture constants for SOPs ─────────────────────────────────
_ARCH = {
    "swin_depths": [2, 2, 6, 2], "swin_num_heads": [3, 6, 12, 24],
    "base_num_channels": 96, "window_size": (2, 9, 9), "mlp_ratio": 4,
    "num_steps": 10, "num_resblocks": 2,
}

def _conv_flops(c_in, c_out, k, h, w, t=10):
    return float(2 * k * k * c_in * c_out * h * w * t)

def _linear_flops(c_in, c_out, tokens):
    return float(2 * c_in * c_out * tokens)

def _compute_total_dense_flops(crop_h, crop_w, config=None):
    """Architecture-level dense FLOPs, attention-type-aware.

    Baseline: float QK attention → full MAC
    TX (ternary_alpha_xnor): popcount-based → 0.1x attention MAC equivalent
    SC (signed_consensus): token-popcount → 0.15x attention MAC equivalent
    BSA (strict_bsa): float QK → same as baseline
    """
    attn_mult = 1.0  # default: float attention
    if config:
        bsa = config.get("bsa_attention", {})
        mode = bsa.get("mode", "")
        if mode.startswith("ternary_alpha_xnor"):
            attn_mult = 0.1   # ternary popcount ≈ 10% of float MAC
        elif mode.startswith("signed_consensus"):
            attn_mult = 0.15  # token popcount ≈ 15%
        elif mode.startswith("strict_bsa"):
            attn_mult = 1.0

    total = 0.0
    T, Cb, depths, heads = 10, 96, _ARCH["swin_depths"], _ARCH["swin_num_heads"]
    win = _ARCH["window_size"]
    H0, W0 = crop_h, crop_w

    # Patch Embed
    C, H, W = Cb, H0 // 2, W0 // 2
    total += _conv_flops(2, C, 3, H0, W0) * 2

    for si in range(4):
        C = Cb * (2 ** si)
        H, W = H0 // (2 ** (si + 1)), W0 // (2 ** (si + 1))
        tokens = T * H * W
        t_win = win[0] * win[1] * win[2]
        nW = (T // win[0]) * (H // win[1]) * (W // win[2])

        for _ in range(depths[si]):
            total += 3 * _linear_flops(C, C, tokens) * attn_mult  # QKV projection
            total += 2 * (2 * t_win * t_win * C) * nW * attn_mult  # attention matmul
            total += _linear_flops(C, C, tokens) * attn_mult  # output projection
            hidden = int(C * _ARCH["mlp_ratio"])
            total += _linear_flops(C, hidden, tokens) + _linear_flops(hidden, C, tokens)

        if si < 3:
            Cn = Cb * (2 ** (si + 1))
            Hn, Wn = H0 // (2 ** (si + 2)), W0 // (2 ** (si + 2))
            total += _conv_flops(C, Cn, 2, Hn, Wn)

    C3, H3, W3 = Cb * 8, H0 // 16, W0 // 16
    total += _conv_flops(C3, C3, 3, H3, W3) * _ARCH["num_resblocks"] * 2

    for di in range(4):
        si = 3 - di
        C = Cb * (2 ** si)
        Hd, Wd = H0 // (2 ** (si + 1)), W0 // (2 ** (si + 1))
        skip_C = Cb * (2 ** (si - 1)) if si > 0 else Cb
        total += _conv_flops(C + skip_C, skip_C if si > 0 else Cb, 3, Hd, Wd)

    for pi in range(4):
        si = 3 - pi
        Cp = Cb * (2 ** (si - 1)) if si > 0 else Cb
        Hp = H0 // (2 ** (si + 1))
        Wp = W0 // (2 ** (si + 1))
        total += _conv_flops(Cp, 2, 3, Hp, Wp)

    return total

# 45nm CMOS energy constants
E_AC = 0.9e-12
E_LOGIC = 0.1e-12

# ── Spike profiling helpers ─────────────────────────────────────────

def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_tensors(v)


class _SpikeProfiler:
    """Lightweight spike counter for eval pipeline."""
    def __init__(self):
        self.handles = []
        self.records = defaultdict(lambda: {"calls": 0, "spikes": 0, "elements": 0})

    def attach(self, model):
        for name, module in model.named_modules():
            cls = module.__class__.__name__
            if any(p in cls or p in name for p in ("Spiking_neuron",)):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _hook(self, name):
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
        return fn

    def summary(self):
        total_spikes = sum(r["spikes"] for r in self.records.values())
        total_elements = sum(r["elements"] for r in self.records.values())
        global_fr = total_spikes / total_elements if total_elements else 0.0
        return {
            "total_spikes": total_spikes,
            "total_elements": total_elements,
            "global_firing_rate": global_fr,
            "profiled_layers": len(self.records),
            "layer_firing_rates": {
                name: {"spikes": r["spikes"], "elements": r["elements"],
                       "firing_rate": r["spikes"] / r["elements"] if r["elements"] else 0}
                for name, r in sorted(self.records.items())
            }
        }

use_ml_flow = os.getenv("SDFORMER_USE_MLFLOW", "1").lower() not in {"0", "false", "no"}

def cal_firing_rate(s_seq: torch.Tensor):
    # s_seq.shape = [T, N, *]
    return s_seq.flatten(1).mean(1)

def valid_test(args, config_parser):
    ########## configs ##########
    config = config_parser.config
    path = 'output/' + str(args.runid) + '/'
    sequence = 'valid'
    if use_ml_flow:
        mlflow.set_tracking_uri(args.path_mlflow)
        run = mlflow.get_run(args.runid)
        config = config_parser.merge_configs(run.data.params)

        # create directory for inference results
        path_results = create_model_dir(args.path_results, args.runid)


        # store validation settings
        eval_id = log_config(path_results, args.runid, config)



    # initialize settings
    if not use_ml_flow:
        config = YAMLParser.combine_entries(config)
        path_results = args.path_results or "results_inference/"
        eval_id = "local"
    device = config_parser.device

    # ── H9 overlay ──
    _install_h9_overlay(args.config)

    # Eval requires batch_size=1 for correct metric computation
    config["loader"]["batch_size"] = 1

    # Fix relative data path (H9 configs use ../../data/... relative to baseline_root)
    if not os.path.isabs(config["data"]["path"]):
        baseline_root = os.path.dirname(os.path.abspath(__file__))
        resolved = os.path.normpath(os.path.join(baseline_root, config["data"]["path"]))
        if os.path.isdir(resolved):
            config["data"]["path"] = resolved
        else:
            repo_root = Path(__file__).resolve().parents[2]
            resolved = os.path.normpath(os.path.join(str(repo_root), config["data"]["path"]))
            if os.path.isdir(resolved):
                config["data"]["path"] = resolved
    loss_function = flow_loss_supervised(config, device)
    vis_cfg = config["vis"]
    vis_enabled = vis_cfg.get("enabled", False)
    vis_store = vis_cfg.get("store", False)
    vis_store_att = vis_cfg.get("store_att", False)
    vis_monitor_fr = vis_cfg.get("monitor_fr", False)
    vis_monitor_v = vis_cfg.get("monitor_v", False)
    # visualization tool
    if vis_enabled or vis_store or vis_store_att:
        vis = Visualization_DSEC(config, eval_id=eval_id, path_results=path_results)


    ########## data loader ##########
    if config["loader"]["crop"] is not None:
        transform_valid = Compose([CenterCrop((config['loader']['crop'][0], config['loader']['crop'][1])) ])
        config["swin_transformer"]["input_size"] = [config['loader']['crop'][0], config['loader']['crop'][1]]
    else:
        transform_valid = None
        config["swin_transformer"]["input_size"] = [config['loader']['resolution'][0], config['loader']['resolution'][1]]



    # Create validation dataset
    file = "valid"

    print("Creating Validation Dataset ...")

    valid_dataset = DSECDatasetLite(
        config,
        file_list=file,
        stereo=False,
        scale_factor=config.get('test', {}).get('scale_factor', 1)
    )



    # Define validation dataloader
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config["loader"]["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    # Transformer config
    if config["swin_transformer"]["use_arc"][0]:
        model = eval(config["model"]["name"])(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = eval(config["model"]["name"])(config["model"].copy())


    model.to(device)
    model.init_weights()

    # Install H9 overlay modules (ATLIF + BSA) before loading checkpoint
    _install_h9_modules(model, config)

    remap = config["loader"]["remap"] if "remap" in config["loader"] else None

    model_source = args.checkpoint if args.checkpoint else args.runid
    if args.checkpoint and os.path.isfile(args.checkpoint):
        try:
            from models.STSwinNet_SNN.h9_load_audit import load_checkpoint_with_h9_audit
        except Exception as exc:
            raise RuntimeError(
                "H9 load audit helper is unavailable. Check that the H9 overlay path was installed "
                "before evaluating a local checkpoint."
            ) from exc
        model = load_checkpoint_with_h9_audit(
            model_source,
            model,
            device,
            config=config,
            remap=remap,
            test=True,
        )
    else:
        model = load_model(model_source, model, device, remap = remap, test = True) # delete the relative positioning bias and index

    functional.reset_net(model)
    functional.set_step_mode(model, config['data']['step_mode'])

    if config["model"]["spiking_neuron"]["neuron_type"] == "if":
        neurontype = getattr(neuron, "IFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "lif":
        neurontype = getattr(neuron, "LIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "plif":
        neurontype = getattr(neuron, "ParametricLIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "glif":
        neurontype = GatedLIFNode
    elif config["model"]["spiking_neuron"]["neuron_type"] == "psn":
        neurontype = PSN
    elif config["model"]["spiking_neuron"]["neuron_type"] == "SLTTlif":
        neurontype = SLTTLIFNode
    else:
        raise "neurontype not implemented!"
    configure_snn_backend(model, device, config, neurontype)
    print(model)
    print_parameters(model)


    # Validation Dataset
    model.eval()

    # Attach spike profiler for sparsity metrics
    spike_profiler = _SpikeProfiler()
    spike_profiler.attach(model)

    print('Validating... (test sequence)')
    sample = 0
    val_results = {}
    for metric in config["metrics"]["name"]:
        val_results[metric] = {}
        val_results[metric]["metric"] = 0
        val_results[metric]["it"] = 0
        if metric == "AEE":
            val_results[metric]["PE1"] = 0
            val_results[metric]["PE2"] = 0
            val_results[metric]["PE3"] = 0
            val_results[metric]["outliers"] = 0

    if vis_monitor_fr:
        # spike_seq_monitor = monitor.OutputMonitor(model, neurontype)
        fr_monitor = monitor.OutputMonitor(model, neurontype, cal_firing_rate)
        fr_monitor.enable()

    if vis_monitor_v:
        for m in model.modules():
            if isinstance(m, neurontype):
                m.store_v_seq = True
        v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=model, instance=neurontype)



    max_samples = int(getattr(args, "max_samples", 0) or 0)
    for chunk, mask, label in tqdm(valid_dataloader):
        if max_samples > 0 and sample >= max_samples:
            break

        functional.reset_net(model)


        chunk = chunk.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)  # [num_batches, 2, H, W]
        mask = mask.to(device=device)
        mask = torch.unsqueeze(mask, dim=1)


        if transform_valid is not None:
            chunk, label, mask = transform_valid((chunk, label, mask.float()))
        with torch.no_grad():
            # forward pass
            if config['model']['encoding'] == 'cnt':
                if vis_enabled or vis_store_att:
                    chunk_vis = torch.sum(chunk, dim=1).detach()
                if config["swin_transformer"]["use_arc"][1] =="PatchEmbed3D":  #B D,P,H,W  -B P D H W
                    chunk = torch.transpose(chunk, 1, 2)
                else:
                    if config['loader']['polarity']:
                        chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))#for EV-FlowNet  #[B,40,H,W] 2+2+2....



            elif config['model']['encoding'] == 'voxel': #B, C, P, H, W
                if config['loader']['polarity']:
                    # ignore polarity
                    neg = torch.nn.functional.relu(-chunk)
                    pos = torch.nn.functional.relu(chunk)
                    # chunk = torch.abs(chunk)
                    chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)),
                                      dim=2)  # B,C=20,P=2,H,W   B C, P, H, W
                    if vis_enabled:
                        chunk_vis = torch.stack((torch.sum(pos, dim=1), torch.sum(neg, dim=1)), dim=1)
                else:
                    if vis_enabled:
                        chunk_vis = torch.sum(chunk, dim=1).detach()


            else:
                print("Config error: Event encoding not support.")
                raise AttributeError

            # normalize input
            if config["model"]["norm_input"]== "minmax":
                min, max = (
                    torch.min(chunk[chunk != 0]),
                    torch.max(chunk[chunk != 0]),
                )
                if not min == max:
                    chunk[chunk != 0] = (chunk[chunk != 0] - min) / (max-min)
            elif config["model"]["norm_input"]== "std":
                mean, stddev = (
                    chunk[chunk != 0].mean(),
                    chunk[chunk != 0].std(),
                )
                if stddev > 0:
                    chunk[chunk != 0] = (chunk[chunk != 0] - mean) / stddev

            # spike input
            if config['data']['spike_th'] is not None:
                chunk[chunk > config['data']['spike_th']] = 1
                chunk[chunk < config['data']['spike_th']] = 0

            pred_list = model(chunk.to(device))
            pred = pred_list["flow"][-1]

            # print(f'spike_seq_monitor.records=\n{spike_seq_monitor.records}')
            if vis_monitor_fr:
                fire_rate_mean = [torch.mean(rate) for rate in fr_monitor.records]
                print(f'firing rate=\n{torch.mean(torch.stack(fire_rate_mean))}')
                save_csv(fr_monitor.records, "firing_rate.csv")
                fr_monitor.records =[]
            if vis_monitor_v:
                print(f'v_seq=\n{v_seq_monitor.records}')
                v_seq_monitor.records =[]



        if config['metrics']['mask_events']:
            # event_mask = torch.sum(chunk, dim=1).bool()
            event_mask = torch.sum(torch.sum(chunk, dim=1), dim=1, keepdim=True).bool()
            mask = mask * event_mask


        total_loss = loss_function([pred], label, mask)
        print(total_loss)




        if vis_enabled or vis_store_att or (vis_store and config["loader"]["batch_size"] == 1):
            flow_vis = pred.clone()
            # flow_vis *= mask


        # validation
        # validation metric
        criteria = []
        if "metrics" in config.keys():
            for metric in config["metrics"]["name"]:
                criteria.append(eval(metric)(pred, label, mask, config['metrics']['flow_scaling']))
        for i, metric in enumerate(config["metrics"]["name"]):
                # compute metric
                val_metric = criteria[i]()

                # # accumulate results
                for batch in range(config["loader"]["batch_size"]):
                    val_results[metric]["it"] += 1
                    if metric == "AEE":
                        val_results[metric]["metric"] += val_metric[0][batch].cpu().numpy()
                        val_results[metric]["PE1"]  += val_metric[1][batch].cpu().numpy()
                        val_results[metric]["PE2"]  += val_metric[2][batch].cpu().numpy()
                        val_results[metric]["PE3"]  += val_metric[3][batch].cpu().numpy()
                        val_results[metric]["outliers"]  += val_metric[4][batch].cpu().numpy()
                    else:
                        val_results[metric]["metric"] += val_metric[batch].cpu().numpy()



        with torch.no_grad():
            if vis_enabled and config["loader"]["batch_size"] == 1:
                vis.update(chunk_vis, label, mask, flow_vis, None)
            if vis_store:
                sequence = sequence
                vis.store(chunk_vis, label, mask, flow_vis, sequence, None)
        sample += 1


    results = {}
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            results[metric] = {}
            if metric == "AEE":
                results[metric + "_PE1"] = {}
                results[metric + "_PE2"] = {}
                results[metric + "_PE3"] = {}
                results[metric + "_outliers"] = {}

            results[metric] = str(val_results[metric]["metric"] / val_results[metric]["it"])
            if metric == "AEE":
                results[metric + "_PE1"] = str(
                    val_results[metric]["PE1"] / val_results[metric]["it"]
                )
                results[metric + "_PE2"] = str(
                    val_results[metric]["PE2"] / val_results[metric]["it"]
                )
                results[metric + "_PE3"] = str(
                    val_results[metric]["PE3"] / val_results[metric]["it"]
                )
                results[metric + "_outliers"] = str(
                    val_results[metric]["outliers"] / val_results[metric]["it"]
                )


            if use_ml_flow:
                log_results(args.runid, results, path_results, eval_id)

            print(results[metric], results[ "AEE_PE1"], results["AEE_PE2"], results["AEE_PE3"], results["AEE_outliers"] )

    # ── Spike & SOPs profiling summary ─────────────────────────────
    spike_profiler.close()
    sp = spike_profiler.summary()

    # SOPs computation
    crop = config["loader"]["crop"]
    dense_flops = _compute_total_dense_flops(crop[0], crop[1], config)
    total_elements = sp["total_elements"]
    effective_flops = 0.0
    synops_mac = 0.0
    synops_logic = 0.0
    energy_j = 0.0

    for name, r in sp["layer_firing_rates"].items():
        # Per-layer dense FLOPs proportional to element count
        flops_share = dense_flops * (r["elements"] / total_elements) if total_elements else 0
        effective_flops += flops_share * r["firing_rate"]

        is_attn = any(k in name for k in ("sn_q", "sn2_q", "sn_k", "attn_sn"))
        if is_attn:
            synops_logic += r["spikes"]
            energy_j += r["spikes"] * E_LOGIC
        else:
            synops_mac += r["spikes"]
            energy_j += r["spikes"] * E_AC

    def _fmt(v): return f"{v/1e9:.4f}G" if v >= 1e9 else f"{v/1e6:.2f}M"

    print(f"\n[SPARSITY] total_spikes={_fmt(sp['total_spikes'])}  "
          f"global_fr={sp['global_firing_rate']:.2%}  "
          f"dense_flops={_fmt(dense_flops)}")
    print(f"[SPARSITY] effective_flops={_fmt(effective_flops)}  "
          f"sparsity={1-effective_flops/dense_flops:.1%}  "
          f"synops_mac={_fmt(synops_mac)}  synops_logic={_fmt(synops_logic)}")
    print(f"[SPARSITY] energy={energy_j*1e6:.2f}uJ  "
          f"(MAC: {synops_mac*E_AC*1e6:.2f}uJ + logic: {synops_logic*E_LOGIC*1e6:.2f}uJ)")

    # Save full profile
    if not use_ml_flow and args.checkpoint:
        profile = {
            "metrics": results,
            "samples": next(iter(val_results.values()))["it"] if val_results else 0,
            "total_spikes": sp["total_spikes"],
            "global_firing_rate": sp["global_firing_rate"],
            "dense_flops": dense_flops,
            "effective_flops": effective_flops,
            "sparsity_ratio": 1 - effective_flops / dense_flops if dense_flops else 0,
            "synops_mac": synops_mac, "synops_logic": synops_logic,
            "synops_total": synops_mac + synops_logic,
            "energy_uj": energy_j * 1e6,
            "profiled_layers": sp["profiled_layers"],
            "layer_firing_rates": sp["layer_firing_rates"],
        }
        spike_path = os.path.join(path_results, "spike_profile.json")
        os.makedirs(path_results, exist_ok=True)
        with open(spike_path, "w") as f:
            json.dump(profile, f, indent=2, default=str)
        print(f"[SPARSITY] profile saved to {spike_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/valid_DSEC_supervised.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )

    parser.add_argument("--runid", default="4b3da75cc15e44da80b84c3fb35ad618", help="mlflow run")
    parser.add_argument("--checkpoint", default="", help="optional local checkpoint path to evaluate")
    parser.add_argument(
        "--save_path",
        default="results/checkpoint_epoch{}.pth",
        help="save the model",
    )
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument("--mode", default="valid")
    parser.add_argument("--max-samples", type=int, default=0, help="optional cap for quick valid smoke tests")

    args = parser.parse_args()

    # launch test
    if args.mode == "valid":
        valid_test(args, YAMLParser(args.config))
