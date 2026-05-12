"""Profile spike activity and estimated SOPs for SDFormerFlow inference.

The script is intentionally kept outside third_party/SDformerFlow. It can run
against the baseline or an experiment overlay by placing the overlay directory
before the baseline on sys.path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import Any, Iterable

import torch


DEFAULT_DENSE_OPS = "42.63G"


def parse_human_number(value: str | float | int) -> float:
    if isinstance(value, Number):
        return float(value)
    text = value.strip()
    if not text:
        raise ValueError("empty numeric value")
    suffixes = {
        "k": 1e3,
        "m": 1e6,
        "g": 1e9,
        "t": 1e12,
        "p": 1e15,
    }
    suffix = text[-1].lower()
    if suffix in suffixes:
        return float(text[:-1]) * suffixes[suffix]
    return float(text)


def format_human_number(value: float) -> str:
    for suffix, scale in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.4f}{suffix}"
    return f"{value:.4f}"


def flatten_numeric_tree(value: Any) -> float:
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, dict):
        return sum(flatten_numeric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(flatten_numeric_tree(item) for item in value)
    return 0.0


def estimate_sops(dense_ops: float, firing_rate: float) -> float:
    return float(dense_ops) * float(firing_rate)


class MetricAccumulator:
    def __init__(self, metrics: Iterable[str]):
        self.metrics = list(metrics)
        self.values: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update_aee(self, values) -> None:
        aee, pe1, pe2, pe3, outliers = values
        self._update_vector("AEE", aee)
        self._update_vector("AEE_PE1", pe1)
        self._update_vector("AEE_PE2", pe2)
        self._update_vector("AEE_PE3", pe3)
        self._update_vector("AEE_outliers", outliers)

    def update_scalar(self, name: str, value: torch.Tensor | float, count: int = 1) -> None:
        scalar = float(value.detach().cpu().item() if torch.is_tensor(value) else value)
        self.values[name] += scalar
        self.counts[name] += int(count)

    def _update_vector(self, name: str, value: torch.Tensor) -> None:
        detached = value.detach().cpu().float().reshape(-1)
        self.values[name] += float(detached.sum().item())
        self.counts[name] += int(detached.numel())

    def summary(self) -> dict[str, float]:
        keys = []
        for metric in self.metrics:
            keys.append(metric)
            if metric == "AEE":
                keys.extend(["AEE_PE1", "AEE_PE2", "AEE_PE3", "AEE_outliers"])
        return {
            key: self.values[key] / self.counts[key]
            for key in keys
            if self.counts[key] > 0
        }


class SpikeActivityProfiler:
    def __init__(self, model: torch.nn.Module, module_name_patterns: Iterable[str] = ("Spiking_neuron",)):
        self.model = model
        self.module_name_patterns = tuple(module_name_patterns)
        self.handles: list[Any] = []
        self.records: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "spikes": 0, "elements": 0, "rate_sum": 0.0}
        )

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__
            if any(pattern in class_name or pattern in name for pattern in self.module_name_patterns):
                self.handles.append(module.register_forward_hook(self._make_hook(name, class_name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str, class_name: str):
        key = name or class_name

        def hook(_module, _inputs, output):
            tensors = list(_iter_tensors(output))
            if not tensors:
                return
            spikes = 0
            elements = 0
            for tensor in tensors:
                detached = tensor.detach()
                spikes += int((detached != 0).sum().item())
                elements += int(detached.numel())
            if elements == 0:
                return
            record = self.records[key]
            record["calls"] += 1
            record["spikes"] += spikes
            record["elements"] += elements
            record["rate_sum"] += spikes / elements

        return hook

    def summary(self) -> dict[str, float]:
        total_spikes = int(sum(record["spikes"] for record in self.records.values()))
        total_elements = int(sum(record["elements"] for record in self.records.values()))
        global_rate = total_spikes / total_elements if total_elements else 0.0
        return {
            "num_layers": len(self.records),
            "total_spikes": total_spikes,
            "total_elements": total_elements,
            "global_firing_rate": global_rate,
        }

    def layer_rows(self) -> list[dict[str, float | str]]:
        rows = []
        for name, record in sorted(self.records.items()):
            elements = record["elements"]
            calls = record["calls"]
            rows.append(
                {
                    "layer": name,
                    "calls": int(calls),
                    "spikes": int(record["spikes"]),
                    "elements": int(elements),
                    "firing_rate": record["spikes"] / elements if elements else 0.0,
                    "mean_call_firing_rate": record["rate_sum"] / calls if calls else 0.0,
                }
            )
        return rows


def _iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def infer_overlay_path(config_path: Path, repo_root: Path) -> Path | None:
    resolved = config_path.resolve()
    experiments_root = (repo_root / "neuron_experiments").resolve()
    try:
        relative = resolved.relative_to(experiments_root)
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    overlay = experiments_root / relative.parts[0] / "overlay"
    return overlay if overlay.exists() else None


def install_project_paths(repo_root: Path, overlay: Path | None) -> Path:
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    if overlay is not None:
        sys.path.insert(0, str(overlay))
    return baseline_root


def resolve_neuron_type(config: dict):
    from spikingjelly.activation_based import neuron
    from models.STSwinNet_SNN.Spiking_submodules import GatedLIFNode, PSN, SLTTLIFNode

    neuron_type = config["model"]["spiking_neuron"]["neuron_type"]
    if neuron_type == "if":
        return getattr(neuron, "IFNode")
    if neuron_type == "lif":
        return getattr(neuron, "LIFNode")
    if neuron_type == "plif":
        return getattr(neuron, "ParametricLIFNode")
    if neuron_type == "glif":
        return GatedLIFNode
    if neuron_type == "psn":
        return PSN
    if neuron_type == "SLTTlif":
        return SLTTLIFNode
    try:
        from models.STSwinNet_SNN.experimental_neurons.factory import resolve_backend_neuron_type
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unsupported neuron_type without experiment overlay: {neuron_type}") from exc
    return resolve_backend_neuron_type(neuron_type)


def build_model_and_data(args: argparse.Namespace, repo_root: Path):
    from configs.parser import YAMLParser
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from DSEC_dataloader.data_augmentation import CenterCrop, Compose
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4, MS_SpikingformerFlowNet, SpikingformerFlowNet
    from spikingjelly.activation_based import functional
    from torch.utils.data import DataLoader
    from utils.runtime_backend import configure_snn_backend
    from utils.utils import load_model

    parser = YAMLParser(str(args.config))
    config = YAMLParser.combine_entries(parser.config)
    if args.batch_size is not None:
        config["loader"]["batch_size"] = args.batch_size
    config["loader"]["n_workers"] = args.num_workers
    config.setdefault("runtime", {})
    config["runtime"]["snn_backend"] = args.snn_backend
    config.setdefault("test", {})
    if args.num_samples is not None:
        config["test"]["sample"] = args.num_samples
    if args.metrics:
        config.setdefault("metrics", {})
        config["metrics"]["name"] = args.metrics

    if config["loader"]["crop"] is not None:
        transform = Compose([CenterCrop((config["loader"]["crop"][0], config["loader"]["crop"][1]))])
        config["swin_transformer"]["input_size"] = [config["loader"]["crop"][0], config["loader"]["crop"][1]]
    else:
        transform = None
        config["swin_transformer"]["input_size"] = [config["loader"]["resolution"][0], config["loader"]["resolution"][1]]

    dataset = DSECDatasetLite(config, file_list=args.split, stereo=False)
    loader = DataLoader(
        dataset=dataset,
        batch_size=config["loader"]["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    model_cls = {
        "SpikingformerFlowNet": SpikingformerFlowNet,
        "MS_SpikingformerFlowNet": MS_SpikingformerFlowNet,
        "MS_SpikingformerFlowNet_en4": MS_SpikingformerFlowNet_en4,
    }[config["model"]["name"]]
    if config["swin_transformer"]["use_arc"][0]:
        model = model_cls(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = model_cls(config["model"].copy())

    device = parser.device if args.device is None else torch.device(args.device)
    model.to(device)
    model.init_weights()
    if args.checkpoint:
        model = load_model(str(args.checkpoint), model, device, remap=args.remap, test=True)
    functional.reset_net(model)
    functional.set_step_mode(model, config["data"]["step_mode"])
    configure_snn_backend(model, device, config, resolve_neuron_type(config))
    model.eval()
    return config, model, loader, transform, device


def prepare_batch(
    chunk: torch.Tensor,
    label: torch.Tensor,
    mask: torch.Tensor,
    config: dict,
    transform,
):
    if transform is not None and label is not None and mask is not None:
        chunk, label, mask = transform((chunk, label, mask.float()))

    if config["model"]["encoding"] == "voxel":
        if config["loader"]["polarity"]:
            neg = torch.nn.functional.relu(-chunk)
            pos = torch.nn.functional.relu(chunk)
            chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)), dim=2)
    elif config["model"]["encoding"] == "cnt":
        if config["swin_transformer"]["use_arc"][1] == "PatchEmbed3D":
            chunk = torch.transpose(chunk, 1, 2)
        elif config["loader"]["polarity"]:
            chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))
    else:
        raise ValueError(f"Unsupported encoding: {config['model']['encoding']}")

    if config["model"]["norm_input"] == "minmax":
        non_zero = chunk != 0
        if non_zero.any():
            min_value = torch.min(chunk[non_zero])
            max_value = torch.max(chunk[non_zero])
            if min_value != max_value:
                chunk[non_zero] = (chunk[non_zero] - min_value) / (max_value - min_value)
    elif config["model"]["norm_input"] == "std":
        non_zero = chunk != 0
        if non_zero.any():
            stddev = chunk[non_zero].std()
            if stddev > 0:
                chunk[non_zero] = (chunk[non_zero] - chunk[non_zero].mean()) / stddev

    if config["data"]["spike_th"] is not None:
        chunk[chunk > config["data"]["spike_th"]] = 1
        chunk[chunk < config["data"]["spike_th"]] = 0
    return chunk, label, mask


def resolve_dense_ops(model: torch.nn.Module, dense_ops_arg: str, fallback: str) -> tuple[float, str]:
    if dense_ops_arg != "auto":
        return parse_human_number(dense_ops_arg), "cli"
    try:
        return flatten_numeric_tree(model.record_flops()), "model.record_flops"
    except Exception as exc:
        return parse_human_number(fallback), f"fallback:{fallback} ({exc.__class__.__name__})"


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    args.config = args.config.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    if args.overlay is not None:
        args.overlay = args.overlay.resolve()
    overlay = args.overlay
    if overlay is None:
        overlay = infer_overlay_path(args.config, repo_root)
    install_project_paths(repo_root, overlay)

    baseline_cwd = repo_root / "third_party" / "SDformerFlow"
    old_cwd = Path.cwd()
    try:
        # Match SDFormerFlow relative data paths such as ../../data/Datasets/...
        import os

        os.chdir(baseline_cwd)
        config, model, loader, transform, device = build_model_and_data(args, repo_root)
        from loss.flow_supervised import AAE, AEE

        metric_acc = MetricAccumulator(config.get("metrics", {}).get("name", []))
        dense_ops, dense_ops_source = resolve_dense_ops(model, args.dense_ops, args.fallback_dense_ops)
        profiler = SpikeActivityProfiler(model, args.module_pattern)
        profiler.attach()
        num_seen = 0
        from spikingjelly.activation_based import functional

        with torch.no_grad():
            for chunk, mask, label in loader:
                functional.reset_net(model)
                chunk = chunk.to(device=device, dtype=torch.float32, non_blocking=True)
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)
                mask = torch.unsqueeze(mask.to(device=device, non_blocking=True), dim=1)
                chunk, label, mask = prepare_batch(chunk, label, mask, config, transform)
                pred_list = model(chunk.to(device))
                pred = pred_list["flow"][-1]
                if config.get("metrics", {}).get("mask_events", False):
                    event_mask = torch.sum(torch.sum(chunk, dim=1), dim=1, keepdim=True).bool()
                    mask = mask * event_mask
                for metric in config.get("metrics", {}).get("name", []):
                    if metric == "AEE":
                        metric_acc.update_aee(AEE(pred, label, mask, config["metrics"]["flow_scaling"])())
                    elif metric == "AAE":
                        metric_acc.update_scalar("AAE", AAE(pred, label, mask, config["metrics"]["flow_scaling"])()[0])
                num_seen += chunk.shape[0]
                if args.num_samples is not None and num_seen >= args.num_samples:
                    break
        profiler.close()
    finally:
        import os

        os.chdir(old_cwd)

    summary = profiler.summary()
    estimated_sops = estimate_sops(dense_ops, summary["global_firing_rate"])
    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "overlay": str(overlay) if overlay else None,
        "split": args.split,
        "samples": num_seen,
        "dense_ops": dense_ops,
        "dense_ops_human": format_human_number(dense_ops),
        "dense_ops_source": dense_ops_source,
        "global_firing_rate": summary["global_firing_rate"],
        "estimated_total_sops": estimated_sops,
        "estimated_total_sops_human": format_human_number(estimated_sops),
        "profiled_layers": summary["num_layers"],
        "total_spikes": summary["total_spikes"],
        "total_elements": summary["total_elements"],
        "metrics": metric_acc.summary(),
        "layer_rows": profiler.layer_rows(),
    }
    return result


def write_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sops_summary.json"
    layers_path = output_dir / "layer_firing_rates.csv"
    summary = {key: value for key, value in result.items() if key != "layer_rows"}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = result["layer_rows"]
    with layers_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "calls", "spikes", "elements", "firing_rate", "mean_call_firing_rate"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, layers_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training/eval config to build the model and data loader.")
    parser.add_argument("--checkpoint", type=Path, help="Local checkpoint .pth to load before profiling.")
    parser.add_argument("--overlay", type=Path, help="Experiment overlay directory. Inferred for neuron_experiments configs.")
    parser.add_argument("--output-dir", type=Path, help="Directory for summary JSON and layer CSV.")
    parser.add_argument("--split", default="valid", choices=["train", "valid"], help="Dataset split to profile.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to profile.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for profiling.")
    parser.add_argument("--device", help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--snn-backend", default="torch", choices=["torch", "cupy", "auto"], help="SNN backend.")
    parser.add_argument(
        "--dense-ops",
        default="auto",
        help="Dense ops per inference. Use auto, or a value like 42.63G. SOPs = dense_ops * firing_rate.",
    )
    parser.add_argument("--fallback-dense-ops", default=DEFAULT_DENSE_OPS, help="Fallback when model.record_flops fails.")
    parser.add_argument("--remap", choices=["v1", "v2"], help="Checkpoint key remap mode.")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric to compute during profiling, e.g. AEE or AAE. Repeat to add metrics.",
    )
    parser.add_argument(
        "--module-pattern",
        action="append",
        default=["Spiking_neuron"],
        help="Module class/name pattern to hook. Repeat to add patterns.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("neuron_experiments") / "_profiles" / f"sops_{stamp}"
    result = run_profile(args)
    summary_path, layers_path = write_outputs(result, args.output_dir)
    print(f"samples: {result['samples']}")
    print(f"dense_ops: {result['dense_ops_human']} [{result['dense_ops_source']}]")
    print(f"global_firing_rate: {result['global_firing_rate']:.6f}")
    print(f"estimated_total_sops: {result['estimated_total_sops_human']}")
    if result["metrics"]:
        print("metrics:")
        for key, value in result["metrics"].items():
            print(f"  {key}: {value:.6f}")
    print(f"profiled_layers: {result['profiled_layers']}")
    print(f"summary: {summary_path}")
    print(f"layer_csv: {layers_path}")


if __name__ == "__main__":
    main()
