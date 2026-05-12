"""Adapter-aware sparse profiler for SDFormerFlow.

Loads the model through the src/ adapter (supports sparse_ops, external_inspirations,
and all plug-in modules), profiles spike activity, and computes AEE + SOPs.

Usage:
    python -m autoresearch_sparsity.entrypoints.profile_sparse \
        --config configs/sdformer_baseline.yaml \
        --checkpoint experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
        --num-samples 40 \
        --metrics AEE AAE
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
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_human_number(value: str | float | int) -> float:
    if isinstance(value, Number):
        return float(value)
    text = value.strip()
    if not text:
        raise ValueError("empty numeric value")
    suffixes = {"k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12, "p": 1e15}
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

    def update_scalar(self, name: str, value, count: int = 1) -> None:
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
        return {key: self.values[key] / self.counts[key] for key in keys if self.counts[key] > 0}


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
            rows.append({
                "layer": name,
                "calls": int(calls),
                "spikes": int(record["spikes"]),
                "elements": int(elements),
                "firing_rate": record["spikes"] / elements if elements else 0.0,
                "mean_call_firing_rate": record["rate_sum"] / calls if calls else 0.0,
            })
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


def resolve_dense_ops(model: torch.nn.Module, dense_ops_arg: str, fallback: str) -> tuple[float, str]:
    if dense_ops_arg != "auto":
        return parse_human_number(dense_ops_arg), "cli"
    try:
        inner = getattr(model, "model", model)
        return flatten_numeric_tree(inner.record_flops()), "model.record_flops"
    except Exception as exc:
        return parse_human_number(fallback), f"fallback:{fallback} ({exc.__class__.__name__})"


def _patch_adapter_spiking_neuron() -> None:
    """Patch build_upstream_config to place spiking_neuron inside model dict."""
    from src.models.sdformer import backbone as _backbone, layers as _layers
    _orig_build = _backbone.build_upstream_config

    def _patched_build(cfg, mode="train"):
        result = _orig_build(cfg, mode)
        if "spiking_neuron" in result and "spiking_neuron" not in result.get("model", {}):
            result.setdefault("model", {})["spiking_neuron"] = result["spiking_neuron"]
        return result

    _backbone.build_upstream_config = _patched_build
    _layers.build_upstream_config = _patched_build


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    # Ensure project root is on path for src/ imports
    sys.path.insert(0, str(REPO_ROOT))
    upstream_root = str(REPO_ROOT / "third_party" / "SDformerFlow")
    sys.path.insert(0, upstream_root)

    from src.utils.config import load_config
    from src.utils.seed import set_seed
    from src.models.registry import build_model
    from src.datasets import build_dataset
    from src.datasets.transforms import move_batch_to_device
    from torch.utils.data import DataLoader
    from loss.flow_supervised import AAE, AEE

    # Fix: the adapter's build_upstream_config puts spiking_neuron at the
    # top level, but the upstream model constructor reads it from inside the
    # model dict. Monkey-patch the adapter to merge it before model init.
    _patch_adapter_spiking_neuron()

    cfg = load_config(str(args.config))
    if args.num_samples is not None:
        cfg.setdefault("runtime", {})["batch_size"] = args.batch_size
    set_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"]["device"] != "cpu" else "cpu")

    dataset = build_dataset(cfg, "eval")
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    model = build_model(cfg).to(device)
    if args.checkpoint:
        ckpt_data = torch.load(str(args.checkpoint), map_location=str(device))
        if hasattr(ckpt_data, "state_dict"):
            # Serialized model object (upstream format)
            model.model.load_state_dict(ckpt_data.state_dict(), strict=False)
        elif isinstance(ckpt_data, dict) and "model" in ckpt_data:
            # Dict-wrapped state dict
            sd = ckpt_data["model"]
            if isinstance(sd, dict):
                # Try adapter format first (with model. prefix), then raw
                try:
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    model.model.load_state_dict(sd, strict=False)
            elif hasattr(sd, "state_dict"):
                model.model.load_state_dict(sd.state_dict(), strict=False)
        else:
            model.load_state_dict(ckpt_data, strict=False)
    if hasattr(model, "configure_backend"):
        model.configure_backend()

    model.eval()

    metric_names = args.metrics or cfg["metrics"].get("names", ["AEE", "AAE"])
    metric_acc = MetricAccumulator(metric_names)

    inner_model = getattr(model, "model", model)
    try:
        dense_ops, dense_ops_source = resolve_dense_ops(model, args.dense_ops, args.fallback_dense_ops)
    except Exception:
        dense_ops = parse_human_number(args.fallback_dense_ops)
        dense_ops_source = f"fallback:{args.fallback_dense_ops}"

    profiler = SpikeActivityProfiler(inner_model, args.module_pattern)
    profiler.attach()

    num_seen = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            pred = outputs["flow_pred"]
            gt_flow = batch["gt_flow"]
            valid_mask = batch["valid_mask"]

            for metric_name in metric_names:
                if metric_name == "AEE":
                    aee_result = AEE(pred, gt_flow, valid_mask, cfg["metrics"]["flow_scaling"])()
                    metric_acc.update_aee(aee_result)
                elif metric_name == "AAE":
                    aae_result = AAE(pred, gt_flow, valid_mask, cfg["metrics"]["flow_scaling"])()[0]
                    metric_acc.update_scalar("AAE", aae_result)

            num_seen += pred.shape[0]
            if args.num_samples is not None and num_seen >= args.num_samples:
                break

    profiler.close()

    summary = profiler.summary()
    estimated_sops = estimate_sops(dense_ops, summary["global_firing_rate"])

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
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
    if rows:
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
    parser.add_argument("--config", type=Path, required=True, help="Project-level config YAML.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint .pth to load.")
    parser.add_argument("--output-dir", type=Path, help="Directory for summary JSON and layer CSV.")
    parser.add_argument("--num-samples", type=int, default=40, help="Number of samples to profile.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Metrics to compute (AEE, AAE).")
    parser.add_argument("--dense-ops", default="auto", help="Dense ops value or 'auto'.")
    parser.add_argument("--fallback-dense-ops", default=DEFAULT_DENSE_OPS, help="Fallback dense ops.")
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
        args.output_dir = REPO_ROOT / "autoresearch_sparsity" / "results" / f"profile_{stamp}"
    result = run_profile(args)
    summary_path, layers_path = write_outputs(result, args.output_dir)

    # Output in autoresearch.sh parseable format
    metrics = result["metrics"]
    aee = metrics.get("AEE", float("nan"))
    sops = result["estimated_total_sops"]
    firing = result["global_firing_rate"]

    print(f"METRIC aee={aee:.6f}")
    print(f"METRIC sops={sops:.6f}")
    print(f"METRIC firing_rate={firing:.6f}")
    if "AAE" in metrics:
        print(f"METRIC aae={metrics['AAE']:.6f}")
    if "AEE_PE1" in metrics:
        print(f"METRIC aee_pe1={metrics['AEE_PE1']:.6f}")

    print(f"\nsamples: {result['samples']}")
    print(f"dense_ops: {result['dense_ops_human']} [{result['dense_ops_source']}]")
    print(f"estimated_total_sops: {result['estimated_total_sops_human']}")
    print(f"profiled_layers: {result['profiled_layers']}")
    print(f"summary: {summary_path}")
    if layers_path.exists():
        print(f"layer_csv: {layers_path}")


if __name__ == "__main__":
    main()
