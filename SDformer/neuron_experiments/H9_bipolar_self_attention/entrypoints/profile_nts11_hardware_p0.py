"""Profile NTS11 hardware-facing H60/ATLIF/skip statistics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torchvision  # noqa: F401 - keep SDFormerFlow import order
import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
BASELINE_ROOT = REPO_ROOT / "third_party" / "SDformerFlow"
OVERLAY_ROOT = EXP_ROOT / "overlay"

sys.path.insert(0, str(OVERLAY_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")

from configs.parser import YAMLParser  # noqa: E402
from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite  # noqa: E402
from DSEC_dataloader.data_augmentation import CenterCrop, Compose  # noqa: E402
from models.STSwinNet_SNN.Spiking_STSwinNet import (  # noqa: E402
    MS_SpikingformerFlowNet,
    MS_SpikingformerFlowNet_en4,
    SpikingformerFlowNet,
)
from spikingjelly.activation_based import functional, neuron  # noqa: E402
from utils.runtime_backend import configure_snn_backend  # noqa: E402


def load_config(path: Path) -> tuple[dict[str, Any], torch.device]:
    parser = YAMLParser(str(path))
    config = YAMLParser.combine_entries(parser.config)
    config["loader"]["batch_size"] = 1
    config["loader"]["shuffle"] = False
    config["loader"]["pin_memory"] = False
    config["loader"]["num_workers"] = 0
    if config["loader"].get("crop") is not None:
        config["swin_transformer"]["input_size"] = [
            int(config["loader"]["crop"][0]),
            int(config["loader"]["crop"][1]),
        ]
    else:
        config["swin_transformer"]["input_size"] = [
            int(config["loader"]["resolution"][0]),
            int(config["loader"]["resolution"][1]),
        ]
    return config, parser.device


def install_overlay_modules(model: torch.nn.Module, config: dict[str, Any]) -> None:
    import models
    import models.STSwinNet_SNN as stsnn

    overlay_models = str(OVERLAY_ROOT / "models")
    overlay_stsnn = str(OVERLAY_ROOT / "models" / "STSwinNet_SNN")
    if overlay_models not in list(models.__path__):
        models.__path__.append(overlay_models)
    if overlay_stsnn not in list(stsnn.__path__):
        stsnn.__path__.append(overlay_stsnn)
    if config.get("atlif_ternary_psn", {}).get("enabled"):
        from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn

        installed = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
        print(f"[profile] installed ATLIF modules: {len(installed)}", flush=True)
    if config.get("bsa_attention", {}).get("enabled"):
        from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, register_shiftmax_pickle_compat

        register_shiftmax_pickle_compat()
        installed = install_shiftmax_attention(model, config.get("bsa_attention"))
        print(f"[profile] installed H60/Shiftmax modules: {len(installed)}", flush=True)


def build_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = eval(config["model"]["name"])(config["model"].copy(), config["swin_transformer"].copy())
    model.to(device)
    model.init_weights()
    install_overlay_modules(model, config)
    from models.STSwinNet_SNN.h9_load_audit import load_checkpoint_with_h9_audit

    remap = config["loader"].get("remap")
    model = load_checkpoint_with_h9_audit(
        str(checkpoint),
        model,
        device,
        config=config,
        remap=remap,
        test=True,
    )
    functional.reset_net(model)
    functional.set_step_mode(model, config["data"]["step_mode"])

    neuron_type = config["model"]["spiking_neuron"]["neuron_type"]
    if neuron_type == "if":
        neurontype = neuron.IFNode
    elif neuron_type == "lif":
        neurontype = neuron.LIFNode
    elif neuron_type == "plif":
        neurontype = neuron.ParametricLIFNode
    elif neuron_type == "psn":
        from models.STSwinNet_SNN.Spiking_submodules import PSN

        neurontype = PSN
    else:
        neurontype = None
    if neurontype is not None:
        configure_snn_backend(model, device, config, neurontype)
    model.eval()
    return model


def iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    shape = list(tensor.shape)
    elements = int(tensor.numel())
    active = int(tensor.detach().ne(0).sum().item())
    return {
        "shape": shape,
        "elements": elements,
        "active": active,
        "density": active / elements if elements else 0.0,
        "bytes_fp16": elements * 2,
        "bytes_fp32": elements * 4,
        "bytes_binary_packed": (elements + 7) // 8,
        "bytes_ternary_packed": (elements * 2 + 7) // 8,
    }


class HardwareProfiler:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: list[Any] = []
        self.h60_records: list[dict[str, Any]] = []
        self.activation_records: list[dict[str, Any]] = []
        self.atlif_records: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "calls": 0,
            "elements": 0,
            "active": 0,
            "pos": 0,
            "neg": 0,
        })

    def attach(self) -> None:
        unet = self.model.sttmultires_unet
        swin = unet.encoders.swin3d
        self.handles.append(swin.patch_embed.register_forward_hook(self._activation_hook("patch_embed", "patch")))
        for stage_idx, layer in enumerate(swin.layers):
            self.handles.append(layer.register_forward_hook(self._stage_hook(stage_idx)))
            if getattr(layer, "downsample", None) is not None:
                self.handles.append(layer.downsample.register_forward_hook(self._activation_hook(f"S{stage_idx}.downsample", "downsample")))
            for block_idx, block in enumerate(layer.swin_blocks):
                self.handles.append(block.register_forward_hook(self._activation_hook(f"S{stage_idx}.B{block_idx}", "swin_block")))
                attn = getattr(block, "attn", None)
                if attn is not None and hasattr(attn, "_h9_shiftmax_cfg"):
                    attn._h9_profile_collector = self._h60_collector(f"S{stage_idx}.B{block_idx}.attn")
        for idx, module in enumerate(unet.resblocks):
            self.handles.append(module.register_forward_hook(self._activation_hook(f"resblock{idx}", "resblock")))
        for idx, module in enumerate(unet.decoders):
            self.handles.append(module.register_forward_hook(self._activation_hook(f"decoder{idx}", "decoder")))
        for idx, module in enumerate(unet.preds):
            self.handles.append(module.register_forward_hook(self._activation_hook(f"pred{idx}", "prediction")))
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "ATLIFTernaryPSN":
                self.handles.append(module.register_forward_hook(self._atlif_hook(name)))

    def close(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "_h9_profile_collector"):
                delattr(module, "_h9_profile_collector")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _h60_collector(self, name: str):
        def collect(_module: torch.nn.Module, stats: dict[str, Any]) -> None:
            stats = dict(stats)
            stats["name"] = name
            self.h60_records.append(stats)

        return collect

    def _activation_hook(self, name: str, kind: str):
        def hook(_module: torch.nn.Module, _inp: Any, out: Any) -> None:
            tensors = list(iter_tensors(out))
            if not tensors:
                return
            rec = tensor_record(tensors[0])
            rec.update({"name": name, "kind": kind})
            self.activation_records.append(rec)

        return hook

    def _stage_hook(self, stage_idx: int):
        def hook(_module: torch.nn.Module, _inp: Any, out: Any) -> None:
            if not isinstance(out, tuple) or len(out) != 2:
                return
            x_out, skip = out
            skip_rec = tensor_record(skip)
            skip_kind = "stage_skip_predownsample" if stage_idx < 3 else "stage_skip_final"
            skip_rec.update({"name": f"S{stage_idx}.skip", "kind": skip_kind})
            self.activation_records.append(skip_rec)
            out_rec = tensor_record(x_out)
            out_rec.update({"name": f"S{stage_idx}.x_out", "kind": "stage_x_out"})
            self.activation_records.append(out_rec)

        return hook

    def _atlif_hook(self, name: str):
        def hook(module: torch.nn.Module, _inp: Any, out: Any) -> None:
            tensors = list(iter_tensors(out))
            if not tensors:
                return
            t = tensors[0].detach()
            rec = self.atlif_records[name]
            rec["calls"] += 1
            rec["elements"] += int(t.numel())
            rec["active"] += int(t.ne(0).sum().item())
            rec["pos"] += int(t.gt(0).sum().item())
            rec["neg"] += int(t.lt(0).sum().item())
            rec["output_mode"] = str(getattr(module, "output_mode", "unknown"))
            rec["threshold_mode"] = str(getattr(module, "threshold_mode", "unknown"))

        return hook

    @staticmethod
    def _aggregate_numeric(records: list[dict[str, Any]], keys: list[str], group_key: str) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for rec in records:
            group = str(rec[group_key])
            row = grouped.setdefault(group, {"group": group, "calls": 0})
            row["calls"] += 1
            for key in keys:
                if key in rec:
                    row[key] = row.get(key, 0.0) + float(rec[key])
        for row in grouped.values():
            calls = max(int(row["calls"]), 1)
            for key in keys:
                if key in row:
                    row[key] /= calls
        return sorted(grouped.values(), key=lambda row: row["group"])

    def summary(self) -> dict[str, Any]:
        h60_keys = [
            "tx_mean", "tx_std", "sc_mean", "sc_std", "fused_mean", "fused_std",
            "gate_entropy_mean", "top1_mass_mean", "top4_mass_mean", "effective_tokens_mean",
            "q_active_density", "k_active_density", "q_token_active_density", "k_token_active_density",
            "ttb1_empty_ratio", "ttb2_empty_ratio", "ttb4_empty_ratio",
            "ttb1_low_density_ratio", "ttb2_low_density_ratio", "ttb4_low_density_ratio",
        ]
        act_by_kind: dict[str, dict[str, Any]] = {}
        for rec in self.activation_records:
            row = act_by_kind.setdefault(rec["kind"], {"kind": rec["kind"], "calls": 0, "elements": 0, "active": 0, "bytes_fp16": 0, "bytes_ternary_packed": 0})
            row["calls"] += 1
            row["elements"] += rec["elements"]
            row["active"] += rec["active"]
            row["bytes_fp16"] += rec["bytes_fp16"]
            row["bytes_ternary_packed"] += rec["bytes_ternary_packed"]
        for row in act_by_kind.values():
            row["density"] = row["active"] / row["elements"] if row["elements"] else 0.0
        atlif_rows = []
        for name, rec in sorted(self.atlif_records.items()):
            row = dict(rec)
            row["name"] = name
            row["activity"] = row["active"] / row["elements"] if row["elements"] else 0.0
            row["pos_rate"] = row["pos"] / row["elements"] if row["elements"] else 0.0
            row["neg_rate"] = row["neg"] / row["elements"] if row["elements"] else 0.0
            atlif_rows.append(row)
        return {
            "h60_records": self.h60_records,
            "h60_by_block": self._aggregate_numeric(self.h60_records, h60_keys, "name"),
            "h60_by_stage": self._aggregate_numeric(self.h60_records, h60_keys, "stage"),
            "activation_records": self.activation_records,
            "activation_by_kind": sorted(act_by_kind.values(), key=lambda row: row["kind"]),
            "atlif_rows": atlif_rows,
        }


def preprocess_chunk(config: dict[str, Any], chunk: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, transform_valid: Any, device: torch.device):
    chunk = chunk.to(device=device, dtype=torch.float32)
    label = label.to(device=device, dtype=torch.float32)
    mask = torch.unsqueeze(mask.to(device=device), dim=1)
    if transform_valid is not None:
        chunk, label, mask = transform_valid((chunk, label, mask.float()))
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
        raise AttributeError("Unsupported event encoding")
    if config["model"]["norm_input"] == "minmax" and torch.any(chunk != 0):
        mn = torch.min(chunk[chunk != 0])
        mx = torch.max(chunk[chunk != 0])
        if mn != mx:
            chunk[chunk != 0] = (chunk[chunk != 0] - mn) / (mx - mn)
    elif config["model"]["norm_input"] == "std" and torch.any(chunk != 0):
        mean = chunk[chunk != 0].mean()
        std = chunk[chunk != 0].std()
        if std > 0:
            chunk[chunk != 0] = (chunk[chunk != 0] - mean) / std
    if config["data"]["spike_th"] is not None:
        chunk[chunk > config["data"]["spike_th"]] = 1
        chunk[chunk < config["data"]["spike_th"]] = 0
    return chunk


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    h60_stage = summary["h60_by_stage"]
    activ = summary["activation_by_kind"]
    atlif = summary["atlif_rows"]
    lines = [
        "# NTS11 硬件 P0 Profiling 报告",
        "",
        f"- 实验：`{result['experiment']}`",
        f"- checkpoint：`{result['checkpoint']}`",
        f"- samples：{result['samples']}",
        f"- H60 调用记录：{len(summary['h60_records'])}",
        f"- ATLIF 记录模块：{len(atlif)}",
        "",
        "## H60 分 stage 统计",
        "",
        "| stage | calls | gate_entropy | top1_mass | top4_mass | effective_tokens | q_active | k_active | TTB1 empty | TTB2 empty | TTB4 empty |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in h60_stage:
        lines.append(
            f"| {row['group']} | {int(row['calls'])} | {row.get('gate_entropy_mean', 0):.4f} | "
            f"{row.get('top1_mass_mean', 0):.4f} | {row.get('top4_mass_mean', 0):.4f} | "
            f"{row.get('effective_tokens_mean', 0):.2f} | {row.get('q_active_density', 0):.5f} | "
            f"{row.get('k_active_density', 0):.5f} | {row.get('ttb1_empty_ratio', 0):.4f} | "
            f"{row.get('ttb2_empty_ratio', 0):.4f} | {row.get('ttb4_empty_ratio', 0):.4f} |"
        )
    lines += [
        "",
        "## Activation / Skip 存储口径",
        "",
        "| kind | calls | elements | density | FP16 bytes | ternary packed bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in activ:
        lines.append(
            f"| {row['kind']} | {row['calls']} | {row['elements']} | {row['density']:.6f} | "
            f"{row['bytes_fp16']} | {row['bytes_ternary_packed']} |"
        )
    ternary = [row for row in atlif if row.get("output_mode") == "ternary"]
    binary = [row for row in atlif if row.get("output_mode") == "binary"]
    def avg(rows: list[dict[str, Any]], key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0
    lines += [
        "",
        "## ATLIF 活性快照",
        "",
        "| group | modules | activity | pos_rate | neg_rate |",
        "|---|---:|---:|---:|---:|",
        f"| ternary | {len(ternary)} | {avg(ternary, 'activity'):.6f} | {avg(ternary, 'pos_rate'):.6f} | {avg(ternary, 'neg_rate'):.6f} |",
        f"| binary | {len(binary)} | {avg(binary, 'activity'):.6f} | {avg(binary, 'pos_rate'):.6f} | {avg(binary, 'neg_rate'):.6f} |",
        "",
        "## 读法",
        "",
        "- `stage_skip_predownsample` 只对应 S0/S1/S2 的 downsample 前 skip。",
        "- `stage_skip_final` 对应 S3 final-stage output，硬件上要跨 bottleneck 保留给 decoder i=0。",
        "- TTB empty/low/high 目前按 H60 Q token 活性估算，是调度可跳过性的保守代理指标。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config, device = load_config(args.config)
    dataset = DSECDatasetLite(config, file_list="valid", stereo=False, scale_factor=config.get("test", {}).get("scale_factor", 1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=False, num_workers=args.num_workers)
    transform_valid = None
    if config["loader"].get("crop") is not None:
        transform_valid = Compose([CenterCrop((config["loader"]["crop"][0], config["loader"]["crop"][1]))])
    model = build_model(config, args.checkpoint, device)
    profiler = HardwareProfiler(model)
    profiler.attach()
    processed = 0
    try:
        with torch.no_grad():
            for chunk, mask, label in loader:
                if processed >= args.samples:
                    break
                functional.reset_net(model)
                x = preprocess_chunk(config, chunk, label, mask, transform_valid, device)
                _ = model(x)
                processed += 1
                if processed % 5 == 0:
                    print(f"[profile] processed {processed}/{args.samples}", flush=True)
    finally:
        profiler.close()

    try:
        from models.STSwinNet_SNN.atlif_ternary_psn import atlif_ternary_summary
        atlif_summary = atlif_ternary_summary(model)
    except Exception as exc:
        atlif_summary = {"error": str(exc)}
    result = {
        "experiment": config.get("experiment", args.config.stem),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "samples": processed,
        "atlif_summary": atlif_summary,
        "summary": profiler.summary(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "nts11_hardware_p0_profile.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output_dir / "h60_by_block.csv", result["summary"]["h60_by_block"])
    write_csv(args.output_dir / "h60_by_stage.csv", result["summary"]["h60_by_stage"])
    write_csv(args.output_dir / "activation_records.csv", result["summary"]["activation_records"])
    write_csv(args.output_dir / "atlif_activity.csv", result["summary"]["atlif_rows"])
    write_md(args.output_dir / "nts11_hardware_p0_profile.md", result)
    print(f"[profile] wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
