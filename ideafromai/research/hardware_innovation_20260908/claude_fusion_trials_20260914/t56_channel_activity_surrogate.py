#!/usr/bin/env python3
"""T56: 逐通道活性 + surrogate 梯度审计，以及「官方 vs 修正」ATLIF 阈值梯度 A/B。

背景（2026-09-19 核对）：
  H9 的 `OfficialATLIFSurrogate` 与上游 putshua/Activity-Pruning-SNN 的
  `Surrogate` **逐行等价** —— `tmp` 连乘两次、缺幅度项 `out` 都是官方原始写法，
  不是移植 bug。所以这里不「修 bug」，而是把数学上正确的阈值梯度做成**单变量对照**：

    前向  y = H(h-θ)·θ
    官方  ∂y/∂h = tmp                    ,  ∂y/∂θ = -mean(grad·tmp²)
    修正  ∂y/∂h = tmp （不变）           ,  ∂y/∂θ = Σ grad·(g - tmp)

  其中 tmp = (1-|（h-θ)/θ|)₊，g = H(h-θ)。
  修正版只改 θ 的梯度、不改 h 的梯度 ⇒ 归因干净。

同时产出 GPT 要的逐通道审计表：活性率 / 膜电位到阈值距离 / surrogate 梯度幅值，
三分法（结构性无用 / 合理低活 / 训练致死）。

用法（远端）：
  python /root/t56_channel_activity_surrogate.py --samples 4 --out /root/t56_audit
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
from pathlib import Path

import torch
import torchvision  # noqa: F401 - keep SDFormerFlow import order

REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
EXP_ROOT = REPO / "neuron_experiments/H9_bipolar_self_attention"
OVERLAY_ROOT = EXP_ROOT / "overlay"
BASELINE_ROOT = REPO / "third_party/SDformerFlow"
ENTRY = EXP_ROOT / "entrypoints"

sys.path.insert(0, str(OVERLAY_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ENTRY))

os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")

from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite  # noqa: E402
from DSEC_dataloader.data_augmentation import CenterCrop, Compose  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402

import profile_nts11_hardware_p0 as P  # noqa: E402


def zif_backward(x: torch.Tensor, thre: torch.Tensor) -> torch.Tensor:
    return (1.0 - (x / thre).abs()).clamp_min(0)


class CorrectedATLIFSurrogate(torch.autograd.Function):
    """官方 ATLIF 前向语义不变，只把 θ 的梯度改成数学上正确的形式。"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, thre: torch.Tensor, sp: float):
        out = (input >= thre).float()
        thre_updates = (sp * zif_backward(input - thre, thre) * out).sum(0).mean().item()
        ctx.save_for_backward(input, thre, out)
        return out * thre, thre_updates

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor, _dummy):
        input, thre, out = ctx.saved_tensors
        tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
        grad_h = grad_input * tmp                      # 与官方一致，单变量
        grad_thre = (grad_input * (out - tmp)).sum()   # 官方漏了幅度项 g，且多乘 tmp、取 mean
        return grad_h, grad_thre, None


class ATLIFAudit:
    """挂在每个 ATLIFTernaryPSN 上，抓真实输入并复算膜电位 h = W·x + b。"""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.handles: list = []

    def attach(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if module.__class__.__name__ != "ATLIFTernaryPSN":
                continue
            self.rows[name] = {
                "name": name,
                "T": int(getattr(module, "T", 0)),
                "output_mode": getattr(module, "output_mode", ""),
                "threshold_mode": getattr(module, "threshold_mode", ""),
                "theta_before": float(module.thresh.detach().float()),
                "calls": 0,
                "elements": 0,
                "active": 0,
                "abs_margin_sum": 0.0,
                "band_ratio": {1.0 / 16: 0, 1.0 / 4: 0, 1.0: 0, 2.0: 0},
                "window_elems": 0,
                "grad_out_abs_sum": 0.0,
            }
            self.handles.append(module.register_forward_hook(self._make_hook(name, module)))

    def _make_hook(self, name: str, module: torch.nn.Module):
        row = self.rows[name]

        def hook(mod, inputs, output):
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            with torch.no_grad():
                T = int(x.shape[0])
                flat = x.detach().float().reshape(T, -1)
                max_vectors = 4096
                stride = max(1, (flat.shape[1] + max_vectors - 1) // max_vectors)
                flat = flat[:, ::stride][:, :max_vectors]
                h = torch.addmm(
                    module.bias.detach().float(),
                    module.weight.detach().float(),
                    flat,
                )
                theta = module.thresh.detach().float()
                margin = (h - theta).abs()
                row["calls"] += 1
                row["elements"] += int(h.numel())
                row["active"] += int((h >= theta).sum().item())
                row["abs_margin_sum"] += float(margin.sum().item())
                for band in row["band_ratio"]:
                    row["band_ratio"][band] += int((margin <= band).sum().item())
                row["window_elems"] += int((margin < theta).sum().item())
            if torch.is_tensor(output) and output.requires_grad:
                output.register_hook(self._make_out_hook(name))

        return hook

    def _make_out_hook(self, name: str):
        def out_hook(grad):
            with torch.no_grad():
                self.rows[name]["grad_out_abs_sum"] += float(grad.detach().float().abs().sum().item())
            return grad

        return out_hook

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def table(self) -> list[dict]:
        out = []
        for row in self.rows.values():
            elems = max(row["elements"], 1)
            out.append({
                "name": row["name"],
                "T": row["T"],
                "output_mode": row["output_mode"],
                "threshold_mode": row["threshold_mode"],
                "theta_before": row["theta_before"],
                "theta_after_official": row.get("theta_after_official"),
                "theta_after_corrected": row.get("theta_after_corrected"),
                "activity_rate": row["active"] / elems,
                "mean_abs_margin": row["abs_margin_sum"] / elems,
                "surrogate_window_frac": row["window_elems"] / elems,
                "band_le_1_16": row["band_ratio"][1.0 / 16] / elems,
                "band_le_1_4": row["band_ratio"][1.0 / 4] / elems,
                "band_le_1": row["band_ratio"][1.0] / elems,
                "band_le_2": row["band_ratio"][2.0] / elems,
                "grad_out_abs_sum": row["grad_out_abs_sum"],
                "grad_theta_official": row.get("grad_theta_official"),
                "grad_theta_corrected": row.get("grad_theta_corrected"),
            })
        return out


def masked_epe(prediction: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, flow_scaling: float) -> torch.Tensor:
    diff = (prediction.float() * float(flow_scaling) - label.float()).abs()
    m = mask.float()
    return (diff * m).sum() / m.sum().clamp_min(1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=REPO / "hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs/"
                                    "dsec_c12_alpha0125_ep29_resume5_20260830.yml")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO / "hw_autoresearch_nts07/system_handoff/incoming/"
                                   "motion_c12_ep34_live93_checkpoint_epoch34.pth")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("/root/t56_audit"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config, device = P.load_config(args.config.resolve())
    print(f"[t56] device={device} config={args.config}", flush=True)

    dataset = DSECDatasetLite(config, file_list="valid", stereo=False,
                              scale_factor=config.get("test", {}).get("scale_factor", 1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         drop_last=False, pin_memory=False, num_workers=0)
    transform_valid = None
    if config["loader"].get("crop") is not None:
        transform_valid = Compose([CenterCrop((config["loader"]["crop"][0], config["loader"]["crop"][1]))])

    flow_scaling = float(config.get("metrics", {}).get("flow_scaling", 128))

    def run(arm: str) -> list[dict]:
        model = P.build_model(config, args.checkpoint.resolve(), device)
        P.configure_batch_norm_evaluation(model, config.get("test", {}).get("bn_policy", "running"))
        audit = ATLIFAudit()
        audit.attach(model)

        if arm == "corrected":
            patched = 0
            for module in model.modules():
                if module.__class__.__name__ == "ATLIFTernaryPSN":
                    module.act = CorrectedATLIFSurrogate.apply
                    patched += 1
            print(f"[t56] arm={arm}: patched {patched} ATLIF modules", flush=True)

        processed = 0
        loss_sum = 0.0
        for chunk, mask, label in loader:
            if processed >= args.samples:
                break
            functional.reset_net(model)
            x, tlabel, tmask = P.preprocess_chunk(config, chunk, label, mask, transform_valid, device)
            for _, module in model.named_modules():
                if module.__class__.__name__ == "ATLIFTernaryPSN":
                    module.thresh.grad = None
            prediction = model(x)["flow"][-1]
            loss = masked_epe(prediction, tlabel, tmask, flow_scaling)
            loss.backward()
            loss_sum += float(loss.detach().cpu())
            processed += 1
            print(f"[t56] arm={arm} sample {processed} loss={float(loss.detach().cpu()):.4f}", flush=True)

        for name, module in model.named_modules():
            if name in audit.rows:
                grad = module.thresh.grad
                audit.rows[name]["grad_theta_" + arm] = None if grad is None else float(grad.detach().float())
                audit.rows[name]["theta_after_" + arm] = float(module.thresh.detach().float())

        audit.detach()
        rows = audit.table()
        for row in rows:
            row["arm"] = arm
            row["mean_loss"] = loss_sum / max(processed, 1)
        return rows

    official = run("official")
    corrected = run("corrected")

    by_name = {row["name"]: row for row in official}
    for row in corrected:
        base = by_name.get(row["name"])
        if base is not None:
            row["grad_theta_official"] = base["grad_theta_official"]
            row["activity_rate"] = base["activity_rate"]

    def classify(row: dict) -> str:
        if not row["activity_rate"]:
            return "structurally_dead"
        if row["surrogate_window_frac"] < 1e-6:
            return "training_dead_no_surrogate_window"
        if row["activity_rate"] < 0.01:
            return "reasonable_low_activity"
        return "healthy"

    for row in corrected:
        row["class"] = classify(row)

    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "samples": args.samples,
        "flow_scaling": flow_scaling,
        "n_modules": len(corrected),
        "grad_theta_official_absmax": max((abs(r["grad_theta_official"] or 0.0) for r in corrected), default=0.0),
        "grad_theta_corrected_absmax": max((abs(r["grad_theta_corrected"] or 0.0) for r in corrected), default=0.0),
        "rows": corrected,
    }
    (args.out / "t56_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    keys = sorted({k for r in corrected for k in r})
    with (args.out / "channel_activity_and_surrogate.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(corrected)

    print("=== T56 SUMMARY ===", flush=True)
    print(f"modules={len(corrected)}", flush=True)
    print(f"grad_theta |official| absmax = {payload['grad_theta_official_absmax']:.6e}", flush=True)
    print(f"grad_theta |corrected| absmax = {payload['grad_theta_corrected_absmax']:.6e}", flush=True)
    live = [r for r in corrected if r["activity_rate"] > 0]
    print(f"live modules={len(live)}  dead={len(corrected) - len(live)}", flush=True)
    win = sorted(r["surrogate_window_frac"] for r in corrected)
    print(f"surrogate_window_frac min/med/max = {win[0]:.3e}/{win[len(win)//2]:.3e}/{win[-1]:.3e}", flush=True)
    print("=== T56 COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
