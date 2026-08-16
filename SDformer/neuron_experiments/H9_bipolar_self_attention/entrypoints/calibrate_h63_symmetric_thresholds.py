"""Calibrate per-module symmetric thresholds to TTX's one-sided event budgets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision  # noqa: F401
import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
OVERLAY = EXP_ROOT / "overlay"
DEFAULT_CONFIG = EXP_ROOT / "configs/generated/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
DEFAULT_SOURCE = (
    EXP_ROOT
    / "results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid"
    / "checkpoint_epoch2.pth"
)
DEFAULT_OUTPUT = EXP_ROOT / "results/h63_checkpoints/ttxep2_symmetric_budget_calibrated.pth"


class BudgetObserver:
    def __init__(self, sample_size: int) -> None:
        self.sample_size = int(sample_size)
        self.total = 0
        self.positive = 0
        self.abs_samples: list[torch.Tensor] = []
        self.step_samples: list[list[torch.Tensor]] = []

    def __call__(self, h_seq: torch.Tensor, threshold: torch.Tensor) -> None:
        flat = h_seq.reshape(-1)
        self.total += flat.numel()
        self.positive += int(flat.ge(threshold).sum().item())
        stride = max(1, flat.numel() // self.sample_size)
        sample = flat[::stride][: self.sample_size].abs().float().cpu()
        self.abs_samples.append(sample)
        if not self.step_samples:
            self.step_samples = [[] for _ in range(h_seq.shape[0])]
        per_step = max(1, self.sample_size // h_seq.shape[0])
        for step, values in enumerate(h_seq.reshape(h_seq.shape[0], -1)):
            step_stride = max(1, values.numel() // per_step)
            self.step_samples[step].append(values[::step_stride][:per_step].float().cpu())

    def threshold(self) -> tuple[float, float, int]:
        if self.total == 0 or not self.abs_samples:
            raise RuntimeError("observer received no activations")
        target_rate = self.positive / self.total
        values = torch.cat(self.abs_samples)
        if values.numel() > self.sample_size:
            stride = max(1, values.numel() // self.sample_size)
            values = values[::stride][: self.sample_size]
        if target_rate <= 0.0:
            calibrated = float(values.max().item()) + 1.0e-6
        else:
            calibrated = float(torch.quantile(values, min(1.0, max(0.0, 1.0 - target_rate))).item())
        return calibrated, target_rate, int(values.numel())

    def centered_threshold(self) -> tuple[torch.Tensor, float, float, int]:
        if self.total == 0 or not self.step_samples:
            raise RuntimeError("observer received no activations")
        centers = []
        residuals = []
        for chunks in self.step_samples:
            values = torch.cat(chunks)
            center = torch.median(values)
            centers.append(center)
            residuals.append((values - center).abs())
        residual = torch.cat(residuals)
        target_rate = self.positive / self.total
        if target_rate <= 0.0:
            threshold = float(residual.max().item()) + 1.0e-6
        else:
            threshold = float(
                torch.quantile(residual, min(1.0, max(0.0, 1.0 - target_rate))).item()
            )
        return torch.stack(centers).reshape(-1, 1), threshold, target_rate, int(residual.numel())


def extract_state_dict(payload):
    if hasattr(payload, "state_dict") and not isinstance(payload, dict):
        return payload.state_dict()
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in payload:
                return payload[key]
    raise TypeError(f"unsupported checkpoint type: {type(payload)!r}")


def normalize_config(config: dict) -> dict:
    cfg = dict(config)
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["spiking_neuron"] = cfg["spiking_neuron"]
    cfg["model"] = model_cfg
    crop = cfg["loader"].get("crop")
    swin = dict(cfg["swin_transformer"])
    swin["input_size"] = list(crop if crop is not None else cfg["loader"]["resolution"])
    cfg["swin_transformer"] = swin
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=65536)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--calibrate-center", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "third_party/SDformerFlow"))
    sys.path.insert(0, str(OVERLAY))
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from DSEC_dataloader.data_augmentation import CenterCrop, Compose
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN, install_atlif_ternary_psn
    from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, register_shiftmax_pickle_compat
    from spikingjelly.activation_based import functional
    from utils.runtime_backend import configure_snn_backend
    from utils.utils import _extract_pretrained_state_dict
    from tools.profile_sops import prepare_batch, resolve_neuron_type

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = normalize_config(raw)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MS_SpikingformerFlowNet_en4(config["model"].copy(), config["swin_transformer"].copy()).to(device)
    model.init_weights()
    register_shiftmax_pickle_compat()
    installed = install_atlif_ternary_psn(model, config["atlif_ternary_psn"])
    attention = install_shiftmax_attention(model, config["bsa_attention"])
    if len(installed) != 105 or len(attention) != 12:
        raise RuntimeError(f"install mismatch: ATLIF={len(installed)} attention={len(attention)}")

    payload = torch.load(args.source, map_location=device, weights_only=False)
    state = _extract_pretrained_state_dict(payload, test=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"source load mismatch: missing={incompatible.missing_keys[:8]} unexpected={incompatible.unexpected_keys[:8]}"
        )

    observers: dict[str, BudgetObserver] = {}
    for name, module in model.named_modules():
        if isinstance(module, ATLIFTernaryPSN):
            observer = BudgetObserver(args.sample_size)
            module._h9_calibration_observer = observer
            observers[name] = observer
    if len(observers) != 105:
        raise RuntimeError(f"observer mismatch: {len(observers)}")

    crop = config["loader"].get("crop")
    transform = Compose([CenterCrop(tuple(crop))]) if crop is not None else None
    old_cwd = Path.cwd()
    os.chdir(REPO_ROOT / "third_party/SDformerFlow")
    try:
        dataset = DSECDatasetLite(config, file_list="valid", stereo=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        config.setdefault("runtime", {})["snn_backend"] = "cupy"
        functional.set_step_mode(model, config["data"]["step_mode"])
        configure_snn_backend(model, device, config, resolve_neuron_type(config))
        model.eval()
        seen = 0
        with torch.no_grad():
            for chunk, mask, label in loader:
                functional.reset_net(model)
                chunk = chunk.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                mask = mask.to(device=device).unsqueeze(1)
                chunk, _, _ = prepare_batch(chunk, label, mask, config, transform)
                model(chunk)
                seen += chunk.shape[0]
                if seen >= args.samples:
                    break
    finally:
        os.chdir(old_cwd)

    output_state = {key: value.detach().cpu().clone() if torch.is_tensor(value) else value for key, value in state.items()}
    rows = []
    for name, observer in observers.items():
        key = f"{name}.thresh"
        if key not in output_state:
            raise KeyError(key)
        old_threshold = float(output_state[key].float().mean())
        if observer.total == 0:
            threshold, target_rate, used = old_threshold, 0.0, 0
            center = output_state[f"{name}.center"].float().clone()
            observed = False
        else:
            if args.calibrate_center:
                center, threshold, target_rate, used = observer.centered_threshold()
            else:
                threshold, target_rate, used = observer.threshold()
                center = output_state[f"{name}.center"].float().clone()
            observed = True
        output_state[key] = torch.full_like(output_state[key], threshold)
        center_key = f"{name}.center"
        if center_key not in output_state:
            raise KeyError(center_key)
        if tuple(center.shape) != tuple(output_state[center_key].shape):
            raise RuntimeError(
                f"center shape mismatch for {name}: calibrated={tuple(center.shape)} "
                f"checkpoint={tuple(output_state[center_key].shape)}"
            )
        output_state[center_key] = center.to(dtype=output_state[center_key].dtype)
        rows.append(
            {
                "module": name,
                "old_threshold": old_threshold,
                "symmetric_threshold": threshold,
                "target_positive_rate": target_rate,
                "calibration_values": used,
                "observed": observed,
                "center_min": float(center.min()),
                "center_mean": float(center.mean()),
                "center_max": float(center.max()),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": output_state,
            "h63_source_checkpoint": str(args.source.resolve()),
            "h63_calibration_samples": seen,
            "h63_calibration": (
                "per-timestep median center plus symmetric residual quantile matching original positive event rate"
                if args.calibrate_center
                else "per-module abs quantile matching original positive event rate"
            ),
        },
        args.output,
    )
    report = args.output.with_suffix(".json")
    report.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    observed_rows = [row for row in rows if row["observed"]]
    values = [row["symmetric_threshold"] for row in observed_rows]
    rates = [row["target_positive_rate"] for row in observed_rows]
    print(f"saved={args.output} report={report}")
    print(
        f"modules={len(rows)} observed={len(observed_rows)} samples={seen} threshold_min={min(values):.6f} "
        f"threshold_mean={sum(values)/len(values):.6f} threshold_max={max(values):.6f} "
        f"target_rate_mean={sum(rates)/len(rates):.6f}"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")
    raise SystemExit(main())
