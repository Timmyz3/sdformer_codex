"""Create a reproducible state-dict warm start with scaled ATLIF thresholds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
DEFAULT_SOURCE = (
    EXP_ROOT
    / "results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid"
    / "checkpoint_epoch2.pth"
)
DEFAULT_OUTPUT = EXP_ROOT / "results/h63_checkpoints/ttxep2_symmetric_threshold_x4.pth"


def extract_state_dict(payload):
    if hasattr(payload, "state_dict") and not isinstance(payload, dict):
        return payload.state_dict()
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in payload:
                return payload[key]
    raise TypeError(f"unsupported checkpoint type: {type(payload)!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale", type=float, default=4.0)
    args = parser.parse_args()
    if args.scale <= 0:
        raise ValueError("scale must be positive")

    sys.path.insert(0, str(REPO_ROOT / "third_party/SDformerFlow"))
    sys.path.insert(0, str(EXP_ROOT / "overlay"))
    from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat

    register_shiftmax_pickle_compat()
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    state = extract_state_dict(payload)
    scaled = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in state.items()}
    threshold_keys = [key for key in scaled if key.endswith(".spiking_neuron.thresh")]
    if len(threshold_keys) != 105:
        raise RuntimeError(f"expected 105 ATLIF thresholds, found {len(threshold_keys)}")
    for key in threshold_keys:
        scaled[key].mul_(float(args.scale))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": scaled,
            "h63_source_checkpoint": str(args.source.resolve()),
            "h63_threshold_scale": float(args.scale),
            "h63_threshold_keys": len(threshold_keys),
        },
        args.output,
    )
    values = [float(scaled[key].float().mean()) for key in threshold_keys]
    print(f"saved={args.output}")
    print(f"threshold_keys={len(values)} min={min(values):.6f} mean={sum(values)/len(values):.6f} max={max(values):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
