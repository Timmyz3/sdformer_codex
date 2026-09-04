"""Generate DATE11 all-binary dual-rail TX configs.

The existing all-binary TX path sees only {0,+1} events, so TX opposite-polarity
terms collapse. These configs use a dedicated dual-rail binary TX attention mode:
the first half of each head is treated as positive rails, the second half as
negative rails, restoring same/opposite polarity scoring while keeping binary
ATLIF everywhere.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
MANIFEST = GEN / "date11_dualrail_tx_manifest.json"

FULL30_BASE = GEN / "date11full_all_binary_atlif_tx_w720_fastlr_full30.yml"
FT5_BASE = GEN / "date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5.yml"

NB0_RESUME = Path("experiments/baseline_stride_upstream/checkpoint_epoch59.pth")
TX_EP19_RESUME = Path(
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/"
    "checkpoint_epoch19.pth"
)


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def beta_tag(beta: float) -> str:
    return f"b{int(round(beta * 100)):03d}"


def apply_dualrail_tx(cfg: dict[str, Any], beta: float) -> None:
    attn = cfg.setdefault("bsa_attention", {})
    attn.update(
        {
            "enabled": True,
            "mode": "dualrail_binary_tx_qkselector_shiftmax",
            "center_scores": True,
            "preserve_mean": True,
            "alpha0": 0.02,
            "mismatch_penalty": float(beta),
            "single_active_penalty": 0.10,
            "single_active_penalty_grad": "ste",
            "score_scale": 1.0,
            "consensus_bias": 0.02,
            "consensus_score_norm": "head_dim",
            "eps": 1.0e-6,
            "relu_k_floor": 0.0,
            "value_mode": "threshold",
            "target_rate": None,
            "sc_mu_schedule_enabled": False,
            "bipolar_mu": 0.0,
            "k_magnitude_alpha": 0.0,
        }
    )


def make_full30(beta: float) -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(FULL30_BASE))
    tag = beta_tag(beta)
    cfg["experiment"] = f"date11full_all_binary_atlif_drtx_{tag}_w720_fastlr_full30"
    cfg["note"] = (
        "DATE11 all-binary dual-rail TX full30: 105 binary ATLIF modules; "
        "12 dualrail_binary_tx_qkselector_shiftmax attention modules; "
        f"opposite-rail penalty beta={beta}; single-active penalty=0.10. "
        "Resume from NB0 ep59."
    )
    apply_dualrail_tx(cfg, beta)
    cfg.setdefault("loader", {})["n_epochs"] = 30
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "full30",
        "config": str(out),
        "resume": str(NB0_RESUME),
        "attention_mode": "dualrail_binary_tx_qkselector_shiftmax",
        "beta": beta,
        "single_active_penalty": 0.10,
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def make_ft5(beta: float) -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(FT5_BASE))
    tag = beta_tag(beta)
    cfg["experiment"] = f"date11full_all_binary_atlif_drtx_{tag}_stdlr_ft_txep19_ft5"
    cfg["note"] = (
        "DATE11 all-binary dual-rail TX FT5: switch existing all-binary+TX ep19 "
        "checkpoint to dual-rail TX attention; 105 binary ATLIF modules; "
        f"opposite-rail penalty beta={beta}; single-active penalty=0.10."
    )
    apply_dualrail_tx(cfg, beta)
    cfg.setdefault("loader", {})["n_epochs"] = 5
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 1, 2, 3, 4]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "ft5_from_tx_ep19",
        "config": str(out),
        "resume": str(TX_EP19_RESUME),
        "attention_mode": "dualrail_binary_tx_qkselector_shiftmax",
        "beta": beta,
        "single_active_penalty": 0.10,
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def main() -> int:
    if not FULL30_BASE.is_file():
        raise FileNotFoundError(f"missing base config: {FULL30_BASE}")
    if not FT5_BASE.is_file():
        raise FileNotFoundError(f"missing base config: {FT5_BASE}")

    rows: list[dict[str, Any]] = []
    for beta in (0.25, 0.50, 1.00):
        for make in (make_full30, make_ft5):
            out, row = make(beta)
            rows.append(row)
            print(out)

    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
