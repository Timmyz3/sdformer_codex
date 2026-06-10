"""Generate NTS10 hardware-friendly H60 configs with expanded target_blocks.

NTS09e (freeze1224) is the best sparse-biased NTS09 line so far. NTS10 keeps the
same qk-threshold freeze policy but sweeps attention replacement scope beyond
S2-only toward S0+S1+S2 (10 blocks) to test whether broader Shiftmax coverage
can improve accuracy without exploding total_spikes.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "nts07b_hw_h60_ffn_update0_act0_s1224.yml"

STAGE_BLOCKS = {
    "s0": ("0:0", "0:1"),
    "s1": ("1:0", "1:1"),
    "s2": ("2:0", "2:1", "2:2", "2:3", "2:4", "2:5"),
    "s3": ("3:0", "3:1"),
}


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def blocks_for(scope: str) -> list[str]:
    if scope == "s2":
        return list(STAGE_BLOCKS["s2"])
    if scope == "s01":
        return list(STAGE_BLOCKS["s0"] + STAGE_BLOCKS["s1"])
    if scope == "s012":
        return list(STAGE_BLOCKS["s0"] + STAGE_BLOCKS["s1"] + STAGE_BLOCKS["s2"])
    if scope == "s23":
        return list(STAGE_BLOCKS["s2"] + STAGE_BLOCKS["s3"])
    raise ValueError(f"unknown scope: {scope}")


def set_runtime(cfg: dict[str, Any], name: str, note: str) -> None:
    cfg["experiment"] = name
    cfg["note"] = note
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 1224
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False


def apply_nts09e_freeze(cfg: dict[str, Any]) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["threshold_eta"] = 6.5e-4
    atlif["threshold_lr_scale"] = 50000.0
    atlif["threshold_freeze_after_step"] = 1224
    atlif["max_threshold"] = 2.0
    atlif["target_rate"] = None
    atlif["target_rate_eta"] = 0.0
    atlif["activity_eta"] = 0.0

    for group in atlif.get("target_groups", []):
        group["threshold_eta"] = 0.0
        group["activity_eta"] = 0.0
        group["target_rate"] = None
        group["target_rate_eta"] = 0.0

    attn = cfg.setdefault("bsa_attention", {})
    attn["mode"] = "h60"
    attn["k_magnitude_alpha"] = 0.0
    attn["mismatch_penalty"] = 0.0
    attn["single_active_penalty"] = 0.0
    attn["target_rate"] = None
    attn["bipolar_mu"] = 0.05
    attn["sc_mu_schedule_enabled"] = True
    attn["sc_mu_start"] = 0.0
    attn["sc_mu_start_step"] = 0
    attn["sc_mu_warmup_steps"] = 720


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    set_runtime(cfg, spec["name"], spec["note"])
    apply_nts09e_freeze(cfg)

    attn = cfg.setdefault("bsa_attention", {})
    attn["target_blocks"] = blocks_for(str(spec["scope"]))

    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "nts10a_hw_h60_s2_freeze1224_s1224",
            "scope": "s2",
            "note": "NTS-10a control: S2-only (6 blocks), same freeze1224 policy as NTS09e.",
        },
        {
            "name": "nts10b_hw_h60_s01_freeze1224_s1224",
            "scope": "s01",
            "note": "NTS-10b: replace S0+S1 attention (4 blocks) with h60 Shiftmax.",
        },
        {
            "name": "nts10c_hw_h60_s012_freeze1224_s1224",
            "scope": "s012",
            "note": "NTS-10c: replace S0+S1+S2 attention (10 blocks) — main expanded-scope candidate.",
        },
        {
            "name": "nts10d_hw_h60_s23_freeze1224_s1224",
            "scope": "s23",
            "note": "NTS-10d: replace S2+S3 attention (8 blocks) — semantic+decoder coverage.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())