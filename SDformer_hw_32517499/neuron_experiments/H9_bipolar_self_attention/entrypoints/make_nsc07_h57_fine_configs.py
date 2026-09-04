"""Generate NSC-07 fine H57 residual sweep configs.

This sweep follows the NSC-06 evidence: full-scope small SC residual improved
the H57 mu=0 control slightly, while stage2-only stronger residuals hurt AEE.
Keep the carrier-preserving H57 form and only sweep small all-scope mu values.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "ntx04h_cptc_ntx01_warm.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def make_config(base: dict[str, Any], name: str, mu: float, lam: float) -> Path:
    cfg = deepcopy(base)
    cfg["experiment"] = name

    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = "tx_sc_residual_selector_shiftmax"
    attn["stage_selection"] = "all"
    attn.pop("target_blocks", None)
    attn["bipolar_mu"] = mu
    attn["bipolar_lambda"] = lam
    attn["mismatch_penalty"] = 0.25
    attn["single_active_penalty"] = 0.0
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["confidence_enabled"] = False
    attn["k_consistency_mod"] = False
    attn["bipolar_gate_min"] = None
    attn["bipolar_gate_max"] = None

    opt = cfg.setdefault("optimizer", {})
    opt["lr_warmup"] = {"enabled": True, "steps": 450, "start_factor": 0.05}
    opt["milestones"] = [22, 27]
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = 1.0e-6
    groups["norm_lr"] = 1.0e-6
    groups["neuron_lr"] = 3.0e-5
    groups["threshold_lr"] = 5.0e-6
    cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = 5.0e-6

    cfg["note"] = (
        "NSC-07 H57 fine sweep: carrier-preserving TX attention with small "
        f"SC agree/disagree residual, all scope, mu={mu}, lambda={lam}. "
        "Generated after NSC-06 showed only all-scope low-mu residual had "
        "a small positive short-test signal."
    )

    out = GENERATED / f"{name}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs = [
        ("nsc07a_h57_all_mu003_l03", 0.03, 0.30),
        ("nsc07b_h57_all_mu005_l03", 0.05, 0.30),
        ("nsc07c_h57_all_mu008_l03", 0.08, 0.30),
        ("nsc07d_h57_all_mu012_l03", 0.12, 0.30),
        ("nsc07e_h57_all_mu010_l02", 0.10, 0.20),
        ("nsc07f_h57_all_mu010_l04", 0.10, 0.40),
    ]
    for name, mu, lam in specs:
        print(make_config(base, name, mu, lam).relative_to(EXP_ROOT / "configs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
