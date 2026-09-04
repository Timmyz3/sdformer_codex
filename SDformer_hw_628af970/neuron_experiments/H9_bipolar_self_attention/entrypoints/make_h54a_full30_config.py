"""Generate H54a full30 config from best short-sweep result.

Converts a short-test config to full30, switching neuron mode from
symmetric_bsa_tsn (free-floating threshold) to symmetric_target_rate
(H49-proven stable recipe) while preserving per-stage threshold_eta.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def make_full30(short_config_path: str, lam: float, lr_strategy: str) -> str:
    base = load_yaml(Path(short_config_path))

    lam_str = str(lam).replace(".", "p")
    name = f"h54a_lam{lam_str}_{lr_strategy}_full30"
    base["experiment"] = name
    base["runtime"]["max_train_steps"] = 0
    base["loader"]["n_epochs"] = 30
    base["runtime"]["force_save_epochs"] = list(range(30))
    base["note"] = (
        f"H54a full30: lambda={lam}, LR={lr_strategy}. "
        "Neuron: symmetric_target_rate tr=0.07 for full30 stability. "
        + base.get("note", "")
    )

    # Switch neuron to symmetric_target_rate for full30 stability
    atlif = base.setdefault("atlif_ternary_psn", {})
    atlif["threshold_mode"] = "symmetric_target_rate"
    atlif["target_rate"] = 0.07
    atlif["target_rate_eta"] = 0.08
    atlif["target"] = "qk"

    for grp in atlif.get("target_groups", []):
        gname = grp.get("name", "")
        if gname.startswith("qk"):
            grp["threshold_mode"] = "symmetric_target_rate"
            grp["target_rate"] = 0.07
            grp["target_rate_eta"] = 0.06

    out_path = CONFIG_DIR / "generated" / f"{name}.yml"
    dump_yaml(out_path, base)
    print(f"wrote {out_path}")
    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: make_h54a_full30_config.py <short_config_path> <lambda> <lr_strategy>")
        sys.exit(1)
    make_full30(sys.argv[1], float(sys.argv[2]), sys.argv[3])
