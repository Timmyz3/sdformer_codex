"""Generate NTS-11 lite ablation configs.

These configs keep the NTS-11 all12 H60 attention datapath but reduce ATLIF
coverage to isolate the hardware area/accuracy tradeoff:

  - qk_only: ternary Q/K only, no non-QK ATLIF wrappers.
  - qk_downsample: ternary Q/K plus ternary downsample nodes, no binary all_non_qk.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11_two_neuron_only_configs import read_yaml, write_yaml
from make_nts11bd_unified_attn_sweep_configs import (
    DOWNSAMPLE_PATHS,
    NB0,
    RECIPES,
    apply_recipe,
    apply_unified_h60_attention,
    ternary_group,
)


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE_CONFIG = EXP_ROOT / "configs/nts11bd_u12_ds_w720_fastlr_full30_20260613_223042.yml"
MANIFEST = GENERATED / "nts11_lite_ablation_manifest.json"


def apply_lite_scope(cfg: dict[str, Any], *, with_downsample: bool) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["enabled"] = True
    atlif["target"] = "qk"
    atlif["stage_selection"] = "all"
    atlif["output_mode"] = "ternary"
    atlif["threshold_mode"] = "symmetric_bsa_tsn"
    atlif["center_mode"] = "bias"
    atlif.pop("target_paths", None)
    atlif["target_groups"] = (
        [ternary_group("downsample_ternary", DOWNSAMPLE_PATHS)] if with_downsample else []
    )


def make_full_config(spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(BASE_CONFIG))
    cfg["experiment"] = spec["name"] + "_full30"
    cfg["note"] = spec["note"]

    apply_lite_scope(cfg, with_downsample=bool(spec["with_downsample"]))
    apply_recipe(cfg, RECIPES["w720_fastlr"])
    apply_unified_h60_attention(cfg)

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    runtime["use_mlflow_model_logging"] = False
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10

    out = GENERATED / f"{spec['name']}_full30.yml"
    write_yaml(out, cfg)
    manifest = {
        "name": spec["name"],
        "config": str(out),
        "resume": str(NB0),
        "track": "full30",
        "full_epochs": 30,
        "attention": "h60_all12",
        "atlif_expected": 27 if spec["with_downsample"] else 24,
        "shiftmax_expected": 12,
        "scope": spec["scope"],
    }
    return out, manifest


def main() -> int:
    if not NB0.is_file():
        raise FileNotFoundError(f"missing NB0 checkpoint: {NB0}")
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"missing base config: {BASE_CONFIG}")

    specs = [
        {
            "name": "nts11lite_u12_qkonly_w720_fastlr",
            "scope": "ternary_qk_only",
            "with_downsample": False,
            "note": (
                "NTS-11-lite ablation: all12 H60 attention, ternary Q/K only; "
                "no downsample ternary, no sn2q binary, no all_non_qk binary ATLIF."
            ),
        },
        {
            "name": "nts11lite_u12_qkds_w720_fastlr",
            "scope": "ternary_qk_plus_downsample",
            "with_downsample": True,
            "note": (
                "NTS-11-lite ablation: all12 H60 attention, ternary Q/K plus "
                "ternary downsample nodes; no sn2q binary and no all_non_qk binary ATLIF."
            ),
        },
    ]

    GENERATED.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in specs:
        out, row = make_full_config(spec)
        manifest.append(row)
        print(out)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
