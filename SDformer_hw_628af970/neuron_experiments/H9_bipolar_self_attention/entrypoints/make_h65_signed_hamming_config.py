"""Generate the all105/all12 H65 signed Hamming DSEC config."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
BASE = GEN / "h63_direct_shiftmax_stc_symtern_s120.yml"
OUTPUT = GEN / "h65_all105_symtern_signed_hamming_s20.yml"
MANIFEST = GEN / "h65_signed_hamming_manifest.json"
CHECKPOINT = (
    EXP_ROOT
    / "results/date11full_all_ternary_atlif_tx_w720_fastlr_full30_bs8_20260616_022014_setsid"
    / "checkpoint_epoch29.pth"
)


def main() -> int:
    cfg = deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    cfg["experiment"] = "h65_all105_symtern_signed_hamming_s20"
    cfg["bsa_attention"].update(
        {
            "mode": "hamming_ternary_active_direct",
            "value_mode": "threshold",
            "bipolar_mu": 0.0,
            "k_magnitude_alpha": 0.0,
            "sc_mu_schedule_enabled": False,
            "target_rate": None,
        }
    )
    cfg["runtime"]["max_train_steps"] = 20
    cfg["note"] = (
        "H65 all105 symmetric ternary ATLIF + all12 signed Hamming linear attention. "
        "No Shiftmax, gate*K, SC, Kmag, target-rate, stage mixture, or partial replacement."
    )
    OUTPUT.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    MANIFEST.write_text(
        json.dumps(
            {
                "name": cfg["experiment"],
                "config": str(OUTPUT),
                "checkpoint": str(CHECKPOINT),
                "atlif_expected": 105,
                "attention_expected": 12,
                "attention_mode": "hamming_ternary_active_direct",
                "uses_gate_k": False,
                "uses_k_value": True,
                "status": "stopped_activity_gate_20step",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
