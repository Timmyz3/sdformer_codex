#!/usr/bin/env python3
"""Directly audit the first full-resolution Local-5 checkpoint object."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
BASELINE = REPO / "third_party/SDformerFlow"
OVERLAY = EXP / "overlay"
RUN = EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
MODEL = RUN / "checkpoint_epoch9.pth"
IDENTITY = RUN / "training_config_identity.json"
REPORT = RUN / "checkpoint_epoch9_structure_audit.json"

sys.path[:0] = [str(OVERLAY), str(BASELINE), str(REPO)]

from models.STSwinNet_SNN.bsa_attention import (  # noqa: E402
    register_shiftmax_pickle_compat,
)


OVERLAY_MARKERS = (
    ".linear_v.",
    ".bn_v.",
    ".sn_v.",
    "._h9_match_code_weight",
    "._h9_lc4_coefficients",
    "._h9_cf10_beta",
    ".spiking_neuron.thresh",
    ".spiking_neuron.center",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    model_sha = sha256(MODEL)
    identity_sha = sha256(IDENTITY)

    register_shiftmax_pickle_compat()
    model = torch.load(str(MODEL), map_location="cpu", weights_only=False, mmap=True)
    modules = list(model.modules())
    atlif = [module for module in modules if module.__class__.__name__ == "ATLIFTernaryPSN"]
    attentions = [module for module in modules if hasattr(module, "_h9_shiftmax_cfg")]
    state_keys = list(model.state_dict())
    overlay_keys = [
        key for key in state_keys if any(marker in key for marker in OVERLAY_MARKERS)
    ]

    attention_modes = sorted(
        {str(module._h9_shiftmax_cfg.mode) for module in attentions}
    )
    value_branches = sorted(
        {str(module._h9_shiftmax_cfg.value_branch) for module in attentions}
    )
    output_modes = sorted({str(module.output_mode) for module in atlif})
    threshold_modes = sorted({str(module.threshold_mode) for module in atlif})
    center_modes = sorted({str(module.center_mode) for module in atlif})

    facts = {
        "module_counts": {
            "ATLIFTernaryPSN": len(atlif),
            "ShiftmaxAttention": len(attentions),
        },
        "model_state_key_count": len(state_keys),
        "checkpoint_overlay_key_count": len(overlay_keys),
        "attention_modes": attention_modes,
        "attention_value_branches": value_branches,
        "atlif_output_modes": output_modes,
        "atlif_threshold_modes": threshold_modes,
        "atlif_center_modes": center_modes,
        "atlif_symmetric_binary_abs_count": sum(
            bool(getattr(module, "symmetric_binary_abs", False)) for module in atlif
        ),
        "atlif_ternary_output_count": sum(
            getattr(module, "output_mode", None) == "ternary" for module in atlif
        ),
    }
    checks = {
        "training_identity_pass": identity.get("status") == "PASS",
        "training_identity_schema": identity.get("schema")
        == "local5_training_config_identity_v1",
        "identity_model_path": Path(str(identity.get("model_path", ""))).resolve()
        == MODEL.resolve(),
        "identity_model_sha256": identity.get("model_sha256") == model_sha,
        "atlif_count_105": len(atlif) == 105,
        "shiftmax_count_12": len(attentions) == 12,
        "overlay_key_count_210": len(overlay_keys) == 210,
        "all_atlif_binary": output_modes == ["binary"],
        "all_atlif_official": threshold_modes == ["official_atlif"],
        "all_atlif_zero_center": center_modes == ["zero"],
        "no_symmetric_binary_abs": facts["atlif_symmetric_binary_abs_count"] == 0,
        "no_ternary_output": facts["atlif_ternary_output_count"] == 0,
        "all_attention_local5": attention_modes
        == ["binary_axnor_local5_shiftmax"],
        "all_attention_reuse_k": value_branches == ["reuse_k"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "schema": "local5_ep9_checkpoint_structure_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "scope": "direct_full_model_object_structure_not_accuracy_or_final_rank1",
        "model_path": str(MODEL.resolve()),
        "model_sha256": model_sha,
        "training_identity_path": str(IDENTITY.resolve()),
        "training_identity_sha256": identity_sha,
        "facts": facts,
        "checks": checks,
        "failed_checks": failed,
    }
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT)
    print(json.dumps(report, indent=2))
    if failed:
        raise RuntimeError(f"Local-5 ep9 structure audit failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
