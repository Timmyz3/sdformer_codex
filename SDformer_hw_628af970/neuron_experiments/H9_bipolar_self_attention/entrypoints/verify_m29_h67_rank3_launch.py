#!/usr/bin/env python3
"""Fail-closed preflight for the M29 H67 factor feasibility screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ENTRYPOINTS = Path(__file__).resolve().parent
if str(ENTRYPOINTS) not in sys.path:
    sys.path.insert(0, str(ENTRYPOINTS))

import profile_nts11_hardware_p0 as profiler  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    receipt_path = args.receipt.resolve()
    if (
        not config_path.is_file()
        or not checkpoint_path.is_file()
        or not receipt_path.is_file()
    ):
        raise FileNotFoundError("M29 preflight config/checkpoint/receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_base = Path(receipt.get("base", "")).resolve()
    if (
        receipt.get("schema") != "m29_h67_rank3_factor_config_receipt_v1"
        or receipt.get("status")
        != "READY_FLOATING_FACTOR_AMP_ACCURACY_SCREEN_NOT_INT8_NOT_SPEEDUP"
        or Path(receipt.get("output", "")).resolve() != config_path
        or receipt.get("output_sha256") != sha256(config_path)
        or not receipt_base.is_file()
        or receipt.get("base_sha256") != sha256(receipt_base)
        or Path(receipt.get("checkpoint", "")).resolve() != checkpoint_path
        or receipt.get("checkpoint_sha256") != sha256(checkpoint_path)
        or int(receipt.get("expected_t10_factorized_modules", -1)) != 45
        or int(receipt.get("expected_t2_dense_fallback_modules", -1)) != 60
        or receipt.get("headline_admitted") is not False
    ):
        raise RuntimeError("M29 generated-config receipt identity drift")

    config, _ = profiler.load_config(config_path)
    atlif = config.get("atlif_ternary_psn") or {}
    runtime = config.get("runtime") or {}
    test = config.get("test") or {}
    optimizer = config.get("optimizer") or {}
    if (
        int(atlif.get("temporal_factor_rank", -1)) != 3
        or str(atlif.get("temporal_factor_init")) != "balanced_svd"
        or str(atlif.get("trainable")) != "temporal_factor_atlif"
        or int(config.get("loader", {}).get("n_epochs", -1)) != 5
        or int(runtime.get("epoch_offset", -1)) != 36
        or str(runtime.get("m29_scope"))
        != "floating_factor_valid40_internal_screen_amp_before_int8_qat"
        or int(test.get("sample", -1)) != 40
        or int(test.get("n_valid", -1)) != 1
        or optimizer.get("use_amp") is not True
        or str(runtime.get("m29_source_checkpoint_sha256"))
        != sha256(checkpoint_path)
    ):
        raise RuntimeError("M29 config or checkpoint identity drift")

    model = profiler.build_model(config, checkpoint_path, torch.device("cpu"))
    audit = profiler.validate_h9_load_audit(model, config)
    from models.STSwinNet_SNN.atlif_ternary_psn import (
        apply_trainable_mode,
        atlif_temporal_factor_diagnostics,
    )

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "ATLIFTernaryPSN"
    ]
    requested = [
        (name, module)
        for name, module in modules
        if int(getattr(module, "temporal_factor_requested_rank", 0)) == 3
    ]
    factorized = [
        (name, module)
        for name, module in modules
        if int(getattr(module, "temporal_factor_rank", 0)) == 3
    ]
    fallback = [
        (name, module)
        for name, module in modules
        if int(getattr(module, "temporal_factor_requested_rank", 0)) == 3
        and int(getattr(module, "temporal_factor_rank", 0)) == 0
    ]
    if (
        len(modules) != 105
        or len(requested) != 105
        or len(factorized) != 45
        or len(fallback) != 60
        or any(module.T != 10 for _, module in factorized)
        or any(module.T != 2 for _, module in fallback)
        or any(
            module.temporal_factor_load_source != "balanced_svd_dense"
            for _, module in factorized
        )
        or int(audit.get("missing_count", -1)) != 0
        or int(audit.get("unexpected_count", -1)) != 0
    ):
        raise RuntimeError("M29 105/45/60 model or load census drift")

    trainable = apply_trainable_mode(model, atlif)
    factor_diagnostics = atlif_temporal_factor_diagnostics(model)
    factor_prefixes = tuple(name + "." for name, _ in factorized)
    illegal_trainable = []
    trainable_names = []
    allowed_suffixes = (
        ".temporal_factor_left",
        ".temporal_factor_right",
        ".bias",
        ".thresh",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_names.append(name)
        if not name.startswith(factor_prefixes) or not name.endswith(allowed_suffixes):
            illegal_trainable.append(name)
    if illegal_trainable or len(trainable_names) != 45 * 4:
        raise RuntimeError(
            "M29 trainable scope drift: {}".format(illegal_trainable[:12])
        )

    result = {
        "schema": "m29_h67_rank3_launch_preflight_v1",
        "status": "PASS_FLOATING_FACTOR_VALID40_INTERNAL_SCREEN_PREFLIGHT_NOT_INT8_NOT_ACCURACY_RESULT",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "modules": len(modules),
        "requested_rank3": len(requested),
        "factorized_t10_rank3": len(factorized),
        "dense_fallback_t2": len(fallback),
        "missing": int(audit["missing_count"]),
        "unexpected": int(audit["unexpected_count"]),
        "trainable_tensors": len(trainable_names),
        "trainable_parameters": int(trainable["trainable_parameters"]),
        "expected_trainable_tensors": 45 * 4,
        "validation_scope": "valid40_internal_screen_not_valid825_admission",
        "factor_diagnostics": factor_diagnostics,
        "headline_admitted": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise ValueError("refusing to overwrite M29 preflight: {}".format(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
