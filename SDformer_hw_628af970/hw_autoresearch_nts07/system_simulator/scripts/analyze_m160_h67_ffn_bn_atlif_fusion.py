#!/usr/bin/env python3
"""Audit checkpoint-bound H67 FFN BN/ATLIF fusion and zero-input semantics.

This is a numeric and algebraic milestone, not a cycle-speedup claim.  It
instantiates the frozen ep35 model through the production H67 loader, checks
all twelve MS-spiking MLPs, proves eval-BN folding against PyTorch, and derives
the separable BN1-to-ATLIF temporal affine.  It also executes the complete FFN
on an all-zero input so that a skip contract cannot silently discard BN
offsets or ATLIF responses.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
EXPERIMENT = ROOT / "neuron_experiments/H9_bipolar_self_attention"
PATHS = {
    "checkpoint": HW / "system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth",
    "config": EXPERIMENT / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml",
    "profile_loader": EXPERIMENT / "entrypoints/profile_nts11_hardware_p0.py",
    "atlif_installer": EXPERIMENT / "overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py",
    "atlif_source": EXPERIMENT / "overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py",
    "spiking_modules": ROOT / "third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py",
    "swin_source": ROOT / "third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py",
    "m159": HW / "results/m159_h67_full_ffn_subgraph_scope_r1_20260824/m159_h67_full_ffn_subgraph_scope.json",
    "ffn_ledger": HW / "results/motion_ffn_resident_fusion_opportunity_review_r1_20260824/ffn_pair_ledger.csv",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "checkpoint": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    "config": "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
    "profile_loader": "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    "atlif_installer": "5873063b98eb4a267afa6513d03b86621f3fb6a885b310b4c5569ef5448ae657",
    "atlif_source": "d9ee7e172f941a53ad1c031b0d5cdbbf7819f521c807e5bc54001a80c41b57f3",
    "spiking_modules": "130e32bba2c0bc23a4da091087d09b71cb50e9381ed2d1c25ff6b447bce52948",
    "swin_source": "8d551eac17e2272813dd0238ea4aee2e84cfe0b7e9435a1407d0787fb5407768",
    "m159": "6c67a75d052080cf58e558f960f23bea64d841087967de044fef898ad46c7f89",
    "ffn_ledger": "dcf183e930372253da96c6ce242289e3e6a5e1b0f76a513e095fae4b0d2ae128",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
FFN_RE = re.compile(
    r"^sttmultires_unet\.encoders\.swin3d\.layers\.(\d+)\.swin_blocks\.(\d+)\.mlp$")
BN1_ELEMENTS_PER_FRAME = 350_208_000
BN2_ELEMENTS_PER_FRAME = 87_552_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out

    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + value)),
        )


def finite(value, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), label + " is non-finite")
    return result


def summary(values) -> dict:
    clean = sorted(finite(value, "summary value") for value in values)
    require(clean, "empty summary")

    def nearest(fraction):
        return clean[max(0, math.ceil(fraction * len(clean)) - 1)]

    return {
        "count": len(clean),
        "minimum": clean[0],
        "mean": math.fsum(clean) / len(clean),
        "p50_nearest_rank": nearest(0.50),
        "p95_nearest_rank": nearest(0.95),
        "maximum": clean[-1],
    }


def load_profile_module():
    entrypoints = EXPERIMENT / "entrypoints"
    overlay = EXPERIMENT / "overlay"
    baseline = ROOT / "third_party/SDformerFlow"
    for path in reversed((str(entrypoints), str(overlay), str(baseline))):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("m160_profile_loader", PATHS["profile_loader"])
    require(spec is not None and spec.loader is not None, "cannot import H67 loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bn_affine(batch_norm):
    require(not batch_norm.training, "BN must be in eval mode")
    require(batch_norm.track_running_stats, "BN running stats disabled")
    tensors = (
        batch_norm.weight.detach().cpu().to(torch.float64),
        batch_norm.bias.detach().cpu().to(torch.float64),
        batch_norm.running_mean.detach().cpu().to(torch.float64),
        batch_norm.running_var.detach().cpu().to(torch.float64),
    )
    require(all(bool(torch.isfinite(item).all()) for item in tensors),
            "BN parameter contains NaN/Infinity")
    gamma, beta, mean, variance = tensors
    require(bool((variance >= 0).all()), "negative BN variance")
    alpha = gamma / torch.sqrt(variance + float(batch_norm.eps))
    offset = beta - alpha * mean
    return alpha, offset


def quantize_rows(weight: torch.Tensor):
    value = weight.detach().cpu().to(torch.float64)
    maximum = value.abs().amax(dim=1)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127.0)
    quantized = torch.clamp(torch.round(value / scale[:, None]), -127, 127).to(torch.int64)
    require(not bool((quantized == -128).any()), "reserved INT8 -128 emitted")
    return scale, quantized


def fold_miter(linear, batch_norm, alpha, offset, seed: int) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    features = int(linear.weight.shape[1])
    inputs = torch.randn((37, features), generator=generator, dtype=torch.float32)
    with torch.no_grad():
        raw = torch_functional.linear(inputs, linear.weight.detach().cpu(), None)
        # The production SpikingJelly BN is in multi-step mode and therefore
        # requires [T,N,C,H,W], even though eval statistics are time-invariant.
        reference = batch_norm(
            raw[:, None, :, None, None]).squeeze(1).squeeze(-1).squeeze(-1)
        folded_weight = linear.weight.detach().cpu().to(torch.float64) * alpha[:, None]
        folded = torch_functional.linear(
            inputs.to(torch.float64), folded_weight, offset)
    error = (reference.to(torch.float64) - folded).abs().max().item()
    require(error <= 2.0e-5, "Linear/BN fold miter exceeded tolerance")
    return finite(error, "fold miter error")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(not output.exists(), "refusing to overwrite M160 output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M160 frozen input identity drift")

    m159 = strict_json(PATHS["m159"])
    require(m159["resolved_topology"]["pairs"] == 12, "M159 FFN population drift")
    require(m159["unmodeled_elementwise_extent_per_frame"]["bn1_elements"] ==
            BN1_ELEMENTS_PER_FRAME, "M159 BN1 extent drift")
    require(m159["unmodeled_elementwise_extent_per_frame"]["bn2_elements"] ==
            BN2_ELEMENTS_PER_FRAME, "M159 BN2 extent drift")
    with PATHS["ffn_ledger"].open(newline="", encoding="utf-8") as handle:
        ledger = {row["pair_id"]: row for row in csv.DictReader(handle)}
    require(len(ledger) == 12, "FFN pair ledger population drift")

    profile = load_profile_module()
    config, _ = profile.load_config(PATHS["config"])
    model = profile.build_model(config, PATHS["checkpoint"], torch.device("cpu"))
    model.eval()
    modules = dict(model.named_modules())
    ffn_names = sorted(name for name in modules if FFN_RE.match(name))
    require(len(ffn_names) == 12 and set(ffn_names) == set(ledger),
            "runtime FFN population/identity drift")

    rows = []
    all_alpha1 = []
    all_offset1 = []
    all_alpha2 = []
    all_offset2 = []
    all_gain1 = []
    all_gain2 = []
    all_sn1_thresholds = []
    all_sn2_thresholds = []
    fold_errors = []
    total_expanded = 0
    total_output = 0
    total_zero_sn1_active = 0
    total_zero_fc1_acc_sn2_active = 0
    total_full_zero_sn2_active = 0
    total_zero_branch_nonzero = 0
    total_zero_branch_values = 0
    total_materialized_temporal_bias = 0
    total_factored_temporal_constants = 0

    for module_index, name in enumerate(ffn_names):
        match = FFN_RE.match(name)
        require(match is not None, "bad FFN name")
        stage, block = map(int, match.groups())
        mlp = modules[name]
        require(mlp.__class__.__name__ == "MS_Spiking_Mlp", "MLP type drift")
        require(mlp.fc1.bias is None and mlp.fc2.bias is None,
                "FFN Linear bias drift")
        require(float(mlp.drop1.p) == 0.0 and float(mlp.drop2.p) == 0.0,
                "FFN dropout drift")
        input_channels = int(mlp.fc1.weight.shape[1])
        expanded_channels = int(mlp.fc1.weight.shape[0])
        require(expanded_channels == 4 * input_channels, "MLP ratio drift")
        require(tuple(mlp.fc2.weight.shape) == (input_channels, expanded_channels),
                "fc2 geometry drift")
        require(int(ledger[name]["input_channels"]) == input_channels and
                int(ledger[name]["expanded_channels"]) == expanded_channels,
                "checkpoint/ledger geometry drift")

        bn1 = mlp.bn1.norm_layer
        bn2 = mlp.bn2.norm_layer
        require(bn1.__class__.__name__ == "BatchNorm2d" and
                bn2.__class__.__name__ == "BatchNorm2d", "BN type drift")
        alpha1, offset1 = bn_affine(bn1)
        alpha2, offset2 = bn_affine(bn2)
        require(alpha1.numel() == expanded_channels and
                alpha2.numel() == input_channels, "BN channel geometry drift")

        sn1 = mlp.sn1.spiking_neuron
        sn2 = mlp.sn2.spiking_neuron
        for label, neuron in (("sn1", sn1), ("sn2", sn2)):
            require(neuron.__class__.__name__ == "ATLIFTernaryPSN",
                    label + " overlay type drift")
            require(int(neuron.T) == 10 and tuple(neuron.weight.shape) == (10, 10),
                    label + " temporal geometry drift")
            require(neuron.output_mode == "binary" and
                    neuron.threshold_mode == "official_atlif" and
                    neuron.center_mode == "zero", label + " mode drift")
            require(math.isfinite(float(neuron.thresh.detach().cpu())) and
                    float(neuron.thresh.detach().cpu()) > 0.0,
                    label + " threshold must be finite and positive")

        scale1, qweight1 = quantize_rows(mlp.fc1.weight)
        scale2, qweight2 = quantize_rows(mlp.fc2.weight)
        gain1 = alpha1 * scale1
        gain2 = alpha2 * scale2
        require(bool(torch.isfinite(gain1).all()) and bool(torch.isfinite(gain2).all()),
                "folded gain is non-finite")
        sumabs1 = qweight1.abs().sum(dim=1)
        sumabs2 = qweight2.abs().sum(dim=1)

        error1 = fold_miter(mlp.fc1, bn1, alpha1, offset1,
                            16000 + module_index * 2)
        error2 = fold_miter(mlp.fc2, bn2, alpha2, offset2,
                            16001 + module_index * 2)
        fold_errors.extend((error1, error2))

        temporal_weight2 = sn2.weight.detach().cpu().to(torch.float64)
        temporal_bias2 = sn2.bias.detach().cpu().to(torch.float64).reshape(10)
        temporal_row_sum2 = temporal_weight2.sum(dim=1)
        folded_temporal_bias = (
            temporal_row_sum2[:, None] * offset1[None, :] +
            temporal_bias2[:, None]
        )
        zero_sn2_formula = folded_temporal_bias >= float(sn2.thresh.detach().cpu())

        zero_input = torch.zeros(
            (10, 1, 1, 1, input_channels), dtype=torch.float32)
        with torch.no_grad():
            sn1_out = mlp.sn1(zero_input)
            fc1_out = mlp.fc1(sn1_out)
            bn1_out = mlp.bn1(
                fc1_out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
            sn2_out = mlp.sn2(bn1_out)
            zero_fc1_bn1 = offset1.to(torch.float32).reshape(
                1, 1, 1, 1, expanded_channels).expand(10, 1, 1, 1, -1)
            zero_fc1_sn2_out = mlp.sn2(zero_fc1_bn1)
            fc2_out = mlp.fc2(sn2_out)
            bn2_out = mlp.bn2(
                fc2_out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
            full_out = mlp(zero_input)
        zero_fc1_sn2_observed = zero_fc1_sn2_out.reshape(
            10, expanded_channels).ne(0).cpu()
        require(torch.equal(zero_fc1_sn2_observed, zero_sn2_formula.cpu()),
                "BN1-to-sn2 zero-input formula mismatch")
        require(torch.allclose(full_out, bn2_out, rtol=0.0, atol=0.0),
                "manual/full FFN zero-input mismatch")

        zero_sn1_active = int(sn1_out.ne(0).sum().item())
        zero_fc1_acc_sn2_active = int(zero_fc1_sn2_observed.sum().item())
        zero_fc1_acc_sn2_channels = int(
            zero_fc1_sn2_observed.any(dim=0).sum().item())
        full_zero_sn2_active = int(sn2_out.ne(0).sum().item())
        zero_branch_nonzero = int(full_out.ne(0).sum().item())
        zero_branch_values = int(full_out.numel())
        total_zero_sn1_active += zero_sn1_active
        total_zero_fc1_acc_sn2_active += zero_fc1_acc_sn2_active
        total_full_zero_sn2_active += full_zero_sn2_active
        total_zero_branch_nonzero += zero_branch_nonzero
        total_zero_branch_values += zero_branch_values
        total_expanded += expanded_channels
        total_output += input_channels
        total_materialized_temporal_bias += 10 * expanded_channels
        total_factored_temporal_constants += expanded_channels + 20

        all_alpha1.extend(alpha1.tolist())
        all_offset1.extend(offset1.tolist())
        all_alpha2.extend(alpha2.tolist())
        all_offset2.extend(offset2.tolist())
        all_gain1.extend(gain1.tolist())
        all_gain2.extend(gain2.tolist())
        all_sn1_thresholds.append(float(sn1.thresh.detach().cpu()))
        all_sn2_thresholds.append(float(sn2.thresh.detach().cpu()))
        rows.append({
            "module": name,
            "stage": stage,
            "block": block,
            "input_channels": input_channels,
            "expanded_channels": expanded_channels,
            "bn1_alpha_min": float(alpha1.min()),
            "bn1_alpha_max": float(alpha1.max()),
            "bn1_offset_min": float(offset1.min()),
            "bn1_offset_max": float(offset1.max()),
            "bn1_exact_zero_offsets": int((offset1 == 0).sum()),
            "bn2_alpha_min": float(alpha2.min()),
            "bn2_alpha_max": float(alpha2.max()),
            "bn2_offset_min": float(offset2.min()),
            "bn2_offset_max": float(offset2.max()),
            "bn2_exact_zero_offsets": int((offset2 == 0).sum()),
            "fc1_bn1_fold_max_abs_error": error1,
            "fc2_bn2_fold_max_abs_error": error2,
            "fc1_int8_sumabs_max": int(sumabs1.max()),
            "fc2_int8_sumabs_max": int(sumabs2.max()),
            "sn1_threshold": float(sn1.thresh.detach().cpu()),
            "sn2_threshold": float(sn2.thresh.detach().cpu()),
            "zero_input_sn1_active_values": zero_sn1_active,
            "zero_fc1_acc_sn2_active_values": zero_fc1_acc_sn2_active,
            "zero_fc1_acc_sn2_active_channels": zero_fc1_acc_sn2_channels,
            "full_zero_input_sn2_active_values": full_zero_sn2_active,
            "zero_input_branch_nonzero_values": zero_branch_nonzero,
            "zero_input_branch_values": zero_branch_values,
            "zero_input_branch_max_abs": float(full_out.abs().max()),
        })

    require(total_expanded == 17_664 and total_output == 4_416,
            "aggregate FFN channel population drift")
    require(total_materialized_temporal_bias == 176_640 and
            total_factored_temporal_constants == 17_904,
            "temporal affine storage arithmetic drift")
    require(all(value > 0 for value in all_alpha1 + all_alpha2),
            "negative/zero BN scale requires a different sign contract")
    require(sum(row["bn1_exact_zero_offsets"] for row in rows) == 0 and
            sum(row["bn2_exact_zero_offsets"] for row in rows) == 0,
            "unexpected exact-zero BN offset")

    output.mkdir(parents=True)
    csv_path = output / "per_ffn_bn_atlif_fusion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "m160_h67_ffn_bn_atlif_fusion_v1",
        "status": "PASS_CHECKPOINT_BOUND_BN_FOLD_AND_ZERO_PATH_AUDIT",
        "identity": {
            "analyzer_start_end_sha256": script_start,
            "inputs_sha256": observed,
            "runtime_model": "MS_SpikingformerFlowNet_en4/ep35/eval",
            "ffn_modules": len(rows),
        },
        "full_ffn_boundary": {
            "source_order": [
                "sn1_ATLIF", "dropout1_p0", "fc1_bias_free", "BN1_eval",
                "sn2_ATLIF", "dropout2_p0", "fc2_bias_free", "BN2_eval",
                "drop_path_eval_off", "residual_add",
            ],
            "why_fc1_fc2_was_incomplete": (
                "fc1/fc2 named only the matrix-engine ledger.  Correct pruning and "
                "fusion semantics include both ATLIFs, both BNs, and residual commit."
            ),
        },
        "checkpoint_census": {
            "expanded_channels_across_12_blocks": total_expanded,
            "output_channels_across_12_blocks": total_output,
            "bn1_alpha": summary(all_alpha1),
            "bn1_offset": summary(all_offset1),
            "bn2_alpha": summary(all_alpha2),
            "bn2_offset": summary(all_offset2),
            "bn1_exact_zero_offsets": sum(row["bn1_exact_zero_offsets"] for row in rows),
            "bn2_exact_zero_offsets": sum(row["bn2_exact_zero_offsets"] for row in rows),
            "all_bn_scales_strictly_positive": True,
            "maximum_float64_fold_miter_abs_error": max(fold_errors),
            "fc1_folded_int8_dequant_gain": summary(all_gain1),
            "fc2_folded_int8_dequant_gain": summary(all_gain2),
            "sn1_threshold": summary(all_sn1_thresholds),
            "sn2_threshold": summary(all_sn2_thresholds),
            "maximum_fc1_int8_binary_input_sumabs_bound": max(
                row["fc1_int8_sumabs_max"] for row in rows),
            "maximum_fc2_int8_binary_input_sumabs_bound": max(
                row["fc2_int8_sumabs_max"] for row in rows),
        },
        "zero_input_semantics": {
            "sn1_active_values": total_zero_sn1_active,
            "sn2_active_values_after_zero_fc1_acc_and_bn1_offset":
                total_zero_fc1_acc_sn2_active,
            "sn2_active_values_on_full_zero_mlp_input": total_full_zero_sn2_active,
            "full_branch_nonzero_values": total_zero_branch_nonzero,
            "full_branch_values": total_zero_branch_values,
            "full_branch_nonzero_fraction": (
                float(total_zero_branch_nonzero) / total_zero_branch_values),
            "decision": (
                "An all-zero FFN input does not authorize dropping the branch.  "
                "BN1 offsets may trigger sn2 and every BN2 offset is nonzero; a skip "
                "implementation must synthesize the proven constant/trigger response."
            ),
        },
        "fusion_candidate": {
            "bn1_sn2_exact_algebra": (
                "h[t,j] = (alpha1[j]*weight_scale1[j]) * "
                "sum_tau(Wt[t,tau]*acc_fc1[tau,j]) + "
                "offset1[j]*sum_tau(Wt[t,tau]) + bias_sn2[t]"
            ),
            "bn1_temporal_bias_materialized_values": total_materialized_temporal_bias,
            "bn1_temporal_affine_factored_values": total_factored_temporal_constants,
            "materialized_to_factored_value_ratio": (
                float(total_materialized_temporal_bias) /
                total_factored_temporal_constants),
            "bn2_commit_exact_algebra": (
                "output[j] = (alpha2[j]*weight_scale2[j])*acc_fc2[j] + "
                "offset2[j] + residual[j]"
            ),
            "standalone_bn_elements_per_frame_eligible_for_no_materialization":
                BN1_ELEMENTS_PER_FRAME + BN2_ELEMENTS_PER_FRAME,
            "residual_elements_per_frame_remain": BN2_ELEMENTS_PER_FRAME,
            "hardware_mapping": (
                "Time-share the existing Linear and ATLIF engines.  Add a lane-local "
                "requant/threshold affine and fuse BN2 into residual commit; do not "
                "instantiate a second MAC pool or a standalone BN traversal."
            ),
        },
        "algorithm_feedback": {
            "paired_expanded_channel_mask": [
                "fc1 output row", "BN1 channel", "sn2 channel state/parameters",
                "fc2 input column",
            ],
            "training_requirement": (
                "PAFT/structured training must mask the complete expanded-channel "
                "tuple and preserve the folded BN1-to-sn2 response; weight-only masks "
                "are not a hardware skip contract."
            ),
            "priority": "Stage 2 remains the first FFN optimization target from M159.",
        },
        "admission": {
            "checkpoint_bound_numeric": True,
            "full_ffn_zero_path_executed": True,
            "bn_fold_algebra_proved": True,
            "bn_materialization_elision_rtl": False,
            "bn_materialization_elision_vcs": False,
            "cycle_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "next_gate": (
                "Build the lane-local folded-affine/threshold and residual-commit RTL, "
                "prove it against exported vectors in VCS, then add explicit port and "
                "overlap recurrence before assigning cycles."
            ),
        },
    }
    result_path = output / "m160_h67_ffn_bn_atlif_fusion.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(sha256(script_path) == script_start, "M160 analyzer changed during run")
    print(json.dumps({
        "status": payload["status"],
        "modules": len(rows),
        "fold_error_max": max(fold_errors),
        "zero_fc1_acc_sn2_active": total_zero_fc1_acc_sn2_active,
        "full_zero_sn2_active": total_full_zero_sn2_active,
        "zero_branch_nonzero_fraction": payload["zero_input_semantics"]["full_branch_nonzero_fraction"],
        "bn_elements_no_materialization_candidate": BN1_ELEMENTS_PER_FRAME + BN2_ELEMENTS_PER_FRAME,
        "temporal_bias_storage_ratio": payload["fusion_candidate"]["materialized_to_factored_value_ratio"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
