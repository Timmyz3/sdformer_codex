"""Evaluation-only shared bounded-group pruning for FC1 and patch Conv.

The mask is derived from symmetric per-output-row INT8 weights.  A source is
removed for one consecutive destination group only when every quantized
weight in that group has magnitude no greater than ``beta``.  ``beta=0`` is
an explicit exact floating-point checkpoint path and changes no weight.
"""

import hashlib

import torch


_STATE_ATTR = "_m301_shared_fc1_patch_group_state"
_FC1_OPERATORS = tuple(
    "sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp.fc1".format(
        stage, block)
    for stage, blocks in ((0, 2), (1, 2), (2, 6))
    for block in range(blocks)
)
_PATCH_CONV_OPERATORS = (
    "sttmultires_unet.encoders.swin3d.patch_embed.conv.conv.0",
    "sttmultires_unet.encoders.swin3d.patch_embed.proj.conv",
    "sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv1.0",
    "sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0",
    "sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.1.conv1.0",
    "sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.1.conv2.0",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _weight_sha256(weight):
    payload = weight.detach().cpu().contiguous().numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _mask_one(module, name, kind, group_size, beta):
    weight = module.weight
    _require(module.bias is None and weight.ndim in (2, 4),
             "M301 geometry/bias drift: " + name)
    output_channels = int(weight.shape[0])
    _require(output_channels % group_size == 0,
             "M301 destination group does not divide output: " + name)
    if kind == "fc1":
        _require(module.__class__.__name__ == "Linear" and weight.ndim == 2 and
                 output_channels == 4 * int(weight.shape[1]),
                 "M301 FC1 identity drift: " + name)
    else:
        _require(module.__class__.__name__ == "Conv2d" and weight.ndim == 4 and
                 output_channels == 96 and tuple(weight.shape[2:]) == (3, 3),
                 "M301 patch Conv identity drift: " + name)

    with torch.no_grad():
        flat = weight.detach().reshape(output_channels, -1)
        row_maximum = flat.abs().amax(dim=1)
        scale = torch.where(row_maximum == 0, torch.ones_like(row_maximum),
                            row_maximum / 127.0)
        quantized = torch.clamp(torch.round(flat / scale[:, None]),
                                -127, 127).to(torch.int16)
        _require(not bool((quantized == -128).any().item()),
                 "M301 quantizer emitted -128")
        source_count = int(flat.shape[1])
        group_count = output_channels // group_size
        maximum = quantized.abs().reshape(
            group_count, group_size, source_count).amax(dim=1)
        prune_pair = (maximum <= beta if beta > 0 else
                      torch.zeros_like(maximum, dtype=torch.bool))
        prune_flat = prune_pair[:, None, :].expand(
            group_count, group_size, source_count).reshape_as(flat)
        prune_weight = prune_flat.reshape_as(weight)
        removed_weights = int(prune_weight.sum().item())
        original_l1 = float(weight.abs().sum().item())
        removed_l1 = float(
            weight.masked_select(prune_weight).abs().sum().item())
        original_l2_squared = float(weight.square().sum().item())
        removed_l2_squared = float(
            weight.masked_select(prune_weight).square().sum().item())
        removed_l1_per_destination = (
            flat.abs() * prune_flat.to(dtype=flat.dtype)).sum(dim=1)
        local_output_linf_bound = float(
            removed_l1_per_destination.max().item())
        mean_output_linf_bound = float(
            removed_l1_per_destination.mean().item())
        if beta > 0:
            module.weight.masked_fill_(prune_weight, 0.0)
        _require(bool(torch.isfinite(module.weight).all().item()),
                 "M301 produced non-finite weight")
        post_install_removed_weight_nonzero = int(
            torch.count_nonzero(module.weight.masked_select(prune_weight)).item())
        _require(post_install_removed_weight_nonzero == 0,
                 "M301 omitted weight remained nonzero after installation")
        post_install_weight_sha256 = _weight_sha256(module.weight)

    return {
        "module": name,
        "kind": kind,
        "input_sources": source_count,
        "output_channels": output_channels,
        "destination_groups": group_count,
        "total_source_group_pairs": int(maximum.numel()),
        "removed_source_group_pairs": int(prune_pair.sum().item()),
        "total_weights": int(weight.numel()),
        "removed_weights": removed_weights,
        "removed_float_weight_l1_fraction":
            removed_l1 / original_l1 if original_l1 else 0.0,
        "removed_float_weight_l2_squared_fraction":
            removed_l2_squared / original_l2_squared
            if original_l2_squared else 0.0,
        "local_binary_input_output_linf_bound": local_output_linf_bound,
        "mean_per_destination_binary_input_output_linf_bound":
            mean_output_linf_bound,
        "local_bound_semantics": "for |x_i|<=1, each destination output perturbation is at most the L1 sum of its omitted floating-point weights; this is a local layer-output bound, not an end-to-end AEE bound",
        "maximum_omitted_int8_weight_per_task": beta,
        "post_install_removed_weight_nonzero":
            post_install_removed_weight_nonzero,
        "post_install_weight_sha256": post_install_weight_sha256,
    }


def install_shared_fc1_patch_group_pruning(model, spec):
    """Install the frozen FC1+patch-Conv mask and return operator names."""
    _require(not hasattr(model, _STATE_ATTR),
             "M301 refuses a stale shared-group installation")
    group_size = int(spec.get("destination_group_size", -1))
    beta = int(spec.get("maximum_absolute_int8_weight", -1))
    allowed_groups = tuple(int(value) for value in
                           spec.get("allowed_group_sizes", ()))
    allowed_betas = tuple(int(value) for value in
                          spec.get("allowed_betas", ()))
    _require(group_size in allowed_groups and beta in allowed_betas,
             "M301 point is outside the frozen DSE grid")
    _require(group_size == 4,
             "M301 S10 screen is frozen to four-output groups")

    modules = dict(model.named_modules())
    state = {
        "schema": "m301_shared_fc1_patch_group_runtime_state_v1",
        "destination_group_size": group_size,
        "maximum_absolute_int8_weight": beta,
        "operator_names": list(_FC1_OPERATORS + _PATCH_CONV_OPERATORS),
        "fc1_operator_names": list(_FC1_OPERATORS),
        "patch_conv_operator_names": list(_PATCH_CONV_OPERATORS),
        "modules": [],
    }
    setattr(model, _STATE_ATTR, state)
    for kind, names in (("fc1", _FC1_OPERATORS),
                        ("patch_conv", _PATCH_CONV_OPERATORS)):
        for name in names:
            module = modules.get(name)
            _require(module is not None, "M301 missing module: " + name)
            state["modules"].append(
                _mask_one(module, name, kind, group_size, beta))
    return list(state["operator_names"])


def shared_fc1_patch_group_summary(model):
    state = getattr(model, _STATE_ATTR, None)
    _require(state is not None, "M301 summary requested before installation")
    named_modules = dict(model.named_modules())
    audited_modules = []
    for row in state["modules"]:
        module = named_modules.get(row["module"])
        _require(module is not None,
                 "M301 post-evaluation module disappeared: " + row["module"])
        audited = dict(row)
        audited["post_evaluation_weight_sha256"] = _weight_sha256(module.weight)
        audited["weight_sha256_stable_through_evaluation"] = (
            audited["post_evaluation_weight_sha256"] ==
            audited["post_install_weight_sha256"])
        _require(audited["weight_sha256_stable_through_evaluation"],
                 "M301 weight changed after post-checkpoint installation: " +
                 row["module"])
        audited_modules.append(audited)

    summaries = {}
    for kind in ("fc1", "patch_conv", "all"):
        rows = (state["modules"] if kind == "all" else
                [row for row in state["modules"] if row["kind"] == kind])
        total_weights = sum(int(row["total_weights"]) for row in rows)
        removed_weights = sum(int(row["removed_weights"]) for row in rows)
        total_pairs = sum(int(row["total_source_group_pairs"]) for row in rows)
        removed_pairs = sum(int(row["removed_source_group_pairs"]) for row in rows)
        summaries[kind] = {
            "modules": len(rows),
            "total_weights": total_weights,
            "removed_weights": removed_weights,
            "removed_weight_fraction":
                removed_weights / float(total_weights),
            "total_source_group_pairs": total_pairs,
            "removed_source_group_pairs": removed_pairs,
            "removed_source_group_pair_fraction":
                removed_pairs / float(total_pairs),
            "one_bit_mask_metadata_bits": total_pairs,
            "one_bit_mask_metadata_bytes": (total_pairs + 7) // 8,
            "mask_bytes_over_int8_weight_bytes":
                ((total_pairs + 7) // 8) / float(total_weights),
        }
    return {
        "schema": state["schema"],
        "destination_group_size": state["destination_group_size"],
        "maximum_absolute_int8_weight":
            state["maximum_absolute_int8_weight"],
        "operator_names": list(state["operator_names"]),
        "fc1_operator_names": list(state["fc1_operator_names"]),
        "patch_conv_operator_names": list(state["patch_conv_operator_names"]),
        "summaries": summaries,
        "modules": audited_modules,
    }
