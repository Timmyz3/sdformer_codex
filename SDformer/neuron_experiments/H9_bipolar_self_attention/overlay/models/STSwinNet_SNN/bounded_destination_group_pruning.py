"""Evaluation-only bounded destination-group pruning for binary FC1 layers."""

import torch


_STATE_ATTR = "_m288_bounded_destination_group_state"
_OPERATORS = tuple(
    "sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp.fc1".format(
        stage, block)
    for stage, blocks in ((0, 2), (1, 2), (2, 6))
    for block in range(blocks)
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def install_bounded_destination_group_pruning(model, spec):
    """Zero selected FC1 weight groups and return the exact operator list."""
    _require(not hasattr(model, _STATE_ATTR),
             "M288 refuses a stale bounded-group installation")
    group_size = int(spec.get("destination_group_size", -1))
    beta = int(spec.get("maximum_absolute_int8_weight", -1))
    allowed_groups = tuple(int(value) for value in
                           spec.get("allowed_group_sizes", ()))
    allowed_betas = tuple(int(value) for value in
                          spec.get("allowed_betas", ()))
    _require(group_size in allowed_groups and beta in allowed_betas,
             "M288 point is outside the frozen DSE grid")
    _require(group_size in (4, 8), "M288 modified forward limits groups to 4/8")
    modules = dict(model.named_modules())
    state = {
        "schema": "m288_bounded_destination_group_runtime_state_v1",
        "destination_group_size": group_size,
        "maximum_absolute_int8_weight": beta,
        "operator_names": list(_OPERATORS),
        "modules": [],
        "total_weights": 0,
        "removed_weights": 0,
        "total_source_group_pairs": 0,
        "removed_source_group_pairs": 0,
    }
    setattr(model, _STATE_ATTR, state)

    for name in _OPERATORS:
        module = modules.get(name)
        _require(module is not None and module.__class__.__name__ == "Linear",
                 "M288 missing FC1 Linear: " + name)
        _require(module.bias is None and module.weight.ndim == 2,
                 "M288 FC1 geometry/bias drift: " + name)
        output_channels, input_channels = (int(value) for value in
                                           module.weight.shape)
        _require(output_channels == 4 * input_channels and
                 output_channels % group_size == 0,
                 "M288 FC1 expansion/group drift: " + name)
        with torch.no_grad():
            weight = module.weight.detach()
            row_maximum = weight.abs().amax(dim=1)
            scale = torch.where(row_maximum == 0,
                                torch.ones_like(row_maximum),
                                row_maximum / 127.0)
            quantized = torch.clamp(torch.round(weight / scale[:, None]),
                                    -127, 127).to(torch.int16)
            _require(not bool((quantized == -128).any().item()),
                     "M288 quantizer emitted -128")
            groups = output_channels // group_size
            maximum = quantized.abs().reshape(
                groups, group_size, input_channels).amax(dim=1)
            # beta=0 is explicitly the exact floating-point checkpoint path.
            prune_pair = (maximum <= beta if beta > 0 else
                          torch.zeros_like(maximum, dtype=torch.bool))
            prune_weight = prune_pair[:, None, :].expand(
                groups, group_size, input_channels).reshape_as(weight)
            removed_weights = int(prune_weight.sum().item())
            original_l1 = float(weight.abs().sum().item())
            removed_l1 = float(weight.masked_select(prune_weight).abs().sum().item())
            original_l2_squared = float(weight.square().sum().item())
            removed_l2_squared = float(
                weight.masked_select(prune_weight).square().sum().item())
            if beta > 0:
                module.weight.masked_fill_(prune_weight, 0.0)
            _require(bool(torch.isfinite(module.weight).all().item()),
                     "M288 produced non-finite FC1 weight")
        row = {
            "module": name,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "destination_groups": groups,
            "total_source_group_pairs": int(maximum.numel()),
            "removed_source_group_pairs": int(prune_pair.sum().item()),
            "total_weights": int(weight.numel()),
            "removed_weights": removed_weights,
            "removed_float_weight_l1_fraction":
                removed_l1 / original_l1 if original_l1 else 0.0,
            "removed_float_weight_l2_squared_fraction":
                removed_l2_squared / original_l2_squared
                if original_l2_squared else 0.0,
            "maximum_omitted_int8_weight_per_task": beta,
        }
        state["modules"].append(row)
        state["total_weights"] += row["total_weights"]
        state["removed_weights"] += row["removed_weights"]
        state["total_source_group_pairs"] += row["total_source_group_pairs"]
        state["removed_source_group_pairs"] += row["removed_source_group_pairs"]
    return list(_OPERATORS)


def bounded_destination_group_summary(model):
    state = getattr(model, _STATE_ATTR, None)
    _require(state is not None, "M288 summary requested before installation")
    total_weights = int(state["total_weights"])
    total_pairs = int(state["total_source_group_pairs"])
    return {
        "schema": state["schema"],
        "destination_group_size": state["destination_group_size"],
        "maximum_absolute_int8_weight":
            state["maximum_absolute_int8_weight"],
        "operator_names": list(state["operator_names"]),
        "modules": list(state["modules"]),
        "total_weights": total_weights,
        "removed_weights": int(state["removed_weights"]),
        "removed_weight_fraction":
            state["removed_weights"] / float(total_weights),
        "total_source_group_pairs": total_pairs,
        "removed_source_group_pairs":
            int(state["removed_source_group_pairs"]),
        "removed_source_group_pair_fraction":
            state["removed_source_group_pairs"] / float(total_pairs),
    }
