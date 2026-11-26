"""Inference-only regular spatial mean/MLP/repeat mechanism adaptation.

This is not DGE/ToMe/VidToMe reproduction. Fixed whole-tile decisions are supplied
by the caller; there is no learned or motion gate. Both producers share the
unchanged native MLP. This module returns only its branch, never the residual.
"""
import contextlib
import hashlib
import math

import torch

TILE = 9
TIME = 10
MODES = ("actual", "dual_reference")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_native(native):
    require(type(native).__name__ == "MS_Spiking_Mlp" and native.norm_layer == "BN",
            "Native MS_Spiking_Mlp with BN required")
    require(all(not m.training for m in native.modules()), "Native inference/eval required")
    require(not any(p.requires_grad for p in native.parameters()), "Native parameters must be frozen")
    c, hidden = native.fc1.in_features, native.fc1.out_features
    require((native.fc2.in_features, native.fc2.out_features) == (hidden, c), "Native FC widths differ")
    for name, width in (("sn1", c), ("sn2", hidden)):
        sn = getattr(native, name).spiking_neuron
        require(type(sn).__name__ == "PSN" and sn.T == TIME and
                tuple(sn.weight.shape) == (TIME, TIME) and tuple(sn.bias.shape) == (TIME, 1),
                "Complete-T10 native PSN required: " + name)
        require(getattr(sn, "step_mode", "m") == "m", "PSN must use multi-step mode")
    for name, width in (("bn1", hidden), ("bn2", c)):
        bn = getattr(native, name).norm_layer
        require(isinstance(bn, torch.nn.modules.batchnorm._BatchNorm) and
                not bn.training and bn.track_running_stats and
                bn.running_mean is not None and bn.running_var is not None,
                "Private frozen running-stat BN required: " + name)
        require(bn.num_features == width and bool(torch.isfinite(bn.running_mean).all()) and
                bool(torch.isfinite(bn.running_var).all()) and bool((bn.running_var >= 0).all()),
                "Invalid native BN buffers: " + name)
        require(getattr(bn, "step_mode", "m") == "m", "Native BN must use multi-step mode")
    for name in ("fc1", "fc2"):
        require(getattr(getattr(native, name), "step_mode", "m") == "m",
                "Native FC must use multi-step mode")
    return c, hidden


def cell_spans(h, w, stride):
    require(stride in (3, 9), "Only stride3 or whole-tile stride9 is supported")
    require(0 < h <= TILE and 0 < w <= TILE, "Valid real tile extent required")
    return [[(y, min(y + stride, h), x, min(x + stride, w))
             for x in range(0, w, stride)] for y in range(0, h, stride)]


def spatial_mean(local, stride):
    """Mean over valid H/W cells, independently for every T/B/C; never pad."""
    spans = cell_spans(local.shape[2], local.shape[3], stride)
    rows = []
    for row in spans:
        columns = []
        for y0, y1, x0, x1 in row:
            area = (y1 - y0) * (x1 - x0)
            columns.append(local[:, :, y0:y1, x0:x1, :].sum((2, 3)) / area)
        rows.append(torch.stack(columns, dim=2))
    pooled = torch.stack(rows, dim=2).contiguous()
    require(bool(torch.isfinite(pooled).all()), "Nonfinite pooled representative; reduction overflow")
    return pooled, spans


def repeat_branch(representatives, spans, h, w):
    """Copy only the representative branch value to each real cell member."""
    t, b, _, _, c = representatives.shape
    out = torch.empty((t, b, h, w, c), dtype=representatives.dtype, device=representatives.device)
    for gy, row in enumerate(spans):
        for gx, (y0, y1, x0, x1) in enumerate(row):
            out[:, :, y0:y1, x0:x1, :] = representatives[:, :, gy:gy+1, gx:gx+1, :]
    return out


def service_ledger(shape, fine_mask, merge_stride, hidden_features):
    """Logical tensor/arithmetic counts, not measured traffic, latency or cycles."""
    t, b, h, w, c = shape
    fine_points = coarse_points = representative_points = coarse_tiles = 0
    all_representative_points = 0
    for gy, row in enumerate(fine_mask):
        for gx, fine in enumerate(row):
            ht, wt = min(TILE, h - TILE*gy), min(TILE, w - TILE*gx)
            reps = math.ceil(ht/merge_stride)*math.ceil(wt/merge_stride)
            all_representative_points += reps
            if fine:
                fine_points += ht*wt
            else:
                coarse_points += ht*wt
                representative_points += reps
                coarse_tiles += 1
    full_rows = t*b*h*w
    selected_rows = t*b*(fine_points + representative_points)
    cancelled_rows = full_rows - selected_rows
    q, r = coarse_points, representative_points
    return {
        "fine_spatial_points": fine_points, "coarse_source_spatial_points": q,
        "coarse_representative_spatial_points": r, "coarse_tiles": coarse_tiles,
        "all_coarse_representative_spatial_points": all_representative_points,
        "dense_fc_rows_per_layer": full_rows,
        "selected_fc_rows_per_layer": selected_rows,
        "cancelled_fc_rows_per_layer": cancelled_rows,
        "cancelled_fc_weight_products_dense_equivalent": cancelled_rows*2*c*hidden_features,
        "cancelled_psn_temporal_macs_nominal": (q-r)*b*t*t*(c+hidden_features),
        "cancelled_bn_elements": cancelled_rows*(c+hidden_features),
        "selected_added_work": {
            "pool_input_elements": t*b*c*q,
            "pool_reduction_additions_logical": t*b*c*(q-r),
            "pool_divisions_logical": t*b*c*r,
            "representative_tensor_elements": t*b*c*r,
            "repeat_dense_write_elements": t*b*c*q,
            "repeat_unique_source_elements": t*b*c*r,
        },
        "common_output_assembly_write_elements": t*b*h*w*c,
        "external_residual_additions_unchanged": t*b*h*w*c,
        "local_contiguous_copy_upper_bound_elements": t*b*h*w*c,
        "scope": "Selected deployment path only; logical arithmetic and elements, not cycle/RTL/traffic results",
        "unmodelled": ["mask delivery and host/device synchronization", "Python dispatch and allocation",
                       "physical pooling/repeat traffic", "layout conversion within native kernels",
                       "audit hashes, finite scans, and their synchronization",
                       "FC spike density and backend kernel scheduling"],
    }


def tile_core(native, u, fine_mask, merge_stride=3, mode="actual", audit_hashes=False):
    require(mode in MODES, "Unknown execution mode")
    require(merge_stride in (3, 9), "Only stride3 or whole-tile stride9 is supported")
    require(not torch.is_grad_enabled(), "Inference only; use torch.inference_mode()")
    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"), "Autocast is prohibited")
    c, hidden = validate_native(native)
    require(u.ndim == 5 and tuple(u.shape[:2]) == (TIME, 1) and
            u.shape[2] > 0 and u.shape[3] > 0 and u.shape[4] == c,
            "Expected complete T10/B1/positive H/W/native C")
    require(u.dtype in (torch.float32, torch.float64) and not u.requires_grad and
            bool(torch.isfinite(u).all()), "Finite frozen FP32/FP64 input required")
    require(u.dtype == native.fc1.weight.dtype and u.device == native.fc1.weight.device,
            "Native/input dtype or device differs")
    t, b, h, w, c = u.shape
    require(isinstance(fine_mask, torch.Tensor) and fine_mask.dtype == torch.bool and
            tuple(fine_mask.shape) == (math.ceil(h/TILE), math.ceil(w/TILE)),
            "Whole-tile boolean fine mask required")
    decisions = fine_mask.detach().cpu().tolist()
    ledger = service_ledger(tuple(u.shape), decisions, merge_stride, hidden)
    keys = ("native_mlp_calls", "fc1_rows", "fc2_rows", "sn1_elements", "sn2_elements",
            "bn1_elements", "bn2_elements")
    observed = {which: {key: 0 for key in keys} for which in ("fine", "coarse")}
    observed["native_input_shapes"] = {"fine": [], "coarse": []}
    active = None
    handles = []

    def native_hook(module, inputs):
        x = inputs[0]
        require(tuple(x.shape[:2]) == (TIME, 1), "A producer lost complete T10/B1")
        observed[active]["native_mlp_calls"] += 1
        observed["native_input_shapes"][active].append(list(x.shape))

    def count_hook(key, row_count=False):
        def hook(module, inputs):
            x = inputs[0]
            observed[active][key] += x.numel()//x.shape[-1] if row_count else x.numel()
        return hook

    handles.append(native.register_forward_pre_hook(native_hook))
    for name in ("fc1", "fc2"):
        handles.append(getattr(native, name).register_forward_pre_hook(count_hook(name+"_rows", True)))
    for name in ("sn1", "sn2"):
        handles.append(getattr(native, name).spiking_neuron.register_forward_pre_hook(count_hook(name+"_elements")))
    for name in ("bn1", "bn2"):
        handles.append(getattr(native, name).norm_layer.register_forward_pre_hook(count_hook(name+"_elements")))
    out = torch.empty_like(u, memory_format=torch.contiguous_format)
    executed_pool_points = executed_representatives = 0
    try:
        for gy, row in enumerate(decisions):
            for gx, selected_fine in enumerate(row):
                y0, y1 = TILE*gy, min(TILE*(gy+1), h)
                x0, x1 = TILE*gx, min(TILE*(gx+1), w)
                local = u[:, :, y0:y1, x0:x1, :].contiguous()
                fine = coarse = None
                if selected_fine or mode == "dual_reference":
                    active = "fine"
                    fine = native(local)
                    require(fine.shape == local.shape and bool(torch.isfinite(fine).all()), "Invalid fine branch")
                    require(fine.dtype == local.dtype and fine.device == local.device, "Fine producer dtype/device differs")
                if not selected_fine or mode == "dual_reference":
                    pooled, spans = spatial_mean(local, merge_stride)
                    executed_pool_points += local.shape[2]*local.shape[3]
                    executed_representatives += pooled.shape[2]*pooled.shape[3]
                    active = "coarse"
                    representative_branch = native(pooled)
                    require(representative_branch.shape == pooled.shape and bool(torch.isfinite(representative_branch).all()),
                            "Invalid coarse representative branch")
                    require(representative_branch.dtype == local.dtype and representative_branch.device == local.device,
                            "Coarse producer dtype/device differs")
                    coarse = repeat_branch(representative_branch, spans, local.shape[2], local.shape[3])
                out[:, :, y0:y1, x0:x1, :] = fine if selected_fine else coarse
    finally:
        for handle in handles:
            handle.remove()
    fine_points = h*w if mode == "dual_reference" else ledger["fine_spatial_points"]
    coarse_reps = (ledger["all_coarse_representative_spatial_points"] if mode == "dual_reference"
                   else ledger["coarse_representative_spatial_points"])
    total_tiles = math.ceil(h/TILE)*math.ceil(w/TILE)
    for which, points, calls in (("fine", fine_points, total_tiles if mode == "dual_reference" else total_tiles-ledger["coarse_tiles"]),
                                 ("coarse", coarse_reps, total_tiles if mode == "dual_reference" else ledger["coarse_tiles"])):
        obs = observed[which]
        require(obs["fc1_rows"] == obs["fc2_rows"] == t*b*points, "Observed FC rows differ: " + which)
        require(obs["native_mlp_calls"] == calls, "Observed native calls differ: " + which)
        require(obs["sn1_elements"] == obs["bn2_elements"] == t*b*points*c and
                obs["sn2_elements"] == obs["bn1_elements"] == t*b*points*hidden,
                "Observed PSN/BN service differs: " + which)
    record = {
        "mode": mode, "shape_TBHWC": list(u.shape), "merge_stride": merge_stride,
        "fine_mask": decisions, "observed": observed, "selected_service": ledger,
        "executed_fc_rows_per_layer": t*b*(fine_points+coarse_reps),
        "executed_pool_input_elements": t*b*c*executed_pool_points,
        "executed_pool_reduction_additions_logical": t*b*c*(executed_pool_points-executed_representatives),
        "executed_pool_divisions_logical": t*b*c*executed_representatives,
        "executed_repeat_write_elements": t*b*c*executed_pool_points,
        "external_residual_added_here": False, "temporal_pooling": False,
        "tail_rule": "real extents; each partial cell divides by its own valid area; no padding",
        "output_stride": list(out.stride()),
        "scope": "same-shape selected producers vs both producers; no task quality or hardware conclusion",
    }
    if audit_hashes:
        record.update(input_sha256=tensor_sha(u), output_sha256=tensor_sha(out))
    return out, record


class StructuredMergeMLP(torch.nn.Module):
    def __init__(self, native, fine_mask, merge_stride=3, mode="actual", expected_shape=None,
                 expected_u_sha=None, audit_hashes=True):
        super().__init__()
        validate_native(native)
        require(isinstance(fine_mask, torch.Tensor) and fine_mask.dtype == torch.bool and fine_mask.ndim == 2,
                "Whole-tile boolean fine mask required")
        require(mode in MODES and merge_stride in (3, 9), "Invalid mode or merge stride")
        self.native = native
        self.register_buffer("fine_mask", fine_mask.detach().clone())
        self.merge_stride, self.mode = merge_stride, mode
        self.expected_shape = tuple(expected_shape) if expected_shape is not None else None
        self.expected_u_sha, self.audit_hashes = expected_u_sha, audit_hashes
        self.records = []
        self.eval()

    def forward(self, u):
        if self.expected_shape is not None:
            require(tuple(u.shape) == self.expected_shape, "Declared native shape differs")
        if self.expected_u_sha is not None:
            require(tensor_sha(u) == self.expected_u_sha, "Frozen upstream input changed")
        output, record = tile_core(self.native, u, self.fine_mask, self.merge_stride, self.mode, self.audit_hashes)
        self.records.append(record)
        return output


@contextlib.contextmanager
def replace_mlp(model, layer_name, fine_mask, merge_stride=3, mode="actual", **kwargs):
    """Target one explicit MLP; surrounding current-stat BN is left untouched."""
    require(all(not m.training for m in model.modules()), "Backbone eval required")
    require(not any(p.requires_grad for p in model.parameters()), "Backbone parameters must be frozen")
    parent_name, child = layer_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    require(getattr(parent, "cnf", None) == "ADD" and not getattr(parent, "use_checkpoint", False),
            "Native ADD residual and no checkpoint required")
    original = getattr(parent, child)
    wrapper = StructuredMergeMLP(original, fine_mask, merge_stride, mode, **kwargs)
    try:
        setattr(parent, child, wrapper)
        yield wrapper
    finally:
        setattr(parent, child, original)
