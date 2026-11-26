#!/usr/bin/env python3
"""QP-paper SVS-only reconstruction for native v2 MLP hidden channels.

Not official QP code; no ReScaW/quantization or training implementation.
Scoring runs on CPU, and compression returns independent deep copies.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

EPSILON = 1e-6
FORMULA = {
    'name': 'QP-paper SVS-only / v2 hidden-channel adaptation',
    'source': 'https://arxiv.org/html/2502.05905v2#S4.SS2',
    'location': 'section4.2 equations11-13; appendixA algorithm1',
    'formula': 'Sbar[b,c,h,w]=mean_t(S[t,b,c,h,w]); SVS[c]=mean_b(sum_i(sigma_i(Sbar[b,c])>1e-6))',
    'epsilon': EPSILON,
    'score_site': 'native MS_Spiking_Mlp.sn2 output BEFORE drop2 and fc2',
    'numerics': 'CPU float64 time mean and SVD; absolute singular-value threshold',
    'cross_batch_reduction': 'sample-weighted mean; equal weight per sample, not per minibatch',
    'provenance': 'Reconstructed from paper; official QP implementation not confirmed',
    'excluded': ['ReScaW', 'quantization', 'spatial_runtime_gate', 'accuracy_or_cycle_claim'],
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def svs_ranks(spikes, layout='TBHWC', channel_chunk=32, expected_steps=10):
    """Return int64 [B,C] effective ranks from binary spike maps, CPU only.

    Includes every spatial point of the provided native MLP tensor, with no GT,
    event-density, keep-mask weighting or spatial pooling. Chunking limits SVD
    working memory but does not change the individual H x W matrix definition.
    """
    require(isinstance(spikes, torch.Tensor) and spikes.device.type == 'cpu',
            'Provide an explicitly detached CPU spike tensor')
    require(not spikes.is_complex(), 'Complex tensors are not binary spike maps')
    require(spikes.ndim == 5 and all(int(s) > 0 for s in spikes.shape), 'Expected a nonempty 5D spike tensor')
    require(layout in ('TBHWC', 'TBCHW'), 'Specify TBHWC or TBCHW explicitly')
    require(type(channel_chunk) is int and channel_chunk > 0, 'channel_chunk must be positive')
    require(spikes.shape[0] == expected_steps, 'Complete PSN time dimension required')
    require(bool(torch.isfinite(spikes).all()), 'Nonfinite spike map')
    require(bool(((spikes == 0) | (spikes == 1)).all()), 'SVS input must be binary sn2 spikes, not membrane or u')
    s = spikes.detach().permute(0, 1, 4, 2, 3) if layout == 'TBHWC' else spikes.detach()
    batch, channels = s.shape[1:3]
    result = torch.empty((batch, channels), dtype=torch.int64)
    with torch.no_grad():
        for c0 in range(0, channels, channel_chunk):
            avg = s[:, :, c0:c0 + channel_chunk].to(torch.float64).mean(dim=0)
            singular = torch.linalg.svdvals(avg)
            result[:, c0:c0 + channel_chunk] = (singular > EPSILON).sum(dim=-1)
    return result


def svs_score(spikes, layout='TBHWC', channel_chunk=32, expected_steps=10):
    """Exact sample mean of per-channel effective rank within one minibatch."""
    return svs_ranks(spikes, layout, channel_chunk, expected_steps).to(torch.float64).mean(dim=0)


class SVSAccumulator:
    """Bounded train-only caller-bound collector; duplicate sample IDs rejected.

    Caller supplies fixed allowed training IDs and associates each sn2 hook with
    the exact current minibatch IDs. This class cannot infer identity from data.
    """
    def __init__(self, allowed_ids, max_samples, expected_steps=10, channel_chunk=32):
        self.allowed_ids = frozenset(allowed_ids)
        require(self.allowed_ids and all(isinstance(x, str) and x for x in self.allowed_ids), 'Explicit allowed IDs required')
        require(type(max_samples) is int and 0 < max_samples <= len(self.allowed_ids), 'Invalid sample cap')
        self.max_samples, self.expected_steps, self.channel_chunk = max_samples, expected_steps, channel_chunk
        self.seen_ids, self.batch_sizes, self.rank_sum = [], [], None

    def update(self, spikes, sample_ids, layout='TBHWC'):
        ids = list(sample_ids)
        require(len(ids) == spikes.shape[1] and len(ids) == len(set(ids)), 'Batch sample IDs do not match B or contain duplicates')
        require(set(ids) <= self.allowed_ids and not (set(ids) & set(self.seen_ids)), 'Out-of-pool or duplicate sample')
        require(len(self.seen_ids) + len(ids) <= self.max_samples, 'SVS sample cap exceeded; do not silently truncate a batch')
        ranks = svs_ranks(spikes, layout, self.channel_chunk, self.expected_steps)
        require(self.rank_sum is None or self.rank_sum.numel() == ranks.shape[1], 'Hidden width changed during collection')
        if self.rank_sum is None:
            self.rank_sum = torch.zeros(ranks.shape[1], dtype=torch.int64)
        self.rank_sum += ranks.sum(dim=0)
        self.seen_ids.extend(ids); self.batch_sizes.append(len(ids))

    def result(self):
        require(self.rank_sum is not None and self.seen_ids, 'No SVS observations')
        scores = self.rank_sum.to(torch.float64) / len(self.seen_ids)
        values, counts = torch.unique(scores, return_counts=True)
        return {'schema': 'flow_svs_score_v1', 'formula': FORMULA, 'sample_ids': self.seen_ids.copy(),
                'sample_count': len(self.seen_ids), 'batch_sizes': self.batch_sizes.copy(),
                'sample_cap': self.max_samples, 'hidden_width': scores.numel(),
                'rank_sums': self.rank_sum.tolist(), 'scores': scores.tolist(),
                'score_histogram': [{'score': float(v), 'channels': int(n)} for v, n in zip(values, counts)],
                'identity_scope': 'IDs bound by caller; must attach data/checkpoint/source/BN/geometry hashes before scientific use'}


def _check_psn(sn, expected_steps):
    require(hasattr(sn, 'spiking_neuron'), 'Expected native neuron wrapper')
    psn = sn.spiking_neuron
    require(type(psn).__name__ == 'PSN' and psn.T == expected_steps, 'Only native complete-T PSN supported')
    require(set(dict(psn.named_parameters())) == {'weight', 'bias'}, 'PSN has unexpected or channel-dependent parameters')
    require(tuple(psn.weight.shape) == (expected_steps, expected_steps) and
            tuple(psn.bias.shape) == (expected_steps, 1), 'PSN dependency differs from shared time-only parameters')
    require(not list(psn.named_buffers()), 'Unexpected PSN state buffers require a new dependency audit')


def validate_native_mlp(mlp, expected_steps=10):
    require(type(mlp).__name__ == 'MS_Spiking_Mlp', 'Only audited native MS_Spiking_Mlp is supported')
    require(mlp.norm_layer == 'BN', 'Only native BN MLP closure is audited; GN/LN/IN/BNTT not supported')
    require(isinstance(mlp.fc1, nn.Linear) and isinstance(mlp.fc2, nn.Linear), 'Expected actual Linear modules')
    require(mlp.fc1.out_features == mlp.fc2.in_features and mlp.fc1.in_features == mlp.fc2.out_features,
            'Hidden dependency or external residual width differs')
    for number, channels in ((1, mlp.fc1.out_features), (2, mlp.fc2.out_features)):
        wrapper = getattr(mlp, 'bn' + str(number)); bn = wrapper.norm_layer
        require(wrapper.norm in ('BN', 'BN_notrack') and isinstance(bn, nn.BatchNorm2d), 'Only per-channel BatchNorm2d supported')
        require(bn.num_features == channels, 'BN width does not match producer')
    for sn in (mlp.sn1, mlp.sn2):
        _check_psn(sn, expected_steps)
    # Reject hidden learned state in an unexpected child rather than silently
    # pretending FC/BN slicing is a complete dependency closure.
    require(set(mlp._modules) == {'fc1', 'fc2', 'bn1', 'bn2', 'sn1', 'sn2', 'drop1', 'drop2'}, 'Unexpected MLP child module')
    require(not list(mlp.named_parameters(recurse=False)) and not list(mlp.named_buffers(recurse=False)), 'Unexpected MLP-owned state')
    require(not list(mlp.drop1.parameters()) and not list(mlp.drop2.parameters()), 'Unexpected learned dropout state')
    return {'input_channels': mlp.fc1.in_features, 'hidden_channels': mlp.fc1.out_features,
            'output_channels': mlp.fc2.out_features, 'T': expected_steps}


def magnitude_score(mlp, expected_steps=10):
    """Simple same-width control: mean absolute FC1 row weight, no BN folding.

    This control is ours, not a replacement interpretation of paper SVS.
    """
    validate_native_mlp(mlp, expected_steps)
    return mlp.fc1.weight.detach().to(device='cpu', dtype=torch.float64).abs().mean(dim=1)


def select_hidden_channels(scores, keep_width, pe_group_width=1):
    """Keep high scores; tie by original channel index. No silent rounding.

    Surviving channels are packed into a physically narrower dense matrix;
    original index adjacency is not required for static channel compaction.
    """
    scores = torch.as_tensor(scores, dtype=torch.float64, device='cpu')
    require(scores.ndim == 1 and scores.numel() > 0 and bool(torch.isfinite(scores).all()), 'Invalid channel scores')
    require(type(keep_width) is int and 0 < keep_width <= scores.numel(), 'Invalid keep width')
    require(type(pe_group_width) is int and pe_group_width > 0 and keep_width % pe_group_width == 0,
            'Requested retained width is not a positive multiple of PE group width')
    ranked = sorted(range(scores.numel()), key=lambda i: (-float(scores[i]), i))
    keep = sorted(ranked[:keep_width])
    cutoff = float(scores[ranked[keep_width - 1]])
    return {'schema': 'flow_svs_hidden_selection_v1', 'original_hidden_width': scores.numel(),
            'keep_width': keep_width, 'pe_group_width': pe_group_width, 'keep_indices': keep,
            'selected_by': 'descending score, original channel index ascending on ties',
            'physical_order': 'retained original indices ascending, compact dense storage',
            'cutoff_score': cutoff, 'channels_tied_at_cutoff': int((scores == cutoff).sum()),
            'spatially_dynamic': False}


def _slice_parameter(value, indices, dim):
    if value is None:
        return None
    out = value.detach().index_select(dim, indices.to(value.device)).clone().contiguous()
    return nn.Parameter(out, requires_grad=value.requires_grad)


def narrow_mlp_deepcopy(mlp, keep_indices, pe_group_width=1, expected_steps=10):
    """Return (new_native_mlp, report), never mutate input or reuse its storage.

    FC1 rows/bias -> BN1 affine/running state -> shared PSN2 -> FC2 columns.
    PSN1/2 and all time dimensions, FC2 output/bias, BN2 and residual C stay.
    Optimizer state MUST be rebuilt for the new parameter objects.
    """
    shape = validate_native_mlp(mlp, expected_steps)
    raw_indices = list(keep_indices)
    require(raw_indices and all(type(i) is int for i in raw_indices), 'Channel indices must be plain integers')
    require(raw_indices == sorted(set(raw_indices)) and min(raw_indices) >= 0 and max(raw_indices) < shape['hidden_channels'],
            'Indices must be sorted unique and within original hidden width')
    require(type(pe_group_width) is int and pe_group_width > 0 and len(raw_indices) % pe_group_width == 0,
            'Retained hidden width is not PE-group aligned')
    require(not any(mod._forward_hooks or mod._forward_pre_hooks or mod._backward_hooks for mod in mlp.modules()),
            'Remove observation hooks before copying; copied external collectors are unsafe')
    before = sum(p.numel() for p in mlp.parameters())
    new = copy.deepcopy(mlp); ids = torch.tensor(raw_indices, dtype=torch.long)
    # SJ dropout can retain a hidden-width-dependent mask from a previous
    # training chunk. The new graph starts a new chunk; do not carry that mask.
    for dropout in (new.drop1, new.drop2):
        if hasattr(dropout, 'reset'):
            dropout.reset()
    new.fc1.weight = _slice_parameter(new.fc1.weight, ids, 0)
    new.fc1.bias = _slice_parameter(new.fc1.bias, ids, 0)
    new.fc1.out_features = len(raw_indices)
    bn = new.bn1.norm_layer
    bn.weight = _slice_parameter(bn.weight, ids, 0)
    bn.bias = _slice_parameter(bn.bias, ids, 0)
    for name in ('running_mean', 'running_var'):
        value = getattr(bn, name)
        if value is not None:
            setattr(bn, name, value.detach().index_select(0, ids.to(value.device)).clone().contiguous())
    bn.num_features = len(raw_indices)
    new.fc2.weight = _slice_parameter(new.fc2.weight, ids, 1)
    new.fc2.in_features = len(raw_indices)
    after_shape = validate_native_mlp(new, expected_steps)
    after = sum(p.numel() for p in new.parameters())
    expected_removed_per_channel = shape['input_channels'] + shape['output_channels'] + int(mlp.fc1.bias is not None)
    if mlp.bn1.norm_layer.affine:
        expected_removed_per_channel += 2
    require(before - after == (shape['hidden_channels'] - len(raw_indices)) * expected_removed_per_channel,
            'Actual parameter removal contradicts dependency closure')
    report = {'schema': 'flow_svs_narrow_mlp_v1', 'scope': 'SVS-only component adaptation; no ReScaW or quantization',
              'before': shape, 'after': after_shape, 'keep_indices': raw_indices, 'pe_group_width': pe_group_width,
              'parameters_before': before, 'parameters_after': after, 'parameters_removed': before - after,
              'changed': ['fc1.output_rows_and_optional_bias', 'bn1.channel_affine_and_running_statistics', 'fc2.input_columns'],
              'preserved': ['sn1_shared_TxT_and_Tx1', 'sn2_shared_TxT_and_Tx1', 'complete_T',
                            'fc2.output_and_optional_bias', 'bn2', 'external_residual_channel_interface'],
              'optimizer_state_reusable': False, 'accuracy_preservation_claimed': False,
              'recovery_and_BN_recalibration': 'Required experiment stages, not performed by this transformation'}
    return new, report


def narrow_model_deepcopy(model, keep_by_path, pe_group_width=1, expected_steps=10):
    """Apply an explicit hidden-channel plan to a new model; tied paths rejected.

    All targeted MLPs must be native and independently owned. The caller must
    bind checkpoint/config/source hashes and save the width map with new weights.
    """
    require(isinstance(keep_by_path, dict) and keep_by_path, 'Nonempty explicit MLP path plan required')
    modules = dict(model.named_modules())
    all_paths = list(model.named_modules(remove_duplicate=False))
    for path in keep_by_path:
        require(path and path in modules, 'Unknown/nonempty MLP path required')
        require(sum(id(mod) == id(modules[path]) for _, mod in all_paths) == 1, 'Shared MLP path requires a separate closure audit')
        validate_native_mlp(modules[path], expected_steps)
        owned = {id(p) for p in modules[path].parameters()}
        names = [n for n, p in model.named_parameters(remove_duplicate=False) if id(p) in owned]
        require(all(n.startswith(path + '.') for n in names), 'Target MLP shares parameters outside its closure')
    new = copy.deepcopy(model); reports = {}
    for path, indices in sorted(keep_by_path.items()):
        original = new.get_submodule(path)
        replacement, report = narrow_mlp_deepcopy(original, indices, pe_group_width, expected_steps)
        parent_path, _, child = path.rpartition('.')
        new.get_submodule(parent_path).add_module(child, replacement)
        reports[path] = report
    return new, {'schema': 'flow_svs_narrow_model_v1', 'blocks': reports,
                 'parameters_before': sum(p.numel() for p in model.parameters()),
                 'parameters_after': sum(p.numel() for p in new.parameters()),
                 'requires_new_optimizer_and_width_aware_reconstruction': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spikes', type=Path, required=True, help='Binary NPY from sn2; no pickle')
    parser.add_argument('--layout', choices=('TBHWC', 'TBCHW'), default='TBHWC')
    parser.add_argument('--keep-width', type=int, required=True)
    parser.add_argument('--pe-group-width', type=int, default=16)
    parser.add_argument('--channel-chunk', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refuse to overwrite an existing score artifact')
    import numpy as np
    torch.set_num_threads(1)
    data = args.spikes.read_bytes()
    import io
    spikes = torch.from_numpy(np.load(io.BytesIO(data), allow_pickle=False))
    scores = svs_score(spikes, args.layout, args.channel_chunk)
    result = {'schema': 'flow_svs_npy_diagnostic_v1', 'formula': FORMULA,
              'input_sha256': hashlib.sha256(data).hexdigest(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'shape': list(spikes.shape), 'layout': args.layout, 'scores': scores.tolist(),
              'selection': select_hidden_channels(scores, args.keep_width, args.pe_group_width),
              'scope': 'Offline scoring only; no dataset/checkpoint provenance inferred from NPY',
              'GPU_initialized': torch.cuda.is_initialized()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'SCORED_CPU', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
