"""Fixed single-matrix BBS migration, using the unchanged authors' functions.

There is no global-network pruning, retraining, BitVert PE/scheduler, or channel
reordering here. The same 32 scale-ranked output rows stay unchanged in all
three controls. Others use C16 groups, three pruned columns / five variable
bits. Official ZPS searches its declared six-bit constant domain, not a sweep
chosen on these captures. Returned reconstructed weights are kept as INT16;
there is no post-pruning INT8 clamp.

The ordinary control uses each original group's minimum signed width, rounds
to five signed bits with RNE, clamps to [-16,15], and restores the group dyadic
shift. Thus it too exploits redundant leading sign columns. Original row and
input scales remain unchanged. It is a specified ordinary group quantizer,
not all calibrated/learned PTQ methods. Group metadata is accounted separately.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'official'))
from binary_pruning import roundAvg_fc, zeroPointShifting_fc
torch.set_num_threads(1)


def save(path, obj):
    def conv(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv)+'\n')


def call_and_observe(function, w, **kwargs):
    """Observe return locals for encoding accounting, without editing the code."""
    captured = {}
    wanted = ('prune_until', 'wq_int_pruned')
    def profile(frame, event, arg):
        if event == 'return' and frame.f_code is function.__code__:
            for key in wanted:
                if key in frame.f_locals:
                    captured[key] = frame.f_locals[key].detach().cpu().numpy().copy()
    old = sys.getprofile()
    sys.setprofile(profile)
    try:
        got = function(torch.from_numpy(w.astype(np.float32).copy()), **kwargs)
    finally:
        sys.setprofile(old)
    result = got.detach().cpu().numpy()
    assert np.array_equal(result, np.rint(result))
    return result.astype(np.int16), captured


def signed_width(lo, hi):
    width = 1
    while lo < -(1 << (width-1)) or hi > (1 << (width-1))-1:
        width += 1
    return width


def ordinary_group5(w):
    groups = w.reshape(-1, 16).astype(np.int64)
    widths = np.array([signed_width(int(g.min()), int(g.max())) for g in groups])
    shifts = np.maximum(0, widths-5)
    code_preclip = np.rint(groups/np.exp2(shifts[:, None])).astype(np.int64)
    code = np.clip(code_preclip, -16, 15)
    restored = code << shifts[:, None]
    return restored.reshape(w.shape).astype(np.int16), dict(
        group_shift=shifts, group_code=code.astype(np.int8),
        ordinary_rounding_clipped_values=int(np.count_nonzero(code != code_preclip)))


def error_metrics(value, reference):
    error = value.astype(np.float64)-reference.astype(np.float64)
    return dict(max_abs=float(np.abs(error).max()), RMSE=float(np.sqrt(np.mean(error**2))),
                MAE=float(np.mean(np.abs(error))),
                relative_RMSE=float(np.sqrt(np.sum(error**2)/np.sum(reference.astype(np.float64)**2))))


def main():
    parameter = HERE.parent.parent/'projection_control_parameters.npz'
    with np.load(parameter) as z:
        base = {k:z[k].copy() for k in z.files}
    W = base['Wq'].astype(np.int16)
    row_scale = base['row_scale']
    order = np.lexsort((np.arange(96), -row_scale))
    sensitive = np.sort(order[:32])
    pruned = np.sort(order[32:])
    xq = base['calibration_Xq'].astype(np.int64)
    x = []
    for file in base['calibration_source_files']:
        with np.load(str(file)) as z:
            x.append(z['r1out'][:, :, z['anchor_mask'].astype(bool)].transpose(0,2,1).reshape(-1,96))
    x = np.concatenate(x).astype(np.float64)
    assert np.array_equal(np.clip(np.rint(x/float(base['input_scale'])), -2048,2047).astype(np.int64), xq)
    output_scale = (float(base['input_scale'])*row_scale).astype(np.float32)
    def decode(w):
        dot = xq @ w.astype(np.int64).T
        return dot.astype(np.float32)*output_scale[None,:]+base['original_bias'][None,:]
    original = x @ base['C'].astype(np.float64).T+base['original_bias'].astype(np.float64)
    base_output = decode(W)
    summary = dict(
        source_parameters=parameter, official_source='https://github.com/yc2367/BBS-MICRO',
        primary_paper='https://arxiv.org/html/2409.05227v1',
        method_read='BBS III-B, Algorithm 1, III-C; IV-A--D separates BitVert from this compile migration',
        scope='one 96x96 continuous PED projection; not whole-network global pruning or BitVert hardware reproduction',
        configuration=dict(group_size=16, num_pruned_column=3, ZPS_const_bitwidth=6,
                           protected_rows=32, pruned_rows=64,
                           selection='row_scale descending, original row index breaks ties'),
        sensitive_rows=sensitive, pruned_rows=pruned, original_output_order_preserved=True,
        actual_vectors=len(xq), actual_values=int(xq.size), calibration_only='four training frames',
        original_W8_local_error_vs_captured_FP=error_metrics(base_output, original), variants={})
    common = dict(w_bitwidth=8, group_size=16, num_pruned_column=3, device='cpu')
    variants = []
    for name, function, extra in (
        ('round_average', roundAvg_fc, {}),
        ('zero_point_shift', zeroPointShifting_fc, dict(const_bitwidth=6))):
        before = time.monotonic()
        sub, locals_ = call_and_observe(function, W[pruned], **common, **extra)
        q = W.copy(); q[pruned] = sub
        encoding = {}
        if name == 'round_average':
            until = locals_['prune_until'].astype(np.int64)
            redundant = until-5
            low_width = np.maximum(0, 8-until)
            group = sub.reshape(-1,16).astype(np.int64)
            constant = group[:,0] % (1 << low_width)
            assert np.all(group % (1 << low_width[:,None]) == constant[:,None])
            encoding.update(group_redundant_columns=redundant, group_low_bit_width=low_width,
                            group_constant=constant)
        else:
            all_restored = locals_['wq_int_pruned'].astype(np.int64)
            groups = W[pruned].reshape(-1,16).astype(np.int64)
            sse = np.sum((all_restored-groups[None,:,:])**2,axis=-1)
            best = np.argmin(sse,axis=0)  # same strict-improvement first tie as official
            selected = all_restored[best,np.arange(len(groups))]
            assert np.array_equal(selected,sub.reshape(-1,16))
            offsets = best-32
            until = locals_['prune_until'].astype(np.int64)[best,np.arange(len(groups))]
            encoding.update(group_redundant_columns=until-5, group_constant=offsets,
                            group_shifted_pruned=selected+offsets[:,None])
        variants.append((name,q,encoding,time.monotonic()-before))
        print('GENERATED',name,'range',int(q.min()),int(q.max()),'seconds',round(time.monotonic()-before,3),flush=True)
    sub, encoding = ordinary_group5(W[pruned])
    q = W.copy(); q[pruned] = sub
    variants.append(('uniform_group5',q,encoding,0.))
    for name,q,encoding,seconds in variants:
        assert np.array_equal(q[sensitive], W[sensitive])
        pos = np.maximum(q.astype(np.int64),0).sum(axis=1)
        neg = np.minimum(q.astype(np.int64),0).sum(axis=1)
        dotlo, dothi = -2048*pos+2047*neg, 2047*pos-2048*neg
        bound = np.maximum(-dotlo,dothi)
        fields = dict(base)
        fields.update(Wq=q.astype(np.int16), weight_integer_range=np.array([q.min(),q.max()],np.int64),
                      dot_absolute_bound=bound, dot_signed_range=np.stack([dotlo,dothi],axis=1),
                      variant=np.array(name), sensitive_rows=sensitive, pruned_rows=pruned,
                      group_size=np.array(16), variable_weight_bits=np.array(5),
                      definition=np.array(
                          f'Stride2 anchor continuous input. Xq=RNE(X/input_scale) clamp[-2048,2047]. '
                          f'Wq is restored integer {name}, protected32 rows retain original W8; no final W8 clamp. '
                          'Exact integer dot via FP64; cast dot toFP32, multiply by unchanged input_scale*row_scale[h], '
                          'add originalFP32 bias once. No theta/BN folding. See parameters_result.json.'),
                      **encoding)
        output=HERE/f'{name}.npz'
        np.savez_compressed(output,**fields)
        result=dict(parameters=output,weight_integer_range=[int(q.min()),int(q.max())],
                    values_outside_original_W8_domain=int(np.count_nonzero((q < -127)|(q > 127))),
                    reconstructed_storage_dtype=str(q.dtype),
                    changed_weights=int(np.count_nonzero(q != W)),nonzero_weights=int(np.count_nonzero(q)),
                    weight_error_vs_W8=error_metrics(q,W),
                    decoded_error_vs_same_input_W8=error_metrics(decode(q),base_output),
                    decoded_error_vs_captured_original_FP=error_metrics(decode(q),original),
                    legal_dot_range=[int(dotlo.min()),int(dothi.max())],
                    maximum_output_signed_bits=max(signed_width(int(lo),int(hi)) for lo,hi in zip(dotlo,dothi)),
                    output_scale_and_bias_unchanged=True, wall_seconds=seconds,
                    encoding=encoding,
                    fixed_weight_payload_bytes=(32*96*8+64*96*5)//8,
                    group_metadata_slots=64*96//16,
                    group_metadata_bytes_if_common_8bit_slot=64*96//16,
                    common_8bit_metadata_total_bytes=(32*96*8+64*96*5)//8+64*96//16,
                    metadata_note='BBS: 2 redundant-column bits +6 constant bits/group. Ordinary: shift only, <=3bits suffices; 8bit slot comparison is explicit padding, not a requirement. Row selection/order metadata and shared scales excluded; no BitVert service simulation.',
                    baseline_note='uniform_group5 is fixed RNE dyadic group PTQ, not an assertion that calibrated or trained PTQ has been exhausted.')
        summary['variants'][name]=result
    save(HERE/'parameters_result.json',summary)
    print('READY',[(k,v['parameters'].name) for k,v in summary['variants'].items()],flush=True)


if __name__ == '__main__': main()
