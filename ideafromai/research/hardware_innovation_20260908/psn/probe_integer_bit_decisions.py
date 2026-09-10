"""Exact MSB decision opportunity on existing S2 integer class-state traces.

Bounds/early termination are established prior art. This reports arithmetic
opportunities only, with fixed SIMD and ideal within-tile compaction bounds.
It does not credit free compaction, input gathering, or a bit-serial speedup
over a word MAC. All intermediate state and theta semantics are preserved.
"""
import argparse
import json
from pathlib import Path
import math
import numpy as np
import torch


def issue_bounds(steps, width):
    # [T,P,H] with H an exact multiple of96. Every dot has the same B row.
    t, p, h = steps.shape
    fixed = int(steps.reshape(t, p, h//96, 96).amax(-1).sum())
    full = t*p*(h//96)*width
    compact = {}
    for spatial in (1, 6, 32):
        total, peak_pending = 0, 0
        for start in range(0, p, spatial):
            block = steps[:, start:start+spatial].reshape(t, -1, h//96, 96)
            for depth in range(1, width+1):
                active = (block >= depth).sum((1, 3))
                total += int(((active+95)//96).sum())
                peak_pending = max(peak_pending, int(active.sum(0).max()))
        compact[str(spatial)] = {'ideal_vector_bit_steps': total,
                                 'ratio_to_full_width': total/full,
                                 'peak_live_gate_contexts_per_H96_tile': peak_pending}
    return {'word_input_valid_signed_bits': width,
            'full_width_vector_bit_steps': full,
            'fixed_SIMD96_vector_bit_steps': fixed,
            'fixed_SIMD96_ratio_to_full_width': fixed/full,
            'mean_scalar_bit_steps': float(steps.float().mean()),
            'independent_DPU_work_lower_bound_ratio': float(steps.float().mean())/width,
            'ideal_compaction_by_spatial_tile': compact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    alg = args.root/'algorithm'
    out = args.root/'psn/bit_decision_probe'
    out.mkdir(parents=True, exist_ok=True)
    params = torch.load(alg/'stage2_temporal_codes/integer_parameters.pt', map_location='cpu', weights_only=False)
    basis = np.load(alg/'stage2_temporal_codes/signed_basis.npz')
    captures = alg/'stage2_deployment_diverse_capture/capture'
    records = []
    files = [p for p in sorted(captures.glob('*.npz'))
             if int(p.name[1:4]) < 4 and p.stem.rsplit('_', 1)[1] in ('s2b0', 's2b3')]
    with torch.no_grad():
        for path in files:
            tag = path.stem.rsplit('_', 1)[1]
            prefix = f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{tag[-1]}.mlp.'
            q = params[prefix]
            assert q['positive_gain'].all() and not q['constant_channels'].any()
            with np.load(path) as capture:
                codes = torch.from_numpy(capture['codes'].astype(np.int64)).cuda()
            E = torch.from_numpy(basis[tag+'_coordinates_int8'].astype(np.float32)).cuda()
            B = torch.from_numpy(basis[tag+'_B_int32'].astype(np.float64)).cuda()
            W = q['weight_int8'].cuda().float()
            features = E[codes].permute(0, 2, 1)
            S = (features @ W.T).double()
            live = features.ne(0).any(-1).T.double()
            U = torch.einsum('tr,prh->tph', B, S)
            tau = q['threshold_int64'].cuda().double()
            truth = U >= tau[:, None, :]
            pos = (B.clamp_min(0) @ live)[:, :, None]
            neg = (B.clamp_max(0) @ live)[:, :, None]
            bound = int(q['weight_int8'].long().abs().sum(1).max())
            width = bound.bit_length()+1
            bit = width-1
            scale = 2**bit
            qt = torch.ceil(tau/scale)
            residue = qt[:, None, :]+torch.einsum('tr,prh->tph', B, S.lt(0).double())
            pending = torch.ones_like(truth)
            steps = torch.zeros_like(truth, dtype=torch.uint8)
            hist, errors, r_min, r_max = [], 0, 0, 0
            for bit in range(width-1, -1, -1):
                scale = 2**bit
                eta = (scale*qt-tau)[:, None, :]
                low = torch.floor(((scale-1)*neg+eta)/scale)
                high = torch.floor(((scale-1)*pos+eta)/scale)
                yes = pending & (residue <= low)
                no = pending & (residue > high)
                errors += int((yes & ~truth).sum())+int((no & truth).sum())
                newly = yes | no
                steps[newly] = width-bit
                pending &= ~newly
                remaining = int(pending.sum())
                if remaining:
                    values = residue[pending]
                    r_min = min(r_min, int(values.min()))
                    r_max = max(r_max, int(values.max()))
                hist.append({'remaining_low_bits': bit, 'newly_decided': int(newly.sum()), 'still_pending': remaining})
                if bit:
                    next_qt = torch.ceil(tau/(scale//2))
                    digits = torch.remainder(torch.floor(S/(scale//2)), 2)
                    terms = torch.einsum('tr,prh->tph', B, digits)
                    residue = torch.where(pending, 2*residue-terms+(next_qt-2*qt)[:, None, :], 0)
                    qt = next_qt
            assert not pending.any() and errors == 0
            residue_bits = 1
            while r_min < -(1 << (residue_bits-1)) or r_max >= 1 << (residue_bits-1):
                residue_bits += 1
            row = {'capture': path.name, 'T_P_H': list(U.shape), 'decision_mismatches': errors,
                   'INT24_S_static_abs_bound': bound, 'observed_S_abs_max': int(S.abs().max()),
                   'observed_pending_normalized_residue_range': [r_min, r_max],
                   'observed_pending_residue_signed_bits': residue_bits,
                   'per_plane': hist, **issue_bounds(steps, width)}
            records.append(row)
            print('CASE', path.name, 'SIMD', row['fixed_SIMD96_ratio_to_full_width'],
                  'ideal_DPU', row['independent_DPU_work_lower_bound_ratio'], flush=True)
            (out/'result.json').write_text(json.dumps({'cases': records,
                'scope': 'two S2 modules, first frame in four sequences; same integer student, no new AEE or timing',
                'limits': 'Bit-step counters are not MAC cycles. Independent async-DPU and ideal compaction omit dispatch/input/threshold access and residue moves. Ordinary streaming DPUs need only one partial per in-flight context, not a full-tile U array. No novelty or hardware speed claim.'}, indent=2)+'\n')
            del codes, features, S, U, live, truth, pos, neg, residue, pending, steps


if __name__ == '__main__':
    main()
