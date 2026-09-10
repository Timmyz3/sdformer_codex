"""Train-calibrated bounded-input PSN: a CPU opportunity screen, not AEE.

Clipping changes the student. Once the bounded student is defined, intervals
use only static legal input ranges, never actual unproduced convolution values.
Reuse the ordinary five-Y/one-U P4/H8 planner so all axes get its same baseline.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PATCH = HERE.parent
ROOT = PATCH.parents[1]
sys.path.insert(0, str(ROOT/'bn_state'))
from support_service_model import read_torch

spec = importlib.util.spec_from_file_location('ordinary_completion', PATCH/'partial_completion/probe.py')
ordinary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ordinary)


def intervals(a, lower, upper):
    midpoint, radius = [], []
    for state in range(1024):
        remaining = a*((state & ordinary.BITS) == 0)[None]
        lo = np.maximum(remaining, 0)@lower+np.minimum(remaining, 0)@upper
        hi = np.maximum(remaining, 0)@upper+np.minimum(remaining, 0)@lower
        center = (hi+lo)*.5
        # Small outward allowance for this Float64 opportunity calculation;
        # this is not an integer/FP32 hardware interval proof.
        guard = 1e-10*(1+np.maximum(np.abs(lo), np.abs(hi)))
        midpoint.append(center)
        radius.append((hi-lo)*.5+guard)
    return np.asarray(midpoint), np.asarray(radius)


def aggregate(rows):
    names = [k for k in rows[0] if k not in ('file', 'split')]
    out = {k:(max(r[k] for r in rows) if k == 'peak_allocated_Y_columns'
              else sum(r[k] for r in rows)) for k in names}
    out['conv1_active_fraction'] = out['conv1_active_terms']/out['full_conv1_active_terms']
    out['clipped_input_fraction'] = out['clipped_inputs']/out['input_values']
    out['clipped_model_gate_difference'] = out['gates_changed_by_clipping']/out['gates']
    out['false_negative_fraction_original_spikes'] = out['clipping_false_negatives']/max(1,out['original_spikes'])
    out['frame_count'] = len(rows)
    return out


def main():
    started = time.monotonic()
    capture = read_torch(PATCH/'partial_completion/capture.pt')
    params = json.loads((PATCH/'dependency/fit.json').read_text())['variants']['row34']
    a = np.asarray(params['weight'], np.float64)
    bias = np.asarray(params['bias'], np.float64).reshape(10)
    theta = capture['metadata']['neuron_theta']
    assert capture['metadata']['center_mode'] == 'zero'
    train = np.concatenate([s['Y'].reshape(10,-1) for s in capture['samples'] if s['split']=='train'],axis=1)
    result = dict(scope='16 train / 4 held-out valid frames; 64 native P4 groups/frame, all 96 channels and T10',
        parameters='saved row34 A/bias, actual neuron_theta from capture; mask-score threshold is unrelated',
        theta=theta, time_order=ordinary.ORDER,
        finite_state='same ordinary planner: at most five Y columns including production plus one working U; no maintained T10-wide U',
        calibration='two presets fixed before screening: per-time train extrema, or per-time [0.5%,99.5%] quantiles; channels share ranges',
        exactness='candidate clips every input to its static range; certified decisions are checked against the FULL clipped Float64 function, not unclipped parent or FP32',
        source_rule='same actual source support; truly empty P4/time windows use known fixed-BN constant in all axes',
        exclusions=['controller/ROM/lookup/interval comparison latency and ports',
                    'full-layer conv2, final optical flow AEE, training or integer proof',
                    'future input values cannot be used in intervals; only labels and issued columns read actual Y'],
        axes={})
    for name, q in [('train_extrema_clip', 0.), ('percentile99_clip', .005)]:
        lower = np.quantile(train,q,axis=1)
        upper = np.quantile(train,1-q,axis=1)
        tm, ts = intervals(a,lower,upper)
        axis = dict(lower=lower.tolist(),upper=upper.tolist(),modes={})
        for mode in ('exact','whole_word','individual'):
            rows=[]
            for sample in capture['samples']:
                original_y = sample['Y'].astype(np.float64)
                clipped_y = np.clip(original_y,lower[:,None,None,None],upper[:,None,None,None])
                original_h = np.einsum('ts,scgp->tcgp',a,original_y)+bias[:,None,None,None]
                clipped_h = np.einsum('ts,scgp->tcgp',a,clipped_y)+bias[:,None,None,None]
                original_gate, clipped_gate = original_h>=theta, clipped_h>=theta
                count = ordinary.measure(dict(sample,Y=clipped_y),a,bias,theta,tm,ts,mode,1.)
                if count['wrong_gates']:
                    raise RuntimeError('Bounded completion disagrees with its own full clipped student')
                count.update(clipped_inputs=int(np.count_nonzero(original_y!=clipped_y)),input_values=original_y.size,
                    gates_changed_by_clipping=int(np.count_nonzero(original_gate!=clipped_gate)),
                    original_spikes=int(original_gate.sum()),
                    clipping_false_negatives=int(np.count_nonzero(original_gate & ~clipped_gate)),
                    clipping_false_positives=int(np.count_nonzero(~original_gate & clipped_gate)))
                rows.append(dict(file=sample['file'],split=sample['split'],**count))
            splits={split:aggregate([r for r in rows if r['split']==split]) for split in ('train','valid')}
            for split,r in splits.items():
                if mode!='exact':
                    reference=axis['modes']['exact']['splits'][split]
                    r.update(W_uses_over_full_clipped= r['logical_W_vector_uses']/reference['logical_W_vector_uses'],
                        PSN_evaluations_over_full_clipped=r['PSN_nonzero_coefficient_evaluations']/reference['PSN_nonzero_coefficient_evaluations'])
            axis['modes'][mode]=dict(frames=rows,splits=splits)
            print(name,mode,json.dumps(splits['valid']),flush=True)
        result['axes'][name]=axis
    result['wall_seconds']=time.monotonic()-started
    (HERE/'bounded_clip_result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
