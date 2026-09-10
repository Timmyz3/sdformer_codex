"""Train-only fit of common temporal columns plus independent diagonal tails.

This is an existing sparse-PSN / conditional-prefix construction. The question
is whether removing cross-time tail fanout reduces expensive Conv production.
No new algebra, exact-model claim, network recovery or RTL timing is implied.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import itertools
import json
from pathlib import Path
import time

import numpy as np
import probe

HERE = Path(__file__).resolve().parent


def row_fit(a, cov, row, cols):
    weight = np.zeros(10)
    columns = np.asarray(cols, dtype=int)
    weight[columns] = np.linalg.lstsq(cov[np.ix_(columns, columns)],
                                    cov[columns] @ a[row], rcond=1e-10)[0]
    delta = a[row] - weight
    return max(0., float(delta @ cov @ delta)), weight


def fit_structure(a, b, mean, cov, anchors, remove):
    # For each fixed three-column set, DP exactly assigns up to three removals
    # among rows. Every diagonal is retained; numerical rank is reported too.
    states = {0: (0., [])}
    for row in range(10):
        extra = sorted(set(anchors) - {row})
        options = []
        for dropped in range(min(remove, len(extra))+1):
            for keep in itertools.combinations(extra, len(extra)-dropped):
                cols = tuple(sorted((row,)+keep))
                error, weight = row_fit(a, cov, row, cols)
                options.append((dropped, error, weight))
        nxt = {}
        for used, (error, weights) in states.items():
            for dropped, local_error, weight in options:
                count = used+dropped
                if count <= remove and (count not in nxt or error+local_error < nxt[count][0]):
                    nxt[count] = (error+local_error, weights+[weight])
        states = nxt
    error, weights = states[remove]
    weight = np.asarray(weights).astype(np.float32)
    bias = (b+(a-weight)@mean).astype(np.float32)
    support = weight != 0
    return dict(anchors=list(anchors), weight=weight.tolist(), bias=bias.tolist(),
                actual_connections=int(support.sum()), matrix_rank=int(np.linalg.matrix_rank(weight)),
                training_membrane_MSE=error/10,
                column_fanout=support.sum(0).tolist())


def aggregate(rows):
    names = [k for k in rows[0] if k not in ('file', 'split')]
    result = {k:(max(r[k] for r in rows) if k == 'peak_allocated_Y_columns'
                 else sum(r[k] for r in rows)) for k in names}
    result['frame_count'] = len(rows)
    return result


def main():
    started = time.monotonic()
    variants = json.loads((HERE.parent/'dependency/fit.json').read_text())['variants']
    native = variants['native']
    a, b = np.asarray(native['weight']), np.asarray(native['bias']).reshape(10)
    moments = np.load(HERE.parent/'dependency/train_moments.npz')
    mean, cov = moments['mean'], moments['covariance']
    candidates = {n:[] for n in (34,37)}
    for anchors in itertools.combinations(range(10), 3):
        for connections in (34,37):
            candidates[connections].append(fit_structure(a,b,mean,cov,anchors,37-connections))
    selected = {}
    for connections, options in candidates.items():
        legal = [v for v in options if v['matrix_rank'] == 10 and v['actual_connections'] == connections]
        selected[f'common3_diagonal_{connections}'] = min(legal, key=lambda v:(v['training_membrane_MSE'],v['anchors']))
    report = dict(
        scope='train32 moment fit; train16/valid4 sampled P4/H8 opportunity only',
        method='120 common-three-column sets; exact row-removal DP for 34; all original t outputs, retained diagonals, actual rank10',
        prior='sparse/masked PSN plus ordinary conditional prefix; column-sparse low rank plus diagonal is not new algebra',
        selection='lowest training membrane MSE, no validation fitting; no gradient/task recovery; GPU first issue width5 subsequently selected for lower train logical W among widths3/5',
        requested_change='each non-anchor Conv time column has exactly one temporal consumer; still shared by 32 p/h lanes',
        control='same raw-zero, lane enable, five Y plus one U; both first-issue widths 3/5; ordinary individual predicate/dependency execution has identical semantics',
        ordinary_row34_training_MSE=variants['row34']['training_membrane_MSE'],
        error_definition='wrong_gates measures approximation versus the SAME selected matrix; full_function errors measure the unpredicted matrix versus native. These errors cannot simply be added or compared as common-target accuracy.',
        selected=selected, axes={},
        exclusions=['complete optical flow AEE', 'physical source/W traffic, banking, coefficient ROM and compensation costs',
                    'service cycles, RTL and PPA'])
    print('SELECTED', json.dumps({k:{n:v[n] for n in ('anchors','actual_connections','matrix_rank','training_membrane_MSE','column_fanout')} for k,v in selected.items()}),flush=True)
    capture = probe.read_torch(HERE/'capture.pt')
    theta = capture['metadata']['neuron_theta']
    structural = selected['common3_diagonal_34']
    old_order = probe.ORDER.copy()
    for variant, params in [('row34',variants['row34']),('common3_diagonal_34',structural)]:
        weight, bias = np.asarray(params['weight']), np.asarray(params['bias']).reshape(10)
        if variant == 'row34':
            probe.ORDER = old_order
        else:
            probe.ORDER = params['anchors']+[t for t in old_order if t not in params['anchors']]
        tm,ts = probe.residual_tables(weight,mean,cov)
        # A three-column first issue is granted equally to the ordinary row34.
        # Subsequent batches fill the same five-slot planner in all modes.
        policies=[(mode,gamma,first) for first in (3,5) for mode,gamma in
                  [('exact',None),('individual',3.),('individual',2.),('whole_word',2.)]]
        for mode,gamma,first in policies:
            key = f'{variant}_{mode}'+('' if gamma is None else f'_g{gamma:g}')
            if first == 5:
                key += '_first5'
            rows=[]
            for sample in capture['samples']:
                counts=probe.measure(sample,weight,bias,theta,tm,ts,mode,gamma,first_batch_size=first)
                y=sample['Y'].astype(np.float64)
                parent_gate=np.einsum('ts,scgp->tcgp',a,y)+b[:,None,None,None]>=theta
                own_gate=np.einsum('ts,scgp->tcgp',weight,y)+bias[:,None,None,None]>=theta
                counts['full_function_gate_difference_vs_native']=int((parent_gate!=own_gate).sum())
                counts['full_function_false_negatives_vs_native']=int((parent_gate & ~own_gate).sum())
                counts['native_active_gates']=int(parent_gate.sum())
                rows.append(dict(file=sample['file'],split=sample['split'],**counts))
            result={split:aggregate([r for r in rows if r['split']==split]) for split in ('train','valid')}
            report['axes'][key]=dict(time_order=probe.ORDER.copy(),first_batch_size=first,splits=result)
            print(key,json.dumps({k:result['valid'][k] for k in ['logical_W_vector_uses','lane_gated_conv1_active_terms','PSN_evaluations_excluding_known_raw_zero_columns','confidence_comparisons','wrong_gates','full_function_gate_difference_vs_native']}),flush=True)
    probe.ORDER=old_order
    report['elapsed_seconds']=time.monotonic()-started
    (HERE/'shared_column_result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
