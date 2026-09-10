"""H8 coefficient-vector request proxies; no new model forward.

The train16/valid4 payload is the fixed-four-BN floating-point parent.  Its
source is unchanged by the later r1 Conv1/PSN integer replacement.  Its Y is
NOT an integer-deployment capture.  P4 OR source words give exact requests
only for all-or-none P4/H8 column enables; arbitrary lane enables need the
four separate source masks available only in the four integer NPZs.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PATCH = HERE.parents[1]
ROOT = PATCH.parents[1]
sys.path.insert(0, str(ROOT / 'bn_state'))
from support_service_model import read_torch


def group_union_cost(source_words, pending_columns, w_any):
    """One K sweep, W[k,H8] used across every requested time in that sweep.

    source_words [G,K], pending_columns [G,Hgroup,T] bool; returns [G,Hgroup].
    A separate sweep/cold reuse epoch must call this again, not reuse its cost.
    """
    need_word = (pending_columns.astype(np.int64) * (1 << np.arange(10))).sum(-1)
    return (((source_words[:, None, :] & need_word[..., None]) != 0)
            & w_any[None]).sum(-1, dtype=np.int64)


def main():
    c = PATCH / 'partial_completion'
    saved = read_torch(c / 'capture.pt')
    dep = np.load(c / 'integer_deployment/common3_diagonal_34.npz')
    w = dep['weight_int8'].reshape(12, 8, 864)
    w_any = np.any(w != 0, axis=1)
    w_fp = np.load(c/'shared_column_deployment_source.npz')['weight'].reshape(12,8,864)
    fp_any = np.any(w_fp != 0, axis=1)
    assert np.array_equal(w_any,fp_any)
    samples = saved['samples']
    words = np.stack([s['source_words'] for s in samples]).astype(np.int16)
    active = (words[..., None] & (1 << np.arange(10))) != 0
    column = np.einsum('sgkt,hk->sght', active.astype(np.int64), w_any.astype(np.int64))
    baseline = np.stack([group_union_cost(z, np.ones((64, 12, 10), bool), w_any) for z in words])
    prefix = np.zeros((64, 12, 10), bool)
    prefix[..., [2, 3, 7]] = True
    prefix_cost = np.stack([group_union_cost(z, prefix, w_any) for z in words])
    np.savez_compressed(
        HERE / 'training_request_costs.npz',
        files=np.asarray([s['file'] for s in samples]),
        splits=np.asarray([s['split'] for s in samples]),
        source_words=words, group_ids=saved['groups'],
        column_H8_vector_uses=column,full_batch_H8_vector_uses=baseline,
        prefix_batch_H8_vector_uses=prefix_cost,
        column_W64_uses=column, full_batch_W64_uses=baseline,
        prefix_batch_W64_uses=prefix_cost,
        W_int8=w, W_nonzero_h_k=(w != 0), W_any_H8_k=w_any,
        W_float32=w_fp,FP32_W_nonzero_h_k=(w_fp!=0),FP32_W_any_H8_k=fp_any,
        source_theta=np.asarray(saved['metadata']['source_theta']),
    )
    alignment = []
    for f in sorted((c / 'integer_valid10').glob('capture_*.npz')):
        z = np.load(f)
        index = next(i for i, s in enumerate(samples) if s['file'] == str(z['file']))
        restored = np.bitwise_or.reduce(z['source_gate_words'], axis=-1)
        counts = (((z['source_gate_words'][..., None] >> np.arange(10)) & 1).sum(1))
        alignment.append(dict(
            file=str(z['file']), source_OR_differences=int(np.count_nonzero(restored != words[index])),
            source_count_differences=int(np.count_nonzero(counts != samples[index]['source_active_terms'])),
            source_entries=int(restored.size),
        ))
    result = dict(
        source='existing capture.pt: train16 + valid4, 64 aligned P4 per frame; full T10 and C96',
        numeric_identity='Y remains the pre-integer fixed-four-BN parent; only source support is shared with integer student',
        weight_identity='exports both original Conv1 FP32 W and later common3/row34 shared BN-gain-folded W8; local trained student still uses original FP32 Conv1',
        arrays='source_words[S,64,864], column_H8_vector_uses[S,64,12,10]; legacy *_W64_uses fields are identical count aliases valid as64-bit words ONLY for W8 deployment',
        original_FP32_weight_elements=int(w_fp.size),original_FP32_weight_nonzero=int(np.count_nonzero(w_fp)),
        original_FP32_H8_vectors_with_nonzero=int(fp_any.sum()),
        layout='FP32 H8 vector=256bit/four64bit beats, fullW331776B; W8 H8 vector=64bit/one beat, fullW82944B. Vector ratios do not prove same pools/service.',
        W64_vectors_with_some_nonzero=int(w_any.sum()), total_W64_vectors=int(w_any.size),
        cost_definition='sum_k OR_t(source_word[g,k,t] AND pending_column[g,H8,t]) AND any_h(W[k,h]!=0)',
        all_or_none_column='pending column is max of unresolved dependent output gates over actual P4/H8 lanes; use E[row,t]',
        arbitrary_lane_limit='P4 OR loses which p generated the bit; do not call the proxy exact for per-p/h retirement',
        schedule_boundary='one physical W reuse epoch; sum column costs only when time-major emits distinct K sweeps',
        source_port_limit='these are H8 coefficient-vector uses, not source-cache/DMA requests or issued physical beats; distinct consumer unions apply there',
        current_W_fact='Every H8 coefficient vector contains a nonzero; all-or-none column cost is identical across H8 groups',
        source_alignment=alignment,
        split_totals={},
    )
    for split in ('train', 'valid'):
        ids=[i for i,s in enumerate(samples) if s['split']==split]
        result['split_totals'][split]=dict(
            frames=len(ids), P4_groups=len(ids)*64,
            time_major_H8_vector_uses=int(column[ids].sum()),
            one_full_T_batch_H8_vector_uses=int(baseline[ids].sum()),
            one_prefix_batch_H8_vector_uses=int(prefix_cost[ids].sum()),
        )
    (HERE/'training_request_costs.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result['split_totals'], ensure_ascii=False))


if __name__ == '__main__':
    main()
