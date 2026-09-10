#!/opt/anaconda3/bin/python3.12
"""M1/M2: all stored ep34 Q/K, with explicit population and arithmetic units."""
from __future__ import annotations

import collections
import json
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "plan.json").read_text())
CAP = Path(PLAN["input_root"])
ROOT = CAP.parents[2]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def histogram(a, size):
    return np.bincount(np.asarray(a).reshape(-1), minlength=size).tolist()


def run_hist(mask):
    rows = mask.reshape(-1, mask.shape[-1])
    delta = np.diff(np.pad(rows.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    sr, start = np.where(delta == 1)
    er, end = np.where(delta == -1)
    assert np.array_equal(sr, er)
    return histogram(end - start, rows.shape[1] + 1)


def analyze(q, k, captured_gate):
    assert q.shape == k.shape and q.shape[0] == 2 and q.shape[-2:] == (225, 32)
    _, w, h, n, d = q.shape
    qp, kp = q.sum(-1, dtype=np.int16), k.sum(-1, dtype=np.int16)
    qd, kd = np.any(q[0] != q[1], -1), np.any(k[0] != k[1], -1)
    dirty = qd | kd
    kz_pair = (kp[0] == 0) & (kp[1] == 0)
    clean = ~dirty
    pair_count = dirty.size
    overlap = (q & k).sum(-1, dtype=np.int16)
    same_zero = d - qp - kp + overlap
    motion = (k[0] ^ k[1]).sum(-1, dtype=np.int16)
    # Exact rational algebra for ep34 config alpha0=1/50, motion_alpha=1/8.
    # This is not a claim about FP32 reassociation or the old dyadic RTL leaf.
    score_num = 200 * overlap.astype(np.int32) + 4 * same_zero + 25 * motion[None]
    equal_score = score_num[0] == score_num[1]
    assert np.all(equal_score[clean])
    # Layout is the actual score view [window, head, T*N], not QK^T.
    num_rows = score_num.transpose(1, 2, 0, 3).reshape(w, h, 2 * n)
    kz_rows = (kp == 0).transpose(1, 2, 0, 3).reshape(w, h, 2 * n)
    native_scores = (overlap.astype(np.float32) + np.float32(.02) * same_zero
                     + np.float32(.125) * motion[None]) / np.float32(d)
    native_rows = native_scores.transpose(1, 2, 0, 3).reshape(w, h, 2 * n)
    centered = native_rows - native_rows.mean(-1, keepdims=True, dtype=np.float32)
    shifted = centered - centered.max(-1, keepdims=True)
    exp = np.exp2(shifted)
    sums = exp.sum(-1, keepdims=True, dtype=np.float32)
    den = np.exp2(np.ceil(np.log2(np.maximum(sums, np.float32(1e-6)))))
    native_gate = exp / den * np.float32(2 * n)
    diag_gate_q17 = np.rint(native_gate * np.float32(128)).astype(np.int32)
    assert captured_gate.shape == (w, h, 2 * n)
    gate_errors = np.abs(diag_gate_q17 - captured_gate.astype(np.int32))
    capture_pair_gate = captured_gate.reshape(w, h, 2, n).transpose(2, 0, 1, 3)

    # Removing zero-value tokens from normalization is a separate, generally
    # invalid transformation. Keep original scores; compare surviving gates.
    active_rows = ~kz_rows
    has_active = active_rows.any(-1, keepdims=True)
    active_max = np.max(np.where(active_rows, native_rows, -np.inf), -1, keepdims=True)
    active_max = np.where(has_active, active_max, 0)
    skip_exp = np.where(active_rows, np.exp2(native_rows - active_max), 0)
    skip_sum = skip_exp.sum(-1, keepdims=True, dtype=np.float32)
    skip_den = np.exp2(np.ceil(np.log2(np.maximum(skip_sum, np.float32(1e-6)))))
    removed_gate = skip_exp / skip_den * np.float32(2 * n)
    changed_active = active_rows & (np.abs(removed_gate - native_gate) > np.float32(1e-6))

    # Count duplicate *score values* among K-zero entries within each real row.
    # Their weighted contribution may be represented compactly but cannot be
    # discarded. Discovery/lookup/exp/reduction costs remain unpaid.
    zero_score_classes = 0
    zero_nonempty_rows = 0
    for nums, zeros in zip(num_rows.reshape(-1, 2*n), kz_rows.reshape(-1, 2*n)):
        if zeros.any():
            zero_nonempty_rows += 1
            zero_score_classes += int(np.unique(nums[zeros]).size)

    c = {
        "paired_head_positions": int(pair_count),
        "temporal_head_tokens": int(2*pair_count),
        "lane_bits_per_q_or_k": int(q.size),
        "q_one_bits": int(q.sum()), "k_one_bits": int(k.sum()),
        "q_dirty_pairs": int(qd.sum()), "k_dirty_pairs": int(kd.sum()),
        "dirty_pairs": int(dirty.sum()), "clean_pairs": int(clean.sum()),
        "q_only_dirty_pairs": int((qd & ~kd).sum()),
        "k_only_dirty_pairs": int((kd & ~qd).sum()),
        "both_dirty_pairs": int((qd & kd).sum()),
        "k_zero_both_times_pairs": int(kz_pair.sum()),
        "k_zero_temporal_tokens": int((kp == 0).sum()),
        "grok_dirty_and_not_pair_kzero": int((dirty & ~kz_pair).sum()),
        "normalization_rows": int(w*h),
        "rows_with_any_dirty_pair": int(dirty.any(-1).sum()),
        "rows_with_all_pair_scores_equal_rational": int(equal_score.all(-1).sum()),
        "pair_scores_equal_rational": int(equal_score.sum()),
        "pair_gate_codes_equal_capture": int((capture_pair_gate[0] == capture_pair_gate[1]).sum()),
        "diagnostic_gate_q17_checked": int(captured_gate.size),
        "diagnostic_gate_q17_mismatch": int(np.count_nonzero(gate_errors)),
        "diagnostic_gate_q17_abs_error_sum": int(gate_errors.sum()),
        "active_tokens_if_zero_K_removed": int(active_rows.sum()),
        "active_gates_changed_if_zero_K_removed_diagnostic": int(changed_active.sum()),
        "rows_with_changed_active_gate_if_zero_K_removed_diagnostic": int(changed_active.any(-1).sum()),
        "zero_K_score_classes_per_row_summed": zero_score_classes,
        "rows_with_zero_K": zero_nonempty_rows,
        "overlap_sum_two_times": int(overlap.sum()),
        "same_zero_sum_two_times": int(same_zero.sum()),
        "motion_sum_two_times": int(2*motion.sum()),
    }
    for zero in (0, 1):
        for change in (0, 1):
            c[f"pair_kzero{zero}_dirty{change}"] = int(((kz_pair == zero) & (dirty == change)).sum())
    assert sum(c[f"pair_kzero{z}_dirty{a}"] for z in (0,1) for a in (0,1)) == pair_count
    hist = {
        "q_popcount_two_times": histogram(qp, 33),
        "k_popcount_two_times": histogram(kp, 33),
        "overlap_two_times": histogram(overlap, 33),
        "same_zero_two_times": histogram(same_zero, 33),
        "motion_one_per_pair": histogram(motion, 33),
        "dirty_run_length_within_225_positions": run_hist(dirty),
        "clean_run_length_within_225_positions": run_hist(clean),
    }
    assert sum(hist['overlap_two_times']) == 2*pair_count
    assert sum(i*v for i,v in enumerate(hist['dirty_run_length_within_225_positions'])) == c['dirty_pairs']
    assert sum(i*v for i,v in enumerate(hist['clean_run_length_within_225_positions'])) == c['clean_pairs']
    return c, hist, int(gate_errors.max()), float(np.max(np.abs(removed_gate-native_gate)[active_rows], initial=0))


def aggregate(records):
    counts = collections.Counter()
    hist = {}
    for r in records:
        counts.update(r['counts'])
        for key, a in r['histograms'].items():
            hist.setdefault(key, np.zeros(len(a), dtype=np.int64))
            hist[key] += a
    pair = counts['paired_head_positions']; token = counts['temporal_head_tokens']
    row = counts['normalization_rows']; bit = counts['lane_bits_per_q_or_k']
    ratios = {
        'dirty_pair_fraction': counts['dirty_pairs']/pair,
        'K_zero_both_times_pair_fraction': counts['k_zero_both_times_pairs']/pair,
        'K_zero_temporal_token_fraction_output_product_only': counts['k_zero_temporal_tokens']/token,
        'grok_heuristic_remaining_pair_fraction_NOT_legal_score_schedule': counts['grok_dirty_and_not_pair_kzero']/pair,
        'row_dirty_pair_OR_fraction': counts['rows_with_any_dirty_pair']/row,
        'row_all_pairs_clean_fraction': 1-counts['rows_with_any_dirty_pair']/row,
        'clean_pair_reuse_savings_upper_bound_of_two_score_evaluations': counts['clean_pairs']/token,
        'rational_score_equal_pair_fraction': counts['pair_scores_equal_rational']/pair,
        'rational_score_reuse_posthoc_upper_bound_of_two_evaluations': counts['pair_scores_equal_rational']/token,
        'capture_Q17_pair_gate_equal_fraction_NOT_FP32_equality': counts['pair_gate_codes_equal_capture']/pair,
        'q_bit_density': counts['q_one_bits']/bit,
        'k_bit_density': counts['k_one_bits']/bit,
        'overlap_mean_per_temporal_head_token': counts['overlap_sum_two_times']/token,
        'same_zero_mean_per_temporal_head_token': counts['same_zero_sum_two_times']/token,
        'motion_mean_per_temporal_head_token': counts['motion_sum_two_times']/token,
        'overlap_exact_zero_fraction': int(hist['overlap_two_times'][0])/token,
        'same_zero_exact_zero_fraction': int(hist['same_zero_two_times'][0])/token,
        'motion_exact_zero_fraction': int(hist['motion_one_per_pair'][0])/pair,
    }
    return {'records': len(records), 'counts': dict(counts), 'ratios': ratios,
            'histograms':{k:a.tolist() for k,a in hist.items()},
            'diagnostic_gate_q17_max_abs_error':max(r['diagnostic_gate_q17_max_abs_error'] for r in records)}


def main():
    out = HERE/'census_r1'
    out.mkdir(exist_ok=True)
    start=time.time()
    manifest=json.loads((CAP/'manifest.json').read_text())
    qm=json.loads((CAP/'attention_qk/manifest.json').read_text())
    identity={'checkpoint':'Motion C12 ep34', 'bn_policy':'no_running',
              'source_capture':str(CAP), 'samples':40, 'quantization_enabled':False}
    cohort={r['global_sample_id']:r for r in manifest['cohort']['samples']}
    all_names={r['name'] for r in qm['records']}
    records=[]
    log=(out/'records.jsonl').open('w')
    for i,r in enumerate(sorted(qm['records'],key=lambda r:(r['sample_id'],r['name']))):
        f=CAP/'attention_qk'/Path(r['file']).name
        with np.load(f,allow_pickle=False) as z:
            shape=tuple(map(int,z['q_shape'])); kshape=tuple(map(int,z['k_shape']))
            assert shape==kshape==(2,r['windows_captured'],r['heads'],225,32)
            q=np.unpackbits(z['q_bits_packed'],bitorder='little',count=int(np.prod(shape))).reshape(shape)
            k=np.unpackbits(z['k_bits_packed'],bitorder='little',count=int(np.prod(shape))).reshape(shape)
            assert q.sum()==r['q_active_bits'] and k.sum()==r['k_active_bits']
            gate=z['gate_q17']
            assert np.count_nonzero(gate)==r['gate_nonzero']
            c,h,ge,changed=analyze(q,k,gate)
        s=cohort[r['sample_id']]
        assert s['sample_key']==r['sample_key']
        item={'sample_id':r['sample_id'],'sample_key':r['sample_key'],'sequence':s['sequence'],
              'block':r['name'],'stage':r['name'].split('.')[0], 'file':f.name,
              'windows_captured':r['windows_captured'],'windows_total':r['windows_total'],
              'counts':c,'histograms':h,'diagnostic_gate_q17_max_abs_error':ge,
              'diagnostic_zero_K_removal_max_active_gate_change':changed}
        records.append(item);log.write(json.dumps(item)+'\n');log.flush()
        if (i+1)%24==0: print(f'QK records {i+1}/480, elapsed {time.time()-start:.1f}s',flush=True)
    log.close()
    agg=aggregate(records)
    per_stage={s:aggregate([r for r in records if r['stage']==s]) for s in ['S0','S1','S2','S3']}
    per_block={s:aggregate([r for r in records if r['block']==s]) for s in sorted(all_names)}
    per_seq={s:aggregate([r for r in records if r['sequence']==s]) for s in sorted({r['sequence'] for r in records})}
    per_sample={str(s):aggregate([r for r in records if r['sample_id']==s]) for s in range(40)}
    coverage={}
    for stage in ['S0','S1','S2','S3']:
        rr=[r for r in records if r['stage']==stage]
        pairs=sum(r['counts']['paired_head_positions'] for r in rr)
        total=sum(r['windows_total']*(3*2**int(stage[1]))*225 for r in rr)
        missing=total-pairs
        coverage[stage]={'captured_windows':sum(r['windows_captured'] for r in rr),
                         'total_windows':sum(r['windows_total'] for r in rr),
                         'captured_paired_head_positions':pairs,'total_paired_head_positions':total,
                         'captured_fraction':pairs/total,'selection':'first min(100,windows_total) windows in native order',
                         'full_population_no_assumption_bounds':{
                             key:[per_stage[stage]['counts'][key]/total,(per_stage[stage]['counts'][key]+missing)/total]
                             for key in ['dirty_pairs','k_zero_both_times_pairs','clean_pairs']}}
    result={'status':'PASS_ARCHIVED_EP34_CAPTURE_CENSUS_NOT_FULL_FRAME_OR_RTL',
            'identity':identity,'records':480,'coverage':coverage,'aggregate_captured_only':agg,
            'per_stage':per_stage,'per_block':per_block,'per_sequence':per_seq,'per_sample':per_sample,
            'elapsed_seconds':time.time()-start,
            'limits':['40 samples from4 sequences, selected earlier for C1/decoder; not valid825 or random sampling',
                      'prefix windows at S0/S1; S2/S3 complete for these samples; no population-average inference',
                      'paired head-position is not one scalar lane or one physical spatial token',
                      'Q/K positive support preserves score events, not the full theta amplitude of K-as-V',
                      'rational ep34 score and NumPy gate diagnostics do not prove FP32 reassociation equivalence',
                      'native config alpha0=.02,motion=.125,quant=false differs from old Q7 RTL alpha0=1/64,motion=1/4',
                      'clean pairs can save at most one of two score computations before control costs',
                      'K-zero local products may vanish while row normalization contributions must remain',
                      'posthoc exact score classes/equality need discovery and storage; no free hardware reuse claim']}
    dump(out/'result.json',result)
    print(json.dumps({'status':result['status'],'aggregate_ratios':agg['ratios'],'gate_diagnostic_mismatches':agg['counts']['diagnostic_gate_q17_mismatch'],'coverage':{s:v['captured_fraction'] for s,v in coverage.items()}},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
