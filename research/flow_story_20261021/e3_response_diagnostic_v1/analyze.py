#!/usr/bin/env python3
"""Frozen CPU-only response representability; no torch/model/fit/GPU imports."""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE/'independent_review'))
import reduce_e1_quality_v1 as old

raw, mc = old.raw, old.mathcore
ACTS = [f'upgrade_{i}' for i in range(6)]
POLICIES = ['response_norm', 'signed_full', 'signed_grid4', 'signed_grid1', 'activity_max_bins']
EPS = 1e-12


def sha(p):
    return old.sha(p)


def read(p):
    return json.loads(Path(p).read_text())


def write(p, d):
    Path(p).write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')


def ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='stable'); out = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i+1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        out[order[i:j]] = (i+j-1)/2+1
        i = j
    return out


def rank_metrics(score, truth):
    a, b = ranks(score), ranks(truth)
    ac, bc = a-a.mean(), b-b.mean()
    denom = float(np.sqrt(np.sum(ac*ac)*np.sum(bc*bc)))
    rho = float(np.dot(ac, bc)/denom) if denom else None
    pair = 0.; predicted_ties = actual_ties = both_ties = 0
    for i in range(6):
        for j in range(i+1, 6):
            aa = int(np.sign(score[i]-score[j])); bb = int(np.sign(truth[i]-truth[j]))
            predicted_ties += aa == 0; actual_ties += bb == 0; both_ties += aa == bb == 0
            pair += 1 if aa == bb else .5 if aa == 0 or bb == 0 else 0
    picked = min(range(6), key=lambda i: (score[i], i))
    oracle = min(range(6), key=lambda i: (truth[i], i))
    return dict(spearman=rho, pairwise_agreement=pair/15, predicted_tie_pairs=predicted_ties,
        actual_tie_pairs=actual_ties, both_tie_pairs=both_ties, picked_action=picked,
        exact_oracle_action=oracle, hit_oracle_action=picked == oracle,
        hit_best_true_value=truth[picked] == min(truth), selected_actual_delta_AEE=truth[picked],
        regret_AEE=truth[picked]-truth[oracle],
        predicted_order=sorted(range(6), key=lambda i:(score[i],i)),
        actual_order=sorted(range(6), key=lambda i:(truth[i],i)))


def grid_labels(vector, mask, grid):
    h, w = mask.shape
    assert h % grid == w % grid == 0 and vector.shape == (2,h,w)
    bh, bw = h//grid, w//grid
    count = mask.reshape(grid,bh,grid,bw).sum(axis=(1,3),dtype=np.int64)
    sums = (vector*mask[None]).reshape(2,grid,bh,grid,bw).sum(axis=(2,4),dtype=np.float64)
    mean = np.divide(sums,count[None],out=np.zeros_like(sums),where=count[None]>0)
    return count, mean


def contraction(count, a, b):
    n = int(count.sum()); assert n > 0
    return float(np.sum(count*np.sum(a*b,axis=0),dtype=np.float64)/n)


def same_metrics(a, b):
    assert set(a) == set(b) == set(mc.REGIONS)
    for reg in mc.REGIONS:
        aa, bb = a[reg], b[reg]
        assert aa['valid_pixels'] == bb['valid_pixels']
        assert math.isclose(aa['error_sum'],bb['error_sum'],rel_tol=1e-12,abs_tol=1e-8)
        assert (aa['AEE'] is None) == (bb['AEE'] is None)
        if aa['AEE'] is not None:
            assert math.isclose(aa['AEE'],bb['AEE'],rel_tol=1e-12,abs_tol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    bindings = read(args.bindings)
    assert bindings['schema'] == 'flow_e3_response_prerun_bindings_v1'
    for rel,digest in bindings['files'].items():
        assert sha(ROOT/rel) == digest, rel
    quality_path = BASE/'independent_review/E3_ACTUAL_RAW_QUALITY_V1.json'
    q = read(quality_path)
    audit = read(BASE/'independent_review/E3_ACTUAL_AUDIT_V1.json')
    run = BASE/'actual/e3_rne_probe_v1'; result = read(run/'results.json')
    assert q['status'] == 'COMPLETE' and q['execution_review']['status'] == 'PASS'
    assert audit['status'] == 'PASS' and audit['quality_analysis_allowed']
    assert audit['result_sha256'] == q['execution_review']['result_sha256'] == sha(run/'results.json')
    assert result['status'] == 'E3_RNE112_AND6_DENSE_REFERENCES_COMPLETE'
    ids = result['selected_ids']; assert len(ids) == len(set(ids)) == 16
    qframes = {f['target_id']:f for f in q['frames']}
    members_path = old.R27/'independent_review/membership_v1.json'
    assert sha(members_path) == result['sources']['members']
    records = {r['target_id']:r for r in read(members_path)['records']}
    regions_path = ROOT/'research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json'
    assert sha(regions_path) == old.REGIONS_SHA
    regions = read(regions_path)
    # Revalidate all48 source bytes and112 predictions before inspecting GT values.
    sources, predictions = {}, {}
    for tid in ids:
        sources[tid] = {}
        for rec in records[tid]['source_files']:
            kind = {'input':'events','gt':'gt','mask':'mask'}[rec['category']]
            path = old.child(old.R27/'dev64_source_snapshot',rec['relative_path'])
            assert sha(path) == rec['sha256'] and path.stat().st_size == int(rec['bytes'])
            sources[tid][kind] = path
    for state in ['background_Q4']+ACTS:
        for frame in result['evaluations'][state]['frames']:
            predictions[state,frame['target_id']] = old.prediction(run,frame)
    source_bound = dict(source_arrays=48,predictions=112,source_bytes_bound_before_GT_values=True)
    foots = []
    first_patch = result['action_contract']['family_targets'][0]
    for g in result['action_contract']['groups']:
        small = np.zeros((240,320),dtype=bool)
        for y,x in g['tiles'][first_patch]:
            small[9*y:9*y+9,9*x:9*x+9] = True
        foots.append(np.repeat(np.repeat(small,2,axis=0),2,axis=1))
    frames, details = [], []
    for tid in ids:
        arrays = {k:np.load(p,allow_pickle=False) for k,p in sources[tid].items()}
        masks = raw.region_masks(arrays['gt'],arrays['mask'],arrays['events'],
            regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels'])
        valid = masks['all_valid']; n = int(valid.sum()); assert n > 0
        with np.load(predictions['background_Q4',tid],allow_pickle=False) as z:
            background = z['head3'].copy()
        f0 = background.astype(np.float64); gt = arrays['gt'].astype(np.float64)
        e = f0-gt; length = np.sqrt(np.sum(e*e,axis=0))
        unit = e/np.maximum(length,EPS)[None]
        measured = {'background_Q4':raw.measure(background,arrays['gt'],masks)}
        same_metrics(measured['background_Q4'],qframes[tid]['states']['background_Q4'])
        cnt4,u4 = grid_labels(unit,valid,4); cnt1,u1 = grid_labels(unit,valid,1)
        score = {p:[] for p in POLICIES}; grid4_responses = []; grid1_responses = []
        truth = []; zero_norm = int((valid & (length == 0)).sum())
        bins = np.count_nonzero(arrays['events'],axis=0)
        for i, state in enumerate(ACTS):
            with np.load(predictions[state,tid],allow_pickle=False) as z:
                pred = z['head3'].copy()
            measured[state] = raw.measure(pred,arrays['gt'],masks)
            same_metrics(measured[state],qframes[tid]['states'][state])
            delta = pred.astype(np.float64)-f0
            score['response_norm'].append(float(np.sum(np.sqrt(np.sum(delta*delta,axis=0))[valid],dtype=np.float64)/n))
            score['signed_full'].append(float(np.sum(np.sum(unit*delta,axis=0)[valid],dtype=np.float64)/n))
            c4,d4 = grid_labels(delta,valid,4); c1,d1 = grid_labels(delta,valid,1)
            assert np.array_equal(c4,cnt4) and np.array_equal(c1,cnt1)
            score['signed_grid4'].append(contraction(cnt4,u4,d4))
            score['signed_grid1'].append(contraction(cnt1,u1,d1))
            grid4_responses.append(d4.tolist()); grid1_responses.append(d1.tolist())
            # Deliberately no valid/GT mask in the activity control.
            activity = int(bins[foots[i]].sum())
            assert activity == q['action_coverage'][tid][state]['nonzero_voxel_bins']
            score['activity_max_bins'].append(-activity)
            truth.append(measured[state]['all_valid']['AEE']-measured['background_Q4']['all_valid']['AEE'])
        metrics = {p:rank_metrics(score[p],truth) for p in POLICIES}
        assert ACTS[metrics['signed_full']['exact_oracle_action']] == q['policy_choices'][tid]['single_action_oracle']
        states = copy.deepcopy(qframes[tid]['states'])
        for p in POLICIES:
            states[p] = measured[ACTS[metrics[p]['picked_action']]]
        frames.append(dict(target_id=tid,sequence=qframes[tid]['sequence'],states=states))
        details.append(dict(target_id=tid,sequence=qframes[tid]['sequence'],valid_pixels=n,
            zero_background_error_valid_pixels=zero_norm,true_delta_AEE=truth,scores=score,rank_metrics=metrics,
            grid4=dict(shape=[4,4],support_counts=cnt4.tolist(),unit_direction_means=u4.tolist(),
                       response_means_by_action=grid4_responses),
            grid1=dict(shape=[1,1],support_counts=cnt1.tolist(),unit_direction_means=u1.tolist(),
                       response_means_by_action=grid1_responses)))
        print(tid, 'complete',flush=True)
    seqs = list(dict.fromkeys(f['sequence'] for f in frames)); assert len(seqs) == 8
    states = q['states']+POLICIES
    summaries = {s:{scope:{reg:mc.aggregate([f['states'][s][reg] for f in frames
        if scope == 'overall' or f['sequence'] == scope]) for reg in mc.REGIONS}
        for scope in ['overall']+seqs} for s in states}
    comparisons = {p+'__vs__'+ref:mc.comparison(frames,summaries,p,ref,seqs)
        for p in POLICIES for ref in ('background_Q4','single_action_oracle','fit_fold_static','response_norm') if p != ref}
    rank_summary = {}
    for p in POLICIES:
        rank_summary[p] = {}
        for scope in ['overall']+seqs:
            rows = [d for d in details if scope == 'overall' or d['sequence'] == scope]
            rms = [d['rank_metrics'][p] for d in rows]
            rho = [m['spearman'] for m in rms if m['spearman'] is not None]
            regret = [m['regret_AEE'] for m in rms]
            regpix = math.fsum(d['valid_pixels']*d['rank_metrics'][p]['regret_AEE'] for d in rows)/sum(d['valid_pixels'] for d in rows)
            rank_summary[p][scope] = dict(identity_count=len(rows),spearman_mean=math.fsum(rho)/len(rho) if rho else None,
                spearman_defined_count=len(rho),pairwise_agreement_mean=math.fsum(m['pairwise_agreement'] for m in rms)/len(rms),
                exact_oracle_action_hits=sum(m['hit_oracle_action'] for m in rms),
                regret_frame_mean_AEE=math.fsum(regret)/len(regret),regret_pixel_weighted_AEE=regpix,
                regret_quantiles=mc.quantiles(regret),
                improved_Q4_frames=sum(m['selected_actual_delta_AEE']<0 for m in rms),
                worsened_Q4_frames=sum(m['selected_actual_delta_AEE']>0 for m in rms))
    output = dict(schema='flow_e3_response_representability_v1',status='COMPLETE',
        sources=dict(pre_run_bindings_sha256=sha(args.bindings),runner_sha256=sha(__file__),
                     raw_math_sha256=sha(raw.__file__),scalar_math_sha256=sha(mc.__file__),**source_bound),
        model_forwards=0,GPU_used=False,head_fitting=False,GT_and_real_counterfactuals_used=True,
        activity_control_uses_GT=False,online_cost_modeled=False,grids_evaluated=[4,1],
        primary='signed_grid4',full_signed_is_not_mathematical_upper_bound=True,
        states=states,frames=frames,summaries=summaries,comparisons=comparisons,
        action_diagnostics=details,rank_summary=rank_summary,
        baseline_pixel_headroom_to_finite_oracle=summaries['background_Q4']['overall']['all_valid']['pixel_weighted_AEE']-summaries['single_action_oracle']['overall']['all_valid']['pixel_weighted_AEE'],
        scope='Post-E3 offline ideal-label representability on16 exposed identities/8 sequences, not predictor success or novelty.',
        activity_scope='One predeclared max full-footprint nonzero-voxel-bin rule, simple control only; no online cost modeled.',
        legacy_thresholds_are_review_references_only=True)
    write(args.output,output)
    print(json.dumps({p:dict(pixel_AEE=summaries[p]['overall']['all_valid']['pixel_weighted_AEE'],
        B=summaries[p]['overall']['B']['pixel_weighted_AEE'],L=summaries[p]['overall']['L']['pixel_weighted_AEE'],
        **rank_summary[p]['overall']) for p in POLICIES},indent=2))


if __name__ == '__main__':
    main()
