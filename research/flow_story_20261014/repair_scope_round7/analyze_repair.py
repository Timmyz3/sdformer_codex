#!/usr/bin/env python3
"""Fixed four-state descriptive decomposition; no ranking, fitting or GPU."""
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'damage_predictor_round6'
SOURCES = {'CC': OLD/'collect_v1/results.json', 'FF': OLD/'matched_fine_check_v1/results.json',
           'native': OLD/'native_check_v1/results.json', 'mixed': HERE/'run_v1/results.json'}
ARM = {'CC': 'coarse_stage02', 'FF': 'fine_stage02', 'native': 'native_dense',
       'CF': 'C0_F2', 'FC': 'F0_C2'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    raw = {k: json.loads(p.read_text()) for k, p in SOURCES.items()}
    mixed = raw['mixed']; assert mixed['status'] == 'ROUND7_MIXED_SCOPE_DIAGNOSTIC_COMPLETE'
    assert mixed['full_forward_count'] == 20 and len(mixed['frames']) == 8
    assert all(x['exact_archived_prediction'] for x in mixed['anchor_replays'].values())
    assert all(x['flow_exact'] and x['all8_branches_exact'] for x in mixed['dual_checks'].values())
    assert all(x['original_state_sha256'] == mixed['original_state_sha256'] for x in raw.values())
    ids = [f['target_id'] for f in mixed['frames']]; assert len(set(ids)) == 8
    by_state = {k: {f['target_id']: f for f in raw['mixed' if k in ('CF','FC') else k]['frames'] if f['target_id'] in ids} for k in ARM}
    frames = []
    for tid in ids:
        entries = {k: v[tid] for k,v in by_state.items()}
        assert all(f['input_evidence'] == entries['CC']['input_evidence'] and f['timestamps'] == entries['CC']['timestamps'] for f in entries.values())
        regions = {k: f['arms'][ARM[k]]['metrics']['regions'] for k,f in entries.items()}
        out = {}
        for r in regions['CC']:
            n = regions['CC'][r]['valid_pixels']
            assert all(x[r]['valid_pixels'] == n for x in regions.values())
            e = {k: x[r]['AEE'] for k,x in regions.items()}
            out[r] = {'valid_pixels': n, 'AEE': e, 'error_sum': {k:x[r]['error_sum'] for k,x in regions.items()},
                      'stage0_alone_damage': e['CF']-e['FF'] if n else None,
                      'stage2_alone_damage': e['FC']-e['FF'] if n else None,
                      'restore_stage0_gain': e['CC']-e['FC'] if n else None,
                      'restore_stage2_gain': e['CC']-e['CF'] if n else None,
                      'interaction': e['CC']-e['CF']-e['FC']+e['FF'] if n else None}
        frames.append({'target_id':tid, 'sequence':entries['CC']['sequence'], 'regions':out})
    summaries = {}
    for scope in ['overall']+sorted({f['sequence'] for f in frames}):
        selected = frames if scope == 'overall' else [f for f in frames if f['sequence']==scope]
        summary = {}
        for r in frames[0]['regions']:
            data = [f['regions'][r] for f in selected if f['regions'][r]['valid_pixels']]
            count = sum(x['valid_pixels'] for x in data)
            summary[r] = {'nonempty_frames':len(data), 'valid_pixels':count,
                          'frame_mean_AEE': {k:mean(x['AEE'][k] for x in data) if data else None for k in ARM},
                          'pixel_weighted_AEE': {k:sum(x['error_sum'][k] for x in data)/count if count else None for k in ARM},
                          'frame_mean_effects': {k:mean(x[k] for x in data) if data else None for k in ('stage0_alone_damage','stage2_alone_damage','restore_stage0_gain','restore_stage2_gain','interaction')}}
        summaries[scope] = summary
    result = {'schema':'flow_round7_four_state_analysis_v1', 'source_sha256':{str(p):sha(p) for p in SOURCES.values()},
              'script_sha256':sha(Path(__file__)), 'frames':frames, 'summaries':summaries,
              'scope':'Reused development members, post-result descriptive factorial; no causal attribution to spikes/BN, no policy or hardware gain'}
    output = HERE/'analysis.json'; assert not output.exists()
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({r:summaries['overall'][r] for r in ('all_valid','B','L')},indent=2))
    print('Per-frame allGT:',json.dumps({f['target_id']:f['regions']['all_valid'] for f in frames},indent=2))


if __name__=='__main__':main()
