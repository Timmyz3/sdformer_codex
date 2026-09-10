"""Correct bookkeeping for already selected organizations, without a new sweep."""
import sys
sys.dont_write_bytecode = True
import json
import time
import numpy as np
from service_model import ROOT, R, Simulation, kernel_trace, Graph


def check_row(meta, data, row):
    counts = row['counters']
    first = int(meta['source_active'])*meta['H']
    expected = 2*first if row['axis']=='recompute' else first
    if row['axis']!='packet':
        assert counts['fc1_scalar_updates']==expected
        assert counts['psn_scalar_macs']==meta['P']*meta['H']*meta['A_nonzero']
    else:
        assert counts['fc1_scalar_updates']>=first
        assert counts['required_failed_h_tiles']==int(data['failure'].sum())
        assert counts['repaired_h_tiles']>=counts['required_failed_h_tiles']
        counts['repair_channel_amplification']=(counts['repaired_h_tiles']/counts['required_failed_h_tiles'])
        counts['repair_scalar_updates']=counts['fc1_scalar_updates']-first
        counts['repair_psn_macs']=counts['psn_scalar_macs']-meta['P']*meta['H']*meta['A_nonzero']
    assert row['state_live_peak_bytes']<=R['state_capacity_bytes']
    assert row['wide_work_live_max_bytes']<=row['reserved_wide_work_bytes']
    if row['axis']=='gram':
        assert row['bytes']['gram_threshold_write']==meta['H']*meta['T']*8
        assert row['bytes']['gram_threshold_read']==meta['H']*meta['T']*8
        assert counts['gram_contraction_macs']==meta['H']*(meta['C']**2+2*meta['C'])


def main():
    started=time.time()
    prior=json.loads((ROOT/'service_result.json').read_text())
    result=dict(scope=prior['scope'],resources=R,layers=[],
                policy='Rerun the five per-axis winners selected in the initial organization screen. No new parameter search.',
                corrections=['Store and service all H*T Float64 thresholds in Gram state.',
                             'Charge immutable affine parameter reads before their producer use.',
                             'Explicit 64KiB moment/parameter scratch and 32KiB metadata directory.',
                             'Charge failure bitmap writes/scans and two-descriptor dispatch.',
                             'Use 32-byte output descriptors, including selective h masks.',
                             'Do not encode nonexistent tail positions in the final partial B32 tile.'],
                limitations=prior['limitations']+[
                    'Initial per-axis winners remain fixed after bookkeeping correction; they are strong implemented controls, not a proof of the globally optimal schedule.',
                    'Each source/FC1/PSN service operation uses real captured support, but the service simulator does not emulate each hardware IEEE-754 FMA rounding.',
                    'A8 packet encoder is evaluated at q96; no claim of a conflict-free arbitrary-q gather network.'])
    sanity=Graph(); a=sanity.add('a','a',7); b=sanity.add('b','b',11); sanity.add('join','a',3,[a,b])
    assert sanity.run()['beats']==14
    for layer in prior['layers']:
        meta=layer['meta']; sid,stage=meta['sample'],meta['stage']
        data=dict(np.load(ROOT/f'trace_s{sid}_stage{stage}.npz'))
        selected=[min((r for r in layer['rows'] if r['axis']==axis),key=lambda r:r['beats'])
                  for axis in ('save_y','save_u','recompute','packet','gram')]
        qs={r['q'] for r in selected}|{r['repair_q'] for r in selected if r['repair_q']}
        S=np.unpackbits(data['source_packed'],axis=1,bitorder='little')[:,:meta['C']]
        kernels={q:kernel_trace(S,meta['P'],meta['C'],q,data['sizes']) for q in sorted(qs)}
        rows=[]
        for original in selected:
            sim=Simulation(meta,data,kernels,original['axis'],original['q'],original['allocation'],
                           original['repair_q'] or 1,original['order'] or 'resident')
            row=sim.run(); row['allocation']=original['allocation']
            row['before_accounting_correction_beats']=original['beats']
            check_row(meta,data,row)
            rows.append(row)
            print('FINAL',sid,stage,row['axis'],row['q'],row['beats'],flush=True)
        baseline=min((r for r in rows if r['axis'] in ('save_y','save_u','recompute')),key=lambda r:r['beats'])
        for row in rows:
            row['relative_service_time_to_strong_baseline']=row['beats']/baseline['beats']
            row['service_time_reduction_vs_strong_baseline']=1-row['beats']/baseline['beats']
        result['layers'].append(dict(meta=meta,rows=rows,strong_baseline_axis=baseline['axis'],
                                     strong_baseline_beats=baseline['beats']))
        (ROOT/'final_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    aggregates=[]
    for stage in (0,3):
        subset=[l for l in result['layers'] if l['meta']['stage']==stage]
        for axis in ('packet','gram'):
            basetotal=sum(l['strong_baseline_beats'] for l in subset)
            rows=[next(r for r in l['rows'] if r['axis']==axis) for l in subset]
            aggregates.append(dict(stage=stage,axis=axis,samples=[l['meta']['sample'] for l in subset],
                mean_per_point_reduction=float(np.mean([r['service_time_reduction_vs_strong_baseline'] for r in rows])),
                sum_service_reduction=1-sum(r['beats'] for r in rows)/basetotal,
                warning='Two hardware samples for one component, not valid825 or a network performance result.'))
    result['aggregates']=aggregates; result['wall_seconds']=time.time()-started
    (ROOT/'final_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',result['wall_seconds'],flush=True)


if __name__=='__main__': main()
