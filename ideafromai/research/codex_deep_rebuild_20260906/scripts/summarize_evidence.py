#!/usr/bin/env python3.12
"""Re-derive report numbers from immutable per-workload results; no EDA admission."""
import sys
sys.dont_write_bytecode=True
from collections import defaultdict
import hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HW=Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
def read(name):return json.loads((ROOT/'results'/name).read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

c1=read('c1_forest_recompute_5640.json');c2=read('c2_merge_controls_2880.json');raw=read('c2_source_staging_2880.json')
summary={'date':'2026-09-06','evidence_level':'CPU operation/intent counts only; PPA_ADMISSION=0; RTL_SPEEDUP_ADMISSION=0',
         'c1':{'tiles':c1['tiles'],'input_rows':360000,'nonzero_rows_per_axis':168756,'arms':{}},'c2':{},'checks':{}}
for order in ('stable','threaded'):
    for slots in (1,2,4):
        a=f'{order}_{slots}_recompute';b=f'{order}_{slots}_backing';x=c1['totals'][a];y=c1['totals'][b]
        assert x['architectural_row_commits']==y['architectural_row_commits']==168756
        assert not x.get('backing_reads',0) and not x.get('backing_writes',0)
        for metric in ('source_issue_slots','architectural_row_commits','checked_outputs'):
            assert x[metric]==sum(r['points'][a].get(metric,0) for r in c1['rows'])
        groups=defaultdict(lambda:[0,0])
        for r in c1['rows']:
            g=groups[(r['sample'],r['operator'])];g[0]+=r['points'][a].get('source_issue_slots',0);g[1]+=r['points'][b].get('source_issue_slots',0)
        summary['c1']['arms'][a]={'slots':slots,'payload_bytes':slots*96*12//8,'backing_source_issues':y['source_issue_slots'],
            'recompute_source_issues':x['source_issue_slots'],'source_issue_increase_pct':100*(x['source_issue_slots']/y['source_issue_slots']-1),
            'miss_rows':x.get('missing_parent_rows',0),'eliminated_backing_accesses':y.get('backing_reads',0)+y.get('backing_writes',0),
            'max_sample_operator_increase_pct':max(100*(u/v-1) for u,v in groups.values())}
for target in ('FC1','FC2'):
    x=c2['weighted'][target+'_window64'];y=raw['weighted'][target]
    assert x['ordinary']==y['ordinary']
    for name,c in ((target+'_window64',x),(target,y)):
        data=c2 if name.endswith('64') else raw
        for key in ('ordinary',):
            assert c[key]==sum(r['counts'][key]*r['output_tiles'] for r in data['rows'] if r['target']==target and r.get('window',64)==64)
    summary['c2'][target]={'baseline_intents':x['ordinary'],
        'partial_fixed01_reduction_pct':100*x['partial_fixed01_reduction'],
        'combined_fixed01_reduction_pct':100*x['combined_fixed01_reduction'],
        'raw_capture0_reduction_pct':100*y['capture0_reduction'],
        'raw_capture1_reduction_pct':100*y['capture1_reduction'],
        'raw_capture0_saved_intents_per_weighted_window':(y['ordinary']-y['capture0'])/y['windows']}
summary['checks']={'c1_numeric_lane_comparisons_all_12_axes':sum(x['checked_outputs'] for x in c1['totals'].values()),
    'c1_zero_mismatches':True,'c2_partial_directed_lane_comparisons':c2['checked_event_replay_outputs'],
    'c2_raw_directed_lane_comparisons':raw['checked_outputs'],'c2_zero_directed_mismatches':raw['mismatches']==c2['event_replay_mismatches']==0}
summary['c1']['integer_scope']='Frozen binary supports with seeded signed INT8 diagnostic weights, including -128 and +127 columns; not frozen FP32 inference or AEE'
summary['c2']['scope']='2880 existing reduced B4 descriptor templates; output-tile weighted; not all-token coverage. Numeric replay is directed subset only.'
(ROOT/'results/evidence_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
inputs=[Path(c1['source']),HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/layers.json']
for prefix in ('m2051_ep34_tsbg_full40_s1920','m2067_ep34_fc2_exact_continuation_s960'):
    inputs.extend(HW/'tb_m2018/fixtures'/(prefix+ext) for ext in ('.json','.memh'))
inputs.extend(HW/'system_simulator/scripts'/name for name in ('m2259_c1_forest_lifetime_probe.py','m2260_c1_hot_parent_probe.py','m2271_destination_signature_screen.py'))
provenance={'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HW,text=True).strip(),
    'python':sys.version,'source_inputs':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in inputs],
    'scripts':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted((ROOT/'scripts').glob('screen_*.py'))],
    'results':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted((ROOT/'results').glob('*.json'))],
    'note':'Hashes record this research snapshot; they do not grant VCS/DC/PT/Formality admission. Later additions to result boundary text are documentation-only.'}
(ROOT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
