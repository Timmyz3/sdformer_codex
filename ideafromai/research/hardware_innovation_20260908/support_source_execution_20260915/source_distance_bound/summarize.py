"""Independently reconcile RTL counters against CPU decision/source work and FSM accounting."""
from pathlib import Path
import csv,json,re
from collections import defaultdict
P=Path(__file__).resolve().parent
probe=json.loads((P/'probe.json').read_text())['variants']
files=['old_small','adapt_small','zero_small','old_expanded','adapt_expanded','zero_expanded']
rows={};summary=[];checks=defaultdict(int)
fields=['cycles_to_last_code','cycles_to_done','words','bytes','config_words','X_words','prefetch_words','produced_pairs','channel_refs','batches','scalar_mac','bound_cycles','popcount16_ops','dominance_tests','reduce_cycles','cache_hits','refetches','req_stall','out_stall']
for stem in files:
    data=[]
    with (P/(stem+'.csv')).open() as f:
        for r in csv.DictReader(f):
            r={k:(v if k in ('case','function') else int(v)) for k,v in r.items()};data.append(r)
    rows[stem]=data
    sets=defaultdict(list)
    for r in data:
        assert sum(r['state'+str(i)] for i in range(17))==r['cycles_to_done']
        assert r['bytes']==r['words']*16 and r['words']==r['config_words']+r['X_words']
        assert r['scalar_mac']==r['produced_pairs']*10
        assert r['config_words']==({1:30,2:32,3:35}[r['mode']] if r['p']==0 else 0)
        assert r['state8']==r['batches']*10
        assert r['state10']==r['batches']
        if r['mode']>=2:
            evaluations=60+r['produced_pairs']
            assert r['reduce_cycles']==r['state16']==evaluations
            assert r['dominance_tests']==evaluations*256
            assert r['popcount16_ops']==evaluations*272
            assert r['bound_cycles']==sum(r['state'+str(i)] for i in (14,15,16))==18*evaluations+r['batches']+6
            assert r['state15']==16*evaluations and r['state14']==evaluations+r['batches']+6
        else:
            assert r['produced_pairs']==640 and r['X_words']==128
            assert r['bound_cycles']==0 and r['popcount16_ops']==960
        checks['P_commands']+=1;checks['output_labels']+=60
        sets[(r['function'],r['case'],r['real'],r['mode'],r['bp'],r['pass'])].append(r)
    for (fn,case,real,mode,bp,pa),rs in sets.items():
        assert sorted(x['p'] for x in rs)==list(range(32))
        checks['P32_frames']+=1
        if real and mode>=2:
            fi=int(re.search(r'\d+',case).group())
            truth=probe['code' if mode==2 else fn]['frame_work'][fi]
            for observed,expected in [('produced_pairs','source_produced_pairs'),('channel_refs','source_channel_refs'),('X_words','source_X_words'),('batches','batches'),('scalar_mac','source_scalar_mac')]:
                assert sum(r[observed] for r in rs)==truth[expected],(stem,case,mode,observed)
                checks['independent_CPU_work_equalities']+=1
    agg=defaultdict(lambda:defaultdict(int))
    for r in data:
        if r['pass']!=0 or not r['real']:continue
        key=(r['function'],r['mode'],r['bp'])
        for k in fields:agg[key][k]+=r[k]
        agg[key]['P_commands']+=1
    for (fn,mode,bp),a in sorted(agg.items()):summary.append(dict(set=stem,function=fn,mode=mode,bp=bp,**a))
# Same exact-code hardware/input must produce identical execution for all W variants.
for scope in ('small','expanded'):
    ref={(r['case'],r['bp'],r['p']):r for r in rows['old_'+scope] if r['mode']==2 and r['pass']==0}
    for name in ('adapt','zero'):
        for r in rows[name+'_'+scope]:
            if r['mode']!=2 or r['pass']!=0:continue
            a=ref[r['case'],r['bp'],r['p']]
            assert all(a[k]==r[k] for k in fields)
            checks['same_input_exact_code_execution_equalities']+=1
out=dict(status='PASS',scope='source PSN to typed code/class only; 32 training frames P32; not network AEE or joined backend',
         clock='one initial reset per executable; P0 cold, P1..31 static reuse; dynamic X/cache cleared perP; BP calendar restarts perP identically in all arms',
         counts=dict(checks),summaries=summary)
(P/'SUMMARY.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
with (P/'summary.jsonl').open('w') as f:
    for x in summary:f.write(json.dumps(x,separators=(',',':'))+'\n')
print(json.dumps(dict(status='PASS',checks=dict(checks)),separators=(',',':')))
for x in summary:
    print(x['set'],x['mode'],x['bp'],'cycles',x['cycles_to_done'],'pairs',x['produced_pairs'],'X',x['X_words'],'bound',x['bound_cycles'])
