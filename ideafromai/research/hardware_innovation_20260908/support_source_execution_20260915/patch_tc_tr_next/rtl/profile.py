"""Independent source/arithmetic/physical-load and complete ready/BP cycle reconciliation."""
from pathlib import Path
from collections import defaultdict
import csv,json,re
import numpy as np
P=Path(__file__).resolve().parent;BASE=P.parents[2]
raw=[]
for ci,path in enumerate(sorted((BASE/'algorithm/patch_probe/partial_completion/integer_valid10').glob('capture_*.npz'))[:4]):
    d=np.load(path)
    for gid,w in zip(d['group_ids'],d['source_gate_words']):raw.append((f'capture{ci}_group{int(gid)}',int(gid),w))
raw.extend([('diagnostic_zero',0,np.zeros((864,4),np.uint16)),('diagnostic_dense',0,np.full((864,4),1023,np.uint16)),
            ('diagnostic_alternating',0,np.where((np.arange(864)[:,None]+np.arange(4)[None,:])%2,341,682).astype(np.uint16))])
source={name:(gid,w) for name,gid,w in raw}
allrows=[]
for stem in ['ordinary_small','regional_small','ordinary_full','regional_full']:
    candidates=list(csv.DictReader((P/(stem+'.csv')).open()))
    if (P/(stem+'_dense.csv')).exists():
        candidates=[r for r in candidates if r['tc']=='1']+list(csv.DictReader((P/(stem+'_dense.csv')).open()))
    for row in candidates:
        row={k:(v if k in ['function','case'] else int(v)) for k,v in row.items()};row['set']=stem;allrows.append(row)
results={};checks=defaultdict(int)
states=re.search(r'typedef enum logic\[7:0\] \{(.*?)\} state_t', (P/'patch_core.sv').read_text(),re.S).group(1)
state_names=[x.strip() for x in states.replace('\n','').split(',')];done_index=state_names.index('DONE')
for fn in ['ordinary_r32','regional_r96_a48']:
    model=np.load(P.parent/(fn+'.npz'));u=model['u'].astype(np.int64);v=model['v'].astype(np.int64);a=model['a'].astype(np.int64);R=u.shape[1]
    an=(a!=0).sum(axis=0)
    for name in set(r['case'] for r in allrows if r['function']==fn):
        gid,w=source[name];x=((w[...,None]>>np.arange(10))&1).transpose(1,2,0).reshape(40,864).astype(np.int64)
        sc=x.sum(0);liveK=int(np.count_nonzero(sc));mask=model['masks'][gid//2400].repeat(4);z=(x@u)*mask;y=z@v;q=np.einsum('ts,psr->ptr',a,z.reshape(4,10,R)).reshape(40,R)
        active=np.flatnonzero(mask);activeR=len(active)
        vn=(v[active].reshape(activeR,12,8)!=0).any(axis=2).sum(axis=1)
        for tc in [0,1]:
            ids=active if tc else np.arange(R);nb=len(ids)//8;lane_live=mask[ids].reshape(nb,8);block_live=lane_live.any(axis=1);B=int(block_live.sum())
            um=u[:,ids]*mask[ids];nonzero=um.reshape(864,nb,8).any(axis=2)
            U_updates=int((nonzero*sc[:,None]).sum());ugroups=int((nonzero*(sc[:,None]!=0)).sum());groups=liveK*B
            seq=[]
            for block in range(nb):
                ranks=ids[block*8:block*8+8][lane_live[block]]
                seq.extend(np.unique(ranks//16).tolist())
            reads=liveK*len(seq);tag=-1;U_words=0
            for k in np.flatnonzero(sc):
                for line in seq:
                    address=int(k)*(R//16)+line
                    if address!=tag:U_words+=1;tag=address
            pU=2882+40*nb+liveK*nb+groups+reads+4*U_words+3*U_updates+ugroups
            for av in [0,1]:
                scalar=q if av else z
                V_updates=int(np.dot(np.count_nonzero(scalar[:,active],axis=0),vn))
                avec=z[:,ids] if av else y
                ncol=avec.shape[1]//8
                nonzero_vec=(avec.reshape(4,10,ncol,8)!=0).any(axis=3).sum(axis=(0,2))
                A_updates=int(np.dot(nonzero_vec,an));A_scalar=int(np.dot((avec.reshape(4,10,-1)!=0).sum(axis=(0,2)),an))
                V_words=activeR*12;words=288+U_words+V_words+13+360+2
                # Every state in the implemented ready schedule, including output retirement.
                pV=12+12*(activeR+1)+3*V_words+480+480*(activeR+1)+480*activeR+V_updates+480+(1440 if av else 1)
                pA=(80*nb+400*B+A_updates+1) if av else (7200+A_updates)
                cycles=61+(R//4 if tc else 0)+pU+pV+pA+41
                results[fn,name,tc,av]=dict(U_updates=U_updates,V_updates=V_updates,A_updates=A_updates,A_scalar_mac=A_scalar,
                    U_words=U_words,U_cache_reads=reads,V_words=V_words,source_words=288,A_words=13,tau_words=360,config_words=2,
                    words=words,bytes=words*16,zero_source_words=864-liveK,Z_writes=40*nb+U_updates,scratch_writes=40*nb if av else 480,
                    ready_cycles=cycles,phase1=pU,phase2=pV,phase3=pA)
for r in allrows:
    expected=results[r['function'],r['case'],r['tc'],r['av']]
    for k,v in expected.items():
        if k=='ready_cycles' or k.startswith('phase'):continue
        assert r[k]==v,(r['function'],r['case'],r['tc'],r['av'],k,r[k],v)
        checks['independent_work_equalities']+=1
    assert sum(r['state'+str(i)] for i in range(64))==r['cycles_done']==sum(r['phase'+str(i)] for i in range(5))
    assert r['state1']==r['words']+r['req_stall']
    extra=r['req_stall']+(r['state2']-r['words'])+r['out_stall']+(r['state'+str(done_index)]-1)
    assert r['cycles_done']==expected['ready_cycles']+extra,(r['function'],r['case'],r['tc'],r['av'],r['bp'],'cycles',r['cycles_done'],expected['ready_cycles'],extra)
    if not r['bp']:
        for key in ['phase1','phase2','phase3']:assert r[key]==expected[key],(r['case'],key,r[key],expected[key])
    checks['complete_cycle_equations']+=1
summary=[]
for scope in ['small','full']:
    for fn in ['ordinary_r32','regional_r96_a48']:
        for tc in [0,1]:
            for av in [0,1]:
                for bp in [0,1]:
                    rows=[r for r in allrows if r['set'].endswith(scope) and r['function']==fn and r['tc']==tc and r['av']==av and r['bp']==bp and r['pass']==0 and r['real']]
                    keys=['cycles_done','words','bytes','source_words','U_words','V_words','A_words','tau_words','config_words','U_updates','V_updates','A_updates','A_scalar_mac','U_cache_reads','zero_source_words','Z_writes','scratch_writes']+[f'phase{i}' for i in range(5)]
                    summary.append(dict(set=scope,function=fn,tc=tc,av=av,bp=bp,commands=len(rows),**{k:sum(r[k] for r in rows) for k in keys}))
out=dict(status='PASS',commands=len(allrows),final_U_gate_values=len(allrows)*3840,checks=dict(checks),state_names=state_names,
         scope='four existing validation source captures; two distinct fixed integer functions; no network AEE',summaries=summary)
(P/'SUMMARY.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
with (P/'summary.jsonl').open('w') as f:
    for r in summary:f.write(json.dumps(r,separators=(',',':'))+'\n')
print(json.dumps(dict(status='PASS',commands=len(allrows),checks=dict(checks))))
for r in summary:
    if r['set']=='full':print(r['function'],r['tc'],r['av'],r['bp'],r['cycles_done'],r['bytes'],r['U_updates'],r['V_updates'],r['A_updates'])
