"""Independent functional/profile accounting from native source, not emitted latent."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from prepare import readhex,native_gold
H=Path(__file__).resolve().parent
OLD=H.parents[1]/'fusion_ten_trials_20260914/decompositions/q2_da'
rows=json.loads((H/'results.json').read_text())
meta=json.loads((H/'definition.json').read_text())
profiles={}
for case in meta['fixtures']:
    name=case['name'];d=H/'fixtures'/name
    words=readhex(d/'source.hex').reshape(96,4,4)
    q1=readhex(d/'q1.hex').reshape(864,8).T
    q2=readhex(d/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
    assert np.array_equal(np.any(q1,axis=0),readhex(d/'k_live.hex'))
    z,gold=native_gold(words,q1,q2,readhex(d/'origin.hex'))
    assert np.array_equal(gold.ravel(),readhex(d/'gold.hex'))
    z=z.reshape(40,8)
    data=dict(strong_selected=0,strong_fallback=0,strong_saved_mac=0,strong_da=0,mac=0,fixed_evaluations=0,fixed_da=0,nonempty_blocks=0,lazy_blocks=0,evaluations=0,selected=0,fallback=0,bypass=0,saved_mac=0,da=0,rank_histogram=Counter(),evaluated_cost_histogram=Counter())
    for og in range(12):
        qs=q2[og*8:og*8+8]
        live=np.any(qs!=0,axis=0)&np.any(z!=0,axis=0)
        data['nonempty_blocks']+=int(np.any(live));did_lazy=False
        for pos in range(40):
            zp=z[pos]*live;rc=int(np.count_nonzero(zp));data['mac']+=rc;data['rank_histogram'][rc]+=1
            if not rc:continue
            width=max((int(v).bit_length()+1 if v>=0 else (~int(v)).bit_length()+1) for v in zp)
            da=0;reconstructed=np.zeros(8,np.int64)
            for bit in range(width):
                for g in range(2):
                    subset=sum((qs[:,r] for r in range(g*4,g*4+4) if (int(zp[r])>>bit)&1),start=np.zeros(8,np.int64))
                    assert np.all(subset>=-131072) and np.all(subset<=131071)
                    da+=int(np.any(subset))
                    reconstructed+=subset*(-(1<<bit) if bit==width-1 else 1<<bit)
            assert np.array_equal(reconstructed,qs@zp)
            data['fixed_evaluations']+=1;data['fixed_da']+=da
            if rc<=2:data['bypass']+=1;continue
            did_lazy=True;data['evaluations']+=1;data['evaluated_cost_histogram'][f'{rc}:{da}']+=1
            if da<rc:
                data['strong_selected']+=1;data['strong_saved_mac']+=rc;data['strong_da']+=da
            else:data['strong_fallback']+=1
            if 2+da<rc:
                data['selected']+=1;data['saved_mac']+=rc;data['da']+=da
            else:data['fallback']+=1
        data['lazy_blocks']+=int(did_lazy)
    profiles[name]=data
for r in rows:
    p=profiles[r['fixture']];m=r['mode'];states=r['state_cycles']
    assert sum(states)==r['cycles']
    assert states[20]==states[21] # DA_POS and DA_ENCODE, including rejected positions.
    assert states[19]==r['aux_writes'] and states[22]==r['aux_events']
    if m==14:blocks=ev=da=saved=0
    elif m==15:blocks=12;ev=p['fixed_evaluations'];da=p['fixed_da'];saved=p['mac']
    else:
        blocks=p['nonempty_blocks'] if m==13 else p['lazy_blocks']
        ev=p['evaluations'];prefix='strong_' if m==11 else '';da=p[prefix+'da'];saved=p[prefix+'saved_mac']
        for field,target in [('hybrid_blocks',blocks),('hybrid_evaluations',ev),('hybrid_selected',p[prefix+'selected']),('hybrid_fallback',p[prefix+'fallback']),('hybrid_bypass',p['bypass']),('hybrid_saved_mac',saved)]:assert r[field]==target,(r,field,target)
    assert r['mac_issues']==p['mac']-saved
    assert states[19]==32*blocks and states[20]==ev and states[22]==da
    assert r['aux_reads']==r['aux_issues']==30*blocks+da
    assert r['z_scalar_reads']==r['mac_issues']
    # Compare each stalled/reconfigured command to same fixture/command standalone baseline.
    base=next(x for x in rows if x['fixture']==r['fixture'] and x['mode']==14 and x['stall']==r['stall'] and x['command']==r['command'] and not x['reconfigured'])
    stall_delta=sum(r[k]-base[k] for k in ['source_stalls','weight_stalls','output_stalls'])
    assert r['cycles']-base['cycles']==32*blocks+2*ev+da-saved+stall_delta
    assert r['z_vector_reads']==base['z_vector_reads']+ev
    for k in ['source_words','second_weight_words','local_source_reads','z_writes','first_issues','dual_updates','psum_reads','psum_writes']:assert r[k]==base[k],(r,k)
# Compare all shared old recorded fields, not only real-eight headline.
oldrows=json.loads((OLD/'results.json').read_text());reproduced=0
for old in oldrows:
    r=next(x for x in rows if x['fixture']==old['fixture'] and x['mode']==old['mode'] and x['stall']==old['stall'] and x['command']==old['command'] and not x['reconfigured'])
    for k,v in old.items():assert r[k]==v,(old['fixture'],old['mode'],old['stall'],old['command'],k,r[k],v)
    reproduced+=1
# Saved actual RTL traces remain readable and exactly equal both command gold vectors.
traces=[]
for seq in sorted({r['sequence'] for r in rows}):
    records=sorted([r for r in rows if r['sequence']==seq],key=lambda r:r['command'])
    path=H/records[0]['trace'];trace=readhex(path)
    assert len(trace)==7680
    for i,r in enumerate(records):assert np.array_equal(trace[i*3840:(i+1)*3840],readhex(H/'fixtures'/r['fixture']/'gold.hex'))
    traces.append(str(path.relative_to(H)))
(H/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
fields=[k for k,v in rows[0].items() if isinstance(v,int) and not isinstance(v,bool) and k not in ['mode','stall','command']]
real={str(m):{str(s):{k:sum(r[k] for r in rows if r['fixture'].startswith('real_') and r['mode']==m and r['stall']==s and r['command']==0 and not r['reconfigured']) for k in fields} for s in [0,1]} for m in [14,15,13,12,11]}
summary=dict(commands=len(rows),outputs_compared=sum(r['outputs'] for r in rows),fixtures=len(meta['fixtures']),original_commands_all_fields_reproduced=reproduced,independent_gold_values=len(meta['fixtures'])*3840,actual_RTL_trace_files=len(traces),commands_in_reconfiguration_sequences=sum(r['reconfigured'] for r in rows),reconfigured_second_commands=sum(r['reconfigured'] and r['command']==1 for r in rows),real_first_commands=real,scope='K864/N96/T10/R8 raw p only; no I24, training, quality rerun, EDA or frame claim')
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
(H/'checks.json').write_text(json.dumps(dict(all_pass=True,state_sums=len(rows),state_and_independent_selection_accounting=len(rows),exact_deltas_including_stall_phase=len(rows),full_output_trace_checks=len(rows),original_reproduction=reproduced),indent=2)+'\n')
print(json.dumps(summary,indent=2))
