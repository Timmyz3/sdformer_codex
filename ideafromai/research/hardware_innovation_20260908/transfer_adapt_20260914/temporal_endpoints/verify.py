"""Native source -> direct/delta/two-domain arithmetic and paid RTL obligations."""
from pathlib import Path
import json
import numpy as np

H=Path(__file__).resolve().parent
B=H.parents[1]

def readhex(path):
    return np.array([int(x,16) for x in path.read_text().split()],np.uint32).view(np.int32).astype(np.int64)

cases=json.loads((H/'fixtures.json').read_text())
rows=json.loads((H/'results.json').read_text())
stream_rows=[]
if (H/'results_stream.json').exists():
    stream_rows=json.loads((H/'results_stream.json').read_text())
    rows+=stream_rows
    native_paths=(B/'transfer_adapt_20260914/pair_sparse/fixtures/stream/manifest.txt').read_text().split()
    cases += [dict(name=f'native_{Path(p).name}',path=p) for p in native_paths]
disjoint_rows=[]
if (H/'results_disjoint.json').exists():
    disjoint_rows=json.loads((H/'results_disjoint.json').read_text())
    rows+=disjoint_rows
    disjoint_paths=(B/'transfer_adapt_20260914/audit/fixtures/disjoint_4000/manifest.txt').read_text().split()
    cases += [dict(name=f'native_{Path(p).name}',path=p) for p in disjoint_paths]
predictions={}
order=json.loads((H/'calibration.json').read_text())['permutation']
inverse=np.argsort(order)
max_endpoint_prefix=0
max_recovered=0
for c in cases:
    p=Path(c['path'])
    word=readhex(p/'source.hex').reshape(96,4,4)
    oy,ox=readhex(p/'origin.hex')
    params=p if (p/'q1.hex').exists() else B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0'
    q=readhex(params/'q1.hex').reshape(864,8)
    v=readhex(params/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
    assert np.all((q>=-4)&(q<=3))
    live=np.any(q!=0,axis=1)
    assert np.array_equal(live,readhex(params/'k_live.hex'))
    events=np.zeros((864,4,10),np.int64)
    valid=int(sum(0<=oy+y<240 and 0<=ox+x<320 for y in range(4) for x in range(4)))
    for k in range(864):
        ch,tap=divmod(k,9)
        for pos in range(4):
            y,x=pos//2+tap//3,pos%2+tap%3
            if 0<=oy+y<240 and 0<=ox+x<320:
                events[k,pos]=(word[ch,y,x]>>np.arange(10))&1
    delta=events-np.pad(events[:,:,:-1],((0,0),(0,0),(1,0)))
    assert np.all((delta>=-1)&(delta<=1))
    direct_support=(events[:,0]!=0)|(events[:,1]!=0)
    direct_support=np.concatenate((direct_support,(events[:,2]!=0)|(events[:,3]!=0)),axis=1)
    delta_support=np.concatenate(((delta[:,0]!=0)|(delta[:,1]!=0),(delta[:,2]!=0)|(delta[:,3]!=0)),axis=1)
    direct_count=direct_support.sum(1)
    endpoint_count=delta_support.sum(1)
    choose=(endpoint_count<direct_count)&live
    adjacent=(events[:,:,1:]*events[:,:,:-1]).any((1,2))
    assert np.all(endpoint_count[~adjacent]>=direct_count[~adjacent])
    z=events.reshape(864,40).T@q
    raw=np.concatenate([z@v[g*8:g*8+8].T for g in range(12)])
    assert np.array_equal(raw.ravel(),readhex(p/'gold.hex')),c['name']
    all_delta=(delta.reshape(864,40)[:,:,None]*q[:,None,:]).cumsum(0)
    assert np.all(all_delta>=-4096)&np.all(all_delta<=4095)
    max_endpoint_prefix=max(max_endpoint_prefix,int(np.abs(all_delta).max()))
    endpoint_z=delta.reshape(864,40).T@q
    reconstructed=endpoint_z.reshape(4,10,8).cumsum(1).reshape(40,8)
    assert np.array_equal(z,reconstructed)
    direct_domain=events.reshape(864,40).T@(q*(~choose)[:,None])
    endpoint_domain=delta.reshape(864,40).T@(q*choose[:,None])
    recovered_endpoint=endpoint_domain.reshape(4,10,8).cumsum(1).reshape(40,8)
    assert np.array_equal(direct_domain+recovered_endpoint,z)
    max_recovered=max(max_recovered,int(np.abs(z).max()))
    vl=np.any(v.reshape(12,8,8)!=0,axis=1)
    M=int(((z!=0)*vl.sum(0)).sum())
    V=int((vl&np.any(z!=0,axis=0)).sum())
    K=int(live.sum())
    Q=int(((direct_count>0)&live).sum())
    assert np.array_equal(direct_count>0,endpoint_count>0)
    native_events=events
    for mode in [0,1,2,3,4,5]:
        if mode>=4:
            events=native_events[:,:,order]
            delta=events-np.pad(events[:,:,:-1],((0,0),(0,0),(1,0)))
            direct_support=np.concatenate(((events[:,0]!=0)|(events[:,1]!=0),(events[:,2]!=0)|(events[:,3]!=0)),axis=1)
            delta_support=np.concatenate(((delta[:,0]!=0)|(delta[:,1]!=0),(delta[:,2]!=0)|(delta[:,3]!=0)),axis=1)
            direct_count=direct_support.sum(1);endpoint_count=delta_support.sum(1)
            choose=(endpoint_count<direct_count)&live
            adjacent=(events[:,:,1:]*events[:,:,:-1]).any((1,2))
            assert np.all(endpoint_count[~adjacent]>=direct_count[~adjacent])
            zd=events.reshape(864,40).T@(q*(~choose)[:,None])
            ze=delta.reshape(864,40).T@(q*choose[:,None])
            zp=zd.reshape(4,10,8)+ze.reshape(4,10,8).cumsum(1)
            assert np.array_equal(zp[:,inverse,:].reshape(40,8),z)
            ze_all=(delta.reshape(864,40).T@q).reshape(4,10,8).cumsum(1)
            assert np.array_equal(ze_all[:,inverse,:].reshape(40,8),z)
        used=events if mode==0 else delta if mode in [1,5] else np.where(choose[:,None,None],delta,events)
        support=np.concatenate(((used[:,0]!=0)|(used[:,1]!=0),(used[:,2]!=0)|(used[:,3]!=0)),axis=1)
        U=int(support[live].sum())
        full_fields=int(np.count_nonzero(used[live]))
        end_U=0 if mode==0 else U if mode in [1,5] else int(endpoint_count[choose].sum())
        mixed=mode in [2,3,4]
        prefix=20 if mode in [1,5] or (mixed and np.any(choose)) else 0
        merge=20 if mixed and np.any(choose) else 0
        clear=40 if mode==2 or (mode in [3,4] and np.any(choose)) else 20
        select=K if mode==2 else int((adjacent&live).sum()) if mode in [3,4] else 0
        expected=dict(outputs=3840,source_words=96*valid,local_source_reads=K,weight_words=Q+V,
                      second_weight_words=V,first_issues=U,dual_updates=full_fields-U,
                      z_vector_reads=U+40+prefix+merge,z_scalar_reads=M,z_writes=clear+U+prefix+merge,
                      psum_reads=480,psum_writes=480,mac_issues=M,direct_issues=U-end_U,endpoint_issues=end_U,
                      selection_issues=select,prefix_issues=prefix,merge_issues=merge,
                      prefix_reads=prefix,merge_reads=merge,zclear_writes=clear,
                      falling_fields=int((used[live]<0).sum()),direct_pair_work=int(direct_count[live].sum()),
                      endpoint_pair_work=int(endpoint_count[live].sum()),
                      selected_endpoint_columns=int(choose.sum()) if mixed else 0,
                      selected_direct_columns=Q-int(choose.sum()) if mixed else 0,
                      preclassified_direct_columns=int(((direct_count>0)&(~adjacent)&live).sum()) if mode in [3,4] else 0,
                      empty_columns_skipped=int(((direct_count==0)&live).sum()) if mode in [3,4] else 0,
                      base_cycles=5437+K+2*Q+3*U+M+(clear-20)+select+2*prefix+2*merge)
        predictions[c['name'],mode]=expected

checked=0
for r in rows:
    p=predictions[r['fixture'],r['mode']]
    for k,v in p.items():
        if k!='base_cycles':assert r[k]==v,(r['fixture'],r['mode'],k,r[k],v)
    assert r['cycles']==p['base_cycles']+r['source_stalls']+r['weight_stalls']+r['output_stalls'],r
    cfg=(1537+(1824 if r['command']==0 else 0)) if r['fixture'].startswith('native_') else (0 if r['command'] else 3361)
    assert r['configuration_cycles']==cfg+r['permutation_configuration_cycles']
    assert r['permutation_configuration_cycles'] in [0,1]
    if not r.get('cross_mode',False):assert r['permutation_configuration_cycles']==int(r['mode']>=4 and r['command']==0)
    assert sum(r['state_cycles'])==r['cycles']
    assert r['state_cycles'][1]+r['state_cycles'][24]==r['zclear_writes']
    assert r['state_cycles'][19]==r['selection_issues']
    assert r['state_cycles'][20]==r['prefix_reads'] and r['state_cycles'][21]==r['prefix_issues']
    assert r['state_cycles'][22]==r['merge_reads'] and r['state_cycles'][23]==r['merge_issues']
    checked+=1

old=json.loads((B/'transfer_adapt_20260914/pair_sparse/results.json').read_text())
lookup={(r['fixture'],r['stall'],r['command']):r for r in old if r['mode']==14}
matched=0
matched_keys=set()
for r in rows:
    key=r['fixture'],r['stall'],r['command']
    if r['mode']==0 and key in lookup:
        oldrow=lookup[key]
        for field in ['cycles','configuration_cycles','outputs','source_words','weight_words','second_weight_words',
                      'z_vector_reads','z_scalar_reads','z_writes','first_issues','dual_updates','psum_reads',
                      'psum_writes','mac_issues','source_stalls','weight_stalls','output_stalls','local_source_reads','state_cycles']:
            assert r[field]==oldrow[field],(key,field,r[field],oldrow[field])
        matched+=1
        matched_keys.add(key)

summary=json.loads((H/'SUMMARY.json').read_text())
stream_controls=0
if stream_rows:
    old_stream=json.loads((B/'transfer_adapt_20260914/pair_sparse/results_stream.json').read_text())
    old_lookup={(r['stall'],r['command']):r for r in old_stream if r['mode']==14}
    for r in stream_rows:
        if r['mode']!=0:continue
        oldrow=old_lookup[r['stall'],r['command']]
        for field in ['cycles','configuration_cycles','outputs','source_words','weight_words','second_weight_words',
                      'z_vector_reads','z_scalar_reads','z_writes','first_issues','dual_updates','psum_reads',
                      'psum_writes','mac_issues','source_stalls','weight_stalls','output_stalls','local_source_reads','state_cycles']:
            assert r[field]==oldrow[field],('stream control',r['command'],field,r[field],oldrow[field])
        stream_controls+=1
real=summary['real']
def coordinates(ids):
    return {(y,x) for tile in ids for y in range(2*(tile//160)-1,2*(tile//160)+3)
            for x in range(2*(tile%160)-1,2*(tile%160)+3) if 0<=y<240 and 0<=x<320}
cal_coords=coordinates(range(32))
held_overlap=len(cal_coords&coordinates(range(128,192)))
disjoint_overlap=len(cal_coords&coordinates(range(4000,4064)))
assert disjoint_overlap==0
positive=min(real['m2_s0']['cycles'],real['m3_s0']['cycles'],real['m4_s0']['cycles'])<real['m0_s0']['cycles']
ablations={}
for label,group in [('held_128',stream_rows),('input_disjoint_4000',disjoint_rows)]:
    if not group:continue
    arms={mode:[r for r in group if r['mode']==mode and r['stall']==0 and r['command']<64] for mode in [0,4,5]}
    assert all(len(arm)==64 for arm in arms.values())
    total=lambda mode,key:sum(r[key] for r in arms[mode])
    for r in arms[5]:
        assert r['selection_issues']==r['merge_issues']==0
        assert r['zclear_writes']==r['prefix_issues']==20
    assert total(4,'configuration_cycles')==total(5,'configuration_cycles')
    saved=total(5,'cycles')-total(4,'cycles')
    assert saved==3*(total(5,'first_issues')-total(4,'first_issues'))-total(4,'selection_issues')-(total(4,'zclear_writes')-total(5,'zclear_writes'))-2*total(4,'merge_issues')
    if label=='held_128':assert total(5,'first_issues')==70535
    ablations[label]=dict(commands_per_arm=64,full_endpoint_updates=total(5,'first_issues'),
                         mixed_updates=total(4,'first_issues'),mixed_saved_vs_full_endpoint=saved,
                         cold_service={f'm{m}':total(m,'cycles')+total(m,'configuration_cycles')+64 for m in [0,4,5]})
result=dict(passed=True,commands=checked,rtl_raw_values=sum(r['outputs'] for r in rows),
            independent_raw_values=len(cases)*3840,native_direct_control_records_equal=matched,
            native_direct_control_unique_records_equal=len(matched_keys),
            held_direct_control_records_equal=stream_controls,
            disjoint_commands=len(disjoint_rows),
            calibration_held_source_overlap_words=held_overlap*96,
            calibration_disjoint_source_overlap_words=disjoint_overlap*96,
            max_observed_partial_delta_abs=max_endpoint_prefix,max_observed_recovered_z_abs=max_recovered,
            real_eight_net_positive=positive,gate_64=positive,
            state_bits_per_mode=8*40*26,z_port_bits=208,producer_alu='8x32 with13bit carry cut and per-field carry-in',
            producer_multiplier='8 signed19x13',new_state_bytes_over_old_native_direct=520,permutation_bits=40,
            same_permutation_full_endpoint_ablation=ablations,
            validation_scope='Fixed cal0..31 order; same-frame output-held128 has overlapping source halo; same-frame4000 source is disjoint. No cross-sequence validation or actual J/I24 RTL.',
            calibration_runtime_scope='Offline DP is outside RTL service; first deployment of its fixed 40bit table is one paid configuration beat.',
            independent_review_followups='Mode5 and input-disjoint4000 measured after audit/temporal_order_review.md; original review judgement retained.',
            predictors={f'{name}_m{mode}':value for (name,mode),value in predictions.items()})
(H/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='predictors'}),flush=True)
