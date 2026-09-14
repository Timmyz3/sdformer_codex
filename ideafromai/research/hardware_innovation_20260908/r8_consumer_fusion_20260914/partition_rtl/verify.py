"""Read-only algebra/ledger validation of the fixed existing fixtures and RTL results."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent
def words(p):return np.array([int(x,16) for x in p.read_text().split()],dtype=np.uint32)
def signed(a,bits):
    a=a.astype(np.int64)&((1<<bits)-1)
    return np.where(a&(1<<(bits-1)),a-(1<<bits),a)
rows=json.loads((H/'results.json').read_text());info=[];checks=0
for name in sorted({r['fixture'] for r in rows}):
    p=H/'fixtures'/name
    q=signed(words(p/'q1.hex'),3).reshape(864,8).T
    v=signed(words(p/'q2.hex'),16).reshape(12,8,8).transpose(0,2,1).reshape(96,8)
    s=((words(p/'source.hex')[None,:]>>np.arange(10)[:,None])&1).reshape(10,96,4,4)
    origin=signed(words(p/'origin.hex'),16);valid=np.zeros((4,4),bool)
    for y in range(4):
        for x in range(4):valid[y,x]=0<=origin[0]+y<240 and 0<=origin[1]+x<320
    s=s*valid[None,None];x=np.stack([s[:,:,a//2:a//2+3,a%2:a%2+3].reshape(10,864) for a in range(4)])
    z=(x@q.T).reshape(40,8);gold=z@v.T
    packed=gold.reshape(4,10,12,8).transpose(2,0,1,3).reshape(-1)
    assert np.array_equal(packed,signed(words(p/'gold.hex'),32));checks+=3840
    live=np.any(q!=0,axis=0);assert np.array_equal(live,words(p/'k_live.hex').astype(bool))
    first=int(np.count_nonzero(x[:,:,live]));ranklive=np.any(z!=0,axis=0)
    vv=np.any(v.reshape(12,8,8)!=0,axis=1)
    second=int(np.count_nonzero(vv&ranklive));qreads=int(np.count_nonzero(np.any(x,axis=(0,1))&live))
    baseline_mac=sum(np.count_nonzero(z[:,vv[g]]) for g in range(12))
    ngroups=0;part_mac=0;absrows=int(np.count_nonzero(np.any(z!=0,axis=1)));zero_coef_groups=0;mincoef=0;maxcoef=0
    for pos,zrow in enumerate(z):
        check=np.zeros(96,np.int64)
        for a in sorted(set(abs(zrow).tolist())-{0}):
            member=abs(zrow)==a;leader=np.flatnonzero(member)[0]
            signs=np.where(member,np.sign(zrow)*np.sign(zrow[leader]),0)
            coeff=v@signs;check+=zrow[leader]*coeff;ngroups+=1
            assert -(1<<18)<=coeff.min() and coeff.max()<(1<<18)
            mincoef=min(mincoef,int(coeff.min()));maxcoef=max(maxcoef,int(coeff.max()))
            for g in range(12):
                effective=member&vv[g]
                if np.any(effective):
                    part_mac+=1;zero_coef_groups+=int(np.all(coeff[g*8:g*8+8]==0))
        assert np.array_equal(check,gold[pos])
    info.append({'fixture':name,'nonzero_z':int(np.count_nonzero(z)),'descriptors':ngroups,'max_descriptors':320,
      'coefficient_min':mincoef,'coefficient_max':maxcoef,'entire_effective_group_coefficient_zero':zero_coef_groups})
    for r in [a for a in rows if a['fixture']==name]:
        assert r['outputs']==3840 and r['cycles']==sum(r['state_cycles'])
        assert r['first_issues']==first and r['z_writes']==first+40 and r['z_vector_reads']==first+40
        assert r['second_weight_words']==second and r['weight_words']==qreads+second
        assert r['psum_reads']==480 and r['psum_writes']==480
        assert r['configuration_cycles']==(3361 if r['command']==0 else 0)
        st=r['state_cycles']
        assert st[1]==40 and st[9]==40 and st[19]==480 and st[20]==480 and st[21]==480+r['output_stalls'] and st[22]==1
        if r['mode']==12:
            assert r['mac_issues']==baseline_mac and st[15]==baseline_mac and r['z_scalar_reads']==baseline_mac
            assert r['descriptor_writes']==r['descriptor_reads']==r['build_issues']==0
        else:
            assert r['descriptor_writes']==ngroups and st[11]==ngroups and st[10]==absrows
            assert r['mac_issues']==part_mac and r['descriptor_reads']==ngroups*12
            assert r['cache_epochs']==12 and st[13]==12
            assert r['cache_misses']==st[18] and r['build_issues']==r['cache_misses']+st[17]
            assert r['mac_issues']==r['singletons']+r['cache_hits']+r['cache_misses']
        checks+=22
    for m in [12,13]:
        for sflag in [0,1]:
            a=[r for r in rows if r['fixture']==name and r['mode']==m and r['stall']==sflag]
            assert len(a)==2
            assert all(a[0][k]==a[1][k] for k in a[0] if k not in ['command','configuration_cycles'])
out={'read_only_checks':checks,'gold_values_recomputed':len(info)*3840,'runs':len(rows),
     'no_new_fixtures_or_rtl_runs':True,'fixtures':info,
     'scope':'Integer partition identity, coefficient bounds, exact source/factor counts, final output obligation, state sums and repeated-command counters. These checks do not replace measured RTL cycles.'}
(H/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
