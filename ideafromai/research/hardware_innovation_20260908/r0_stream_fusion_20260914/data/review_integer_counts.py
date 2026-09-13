"""Independent factor-function and complete-cycle audit; no RTL/GPU rerun."""
from pathlib import Path
import json
import numpy as np
from model_access import HERE
RTL=HERE.parent/'integer_factor'

def hx(p,signed=False):
    a=np.array([int(x,16) for x in p.read_text().split()],dtype=np.int64)
    if signed:a=np.where(a>=2**31,a-2**32,a)
    return a

def predict(name):
    f=RTL/'fixtures'/name
    words=hx(f/'source.hex').reshape(96,4,4)&1023
    oy,ox=hx(f/'origin.hex',True)
    valid=np.array([[0<=oy+y<240 and 0<=ox+x<320 for x in range(4)] for y in range(4)])
    words=words*valid
    g=((words[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
    q1=hx(f/'q1.hex',True).reshape(864,8).T
    q2=hx(f/'q2.hex',True).reshape(12,8,8).transpose(0,2,1).reshape(96,8)
    w=hx(f/'weight.hex',True).reshape(12,864,8).transpose(0,2,1).reshape(96,864)
    kl=hx(f/'k_live.hex').astype(bool);live=hx(f/'mask.hex').reshape(12,24).astype(bool)
    assert np.array_equal(w,q2@q1)
    assert np.array_equal(kl,np.any(q1!=0,axis=0))
    assert np.max(np.abs(q1))<=3 and np.max(np.abs(q2))<=32768
    patches=np.stack([g[:,:,py:py+3,px:px+3].reshape(10,864) for py in range(2) for px in range(2)])
    z=patches@q1.T;p=z@q2.T
    assert np.array_equal(p,patches@w.T)
    oracle=hx(f/'gold.hex',True).reshape(12,4,10,8).transpose(1,2,0,3).reshape(4,10,96)
    assert np.array_equal(p,oracle)
    assert np.abs(z).max()<=2592 and np.abs(p).max()<=679477248
    A=int(patches[:,:,kl].sum());Q=int(patches[:,:,kl].any((0,1)).sum());K=int(kl.sum());Z=int(np.count_nonzero(z));M=12*Z
    source=0
    for k in np.flatnonzero(kl):
        ky,kx=(k%9)//3,k%3
        source+=sum(valid[py+ky,px+kx] for py in range(2) for px in range(2))
    first_phase=2*(864-K)+6*K+2*Q+3*A
    common=dict(source_words=int(source),z_reads=A+3840,z_writes=40+A,first_issues=A,mac_issues=M,
        sum_issues=0,merge_issues=0,pattern_copies=0,outputs=3840)
    modes={7:dict(common,cycles=1482+first_phase+7776+2*M,weight_words=Q+96,second_weight_words=96,
                psum_reads=480+M,psum_writes=480+M,update_issues=M),
           8:dict(common,cycles=1482+first_phase+8640+2*M,weight_words=Q+M,second_weight_words=M,
                psum_reads=480,psum_writes=960,update_issues=480)}
    # Direct uses the same static k_live per actual output destination.
    d=dict(cycles=1442,source_words=0,weight_words=0,psum_reads=480,psum_writes=480,
        sum_issues=0,merge_issues=0,update_issues=0,z_reads=0,z_writes=0,first_issues=0,mac_issues=0,second_weight_words=0,pattern_copies=0,outputs=3840)
    for cg in range(24):
        G=int(live[:,cg].sum());d['cycles']+=112 if G else 2
        if not G:continue
        d['source_words']+=4*int(valid.sum())
        for sy in range(4):
            for sx in range(4):
                raw=words[4*cg:4*cg+4,sy,sx]
                if not np.any(raw):continue
                for py in range(2):
                    for px in range(2):
                        if not(0<=sy-py<3 and 0<=sx-px<3):continue
                        ks=np.arange(4*cg,4*cg+4)*9+(sy-py)*3+(sx-px)
                        a,b,c,e=[int(x) for x in raw*kl[ks]]
                        u=a|b;v=c|e
                        if not(u|v):d['cycles']+=G*2;continue
                        q=sum(bool(x) for x in (a,b,c,e));j=int(bool(a&b))+int(bool(c&e));h=(u&v).bit_count();l=(u|v).bit_count()
                        d['cycles']+=G*(3+q+j+3*l+h);d['weight_words']+=G*q
                        d['sum_issues']+=G*(j+h);d['merge_issues']+=G*h
                        d['update_issues']+=G*l;d['psum_reads']+=G*l;d['psum_writes']+=G*l
    modes[6]=d
    return modes,dict(fixture=name,K_live=K,active_Q1_words=Q,first_activity_A=A,nonzero_z=Z,MACs_per8lane=M,
        first_phase_cycles=first_phase,z_range=[int(z.min()),int(z.max())],p_range=[int(p.min()),int(p.max())],raw_gold_values=p.size)

def main():
    rows=json.loads((RTL/'results.json').read_text());cache={};checks=0
    for r in rows:
        if r['fixture'] not in cache:cache[r['fixture']]=predict(r['fixture'])
        pred=cache[r['fixture']][0][r['mode']]
        for field,expected in pred.items():
            observed=r[field]
            if field=='cycles':observed-=r['source_stalls']+r['weight_stalls']+r['output_stalls']
            assert observed==expected,(r['fixture'],r['mode'],r['stall'],r['command'],field,observed,expected)
            checks+=1
        assert r['configuration_cycles']==14017;checks+=1
    out=dict(complete=True,RTL_rerun=False,GPU_rerun=False,runs=len(rows),checked_RTL_outputs=sum(r['outputs'] for r in rows),
        independently_rebuilt_gold_values=sum(v[1]['raw_gold_values'] for v in cache.values()),counter_checks=checks,all_match=True,
        fixtures=[dict(v[1],predictions=v[0]) for v in cache.values()])
    (HERE/'review_integer_counts.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='fixtures'},indent=2))
if __name__=='__main__':main()
