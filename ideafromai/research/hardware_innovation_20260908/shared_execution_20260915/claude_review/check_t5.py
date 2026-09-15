"""Independent bounded T5 arithmetic/protocol audit; original package read-only."""
from pathlib import Path
import json, subprocess
import numpy as np

H=Path(__file__).resolve().parent
B=H.parents[1]
C=B/'claude_fusion_trials_20260914'
WIDTH=24
FULL=(1<<10)-1

def jwrite(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def exponent_abs(y):return max(abs(int(v)) for v in y).bit_length()
def exponent_signed(y):return max((int(v) if v>=0 else ~int(v)).bit_length() for v in y)
def signed(v,b):return (int(v)+(1<<(b-1)))%(1<<b)-(1<<(b-1))

def first_lock(y,A,thr,e):
    P=np.maximum(A,0).sum(1);N=np.minimum(A,0).sum(1)
    actual=A@y;old_lo=old_hi=None;locked=np.zeros(10,bool);decision=np.zeros(10,bool)
    for fed,j in enumerate(range(e-1,-1,-1) if e else [0],1):
        vtop=A@(y>>j)
        lo=(vtop<<j)+N*((1<<j)-1);hi=(vtop<<j)+P*((1<<j)-1)
        assert np.all(lo<=actual) and np.all(actual<=hi)
        if old_lo is not None:assert np.all(lo>=old_lo) and np.all(hi<=old_hi)
        new=((lo>=thr)|(hi<thr))&~locked
        decision[new]=lo[new]>=thr[new];locked|=new
        old_lo,old_hi=lo,hi
        if locked.all():
            assert np.array_equal(decision,actual>=thr)
            return fed
    raise AssertionError('no final lock')

def write_case(directory,A,Y,thresholds,directions,exponents=None):
    directory.mkdir(exist_ok=True)
    def hx(path,arr,bits):path.write_text(''.join(f'{int(x)&((1<<bits)-1):0{bits//4}x}\n' for x in np.asarray(arr,dtype=object).ravel()))
    hx(directory/'a.hex',A,16)
    hx(directory/'pn.hex',np.stack([np.maximum(A,0).sum(1),np.minimum(A,0).sum(1)],axis=1),48)
    for t in range(10):
        vals=[(int(directions[g])<<63)|(int(thresholds[g,t])&((1<<48)-1)) for g in range(len(Y))]
        vals+= [0]*(3072-len(vals))
        hx(directory/f'tau_t{t}.hex',vals,64)
    for bf in [False,True]:
        out=[]
        for g,y in enumerate(Y):
            e=(int(exponents[g]) if exponents is not None else exponent_abs(y)) if bf else 24
            signw=sum((int(y[s])<0)<<s for s in range(10))
            out.append(f'G {g} {signw:03x}'+(f' {e}' if bf else ''))
            for bit in range(e-1,-1,-1):
                plane=sum(((int(y[s])>>bit)&1)<<s for s in range(10))
                out.append(f'B {plane:03x}')
        (directory/('stim_bf.txt' if bf else 'stim_fx.txt')).write_text('\n'.join(out)+'\n')

def run(directory,mode):
    output=directory/f'rtl_{mode}.txt'
    args=[str(H/'obj/Vcert_gate_core'),f'+a={directory}/a.hex',f'+pn={directory}/pn.hex']
    args += [f'+tau_t{t}={directory}/tau_t{t}.hex' for t in range(10)]
    args += [f'+stim={directory}/stim_{"bf" if mode.startswith("bf") else "fx"}.txt',f'+mode={mode}',f'+out={output}']
    r=subprocess.run(args,cwd=H,check=True,text=True,capture_output=True)
    records=[]
    for line in output.read_text().splitlines():
        if line.startswith('R '):
            _,d,n=line.split();records.append((int(d,16),int(n)))
    return records

def main():
    rng=np.random.default_rng(20260915)
    # Dense mixed-sign A, including exact signed16 extrema.
    A=rng.integers(-32768,32768,(10,10),dtype=np.int64);A[0,0]=-32768;A[1,1]=32767
    cases=[np.zeros(10,np.int64),np.full(10,-(1<<23),np.int64),np.full(10,(1<<23)-1,np.int64),
           np.full(10,-1,np.int64),np.array([-(1<<23),(1<<23)-1,0,-1,1,-2,2,-4,4,8],np.int64)]
    for k in range(24):
        y=np.zeros(10,np.int64);y[k%10]=-(1<<k);cases.append(y)
    for k in range(23):
        y=np.zeros(10,np.int64);y[k%10]=(1<<k);cases.append(y)
    cases += list(rng.integers(-(1<<23),1<<23,(256,10),dtype=np.int64))
    Y=np.stack(cases);G=len(Y)
    V=Y@A.T
    delta=np.resize(np.array([-10000000,-2,-1,0,1,2,10000000],np.int64),(G,10))
    thresholds=V+delta
    directions=np.arange(G)%2
    assert np.max(abs(thresholds))<1<<47
    for y in Y:
        e=exponent_abs(y)
        assert 0<=e<=24
        assert np.array_equal(y>>e,-(y<0).astype(np.int64))
        minimal=exponent_signed(y)
        assert 0<=minimal<=23 and np.array_equal(y>>minimal,-(y<0).astype(np.int64))
        if minimal:assert not np.array_equal(y>>(minimal-1),-(y<0).astype(np.int64))
    d=H/'legal';write_case(d,A,Y,thresholds,directions)
    expected=np.where(directions[:,None]>0,V>=thresholds,V<thresholds)
    packed=(expected.astype(np.int64)*(1<<np.arange(10))).sum(1)
    results={}
    for mode in ['fx_full','fx_cert','bf_full','bf_cert']:
        records=run(d,mode);assert len(records)==G
        diff=sum(int(dec!=packed[g]) for g,(dec,n) in enumerate(records))
        cycle_diff=0
        for g,(dec,n) in enumerate(records):
            e=exponent_abs(Y[g]) if mode.startswith('bf') else 24
            want=first_lock(Y[g],A,thresholds[g],e) if mode.endswith('cert') else max(e,1)
            cycle_diff+=n!=want
        assert diff==0 and cycle_diff==0,(mode,diff,cycle_diff)
        results[mode]=dict(groups=G,decisions=G*10,mismatch_groups=diff,plane_count_mismatches=int(cycle_diff))
    # The original generator admits signed49, then drops bit48/bit47 semantics into signed48 RTL.
    zero=np.zeros((1,10),np.int64)
    wrong_thr=np.full((1,10),1<<47,np.int64)
    d=H/'threshold49';write_case(d,A,zero,wrong_thr,np.ones(1,np.int64))
    got=run(d,'bf_cert')[0]
    assert got[0]==FULL and np.all((zero@A.T)<wrong_thr)
    trunc=dict(model_signed49_admits=True,threshold_before=int(1<<47),
               threshold_after_signed48=signed(1<<47,48),expected_decision_word=0,rtl_decision_word=got[0])
    # If a new front-end optimizes e, e=0 can also encode {-1,0}: original dummy B=0 is then wrong.
    diag=np.eye(10,dtype=np.int64)*4096
    minus=np.full((1,10),-1,np.int64);boundary=np.full((1,10),-6144,np.int64)
    d=H/'minimal_exp_zero';write_case(d,diag,minus,boundary,np.ones(1,np.int64),[0])
    got=run(d,'bf_cert')[0]
    assert got[0]==0 and np.all((minus@diag.T)>=boundary)
    minimal=dict(original_abs_policy_e=1,new_minimal_policy_e=0,expected_decision_word=FULL,
                 rtl_with_unchanged_zero_dummy=got[0],classification='not a bug under original maxabs contract; front-end integration trap')
    # Exact unquantized BN example: gamma=-1, mu=0, sigma=1, A=I, theta=1, Y=0.
    # Correct predicate is Y<=-1. T5 tau flips to +1 and flips direction again => Y<=+1.
    wrong_tau=np.full((1,10),(1<<26)+1,np.int64)
    d=H/'negative_gamma';write_case(d,diag,zero,wrong_tau,np.zeros(1,np.int64))
    got=run(d,'bf_cert')[0]
    assert got[0]==FULL
    gamma=dict(gamma=-1,Y=0,mu=0,sigma=1,theta=1,A_diagonal=1,
               native_BN_gate=False,t5_reference_gate=True,rtl_word=got[0],
               bug='tau is already sign(gamma)-multiplied while V is not, then comparator direction is flipped again')
    # Read-only checks of stored four traces and threshold outputs.
    stored={};threshold_range=[]
    for name in ['s0_stage0','s0_stage3','s10_stage0','s10_stage3']:
        rd=C/'results/t5_rtl'/name
        exp=np.load(rd/'expected.npz');count={}
        for mode in ['fx_full','fx_cert','bf_full','bf_cert']:
            records=[line.split() for line in (rd/f'rtl_{mode}.txt').read_text().splitlines() if line.startswith('R ')]
            pred=np.array([int(r[1],16) for r in records],np.int64)
            pred=(pred[:,None]>>np.arange(10))&1
            assert np.array_equal(pred,exp['dec'])
            cycles=np.array([1+int(r[2]) for r in records])
            if mode=='bf_full':
                e=[int(line.split()[3]) for line in (rd/'stim_bf.txt').read_text().splitlines() if line.startswith('G ')]
                assert np.array_equal(cycles,1+np.maximum(e,1))
            elif mode=='fx_full':assert np.all(cycles==25)
            else:assert np.array_equal(cycles,exp['cyc_bf' if mode=='bf_cert' else 'cyc_fx'])
            count[mode]=dict(groups=len(records),mean_cycles=float(cycles.mean()),mismatches=0)
        trace=np.load(B/'bn_state'/f'trace_{name}.npz')
        ts=[]
        for t in range(10):
            ts += [signed(int(x,16)&((1<<48)-1),48) for x in (rd/f'tau_t{t}.hex').read_text().split()]
        threshold_range+=ts
        stored[name]=dict(gamma_negative=int((trace['gamma']<0).sum()),gamma_zero=int((trace['gamma']==0).sum()),
                          threshold_abs_max=max(abs(x) for x in ts),modes=count)
    tau_changes={}
    for stage in [0,3]:
        aa=np.load(B/'bn_state'/f'trace_s0_stage{stage}.npz');bb=np.load(B/'bn_state'/f'trace_s10_stage{stage}.npz')
        same=all(np.array_equal(aa[k],bb[k]) for k in ['W','A','gamma','beta','bias','center'])
        total=diff=0
        for t in range(10):
            x=(C/f'results/t5_rtl/s0_stage{stage}/tau_t{t}.hex').read_text().split()
            y=(C/f'results/t5_rtl/s10_stage{stage}/tau_t{t}.hex').read_text().split()
            assert len(x)==len(y);total+=len(x);diff+=sum(a!=b for a,b in zip(x,y))
        tau_changes[str(stage)]=dict(same_model_fields=same,threshold_words=total,changed_between_frames=diff)
    result=dict(passed=True,legal_RTL=results,threshold49_counterexample=trunc,negative_gamma_counterexample=gamma,
                optimized_e0_warning=minimal,stored_trace_readback=stored,tau_between_frames=tau_changes,
                tests_scope='unmodified original DUT/TB; small new RTL fixtures and stored-result replay, no original trace regeneration')
    jwrite(H/'audit_results.json',result)
    print(json.dumps(dict(passed=True,legal_groups=G,total_new_RTL_decisions=G*40+30,
                         threshold49_counterexample=trunc,negative_gamma_counterexample=gamma,tau_changes=tau_changes)))

if __name__=='__main__':main()
