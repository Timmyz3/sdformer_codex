"""One fixed K=4 split; optimistic queue lower bound, not RTL cycles."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
d=np.load(HERE.parents[1]/'support_lut_execution_20260915/cases.npz')
a=d['A'].astype(np.int64)
P=np.maximum(a,0).sum(1)[:,None]
N=np.minimum(a,0).sum(1)[:,None]
rows=[]

def pooled_timeline(groups):
    # Fixed alternate interface: 4 group descriptors, one extracted target/clock,
    # 10 target slots, 10 MACs, 10 source cycles + 2 drain per batch.
    # Whole P remains in existing ybuf. No fractional issue or oracle completion.
    descriptors=[]; queue=0; busy=0; t=0; gi=0
    main_left=groups[0]['main']; main_done=False
    stall=0; batches=0; peak_desc=0; peak_queue=0
    while not (main_done and not descriptors and queue==0 and busy==0):
        t+=1
        if busy:busy-=1
        if not busy and (queue==10 or (main_done and not descriptors and queue)):
            queue=0;busy=12;batches+=1
        if descriptors and queue<10:
            descriptors[0]-=1;queue+=1
            if descriptors[0]==0:descriptors.pop(0)
        if not main_done:
            if main_left>1:main_left-=1
            elif len(descriptors)<4 or groups[gi]['targets']==0:
                if groups[gi]['targets']:descriptors.append(groups[gi]['targets'])
                gi+=1
                if gi==len(groups):main_done=True
                else:main_left=groups[gi]['main']
            else:stall+=1
        peak_desc=max(peak_desc,len(descriptors));peak_queue=max(peak_queue,queue)
    return dict(cycles=t,descriptor_stall=stall,batches=batches,
        peak_descriptors=peak_desc,peak_target_queue=peak_queue)

for i in np.flatnonzero(d['is_real']):
    y=d['S'][i].astype(np.int64)@d['W'][i].astype(np.int64)
    tau=d['tau'][i].astype(np.int64)
    pos=d['positive_gain'][i]
    const=d['constant_channels'][i]
    baseline=main=finish=jobs=undecided=pooled=0
    ps=[]
    for p in range(32):
        direct=producer=tail_done=nj=nu=0;groups=[]
        for first in range(0,96,8):
            yy=y[p,:,first:first+8]
            e=int(abs(yy).max()).bit_length()
            e=max(e,1)
            hits=[]
            for n in range(1,e+1):
                m=e-n
                v=a@(yy>>m)
                lo=(v<<m)+N*((1<<m)-1)
                hi=(v<<m)+P*((1<<m)-1)
                positive=pos[first:first+8][None,:]
                locked=const[first:first+8][None,:] | np.where(positive,(lo>=tau[:,first:first+8])|(hi<tau[:,first:first+8]),(lo>tau[:,first:first+8])|(hi<=tau[:,first:first+8]))
                hits.append(locked)
                if locked.all():break
            depth=len(hits)
            direct+=depth+2 # actual GROUP + planes + pack store
            consumed=min(depth,4)
            producer+=consumed+2
            remaining=~hits[consumed-1]
            count=int(remaining.any(0).sum())
            # A 10-MAC T10 finisher spends ten source terms per active h.
            # Best-case no drain/init, no port clash, no pack write fee.
            if count:tail_done=max(tail_done,producer)+10*count
            nj+=count;nu+=int(remaining.sum())
            groups.append(dict(main=consumed+2,targets=int(remaining.sum())))
        critical=max(producer,tail_done)
        packed=pooled_timeline(groups)
        ps.append(dict(p=p,cert_compute=direct,main_compute=producer,
            finisher_end_lower_bound=tail_done,critical_lower_bound=critical,
            unresolved_h=nj,unresolved_t_h=nu,pooled=packed))
        baseline+=direct;main+=producer;finish+=critical;jobs+=nj;undecided+=nu
        pooled+=packed['cycles']
    rows.append(dict(case=str(d['case_name'][i]),cert_compute=baseline,main_compute=main,
        critical_lower_bound=finish,pooled_model_cycles=pooled,unresolved_h=jobs,unresolved_t_h=undecided,positions=ps))
result=dict(scope='fixed K4 CPU opportunity / optimistic real-release schedule lower bound, not hardware cycles',
    spare_engine='10 of existing multiplier/ALU lanes; ten noncausal source terms per h; omitted initialization/drain/port/pack conflicts only favor candidate',
    pooled_scope='CPU finite-control estimate only; bitmap extraction/data Y mux/pack interference not yet RTL; not service or Fmax',
    cases=rows,totals={k:sum(r[k] for r in rows) for k in ['cert_compute','main_compute','critical_lower_bound','pooled_model_cycles','unresolved_h','unresolved_t_h']})
(HERE/'group_tail_probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['totals']))
