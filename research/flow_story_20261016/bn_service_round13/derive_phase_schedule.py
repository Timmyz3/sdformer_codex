#!/usr/bin/env python3
"""Independent phase-by-phase derivation from the frozen complete service result.

Does not import service_model. Each channel group is fully retired before reuse.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=HERE/'actual54_serial_service.json'; data=json.loads(source.read_text()); cfg=data['config']
rows=[]
for row in data['records']:
    geo=row['geometry']; g=geo['group_channels']; m=geo['samples']; e=m*g
    k=row['chunks_per_group_per_pass']; p=k*g
    carry=0
    for motif in row['chunk_motifs']:
        n=motif['samples']; local=0
        while n>1:
            local+=(n%2)*g; n=(n+1)//2
        carry+=local*motif['repeats_per_pass_per_group']
    pair=e-p
    shared=dict(scratch_read=2*pair+carry+p,scratch_write=e+pair+carry,
                copy=carry,command_accept=pair+carry,command_retire=pair+carry,
                chunk_config=k,chunk_finish=k)
    mean=Counter(shared); mean.update(external_read=e,state_read=p,state_write=p+g,add=e-g,mul=g,mean_barrier=1)
    variance=Counter(shared); variance.update(external_read=e,state_read=e+p-g,state_write=p,add=e-g,sub=e,mul=e)
    affine=Counter(external_read=2*g,state_read=2*g,state_write=2*g,add=g,sub=g,mul=3*g,rsqrt=g,affine_barrier=1)
    normalize=Counter(external_read=e,external_write=e,state_read=2*e,add=e,mul=e)
    now=0; phases=[]; combined=Counter()
    for name,count in [('mean_and_barrier',mean),('centered_variance',variance),('affine_and_barrier',affine),('normalize',normalize)]:
        cycles=sum(v*max(cfg['latency'][op],cfg['initiation_interval'].get(op,1)) for op,v in count.items())
        phases.append(dict(name=name,start=now,retire=now+cycles,operations=dict(count)))
        combined.update(count); now+=cycles
    assert {key:val*geo['groups'] for key,val in combined.items()}==row['transactions_and_operations']
    assert now*geo['groups']==row['conditional_serial_service_cycles']
    rows.append(dict(name=row['name'],groups=geo['groups'],group_channels=g,cycles_per_group=now,
                     first_group_phases=phases,repeat_start_formula='group_id * cycles_per_group',
                     mean_ready_first_group=phases[0]['retire'],normalize_ready_first_group=phases[2]['retire'],
                     BN_done=row['conditional_serial_service_cycles'],group_reuse_requires_previous_group_normalize_retired=True))
out=HERE/'actual54_phase_schedule.json'; assert not out.exists()
out.write_text(json.dumps(dict(source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
                              rows=rows,all_phase_counts_match=True,whole_network_schedule=False),indent=2,sort_keys=True)+'\n')
print('All 54 per-phase operation and cycle sums independently match; output',out.name)
