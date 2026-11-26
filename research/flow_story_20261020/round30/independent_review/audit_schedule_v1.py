#!/usr/bin/env python3
"""Independent metadata-only reconstruction; never imports the implementation."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
MEMBERS=HERE.parents[2]/'flow_story_20261019/round27/independent_review/membership_v1.json'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mode(row,sequences):
    v=row['sequence_visit_index'];p=row['pass_index'];s=sequences.index(row['sequence'])
    require(v in range(1,33) and p in (1,2),'Fixed two-pass sequence visits')
    return ('Q4','Q4','Q8','M')[(v-1+s+2*(p-1))%4]
def audit(m):
    seqs=m['sequences'];rows=m['train_order']
    require(len(seqs)==8 and len(set(seqs))==8 and len(rows)==512,'Fixed eight-sequence512 schedule')
    schedule=[];by_id=collections.defaultdict(list)
    for i,r in enumerate(rows,1):
        require(r['step']==i and r['pass_index']==1+(i-1)//256,'Exact step/pass sequence')
        require(r['sequence']==seqs[(i-1)%8] and r['sequence_visit_index']==1+((i-1)%256)//8,
                'Original balanced visit order')
        expected=hashlib.sha256(f"flow-r27-crop-20261019:{i}:{r['target_id']}".encode()).digest()
        require(r['crop']==dict(top=int.from_bytes(expected[:8],'big')%193,
                                left=int.from_bytes(expected[8:16],'big')%257,
                                height=288,width=384),'Original independently hashed crop unchanged')
        chosen=mode(r,seqs)
        schedule.append(dict(step=i,target_id=r['target_id'],sequence=r['sequence'],
                             pass_index=r['pass_index'],mode=chosen,crop=r['crop']))
        by_id[r['target_id']].append(chosen)
    require(len(by_id)==256 and all(len(v)==2 and v.count('Q4')==1
            and (v.count('Q8')+v.count('M'))==1 for v in by_id.values()),
            'Every identity sees Q4 and exactly one other mode; not mode-locked')
    count=dict(collections.Counter(r['mode'] for r in schedule))
    require(count==dict(Q4=256,Q8=128,M=128),'Exact512 student-mode allocation')
    by_seq_pass={}
    for s in seqs:
        by_seq_pass[s]={}
        for p in (1,2):
            counts=dict(collections.Counter(r['mode'] for r in schedule if r['sequence']==s and r['pass_index']==p))
            require(counts==dict(Q4=16,Q8=8,M=8),'Per sequence per pass16/8/8')
            by_seq_pass[s][str(p)]=counts
    crop={}
    for selected in ('Q4','Q8','M'):
        subset=[r for r in schedule if r['mode']==selected]
        crop[selected]={k:dict(min=min(r['crop'][k] for r in subset),max=max(r['crop'][k] for r in subset),
                            mean=statistics.mean(r['crop'][k] for r in subset),
                            population_sd=statistics.pstdev(r['crop'][k] for r in subset))
                        for k in ('top','left')}
    return dict(schema='flow_r30_independent_schedule_audit_v1',status='PASS',
                mode_counts=count,per_sequence_per_pass=by_seq_pass,identity_count=256,
                identity_modes_Q4_once_plus_Q8_or_M=True,crop_descriptive=crop,schedule=schedule,
                metadata_only=True,GT_or_event_values_read=False,model_forwards=0,
                scope='Deterministic counterbalancing, not a claim of randomization or statistical independence. '
                      'No crop or mode selected using quality; two-pass online losses are not paired same-input outcomes.')
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Fresh audit output')
    require(sha(MEMBERS)==MEMBERS_SHA,'Fixed original membership bytes')
    result=audit(json.loads(MEMBERS.read_text()))
    result.update(auditor_sha256=sha(__file__),membership_sha256=MEMBERS_SHA)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],mode_counts=result['mode_counts'],sha256=sha(a.output))))
if __name__=='__main__':main()
