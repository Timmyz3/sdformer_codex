#!/usr/bin/env python3
"""M1086 additive zero-work repair over frozen M1072/M1056.

The only semantic change is ``work_cycles == 0``: no psum event or grant is
created, no last-write state changes, and the effective work end equals the
already-paid stream work start.  Every positive-work call delegates directly
to frozen M1056.  The production work-domain preflight and full iterator are
zero-argument functions; neither accepts caller cycles, records or coverage.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

HERE=Path(__file__).resolve().parent; HW=HERE.parent.parent
M1072_PATH=HERE/'run_m1072_c1_row_provenance_exact_1rw_source.py'
M1085=HW/'reviews/m1085_m1074_c1_full_replay_failure_audit_r1_20260830'
DOCS359=HW/'docs/359_DATE终局冻结_20260813.md'
M1072_SHA='879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c'
M1085_OUTER='ea6a4f8853ccc534be36db355b7c2e57612b2dae8af4681b500134961d2ec2a9'
DOCS359_SHA='dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4'

def require(v:bool,m:str)->None:
    if not v: raise RuntimeError(m)
def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def strict_json(path:Path)->Any:
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key: '+k);out[k]=v
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in()).throw(RuntimeError('nonfinite '+x)))
def verify_dir(path:Path,outer:str)->None:
    require(path.is_dir() and not path.is_symlink(),'M1085 directory drift')
    man=path/'SHA256SUMS'
    for line in man.read_text().splitlines():
        d,r=line.split(None,1);p=path/r.strip().lstrip('*')
        require(p.is_file() and not p.is_symlink() and sha256(p)==d,'M1085 member drift')
    inner=path/'SHA256SUMS.seal.sha256'
    require(inner.read_text().split()==[sha256(man),'SHA256SUMS'] and sha256(inner)==outer,
            'M1085 outer drift')
def load_frozen():
    require(M1072_PATH.is_file() and not M1072_PATH.is_symlink() and
            sha256(M1072_PATH)==M1072_SHA,'M1072 identity drift')
    spec=importlib.util.spec_from_file_location('m1086_frozen_m1072',M1072_PATH)
    require(spec is not None and spec.loader is not None,'cannot load M1072')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module);return module
M1072=load_frozen();M1056=M1072.M1056;M1064=M1072.M1064
DESIGNS=M1072.DESIGNS

def validate_frozen_authorities()->dict[str,Any]:
    verify_dir(M1085,M1085_OUTER)
    review=strict_json(M1085/'review.json')
    require(review['status']=='PASS_M1085_M1074_FAILURE_AUDIT__ADDITIVE_ZERO_WORK_REPAIR_ALLOWED__M1074_DO_NOT_RETRY' and
            review['repair_recommendation']['m1056_m1072_m1074_must_remain_frozen'] is True and
            sha256(M1072_PATH)==M1072_SHA and sha256(DOCS359)==DOCS359_SHA,
            'frozen authority drift')
    return {'status':'PASS_M1086_FROZEN_AUTHORITIES','m1085_outer':M1085_OUTER}

def validate_production_work(value:Any)->int:
    require(type(value) is int and value>=0,'work must be exact nonnegative int')
    require(value==0 or value>=15,'unsupported positive work interval 1..14')
    return value

def validate_dependencies(events:Any)->None:
    require(type(events) is list and all(
        bool(dep.event_id) and type(dep.delay_cycles) is int and
        dep.delay_cycles>=0 for event in events for dep in event.dependencies),
        'dependency type/value drift')

def schedule_task(plan:Any,work_start:int,
                  last_write_cycle:dict[tuple[int,int],int],
                  config:Any=M1056.ArbiterConfig())->Any:
    """Zero-work repair; all positive work is the exact frozen M1056 call."""
    require(type(plan) is M1056.TaskPlan and type(plan.work_cycles) is int,
            'exact M1056 plan/work type required')
    plan.validate();require(type(work_start) is int and work_start>=0,
                            'exact nonnegative work start required')
    if plan.work_cycles>0:
        result=M1056.schedule_task(plan,work_start,last_write_cycle,config)
        validate_dependencies(result.events)
        return result
    before=dict(last_write_cycle)
    result=M1056.TaskResult(
        task_id=plan.task_id,work_start=work_start,
        nominal_work_end=work_start,effective_work_end=work_start,
        events=[],grants={},queue_peak=0,nominal_excess_accesses=0,
        delayed_accesses=0,maximum_read_write_lifetime=0,
        raw_dependencies_pass=True)
    require(last_write_cycle==before and result.events==[] and result.grants=={} and
            result.queue_peak==result.nominal_excess_accesses==
            result.delayed_accesses==result.maximum_read_write_lifetime==0 and
            result.raw_dependencies_pass is True and
            result.nominal_work_end==result.effective_work_end==work_start,
            'zero-work semantic invariant drift')
    return result

@dataclass
class DesignStream:
    last_write:dict[tuple[int,int],int]=field(default_factory=dict)
    previous_start:int|None=None
    previous_effective_end:int|None=None
    delayed_accesses:int=0
    nominal_excess_accesses:int=0
    last_result:Any|None=None
    def consume_internal(self,plan:Any)->Any:
        require(type(plan) is M1056.TaskPlan,'internal plan type drift')
        validate_production_work(plan.work_cycles)
        if self.previous_start is None:start=plan.preprocess_cycles
        else:
            require(self.previous_effective_end is not None,'stream state drift')
            start=max(self.previous_effective_end,
                      self.previous_start+plan.preprocess_cycles)+2
        result=schedule_task(plan,start,self.last_write)
        self.previous_start=start;self.previous_effective_end=result.effective_work_end
        self.delayed_accesses+=result.delayed_accesses
        self.nominal_excess_accesses+=result.nominal_excess_accesses
        self.last_result=result;return result
    def finish_sample(self)->dict[str,int]:
        require(self.previous_effective_end is not None,'empty design sample')
        return {'cycles_after_commit':self.previous_effective_end+2+M1064.COMMIT_CYCLES_PER_SAMPLE,
                'delayed_accesses':self.delayed_accesses,
                'nominal_excess_accesses':self.nominal_excess_accesses}

def canonical_work_domain_preflight()->dict[str,Any]:
    """Exhaustive work-only gate; derives no cycle or arbitration result."""
    validate_frozen_authorities();coverage=M1072.ProvenanceCoverage()
    counts={name:Counter() for name in DESIGNS};digest=hashlib.sha256()
    with M1072.CanonicalRowReader() as reader:
        for task_id in range(M1072.TASKS):
            record=reader.derive(task_id);coverage.consume_internal(record)
            for name in DESIGNS:
                work=validate_production_work(record.works[name])
                counts[name]['zero' if work==0 else 'positive']+=1
                digest.update(f'{task_id}:{name}:{work}\n'.encode())
    proof=coverage.proof();require(proof['full_coverage_pass'],'preflight provenance failed')
    require(all(sum(row.values())==M1072.TASKS for row in counts.values()),
            'preflight design population drift')
    return {'schema':'m1086_canonical_work_domain_preflight_v1',
            'status':'PASS_M1086_ALL_TASK_DESIGN_WORK_VALUES_DEFINED',
            'tasks':M1072.TASKS,'designs':list(DESIGNS),'values_checked':M1072.TASKS*3,
            'domain':'exact_int && (work==0 || work>=15)',
            'counts':{name:dict(row) for name,row in counts.items()},
            'task_design_work_digest_sha256':digest.hexdigest(),
            'row_work_execution_provenance_digest_sha256':
                proof['execution_provenance_digest_sha256'],
            'cycles_derived_or_exported':False,'caller_supplied_work':False}

def iter_canonical_full_replay_results()->Iterator[dict[str,Any]]:
    """Zero-argument repaired production cycle iterator."""
    validate_frozen_authorities();capacity=M1064.derive_physical_capacity()
    coverage=M1072.ProvenanceCoverage();sample_rows=[]
    with M1072.CanonicalRowReader() as reader:
        streams={name:DesignStream() for name in DESIGNS}
        for task_id in range(M1072.TASKS):
            record=reader.derive(task_id);coverage.consume_internal(record)
            for name in DESIGNS:
                work=validate_production_work(record.works[name])
                streams[name].consume_internal(M1056.TaskPlan(
                    record.task_id,record.shared_preprocess_cycles,work,record.row))
            if (task_id+1)%M1072.TASKS_PER_SAMPLE==0:
                sample=task_id//M1072.TASKS_PER_SAMPLE
                sample_rows.append({'sample':sample,
                    'first_task_id':sample*M1072.TASKS_PER_SAMPLE,
                    'last_task_id':task_id,
                    'designs':{name:streams[name].finish_sample() for name in DESIGNS}})
                streams={name:DesignStream() for name in DESIGNS}
    proof=coverage.proof();require(proof['full_coverage_pass'] and
        len(sample_rows)==M1072.SAMPLES,'full provenance coverage failed')
    yield {'schema':'m1086_canonical_full_zero_work_exact_1rw_replay_result_v1',
           'status':'PASS_M1086_RAW_FULL_REPLAY_PENDING_RESULT_HAMMER',
           'samples':sample_rows,'coverage':proof,'capacity':capacity,
           'claim_boundary':{'capacity_only_214912B_admitted':False,
             'matched_cycles_admitted':False,'speedup_admitted':False,
             'rtl_cycles':False,'paper_ppa_ready':False,
             'independent_result_hammer_required':True}}

def source_small_oracle()->dict[str,Any]:
    validate_frozen_authorities();sentinel={(0,7):91};before=dict(sentinel)
    zero=M1056.TaskPlan(207,146,0,15);z=schedule_task(zero,500,sentinel)
    require(sentinel==before and z.effective_work_end==z.work_start==500,
            'zero work state mutation')
    positives=[]
    for work in (8,15,16,224,280):
        a={};b={};plan=M1056.TaskPlan(208,158,work,16)
        repaired=schedule_task(plan,700,a);frozen=M1056.schedule_task(plan,700,b)
        require(repaired==frozen and a==b,'positive-work behavior drift')
        positives.append(work)
    rejected=[]
    for value in (True,False,-1,*range(1,15)):
        try:validate_production_work(value)
        except RuntimeError:rejected.append(value)
        else:raise RuntimeError('unsupported work admitted')
    frozen=M1056.small_oracle()
    require(frozen['same_address_raw_enforced'] and
            frozen['cascade']['nominal_cycles']==20 and
            frozen['cascade']['arbitrated_cycles']==22,
            'frozen RAW/cascade drift')
    return {'status':'PASS_M1086_ZERO_WORK_SOURCE_SMALL_ORACLE',
      'zero_events':len(z.events),'zero_grants':len(z.grants),
      'zero_last_write_unchanged':sentinel==before,
      'zero_effective_end_equals_work_start':True,
      'positive_behavior_equivalent_work_values':positives,
      'production_undefined_bool_negative_1_to_14_rejected':len(rejected),
      'frozen_cascade_20_to_22':True,'full_replay_executed':False}

def real_task207_next_regression()->dict[str,Any]:
    """Bounded real rows 207/208 only; never calls either production iterator."""
    validate_frozen_authorities()
    with M1072.CanonicalRowReader() as reader:
        r207=reader.derive(207);r208=reader.derive(208)
    require((r207.sample,r207.operator,r207.chunk,r207.partition)==(0,0,0,207) and
            r207.row==15 and r207.raw_row_bytes_sha256==
            'e8636aaf63033f5c8520c127205c519a0da3f3b4e599888dcb8fe5569446f9e9' and
            all(r207.works[name]==0 for name in DESIGNS),'task207 identity')
    require((r208.sample,r208.operator,r208.chunk,r208.partition)==(0,0,0,208) and
            r208.row==16 and r208.works=={'candidate':224,
              'strongest_zero':280,'same_coordinate_bit':280},'task208 identity')
    rows={}
    for name in DESIGNS:
        stream=DesignStream(last_write={(0,16):1000})
        first=stream.consume_internal(M1056.TaskPlan(207,r207.shared_preprocess_cycles,0,r207.row))
        state=dict(stream.last_write)
        second=stream.consume_internal(M1056.TaskPlan(208,r208.shared_preprocess_cycles,
                                                     r208.works[name],r208.row))
        expected_start=max(first.effective_work_end,
                           first.work_start+r208.shared_preprocess_cycles)+2
        require(first.events==[] and state=={(0,16):1000} and
                second.work_start==expected_start and
                all(g.cycle>=1001 for eid,g in second.grants.items()
                    if eid.endswith(':b0:R')),'task208 RAW/state regression')
        rows[name]={'task207_start_end':first.work_start,
                    'task208_start':second.work_start,
                    'last_write_unchanged_across_zero':True,
                    'next_raw_predecessor_cycle':1000}
    return {'status':'PASS_M1086_REAL_TASK207_NEXT_RAW_REGRESSION',
            'task207_coordinate':[0,0,0,207],
            'task208_coordinate':[0,0,0,208],
            'designs':rows,'production_iterator_called':False}

def main()->None:
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true')
    p.add_argument('--task207-regression',action='store_true')
    a=p.parse_args();require(a.self_test^a.task207_regression,'select one source test')
    out=source_small_oracle() if a.self_test else real_task207_next_regression()
    print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
