#!/usr/bin/env python3
"""Compare the frozen independent raw reduction with the complete primary analysis."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ATOL,RTOL=1e-10,1e-12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    independent_path=HERE/'actual_reduction_v1/independent_reduction.json'
    primary_path=ROOT/'analysis_v1/analysis.json'
    independent=json.loads(independent_path.read_text());primary=json.loads(primary_path.read_text())
    assert independent['status']=='PASS' and primary['state_layout_count']==40 and primary['evaluation_endpoint_count']==320
    assert independent['analysis_protocol_sha256']==primary['analysis_protocol_sha256']
    stats={'numeric_comparisons':0,'exact_non_numeric_comparisons':0,'maximum_absolute_difference':0.,'maximum_relative_difference':0.,'maximum_location':None}
    def compare(a,b,path):
        if isinstance(a,dict):
            assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
            for k in a:compare(a[k],b[k],path+'/'+k)
        elif isinstance(a,list):
            assert isinstance(b,list) and len(a)==len(b),(path,'length')
            for i,(v,w) in enumerate(zip(a,b)):compare(v,w,path+'/'+str(i))
        elif type(a) in (float,int) and type(b) in (float,int):
            assert math.isfinite(a) and math.isfinite(b),path
            difference=abs(a-b);relative=difference/abs(b) if b else difference
            stats['numeric_comparisons']+=1
            if difference>stats['maximum_absolute_difference']:
                stats.update(maximum_absolute_difference=difference,maximum_location=path)
            stats['maximum_relative_difference']=max(stats['maximum_relative_difference'],relative)
            assert math.isclose(a,b,rel_tol=RTOL,abs_tol=ATOL),(path,a,b)
        else:
            stats['exact_non_numeric_comparisons']+=1
            assert type(a)==type(b) and a==b,(path,a,b)
    frame_regions=scope_regions=gate_families=0
    for condition,value in independent['conditions'].items():
        for cell,frames in value['all_endpoints'].items():
            original=primary['all_endpoints'][condition][cell]
            assert len(frames)==len(original)==8
            for frame,record in zip(frames,original):
                assert frame['target_id']==record['target_id'] and frame['sequence']==record['sequence']
                compare(frame['regions'],record['regions'],condition+'/'+cell+'/'+frame['target_id'])
                frame_regions+=len(frame['regions'])
        primary_summaries={cell:{scope:{region:dict(metric,empty_region_frames=(8 if scope=='overall' else 4)-metric['frames_with_region'])
            for region,metric in regions.items()} for scope,regions in scopes.items()}
            for cell,scopes in primary['summaries'][condition].items()}
        compare(value['summaries'],primary_summaries,condition+'/summaries')
        scope_regions+=sum(len(regions) for scopes in value['summaries'].values() for regions in scopes.values())
        for arm,gate in value['candidate_gates'].items():
            other=primary['candidate_gates'][condition][arm]
            for family in ('normal_references','matched_recovered_references','own_checkpoint_references','shared_fine_preserved'):
                compare(gate[family],other[family],condition+'/'+arm+'/'+family);gate_families+=1
            compare(gate['ordinary_base_quality_pass'],other['ordinary_base_quality_pass'],condition+'/'+arm+'/all_gates')
        normal=all(g['pass'] for g in value['candidate_gates']['CC']['normal_references'].values())
        compare(normal,primary['ordinary_reference_normality'][condition],condition+'/ordinary_normality')
    for key,value in independent['input_results'].items():
        assert value['result_sha256']==primary['inputs'][key]['sha256']
    assert (frame_regions,scope_regions,gate_families)==(5120,1920,16)
    receipt={'schema':'flow_r13_independent_primary_agreement_v1','status':'PASS','comparator_sha256':sha(Path(__file__)),
             'independent_raw_reduction_sha256':sha(independent_path),'primary_analysis_sha256':sha(primary_path),
             'raw_reducer_sha256':independent['auditor_sha256'],'primary_analyzer_sha256':primary['script_sha256'],
             'counts':{'frame_regions':frame_regions,'scope_regions':scope_regions,'candidate_condition_gate_families':gate_families,
                       'paired_reference_gates':32,'region_scope_gate_decisions':288},
             'tolerance':{'atol':ATOL,'rtol':RTOL},'comparison':stats,
             'schema_normalization':'Primary omits empty_region_frames; independently derive as fixed scope size (8 overall / 4 per sequence) minus frames_with_region. All five primary fields compared directly.',
             'scope':'Complete independent raw metrics vs complete primary analysis; all four gate families, no threshold change or refitting.'}
    out=HERE/'actual_reduction_v1/primary_agreement_v1.json'
    assert not out.exists(),'Fresh receipt required'
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
