"""One fixed <=2 signed-power projection; actual CPU gold and common source RTL."""
from pathlib import Path
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','DA_DEFAULT_THREADS'):
    os.environ[_name]='1'
import argparse
import importlib.util
import json
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
spec=importlib.util.spec_from_file_location('new_source_compiler',NEW/'source_execution/run.py')
source=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def candidate_values():
    terms=[sign*(1<<exponent) for exponent in range(16) for sign in (-1,1)]
    values={0,*terms,*(a+b for a in terms for b in terms)}
    return np.asarray(sorted(v for v in values if -32768<=v<=32767),np.int64)


def project(values,candidates):
    original=np.asarray(values,np.int64)
    high=np.searchsorted(candidates,original)
    high=np.minimum(high,len(candidates)-1)
    low=np.maximum(high-1,0)
    a,b=candidates[low],candidates[high]
    da,db=np.abs(original-a),np.abs(original-b)
    selected=np.where((da<db)|((da==db)&(np.abs(a)<=np.abs(b))),a,b)
    return selected.astype(np.int16)


def validate_projection(candidates):
    # A separate brute-force nearest/tie oracle for every actually used q.
    for structure,key in [('dense','As_q16'),('lifting40','lifting_q12')]:
        p=dict(np.load(NEW/'algorithm/matched_training'/structure/'stage320/deployed_constants.npz'))
        for q,out in zip(p[key].reshape(-1),project(p[key],candidates).reshape(-1)):
            expected=min(map(int,candidates),key=lambda v:(abs(v-int(q)),abs(v),v))
            assert int(out)==expected
    assert project(np.asarray([0,-32768,32767,23,-23,22,-22]),candidates).tolist()==[0,-32768,32767,24,-24,20,-20]


def run(structure,candidates):
    key='lifting_q12' if structure=='lifting40' else 'As_q16'
    endpoint=NEW/'algorithm/matched_training'/structure/'stage320'
    original={k:np.asarray(v).copy() for k,v in dict(np.load(endpoint/'deployed_constants.npz')).items()}
    p={k:v.copy() for k,v in original.items()}
    p[key]=project(original[key],candidates).astype(original[key].dtype)
    assert p[key].shape==original[key].shape and p[key].dtype==original[key].dtype
    for k in p:
        if k!=key:assert np.array_equal(p[k],original[k]),k
    out=HERE/structure
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'deployed_constants.npz',**p)
    delta=p[key].astype(np.int64)-original[key].astype(np.int64)
    quant=dict(structure=structure,changed_key=key,coefficients=int(delta.size),
        changed_coefficients=int(np.count_nonzero(delta)),max_abs_integer_error=int(np.abs(delta).max()),
        squared_integer_error=int(np.square(delta).sum()),candidate_integer_values=len(candidates),
        original_values=original[key],projected_values=p[key],integer_delta=delta,
        exponent=int(p['lifting_fraction_bits'] if structure=='lifting40' else p['As_exponent']),
        unchanged_fields=[k for k in p if k!=key],literal_forward_structure=structure,
        literal_forward_constants='deployed_constants.npz',new_function=bool(np.any(delta)),
        new_AEE10=None,new_AEE825=None,no_parent_quality_inheritance=True,
        rule='Nearest signed16 integer representable by <=2 signed powers, tie smaller absolute integer; fixed existing exponent.')
    source.dump(out/'quantization.json',quant)
    # Package and CPU gold are available before potentially slower compilation.
    inputs=source.PREV/'hardware/source_rtl_inputs'
    cases=[c for c in json.loads((inputs/'manifest.json').read_text())['cases'] if c['axis']=='ordinary']
    gold_package={}
    checks=[]
    for case in cases:
        label=case['window'];input_file=inputs/case['input_file']
        tiles=np.fromfile(input_file,dtype='<i4').reshape(-1,10,8)
        x=tiles.transpose(1,0,2)
        baseline=source.literal_gate(x,original,structure)
        gate=source.literal_gate(x,p,structure)
        gold=sum(gate[t].astype(np.uint16)<<t for t in range(10))
        gold.astype('<u2').tofile(out/(label+'_gates.bin'))
        parent_report=json.loads((NEW/'source_execution'/structure/'results.json').read_text())
        parent_gold=np.fromfile(NEW/'source_execution'/structure/(label+'_gates.bin'),'<u2').reshape(gold.shape)
        assert np.array_equal(sum(baseline[t].astype(np.uint16)<<t for t in range(10)),parent_gold)
        height,width=case['height'],case['width']
        def spatial(a):return a.reshape(10,height,width,12,8).transpose(0,3,4,1,2).reshape(10,96,height,width)
        gold_package[label+'_I24']=spatial(x).astype(np.int32)
        gold_package[label+'_source_gate']=spatial(gate).astype(bool)
        gold_package[label+'_parent_source_gate']=spatial(baseline).astype(bool)
        checks.append(dict(window=label,source_values=int(x.size),gate_bits=int(gate.size),
            gate_bit_differences_vs_parent=int(np.count_nonzero(gate!=baseline)),
            gate_bit_difference_rate_vs_parent=float(np.mean(gate!=baseline)),
            gate_words_different_vs_parent=int(np.count_nonzero(gold!=parent_gold)),
            parent_gate_ones=int(baseline.sum()),new_gate_ones=int(gate.sum()),
            input=str(input_file.relative_to(source.OPEN))))
    np.savez_compressed(out/'cpu_gold.npz',**gold_package)
    source.dump(out/'cpu_checks.json',dict(rows=checks,source_layout='T,C,H,W in NPZ; tile,T,lane in tracked input',
        parent_gate_recomputed_and_archived_gold_equal=True,no_new_AEE=True))
    print('CPU_PACKAGE_READY',structure,quant['changed_coefficients'],checks,flush=True)
    program,accounting=source.compile_graph(p,structure,out)
    compilation=json.loads((out/'compilation.json').read_text())
    compilation['constant_function']='Fixed <=2 signed-power coefficient projection of new320 endpoint; new function, no training or AEE inheritance.'
    assert compilation['full_halfstep_RNEs']==(35 if structure=='lifting40' else 0)
    source.dump(out/'compilation.json',compilation)
    source.fields(program,out/'program.txt')
    binary=source.PREV/'hardware/rtl_source/obj_dir/Vtemporal_source'
    rows=[]
    for case in cases:
        for mode in ('ready','stress'):
            row=json.loads(subprocess.run([str(binary),str(out/'program.txt'),str(inputs/case['input_file']),
                str(out/(case['window']+'_gates.bin')),mode],check=True,capture_output=True,text=True).stdout)
            row.update(window=case['window'],source_values=case['source_values'])
            rows.append(row)
    report=dict(structure=structure,quantization=quant,cpu_checks=checks,accounting=accounting,rows=rows,
        parent_accounting=parent_report['accounting'],parent_RTL_rows=parent_report['rows'],
        evidence='Same common isolated source RTL; Verilator CPU replay, no EDA/PPA/full-chain/AEE.',
        unchanged_cutoff_exponents_downstream=True,lifting_materialized_norm24=compilation['full_halfstep_RNEs'],
        quality={'AEE10':None,'AEE825':None,'new_function':True},new_X=False)
    source.dump(out/'results.json',report)
    print('SOURCE_CONSTANT_COMPLETE',structure,json.dumps(accounting),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--structure',choices=['dense','lifting40'])
    args=parser.parse_args()
    candidates=candidate_values();validate_projection(candidates)
    source.dump(HERE/'projection_rule.json',dict(values=candidates,rule='Fixed nearest, ties smaller absolute value',
        at_most_terms=2,signed16_range=[-32768,32767],no_search_over_bitwidth_or_terms=True))
    for structure in ([args.structure] if args.structure else ['dense','lifting40']):run(structure,candidates)


if __name__=='__main__':main()
