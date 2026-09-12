"""Reuse the unchanged fixed two-chain CSD generator on the new two-term dense."""
from pathlib import Path
from collections import Counter
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','DA_DEFAULT_THREADS'):
    os.environ[_name]='1'
import importlib.util
import json
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
NEW=HERE.parents[1]
spec=importlib.util.spec_from_file_location('fixed_two_chain_source',NEW/'hardware/dense_low_state_source/run.py')
fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)


def main():
    parent=HERE.parent/'dense'
    p=dict(np.load(parent/'deployed_constants.npz'))
    program,per_row=fixed.compile_fixed(p)
    checks=fixed.sched.validate_program('ordinary',program,{k:v.tolist() for k,v in p.items()})
    fixed.dump(HERE/'program.json',program)
    fixed.source.fields(program,HERE/'program.txt')
    capacity=fixed.sched.CONTRACT['instruction_ROM']['words']
    admitted=len(program)<=capacity
    counts=dict(Counter(i['kind'] for i in program))
    root=fixed.OPEN/'stage_20260912/hardware/source_rtl_inputs'
    cases=[c for c in json.loads((root/'manifest.json').read_text())['cases'] if c['axis']=='ordinary']
    actual=[];rtl=[]
    for c in cases:
        x=np.fromfile(root/c['input_file'],'<i4')
        actual.append(dict(window=c['window'],**fixed.actual_window(program,x,p)))
        if admitted:
            for mode in ('ready','stress'):
                binary=fixed.OPEN/'stage_20260912/hardware/rtl_source/obj_dir/Vtemporal_source'
                row=json.loads(subprocess.run([str(binary),str(HERE/'program.txt'),str(root/c['input_file']),
                    str(parent/(c['window']+'_gates.bin')),mode],capture_output=True,text=True,check=True).stdout)
                row['window']=c['window'];rtl.append(row)
    baseline=json.loads((parent/'results.json').read_text())
    costs=[]
    for mode in ('ready','stress'):
        new=[r for r in rtl if r['mode']==mode]
        old=[r for r in baseline['rows'] if r['mode']==mode]
        if not new:continue
        for a,b in zip(new,old):
            assert a['window']==b['window']
            for k in ('tiles','SR64_reads','SW64_writes','gate_bits_checked'):assert a[k]==b[k]
        cycles=sum(r['cycles'] for r in new);prior=sum(r['cycles'] for r in old)
        tiles=sum(r['tiles'] for r in new)
        costs.append(dict(mode=mode,tiles=tiles,cycles=cycles,cycles_per_tile=cycles/tiles,
            full_CSE_cycles=prior,full_CSE_cycles_per_tile=prior/tiles,
            change_vs_full_CSE=cycles/prior-1))
    result=dict(status='ADMITTED' if admitted else 'NOT_ADMITTED_ROM_CAPACITY',
        same_function_as='source_constant_probe/dense/deployed_constants.npz',
        generator='hardware/dense_low_state_source/run.py::compile_fixed, unchanged',
        instructions=len(program),ROM_words=capacity,instruction_counts=counts,
        base_without_nops=len(program)-counts.get('nop',0),
        RAW_nops=sum(i.get('reason')=='RAW_wait' for i in program),
        drain_nops=sum(i.get('reason')=='pipeline_drain' for i in program),
        allocated_work_RF_vectors=13,reserved_gate_RF=95,gate_RF_vectors=1,
        RF_layout=dict(input=[0,9],two_chains=[10,11],row_sum=12),
        RF_count_is_allocation_not_minimum_proof=True,
        per_row=per_row,scalar_checks=checks,actual_window_checks=actual,RTL_rows=rtl,cost_comparison=costs,
        no_new_function=True,no_new_AEE_arm=True,RTL_executed=admitted,
        interleave_executed=False,full_CSE_control_retained=True,
        evidence='Same isolated source RTL and real two I24 halos; no extra ISA/ROM, no PPA or consumer overlap.')
    fixed.dump(HERE/'results.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
