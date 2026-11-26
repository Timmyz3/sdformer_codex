#!/usr/bin/env python3
"""Independent source-vector/table audit and replay of existing Verilator binary.

No synthesis, model forwards, GT, R30 predictions or old source modifications.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
HW=HERE.parent/'hardware'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def payload(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    require(not args.output.exists(),'New independent receipt')
    rt=HW/'rtl_results';manifest=read(rt/'ACTUAL_VECTOR_MANIFEST_V1.json')
    for path,digest in manifest['source_sha256'].items():
        require(sha(path)==digest,'Original source bytes '+path)
    vector=rt/'actual_vectors_v1.txt'
    require(sha(vector)==manifest['vectors_sha256'],'Original vector bytes')
    original=next(Path(x) for x in manifest['source_sha256'] if x.endswith('/results.json'))
    result=read(original);record=result['forward_records'][0]
    require(record['arm']=='native','Only actual native operand source; not R30 mixed output')
    archive=original.parent/record['prediction_file'];quant=original.parent/result['quantization_archive']['path']
    require(sha(archive)==record['prediction_file_sha256']
            and sha(quant)==result['quantization_archive']['sha256'],'Audited R26 artifacts')
    with np.load(archive,allow_pickle=False) as z:
        spikes={k:z[k] for k in ('sn1','sn2')}
    with np.load(quant,allow_pickle=False) as z:
        weights={k:z[k+'__q8'] for k in ('fc1','fc2')}
    for key,value in spikes.items():
        require(payload(value)==record['tensor_records'][key]['sha256']
                and np.isin(value,[0,1]).all(),'Real binary activation '+key)
    numbers=np.fromstring(vector.read_text(),sep=' ',dtype=np.int64)
    require(numbers[0]==manifest['cases']==len(manifest['records'])==96,'Exactly96 actual transactions')
    ptr=1;unique=set();golden=[];mode_counts=collections.Counter();beat_binary=beat_parallel=0
    for row in manifest['records']:
        k,q8=map(int,numbers[ptr:ptr+2]);ptr+=2
        x=numbers[ptr:ptr+k];ptr+=k
        w=numbers[ptr:ptr+16*k].reshape(16,k);ptr+=16*k
        require(q8 in (0,1) and row['mode']==('Q8' if q8 else 'Q4'),'Declared integer mode')
        fc=row['fc'];sn='sn1' if fc=='fc1' else 'sn2'
        v=spikes[sn][tuple(row['coordinate'])].astype(np.int8)
        lo,hi=row['outputs'];q=weights[fc][lo:hi]
        require(hi-lo==16 and len(v)==k and q.shape==(16,k),'Explicit16 output slice')
        require(np.array_equal(x,v) and np.array_equal(w,q),'Every text operand equals actual frozen array')
        require(payload(v)==row['input_sha256'] and payload(q)==row['q8_weight_sha256'],
                'Every actual slice payload')
        effective=w if q8 else np.floor_divide(w,16)*16
        y=effective@x
        require(np.all(y>=np.iinfo(np.int32).min) and np.all(y<=np.iinfo(np.int32).max),'Int32 domain')
        golden.append(y.astype(np.int32))
        unique.add((fc,tuple(row['coordinate']),tuple(row['outputs']),q8))
        mode_counts['Q8' if q8 else 'Q4']+=1
        beat_binary+=(8 if q8 else 4)*((k+127)//128)
        beat_parallel+=(k+15)//16
    require(ptr==len(numbers),'No extra/unparsed vector operands')
    binary=rt/'verilator/obj/Vpe_compare_top'
    replay=subprocess.run([str(binary),str(vector)],text=True,capture_output=True,timeout=60)
    require(replay.returncode==0,'Existing compiled RTL replay '+replay.stderr[-1000:])
    observed=json.loads(replay.stdout)
    require(observed==read(rt/'verilator/ACTUAL_PLUS_SYNTHETIC_V1.json'),'Exact original transaction receipt replay')
    require(observed['case_count']==206 and len(observed['runs'])==4
            and sum(x['checked_int32_values'] for x in observed['runs'])==13184,
            'Four engine/stall modes,13184 checked integer values')
    for row in observed['runs']:
        require(row['retired_rows']==206 and row['checked_int32_values']==3296,'All transactions and lanes')
        if not row['stalls']:require(row['cycles']==row['accepted_beats']+1,'Steady accepted beat plus final retirement')
    dc={}
    for folder in ('dc_binary','dc_parallel'):
        p=rt/folder;s=read(p/'summary.json')
        area=float(re.search(r'Total cell area:\s+([0-9.]+)',(p/'area.rpt').read_text())[1])
        require(area==s['area_library_units'] and s['clock_period_ns']==4.
                and s['worst_setup_slack_ns']>0,'Same4ns constrained area/timing')
        require('This design has no violated constraints.' in (p/'constraints.rpt').read_text(),'No reported constraint violation')
        command=(p/'command.log').read_text()
        require(f"area={area}" in command and f"slack={s['worst_setup_slack_ns']}" in command,
                'Exact DC summary matches original run completion')
        require(all(s[k] is False for k in ('SRAM_included','FP_scale_or_BN_included','whole_accelerator')),
                'Local integer scope only')
        dc[folder]=s
    sources=[HW/'rtl/plane_pe.sv',HW/'rtl/tb_plane_pe.cpp',HW/'rtl/synth_pe.tcl',
             HW/'rtl/export_actual_vectors.py',binary,vector,rt/'ACTUAL_VECTOR_MANIFEST_V1.json']
    sources += [rt/f/n for f in ('dc_binary','dc_parallel')
                for n in ('summary.json','area.rpt','timing.rpt','constraints.rpt','constraints.sdc','mapped.v','command.log')]
    out=dict(schema='flow_r30_independent_RTL_component_review_v1',status='PASS_WITH_COMPONENT_SCOPE',
        auditor_sha256=sha(__file__),source_sha256={str(p):sha(p) for p in sources},
        actual_transactions=96,unique_coordinate_output_mode_transactions=len(unique),
        actual_mode_counts=dict(mode_counts),actual_independent_int64_golden_sha256=payload(np.stack(golden)),
        actual_vector_beats=dict(binary=beat_binary,ordinary=beat_parallel),
        verilator_existing_binary_replay=observed,DC=dc,
        binary_to_gateadd_cell_area_ratio=dc['dc_binary']['area_library_units']/dc['dc_parallel']['area_library_units'],
        same_weight_port_bits=2048,same_int32_output_accumulators=16,
        activation_port_bits=dict(binary=128,ordinary=16),GT_read=False,model_forwards=0,
        new_synthesis=False,new_verilator_build=False,R30_quality_read=False,
        scope='Existing compiled binary replay plus independent original operand extraction/int64 arithmetic '
              'and actual DC report/source review. Not independent testbench generation, postlayout, '
              'power, gate-level equivalence, SRAM/FP32 BN, or a whole-network cycle result.')
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ('status','actual_transactions','unique_coordinate_output_mode_transactions',
        'actual_mode_counts','binary_to_gateadd_cell_area_ratio')}))
if __name__=='__main__':main()
