"""Run the four captured axes/frames; aggregate counts without inventing cycles."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import argparse
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
JOINT=HERE.parent
CAP=JOINT/'full_capture4/capture'


def job(pair):
    axis,frame=pair
    predictor=(JOINT/'optimized_prefix_train16/row34_packed_word.npz'
               if 'row34' in axis else JOINT/'local_train16/common3_group.npz')
    out=HERE/'full_frame_results'/axis/(frame+'.json')
    command=[sys.executable,str(HERE/'full_frame_requests.py'),
             '--capture-dir',str(CAP/axis/frame),'--parameters',str(CAP/'parameters.npz'),
             '--predictor',str(predictor),'--output',str(out)]
    p=subprocess.run(command,capture_output=True,text=True)
    if p.returncode:raise RuntimeError(axis+'/'+frame+': '+p.stderr)
    return axis,frame,json.loads(out.read_text())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--refresh-source-layouts-only',action='store_true',help='reuse already computed full masks/request counts; update explicit static layouts and aggregate')
    args=ap.parse_args()
    jobs=[(p.parent.parent.name,p.parent.name) for p in sorted(CAP.glob('*/*/gates.npz'))]
    assert len(jobs)==16,len(jobs)
    started=time.monotonic();records={}
    if args.refresh_source_layouts_only:
        import numpy as np
        from full_frame_requests import source_layouts
        params=np.load(CAP/'parameters.npz')
        for axis,frame in jobs:
            path=HERE/'full_frame_results'/axis/(frame+'.json')
            r=json.loads(path.read_text())
            shape_only=np.broadcast_to(np.array(False),r['shape'])
            r['source_ports']['Conv1']=source_layouts(shape_only,params['W1'])['Conv1_temporal_word']
            r['source_ports']['Conv2']=source_layouts(shape_only,params['W2'])['Conv2_time_row_cache']
            r['Conv1']['source_NRV64_build_writes_H32_stripes']=r['Conv1']['source_NRV64_build_writes']*3
            path.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
            records.setdefault(axis,[]).append(r)
    else:
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(job,pair) for pair in jobs]
            for f in as_completed(futures):
                axis,frame,r=f.result();records.setdefault(axis,[]).append(r)
                print('DONE',axis,frame,'gate_diff',r['windows']['gate_unpack_differences'],flush=True)
    axes={}
    for axis,rows in sorted(records.items()):
        n=len(rows);mode=rows[0]['execution_mode']
        counts={}
        for op in ('Conv1','Conv2'):
            keys=('time_vector_uses','full_T_vector_uses','staged_vector_uses',
                  'needed_time_vector_uses','active_scalar_terms','needed_scalar_terms',
                  'source_NRV64_full_epoch_reads','source_NRV64_time_major_replay_reads',
                  'source_NRV64_staged_replay_reads',
                  'assembled_source64_header_filtered_time_major_reads',
                  'assembled_source64_header_filtered_needed_time_major_reads')
            counts[op]={k:sum(r[op][k] for r in rows)/n for k in keys}
            counts[op]['coefficient_dtype']=rows[0][op]['coefficient_dtype']
            counts[op]['coefficient_vector_bytes']=rows[0][op]['coefficient_vector_bytes']
            if op=='Conv1':counts[op]['source_NRV64_build_writes_H32_stripes']=sum(r[op]['source_NRV64_build_writes_H32_stripes'] for r in rows)/n
        psn={k:sum(r['PSN_prediction_and_recompute'][k] for r in rows)/n for k in (
            'full_terms','prefix_all','ideal_retained_C_terms','keep_Y_recompute_terms','keep_Y_skip_prefix_only_fallback_terms')}
        c1=counts['Conv1'];c2=counts['Conv2']
        coeff_time=c1['needed_time_vector_uses']+c2['time_vector_uses']
        coeff_T=(c1['full_T_vector_uses'] if mode=='exact' else c1['staged_vector_uses'])+c2['full_T_vector_uses']
        axes[axis]=dict(frames=n,execution_mode=mode,counts_per_frame=counts,PSN_scalar_terms_per_frame=psn,
            paired_FP32_coefficient_vector_uses=dict(time_major_both=coeff_time,full_T_both_or_conditional_C1_two_epochs=coeff_T),
            window_gate_differences=sum(r['windows']['gate_unpack_differences'] for r in rows),
            window_values_per_numeric_field=sum(r['windows']['conv1_FP64_reference_vs_GPU']['values'] for r in rows),
            window_max_errors={k:max(r['windows'][k]['max_abs'] for r in rows) for k in (
                'conv1_FP64_reference_vs_GPU','conv2_FP64_reference_vs_GPU','fixed_BN1_using_captured_raw','BN2_plus_shortcut_using_captured_raw')},
            source_layout_common=rows[0]['source_ports'])
    comparisons={}
    for axis,a in axes.items():
        if a['execution_mode']!='conditional':continue
        exact=axes[axis.removesuffix('_conditional')+'_exact']
        c={}
        for op,k in [('Conv1','needed_time_vector_uses'),('Conv1','needed_scalar_terms'),
                     ('Conv2','time_vector_uses'),('Conv2','full_T_vector_uses'),('Conv2','active_scalar_terms')]:
            c[op+'_'+k+'_ratio']=a['counts_per_frame'][op][k]/exact['counts_per_frame'][op][k]
        for k,v in a['paired_FP32_coefficient_vector_uses'].items():
            c['pair_'+k+'_ratio']=v/exact['paired_FP32_coefficient_vector_uses'][k]
        comparisons[axis]=c
    result=dict(scope='four actually captured FP32 student/mode axes x four complete240x320/T10/C96 frames; whole-column demand and realConv2 support',
        files='full_frame_results/<axis>/<frame>.json',axes=axes,same_student_comparisons=comparisons,
        precision='both captured Conv weightsFP32; H8 vector256bit/four64bit beats; noINT8 execution claim',
        limit='paired coefficient ratios are declared-reuse address/traffic comparisons, not completion time; lane-enable, finite source/state arbitration, FP datapath and packing service remain open',
        all_window_gate_differences=sum(a['window_gate_differences'] for a in axes.values()),
        wall_seconds=time.monotonic()-started)
    (HERE/'full_frame_request_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(comparisons,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
