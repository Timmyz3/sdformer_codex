"""Run existingFP capture through the same bounded model, at most2CPU jobs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
import subprocess
import sys
import time
import argparse

HERE=Path(__file__).resolve().parent;JOINT=HERE.parent


def one(task):
    family,axis,mode,cap,pred,output=task
    if output.exists():
        old=json.loads(output.read_text())
        if old.get('gate_pack_ordered') and old.get('partial_carry_commit_steps')==1:
            return dict(family=family,axis=axis,mode=mode,frame=old['frame'],path=str(output),result=old)
    cmd=['/opt/anaconda3/bin/python3.12',str(HERE/'finite_frame_service.py'),
         '--capture-dir',str(cap),'--parameters',str(JOINT/'full_capture4/capture/parameters.npz'),
         '--predictor',str(pred),'--mode',mode,'--output',str(output)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    p=subprocess.run(cmd,env=env,text=True,capture_output=True)
    if p.returncode:raise RuntimeError(str(cap)+'\n'+p.stdout+p.stderr)
    z=json.loads(output.read_text());print(json.dumps(dict(family=family,mode=mode,frame=z['frame'],steps=z['best_total_steps'],route=z['best_legal_route']),ensure_ascii=False),flush=True)
    return dict(family=family,axis=axis,mode=mode,frame=z['frame'],path=str(output),result=z)


def main():
    out=HERE/'finite_frame_results';out.mkdir(exist_ok=True)
    exe=HERE/'finite_frame_core';cpp=HERE/'finite_frame_core.cpp'
    if not exe.exists() or exe.stat().st_mtime<cpp.stat().st_mtime:
        subprocess.run(['g++','-std=c++17','-O3',str(cpp),'-o',str(exe)],check=True)
    tasks=[]
    for family,base,pred in [('row34','00_optimized_prefix_train16_row34_packed_word',JOINT/'optimized_prefix_train16/row34_packed_word.npz'),
                             ('common3','01_local_train16_common3_group',JOINT/'local_train16/common3_group.npz')]:
        for mode in ('exact','conditional'):
            folder=JOINT/('full_capture4' if mode=='exact' else 'accepted_capture4')/'capture'/(base+'_'+mode)
            frames=sorted(p.parent for p in folder.glob('*/gates.npz'))
            if len(frames)!=4:raise RuntimeError(f'{folder}:expected4frames,got{len(frames)}')
            for cap in frames:
                output=out/(family+'_'+mode+'_'+cap.name+'.json')
                tasks.append((family,base,mode,cap,pred,output))
    started=time.monotonic();rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,t) for t in tasks]):rows.append(f.result())
    summary=dict(scope='Four completeFP32frames per same-studentexact/conditional axis; eight explicit finite schedules per frame. Issue-service quanta,notRTLcycles/PPA.',
                 files=[r['path'] for r in rows],families={},accepted_check_differences=0,wall_seconds=time.monotonic()-started)
    for family in ('row34','common3'):
        by={m:{r['frame']:r['result'] for r in rows if r['family']==family and r['mode']==m} for m in ('exact','conditional')}
        per=[]
        for frame,a in sorted(by['exact'].items()):
            b=by['conditional'][frame];x=a['best_total_steps'];y=b['best_total_steps']
            per.append(dict(frame=frame,ordinary_steps=x,conditional_steps=y,relative_change=y/x-1,
                            ordinary_route=a['best_legal_route'],conditional_route=b['best_legal_route'],
                            Conv1_ordinary=a['schedules'][a['best_legal_route']]['Conv1_BN1_PSN']['steps'],
                            Conv1_conditional=b['schedules'][b['best_legal_route']]['Conv1_BN1_PSN']['steps'],
                            Conv2_ordinary=a['schedules'][a['best_legal_route']]['Conv2_BN2_shortcut']['steps'],
                            Conv2_conditional=b['schedules'][b['best_legal_route']]['Conv2_BN2_shortcut']['steps']))
            summary['accepted_check_differences']+=b['accepted_review']['need_column_differences']+int(b['accepted_review']['accepted_H_T_count_differences'] or 0)
        s0=sum(p['ordinary_steps'] for p in per);s1=sum(p['conditional_steps'] for p in per)
        summary['families'][family]=dict(frames=per,ordinary_steps_mean=s0/4,conditional_steps_mean=s1/4,
                                        ratio_of_sums=s1/s0,relative_change=s1/s0-1,
                                        averaging='Totals overthe same4frames; no unweightedmixture of ratios')
    (HERE/'finite_frame_service_result.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary['families'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
