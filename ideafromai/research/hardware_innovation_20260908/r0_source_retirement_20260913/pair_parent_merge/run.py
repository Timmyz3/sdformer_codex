#!/usr/bin/env python3
"""One fixed pair-parent forwarding/merge interface versus the exact C4-hold pair control."""
from pathlib import Path
import argparse
import concurrent.futures
import importlib.util
import json
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent.parent/'r0_execution_trials_20260913'
SELECTOR=HERE.parent/'selector'
FIELDS=('cycles','source_words','weight_words','psum_reads','psum_writes','sum_issues','update_issues','pattern_copies','merge_issues','source_stalls','weight_stalls','output_stalls')

def build():
    with (HERE/'build.log').open('w') as log:
        for cmd in (["verilator","--cc","--exe","-Wall","--top-module","pair_parent_merge","--Mdir",str(HERE/'obj_dir'),str(HERE/'pair_parent_merge.sv'),str(HERE/'tb.cpp')],
                    ["make","-C",str(HERE/'obj_dir'),"-f","Vpair_parent_merge.mk","-j2"]):
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)

def support_fixture():
    # Reuse only the independent convolution/hex writer, directed into this
    # owned directory. The reference never computes a support sum or schedule.
    spec=importlib.util.spec_from_file_location('native_reference',OLD/'native_sparse/run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.HERE=HERE
    src=np.zeros((10,96,4,4),dtype=np.uint8)
    for t in range(10):
        for c in range(96):
            for y in range(4):
                for x in range(4):
                    time_code=t%5 if (c//4)%3==0 else t
                    code=(time_code+y*4+x+c//4)%16
                    src[t,c,y,x]=(code>>(c%4))&1
    # All four terms of output lane 0 are -32768; lane 1 is +32767.
    # This exercises the exact signed18 extremes -131072 and +131068.
    n,c,ky,kx=np.indices((96,96,3,3))
    sign=np.where(n%4==0,True,np.where(n%4==1,False,(c+ky+kx)%2==0))
    weight=np.where(sign,-32768,32767).astype(np.int32)
    return module.fixture('all_supports_extreme',src,weight,np.ones((12,24),dtype=bool),
        {'kind':'all sixteen support codes, repeated support times, mixed signs, and signed18 four-term extrema; full C96/N96/T10'})

def fixtures():
    out=[]
    for arm in ('dense','block_magnitude25','cin_magnitude25','cin_fullcost25','mixed_retirement25'):
        for tile in range(8):
            p=SELECTOR/'fixtures'/f'{arm}_tile{tile}'
            assert (p/'gold.hex').exists(),p
            out.append((p.name,p,'formal',arm,tile))
    for name in ('zero','one','corners','all_masked','padding_poison'):
        p=OLD/'native_sparse/fixtures'/name
        out.append((name,p,'functional',name,-1))
    p=support_fixture();out.append((p.name,p,'functional',p.name,-1))
    return out

def execute(item):
    fixture,mode,stall=item
    name,path,scope,arm,tile=fixture
    p=subprocess.run([str(HERE/'obj_dir/Vpair_parent_merge'),str(path),str(mode),str(stall)],check=True,text=True,capture_output=True)
    rows=[]
    for line in p.stdout.splitlines():
        r=json.loads(line);r.update(fixture=name,fixture_path=str(path),scope=scope,arm=arm,tile=tile)
        r['control_cycles']=r['cycles']-sum(r[k] for k in ('source_words','weight_words','psum_reads','psum_writes','sum_issues','pattern_copies','source_stalls','weight_stalls','output_stalls'))-480
        assert r['control_cycles']>=0
        assert r['psum_reads']==r['psum_writes']==r['update_issues']+480
        rows.append(r)
    assert len(rows)==2
    print('RTL_PASS',name,mode,stall,flush=True)
    return rows

def summarize(rows):
    index={(r['fixture'],r['mode'],r['stall'],r['command']):r for r in rows}
    reference=json.loads((HERE.parent/'rtl/results.json').read_text())
    ref_index={(r['fixture'],r['mode'],r['stall'],r['command']):r for r in reference}
    for r in rows:
        peer=index[(r['fixture'],5,r['stall'],r['command'])]
        for field in ('source_words','weight_words','outputs'):
            assert r[field]==peer[field],(r['fixture'],field)
        if r['mode']==5:
            ref=ref_index[(r['fixture'],5,r['stall'],r['command'])]
            for field in FIELDS:
                assert r[field]==ref.get(field,0),(r['fixture'],field,r[field],ref.get(field))
        assert r['pattern_copies']==0
        if r['mode']==6:
            assert peer['update_issues']-r['update_issues']==r['merge_issues']
            assert r['sum_issues']-r['merge_issues']==peer['sum_issues']
        unstalled=index[(r['fixture'],r['mode'],0,r['command'])]
        assert r['cycles']==unstalled['cycles']+r['source_stalls']+r['weight_stalls']+r['output_stalls']
    summary={}
    for arm in ('dense','block_magnitude25','cin_magnitude25','cin_fullcost25','mixed_retirement25'):
        summary[arm]={}
        for mode in (5,6):
            summary[arm][str(mode)]={}
            for stall in (0,1):
                rs=[r for r in rows if r['scope']=='formal' and r['arm']==arm and r['mode']==mode and r['stall']==stall and r['command']==0]
                s={f:sum(r[f] for r in rs) for f in FIELDS+('control_cycles',)}
                s.update(tiles=len(rs),cycles_with_fresh_source_origin=s['cycles']+1537*len(rs))
                summary[arm][str(mode)][str(stall)]=s
    result=dict(runs=len(rows),checked_outputs=sum(r['outputs'] for r in rows),mode5_reference_exact=True,
                source_weight_requests_identical=True,pair_parent_builds_identical=True,
                formal_arms=summary,scope='Full native C96/N96/T10 4x4-to-2x2 Q16 linear leaf. Same hardware, same mask, same precision per arm. Not full network/EDA.')
    (HERE/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-build',action='store_true');args=parser.parse_args()
    if not args.no_build:build()
    items=[(f,m,s) for f in fixtures() for m in (5,6) for s in (0,1)]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for group in pool.map(execute,items):
            rows.extend(group)
            (HERE/'results.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    (HERE/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    summarize(rows)
