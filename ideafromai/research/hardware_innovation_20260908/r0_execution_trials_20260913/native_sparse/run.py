#!/usr/bin/env python3
"""Native fixtures, independent integer convolution reference, and Verilator runs.

No source gather, endpoint, mode choice, partial sum, or conflict decision is
fed to the RTL. Fixture source words are native (c,y,x) with ten temporal bits.
"""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def dump(path, values):
    path.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(values).reshape(-1)))

def fixture(name, source, weight, live, metadata=None, origin=(0,0)):
    source=np.asarray(source)
    weight=np.asarray(weight)
    live=np.asarray(live).astype(bool)
    assert source.shape==(10,96,4,4) and np.all((source==0)|(source==1))
    assert weight.shape==(96,96,3,3) and weight.min()>=-32768 and weight.max()<=32767
    assert live.shape==(12,24)
    valid=((np.arange(4)+origin[0]>=0)&(np.arange(4)+origin[0]<240))[:,None] & ((np.arange(4)+origin[1]>=0)&(np.arange(4)+origin[1]<320))[None,:]
    functional_source=source*valid[None,None,:,:]
    effective=weight.astype(np.int64)*np.repeat(np.repeat(live,8,axis=0),4,axis=1)[:,:,None,None]
    # Output reference is the mathematical 3x3 valid convolution, not the
    # scheduling/pairing implementation. Full K=864, all N=96, all T=10.
    out=np.zeros((10,96,2,2),dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            out += np.einsum('tcyx,nc->tnyx',functional_source[:,:,ky:ky+2,kx:kx+2].astype(np.int64),effective[:,:,ky,kx])
    assert np.abs(out).max()<2**31
    path=HERE/'fixtures'/name;path.mkdir(parents=True,exist_ok=True)
    packed=np.sum(source.astype(np.uint16)*(1<<np.arange(10,dtype=np.uint16))[:,None,None,None],axis=0,dtype=np.uint16)
    dump(path/'source.hex',packed)
    # row=og*864+c*9+tap; lane selects output n=og*8+lane.
    dump(path/'weight.hex',weight.reshape(12,8,864).transpose(0,2,1))
    dump(path/'mask.hex',live)
    dump(path/'origin.hex',origin)
    dump(path/'gold.hex',out.reshape(10,12,8,4).transpose(1,3,0,2))
    info={'shape':[10,96,4,4], 'output_shape':[10,96,2,2],
          'active_source_bits':int(source.sum()),'nonzero_source_words':int(np.count_nonzero(packed)),
          'live_blocks':int(live.sum()),'masked_blocks':int((~live).sum()),
          'input_origin_yx':list(map(int,origin)),'valid_native_pixels':int(valid.sum()),
          'max_abs_integer_output':int(np.abs(out).max()),'metadata':metadata or {}}
    (path/'metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    return path

def build():
    commands=[['verilator','--cc','--exe','--top-module','native_sparse','--Mdir','obj_dir',
               '-Wall','native_sparse.sv','tb.cpp'],
              ['make','-C','obj_dir','-f','Vnative_sparse.mk','-j4']]
    log=[]
    for cmd in commands:
        p=subprocess.run(cmd,cwd=HERE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log.append(p.stdout)
        (HERE/'build.log').write_text('\n'.join(log))
        if p.returncode:raise RuntimeError(p.stdout)

def execute(names, output):
    records=[]
    for name in names:
        for mode in range(3):
            for stall in range(2):
                p=subprocess.run([str(HERE/'obj_dir/Vnative_sparse'),str(HERE/'fixtures'/name),str(mode),str(stall)],
                    text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                if p.returncode:raise RuntimeError(f'{name} mode={mode} stall={stall}: {p.stdout}\n{p.stderr}')
                for line in p.stdout.splitlines():
                    record=json.loads(line);record['fixture']=name
                    record['control_cycles']=record['cycles']-(record['source_words']+record['weight_words']+
                        record['psum_reads']+record['psum_writes']+record['sum_issues']+480+
                        record['source_stalls']+record['weight_stalls']+record['output_stalls'])
                    assert record['psum_reads']==record['update_issues']+480
                    assert record['psum_writes']==record['update_issues']+480
                    assert record['control_cycles']>=0
                    records.append(record)
    (HERE/output).write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'runs':len(records),'checked_outputs':sum(r['outputs'] for r in records),'result':str(HERE/output)}))

def controls():
    rng=np.random.default_rng(913)
    w=rng.integers(-32768,32768,size=(96,96,3,3),dtype=np.int32)
    live=np.ones((12,24),dtype=bool)
    source=np.zeros((10,96,4,4),dtype=np.uint8)
    fixture('zero',source,w,live,{'kind':'full-size functional control'})
    fixture('one',np.ones_like(source),w,live,{'kind':'full-size signed accumulation control'})
    fixture('all_masked',np.ones_like(source),w,np.zeros_like(live),{'kind':'all consumers statically absent'})
    fixture('padding_poison',np.ones_like(source),w,live,{'kind':'out-of-image inputs deliberately nonzero; DUT must suppress them'},origin=(-1,-1))
    # Alternating temporal pair states, four native corner phases, and final
    # channel/output groups stress indexing and all 00/01/10/11 product cases.
    source[:,0,0,0]=np.arange(10)%2
    source[:,1,0,0]=1
    source[9,95,3,3]=1
    source[0,94,0,3]=1
    source[5,63,3,0]=1
    source[:,48,1,1]=1
    live[:,::4]=False
    live[0,0]=True;live[11,23]=True
    fixture('corners',source,w,live,{'kind':'native boundary and physical mask control'})
    return ['zero','one','corners','all_masked','padding_poison']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-build',action='store_true')
    parser.add_argument('--fixtures',nargs='*');parser.add_argument('--output',default='control_results.json')
    args=parser.parse_args()
    if not args.no_build:build()
    execute(args.fixtures or controls(),args.output)
