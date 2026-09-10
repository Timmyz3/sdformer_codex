"""Actual whole-frame U requests for the QDQ captures; not integer-bridge AEE.

Shared/tail separate, K-major ascending byte addresses in each stage; one
current physical word only. No free cross-P4 cache. Python set-based selected
P4 checks are independent of the C++ register walk.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np

HERE=Path(__file__).resolve().parent


def unpack(data,name):
    shape=tuple(int(x) for x in data[name+'_shape'])
    return np.unpackbits(data[name+'_bits'],bitorder='little',count=int(np.prod(shape))).reshape(shape).astype(bool)


def source_words(gates):
    t,c,ny,nx=gates.shape
    pixel=np.zeros((c,ny,nx),np.uint16)
    for s in range(t):pixel|=gates[s].astype(np.uint16)<<s
    padded=np.pad(pixel,((0,0),(1,1),(1,1)))
    words=np.empty((ny,nx//4,c,3,3),np.uint64)
    for kh in range(3):
        for kw in range(3):
            values=padded[:,kh:kh+ny,kw:kw+nx].reshape(c,ny,nx//4,4)
            word=sum(values[...,p].astype(np.uint64)<<(10*p) for p in range(4))
            words[:,:,:,kh,kw]=word.transpose(1,2,0)
    return np.ascontiguousarray(words.reshape(-1,c*9))


def pack_need(values):
    # G,S,P,J2 -> G,J2, bit=10*p+s; no OR across P4/consumer groups.
    return sum(values[:,s,p].astype(np.uint64)<<(10*p+s) for p in range(4) for s in range(10))


def prepare_need(data,params,words):
    tail=unpack(data,'need_Z')
    empty=unpack(data,'source_empty')
    T,ny,nx=empty.shape;G=ny*(nx//4)
    empty=empty.reshape(T,ny,nx//4,4).transpose(1,2,0,3).reshape(G,T,4)
    source_nonempty=np.bitwise_or.reduce(words,axis=1)
    empty_mask=sum((~empty[:,s,p]).astype(np.uint64)<<(10*p+s) for p in range(4) for s in range(T))
    empty_diff=int(np.count_nonzero(source_nonempty!=empty_mask))
    a=params['a']!=0;v=params['v']!=0;shared=int(params['shared_rank']);R=v.shape[0]
    # Captured J2 need can serve per-coefficient enables only because these
    # actual two columns have exactly the same consumer support.
    pair_equal=bool(np.array_equal(v[::2],v[1::2]))
    if not pair_equal:raise ValueError('This capture needs per-latent (not only J2) demand for exact lane cancellation.')
    accepted=unpack(data,'accepted_gate')
    grouped=accepted.reshape(T,96,ny,nx//4,4).transpose(2,3,0,4,1).reshape(G,T,4,96)
    picks=np.unique(np.linspace(0,G-1,64,dtype=int))
    need_y=np.einsum('gtph,ts->gsph',(~grouped[picks]).astype(np.int32),a.astype(np.int32))>0
    need_y&=~empty[picks,...,None]
    expected=np.einsum('gsph,jh->gspj',need_y.astype(np.int32),v[shared::2].astype(np.int32))>0
    need_diff=int(np.count_nonzero(expected!=tail[picks]))
    if empty_diff or need_diff:raise AssertionError((empty_diff,need_diff))
    prefix=np.broadcast_to((~empty)[...,None],(G,T,4,shared//2)).copy()
    prefix&=a.any(0)[None,:,None,None]
    prefix&=v[:shared:2].any(1)[None,None,None,:]
    masks=np.ascontiguousarray(np.concatenate((pack_need(prefix),pack_need(tail)),axis=1))
    return masks,dict(source_empty_full_P4_differences=empty_diff,
        sampled_need_Z_differences=need_diff,checked_native_P4=picks.tolist(),
        J2_individual_columns_have_identical_V_support=pair_equal,
        accepted_gate_count=int(accepted.sum()),source_empty_TP=int(empty.sum()))


def independent_group(words,need,uq,shared,g):
    out={}
    for phase,(first,last) in [('shared',(0,shared)),('tail',(shared,uq.shape[1]))]:
        nr=last-first;base=0 if first==0 else len(uq)*shared
        entries={b:{} for b in (2,8,32)}
        for k in np.flatnonzero(words[g]):
            src=int(words[g,k])
            for r in range(first,last):
                if uq[k,r]==0:continue
                events=src&int(need[g,r//2])
                if not events:continue
                byte=base+int(k)*nr+r-first
                for b in entries:
                    record=entries[b].setdefault(byte//b,[0,0,{}])
                    record[0]+=1;record[1]+=events.bit_count()
                    record[2][int(k)]=record[2].get(int(k),0)|events
        out[phase]={str(b):dict(requests=len(e),useful_coefficients=sum(x[0] for x in e.values()),
            scalar_updates=sum(x[1] for x in e.values()),
            source_TP_word_updates=sum(mask.bit_count() for x in e.values() for mask in x[2].values())) for b,e in entries.items()}
    return out


def summarize(result):
    for phase in ('shared','tail','total'):
        for b,record in result[phase]['words'].items():
            n=record['requests']
            record['mean_useful_coefficients_per_word']=record['useful_coefficients']/max(n,1)
            record['mean_scalar_updates_per_word']=record['scalar_updates']/max(n,1)
            record['mean_source_TP_updates_per_word']=record['source_TP_word_updates']/max(n,1)
            record['coefficient_payload_utilization']=record['useful_coefficients']/max(n*int(b),1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,default=HERE.parent/'network_recovery64_diverse10'/'capture')
    parser.add_argument('--frames',type=int,nargs='+',default=[0])
    parser.add_argument('--output',type=Path,default=HERE/'physical_u_requests_frame0.json')
    args=parser.parse_args();started=time.monotonic()
    exe=HERE/'u_request_core';cpp=HERE/'u_request_core.cpp'
    subprocess.run(['g++','-O3','-std=c++17',str(cpp),'-o',str(exe)],check=True)
    axes=sorted(p for p in args.capture.iterdir() if p.is_dir() and 'u8_vq5' in p.name)
    result=dict(scope='Whole QDQ/dequantized-FP32 capture with U8 and VQ5 weights. Not the later Aq14 integer deployment, not cycles, and not an AEE-matched winner selection.',
      contract=dict(P=4,T=10,K=864,shared_rank=32,
        address='Uq8 shared at byte k*32+r. Tail at 864*32 + k*(R-32)+(r-32). Each stage separately K/r ascending. Physical address=floor(byte/word_bytes).',
        current_word='One current 2/8/32-byte word per alternative. Within a P4/stage, a 32B tail word may cover two adjacent k rows and is retained. Explicitly invalidate at P4/stage boundary; no extra cross-P4 weight copy.',
        controls='Both full and conditional get two-stage storage/traversal, complete T/P4 coefficient reuse, actual source zero, actual U8 zero, and actual per-lane consumer enables. Full can omit the predictor; no predictor cost is inferred here.',
        J2='2-byte logical request unit only, not an asserted physical port.',
        full_scan_boundary='For full mode, these separate-stage W reads require the corresponding source traversals. A one-pass K loop interleaving shared/tail can save a local source pass but loses some 32B tail carry with one word register; its best combined service is outside this request census.',
        metadata='Exact per-coefficient U-zero pruning before mem_req needs a static 64-bit nonzero mask per k (6912B for this padded implementation); reported mask reads are additional logical 64-bit accesses. P4 source K directory is 108B payload/112B at64-bit alignment and must be generated/read. No free metadata port or cold load is claimed.',
        source='Complete actual sn1 bitmap; stride1/pad1/dilation1 3x3, k=((c*3)+kh)*3+kw. Same full source cache may serve both stages; K-word examinations are local reads, not off-chip bytes.',
        excluded='Predictor/V/A/Conv2/BN2/shortcut, fill/port timing/latency, bank arbiters and SRAM area. Scalar updates are useful conditional adds, not cycles.'),frames={},complete=False)
    with tempfile.TemporaryDirectory(prefix='_u_request_work_',dir=HERE) as temporary:
        tmp=Path(temporary)
        for frame_index in args.frames:
            first=next(axes[0].glob(f'{frame_index:03d}_*/gates.npz'))
            with np.load(first) as data:
                first_source=data['source_gate_bits'].copy();source=unpack(data,'source_gate');frame_name=str(data['frame_name'])
            words=source_words(source);words.tofile(tmp/'source.bin');del source
            frame_result={}
            for axis in axes:
                capture=next(axis.glob(f'{frame_index:03d}_*/gates.npz'))
                with np.load(axis/'student_parameters.npz') as z:params={k:z[k].copy() for k in z.files}
                with np.load(capture) as data:
                    if not np.array_equal(data['source_gate_bits'],first_source):raise AssertionError('Sources differ across the named axes; generate this axis separately.')
                    need,checks=prepare_need(data,params,words)
                    conditional=bool(data['conditional'])
                uq=params['u_int8'];R=uq.shape[1];shared=int(params['shared_rank'])
                need.tofile(tmp/'need.bin');uq.tofile(tmp/'weights.bin')
                subprocess.run([str(exe),str(tmp/'source.bin'),str(tmp/'need.bin'),str(tmp/'weights.bin'),str(len(words)),str(words.shape[1]),str(R),str(shared),str(tmp/'count.json')],check=True)
                record=json.loads((tmp/'count.json').read_text())
                for small in record['selected_P4']:
                    ref=independent_group(words,need,uq,shared,small['G'])
                    for phase in ('shared','tail'):
                        for b,r in ref[phase].items():
                            for key,value in r.items():
                                if small[phase]['words'][b][key]!=value:raise AssertionError((axis.name,small['G'],phase,b,key))
                summarize(record)
                record.update(capture=str(capture),parameters=str(axis/'student_parameters.npz'),
                    model=str(params['structure']),conditional=conditional,R=R,checks=checks,
                    independent_full_K_selected_P4=[x['G'] for x in record['selected_P4']],
                    independent_address_set_differences=0,U8_nonzero_coefficients=int(np.count_nonzero(uq)),
                    source_theta=float(params['theta_source']),output_theta=float(params['theta_output']))
                frame_result[axis.name]=record
                print(frame_name,axis.name,json.dumps({b:record['total']['words'][b]['requests'] for b in ('2','8','32')}),flush=True)
            result['frames'][frame_name]=frame_result
    comparisons={}
    for frame,axes_result in result['frames'].items():
        by_model={}
        for name,record in axes_result.items():by_model.setdefault(record['model'],{})['conditional' if record['conditional'] else 'full']=record
        comparisons[frame]={}
        for model,pair in by_model.items():
            comparisons[frame][model]={b:dict(conditional_over_full=pair['conditional']['total']['words'][b]['requests']/pair['full']['total']['words'][b]['requests']) for b in ('2','8','32')}
        shared=by_model['shared56']['conditional'];private=by_model['shared32_private2']['conditional']
        comparisons[frame]['private56_conditional_vs_shared56_conditional']={b:private['total']['words'][b]['requests']/shared['total']['words'][b]['requests'] for b in ('2','8','32')}
    result.update(comparisons=comparisons,complete=True,wall_seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
