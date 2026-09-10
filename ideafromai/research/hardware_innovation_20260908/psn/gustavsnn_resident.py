"""Whole-operator compact-code residency / serial W-strip service model.

Both class and time get identical source format, ports and strip choices.
This is a CPU service model, not RTL or an official GustavSNN artifact.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from gustavsnn_reference import ROOT, HERE, identities, specialize_routes, read_torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=6)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--points',nargs='+',default=['8:1','4:1','4:2','4:4','4:8','4:16','2:16'])
    ap.add_argument('--overlap',type=int,default=1)
    ap.add_argument('--reduce',type=int,default=0)
    ap.add_argument('--variants',nargs='+',default=['integer'])
    ap.add_argument('--output',default='gustavsnn_resident_result.json')
    ap.add_argument('--no-build',action='store_true')
    ap.add_argument('--capture-root',type=Path)
    ap.add_argument('--blocks',type=int,nargs='+',default=list(range(6)),help='whole captured blocks to replay')
    ap.add_argument('--weight-override',type=Path,help='NPZ: weight_int8, optional hidden_keep; only override-block')
    ap.add_argument('--override-block',type=int,default=3)
    ap.add_argument('--block-skip',action='store_true',help='C16 shared block mask, compressed W, before source read')
    ap.add_argument('--weight-mode',choices=['dense','block16','row2of4'],default='dense')
    ap.add_argument('--override-F',type=int,help='W residency rows only for override-block; other blocks use points')
    args=ap.parse_args()
    core=HERE/'gustavsnn_resident_core'
    if not args.no_build:
        subprocess.run(['g++','-O3','-std=c++17',str(core.with_suffix('.cpp')),'-o',str(core)],check=True)
    params=read_torch(ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    books=np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    basis=np.load(ROOT/'algorithm/stage2_temporal_codes/signed_basis.npz')
    consumers=json.loads((ROOT/'algorithm/stage2_class_shift/consumers.json').read_text())
    producers=read_torch(ROOT/'algorithm/direct_code_stage2/producers.pt')
    controls=json.loads((HERE/'class_reconstruct_control.json').read_text())['records']
    controls={(r['variant'],r['block']):r for r in controls}
    specs={}
    for variant in args.variants:
      for b in range(6):
        W,routes,identity=identities(b,params,books,basis,consumers)
        current_consumers=consumers
        weight_mask=None
        if args.weight_override and b==args.override_block:
            with np.load(args.weight_override) as z:
                W=np.ascontiguousarray(z['weight_int8'],dtype=np.int8)
                newtau=np.asarray(z['threshold_int32'] if 'threshold_int32' in z else
                          z['tau_int32'] if 'tau_int32' in z else
                          consumers['power2_trained'][f's2b{b}']['tau_int32'],dtype=np.int64)
                if 'weight_mask' in z:weight_mask=np.asarray(z['weight_mask'],dtype=bool)
                if 'hidden_keep' in z:
                    keep=z['hidden_keep']
                    if keep.dtype==bool:
                        W=W[keep];newtau=newtau[:,keep]
                        if weight_mask is not None:weight_mask=weight_mask[keep]
                    elif len(keep)<len(W):
                        W=W[keep.astype(np.int64)];newtau=newtau[:,keep.astype(np.int64)]
                        if weight_mask is not None:weight_mask=weight_mask[keep.astype(np.int64)]
                assert np.maximum(W.astype(np.int64),0).sum(1).max()<16384
                assert np.minimum(W.astype(np.int64),0).sum(1).min()>=-16384
                current_consumers=copy.deepcopy(consumers)
                current_consumers['power2_trained'][f's2b{b}']['tau_int32']=newtau.tolist()
        routes,identity,costs,tau=specialize_routes(variant,b,W,routes,identity,producers,current_consumers)
        guide=controls['rows3' if variant=='rows3' else 'bits3',b]
        cc=dict(costs['packed_class'])
        cc.update(restore_loads=0,restore_adds=0,tail='direct_B')
        if guide['blockwise_choose']=='restore_then_A':
            cc['CSD_program_issues_per_p']=guide['restore_one_read_found']+10
            cc.update(restore_loads=guide['plan']['U_loads'],
                      restore_adds=guide['plan']['cache_writes'],tail='restore_then_A')
        tc=dict(costs['packed_time'],restore_loads=0,restore_adds=0,tail='A_eff')
        specs[variant,b]=(W,routes,{'class':cc,'time':tc},identity,weight_mask)
    jobs=[]
    for variant in args.variants:
        folder=(ROOT/'algorithm/direct_code_integer/deployment/capture10' if variant=='integer'
                else ROOT/'algorithm/direct_code_stage2'/('compiled_'+variant)/'capture')
        if args.capture_root:folder=args.capture_root
        files=[f for f in sorted(folder.rglob('*.npz')) if int(f.stem[-1]) in args.blocks]
        if args.limit:files=files[:args.limit]
        for filename in files:
            for point in args.points:
                P,F=map(int,point.split(':'))
                for route in ('class','time'):jobs.append((variant,filename,P,F,route))
    result=dict(kind='WHOLE_OPERATOR_CPU_SERVICE_MODEL',
        scope='six S2 FC1 + same-student noncausal T10 PSN; source code already produced',
        model='new student, theta preserved in compiled coefficients; not native ep34 arithmetic',
        common=dict(tiles=8,PE_per_tile=8,W_ports_per_tile=2,W_bytes_per_port_per_cycle=1,
                    W_local_bytes_per_tile=8192,source_bank_bytes_per_ID=1024,
                    source_bank_ports='64-bit 1R1W',source_extract_bits_per_ID=128,
                    source_bridge_entries_per_ID=2,NR4_private_W_compaction=True,
                    packets_per_PE=2,state_words_per_PE=56,state_word_bits=15,
                    consumer_cache_words_per_PE=14,consumer_acc_bits=24,
                    gate_bits_per_PE=80,global_gate_descriptor_bytes=32,
                    global_staging_bytes=64,global_bus='one 32-byte transaction / 5 cycles',
                    source_DMA_layout='64-bit words striped across active k-IDs; 256-bit staging dequeue to four distinct banks',
                    output_gather_bits_per_cycle=64,
                    program_replicas=8,program_replica_axis='k-ID, broadcast to eight tiles',
                    program_capacity_instructions_per_replica=85,program_instruction_bits=12,
                    threshold_register_bits_per_tile=240,threshold_select_ports_per_tile=8,
                    weight_mask_select_ports_per_tile=8,
                    row2of4_metadata_bits_per_row=288,block16_metadata_bits_per_row=24),
        schedule='W strip -> source wave -> each output row sequentially completes C and PSN',
        F_meaning='resident W rows, not concurrent output partial states',
        limits=['no full source producer/quantizer/layout conversion', 'no FC2/BN2/shortcut timeline',
                'fixed static strip search, not complete HYTE implementation',
                'bulk DMA/gate transfers use finite-staging throughput formulas, not queue RTL',
                'no prefetch of a complete next source wave into occupied bank',
                'model uses two PE addition paths; PSN does not overlap synaptic execution',
                'optional member reduction needs different arithmetic area/timing',
                'not physically mapped or complete layer RTL'],
        overlap=args.overlap,member_reduce=args.reduce,
        capture_root=str(args.capture_root) if args.capture_root else None,records=[])
    result.update(weight_override=str(args.weight_override) if args.weight_override else None,
                  override_block=args.override_block,block_skip=args.block_skip,
                  weight_mode=args.weight_mode,override_F=args.override_F)
    def run(job):
        variant,filename,P,F,route=job;b=int(filename.stem[-1])
        W,routes,costs,identity,weight_mask=specs[variant,b]
        if b==args.override_block and args.override_F:F=args.override_F
        decode=np.ascontiguousarray(routes['packed_'+route][0],dtype=np.uint8)
        cost=costs[route]
        with np.load(filename) as z:codes=np.ascontiguousarray(z['codes'],dtype=np.uint8)
        N,C=codes.shape;H=len(W);R=len(decode)
        payload=np.asarray([N,C,H,R],dtype=np.int32).tobytes()+W.tobytes()+codes.tobytes()+decode.tobytes()
        start=time.monotonic()
        weight_mode={'dense':0,'block16':1,'row2of4':2}[args.weight_mode]
        if args.block_skip:weight_mode=1
        if args.weight_override and b!=args.override_block:weight_mode=0
        if weight_mode==2:
            assert weight_mask is not None,'row2of4 requires actual fixed training mask'
            mask4=weight_mask.reshape(H,C//4,4)
            assert np.all(mask4.sum(2)==2)
            assert not np.any(W[~weight_mask])
            masks=np.sum(mask4*np.array([1,2,4,8],dtype=np.uint8),axis=2,dtype=np.uint8)
            payload+=masks.tobytes()
        cmd=[str(core),str(P),str(F),str(cost['CSD_program_issues_per_p']),str(cost['restore_loads']),
             str(cost['restore_adds']),str(args.overlap),str(args.reduce),str(weight_mode)]
        p=subprocess.run(cmd,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        record=json.loads(p.stdout)
        # Independent complete-input operation conservation; no schedule replay.
        events=decode[:,codes].sum(axis=(0,1),dtype=np.int64)
        expected=int(events@np.count_nonzero(W,axis=0).astype(np.int64))
        assert record['selected']==expected,(filename,route,record['selected'],expected)
        if not weight_mode:assert record['weight_bytes']==H*C
        if weight_mode==2:assert record['weight_bytes']==H*C//2
        assert record['consumer_output_bits']==N*H*10
        assert record['source_bytes']==N*C*3//8*((H+8*F-1)//(8*F) if record['resident'] else H//8)
        assert record['max_buffers']<=2
        record.update(variant=variant,capture=filename.name,capture_path=str(filename),block=b,P=P,F_cache=F,route=route,
                      shape=[N,C,H],R=R,active_state_words_per_PE=P*R,weight_mode=weight_mode,
                      consumer=cost,elapsed_wall_seconds=time.monotonic()-start)
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,j):j for j in jobs}
        for f in as_completed(futures):
            r=f.result();result['records'].append(r)
            print(f"{len(result['records'])}/{len(jobs)} {r['variant']} {r['capture']} P{r['P']} F{r['F_cache']} {r['route']} {r['elapsed']}",flush=True)
    result['records'].sort(key=lambda x:(x['variant'],x['capture'],x['P'],x['F_cache'],x['route']))
    target=HERE/args.output;target.write_text(json.dumps(result,indent=2)+'\n')
    print(target,flush=True)


if __name__=='__main__':main()
