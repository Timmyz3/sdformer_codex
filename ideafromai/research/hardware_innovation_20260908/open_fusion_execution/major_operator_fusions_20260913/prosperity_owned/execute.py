"""Full official outer PPU + held-out source representation experiment.

No training, EDA or production writes. Python 3.12; CPU only.
The reused fast relation callback is checked against the official callback.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, time, contextlib, io, copy
from collections import Counter, OrderedDict
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/zhumd/work/sdformer_codex')
HW = ROOT/'SDformer/hw_autoresearch_nts07'
sys.path.insert(0,str(ROOT/'ideafromai/research/complete_transfer_20260907'))
import c1_full_layer as prior
from run_prosperity_official_probe import load_official_api, run_official_fc

CAP = HW/'results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831'
MODULE = 'sttmultires_unet.resblocks.0.conv1.0'

def capture(sample):
    records=[json.loads(s) for s in (CAP/'unified_ordered_records.jsonl').read_text().splitlines()]
    r=next(r for r in records if r.get('cohort')=='c1' and r.get('category')=='c1_conv3x3' and r.get('global_sample_id')==sample and r.get('name')==MODULE)
    p=r['payload'];raw=(CAP/p['support_sign']).read_bytes();nb=p['positive_plane_bytes']
    assert not any(raw[nb:])
    a=np.unpackbits(np.frombuffer(raw[:nb],np.uint8),bitorder='little').reshape(10,768,15,20).astype(bool)
    assert int(a.sum())==r['input']['active']
    window=np.lib.stride_tricks.sliding_window_view(np.pad(a,((0,0),(0,0),(1,1),(1,1))),(3,3),axis=(2,3))
    matrix=window.transpose(0,2,3,1,4,5).reshape(3000,6912)
    return a,matrix

def channel_groups(a):
    """Single calibration sample; greedy average Jaccard, no holdout search."""
    x=a.transpose(1,0,2,3).reshape(768,-1).astype(np.float32)
    overlap=x@x.T;nnz=x.sum(1)
    sim=overlap/np.maximum(1,nnz[:,None]+nnz[None,:]-overlap)
    available=np.ones(768,bool);out=[]
    for group in range(48):
        anchor=int(np.argmax(np.where(available,nnz,-1)))
        members=[anchor];available[anchor]=False;score=sim[anchor].copy()
        for _ in range(15):
            nxt=int(np.argmax(np.where(available,score,-1)))
            members.append(nxt);available[nxt]=False;score+=sim[nxt]
        out.extend(members)
    assert sorted(out)==list(range(768))
    return np.asarray(out,np.int32)

def source_code(a,channels,extended):
    """LoAS-style T10 words; optional exact modal word + XOR exceptions.

    Two-bit mode and 16-bit channel bitmap, 10-bit temporal words, packet
    aligned to two bytes, per spatial/group 32-bit address directory. Raw
    fallback and the same directory/cost are available to both controls.
    """
    words=sum(a[t].astype(np.uint16)<<t for t in range(10)).reshape(768,300).T[:,channels]
    packets=[];offsets=[];modes=Counter();decodes=Counter();total=300*48*4
    for p in range(300):
        for g in range(48):
            w=words[p,g*16:(g+1)*16]
            mask=sum((int(v)!=0)<<i for i,v in enumerate(w))
            opts=[(2+160,'raw',None),(2+16+10*int(np.count_nonzero(w)),'silent',mask)]
            if extended:
                values,counts=np.unique(w,return_counts=True)
                base=int(values[np.argmax(counts)])
                ex=w^base;em=sum((int(v)!=0)<<i for i,v in enumerate(ex))
                opts.append((2+10+16+10*int(np.count_nonzero(ex)),'modal_xor',(base,em,ex)))
            bits,mode,extra=min(opts,key=lambda v:v[0]);value={'raw':0,'silent':1,'modal_xor':2}[mode];shift=2
            def emit(v,b):
                nonlocal value,shift
                value|=int(v)<<shift;shift+=b
            if mode=='raw':
                for v in w:emit(v,10)
            elif mode=='silent':
                emit(extra,16)
                for v in w:
                    if v:emit(v,10)
            else:
                base,em,ex=extra;emit(base,10);emit(em,16)
                for v in ex:
                    if v:emit(v,10)
            assert shift==bits
            b=value.to_bytes(2*((bits+15)//16),'little')
            # Independent decoder, including nonzero nonunit theta-free codes.
            v=int.from_bytes(b,'little');code=v&3;v>>=2;decoded=np.zeros(16,np.uint16)
            if code==0:
                for i in range(16):decoded[i]=v&1023;v>>=10
            else:
                base=0
                if code==2:base=v&1023;v>>=10
                bm=v&65535;v>>=16;decoded[:]=base
                for i in range(16):
                    if bm&(1<<i):decoded[i]^=v&1023;v>>=10
            assert np.array_equal(decoded,w)
            offsets.append((total,len(b)));total+=len(b);packets.append(b);modes[mode]+=1
            decodes[mode]+=1
    return offsets,dict(encoded_bytes=total,directory_bytes=57600,payload_bytes=total-57600,
        decoded_source_bits=int(a.size),packet_modes=dict(modes),packet_count=len(packets),
        exact_decoded_words=len(packets)*16,raw_source_bits_packed_bytes=a.size//8,
        compression_vs_same_directory_raw=1-total/(57600+22*14400)), packets

def producer_service(offsets,extended,packets):
    """One 8KiB LRU, 32B transaction, exact im2col source addresses.

    N loop deliberately follows official m,n,k order; all layouts have the
    same source cache and one 16-lane T10 decoder. Padding synthesizes zero.
    There is no hidden full im2col materialization. Two TCAM banks receive
    decoded rows. Source packets are independent of convolution taps.
    """
    cache=OrderedDict();stats=Counter();tile_cycles=[]
    def load(addr,size):
        misses=0
        for line in range(addr//32,(addr+size-1)//32+1):
            stats['source_line_requests']+=1
            if line in cache:cache.move_to_end(line)
            else:
                misses+=1;stats['source_line_misses']+=1;cache[line]=None
                if len(cache)>256:cache.popitem(last=False)
        return misses
    for m in range(0,3000,256):
        pos=sorted(set(i//10 for i in range(m,min(m+256,3000))))
        for n in range(6):
            for tap in range(9):
                ky,kx=divmod(tap,3)
                for g in range(48):
                    cyc=0
                    for p in pos:
                        y,x=divmod(p,20);sy=y+ky-1;sx=x+kx-1
                        stats['producer_address_generations']+=1
                        if not (0<=sy<15 and 0<=sx<20):continue
                        index=(sy*20+sx)*48+g;addr,size=offsets[index]
                        misses=load(index*4,4)+load(addr,size)
                        # Directory address, data request and decoding are
                        # charged even on hit; miss service 32B/128B per cycle
                        # conservatively rounded to one cycle each.
                        extra=int(extended and (packets[index][0]&3)==2)
                        cyc+=3+misses+extra;stats['packet_decodes']+=1
                        stats['extra_XOR_decode_cycles']+=extra
                    tile_cycles.append(cyc)
    stats['source_bytes_fetched']=stats['source_line_misses']*32
    stats['producer_nonoverlapped_cycles']=sum(tile_cycles)
    return np.asarray(tile_cycles,np.int64),dict(stats)

def pipeline(source,detector,compute):
    """Three stages with two source/TCAM slots and two metadata slots."""
    s=d=c=0;ds=[];cs=[]
    for i,(ss,dd,cc) in enumerate(zip(source,detector,compute)):
        s=max(s,ds[i-2] if i>=2 else 0)+int(ss)
        d=max(d,s,cs[i-2] if i>=2 else 0)+int(dd)
        c=max(c,d)+int(cc);ds.append(d);cs.append(c)
    return c

def main():
    assert sys.version_info[:2]==(3,12)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    Accelerator,FC,Simulator,create_network=load_official_api()
    official_relation=Simulator.run_fc.__globals__['find_product_sparsity']
    cal,_=capture(0);order=channel_groups(cal)
    (HERE/'calibrated_channels.json').write_text(json.dumps(order.tolist())+'\n')
    permutations={'channel_major':np.arange(6912),
        'tap_major':np.arange(6912).reshape(768,9).T.reshape(-1),
        'calibrated_tap_major':np.arange(6912).reshape(768,9)[order].T.reshape(-1)}
    report=dict(scope='Complete captured ep34 Conv1; all M3000 K6912 N768. Official outer cycle model plus separate charged producer model; not physical PPU.',
        calibration_samples=[0],heldout_samples=[1,9],permutation_objective='Greedy average source spike Jaccard, 48 groups of16 channels; no heldout selection.',
        official_tile=[256,16,128],official_mem_if_bits=1024,official_weight_output_bits=8,
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),samples={},
        coefficient_permutation=dict(trained_FP32_bytes=6912*768*4,offline_read_plus_write_bytes=2*6912*768*4,
            artifact_INT8_bytes=6912*768,channel_directory_bytes=768*2,online_weight_permutation=False))
    for sample in (1,9):
        a,matrix=capture(sample);out={}
        for name,perm in permutations.items():
            mat=np.ascontiguousarray(matrix[:,perm]);relation=prior.Relation()
            ordered=mat.reshape(10,300,6912).transpose(1,0,2).reshape(3000,6912)
            checks=0
            for m,k in ((0,0),(256,288),(2816,6896),(0,16)):
                tile=torch.from_numpy(ordered[m:min(m+256,3000),k:k+16])
                x,y=official_relation(tile);u,v=relation(tile)
                assert torch.equal(x,u) and torch.equal(y,v);checks+=1
            acc=Accelerator('Prosperity',128,32,256,16,product_sparsity=True,issue_type=2,mem_if_width=1024)
            got,stdout,elapsed=prior.run(FC,Simulator,acc,mat,10,300,768,relation)
            ref,counts,residency=prior.reference(ordered,acc,True,relation)
            assert got==ref
            row=dict(official_outer_counts=got,official_stdout=stdout,relation_counts_one_N_slice=counts,
                original_official_relation_checks=checks,host_seconds_not_cycles=elapsed,residency=residency)
            baseline_cycles=got['total_cycles'] if name=='channel_major' else out['channel_major']['official_outer_counts']['total_cycles']
            row['cycle_delta_vs_original_pct']=(got['total_cycles']/baseline_cycles-1)*100
            row['official_counter_bytes']={direction:{key:value//8 for key,value in got[direction].items()}
                                           for direction in ('reads','writes')}
            # Materialize full execution plan once for actual trained W test.
            parents=[];residuals=[];orders=[];tile_compute=[];tile_pre=[]
            for m in range(0,3000,256):
                p_local=[];r_local=[];o_local=[];c_local=[];pre_local=[]
                for k in range(0,6912,16):
                    tile=ordered[m:min(m+256,3000),k:k+16]
                    residual,p,c=relation.entry(tile)
                    masks=(residual.numpy().astype(np.uint16)*(1<<np.arange(16,dtype=np.uint16))).sum(1,dtype=np.uint16)
                    p_local.append(p.numpy().astype(np.int16));r_local.append(masks)
                    o_local.append(np.lexsort((np.arange(len(tile)),tile.sum(1))).astype(np.uint16))
                    c_local.append(c['residual_nnz']+c['equal_rows']);pre_local.append(c['preprocess'])
                parents.append(np.stack(p_local));residuals.append(np.stack(r_local));orders.append(np.stack(o_local))
                tile_compute.extend(c_local*6);tile_pre.extend(pre_local*6)
            compute=np.asarray(tile_compute,np.int64);pre=np.asarray(tile_pre,np.int64)
            row['finite_stage_model']=dict(compute_cycles=int(compute.sum()),preprocess_cycles=int(pre.sum()),
                paired_tile_pipeline_cycles=int(np.maximum(compute,pre).sum()+pre[0]))
            if name!='channel_major':
                row['source_encodings']={}
                channels=np.arange(768) if name=='tap_major' else order
                for extended in (False,True):
                    offsets,codec,packets=source_code(a,channels,extended)
                    pc,service=producer_service(offsets,extended,packets)
                    assert len(pc)==len(compute)
                    codec.update(service)
                    codec['full_operator_pipeline_cycles']=pipeline(pc,pre,compute)
                    # Strong shared encoder: sixteen temporal words available,
                    # silent-mask generation and packing, one vector cycle
                    # each. Extension gets a16-entry10-bit match table but
                    # still pays16 queries and1 selection for every packet.
                    codec['encoder_cycles']=14400*(2+(17 if extended else 0))
                    codec['encoder_extra_TCAM_ternary_cells']=160 if extended else 0
                    codec['encoder_extra_count_bits']=16*5 if extended else 0
                    codec['encoder_input_output_bytes']=a.size//8+codec['encoded_bytes']
                    codec['encoder_creation_128B_service_cycles']=(codec['encoder_input_output_bytes']+127)//128
                    codec['cold_encoder_plus_complete_operator_cycles']=codec['full_operator_pipeline_cycles']+codec['encoder_cycles']+codec['encoder_creation_128B_service_cycles']
                    weight_bits=got['writes']['g_wgt']
                    codec['common_weight_refill_bytes']=weight_bits//8
                    codec['common_weight_refill_1024bit_service_cycles']=(weight_bits+1023)//1024
                    codec['cold_with_serial_weight_refill_upper_cycles']=codec['cold_encoder_plus_complete_operator_cycles']+codec['common_weight_refill_1024bit_service_cycles']
                    codec['conservative_serial_producer_plus_PPU_cycles']=int(np.maximum(compute,pre).sum()+pc.sum())
                    row['source_encodings']['modal_xor_extension' if extended else 'LoAS_temporal_words']=codec
            if sample==1:
                # Ragged final M tile; raw records retain every destination.
                planfile=HERE/(name+'_numeric_plan.npz')
                np.savez(planfile,perm=perm,**{f'p{i}':p for i,p in enumerate(parents)},
                         **{f'r{i}':r for i,r in enumerate(residuals)},**{f'o{i}':o for i,o in enumerate(orders)})
            out[name]=row
            print(sample,name,'official cycles',got['total_cycles'],'ops',got['num_ops'],flush=True)
            report['samples'][str(sample)]=out
            (HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
