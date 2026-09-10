"""Finite full-frame FP32 issue-service model driven by captured decisions.

Two bounded P4/H32 contexts. Each physical update is P4/H8 on 32 add
lanes; four H8 contexts share an ordinary aligned six-pixel source word.
This models a declared abstract recurrence/port quantum, not a silicon clock.
"""
from pathlib import Path
import argparse
import heapq
import json
import math
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode=True
import numpy as np
from full_frame_requests import unpack

HERE=Path(__file__).resolve().parent
JOINT=HERE.parent
NAMES=['C_PRE_K','C_PRE_T','C_TAIL_K','C_TAIL_T','F_PRE','F_TAIL',
       'CONV_VEC','W_K','W_T','SRC_K','SRC_T','FMA','CMP','PARAM',
       'BN_STEPS','BN_NONEMPTY','DIR_READS','PSN_TICKS','PSN_FMA','PSN_CMP']+[f'C2_TIME{t}' for t in range(10)]+['C_PRE_P','C_TAIL_P','CONV_VEC_P','C_PRE_R','C_TAIL_R','CONV_VEC_R']
I={n:i for i,n in enumerate(NAMES)}


def resource_contract():
    # Maxima are phase-specific allocations inside the same 32KiB pool.
    return dict(
        quantum='One abstract issue step: registerfile read plus vector add/FMA/comparison result is available at its end (dependent latency=1 by assumption). RFreads are combinational in this service abstraction; no claim of a synchronousSRAMread plusFMA fitting3ns. No conversion to ns/MHz; realFP32 pipeline/forwarding is unimplemented.',
        compute=dict(conv='32 conditional FP32 add lanes, oneP4/H8 vector/step',
                     psn_bn='8FP32 FMA lanes, one dependentH8 vector/step; reused forBN1/PSN andBN2+shortcut',
                     compare='8FP32 compares/step; oneU8 result register bounds comparison overlap',
                     ordinary_packing='P:fourPbanks independentlyselectactiveT;R:fixedbank=(p+t)%4,row=10*hctx+t withfour10-bitselectors. W8vectorbroadcast,oneupdateperbank/step. No extraYread/writeport or arbitrarycrossbar;40-bitpermutation andaddress/L1Dlogic are additionalunmappedcontrol.'),
        coefficients=dict(pool_bytes=131072,read_port_bits=256,read_latency_steps=1,
                          current_vector_register_bytes=32,
                          stripe='H32xC96x3x3 FP32=110592B; W1 andW2 each have3stripes. They are never simultaneously fully resident.',
                          cold_fill='32B sharedDMA transactions,5steps each; constants are charged with their stripe'),
        source='Two bounded aligned6-pixel/T10 buffers,2304B each, plus288-bit live-c/kh map and40-bitP4header. Creation reads real source addresses including zeros. L1D consumes one livec/kh per64-bit read; one word feeds3kw and4H8, whose W/ALU work remains serial.',
        state_pool_bytes=32768,
        state=dict(Y_banks='2xP4xH32xT10xFP32=10240B; each bank has four256-bit1R1W P-lane banks and separateH8 addresses',
                   source_payload_bytes=4608,
                   Conv1_gate_row_temporal_bytes=13824,
                   Conv1_gate_row_timeplane_bytes=12800,
                   predictor_cache_bytes=2560,
                   BN1_gain_offset_bytes=256,
                   A_Q14_and_temporal_bias_bytes=108,
                   accept_output_maps_bytes=640,
                   gate_partial_carry_max_bytes=160,
                   U_and_compare_register_bytes=64,
                   source_map_and_header_bytes=96,
                   current_offset_radius_registers_bytes=64,
                   W_register_bytes=32,DMA_staging_bytes=64,
                   worst_listed_Conv1_bytes=32716,
                   remaining_bytes_for_counters_and_tags=52,
                   packing_masks_and_indices_within_remaining_bytes=7,
                   zero_initialization='No perYvalidbitmap is omitted:40P4/H8 zero-vector writes initialize all10Y of aH32context (4for aoneTConv2context). Constantzero writes use thefreshYbank during sourcefill; everydeclaredfill is longerthan bothcontexts initialization, so it adds paidbankwrites butno completiondelay.',
                   Conv2_three_row_gate_cache_bytes=11520,
                   limit='Bare allocated bits and separately specified ports, not SRAM macro area. Conv2 reuses the Conv1 row/predictor space. A physical shared pool/mux map is not implemented.'),
        memory=dict(global_transaction_bytes=32,global_transaction_steps=5,
                    global_bus='One shared6.4B/step-equivalent transaction bus for source,W/constants,gate map,identity andoutput; no overlapping independent copies of this bus.',
                    gate_temporal_layout='[y,c,ceil(x/6)]64-bit words, bit10*(x%6)+t;9953280B/frame',
                    gate_timeplane_layout='[y,Cstripe32,t,channel_in_stripe,xword64], five64-bitwords/outputrow;9216000B/frame. Gate row is assembled with paid64-bitRMW.',
                    identity_output_layout='Declared blocked32B H8 vectors at(p,t,H8), not a free transpose from the capture serialization. Input identity and output each294912000B/frame.'),
        schedule='Two finite contexts; oldest-ready stage wins each resource,exceptgatePACKcommits strictlyinP4/xorder. Ayoungercompletedgroup remains initschargedoutputbitmap until olderPACKcompletes. Conv32 overlaps anothercontextPSN/BN8 onadistinctYbank. DMAandconstruction overlapcomputation. At aConv1rowboundary,thegaterowflushesbeforereuse. Completegatemap separatesConv1/PSN andConv2. Nofreeinterlayerhalofusion.',
        strong_controls=['Ordinary/candidate both retain10Y and use an ordinaryc/kh source broadcast across4H8 underH32 residentW.',
                         'K-major fullT sharing and time-major loops are both timed; the candidate retainsY and recomputes the full row for failed outputs.',
                         'Conv2 fullT/temporal layout and direct timeplane/three-row-cache are coherent alternative pipelines. Different lower bounds are never added as one schedule.',
                         'BN2 andshortcut share the8FMA unit during DMA epilogue. Two32B staging slots hide the twoFMA operations behind5step transactions.'],
        limitations=['Capture drives decisions; model accounts their producer work but is not a numericalFP RTL replay.',
                     'DependentFP completion=1abstractstep is explicit and optimistic; physicalFP latency,area andclock remain open.',
                     'No generic constant-matrix CSE compiler or source-emptyBN/PSN constant folding. EmptyY is supplied as aBNconstant, but the ordinaryPSN still consumes it: an explicitly unoptimized strongbaseline opportunity.',
                     'No across-rowConv2/output overlap, CFMP/fusedConv1-toConv2 tile implementation or larger-context search. Not a completeGustavSNNimplementation.'],
    )


def words_from_gates(gates):
    t,c,ny,nx=gates.shape
    word=np.zeros((c,ny,nx),np.uint16)
    for ti in range(t):word|=gates[ti].astype(np.uint16)<<ti
    pad=np.pad(word,((0,0),(1,1),(1,1)))
    out=np.empty((ny,nx//4,c,3,3),np.uint64)
    for kh in range(3):
        for kw in range(3):
            v=pad[:,kh:kh+ny,kw:kw+nx].reshape(c,ny,nx//4,4)
            packed=sum(v[...,p].astype(np.uint64)<<(10*p) for p in range(4))
            out[:,:,:,kh,kw]=packed.transpose(1,2,0)
    return np.ascontiguousarray(out.reshape(-1,864))


def accepted_and_needs(z,accept_file,a,prefix,mode):
    shape=tuple(int(x) for x in z['source_gate_shape']);T,H,ny,nx=shape
    if mode=='exact':
        accepted=np.zeros(shape,bool);lane_need=np.ones(shape,bool)
        accepted_source='ordinaryfull:no prediction'
    else:
        az=np.load(accept_file) if accept_file else z
        if 'accepted_gate_bits' in az:
            accepted=unpack(az,'accepted_gate_bits','accepted_gate_shape')
        elif 'accepted_bits' in az:
            accepted=unpack(az,'accepted_bits','accepted_shape')
        else:raise ValueError('Exact localPSN timing requires actual acceptedbits; do not infer them from finalgate.')
        accepted_source=str(accept_file or 'gates.npz')
        lane_need=np.empty_like(accepted)
        for s in range(T):
            use=np.flatnonzero(a[:,s])
            lane_need[s]=True if s in prefix else (~accepted[use]).any(0) if len(use) else False
    # H8 andP4 both retain their true positions. A source p is needed if any
    # of the8H lanes needs thatY. IndividualH lane enables do not create morePEs.
    byp=lane_need.reshape(T,12,8,ny,nx//4,4).any(2).transpose(2,3,1,4,0)
    needs=sum(byp[...,p,ti].astype(np.uint64)<<(10*p+ti) for p in range(4) for ti in range(T)).reshape(-1,12)
    bits=accepted.reshape(T,12,8,ny,nx//4,4).transpose(3,4,1,5,0,2)
    am=sum(bits[...,h].astype(np.uint8)<<h for h in range(8)).reshape(-1,12,4,T)
    observed=byp.any(3)
    saved=unpack(z,'need_column_bits','need_column_shape')
    diff=int(np.count_nonzero(observed!=saved))
    counts=z['accepted_gate_counts_H_T'] if 'accepted_gate_counts_H_T' in z else None
    count_diff=None if counts is None else int(np.count_nonzero(accepted.sum((2,3)).T!=counts))
    return needs,am,dict(source=accepted_source,need_column_differences=diff,
                         accepted_H_T_count_differences=count_diff,
                         accepted_bits=int(accepted.sum()),source_lanes='trueP4 positions andH8; group-column capture cross-checked before finer ordinarylane enable')


def run_core(words,needs,am,a,prefix,conditional,c2,tmp):
    exe=HERE/'finite_frame_core'
    cpp=HERE/'finite_frame_core.cpp'
    if not exe.exists() or exe.stat().st_mtime<cpp.stat().st_mtime:
        subprocess.run(['g++','-std=c++17','-O3',str(cpp),'-o',str(exe)],check=True)
    files=[tmp/n for n in ('words.bin','needs.bin','accepted.bin','A.bin')]
    for f,v in zip(files,(words,needs,am,a.astype(np.uint8))):v.tofile(f)
    output=tmp/'services.bin'
    subprocess.run([str(exe),*(str(f) for f in files),str(sum(1<<t for t in prefix)),
                    str(int(conditional)),str(int(c2)),str(output)],check=True)
    return np.fromfile(output,np.uint64).reshape(len(words),3,len(NAMES))


def geometry(ny=240,nx=320,cin=96):
    temporal=np.zeros((ny,nx//4),np.int64);rowcache=np.zeros_like(temporal)
    for y in range(ny):
        ys=range(max(0,y-1),min(ny,y+2))
        for gi,x in enumerate(range(0,nx,4)):
            wx=range(max(0,x-1)//6,(min(nx,x+5)-1)//6+1)
            lines={(((sy*cin+c)*math.ceil(nx/6)+q)//4) for sy in ys for c in range(cin) for q in wx}
            temporal[y,gi]=len(lines)*5
            q0=max(0,x-1)//64;q1=(min(nx,x+5)-1)//64
            rowcache[y,gi]=len(ys)*cin*(q1-q0+1)
    return temporal,rowcache


def run_jobs(jobs,slots=2):
    """Event-level FIFO resources and finite contexts, with real stage edges.

    Each stage=(resource,duration,label). Jobs are allowed to interleave only
    after the prior stage has completed. There is no sum/max of unrelated
    lower bounds. Zero-duration stages are elided.
    """
    active={};next_job=0;events=[];busy={};usage={};labels={};now=0;done=0;peak=0;timeline=[];next_pack=0
    while done<len(jobs):
        while events and events[0][0]<=now:
            end,j,res=heapq.heappop(events);busy.pop(res);active[j]['stage']+=1
            if res=='PACK':next_pack+=1
            if active[j]['stage']==len(active[j]['work']):del active[j];done+=1
        while len(active)<slots and next_job<len(jobs):
            work=[x for x in jobs[next_job] if x[1]]
            if work:active[next_job]=dict(stage=0,work=work)
            else:done+=1
            next_job+=1
        peak=max(peak,len(active))
        progressed=True
        while progressed:
            progressed=False
            for j in sorted(active):
                q=active[j]
                if any(e[1]==j for e in events):continue
                res,d,label=q['work'][q['stage']]
                if res in busy:continue
                if res=='PACK' and j!=next_pack:continue
                d=int(d);busy[res]=j;heapq.heappush(events,(now+d,j,res));usage[res]=usage.get(res,0)+d;labels[label]=labels.get(label,0)+d
                if len(timeline)<30:timeline.append([now,now+d,j,res,label])
                progressed=True
        if events:now=events[0][0]
        elif done<len(jobs):raise RuntimeError('stalledfiniteDAG')
    return dict(steps=int(now),busy_steps=usage,work_steps=labels,context_peak=peak,first_events=timeline)


def add_stats(target,part):
    target['steps']+=int(part['steps'])
    for name in ('busy_steps','work_steps'):
        for k,v in part[name].items():target[name][k]=target[name].get(k,0)+int(v)
    target['context_peak']=max(target['context_peak'],part['context_peak'])


def layer_schedule(core,geom,mode,c2,layout,packing='K'):
    ny,nxg=geom[0].shape
    total=dict(steps=0,busy_steps={},work_steps={},context_peak=0,first_row=None,
               zero_vector_initialization_writes=ny*nxg*3*40,
               zero_initialization_note='40zero-vector writes perfullT context,or4perT context; constantwrites tofreeYbank overlap sourceDMA/localfill. Bothaxes ownthesameinitializationport; no implicitperYvalidbits.')
    temporal,timecache=geom
    def extra(res,d,label):
        total['steps']+=int(d);total['busy_steps'][res]=total['busy_steps'].get(res,0)+int(d);total['work_steps'][label]=total['work_steps'].get(label,0)+int(d)
    for stripe in range(3):
        constants=(256+68+40)+(2560 if not c2 and mode=='conditional' else 0)
        extra('DMA',5*math.ceil((110592+constants)/32),'W_and_constant_cold_fill')
        times=range(10) if c2 and layout=='three_row' else [None]
        for ti in times:
            for y in range(ny):
                if c2 and layout=='three_row':
                    newrows=2 if y==0 else int(y+1<ny)
                    extra('DMA',newrows*3840//32*5,'gate_three_row_refill')
                jobs=[]
                for x in range(nxg):
                    g=y*nxg+x;r=core[g,stripe]
                    if c2 and layout=='three_row':
                        stages=[('SUPPLY',int(timecache[y,x])+2,'three_row_source_scan'),
                                ('CONV',int(r[I[f'C2_TIME{ti}']])+6,'Conv2_time_parallelH8_serial'),
                                ('DMA',160,'BN2_shortcut_output_epilogue')]
                    else:
                        stages=[('DMA',int(temporal[y,x]),'source_halo_transactions'),('BUILD',4,'source_align_directory_tail')]
                        if c2:
                            stages += [('CONV',int(r[I['C_PRE_'+packing]])+6,'Conv2_fullT'),
                                       ('DMA',1600,'BN2_shortcut_output_epilogue')]
                        else:
                            k=layout.rsplit('_',1)[1]
                            stages += [('CONV',int(r[I['C_PRE_'+k]])+6,'Conv1_prefix_or_full'),
                                       ('FMA',int(r[I['F_PRE']]),'BN1_prefix_and_PSN_or_full')]
                            if mode=='conditional':
                                stages += [('CONV',int(r[I['C_TAIL_'+k]]),'Conv1_tail'),
                                           ('FMA',int(r[I['F_TAIL']]),'BN1_tail_and_failed_PSN_recompute')]
                            if layout.startswith('temporal'):
                                x0=x*4;whole=(x0+4)//6-x0//6
                                if x==nxg-1 and (x0+4)%6:whole+=1
                                pack=max(1,whole*32) #even nofullword mustcommit thepartialcarry inP4order
                            else:pack=321 #32064-bitRMWs:1R1W pipeline,oneinitialreadlatency
                            stages.append(('PACK',pack,'gate_row_pack'))
                    jobs.append(stages)
                part=run_jobs(jobs)
                if total['first_row'] is None:total['first_row']=part
                add_stats(total,part)
                if not c2:
                    rowbytes=13824 if layout.startswith('temporal') else 12800
                    extra('DMA',rowbytes//32*5,'gate_row_flush')
    total['model_only']=True
    return total


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture-dir',type=Path,required=True)
    ap.add_argument('--parameters',type=Path,required=True)
    ap.add_argument('--predictor',type=Path,required=True)
    ap.add_argument('--accepted-file',type=Path)
    ap.add_argument('--mode',choices=['exact','conditional'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();started=time.monotonic()
    z=np.load(args.capture_dir/'gates.npz');pars=np.load(args.parameters);pred=np.load(args.predictor)
    source=unpack(z,'source_gate_bits','source_gate_shape');gates=unpack(z,'output_gate_bits','output_gate_shape')
    assert source.shape==(10,96,240,320) and gates.shape==source.shape
    for key in ('W1','W2'):
        assert pars[key].shape==(96,96,3,3) and pars[key].dtype==np.float32 and np.all(pars[key]!=0), 'This boundedFP reference requires the actual all-nonzeroW; do not silently mapW8 into it.'
    a=pred['temporal_q14']!=0;prefix=pred['prefix'].astype(int).tolist()
    needs,am,review=accepted_and_needs(z,args.accepted_file,a,prefix,args.mode)
    assert review['need_column_differences']==0
    assert review['accepted_H_T_count_differences'] in (0,None)
    with tempfile.TemporaryDirectory(prefix='finite_work_',dir=HERE) as tmpdir:
        tmp=Path(tmpdir)
        w1=words_from_gates(source)
        c1=run_core(w1,needs,am,a,prefix,args.mode=='conditional',False,tmp)
        del w1
        w2=words_from_gates(gates)
        fullneeds=np.full_like(needs,(1<<40)-1)
        c2=run_core(w2,fullneeds,np.zeros_like(am),a,prefix,False,True,tmp)
        del w2
    geom=geometry()
    schedules={};second_cache={}
    for route,clayout in [('temporal','full_T'),('timeplane','three_row')]:
        for order in ('K','T','P','R'):
            corder=order if order in ('P','R') else 'K'
            key=(clayout,corder if clayout=='full_T' else 'T')
            if key not in second_cache:second_cache[key]=layer_schedule(c2,geom,'exact',True,clayout,packing=corder)
            second=second_cache[key]
            first=layer_schedule(c1,geom,args.mode,False,route+'_'+order)
            name=route+'_'+order
            schedules[name]=dict(Conv1_BN1_PSN=first,Conv2_BN2_shortcut=second,total_steps=first['steps']+second['steps'])
    best=min(schedules,key=lambda k:schedules[k]['total_steps'])
    result=dict(frame=str(z['frame_name']),mode=args.mode,prefix=prefix,
                gate_pack_ordered=True,partial_carry_commit_steps=1,
                precision='ActualFP32W1/W2 captured student;A has its capturedQ14 coefficient values. No oldINT8need or integerPE is used.',
                resources=resource_contract(),accepted_review=review,
                actual_theta_source=float(z['theta_source']),actual_theta_output=float(z['theta_output']),
                same_frame_numeric_checks='See correspondingfull_frame_results:fullgate/windows and realW/BN/identity already checked. This program times the captured decisions,not a newFP numericalimplementation.',
                per_operator_service={layer:{n:int(core[...,i].sum()) for i,n in enumerate(NAMES) if not n.startswith('C2_TIME')} for layer,core in [('Conv1',c1),('Conv2',c2)]},
                schedules=schedules,best_legal_route=best,best_total_steps=schedules[best]['total_steps'],
                best_selection='Minimum of eight explicit whole-layer schedules (two gate/layoutpipelines x four ordinaryloop/addresscontrols); each must be implemented. Not a mixture of independentlychosen lowerbounds.',
                complete_scope='source-ready packed sn1 + blocked identity -> full r1block output in external memory, one complete240x320/T10/C96 frame; explicit materializedgate barrier, noConv2halo omitted.',
                wall_seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(frame=result['frame'],mode=args.mode,best=best,steps=result['best_total_steps'],per_route={k:v['total_steps'] for k,v in schedules.items()},wall_seconds=result['wall_seconds']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
