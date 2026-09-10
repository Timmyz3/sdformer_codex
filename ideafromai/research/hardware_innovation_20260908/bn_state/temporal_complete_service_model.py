"""Complete S->restoreY->foldedA reference and seven-bank source sensitivity."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
import json
import math
import time
import numpy as np
from temporal_basis_service_model import (
    HERE, ALG, CAP, T, P, C, H, TILE, LANES,
    compile_routes, psn_lookup, pool as base_pool, read_torch,
    schedule_fc1, serializable)

CONFIGS={
    'direct_T10_1port':('direct_T10_zero_skip','direct_T10_zero_skip',1,False,False,'single'),
    'folded_1port':('folded_time_rows','folded_time_rows',1,False,False,'single'),
    'basis_B_1port':('code_basis_B','code_basis_B',1,False,False,'single'),
    'shared_restore8_1port':('code_basis_B','folded_time_rows',1,True,False,'single'),
    'shared_restore_hidden_bound_1port':('code_basis_B','folded_time_rows',1,True,True,'single'),
    'folded_7bank_slot':('folded_time_rows','folded_time_rows',7,False,False,'slot'),
    'basis_B_7bank_slot':('code_basis_B','code_basis_B',7,False,False,'slot'),
    'shared_restore_hidden_bound_7bank_slot':('code_basis_B','folded_time_rows',7,True,True,'slot'),
    'folded_7bank_interleaved':('folded_time_rows','folded_time_rows',7,False,False,'interleaved'),
    'basis_B_7bank_interleaved':('code_basis_B','code_basis_B',7,False,False,'interleaved'),
    'shared_restore_hidden_bound_7bank_interleaved':('code_basis_B','folded_time_rows',7,True,True,'interleaved'),
}


def encoder_equivalence(D):
    """Exhaustive small exact-ROM check, including lowest-code-index ties."""
    raw=((np.arange(1024)[:,None]>>np.arange(T))&1).astype(bool)
    full=np.count_nonzero(raw[:,None,:]!=D[None,:,:],axis=2).argmin(1)
    table=np.count_nonzero(raw[:256,None,:8]!=D[None,:,:8],axis=2).argmin(1)
    recovered=table[np.arange(1024)&255]
    assert np.array_equal(full,recovered)
    return dict(raw_words=1024,mismatches=int(np.count_nonzero(full!=recovered)),
                table_entries=256,index_time_rows=list(range(8)),
                ignored_time_rows=[8,9],tie_rule='lowest code index',
                note='Raw t1 and t4 are both retained; duplicate codebook rows do not imply duplicate raw gates.')


def encoder_frontend(positions,output_beats):
    """Two128B packets; six exact ROM reads/code writes per beat.

    One position requires16 lookup beats plus a final two-beat ROM/write
    tail for the tile. Both code buffers are compact; no multirow mask
    expansion is hidden in this frontend.
    """
    now,returned,requested,waiting,started=0,-1,0,0,0
    active,ready,peak,conflicts=False,0,0,0
    while True:
        if active and now>=ready:
            active=False
            if started==positions:
                assert requested==positions and waiting==0 and returned<0
                return dict(beats=now+2,raw_packets=requested,
                            packet_peak=peak,bus_conflicts=conflicts)
        if returned==now:
            waiting+=1;returned=-1
        if not active and waiting:
            waiting-=1;started+=1;active=True;ready=now+16
        occupied=int(active)+waiting+int(returned>=0)
        peak=max(peak,occupied)
        if requested<positions and occupied<2 and returned<0:
            if now in output_beats:conflicts+=1
            else:
                requested+=1;returned=now+1;peak=max(peak,occupied+1)
        now+=1
        if returned<0 and active and now<ready and (requested==positions or int(active)+waiting>=2):
            now=ready


def pipeline_step(previous,compute,outputs,positions):
    if previous:
        start=previous['compute_start'];blocked=set(previous['outputs']);old_end=previous['end']
    else:start,blocked,old_end=0,set(),0
    front=encoder_frontend(positions,blocked)
    compute_start=max(old_end,start+front['beats'])
    return dict(compute_start=compute_start,end=compute_start+compute,outputs=outputs,
                compute_idle=previous.get('compute_idle',0)+compute_start-old_end,
                frontend_service=previous.get('frontend_service',0)+front['beats'],
                raw_input_packets=previous.get('raw_input_packets',0)+front['raw_packets'],
                raw_packet_peak=max(previous.get('raw_packet_peak',0),front['packet_peak']),
                input_output_bus_conflicts=previous.get('input_output_bus_conflicts',0)+front['bus_conflicts'])


def pool(coeff):
    p=base_pool(coeff)
    p['parts']['six_exact_encoder_ROMs']=576
    p['parts']['thirtytwo_code_to_time_decoders']=320
    p['payload_bytes']=sum(p['parts'].values())
    p['slack_bytes']=p['allocated_bytes']-p['payload_bytes']
    p['cold_fill_beats']=math.ceil(p['payload_bytes']/128)
    return p


def restoration(pattern):
    s=[bool(pattern>>i&1) for i in range(7)]
    y2=s[0] or s[3];y3=s[0] or s[6];y0=y2 or s[6]
    y7=y0 or s[1];y6=s[2] or s[4]
    return dict(beats=8 if pattern else 0,
                reads=sum(s[i] for i in (0,6,3,1,2,4)),
                writes=sum((y3,y2,y0,y7,y6)),
                add_instructions=5 if pattern else 0,
                nontrivial_adds=sum((s[0] and s[3],s[0] and s[6],
                                    y2 and s[6],y0 and s[1],s[2] and s[4])))


def numeric_restore(codes,W,routes,params):
    positions=np.array([0,1,31,32,127,P-1])
    source=routes['code_basis_B'][0][:,codes[positions]].astype(np.int64)
    slots=np.einsum('rpc,hc->rph',source,W.astype(np.int64))
    r0=slots[0].copy();r1=slots[6].copy()
    slots[6]=r0+r1
    slots[3]=r0+slots[3];r0=slots[3].copy()
    slots[0]=r0+r1;r0=slots[0].copy()
    slots[1]=r0+slots[1]
    r0=slots[2].copy();r1=slots[4].copy();slots[4]=r0+r1
    restored=slots[[0,2,3,6,5,4,1]]
    folded_source=routes['folded_time_rows'][0][:,codes[positions]].astype(np.int64)
    yref=np.einsum('rpc,hc->rph',folded_source,W.astype(np.int64))
    actual=np.einsum('tr,rph->tph',routes['folded_time_rows'][1],restored)
    reference=np.einsum('tr,rph->tph',routes['folded_time_rows'][1],yref)
    tau=params['threshold_int64'][:,None,:]
    pos=params['positive_gain'][None,None,:]
    const=params['constant_channels'][None,None,:]
    cg=params['constant_gate'][:,None,:]
    def gate(u):return np.where(const,cg,np.where(pos,u>=tau,u<=tau))
    out=dict(Y_values=int(yref.size),U_values=int(reference.size),
             Y_mismatches=int(np.count_nonzero(restored!=yref)),
             U_mismatches=int(np.count_nonzero(actual!=reference)),
             gate_mismatches=int(np.count_nonzero(gate(actual)!=gate(reference))))
    assert out['Y_mismatches']==out['U_mismatches']==out['gate_mismatches']==0
    return out


def evaluate(variant,name,routes,params):
    with np.load(CAP/f'{variant}_{name}_codes.npz') as f:codes=f['codes']
    W=read_torch(CAP/f'{variant}_weight_int8.pt')
    assert codes.shape==(P,C)
    counters={key:Counter() for key in CONFIGS}
    pipe={key:{} for key in CONFIGS}
    consumers={key:psn_lookup(coeff) for key,(_,coeff) in routes.items()}
    for lo in range(0,P,TILE):
        code=codes[lo:lo+TILE];n=len(code)
        feature={key:decode[:,code] for key,(decode,_) in routes.items()}
        live={key:f.any(-1) for key,f in feature.items()}
        patterns={key:(v.T*(1<<np.arange(len(v)))).sum(1) for key,v in live.items()}
        service={}
        for src in ('direct_T10_zero_skip','folded_time_rows','code_basis_B'):
            f=feature[src]
            layouts=((1,'single'),) if src=='direct_T10_zero_skip' else ((1,'single'),(7,'slot'),(7,'interleaved'))
            for banks,mapping in layouts:
                if banks==1:count=f.sum((0,1))
                elif mapping=='slot':count=f.sum(1).max(0)
                else:
                    bank=(np.arange(7)[:,None]*TILE+np.arange(n)[None,:])%7
                    count=np.stack([f[bank==b].sum(0) for b in range(7)]).max(0)
                jobs=[(c,int(count[c])) for c in range(C) if count[c]]
                service[src,banks,mapping]=schedule_fc1(jobs)
        assert len({v['memory_words'] for v in service.values()})==1
        for key,(src,consumer,banks,restore,hidden,mapping) in CONFIGS.items():
            ctr=counters[key];s=service[src,banks,mapping]
            updates=int(feature[src].sum())*4;assigned=int(live[src].sum())*4
            ctr['FC1_beats']+=s['beats']*4
            ctr['FC1_issue_rounds']+=s['updates']*4
            ctr['FC1_vector_updates']+=updates
            ctr['FC1_first_assignments']+=assigned
            ctr['FC1_adds_after_first']+=updates-assigned
            ctr['W_read_bytes']+=s['memory_words']*16*4
            ctr['slot_write_bytes']+=updates*LANES*3
            ctr['slot_read_for_add_bytes']+=(updates-assigned)*LANES*3
            ctr['source_code_column_read_bytes']+=len(np.flatnonzero(feature[src].any((0,1))))*12*4
            elapsed=0;outputs=[]
            for q in range(4):
                elapsed+=s['beats']+46
                ctr['tau_read_beats']+=46
                for p in range(n):
                    if restore:
                        rr=restoration(int(patterns[src][p]))
                        elapsed+=0 if hidden else rr['beats']
                        ctr['restore_scheduled_beats']+=0 if hidden else rr['beats']
                        ctr['restore_vector_add_instructions']+=rr['add_instructions']
                        ctr['restore_nontrivial_vector_adds']+=rr['nontrivial_adds']
                        ctr['restore_slot_read_bytes']+=rr['reads']*LANES*3
                        ctr['restore_slot_write_bytes']+=rr['writes']*LANES*3
                    spec=consumers[consumer][int(patterns[consumer][p])]
                    elapsed+=spec['beats'];outputs.append(elapsed-1)
                    ctr['PSN_beats_including_output']+=spec['beats']
                    ctr['PSN_scalar_MAC_issues']+=spec['vector_MACs']*LANES
                    ctr['PSN_slot_read_bytes']+=spec['slot_reads']*LANES*3
                    ctr['PSN_U_read_bytes']+=(spec['vector_MACs']-spec['U_first_products'])*LANES*6
                    ctr['PSN_U_write_bytes']+=spec['vector_MACs']*LANES*6
            pipe[key]=pipeline_step(pipe[key],elapsed,outputs,n)
    base_resources=json.loads((HERE/'temporal_basis_service_resources.json').read_text())
    extra=json.loads((HERE/'temporal_complete_service_resources.json').read_text())
    allocation=base_resources['common_other_raw_bytes'].copy()
    for old in extra['remove_common_raw_allocations']:del allocation[old]
    allocation.update(extra['additional_common_raw_bytes'])
    common=sum(allocation.values())
    modes={}
    for key,(src,consumer,banks,restore,hidden,mapping) in CONFIGS.items():
        v=dict(counters[key]);p=pool(routes[consumer][1])
        if restore:
            p['parts']['restore_control_and_slot_map']=80
            p['payload_bytes']+=80;p['slack_bytes']-=80
            p['cold_fill_beats']=math.ceil(p['payload_bytes']/128)
        v.update(pool=p,kind='optimistic bound, not a demonstrated overlapping schedule' if hidden else 'finite service schedule',
                 slot_banks=banks,wide_slots_per_p=len(routes[src][0]),
                 wide_slot_raw_bytes=len(routes[src][0])*TILE*LANES*3,
                 common_other_raw_bytes=common,common_allocation=allocation,
                 source_add_lanes=96*banks,PSN_MAC_lanes=96,address_mapping=mapping,
                 encoder_ROM_lookups=P*C,encoder_core_service_beats=P*C//6,
                 source_code_write_bytes=P*C*3//8,raw_source_input_bytes=P*128,
                 pipeline={k:x for k,x in pipe[key].items() if k not in ('outputs','compute_start')},
                 warm_chain_beats=pipe[key]['end'],cold_chain_beats=pipe[key]['end']+p['cold_fill_beats'])
        modes[key]=v
    return dict(variant=variant,frame=name,numeric=numeric_restore(codes,W,routes,params),modes=modes)


def summarize(frames):
    out={}
    for variant in sorted({f['variant'] for f in frames}):
        fs=[f for f in frames if f['variant']==variant];modes={}
        for key in CONFIGS:
            vv=[f['modes'][key] for f in fs]
            names=('FC1_beats','FC1_issue_rounds','FC1_vector_updates','W_read_bytes',
                   'slot_write_bytes','slot_read_for_add_bytes','restore_scheduled_beats',
                   'restore_vector_add_instructions','restore_nontrivial_vector_adds',
                   'restore_slot_read_bytes','restore_slot_write_bytes','PSN_beats_including_output',
                   'PSN_scalar_MAC_issues','PSN_slot_read_bytes','source_code_column_read_bytes',
                   'encoder_ROM_lookups','encoder_core_service_beats','source_code_write_bytes',
                   'raw_source_input_bytes','warm_chain_beats','cold_chain_beats')
            modes[key]={n:sum(v.get(n,0) for v in vv) for n in names}
            modes[key]['mean_compute_idle']=sum(v['pipeline']['compute_idle'] for v in vv)/len(vv)
        comp={}
        for label,num,den in (
            ('B_vs_folded_1port','basis_B_1port','folded_1port'),
            ('B_vs_complete_restore8_1port','basis_B_1port','shared_restore8_1port'),
            ('B_vs_complete_hidden_bound_1port','basis_B_1port','shared_restore_hidden_bound_1port'),
            ('B_vs_folded_7bank_slot','basis_B_7bank_slot','folded_7bank_slot'),
            ('B_vs_complete_hidden_bound_7bank_slot','basis_B_7bank_slot','shared_restore_hidden_bound_7bank_slot'),
            ('B_vs_folded_7bank_interleaved','basis_B_7bank_interleaved','folded_7bank_interleaved'),
            ('B_vs_complete_hidden_bound_7bank_interleaved','basis_B_7bank_interleaved','shared_restore_hidden_bound_7bank_interleaved')):
            ratios=[1-f['modes'][num]['warm_chain_beats']/f['modes'][den]['warm_chain_beats'] for f in fs]
            comp[label]=dict(warm_reduction=1-modes[num]['warm_chain_beats']/modes[den]['warm_chain_beats'],
                             cold_reduction=1-modes[num]['cold_chain_beats']/modes[den]['cold_chain_beats'],
                             per_frame_min=min(ratios),per_frame_max=max(ratios))
        best={stem:min((stem+'_7bank_slot',stem+'_7bank_interleaved'),
                       key=lambda x:modes[x]['warm_chain_beats'])
              for stem in ('folded','basis_B','shared_restore_hidden_bound')}
        best_comparisons={}
        for den in ('folded','shared_restore_hidden_bound'):
            num_key,den_key=best['basis_B'],best[den]
            ratios=[1-f['modes'][num_key]['warm_chain_beats']/f['modes'][den_key]['warm_chain_beats'] for f in fs]
            best_comparisons['B_vs_'+den]=dict(numerator=num_key,denominator=den_key,
                warm_reduction=1-modes[num_key]['warm_chain_beats']/modes[den_key]['warm_chain_beats'],
                per_frame_min=min(ratios),per_frame_max=max(ratios))
        out[variant]=dict(frames=len(fs),modes=modes,comparisons=comp,
                         best_static_7bank_mapping=best,best_static_7bank_comparisons=best_comparisons,
                         mapping_selection='One mapping selected per student and route from whole ten-frame totals; no free per-column/frame layout switching. Post-hoc hardware sensitivity, not held-out tuning.')
    return out


def main():
    params=read_torch(ALG/'integer_s0_valid825/integer_parameters.pt')['sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.']
    D=np.load(CAP/'dictionary.npy')
    routes,groups,zeros=compile_routes(D,params['temporal_int16'].astype(np.int64))
    encoder_check=encoder_equivalence(D)
    records=json.loads((CAP/'frames.json').read_text())
    start=time.monotonic();frames=[]
    for variant in ('untrained_code8','trained_code8'):
        for rec in records:
            if rec['variant']!=variant:continue
            name=Path(rec['file']).stem
            f=evaluate(variant,name,routes,params);frames.append(f)
            quick={k:round(v['warm_chain_beats']/1e6,5) for k,v in f['modes'].items()}
            print(variant,name,quick,flush=True)
            result=dict(kind='finite schedule plus explicitly labeled optimistic bounds',
                        resources='temporal_complete_service_resources.json',
                        original_three_route_result='temporal_basis_service_result.json',
                        note='No original results overwritten. Main new denominator includes shared S formation AND restored Y followed by the traditional folded PSN.',
                        encoder_equivalence=encoder_check,
                        frames=frames,aggregate=summarize(frames),elapsed_wall_s=time.monotonic()-start)
            (HERE/'temporal_complete_service_result.json').write_text(json.dumps(serializable(result),ensure_ascii=False,indent=2)+'\n')
    print('FINISHED',len(frames),time.monotonic()-start,flush=True)


if __name__=='__main__':main()
