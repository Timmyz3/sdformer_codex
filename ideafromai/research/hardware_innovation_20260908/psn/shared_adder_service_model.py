"""Static shared-Acc24 PSN schedules on actual trained class-shift captures.

No RTL or new bank simulator. Source RF reads, CSD substeps, finite register
budgets and two output organizations are explicit. Dedicated-MAC FC1/overlap
references are imported from the root's separate measured-input CPU model.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import itertools
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bn_state'))
from support_service_model import read_torch

LANES,TILE,T=96,32,10
ARENA,COMMON_SOURCE_CACHE,CONTROL=5760,7*96*3,64


def csd(value):
    sign=1 if value>=0 else -1
    n=abs(int(value));shift=0;digits=[]
    while n:
        if n&1:
            digit=2-(n&3)
            digits.append((sign*digit,shift));n-=digit
        n//=2;shift+=1
    assert sum(s*(1<<k) for s,k in digits)==value
    return digits


def width(low,high):
    bits=1
    while low<-(1<<(bits-1)) or high>=(1<<(bits-1)):
        bits+=1
    return bits


def prefix_bound(C,decode,W,tau,orders):
    wp=np.maximum(W,0).sum(1);wn=np.minimum(W,0).sum(1)
    low,high=int((-tau).min()),int((-tau).max())
    final=[]
    for t in range(T):
        effective=np.zeros(8,dtype=np.int64)
        for r in orders[t]:
            for sign,shift in csd(int(C[t,r])):
                effective+=sign*(1<<shift)*decode[r]
                lo,hi=int(effective.min()),int(effective.max())
                low=min(low,int((wp*lo+wn*hi-tau[t]).min()))
                high=max(high,int((wp*hi+wn*lo-tau[t]).max()))
        final.append(effective.tolist())
    return dict(residual_range=[low,high],residual_signed_bits=width(low,high),
                final_code_coefficients=final)


def definitions(tag,record,q,books,basis):
    B=np.asarray(record.get('coefficients',record.get('B_int8')),dtype=np.int64)
    tau=np.asarray(record.get('tau',record.get('tau_int32')),dtype=np.int64)
    W=q['weight_int8'].astype(np.int64)
    E=basis[tag+'_coordinates_int8'].astype(np.int64).T
    D=books[tag+'_dictionary'].astype(np.int64)
    selected=basis[tag+'_selected_code_indices']
    basis_time=D[selected].T
    rows=[];seen=set()
    for t in range(T):
        key=tuple(D[:,t])
        if any(key) and key not in seen:rows.append(t);seen.add(key)
    assert len(rows)==6
    M=basis_time[rows]
    inverse=np.rint(np.linalg.inv(M)).astype(np.int64)
    assert np.array_equal(M@inverse,np.eye(6,dtype=np.int64))
    Y=D[:,rows].T
    Aeff=B@inverse
    assert np.array_equal(Aeff@Y,B@E)
    wp=np.maximum(W,0).sum(1);wn=np.minimum(W,0).sum(1)
    result={}
    onehot=np.eye(8,dtype=np.int64)[1:]
    for name,(C,decode) in {'signed':(B,E),'exact_row':(Aeff,Y),
                           'onehot7':((B@E)[:,1:],onehot)}.items():
        rank=C.shape[1]
        orders=[list(range(rank)) for _ in range(T)]
        bounds=prefix_bound(C,decode,W,tau,orders)
        effective=C@decode
        final_low=min(int((wp*row.min()+wn*row.max()-tau[t]).min()) for t,row in enumerate(effective))
        final_high=max(int((wp*row.max()+wn*row.min()-tau[t]).max()) for t,row in enumerate(effective))
        # Try static source permutations only if the natural row-domain order
        # exceeds Acc24; no capture-dependent or runtime order is selected.
        searched=False
        if name=='exact_row' and bounds['residual_signed_bits']>24 and width(final_low,final_high)<=24:
            for t in range(T):
                best=None
                for order in itertools.permutations(range(rank)):
                    effective=np.zeros(8,dtype=np.int64)
                    lo_t,hi_t=int((-tau[t]).min()),int((-tau[t]).max())
                    for r in order:
                        for sign,shift in csd(int(C[t,r])):
                            effective+=sign*(1<<shift)*decode[r]
                            lo,hi=int(effective.min()),int(effective.max())
                            lo_t=min(lo_t,int((wp*lo+wn*hi-tau[t]).min()))
                            hi_t=max(hi_t,int((wp*hi+wn*lo-tau[t]).max()))
                    score=(width(lo_t,hi_t),max(abs(lo_t),abs(hi_t)),order)
                    if best is None or score<best:best=score
                orders[t]=list(best[2])
            bounds=prefix_bound(C,decode,W,tau,orders);searched=True
        source_low=0;source_high=0
        for row in decode:
            lo,hi=int(row.min()),int(row.max())
            source_low=min(source_low,int((wp*lo+wn*hi).min()))
            source_high=max(source_high,int((wp*hi+wn*lo).max()))
        source_bits=width(source_low,source_high)
        digits=[[len(csd(int(c))) for c in row] for row in C]
        acc_bits=max(24,bounds['residual_signed_bits'])
        result[name]=dict(coefficients=C.tolist(),decode=decode.tolist(),
            orders=orders,rank=rank,csd_digits=digits,nonzero_coefficients=int(np.count_nonzero(C)),
            coefficient_signed_bits=width(int(C.min()),int(C.max())),
            total_CSD_digits=sum(map(sum,digits)),accumulator_bits=acc_bits,
            accumulator_24bit_limbs=(acc_bits+23)//24,
            source_static_bits=source_bits,source_static_range=[source_low,source_high],
            source_order_searched=searched,final_residual_range=[final_low,final_high],**bounds)
    assert result['signed']['final_code_coefficients']==result['exact_row']['final_code_coefficients']==result['onehot7']['final_code_coefficients']
    return dict(time_rows=rows,integer_inverse=inverse.tolist(),routes=result,
                theta_source=float(q['theta_source']),theta_output=float(q['theta_output']))


def configuration(route,cache,output):
    acc=route['accumulator_bits'];sourcebits=max(15,route['source_static_bits'])
    rank=len(route['decode'])
    reallocated=cache.endswith('_reallocated')
    stream=cache.startswith('stream')
    G=7 if stream or reallocated else 2 if cache=='resident2_24' else 4 if cache=='resident4_narrow' else 3
    cachebits=sourcebits if reallocated or 'narrow' in cache else 24
    statebits=sourcebits if reallocated else 24
    statebytes=7*TILE*LANES*statebits//8
    arena_limit=64512+ARENA-statebytes if reallocated else ARENA
    while G:
        source_bytes=7*LANES*cachebits//8 if stream else G*rank*LANES*cachebits//8
        gate_bytes=G*128 if output=='packets' else (G*LANES+7)//8
        U_bytes=G*LANES*acc//8
        carry_bytes=(G*LANES+7)//8 if acc>24 else 0
        total=source_bytes+gate_bytes+U_bytes+carry_bytes+CONTROL
        if total<=arena_limit:break
        G-=1
    return dict(positions_per_group=G,cache=cache,output=output,
                source_cache_bits=cachebits,source_cache_bytes=source_bytes,
                main_state_bits=statebits,main_state_bytes=statebytes,
                U_bytes=U_bytes,carry_bytes=carry_bytes,output_buffer_bytes=gate_bytes,
                control_bytes=CONTROL,arena_bytes=total,
                arena_limit_bytes=arena_limit,total_state_plus_arena_bytes=statebytes+total,
                common_tau_broadcast_latch_bytes=LANES*max(acc,24)//8)


def tile_schedule(live,route,cfg):
    digits=np.asarray(route['csd_digits']);C=np.asarray(route['coefficients'])
    limbs=route['accumulator_24bit_limbs'];G=cfg['positions_per_group']
    core=out_end=0;ctr=Counter();bank_reads=np.zeros(7,dtype=np.int64)
    for start in range(0,len(live),G):
        ps=np.arange(start,min(start+G,len(live)))
        group=live[ps];count=group.sum(0);n=len(ps)
        if not cfg['cache'].startswith('stream'):
            needed=np.flatnonzero((C!=0).any(0)&(count!=0))
            core+=len(needed)+bool(len(needed))
            ctr['resident_preload_beats']+=len(needed)+bool(len(needed))
            ctr['state_RF_parallel_read_issue_beats']+=len(needed)
            for r in needed:
                for p in ps[live[ps,r]]:bank_reads[(r*TILE+int(p))%7]+=1
            ctr['state_RF_vector_reads']+=int(count[needed].sum())
        for t in range(T):
            active=(C[t]!=0)&(count!=0)
            issue=int(digits[t,active].sum())*limbs
            # One synchronous source/tau setup, then CSD limbs at II1 and
            # one visible result beat. First operation takes -tau directly.
            local=issue+2
            ctr['CSD_array_issue_beats']+=issue
            ctr['row_setup_and_result_beats']+=2
            ctr['active_lane_vector_adds']+=int((digits[t]*count).sum())*limbs
            ctr['S_cache_vector_operand_reads']+=int((digits[t]*count).sum())*limbs
            ctr['U_first_operation_vectors']+=int(np.any(group[:,C[t]!=0],axis=1).sum())
            if cfg['cache'].startswith('stream'):
                for r in np.flatnonzero(active):
                    for p in ps[live[ps,r]]:bank_reads[(r*TILE+int(p))%7]+=1
                ctr['state_RF_vector_reads']+=int(count[active].sum())
                ctr['state_RF_parallel_read_issue_beats']+=int(active.sum())
            finish=core+local
            if cfg['output']=='fragments' or t==0:
                wait=max(0,out_end-finish);finish+=wait
                ctr['output_backpressure_wait_beats']+=wait
            core=finish
            if cfg['output']=='fragments':out_end=finish+n
        if cfg['output']=='packets':out_end=core+n
        ctr['output_bus_beats']+=n if cfg['output']=='packets' else n*T
    ctr['local_service_beats']=max(core,out_end)
    ctr['output_payload_bytes']=len(live)*T*LANES//8
    ctr['output_transport_bytes']=len(live)*128 if cfg['output']=='packets' else len(live)*T*12
    ctr['state_RF_read_bytes']=ctr['state_RF_vector_reads']*LANES*cfg['main_state_bits']//8
    ctr['S_cache_write_bytes']=ctr['state_RF_vector_reads']*LANES*cfg['source_cache_bits']//8
    ctr['S_cache_operand_read_bytes']=ctr['S_cache_vector_operand_reads']*LANES*cfg['source_cache_bits']//8
    ctr['U_data_write_bits']=ctr['active_lane_vector_adds']*LANES*route['accumulator_bits']/limbs
    for b,count in enumerate(bank_reads):ctr['state_bank'+str(b)+'_vector_reads']=int(count)
    return ctr


def original_reference(params,books,basis):
    defs={};totals=defaultdict(Counter)
    directory=ROOT/'algorithm/stage2_deployment_diverse_capture/capture'
    for path in sorted(directory.glob('*.npz')):
        tag=path.stem.rsplit('_',1)[1]
        if tag not in defs:
            prefix=f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{tag[-1]}.mlp.'
            q=params[prefix]
            record=dict(coefficients=basis[tag+'_B_int32'],tau=q['threshold_int64'])
            defs[tag]=definitions(tag,record,q,books,basis)['routes']['signed']
        route=defs[tag];cfg=configuration(route,'stream7','packets')
        with np.load(path) as f:codes=f['codes']
        live=np.any(np.asarray(route['decode'])[:,codes]!=0,axis=-1).T
        for lo in range(0,len(codes),TILE):
            one=tile_schedule(live[lo:lo+TILE],route,cfg)
            totals[tag].update({k:int(v)*16 for k,v in one.items()})
            totals[tag]['tau_template_load_beats']+=31*16
    return dict(scope='Original B16 and original captured teacher sources; separate numerical model from the two new quantized students.',
        resources='Same Acc24 adders, original exact accumulator widths stored; >24bit CSD additions take two limb issues and84B carry state, within the5760B arena. Original tau fits32bits, so tau/template load31beats, coefficient stripe40832B/319busbeats.',
        definitions=defs,per_module_per_frame={tag:{k:v/10 for k,v in row.items()} for tag,row in totals.items()})


def attach_fc1_references(result):
    """Join each route to its own supplied FC1 trace, without rerunning that model."""
    for route,suffix in (('signed',''),('exact_row','_exact_row'),('onehot7','_onehot7')):
        path=ROOT/'psn'/('class_shift_fc1_services'+suffix+'.json')
        if not path.exists():continue
        refs=json.loads(path.read_text())
        for variant,x in result['variants'].items():
            reference=refs['variants'][variant]
            x.setdefault('route_FC1_and_MAC_references',{})[route]=dict(
                file=path.name,per_frame=reference['per_frame'])
            by_capture={row['capture']:row for row in reference['captures']}
            total=Counter();modules=defaultdict(Counter)
            for capture in x['captures']:
                tag=capture['capture'].removesuffix('.npz').rsplit('_',1)[1]
                for mode,values in capture['modes'].items():
                    if not mode.startswith(route+'/'):continue
                    value=by_capture[capture['capture']]['P32_fc1']+values['PSN_total_local_beats']
                    values['shared_F32_plus_PSN_local_beats']=value
                    total[mode]+=value;modules[tag][mode]+=value
            for mode,value in total.items():
                x['aggregate'][mode]['shared_F32_plus_PSN_local_beats']=value
                x['per_full_frame'][mode]['shared_F32_plus_PSN_local_beats']=value/10
            for tag,by_mode in modules.items():
                for mode,value in by_mode.items():
                    x['modules'][tag][mode]['shared_F32_plus_PSN_local_beats']=value


def append_onehot(args,consumers,params,books,basis):
    """Add only the missing ordinary category baseline; retain prior captures' schedules."""
    result=json.loads(args.output.read_text())
    for variant,x in result['variants'].items():
        counters=defaultdict(Counter);modules=defaultdict(lambda:defaultdict(Counter))
        capture_index={row['capture']:row for row in x['captures']}
        for path in sorted((args.consumers.parent/(variant+'_capture')).glob('*.npz')):
            tag=path.stem.rsplit('_',1)[1]
            if 'onehot7' not in x['definitions'][tag]['routes']:
                prefix=f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{tag[-1]}.mlp.'
                x['definitions'][tag]=definitions(tag,consumers[variant][tag],params[prefix],books,basis)
            route=x['definitions'][tag]['routes']['onehot7']
            with np.load(path) as f:codes=f['codes']
            live=np.any(np.asarray(route['decode'])[:,codes]!=0,axis=-1).T
            for cache in ('stream7_reallocated','resident7_reallocated'):
                key='onehot7/'+cache+'/packets';cfg=configuration(route,cache,'packets');total=Counter()
                for lo in range(0,len(codes),TILE):
                    one=tile_schedule(live[lo:lo+TILE],route,cfg)
                    total.update({k:int(v)*16 for k,v in one.items()})
                    total['tau_template_load_beats']+=24*16
                total['PSN_total_local_beats']=total['local_service_beats']+total['tau_template_load_beats']
                capture_index[path.name]['modes'][key]=dict(config=cfg,**total)
                counters[key].update(total);modules[tag][key].update(total)
        for key,total in counters.items():
            x['aggregate'][key]=dict(total)
            x['per_full_frame'][key]={metric:value/10 for metric,value in total.items()}
            print(variant,key,total['PSN_total_local_beats']/10,flush=True)
        for tag,by_mode in modules.items():
            for key,total in by_mode.items():x['modules'][tag][key]=dict(total)
    result['resources']['narrow_common_storage_bits']=15
    result['resources']['carry_state']='Original two-limb reference uses84B carry state; still within its5760B arena. New trained routes need no carry extension.'
    result['limitations'].append('onehot7 is ordinary exact category aggregation with B7=(B6 E6)[:,1:], charged for seven source slots and CSD of its own coefficients; no FC1 formation cost is borrowed from signed6.')
    attach_fc1_references(result)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumers',type=Path,default=ROOT/'algorithm/stage2_class_shift/consumers.json')
    parser.add_argument('--output',type=Path,default=ROOT/'psn/shared_adder_service_result.json')
    parser.add_argument('--append-onehot',action='store_true')
    args=parser.parse_args()
    consumers=json.loads(args.consumers.read_text())
    params=read_torch(ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    books=np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    basis=np.load(ROOT/'algorithm/stage2_temporal_codes/signed_basis.npz')
    if args.append_onehot:
        append_onehot(args,consumers,params,books,basis)
        return
    fc1=json.loads((ROOT/'psn/class_shift_fc1_services.json').read_text())
    result=dict(scope='actual two trained students, ten frames each, six S2 modules; PSN-local finite static schedules',
                resources=dict(shared_adders='7x96 signed24 adders; no dedicated multiplier in this point',
                    state_RF='7banks x32words x96lanes x24bits; striped address r*32+p',
                    arena_bytes=ARENA,source_read_latency=1,adder_register_feedback_latency=1,
                    common_tau_scratch_bytes=2880,common_operand_staging_bytes=576,
                    tau_template_load_beats_per_H96_tile=24,coefficient_stripe_bytes=39936,
                    coefficient_fill_bus_beats=312,coefficient_pool_bytes=131072,
                    source_order='P32 tile; each group has distinct p modulo7, so one r burst has no bank conflicts',
                    permutation='7 cyclic vector inputs, each96x24bits, route bank outputs to position engines; selector/control and delay are unimplemented',
                    first_add='use -tau in place of previous U; final sign emits gate. Prefix bounds include threshold offset and every CSD substep.',
                    carry='Each >24bit accumulator operation uses low24 then upper limb with carry; store the actual proven accumulator width.',
                    output='packets: finite Gx128B buffer, drain one128B packet/beat. fragments: Gx96bit buffer, one96bit fragment/beat with sideband tags. Core holds completed results until the buffer is free.'),
                limitations=['No RTL, clock, equal-area or energy claim; II1 includes the unproven wide cyclic permutation and shift/add path.',
                    'Source RF bytes are local wide-bank reads, not external DRAM bytes. Stream7 rereads sources across t; resident policies move those reads to an explicitly sized local cache.',
                    'Reallocated point: all routes may use15bit main state and spend the released raw bits on P7 full-source cache. Same total raw capacity is not equal area or proof of the new cache ports.',
                    'CSD and ordinary temporal-row representation are strong controls, not new arithmetic.',
                    'Shared-adder FC1 and PSN cannot overlap on the same adders. Dedicated96-MAC P16 overlap is imported only as a separate-resource optimistic bound.',
                    'Row-domain PSN is exactly the same gate function, but its separate FC1 formation cost is not replaced by signed-source FC1 numbers.',
                    'Fragment sideband/receiver adaptation and global coefficient/output bus arbitration remain outside these local schedules.'],
                variants={})
    for variant in ('integer_trained','power2_trained'):
        definitions_all={};captures=[];aggregate=defaultdict(Counter);modules=defaultdict(lambda:defaultdict(Counter))
        capture_dir=args.consumers.parent/(variant+'_capture')
        files=sorted(capture_dir.glob('*.npz'))
        assert len(files)==60
        f_index={f['capture']:f for f in fc1['variants'][variant]['captures']}
        for path in files:
            tag=path.stem.rsplit('_',1)[1]
            if tag not in definitions_all:
                prefix=f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{tag[-1]}.mlp.'
                definitions_all[tag]=definitions(tag,consumers[variant][tag],params[prefix],books,basis)
            definition=definitions_all[tag]
            with np.load(path) as f:codes=f['codes']
            modes={}
            for route_name,route in definition['routes'].items():
                decode=np.asarray(route['decode'])
                live=np.any(decode[:,codes]!=0,axis=-1).T
                for cache in ('stream7','resident2_24','resident3_narrow','resident4_narrow',
                              'stream7_reallocated','resident7_reallocated'):
                    for output in ('packets','fragments'):
                        if cache=='resident4_narrow' and output=='packets':continue
                        if route_name=='onehot7' and (output!='packets' or cache not in ('stream7_reallocated','resident7_reallocated')):continue
                        key=route_name+'/'+cache+'/'+output
                        cfg=configuration(route,cache,output);total=Counter()
                        for lo in range(0,len(codes),TILE):
                            one=tile_schedule(live[lo:lo+TILE],route,cfg)
                            total.update({k:int(v)*16 for k,v in one.items()})
                            total['tau_template_load_beats']+=24*16
                        total['PSN_total_local_beats']=total['local_service_beats']+total['tau_template_load_beats']
                        modes[key]=dict(config=cfg,**total)
                        aggregate[key].update(total);modules[tag][key].update(total)
            ref=f_index[path.name]
            for key,mode in modes.items():
                if key.startswith('signed/'):
                    mode['shared_F32_plus_PSN_local_beats']=ref['P32_fc1']+mode['PSN_total_local_beats']
                    aggregate[key]['shared_F32_plus_PSN_local_beats']+=mode['shared_F32_plus_PSN_local_beats']
                    modules[tag][key]['shared_F32_plus_PSN_local_beats']+=mode['shared_F32_plus_PSN_local_beats']
            captures.append(dict(capture=path.name,modes=modes,dedicated_MAC_reference=ref))
        result['variants'][variant]=dict(definitions=definitions_all,captures=captures,
            aggregate={k:dict(v) for k,v in aggregate.items()},
            per_full_frame={k:{metric:value/10 for metric,value in v.items()} for k,v in aggregate.items()},
            modules={tag:{k:dict(v) for k,v in mode.items()} for tag,mode in modules.items()},
            dedicated_MAC_reference=fc1['variants'][variant]['per_frame'])
        print(variant,flush=True)
        for key,data in result['variants'][variant]['per_full_frame'].items():
            print(key,round(data['PSN_total_local_beats']/1e6,6),round(data['state_RF_read_bytes']/1e6,3),flush=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    result['original_B16_reference']=original_reference(params,books,basis)
    attach_fc1_references(result)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('wrote',args.output,flush=True)


if __name__=='__main__':main()
