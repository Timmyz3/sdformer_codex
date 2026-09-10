"""Paper-guided GustavSNN CPTB/NRV synaptic reference; not an official artifact.

Run real power2_trained S2 codes, onehot7 and ordinary unique time rows.
The endpoint exports integer partial sums to a ready abstract consumer. Noncausal T10
PSN, original source producer, BN2, FC2, bus refill and PPA are not measured.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import sys, subprocess, json, argparse, time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'bn_state'))
from support_service_model import read_torch
from shared_adder_service_model import csd,prefix_bound


def identities(b,params,books,basis,consumers):
    tag=f's2b{b}'
    q=params[f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{b}.mlp.']
    W=np.ascontiguousarray(q['weight_int8'],dtype=np.int8)
    D=books[tag+'_dictionary'].astype(np.int64)
    E=basis[tag+'_coordinates_int8'].astype(np.int64).T
    B=np.asarray(consumers['power2_trained'][tag]['B_int8'],dtype=np.int64)
    unique=[];seen=set()
    for t in range(10):
        key=tuple(D[:,t])
        if any(key) and key not in seen:unique.append(t);seen.add(key)
    rows=D[:,unique].T
    M=D[basis[tag+'_selected_code_indices']].T[unique]
    inv=np.rint(np.linalg.inv(M)).astype(np.int64)
    assert np.array_equal(M@inv,np.eye(6,dtype=np.int64))
    onehot=np.eye(8,dtype=np.int64)[1:]
    routes={'onehot7':(onehot,(B@E)[:,1:]),'time_rows':(rows,B@inv),
            'packed_class':(onehot,(B@E)[:,1:]),'packed_time':(rows,B@inv)}
    for decode,A in routes.values():assert np.array_equal(A@decode,B@E)
    wp=np.maximum(W.astype(np.int64),0).sum(1)
    wn=np.minimum(W.astype(np.int64),0).sum(1)
    assert wn.min()>=-16384 and wp.max()<16384
    return W,routes,dict(tag=tag,shape=list(W.shape),unique_time_rows=unique,
        zero_weight_fraction=float(np.mean(W==0)),integer_source_range=[int(wn.min()),int(wp.max())],
        source_theta_in_weight=True,source_theta_assumed_one=False,
        final_consumer_identity='B7*onehot == A_eff*unique_rows == B6*E6; verified for all 8 codes')


def specialize_routes(variant,b,W,routes,identity,producers,consumers):
    prefix=f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{b}.mlp.'
    reachable=(list(range(8)) if variant=='teacher' else
               sorted(set(np.asarray(producers[prefix]['bits3' if variant=='integer' else variant]['mapping']).tolist())))
    zero_cols=[c for c in range(8) if c not in reachable]
    onehot,A7=routes['onehot7']
    keep=[c-1 for c in reachable if c]
    dec=onehot[keep].copy();C=A7[:,keep].copy()
    dec[:,zero_cols]=0
    result={'onehot7':(dec,C),'packed_class':(dec,C)}
    oldD,oldA=routes['time_rows'];cols=[];As=[];seen={}
    for j,row in enumerate(oldD):
        key=tuple(row[reachable])
        if not any(key):continue
        if key in seen:As[seen[key]]+=oldA[:,j]
        else:
            seen[key]=len(cols);newrow=row.copy();newrow[zero_cols]=0
            cols.append(newrow);As.append(oldA[:,j].copy())
    dec=np.asarray(cols,dtype=np.int64);C=np.asarray(As,dtype=np.int64).T
    result.update(time_rows=(dec,C),packed_time=(dec,C))
    expected=A7@onehot
    tau=np.asarray(consumers['power2_trained'][f's2b{b}']['tau_int32'],dtype=np.int64)
    details={}
    for name,(decode,A) in result.items():
        assert np.array_equal((A@decode)[:,reachable],expected[:,reachable])
        R=len(decode);digits=sum(len(csd(v)) for v in A.flat)
        bound=prefix_bound(A,decode,W.astype(np.int64),tau,[list(range(R)) for _ in range(10)])
        assert bound['residual_signed_bits']<=24
        details[name]=dict(reachable_codes=reachable,rank=R,csd_digits_per_p=digits,
            nonzero_coefficients=int(np.count_nonzero(A)),coefficient_range=[int(A.min()),int(A.max())],
            residual_prefix_bound=bound,source_cache_bits=2*R*15*64,U_bits=24*64,
            threshold_register_bits=8*10*24,threshold_read_muxes='8 independent 10:1 x24 register muxes per tile',
            microprogram_replica_bits=8*(digits+11)*12,source_state_bits_per_P=64*R*15,
            source_reads_per_p=R,threshold_inits_per_p=10,CSD_program_issues_per_p=digits+10,
            PSN_partition_service='R+1 startup + P*(actual CSD digits +10 threshold inits) +1 final bit',
            gate_FIFO_bits=2*64,output_bits_per_p=10,
            common_adder_extension='one existing partial accumulator per PE extended to24; separate synaptic state-adder remains15',
            shared_adders=64,extra_MACs=0,
            boundary='FC1->noncausal T10 integer PSN->gate to ready sink; source producer/FC2/BN2/global bus excluded')
    newident=dict(identity,reachable_codes=reachable,
                  statically_retained_nonzero_classes=len(keep),
                  statically_retained_time_rows=len(cols),
                  final_consumer_identity='equal on every statically reachable producer code; dead codes excluded by actual saved mapping')
    return result,newident,details,tau


def finalize_metadata(out):
    consumer=out.get('consumer_CSD_enabled',False)
    finite=out.get('finite_supply_enabled',False)
    out['scope']=('integer FC1 + exact compiled noncausal T10/CSD PSN + finite serial source/weight/tau/bit bus' if finite else
                  'integer FC1 + exact compiled noncausal T10/CSD PSN to ready gate sink' if consumer else
                  'synaptic integer FC1 to ready partial-sum sink')
    out['identity_note']='Current trained B/tau and restricted temporal dictionary form the same-student noncausal T10 consumer; no claim of rank10 or exact original ep34 arithmetic.'
    out['excluded']=['source producer arithmetic','FC2','dynamic BN2','shortcut','mapped SRAM/logic PPA']
    if not finite:out['excluded']+=['finite global refill/backpressure']
    out['common']['same_total_raw_capacity']=dict(
        partial_state_data_bits=64*7*8*15,partial_state_valid_bits=64*7*8,
        double_source_cache_bits=64*2*7*15,U_bits=64*24,threshold_bits=8*10*24,
        microprogram_bits=8*85*12,packed_code_decode_bits=64*4*8*7,
        packed_time_cursor_bits=64*4*7,gate_buffer_bits=64*8*10 if finite else 64*2,
        partial_packet_buffers_bits=64*2*4*(8*4+8+9),
        source_buffer_bytes=8*1024,source_extract_register_bytes=8*16 if finite else 0)
    out['common']['same_total_raw_capacity_note']='P8 shared hardware maxima; time/rows3 may clock-gate unused space. Other uses of slack, such as additional position prefetch, are not optimized.'
    if consumer:
        out['common']['shared_CSD_adders']=64
        out['common']['dedicated_MACs']=0
        out['common']['synaptic_partial_adder_width']=24
        out['common']['synaptic_local_state_adder_width']=15
        out['common']['source_RF_ports_per_PE']='1R1W, data15+valid; not a mapped memory'
        out['timing_assumptions']=[x for x in out['timing_assumptions'] if 'ready sink drain' not in x and 'PSN/BN2' not in x]
        out['timing_assumptions']+=['CSD uses same64 PE partial adders extended to24, double R-word caches hide next-p S reads',
            'program and tau loaded before work; same-ID CSD sequence broadcast across output tiles',
            'compare/bit enqueue overlaps next threshold initialization; final bit paid; no downstream FC2 modeled']
        out['limits']=[x for x in out['limits'] if 'future PSN' not in x]
        out['limits']+=['integer noncausal T10 PSN modeled, FP32 source producer/FC2/BN2 and mapped timing remain outside']
    if finite:
        out['common']['source_buffer_ports_per_ID']='64-bit 1R1W, specified model choice'
        out['common']['global_bus']='one shared32B/5beats =6.4B/beat; W/tau/source/program/output serialized'
        out['common']['source_decoder_rows_per_ID_per_beat']=1
        out['common']['source_D_chunk']=128
        out['common']['source_NRV_to_W_latency']=2
        out['common']['source_NRV_to_W_outstanding_per_ID']=1
        out['finite_schedule']=['cold W/tau bus -> explicit one-byte local writes -> tau register preload',
            'for each spatial wave, for each of three D128 chunks: dense-code bus load -> reverse scan/NRV encode including zero rows -> CPTB execute',
            'R*P partials and validity survive all three chunks, then CSD consumes them',
            'all P*T gates held in common640B before serial output bus drain; next spatial wave starts afterward']
        out['finite_layout_check']='all-live128-row case + every real staged record decoded back to original row/code; reverse read and reverse high-address write verified'
        out['finite_limit']='deliberately serial global refill/encode/execute/output, not an optimal-overlap schedule or official cycles'
        out['timing_assumptions']=[x for x in out['timing_assumptions'] if 'current NRV record' not in x]
        out['limits']=[x for x in out['limits'] if 'NRV plane bytes' not in x]
    for record in out['records']:
        R=record['rank'];P=record['P']
        record.setdefault('logical_partial_valid_bits',64*R*P)
        record['common_physical_partial_state_bits']=64*7*P*15
        record['common_physical_partial_valid_bits']=64*7*P
        record['common_physical_double_source_cache_bits']=64*2*7*15
        record['common_physical_microprogram_bits']=8*85*12
        record['common_physical_gate_buffer_bits']=64*P*10 if finite else 64*2
        if finite:
            record['bus_actual_transferred_bytes']=record['bus_beats']//5*32
            record['bus_padding_bytes']=record['bus_actual_transferred_bytes']-sum(record[k] for k in ('bus_input_bytes','bus_weight_bytes','bus_tau_bytes','bus_program_bytes','bus_output_bytes'))
            record['service_component_sum']=(record['bus_beats']+record['input_decode_beats']+
                record['local_weight_write_beats']+record['local_tau_preload_beats']+record['finite_compute_beats'])
            assert record['service_component_sum']==record['beats']
            assert record['source_buffer_peak_bytes']<=1024
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=1)
    ap.add_argument('--p',type=int,nargs='+',default=[8])
    ap.add_argument('--groups',type=int,default=0,help='0 = all 192 output groups')
    ap.add_argument('--output',default='gustavsnn_reference.json')
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--variants',nargs='+',default=['teacher'])
    ap.add_argument('--routes',nargs='+',default=['onehot7','time_rows','packed_class','packed_time'])
    ap.add_argument('--consumer-csd',action='store_true')
    ap.add_argument('--finite-supply',action='store_true')
    args=ap.parse_args()
    core=HERE/('gustavsnn_reference_finite_core' if args.finite_supply else 'gustavsnn_reference_consumer_core' if args.consumer_csd else 'gustavsnn_reference_core')
    subprocess.run(['g++','-O3','-std=c++17',str(HERE/'gustavsnn_reference_core.cpp'),'-o',str(core)],check=True)
    data=ROOT/'algorithm'/'stage2_temporal_codes'
    params=read_torch(data/'integer_parameters.pt')
    books=np.load(data/'codebooks.npz');basis=np.load(data/'signed_basis.npz')
    consumers=json.loads((ROOT/'algorithm/stage2_class_shift/consumers.json').read_text())
    producers=read_torch(ROOT/'algorithm/direct_code_stage2/producers.pt')
    captures=[]
    for variant in args.variants:
        folder=(ROOT/'algorithm/stage2_class_shift/power2_trained_capture' if variant=='teacher' else
                ROOT/'algorithm/direct_code_integer/deployment/capture10' if variant=='integer' else
                ROOT/'algorithm/direct_code_stage2'/('compiled_'+variant)/'capture')
        files=sorted(folder.glob('*.npz'))
        if args.limit:files=files[:args.limit]
        captures.extend((variant,p) for p in files)
    out=dict(finite_supply_enabled=args.finite_supply,consumer_CSD_enabled=args.consumer_csd,kind='PAPER_GUIDED_CPU_SERVICE_MODEL',paper_doi='10.1109/HPCA68181.2026.11408587',
      input='actual power2_trained S2 projected codes; source producer and encoder outside boundary',
      measured_capture_count=0,expected_available_captures=60*len(args.variants),variants=args.variants,records=[],
      common=dict(tiles=8,PE_per_tile=8,merger_rows=4,weight_bits=8,partial_bits=15,
        local_weight_ports_per_tile=2,weight_words_per_port_per_beat=1,weight_return_latency=1,
        weight_fetch_outstanding_per_PE=1,execute_selected_spikes_per_PE_per_beat=1,
        PE_packet_buffers=2,shared_NRV_fetch_rows_per_ID_per_beat=1,
        output_partial_words_per_PE_per_beat=1,output_sink_ready=True,
        local_weight_capacity_bytes_per_tile=8192,weight_global_capacity_bytes=131072,
        original_spike_global_capacity_bytes_per_ID=1024),
      paper_facts=['CPTB: parallel output-row tiles, parallel position partitions, time-second',
       'NRV: row index plus P-bit mask, all-zero row omitted',
       'fetch fills NR registers with nonzero-weight rows; W=0 consumes a read but no register',
       'L1D/merger selects one smallest-column spike per PE beat; same-column partial aggregation',
       'two pipeline stages, shared dual-port local W, same-ID cross-tile synchronization'],
      timing_assumptions=['same-ID raw NRV row broadcast uses common ready of all 8 tiles, conservative bounded multicast',
       'two shared W grant IDs per beat, fair round-robin, at most one read/ID, one-beat response',
       'two NR4 buffers per PE, independent compaction of W=0, next fetch overlaps execution',
       'complete partition/plane retires only when all same-ID tiles finish; one descriptor beat',
       'packet boundary flush uses separate local-state update path; last flush costs one drain beat',
       'R*P partials retained for future noncausal PSN; one-word/PE/beat ready sink drain before next partition',
       'current NRV record and current output W row ready; global refill/encoder/PSN/BN2/FC2 excluded',
       'packed routes use P-bit L1D, 3-bit code select and class/time cursor; no P*R-bit L1D' ],
      limits=['not official software or RTL, not original-paper measured cycles or PPA',
       'raw-row multicast backpressure is our scheduling choice, paper does not specify its exact arbiter',
       'P16/P32 keep 8 PE per tile: resource sensitivity, not paper equal-area sweep',
       'NRV plane bytes beyond original 1KiB require input streaming/staging not timed here',
       'future PSN full-layer consumer and contention/backpressure are not closed'])
    by_block={b:identities(b,params,books,basis,consumers) for b in range(6)}
    specialized={}
    control=json.loads((HERE/'class_reconstruct_control.json').read_text())
    control={(r['variant'],r['block']):r for r in control['records']}
    for variant in args.variants:
        for b,(W,rr,ident) in by_block.items():
            spec=specialize_routes(variant,b,W,rr,ident,producers,consumers)
            routes,identity,costs,tau=spec
            if variant in ('teacher','bits3','integer','rows3'):
                guide=control['rows3' if variant=='rows3' else 'bits3',b]
                routes['packed_class_best_tail']=routes['packed_class']
                cost=dict(costs['packed_class'],tail_choice=guide['blockwise_choose'],
                          restore_loads_per_p=0,restore_add_stores_per_p=0,
                          restore_added_hardware='U24 result-to-cache15 write mux, enabled identically in the common hardware superset')
                if guide['blockwise_choose']=='restore_then_A':
                    plan=guide['plan']
                    cost.update(csd_digits_per_p=guide['time_A_CSD_digits'],
                        CSD_program_issues_per_p=guide['restore_one_read_found']+10,
                        restore_loads_per_p=plan['U_loads'],restore_add_stores_per_p=plan['cache_writes'],
                        restore_microprogram=plan['microprogram'],restore_alias_slots=plan['output_alias_slots'],
                        restored_time_coefficients=routes['packed_time'][1].tolist(),
                        residual_prefix_bound=costs['packed_time']['residual_prefix_bound'],
                        restore_peak_active_cache_words=plan['active_cache_peak_words'])
                costs['packed_class_best_tail']=cost
            specialized[variant,b]=spec
    def execute(item):
        variant,cap,P,route=item
        b=int(cap.stem[-1])
        W,_,_=by_block[b]
        routes,identity,costs,tau=specialized[variant,b]
        decode,A=routes[route]
        codes=np.ascontiguousarray(np.load(cap)['codes'],dtype=np.uint8)
        assert codes.shape==(1200,384) and codes.max()<8
        assert set(np.unique(codes)).issubset(identity['reachable_codes'])
        payload=np.array([1200,384,1536,len(decode)],dtype='<i4').tobytes()+W.tobytes()+codes.tobytes()+np.asarray(decode,dtype=np.uint8).tobytes()
        start=time.monotonic()
        proc=subprocess.run([str(core),str(P),str(args.groups),str(int(route.startswith('packed'))),str(costs[route]['CSD_program_issues_per_p'] if args.consumer_csd else 0),str(int(args.finite_supply)),str(costs[route].get('restore_loads_per_p',0)),str(costs[route].get('restore_add_stores_per_p',0))],input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        record=json.loads(proc.stdout)
        # These independent dot-count checks do not use the scheduling loops.
        nonzero_weight_per_c=np.count_nonzero(W,axis=0)
        events_per_code=np.asarray(decode).sum(0)
        per_c=events_per_code[codes].sum(0)
        expected_selected=int(per_c@nonzero_weight_per_c)
        if args.groups==0:assert expected_selected==record['selected']
        record.update(finite_supply_enabled=args.finite_supply,consumer_CSD_enabled=args.consumer_csd,consumer_cost=costs[route] if args.consumer_csd else None,
            capture=cap.name,variant=variant,block=b,P=P,route=route,rank=len(decode),wall_seconds=time.monotonic()-start,
            module_identity=identity,consumer_coefficients=A.tolist(),
            independent_expected_full_H_selected=expected_selected,
            complete_P_C_H=args.groups==0,
            cold_weight_fill_bytes=int(W.size),
            cold_tau_fill_bytes=1536*10*3 if args.consumer_csd else 0,
            consumer_source_read_bytes_15bit=record.get('consumer_source_words',0)*15/8,
            local_weight_scalar_read_bytes=record['local_w_reads'],
            local_state_write_bytes_15bit=record['state_writes']*15/8,
            local_state_read_bytes_15bit=record['state_reads']*15/8,
            shared_NRV_payload_bytes=record['input_payload_bits']/8*record['groups'],
            stored_NRV_source_plus_W_6p4B_per_beat_bound=(record['input_payload_bits']/8*record['groups']+W.size)/6.4,
            dense_code_source_plus_W_6p4B_per_beat_bound=(1200*384*3/8*record['groups']+W.size)/6.4,
            dense_code_replay_bytes=1200*384*3/8*record['groups'],
            dense_code_input_decode_scope='alternative common global format; D scan/NRV production and refill schedule not timed',
            retained_partial_state_bits=8*8*P*len(decode)*15,
            common_physical_partial_state_bits=8*8*P*7*15,
            common_physical_partial_valid_bits=8*8*P*7,
            logical_partial_valid_bits=8*8*P*len(decode),
            common_physical_double_source_cache_bits=64*2*7*15,
            common_physical_U_bits=64*24,
            common_physical_threshold_bits=8*10*24,
            common_physical_microprogram_bits=8*(74+11)*12,
            common_physical_packed_decode_ROM_bits=64*4*8*7,
            common_physical_packed_cursor_bits=64*4*7,
            common_physical_gate_buffer_bits=64*P*10 if args.finite_supply else 64*2,
            unused_capacity_note='shorter R may clock-gate unused slots; extra time-side position prefetch not optimized',
            original_LIF_potential_slots=8*8*P,
            scalar_consumer_stages='B/A_eff fixed CSD on same64 PE adders; actual prefix bound checked' if args.consumer_csd else 'unmodeled',
            per_PE_packet_raw_bits=2*4*(P*(4 if route.startswith('packed') else 1)+8+9),
            per_PE_packed_time_cursor_bits=4*len(decode) if route=='packed_time' else 0,
            NRV_plane_fits_original_1KiB=record['peak_nrv_plane_bytes']<=1024,
            output_partial_bytes_15bit=record['boundary_words']*15/8,
            selected_spike_PE_utilization=record['selected']/(record['beats']*64),
            state_update_arithmetic='separate 15-bit local potential adder overlaps partial selected-weight adder',
            arbiter_denied_requests_scope='eligible requests denied by two-port limit',
            bank_backpressure_scope='event-time observations of any tile packet-buffer blockage; not integrated stall cycles')
        if args.consumer_csd:
            ps=[0,1,37,511,1199];hs=[0,8,17,511,1535]
            S=decode[:,codes[ps]].transpose(1,0,2).astype(np.int64)@W[hs].astype(np.int64).T
            u=np.einsum('tr,prh->tph',A,S)
            direct=by_block[b][1]['onehot7']
            Dref,Aref=direct
            Sref=Dref[:,codes[ps]].transpose(1,0,2).astype(np.int64)@W[hs].astype(np.int64).T
            ref=np.einsum('tr,prh->tph',Aref,Sref)
            assert np.array_equal(u,ref)
            CSD_A=A;CSD_S=S
            if costs[route].get('tail_choice')=='restore_then_A':
                cache=np.zeros((7,len(ps),len(hs)),dtype=np.int64)
                cache[:len(decode)]=S.transpose(1,0,2)
                tmp=np.zeros((len(ps),len(hs)),dtype=np.int64)
                for instruction in costs[route]['restore_microprogram']:
                    if instruction['op']=='load_U':tmp=cache[instruction['src_slot']].copy()
                    elif instruction['op']=='add_store':
                        tmp=tmp+cache[instruction['src_slot']]
                        assert tmp.min()>=-16384 and tmp.max()<16384
                        cache[instruction['dst_slot']]=tmp
                    else:raise ValueError(instruction)
                CSD_S=cache[costs[route]['restore_alias_slots']].transpose(1,0,2)
                CSD_A=np.asarray(costs[route]['restored_time_coefficients'],dtype=np.int64)
                Dtime=routes['packed_time'][0]
                Y=Dtime[:,codes[ps]].transpose(1,0,2).astype(np.int64)@W[hs].astype(np.int64).T
                assert np.array_equal(CSD_S,Y)
                record['real_restored_Y_values_checked']=int(Y.size)
            accum=np.zeros_like(u)-tau[:,None,hs]
            for t in range(10):
                for r in range(CSD_A.shape[1]):
                    for sign,shift in csd(int(CSD_A[t,r])):accum[t]+=sign*(CSD_S[:,r]<<shift)
            assert np.array_equal(accum,u-tau[:,None,hs])
            record['real_U_and_gate_values_checked']=int(u.size)
            record['real_U_mismatches']=0
        return record
    tasks=[(variant,cap,P,route) for variant,cap in captures for P in args.p for route in args.routes]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,item) for item in tasks]
        for future in as_completed(futures):
            record=future.result();out['records'].append(record)
            out['measured_capture_count']=len({(r['variant'],r['capture']) for r in out['records']})
            (HERE/args.output).write_text(json.dumps(out,indent=2)+'\n')
            print(json.dumps({k:record[k] for k in ('variant','capture','P','route','beats','selected','local_w_reads','state_writes','wall_seconds')}),flush=True)
    out['records'].sort(key=lambda r:(r['variant'],r['capture'],r['P'],r['route']))
    out['summary']={}
    for variant in args.variants:
        out['summary'][variant]={}
        for P in args.p:
            routes={}
            for route in args.routes:
                rows=[r for r in out['records'] if r['variant']==variant and r['P']==P and r['route']==route]
                routes[route]={k:sum(r[k] for r in rows) for k in ('beats','selected','local_w_reads','state_reads','state_writes','synchronization_wait','boundary_words','wall_seconds')}
                routes[route]['complete_capture_count']=len(rows)
            if all(x in routes for x in ('onehot7','time_rows')):
                routes['onehot7_to_time_rows_service_ratio']=routes['onehot7']['beats']/routes['time_rows']['beats']
            if all(x in routes for x in ('packed_class','packed_time')):
                routes['packed_class_to_packed_time_service_ratio']=routes['packed_class']['beats']/routes['packed_time']['beats']
            out['summary'][variant][str(P)]=routes
    finalize_metadata(out)
    (HERE/args.output).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
