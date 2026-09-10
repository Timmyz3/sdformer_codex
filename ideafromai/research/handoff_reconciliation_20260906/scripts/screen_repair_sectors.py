"""One authorized CPU increment: existing sample0/two layers/B32K4 only.

Numeric proxy reproduces GH float64 order. Service counters are a specified
logical replay schedule, never measured RTL latency or frozen FP equivalence.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
from pathlib import Path
import argparse
import json
import math
import resource
import struct
import sys
import time
import zlib
import numpy as np

BASE = Path(__file__).resolve().parents[1]
GH = BASE.parent/'mechanism_rebuild_gh_20260906'
HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
CAP = HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
CKPT = HW/'system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth'
sys.path.insert(0, str(GH/'scripts'))
sys.dont_write_bytecode = True
from checkpoint_numpy import read_checkpoint
HEADER = struct.Struct('<8sHH11I')
B, K, T = 32, 4, 10


def load_sources():
    """Same selected-frame CRC/order/code checks, without full-file rehash."""
    specs = {r['layer_id']: r for r in json.loads((CAP/'layers.json').read_text())['layers']}
    selected = {k:v for k,v in specs.items() if any(
        f'layers.{stage}.swin_blocks.0.mlp.fc' in v['module_name'] for stage in (0,3))}
    assert len(selected) == 4
    arrays = {k:[] for k in selected}
    counts = dict.fromkeys(selected, 0)
    frames = dict.fromkeys(selected, 0)
    with (CAP/'fc_frames.bin').open('rb') as f:
        while True:
            raw_header = f.read(HEADER.size)
            assert len(raw_header) == HEADER.size
            magic,version,hs,lid,sid,fi,start,n,C,br,nnz,rb,cb,crc = HEADER.unpack(raw_header)
            assert magic == b'M1558F01' and version == 1 and hs == HEADER.size
            if sid > 0:
                break
            assert sid == 0 and lid in specs
            if lid not in selected:
                f.seek(cb, 1)
                continue
            assert fi == frames[lid] and start == counts[lid]
            assert C == selected[lid]['input_channels'] and br == (C+7)//8
            dec = zlib.decompressobj()
            raw = dec.decompress(f.read(cb))+dec.flush()
            assert dec.eof and not dec.unused_data and not dec.unconsumed_tail
            assert len(raw) == rb and zlib.crc32(raw)&0xffffffff == crc
            mb = n*br
            assert rb == 3*mb+2*n+nnz and not any(raw[mb:3*mb])
            bits = np.unpackbits(np.frombuffer(raw[:mb],dtype=np.uint8).reshape(n,br),axis=1,bitorder='little')
            assert not bits[:,C:].any()
            bits = bits[:,:C]
            row_counts = np.frombuffer(raw[3*mb:3*mb+2*n],dtype='<u2')
            assert np.array_equal(row_counts,bits.sum(1)) and int(row_counts.sum()) == nnz
            assert np.all(np.frombuffer(raw[3*mb+2*n:],dtype=np.int8) == 1)
            arrays[lid].append(bits)
            counts[lid] += n
            frames[lid] += 1
    out = {}
    for lid,spec in selected.items():
        value = np.concatenate(arrays[lid],axis=0)
        assert value.shape == (spec['tokens_per_call'],spec['input_channels'])
        out[spec['module_name']] = (spec,value)
    return out, {str(k):frames[k] for k in selected}


def numeric_layer(stage, state, source, expected, out):
    pre = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.'
    spec,S = source[pre+'fc1']
    _,archived = source[pre+'fc2']
    N,C,H = len(S),S.shape[1],spec['output_channels']
    P = N//T
    assert spec['input_shape'][0] == T
    W = state[pre+'fc1.weight'].astype(np.float64)
    theta_source = float(state[pre+'sn1.spiking_neuron.thresh'])
    gamma = state[pre+'bn1.norm_layer.weight'].astype(np.float64)
    beta = state[pre+'bn1.norm_layer.bias'].astype(np.float64)
    A = state[pre+'sn2.spiking_neuron.weight'].astype(np.float64)
    bias = state[pre+'sn2.spiking_neuron.bias'].astype(np.float64).reshape(T,1)
    center = state[pre+'sn2.spiking_neuron.center'].astype(np.float64).reshape(T,1)
    theta = float(state[pre+'sn2.spiking_neuron.thresh'])
    assert A.shape == (T,T) and W.shape == (H,C) and np.all(gamma != 0)
    R = A.sum(1).reshape(T,1)
    nb = (P+B-1)//B
    sizes = np.minimum(B,P-np.arange(nb)*B)
    per_position = S.reshape(T,P,C).sum((0,2),dtype=np.int64)
    tile_terms = np.add.reduceat(per_position,np.arange(nb)*B)
    failure = np.zeros((nb,H),dtype=bool)
    fp = changed_count = patched = mismatch_archive = mismatch_algebra = 0
    for lo in range(0,H,32):
        hi = min(H,lo+32)
        hc = hi-lo
        # Deliberately identical chunking/operations to original GH screen.
        Y = (S.astype(np.float64)@W[lo:hi].T*theta_source).reshape(T,P,hc)
        mu = Y.mean((0,1))
        var = ((Y-mu)**2).mean((0,1))
        d = np.sqrt(var+1e-5)
        U = np.einsum('ts,sph->tph',A,Y,optimize=True)
        tau = mu*R+d/gamma[lo:hi]*(theta+center-bias-beta[lo:hi]*R)
        direction = np.sign(gamma[lo:hi])
        V,final_tau = U*direction,tau*direction
        actual = V >= final_tau[:,None,:]
        normalized = gamma[lo:hi]*(Y-mu)/d+beta[lo:hi]
        original = np.einsum('ts,sph->tph',A,normalized,optimize=True)+bias[:,None,:]-center[:,None,:]
        mismatch_algebra += int(np.count_nonzero(actual != (original >= theta)))
        mismatch_archive += int(np.count_nonzero(actual != archived.reshape(T,P,H)[:,:,lo:hi]))
        cs,cq = np.cumsum(Y.sum(0),0),np.cumsum((Y*Y).sum(0),0)
        end = np.minimum((np.arange(nb)+1)*B,P)-1
        denom = ((end+1)*T)[:,None]
        pmu = cs[end]/denom
        pvar = np.maximum(cq[end]/denom-pmu*pmu,0)
        ptau = (pmu[None,:,:]*R[:,None,:]+np.sqrt(pvar+1e-5)[None,:,:]/gamma[lo:hi]
                *(theta+center[:,None,:]-bias[:,None,:]-beta[lo:hi]*R[:,None,:]))*direction
        padded = np.full((T,nb*B,hc),np.nan)
        padded[:,:P] = V
        packed = padded.reshape(T,nb,B,hc)
        valid = np.arange(B)[None,None,:,None] < sizes[None,:,None,None]
        predicted = packed >= ptau[:,:,None,:]
        truth = packed >= final_tau[:,None,None,:]
        changed = (predicted != truth)&valid
        distance = np.where(valid,np.abs(packed-ptau[:,:,None,:]),np.inf)
        order = np.argsort(distance,axis=2,kind='stable')
        by_rank = np.take_along_axis(changed,order,axis=2)
        fail = by_rank[:,:,K:,:].any(2)
        failure[:,lo:hi] = fail.any(0)
        fp += int(fail.sum())
        changed_count += int(changed.sum())
        patched += int((by_rank[:,:,:K,:].sum(2)*(~fail)).sum())
        if hi == H or hi % 512 == 0:
            print('NUMERIC',stage,hi,H,flush=True)
    group96 = failure.reshape(nb,H//96,96).any(2)
    counts = dict(failed_packets=fp,prediction_bit_changes=changed_count,certified_patch_bits=patched,
                  full_FC1_active_scalar_terms=int(S.sum(dtype=np.int64))*H,
                  replay_FC1_terms_by_h=int(np.dot(failure.sum(1,dtype=np.int64),tile_terms)),
                  replay_FC1_terms_fixed96=int(np.dot(group96.sum(1,dtype=np.int64)*96,tile_terms)),
                  full_PSN_scalar_terms=T*T*P*H,
                  replay_PSN_terms_by_h=int(np.dot(failure.sum(1,dtype=np.int64),sizes))*T*T,
                  replay_PSN_terms_fixed96=int(np.dot(group96.sum(1,dtype=np.int64)*96,sizes))*T*T)
    for key,value in counts.items():
        assert value == expected[key], (stage,key,value,expected[key])
    assert float(failure.mean()) == expected['any_T_failure_h_tile_fraction']
    assert float(group96.mean()) == expected['fixed96_any_T_h_group_failure_fraction']
    assert mismatch_archive == mismatch_algebra == 0
    maskfile = out/f'stage{stage}_B32K4_anyT_h.le.bitpack'
    np.packbits(failure,axis=1,bitorder='little').tofile(maskfile)
    assert maskfile.stat().st_size == nb*H//8
    # Compact trace sufficient to reproduce grouping and source demand counts.
    nc = np.zeros((nb,C),dtype=np.uint16)
    for b,n in enumerate(sizes):
        nc[b] = S.reshape(T,P,C)[:,b*B:b*B+int(n),:].sum((0,1),dtype=np.uint16)
    np.save(out/f'stage{stage}_source_active_positions_per_c.npy',nc,allow_pickle=False)
    return dict(stage=stage,N=N,P=P,C=C,H=H,T=T,B=B,K=K,
                source_theta=theta_source,source_frame_codes='all +1',BN_eps_assumption=1e-5,
                numeric_proxy_mismatches=0,archive_support_mismatches=0,
                exact_GH_counter_matches=counts,
                anyT_failure_bitmap=maskfile.name,bitmap_shape=[nb,H],bitmap_bytes=maskfile.stat().st_size,
                failed_h_tiles=int(failure.sum()),
                active_position_counts_file=f'stage{stage}_source_active_positions_per_c.npy'),failure,nc,sizes


def service(layer,failure,nc,sizes,out):
    """Literal finite-packet schedule counts, no free full-layer S cache."""
    C,H,P = layer['C'],layer['H'],layer['P']
    layer['source_backing_lifetime'] = dict(
        entire_domain_bits=layer['N']*C,entire_domain_packed_bytes=layer['N']*C//8,
        retain_write_128bit_words_if_not_already_stored=math.ceil(layer['N']*C/128),
        lifetime='Retain immutable S from first FC1 pass until last repair completes after final BN statistics seal.',
        placement='Required backing storage,not assumed on-chip. One-packet scratch is separate.',
        sharing='May reuse an already-resident upstream S tensor only with explicit capacity/lifetime/address contract;otherwise charge the full retain write and backing allocation. Fused recompute baseline may share the same retained S.',
        not_in_repair_ingress='First-pass retention traffic and backing capacity are separate from the repair-only source ingress below.')
    repaired_packets = np.flatnonzero(failure.any(1)).tolist()
    ingress_words = 0
    ranges = []
    for b in repaired_packets:
        n = int(sizes[b])
        # Original packed binary S address: row-major (t,p,c).
        packet_ranges = []
        for t in range(T):
            bit_start = (t*P+b*B)*C
            bit_end = bit_start+n*C
            first,last = bit_start//128,(bit_end+127)//128
            ingress_words += last-first
            packet_ranges.append([first,last])
        ranges.append(dict(packet=b,valid_spatial_positions=n,source_128bit_word_halfopen_ranges=packet_ranges))
    layer['source_packet_trace'] = ranges
    layer['source_ingress'] = dict(repaired_packets=len(repaired_packets),source_128bit_word_reads=ingress_words,
        source_bytes_transferred=16*ingress_words,whole_layer_source_cache=False,
        max_raw_packet_bytes=max((T*int(sizes[b])*C//8 for b in repaired_packets),default=0),
        max_transposed_padded_packet_bytes=max((C*math.ceil(T*int(sizes[b])/128)*16 for b in repaired_packets),default=0))
    rows = []
    total = layer['exact_GH_counter_matches']['full_FC1_active_scalar_terms']
    for q in (4,16,96):
        assert H%q == 0 and 96%q == 0
        groups = failure.reshape(len(sizes),H//q,q).any(2)
        packed = np.packbits(groups,axis=1,bitorder='little')
        packed.tofile(out/f"stage{layer['stage']}_q{q}_sector_mask.le.bitpack")
        wps = q//4  # FP32 x q, 128-bit words, no INT8 assumption.
        spatial_lanes = 96//q
        count = dict(sector_packet_count=0,replay_FC1_active_scalar_terms=0,PSN_scalar_terms=0,
            weight_coefficients_loaded_FP32=0,weight_128bit_word_reads=0,weight_bank_read_beats=0,
            source_scratch_column_128bit_reads=0,source_nonzero_summary_scan_128bit_reads=0,
            fixed_spatial_scan_issue_batches=0,compacted_spatial_issue_batches=0,
            selector_position_128bit_scan_words=0,repair_Y_scalar_init_or_zero=0,
            repair_Y_scalar_accumulator_updates=0,final_repaired_output_bits=0)
        bank_reads = np.zeros(8,dtype=np.int64)
        for b in repaired_packets:
            n = int(sizes[b]); bt=T*n
            active_cs = np.flatnonzero(nc[b])
            source_ones = int(nc[b].sum())
            colwords = math.ceil(bt/128)
            for s in np.flatnonzero(groups[b]):
                count['sector_packet_count'] += 1
                count['replay_FC1_active_scalar_terms'] += q*source_ones
                count['PSN_scalar_terms'] += q*n*T*T
                count['repair_Y_scalar_init_or_zero'] += q*bt
                count['repair_Y_scalar_accumulator_updates'] += q*source_ones
                count['final_repaired_output_bits'] += q*bt
                count['source_nonzero_summary_scan_128bit_reads'] += math.ceil(C/128)
                count['source_scratch_column_128bit_reads'] += len(active_cs)*colwords
                count['selector_position_128bit_scan_words'] += len(active_cs)*colwords
                count['fixed_spatial_scan_issue_batches'] += len(active_cs)*math.ceil(bt/spatial_lanes)
                count['compacted_spatial_issue_batches'] += int(((nc[b,active_cs].astype(np.int64)+spatial_lanes-1)//spatial_lanes).sum())
                for c in active_cs:
                    # Static s-major,c-major FP32 sector layout, interleaved 8 banks.
                    first = (int(s)*C+int(c))*wps
                    owners = (first+np.arange(wps))%8
                    loads = np.bincount(owners,minlength=8)
                    bank_reads += loads
                    count['weight_bank_read_beats'] += int(loads.max())
                    count['weight_128bit_word_reads'] += wps
                    count['weight_coefficients_loaded_FP32'] += q
        assert count['weight_128bit_word_reads']*4 == count['weight_coefficients_loaded_FP32']
        assert count['weight_128bit_word_reads'] == int(bank_reads.sum())
        if q == 96:
            assert count['replay_FC1_active_scalar_terms'] == layer['exact_GH_counter_matches']['replay_FC1_terms_fixed96']
            assert count['PSN_scalar_terms'] == layer['exact_GH_counter_matches']['replay_PSN_terms_fixed96']
        if q == 4:
            F=layer['exact_GH_counter_matches']['replay_FC1_terms_by_h']; G=layer['exact_GH_counter_matches']['replay_FC1_terms_fixed96']
            assert max(F,G/24) <= count['replay_FC1_active_scalar_terms'] <= min(4*F,G)
        rows.append(dict(q=q,spatial_lanes=spatial_lanes,FP32_weight_words_per_c=wps,
            weight_layout_payload_bytes=H*C*4,weight_layout_duplicate_copy=False,
            source_ingress_128bit_reads_same_packet_cache=ingress_words,
            max_Y_scalar_buffer=q*T*B,max_Y_buffer_bytes_if32bit=q*T*B*4,
            max_Y_buffer_bytes_if_float64_proxy=q*T*B*8,
            PSN_per_position_temporal_input_scalars=q*T,
            PSN_per_position_input_bytes_if32bit=q*T*4,
            PSN_output_accumulator_scalars=q,
            full_packet_U_copy_required=False,
            selector_scan_aliases_source_scratch_column_reads=True,
            PSN_storage_contract='After all Y complete and final stats seal,load one spatial position q*T Y into temporal registers,compute each t output with full signed A,compare final threshold immediately;do not overwrite Y until this position is consumed. No full U packet retained;PSN scalar work/read ports still charged separately.',
            weight_bank_word_reads=bank_reads.tolist(),**count,
            replay_FC1_fraction=count['replay_FC1_active_scalar_terms']/total,
            weight_bytes_transferred=count['weight_128bit_word_reads']*16,
            compacted_lane_useful_fraction=count['replay_FC1_active_scalar_terms']/(96*count['compacted_spatial_issue_batches']),
            fixed_scan_lane_useful_fraction=count['replay_FC1_active_scalar_terms']/(96*count['fixed_spatial_scan_issue_batches'])))
    layer['service'] = rows


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=BASE/'records/repair_sector_sample0_r1',
                        help='New result directory; existing directories are never overwritten')
    out = parser.parse_args().output_dir
    out.mkdir(parents=True,exist_ok=False)
    original = json.loads((GH/'records/threshold_packet_sample0_screen.json').read_text())
    plan = dict(status='PREDECLARED_BEFORE_EXECUTION',sample_ids=[0],stages=[0,3],B=32,K=4,q=[4,16,96],
        numeric='Identical GH float64 proxy,32h chunks,eps1e-5; not GPU FP32 order',
        weight_format='FP32,128bit word,q4/16/96=1/4/24 words per c; static layout,no duplication',
        resources='96 logical arithmetic lanes;8 weight banks,1 128bit read/bank/beat;one packet source buffer plus retained whole-domain bitpacked S backing;no inter-packet weight cache',
        scheduling='After final-statistics seal,tile-major then fixed contiguous qh-sector then c;all qh recomputed;zero source columns skipped using charged summary',
        no_claim='Counts are logical service-model events,not VCS clocks/energy/PPA;selector/transpose/PSN pipeline not timed',
        source_identity='Use same sealed paths and inherited hashes,check selected frame CRC/order/codes;no repeated full-file hash',
        BLAS_threads=4)
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    source,frames = load_sources()
    state = read_checkpoint(CKPT)['model_state_dict']
    layers = []
    for stage in (0,3):
        ex_layer = next(l for l in original['layers'] if f'layers.{stage}.' in l['module'])
        ex = next(r for r in ex_layer['rows'] if (r['B'],r['K']) == (B,K))
        layer,failure,nc,sizes = numeric_layer(stage,state,source,ex,out)
        service(layer,failure,nc,sizes,out)
        layers.append(layer)
        print('SERVICE_DONE',stage,json.dumps(layer['service']),flush=True)
    result = dict(status='PASS_GH_B32K4_EXACT_COUNTER_REPRODUCTION_AND_CPU_SERVICE_ENUMERATION',
        plan=plan,selected_frame_counts=frames,source_hashes_inherited_not_recomputed=original['source_hashes'],
        checkpoint_sha256_inherited_not_recomputed=original['checkpoint_sha256'],layers=layers,
        wall_seconds=time.time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations=['Only existing sample0/two layers;not distribution or accuracy evidence.',
          'Y/U numerical audit remains float64;FP32 weights are layout payload only,not proof32bit replay arithmetic preserves frozen output.',
          'External source ingress is once per repair packet;all source scratch rescans and nonzero summary reads counted.',
          'Row-to-column transpose formation and memory allocation beyond bare packet are unresolved;source column padding explicitly counted.',
          'Compacted arithmetic batches require320-bit source-position selector with24/6/1 emitted destinations per batch;128bit selector scan reads counted but logic delay/area not measured.',
          'PSN terms,patch bits and Y updates are separate counts;no free PSN throughput or commit.',
          'Weight bank beats are isolated per(c,sector) transfers with1RW no writes;no overlap assumed,no bank scheduler claimed.',
          'No end-to-end cycle reported;CPU runtime is research runtime only.'],
        PPA_ADMISSION=0,RTL_SPEEDUP_ADMISSION=0,FROZEN_FP_EQUIVALENCE=0,AEE_ADMISSION=0)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('DONE',result['wall_seconds'],result['peak_rss_kib'],str(out/'result.json'),flush=True)


if __name__ == '__main__':
    main()
