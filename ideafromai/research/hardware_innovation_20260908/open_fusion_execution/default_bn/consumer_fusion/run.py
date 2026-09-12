"""Complete normalize+real-PED join, starting with computed BN statistics.

The full native producer and BN statistics service are outside this suffix.
Both old students and one fixed pressure trace use the same original pairwise
FP32 statistics, actual packed24 PED, and source-proven full-K code0 tags.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location('code0_prepare',PARENT/'run.py')
code0=importlib.util.module_from_spec(spec);spec.loader.exec_module(code0)
MODES={1:'materialized_BN_then_PED',2:'ordinary_normalize_PED_fusion',3:'default_reference_join'}


def prepare(axis,work,condition):
    source=code0.prepare(axis,work)
    folder=code0.FULL/'capture_full_producers'/axis
    with np.load(folder/'000_zurich_city_09_a_0001.npz') as z:
        q=z['full_continuous_q24']
        cuda_norm=z['proj_bn_full_output_fp32']
        geometry=json.loads(str(z['window_geometry_json']))
        local={label:z[label+'_ped_output_fp32'] for label in ['corner','interior']}
    assert q.dtype.kind in 'iu' and q.min()>=-2**23 and q.max()<2**23
    # Preserve the declared existing producer spill layout: spatial,P/T/C.
    spatial=q.transpose(2,3,0,1).copy().reshape(-1)
    bits=spatial.astype(np.uint32)&0xffffff
    packed=np.stack([bits&255,(bits>>8)&255,(bits>>16)&255],axis=1).astype(np.uint8)
    packed.tofile(work/'PED_spatial_T_C.i24')
    fp=np.float32(q)*np.float32(2**-14)
    # Signed24 fits exactly in FP32; scaling by2^-14 is also exact.
    assert np.array_equal(fp.astype(np.float64),q.astype(np.float64)*2**-14)
    cuda_final=np.float32(cuda_norm+fp)
    checks={}
    for label,geo in geometry.items():
        if label not in local:continue
        y,x=geo['output_origin'];h,w=geo['output_shape']
        actual=cuda_final[:,:,y:y+h,x:x+w]
        diff=int(np.count_nonzero(actual.view(np.uint32)!=local[label].view(np.uint32)))
        assert diff==0,(axis,label,diff)
        checks[label]=dict(values=int(actual.size),bit_differences=diff)
    prior=Path('/tmp/default_bn_20260912')/condition/axis
    stats=np.fromfile(prior/'dense_stats.f32','<f4').reshape(3,96)
    with np.load(folder/'live_parameters.npz') as z:
        coeff=np.concatenate([stats[0],stats[2],z['proj_bn_gamma'],z['proj_bn_beta'],np.full(8,2**-14)]).astype('<f4')
    coeff.tofile(work/'join_coeff.f32')
    norm=np.memmap(prior/'dense_output.f32',dtype='<f4',mode='r').reshape(192000,96)
    ped=fp.transpose(0,2,3,1).reshape(192000,96)
    reference=np.float32(norm+ped)
    reference.tofile(work/'reference_final.f32')
    delta=reference-cuda_final.transpose(0,2,3,1).reshape(192000,96)
    return dict(source=source,statistics_from=str(prior/'dense_stats.f32'),
        statistics_service_included=False,stats='Already computed by original pairwise FP32 tree; no new statistic algorithm.',
        actual_PED_format='signed24/f14; packed spatial,T10,C96,2880B per spatial position; BN order T10,spatial,C96.',
        PED_input_bytes=int(packed.nbytes),PED_q24_range=[int(q.min()),int(q.max())],
        conversion='signed24 to FP32 exactly, then charged multiplication by2^-14; final ADD(normalized,PED) in FP32.',
        existing_helper_local_final_checks=checks,
        reference_final_vs_CUDA_BN_plus_actual_PED=dict(max_abs=float(np.max(np.abs(delta))),
            differences=int(np.count_nonzero(delta)),AEE_revalidated=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');args=ap.parse_args()
    condition='stress' if args.stress else 'ready'
    work=Path('/tmp/default_bn_consumer_fusion_20260912')/condition;work.mkdir(parents=True,exist_ok=True)
    (work/'global_bn_engine.hpp').write_text((code0.OLD/'global_bn.cpp').read_text().split('\nint main(')[0]+'\n')
    (work/'default_bn_stream.hpp').write_text((PARENT/'default_bn.cpp').read_text().split('\nint main(')[0]+'\n')
    binary=work/'consumer_fusion'
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I',str(work),str(HERE/'consumer_fusion.cpp'),'-o',str(binary)],check=True)
    result=dict(scope=__doc__,evidence='Complete-domain CPU FP32 payload suffix model; not RTL, PPA, new AEE or full-network time.',
        stress=args.stress,axes={},native_producer_service_included=False,BN_statistics_service_included=False,
        strong_control='Ordinary normalization+PED fusion receives the same code0, default-vector and rounding privileges.',
        candidate_identity='Deferred default reference at the consumer is already expressible by arm2; mode3 intentionally executes the same legal function to test/record no extra X.',
        resources=dict(state_bytes=131072,coefficient_bytes=131072,RF_vectors=96,lanes=8,
            state_ports='1R64/1W64',coefficient_port='1R256',FP_latency=4,
            normalized_position_RF_vectors=list(range(12)),scale_RF_vectors=list(range(12,24)),
            bias_RF_vectors=list(range(24,36)),default_RF_vectors=list(range(48,60)),tag_RF_vectors=[88,89,90,91],
            tag_state_bytes=32,PED_state_bytes=288,PED_state_address=16384),
        PED_decode_conversion_cost='Each H8 consumes3 real SR64 responses, one paid vector packed24 decode, one latency4 signed24-to-FP32 RF operation and one latency4 power-of-two multiply.',
        DRAM_contract='32B per5 transfer slots as prior model; address selection charged. Spatial-T10 PED input is traversed with its real2880B stride; no new DRAM row-buffer model.',
        no_binary_gate_created_from_default=True)
    path=HERE/f'results_{condition}.json'
    for axis in (['ordinary'] if args.stress else ['ordinary','lifting_raw']):
        awork=work/axis;awork.mkdir(exist_ok=True)
        record=dict(input=prepare(axis,awork,condition),arms={})
        for mode,name in MODES.items():
            intermediate=awork/f'{name}_intermediate.f32';output=awork/f'{name}_final.f32'
            print(axis,name,'start',flush=True);begin=time.monotonic()
            proc=subprocess.run([str(binary),str(awork/'live.f32'),str(awork/'tags.u8'),str(awork/'join_coeff.f32'),
                str(awork/'PED_spatial_T_C.i24'),str(intermediate),str(output),str(int(args.stress)),str(mode)],capture_output=True,text=True,check=True)
            r=json.loads(proc.stdout);r['wall_seconds']=time.monotonic()-begin
            r['strict_final_bits']=code0.bit_check(output,awork/'reference_final.f32')
            if mode==1:
                prior=Path('/tmp/default_bn_20260912')/condition/axis
                r['strict_intermediate_BN_bits']=code0.bit_check(intermediate,prior/'dense_output.f32')
            r['phase_service']=dict(zip(['setup_affine','raw_tag_normalization','PED_input_conversion','final_add_and_output','intermediate_materialize_reload'],r['stages']))
            assert sum(r['stages'])==r['service_slots']
            r['physical_port_bytes']=dict(state_read=r['SR64_reads']*8,state_write=r['SW64_writes']*8,
                coefficient_read=r['CR256_reads']*32,coefficient_fill=r['CW256_writes']*32)
            r['external_read_bytes']=r['raw_read_bytes']+r['PED_packed24_read_bytes']+r['tag_read_bytes']+r['intermediate_BN_read_bytes']
            r['external_write_bytes']=r['intermediate_BN_write_bytes']+r['final_write_bytes']
            record['arms'][name]=r
            for arm in record['arms'].values():
                arm['service_reduction_vs_materialized']=1-arm['service_slots']/record['arms'][MODES[1]]['service_slots']
                if MODES[2] in record['arms']:
                    arm['service_reduction_vs_ordinary_fusion']=1-arm['service_slots']/record['arms'][MODES[2]]['service_slots']
            if mode==3:
                compared=record['arms'][MODES[2]]
                counters=['service_slots','stages','SR64_reads','SW64_writes','ALU_vector_issues','DMA_slots',
                    'raw_read_bytes','PED_packed24_read_bytes','tag_read_bytes','final_ADD_vector_issues','external_read_bytes','external_write_bytes']
                assert all(r[k]==compared[k] for k in counters)
                record['arm2_arm3_equal_counters']=counters
                record['arm2_arm3_same_output']=code0.bit_check(output,awork/(MODES[2]+'_final.f32'))
            result['axes'][axis]=record
            path.write_text(json.dumps(result,indent=2)+'\n')
            print(axis,name,r['service_slots'],r['service_reduction_vs_materialized'],'BITS_EQUAL',flush=True)


if __name__=='__main__':main()
