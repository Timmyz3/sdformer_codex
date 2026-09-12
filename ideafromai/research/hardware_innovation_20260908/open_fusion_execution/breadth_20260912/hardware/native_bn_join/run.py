from pathlib import Path
import argparse
import ctypes
import json
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[2]
BASE=OPEN.parent
DEFAULT=OPEN/'default_bn'
FULL=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
EXPORT=OPEN/'stage_20260912/algorithm/hardware_exports/ordinary'
WORK=Path('/tmp/native_bn_join_20260912')


def build():
    (WORK/'global_bn_engine.hpp').write_text((FULL/'preview_sn2_chain/global_bn.cpp').read_text().split('\nint main(')[0]+'\n')
    (WORK/'default_stream.hpp').write_text((DEFAULT/'default_bn.cpp').read_text().split('\nint main(')[0]+'\n')
    (WORK/'onepass_core.hpp').write_text((DEFAULT/'onepass/onepass.cpp').read_text().split('\nint main(')[0]+'\n')
    join=(DEFAULT/'consumer_fusion/consumer_fusion.cpp').read_text().split('\nint main(')[0]
    join=join.replace('#include "default_bn_stream.hpp"\n','').replace('constexpr int PED_STATE=16384;','constexpr int PED_STATE=24576;')
    (WORK/'consumer_join.hpp').write_text(join+'\n')
    binary=WORK/'pipeline'
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-I',str(WORK),
                    str(HERE/'pipeline.cpp'),'-o',str(binary)],check=True)
    math=WORK/'math.so'
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-shared','-fPIC',
        str(OPEN/'stage_20260912/algorithm/onepass_math/numeric.cpp'),'-o',str(math)],check=True)
    return binary,math


def prepare(capture=None):
    data=np.load(capture if capture is not None else WORK/'capture/ordinary.npz')
    live=dict(np.load(EXPORT/'live_parameters.npz'))
    assert float(live['proj_theta_output'])==1.0 and not bool(live['proj_has_bias'])
    assert np.array_equal(live['proj_stride'],[2,2]) and np.array_equal(live['proj_padding'],[1,1])
    assert np.array_equal(live['proj_dilation'],[1,1]) and int(live['proj_groups'])==1
    words=data['full_proj_words'];assert words.shape==(96,240,320) and words.max()<1024
    words.transpose(1,2,0).copy().astype('<u2').tofile(WORK/'gates_HWC.u16')
    weight=live['proj_weight_fp32'].reshape(96,864).copy()
    u=weight.view(np.uint32);weight=((u+np.uint32(0xfff)+((u>>13)&1))&np.uint32(0xffffe000)).view(np.float32)
    assert np.all(np.isfinite(weight));weight.astype('<f4').tofile(WORK/'weights_TF32.f32')
    coef=np.concatenate([live['proj_bn_gamma'],live['proj_bn_beta'],np.full(8,1/192000),
        np.full(8,.5),np.full(8,float(live['proj_bn_eps'])),np.full(8,2**-14)]).astype('<f4')
    coef.tofile(WORK/'bn_coeff.f32')
    ped=data['full_continuous_q24'];assert ped.shape==(10,96,120,160) and ped.min()>=-2**23 and ped.max()<2**23
    bits=ped.transpose(2,3,0,1).copy().reshape(-1).astype(np.uint32)&0xffffff
    np.stack([bits&255,(bits>>8)&255,(bits>>16)&255],1).astype(np.uint8).tofile(WORK/'PED.i24')
    return data,live,ped


def difference(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32);assert a.shape==b.shape
    return dict(values=int(a.size),uint32_bit_differences=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),
        max_abs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture',type=Path,default=WORK/'capture/ordinary.npz')
    args=ap.parse_args()
    WORK.mkdir(exist_ok=True);data,live,ped=prepare(args.capture);binary,math=build()
    print('FULL_NATIVE_START',flush=True);start=time.monotonic()
    with (WORK/'execution.json').open('w') as output:
        subprocess.run([str(binary),str(WORK/'gates_HWC.u16'),str(WORK/'weights_TF32.f32'),
            str(WORK/'bn_coeff.f32'),str(WORK/'PED.i24'),str(WORK)],stdout=output,check=True)
    result=json.loads((WORK/'execution.json').read_text());result['wall_seconds']=time.monotonic()-start
    raw=np.fromfile(WORK/'native_raw.f32','<f4').reshape(192000,96)
    lib=ctypes.CDLL(str(math));ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.moments.argtypes=[ptr,ctypes.c_int,ptr,ptr,ctypes.c_float,ptr];lib.moments.restype=ctypes.c_int
    lib.normalize.argtypes=[ptr,ctypes.c_int,ptr,ptr];lib.normalize.restype=None
    stats=np.empty((5,96),np.float32)
    assert lib.moments(raw,192000,np.ascontiguousarray(live['proj_bn_gamma']),np.ascontiguousarray(live['proj_bn_beta']),float(live['proj_bn_eps']),stats)==0
    normalized=np.empty_like(raw);lib.normalize(raw,192000,stats,normalized)
    computed=np.fromfile(WORK/'statistics.f32','<f4').reshape(5,96)
    result['computed_statistics_check']=difference(computed,stats);assert result['computed_statistics_check']['uint32_bit_differences']==0
    fullped=np.float32(ped.transpose(0,2,3,1).reshape(192000,96))*np.float32(2**-14)
    reference=np.float32(normalized+fullped)
    for arm in result['arms']:
        y=np.fromfile(WORK/(arm['mode']+'_final.f32'),'<f4').reshape(192000,96)
        arm['independent_final_check']=difference(y,reference);assert arm['independent_final_check']['uint32_bit_differences']==0
        arm['physical_port_bytes']=dict(SR64=8*arm['SR64_reads'],SW64=8*arm['SW64_writes'],CR256=32*arm['CR256_reads'],CW256=32*arm['CW256_writes'])
        arm['external_read_bytes']=sum(arm[k] for k in ('moment_read_bytes','normalize_read_bytes','PED_read_bytes','intermediate_BN_read_bytes'))+result['native']['gate_read_bytes']+result['native']['weight_fill_bytes']+896
        arm['external_write_bytes']=result['native']['raw_spill_bytes']+arm['intermediate_BN_write_bytes']+arm['final_write_bytes']
        arm['service_reduction_vs_materialized']=1-arm['service_slots']/result['arms'][0]['service_slots']
    mid=np.fromfile(WORK/'materialized_BN.f32','<f4').reshape(192000,96)
    result['materialized_normalized_check']=difference(mid,normalized);assert result['materialized_normalized_check']['uint32_bit_differences']==0
    gpu=data['proj_bn_full_input_fp32'].transpose(0,2,3,1).reshape(192000,96)
    result['native_vs_captured_GPU_raw']=difference(raw,gpu)
    gpu_bn=data['proj_bn_full_output_fp32'].transpose(0,2,3,1).reshape(192000,96)
    result['native_BN_vs_captured_onepass_output']=difference(normalized,gpu_bn)
    result['native_join_vs_captured_onepass_plus_same_PED']=difference(reference,np.float32(gpu_bn+fullped))
    result.update(scope='Actual ordinary currentR24+onepass complete gate/PED external input -> native whole frame -> actual onepass statistics -> normalized+PED output, one Engine timeline.',
        input_capture='stage_20260912/algorithm/captures/ordinary/000_zurich_city_09_a_0001.npz from A800; already-existing current combination, no new GPU capture.',
        evidence='CPU complete-domain payload/port prototype, not RTL/PPA/AEE or complete I24-to-network execution.',
        new_ISA=False,new_X=False,profile='ready',domain=[10,96,120,160],
        native_function='Full K864 ascending original channel/ky/kx; TF32-rounded real W; original gate/ theta1; Engine ADD vs independent fmaf(1,W,acc). Not CUDA reduction order.',
        resource=dict(RF_vectors=96,lanes=8,state_capacity=131072,coefficient_capacity=131072,
            ports='SR64/SW64/CR256',external_transfer='existing32B/5slots, no additional DRAM latency model',
            native_state=dict(directories=[0,55296],headers=[56000,56064],gates=[65536,81088],output_staging=[32,64]),
            native_RF='0..79 accumulator after directory phase; RF64..72 H4 source cache during directory; RF80 coefficient.',
            BN_tree=[16384,24064],PED_staging=[24576,24864],state_high_water=81088,
            coefficient_high_water=110592,external_raw_buffer=73728000,external_PED_payload=55296000),
        limits=['Full upstream producer service and gate packing excluded by explicit external input boundary.',
            'Dense BN payload baseline; no source-code0 format, cross-tile directory reuse or best compiler claim.',
            'Two suffixes fork the complete actual native+statistics Machine state, not separately timed historical tables.',
            'FP32 native differs from GPU reduction; do not inherit existing825 quality.'],
        phase_names=['native','BN_coefficients','onepass_moments','variance_rsqrt_affine','normalization',
            'PED_external_input','PED_conversion','join_output','BN_intermediate_write','BN_intermediate_read'])
    (HERE/'results_ready.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('native_vs_captured_GPU_raw','computed_statistics_check','wall_seconds')},indent=2),flush=True)
    print([(r['mode'],r['service_slots'],r['service_reduction_vs_materialized']) for r in result['arms']],flush=True)
    print('FULL_NATIVE_BN_JOIN_COMPLETE',flush=True)


if __name__=='__main__':main()
