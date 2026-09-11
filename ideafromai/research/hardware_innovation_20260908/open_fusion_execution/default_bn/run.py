"""Complete-domain code0 BN with strict comparisons to the existing FP32 tree.

This independently executes BN on captured projection payloads and actual
full-K producer-derived tags. Producer replay, PED and later consumers are
outside its cost scope. Every normalized output is still materialized.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
FULL=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
OLD=FULL/'preview_sn2_chain'
MODES={0:'dense',1:'tag_same_arithmetic',2:'default_ops',3:'zero_block_replay'}


def bit_check(path,reference):
    x=np.memmap(path,dtype='<u4',mode='r')
    y=np.memmap(reference,dtype='<u4',mode='r')
    assert x.shape==y.shape
    mismatches=int(np.count_nonzero(x!=y))
    assert mismatches==0,(str(path),str(reference),mismatches)
    return dict(values=int(x.size),uint32_bit_differences=mismatches)


def prepare(axis,work):
    path=FULL/'capture_full_producers'/axis
    with np.load(path/'000_zurich_city_09_a_0001.npz') as z:
        words=z['full_proj_words']
        raw=z['proj_bn_full_input_fp32'].transpose(0,2,3,1).reshape(192000,96).copy()
    with np.load(path/'live_parameters.npz') as z:
        assert not bool(z['proj_has_bias'])
        coeff=np.concatenate([z['proj_bn_gamma'],z['proj_bn_beta'],np.full(8,1/192000),
            np.full(8,.5),np.full(8,float(z['proj_bn_eps']))]).astype('<f4')
    padded=np.pad(words,((0,0),(1,1),(1,1)))
    support=np.zeros((120,160),np.uint16)
    for ky in range(3):
        for kx in range(3):
            support|=np.bitwise_or.reduce(padded[:,ky:ky+240:2,kx:kx+320:2],axis=0)
    zero=np.stack([((support>>t)&1)==0 for t in range(10)]).reshape(-1)
    raw_bits=raw.view(np.uint32)
    # Tag is generated from the real COMPLETE K864 input, not raw==0.
    assert np.all(raw_bits[zero]==0), 'Producer tag did not prove bitwise +0.'
    assert np.array_equal(zero,np.all(raw==0,axis=1))
    live=raw[~zero]
    tags=np.packbits(zero,bitorder='little')
    raw.astype('<f4').tofile(work/'dense.f32')
    live.astype('<f4').tofile(work/'live.f32')
    tags.tofile(work/'tags.u8')
    coeff.tofile(work/'coeff.f32')
    return dict(domain=[10,96,120,160],logical_positions=192000,values_per_channel=192000,
        complete_K=864,tag_source='OR of actual full projection gate T10 words over all96 channels and all9 kernel offsets, including image edges.',
        zero_vectors=int(zero.sum()),live_vectors=int((~zero).sum()),
        all_tagged_raw_bits_are_positive_zero=True,observed_zeros_without_tag=0,
        original_dense_bytes=int(raw.nbytes),packed_live_bytes=int(live.nbytes),tag_bytes_per_pass=int(tags.nbytes),
        complete_zero256_leaves=int(zero.reshape(-1,256).all(1).sum()),total256_leaves=750,
        producer_tag_formation_service='Outside this BN-only scope; original native full-K directory already supplies empty status, but full-frame producer has not been replayed.',
        packer='Host serializes an explicit candidate input format; no compression-throughput or whole-producer claim.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--modes',nargs='+',type=int,choices=list(MODES),default=[0,1,2])
    ap.add_argument('--stress',action='store_true')
    args=ap.parse_args()
    work=Path('/tmp/default_bn_20260912')/('stress' if args.stress else 'ready')
    work.mkdir(parents=True,exist_ok=True)
    # Import the actual existing machine implementation, not a faster substitute.
    prefix=(OLD/'global_bn.cpp').read_text().split('\nint main(')[0]
    (work/'global_bn_engine.hpp').write_text(prefix+'\n')
    binary=work/'default_bn'
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I',str(work),
                    str(HERE/'default_bn.cpp'),'-o',str(binary)],check=True)
    path=HERE/('results_stress.json' if args.stress else 'results_ready.json')
    result=json.loads(path.read_text()) if path.exists() else dict(scope=__doc__,axes={})
    result.update(stress=args.stress,evidence='Actual CPU payload FP32 operator/port model, not RTL, PPA, CUDA bit-equivalence or AEE.',
        arithmetic='Original FP32 four-striped256-leaf tree, centered two-pass variance, original rsqrt seed and3 Newton iterations; no multiplicity-times-square substitution.',
        resources=dict(state_bytes=131072,coefficient_bytes=131072,RF_vectors=96,lanes=8,
            state_ports='1R64/1W64',coefficient_port='1R256',integer_seed_latency=2,FP_latency=4,
            tree_bytes=3840,tag_state_staging_bytes=32,tag_RF_vectors=[88,89,90,91],
            default_normalized_RF_vectors=list(range(48,60)),positive_zero_RF_vector=95,
            zero_block_cache_bytes=384,zero_block_cache_address=12288),
        tag_decode_contract='One charged scalar bit extraction per logical position; four charged packed-byte unpack/RF LOADs per256 positions from actual SR64 responses. Optional all-code0 leaf test costs four control slots.',
        outputs='Every arm writes all73728000 normalized FP32 output bytes; no free PED/default consumer or reduced final output.',
        full_producer_or_PED_replayed=False,new_training=False,new_quantization=False,new_AEE=False)
    axes=['ordinary'] if args.stress else ['ordinary','lifting_raw']
    old_result=json.loads((OLD/('global_bn_pairwise_stress.json' if args.stress else 'global_bn_pairwise.json')).read_text())
    for axis in axes:
        awork=work/axis;awork.mkdir(exist_ok=True)
        source=prepare(axis,awork)
        record=result['axes'].setdefault(axis,dict(source=source,arms={}))
        record['source']=source
        for mode in args.modes:
            name=MODES[mode]
            output=awork/f'{name}_output.f32';stats=awork/f'{name}_stats.f32'
            command=[str(binary),str(awork/'dense.f32'),str(awork/'coeff.f32'),str(awork/'tags.u8'),
                     str(awork/'live.f32'),str(output),str(stats),str(int(args.stress)),str(mode)]
            print(axis,name,'start',flush=True);begin=time.monotonic()
            proc=subprocess.run(command,text=True,capture_output=True,check=True)
            report=json.loads(proc.stdout);report['wall_seconds']=time.monotonic()-begin
            report['variance_uses_shared_negative_mu']=mode>=2
            dense_output=awork/'dense_output.f32';dense_stats=awork/'dense_stats.f32'
            if mode==0:
                prior=Path('/tmp/tcasii_global_bn_work_20260911')/('pairwise_stress' if args.stress else 'pairwise_ready')
                report['dense_reproduction']=dict(
                    output=bit_check(output,prior/(axis+'_output.f32')),
                    statistics=bit_check(stats,prior/(axis+'_stats.f32')),
                    previous_service=old_result['axes'][axis]['service_slots'])
                assert report['service_slots']==old_result['axes'][axis]['service_slots']
            report['strict_bit_checks']=dict(normalized=bit_check(output,dense_output),statistics=bit_check(stats,dense_stats))
            report['physical_port_bytes']=dict(state_read=report['SR64_reads']*8,state_write=report['SW64_writes']*8,
                coefficient_read=report['CR256_reads']*32,coefficient_fill=report['CW256_writes']*32)
            report['external_read_bytes']=report['active_payload_read_bytes']+report['tag_read_bytes']
            report['external_output_bytes']=73728000
            report['dense_service_reduction']=1-report['service_slots']/old_result['axes'][axis]['service_slots']
            report['stage_names']=['setup','mean','variance','rsqrt_affine_default','normalize_full_materialization']
            assert sum(report['stages'])==report['service_slots']
            record['arms'][name]=report
            for arm in record['arms'].values():
                if 'tag_same_arithmetic' in record['arms']:
                    arm['service_reduction_vs_tag_same_arithmetic']=1-arm['service_slots']/record['arms']['tag_same_arithmetic']['service_slots']
                if 'default_ops' in record['arms']:
                    arm['service_reduction_vs_default_ops']=1-arm['service_slots']/record['arms']['default_ops']['service_slots']
            path.write_text(json.dumps(result,indent=2)+'\n')
            print(axis,name,report['service_slots'],report['dense_service_reduction'],'BITS_EQUAL',flush=True)


if __name__=='__main__':main()
