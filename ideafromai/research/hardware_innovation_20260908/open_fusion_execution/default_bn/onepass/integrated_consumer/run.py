"""Single Engine from real one-pass BN statistics to normalized+actual PED.

BNFF/FlexAcc statistics plus default propagation and ordinary consumer fusion
are the shared baseline (X=0). Prior output files are numerical references
only: neither statistics values nor separately measured cycles feed execution.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ONEPASS = HERE.parent
DEFAULT = ONEPASS.parent
spec = importlib.util.spec_from_file_location('code0_prepare', DEFAULT/'run.py')
code0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code0)
MODES = {1: 'onepass_materialized_BN_then_PED', 2: 'onepass_ordinary_normalize_PED_fusion'}
STAGES = ['setup', 'joint_moments', 'variance_rsqrt_affine_scale_switch',
          'normalization_with_code0', 'PED_input_conversion', 'final_add_output',
          'intermediate_BN_materialize_reload']


def prepare(axis, work, profile):
    source = code0.prepare(axis, work)
    coeff = np.fromfile(work/'coeff.f32', '<f4')
    assert coeff.size == 216
    np.concatenate([coeff, np.full(8, 2**-14, np.float32)]).astype('<f4').tofile(work/'integrated_coeff.f32')
    folder = code0.FULL/'capture_full_producers'/axis
    with np.load(folder/'000_zurich_city_09_a_0001.npz') as z:
        q = z['full_continuous_q24']
    assert q.dtype.kind in 'iu' and q.min() >= -2**23 and q.max() < 2**23
    spatial = q.transpose(2, 3, 0, 1).copy().reshape(-1)
    bits = spatial.astype(np.uint32) & 0xffffff
    packed = np.stack([bits & 255, (bits >> 8) & 255, (bits >> 16) & 255], axis=1).astype(np.uint8)
    packed.tofile(work/'PED_spatial_T_C.i24')
    fp = np.float32(q)*np.float32(2**-14)
    assert np.array_equal(fp.astype(np.float64), q.astype(np.float64)*2**-14)
    old = Path('/tmp/flexacc_onepass_20260912')/profile/axis
    normalized = np.memmap(old/'tagged_default_output.f32', dtype='<f4', mode='r').reshape(192000, 96)
    ped = fp.transpose(0, 2, 3, 1).reshape(192000, 96)
    np.float32(normalized+ped).tofile(work/'reference_final.f32')
    return dict(source=source,
        PED_input_bytes=int(packed.nbytes), PED_q24_range=[int(q.min()), int(q.max())],
        PED_format='signed24/f14; actual spatial,T10,C96 packed24; BN traverses T10,spatial,C96.',
        reference='Prior same-formula onepass/tagged BN output plus actual q24*2^-14, separate FP32 ADD.',
        reference_BN_output=str(old/'tagged_default_output.f32'),
        reference_BN_statistics=str(old/'tagged_default_stats.f32'),
        execution_has_no_statistics_input=True,
        producer_scope='Starts at captured complete native projection payload and complete-K source-derived code0; producer replay/packing/tag formation service excluded.')


def build(work):
    # These temporary imports retain the common Engine and original arithmetic.
    (work/'global_bn_engine.hpp').write_text((code0.OLD/'global_bn.cpp').read_text().split('\nint main(')[0]+'\n')
    (work/'default_stream.hpp').write_text((DEFAULT/'default_bn.cpp').read_text().split('\nint main(')[0]+'\n')
    (work/'onepass_core.hpp').write_text((ONEPASS/'onepass.cpp').read_text().split('\nint main(')[0]+'\n')
    join = (DEFAULT/'consumer_fusion/consumer_fusion.cpp').read_text().split('\nint main(')[0]
    join = join.replace('#include "default_bn_stream.hpp"\n', '')
    assert 'constexpr int PED_STATE=16384;' in join
    join = join.replace('constexpr int PED_STATE=16384;', 'constexpr int PED_STATE=24576;')
    (work/'consumer_join.hpp').write_text(join+'\n')
    binary = work/'integrated_consumer'
    subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-I', str(work),
                    str(HERE/'integrated.cpp'), '-o', str(binary)], check=True)
    return binary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stress', action='store_true')
    args = ap.parse_args()
    profile = 'stress' if args.stress else 'ready'
    work = Path('/tmp/onepass_integrated_consumer_20260912')/profile
    work.mkdir(parents=True, exist_ok=True)
    binary = build(work)
    result = dict(scope=__doc__, profile=profile, axes={},
        evidence='CPU complete-domain actual-payload port service model, not RTL/PPA/new AEE or full-network service.',
        common_baseline_X=0, statistics_service_included=True, native_producer_service_included=False,
        complete_domain=dict(T=10, C=96, H=120, W=160, positions=192000, final_FP32_values=18432000),
        formula='sum(x)/n; sum(FMA(x,x,acc))/n - mean*mean, two stripes and paired256-leaf tree; original rsqrt seed/three Newton steps, separate affine MUL/ADD, then FP32 ADD with actual q24/f14 PED.',
        source_identity='AT-LIF={0,theta}; theta can be folded in next weights. PED continuous q24 is a separate real consumer, not a second continuous spike amplitude.',
        resources=dict(state_capacity_bytes=131072, coefficient_capacity_bytes=131072,
            coefficient_used_bytes=896, RF_vectors=96, lanes=8, state_ports='1R64/1W64', coefficient_port='1R256',
            FP_latency=4, integer_seed_latency=2, tree_state_interval=[16384,24064],
            PED_state_interval=[24576,24864], tag_state_interval=[8192,8224],
            shared_input_staging_interval=[0,32], shared_output_staging_interval=[32,64],
            state_address_high_water=24864, active_state_bytes=64+32+7680+288,
            moments_RF_vectors=list(range(48)), affine_scale_RF_vectors=list(range(12,24)),
            affine_bias_RF_vectors=list(range(24,36)), default_RF_vectors=list(range(48,60)),
            normalized_position_RF_vectors=list(range(12)), tag_RF_vectors=[88,89,90,91],
            RF74_lifetime='eps during variance; after all affine steps, charged CR256/LOAD of f14scale at coefficient864, wait before PED use.',
            RF_vectors_named_across_all_phases=76,
            named_RF_union='0..59,72..74,80..91,95; phase lifetimes overlap only within these existing96 vectors.',
            off_chip_BN_intermediate_bytes=73728000),
        pressure='SR blocked last8/32 and SW blocked last4/32 globally from Engine cycle0' if args.stress else 'always_ready',
        external_transfer='Existing32B/5slot contract; actual PED spatial-T10 layout and2880B stride/address issue retained, no DRAM-row model.',
        omitted_third_arm='Prior tag/default-reference arm was identical to ordinary fusion; no extra X claimed.',
        arithmetic_change='Onepass variance differs from old centered two-pass FP32 order; old AEE/CUDA identity not inherited.')
    path = HERE/f'results_{profile}.json'
    for axis in (['ordinary'] if args.stress else ['ordinary','lifting_raw']):
        aw = work/axis
        aw.mkdir(exist_ok=True)
        info = prepare(axis, aw, profile)
        record = dict(input=info, arms={})
        for mode, name in MODES.items():
            intermediate = aw/f'{name}_BN.f32'
            output = aw/f'{name}_final.f32'
            stats = aw/f'{name}_stats.f32'
            print(axis, name, 'start', flush=True)
            begin = time.monotonic()
            proc = subprocess.run([str(binary), str(aw/'live.f32'), str(aw/'integrated_coeff.f32'),
                str(aw/'tags.u8'), str(aw/'PED_spatial_T_C.i24'), str(intermediate), str(output), str(stats),
                str(int(args.stress)), str(mode)], text=True, capture_output=True, check=True)
            arm = json.loads(proc.stdout)
            arm['wall_seconds'] = time.monotonic()-begin
            assert sum(arm['stages']) == arm['service_slots']
            arm['phase_service'] = dict(zip(STAGES, arm['stages']))
            arm['strict_final_bits'] = code0.bit_check(output, aw/'reference_final.f32')
            arm['strict_computed_statistics_bits'] = code0.bit_check(stats, info['reference_BN_statistics'])
            if mode == 1:
                arm['strict_BN_intermediate_bits'] = code0.bit_check(intermediate, info['reference_BN_output'])
            else:
                arm['strict_final_vs_materialized_bits'] = code0.bit_check(output, aw/f'{MODES[1]}_final.f32')
            arm['physical_port_bytes'] = dict(state_read=8*arm['SR64_reads'], state_write=8*arm['SW64_writes'],
                coefficient_read=32*arm['CR256_reads'], coefficient_fill=32*arm['CW256_writes'])
            arm['external_read_bytes'] = sum(arm[k] for k in ('moment_payload_bytes', 'normalization_payload_bytes',
                'PED_packed24_read_bytes', 'intermediate_BN_read_bytes', 'tag_read_bytes'))
            arm['external_write_bytes'] = arm['intermediate_BN_write_bytes']+arm['final_write_bytes']
            arm['algorithm_statistics_output_file_is_reference_only'] = True
            record['arms'][name] = arm
            arm['service_reduction_vs_materialized'] = 1-arm['service_slots']/record['arms'][MODES[1]]['service_slots']
            result['axes'][axis] = record
            path.write_text(json.dumps(result, indent=2)+'\n')
            print(axis, name, arm['service_slots'], 'FULL_FINAL_BITS_EQUAL', flush=True)


if __name__ == '__main__':
    main()
