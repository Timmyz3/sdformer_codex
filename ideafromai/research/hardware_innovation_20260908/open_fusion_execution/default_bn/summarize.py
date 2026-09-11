"""Report exact full-domain comparisons, keeping every baseline visible."""
from pathlib import Path
import json
from run import bit_check

HERE=Path(__file__).resolve().parent
result=dict(scope='Complete192000-position/C96 BN operator on original ordinary/lifting captures.',
    evidence='CPU payload slots, not RTL, ASIC PPA, CUDA equivalence, new AEE or full-network performance.',
    comparison='Generic code0 compression/default normalization are strong baselines. Full-zero-leaf replay is measured only against default_ops for its incremental result.',
    rows=[],ready_stress_checks={})
for condition in ['ready','stress']:
    d=json.loads((HERE/f'results_{condition}.json').read_text())
    for axis,record in d['axes'].items():
        strong=record['arms']['default_ops']
        for name,r in record['arms'].items():
            result['rows'].append(dict(condition=condition,axis=axis,mode=name,
                service_slots=r['service_slots'],phase_service=dict(zip(r['stage_names'],r['stages'])),
                reduction_vs_dense=r['dense_service_reduction'],
                reduction_vs_tag_same_arithmetic=r['service_reduction_vs_tag_same_arithmetic'],
                incremental_reduction_vs_default_ops=r['service_reduction_vs_default_ops'],
                external_read_bytes=r['external_read_bytes'],external_output_bytes=r['external_output_bytes'],
                physical_port_bytes=r['physical_port_bytes'],
                extra_state_reads_vs_default_ops=r['physical_port_bytes']['state_read']-strong['physical_port_bytes']['state_read'],
                default_cache_prepare_slots=r['default_prepare_slots'],zero256_leaves_replayed=r['zero_blocks_replayed'],
                shared_negative_mu_prepare_slots=r.get('negative_mu_prepare_slots',0),
                variance_uses_shared_negative_mu=r.get('variance_uses_shared_negative_mu',False),
                tag_bytes=r['tag_read_bytes'],tag_decode_slots=r['tag_decode_slots'],
                tag_unpack_vector_issues=r['tag_unpack_vector_issues'],block_test_slots=r['block_test_slots'],
                stats_and_all_normalized_bits=r['strict_bit_checks']))
            if condition=='stress':
                ready=Path('/tmp/default_bn_20260912/ready')/axis
                stress=Path('/tmp/default_bn_20260912/stress')/axis
                result['ready_stress_checks'][axis+'_'+name]=dict(
                    output=bit_check(stress/f'{name}_output.f32',ready/f'{name}_output.f32'),
                    statistics=bit_check(stress/f'{name}_stats.f32',ready/f'{name}_stats.f32'))
result['excluded_costs']=['Full native projection producer and generation of its K864 empty witness',
    'Formation throughput of the external compressed stream',
    'Continuous PED and final residual addition',
    'Other layers; no addition of this full BN domain to the earlier4x4 windows']
result['not_claimed_prior_baselines']=['Projection-fused first-pass BN statistics have not been executed in this comparison',
    'No Welford or full optimized BN compiler reproduction is claimed']
(HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
