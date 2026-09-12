from pathlib import Path
import json
from run import code0,MODES

HERE=Path(__file__).resolve().parent
result=dict(scope='Computed-statistics/raw-projection boundary through complete192000x96 normalized+real-PED output.',
    evidence='CPU payload/port suffix model, not whole-network/RTL/PPA/new AEE.',
    conclusion='Ordinary BN normalization+PED fusion already retains the code0 default reference until final ADD; proposed representation has zero incremental service.',
    rows=[],ready_stress_bits={})
for condition in ['ready','stress']:
    d=json.loads((HERE/f'results_{condition}.json').read_text())
    for axis,record in d['axes'].items():
        for name,r in record['arms'].items():
            result['rows'].append(dict(condition=condition,axis=axis,mode=name,
                service_slots=r['service_slots'],phase_service=r['phase_service'],
                reduction_vs_materialized=r['service_reduction_vs_materialized'],
                candidate_increment_vs_ordinary_fusion=r['service_reduction_vs_ordinary_fusion'],
                external_read_bytes=r['external_read_bytes'],external_write_bytes=r['external_write_bytes'],
                physical_port_bytes=r['physical_port_bytes'],PED_read_bytes=r['PED_packed24_read_bytes'],
                PED_conversion_vector_issues=r['PED_conversion_vector_issues'],
                final_ADD_vector_issues=r['final_ADD_vector_issues'],
                full_final_bit_check=r['strict_final_bits']))
            if condition=='stress':
                a=Path('/tmp/default_bn_consumer_fusion_20260912/ready')/axis/(name+'_final.f32')
                b=Path('/tmp/default_bn_consumer_fusion_20260912/stress')/axis/(name+'_final.f32')
                result['ready_stress_bits'][axis+'_'+name]=code0.bit_check(a,b)
result['excluded']=['Complete native producer and encoded-stream formation throughput',
    'BN mean/variance/rsqrt service already completed before this suffix',
    'Current-frame recovered pruning students, new AEE, next-layer gates and other network stages']
(HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
