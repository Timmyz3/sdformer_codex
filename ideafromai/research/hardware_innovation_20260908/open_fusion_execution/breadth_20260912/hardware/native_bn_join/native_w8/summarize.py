from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=Path('/tmp/native_bn_join_20260912/native_w8')


def main():
    a=json.loads((HERE/'expanded_results.json').read_text())
    b=json.loads((HERE/'compact_results.json').read_text())
    comparisons={}
    for name in ['native_raw.f32','statistics.f32','materialized_BN.f32','materialized_final.f32','fused_final.f32']:
        x=np.memmap(WORK/'expanded'/name,dtype='<f4',mode='r')
        y=np.memmap(WORK/'compact'/name,dtype='<f4',mode='r')
        assert x.shape==y.shape
        n=int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32)))
        comparisons[name]=dict(values=int(x.size),bitwise_element_differences=n)
        assert n==0
    rows=[]
    for arm_a,arm_b in zip(a['arms'],b['arms']):
        assert arm_a['mode']==arm_b['mode']
        rows.append(dict(suffix=arm_a['mode'],expanded_service=arm_a['service_slots'],compact_service=arm_b['service_slots'],
            service_reduction=1-arm_b['service_slots']/arm_a['service_slots'],
            saved_slots=arm_a['service_slots']-arm_b['service_slots'],
            expanded_external_read_bytes=arm_a['external_read_bytes'],compact_external_read_bytes=arm_b['external_read_bytes'],
            saved_external_read_bytes=arm_a['external_read_bytes']-arm_b['external_read_bytes'],
            same_external_write_bytes=arm_a['external_write_bytes']==arm_b['external_write_bytes'],
            external_write_bytes=arm_a['external_write_bytes'],
            expanded_physical_port_bytes=arm_a['physical_port_bytes'],compact_physical_port_bytes=arm_b['physical_port_bytes']))
    r=dict(status='COMPLETE_SAME_FUNCTION_FULL_DOMAIN_READY',evidence='CPU payload/port prototype, not RTL/PPA/AEE.',
        new_X=False,scope='External real old ordinary R24+onepass g/PED -> full native -> actual onepass BN -> PED join.',
        quantization=json.loads((HERE/'deployment.json').read_text()),comparisons=comparisons,rows=rows,
        native_expanded=a['native'],native_compact=b['native'],
        attribution=dict(weight_fill_saved_slots=a['native']['phase_slots'][0]-b['native']['phase_slots'][0],
            extra_decode_compute_slots=b['native']['phase_slots'][3]-a['native']['phase_slots'][3],
            extra_scale_output_slots=b['native']['phase_slots'][4]-a['native']['phase_slots'][4],
            same_source_and_directory_cost=a['native']['phase_slots'][1:3]==b['native']['phase_slots'][1:3]),
        AEE='Fresh ordinary W8 diverse10 queued separately; no inherited valid825.',
        retained_scope_limits=['Only one fixed ready full-frame layout, no latency/energy technology claim.',
            'Compact residency is an ordinary quantization/layout control available to any subsequent candidate.',
            'No general optimality claim over every possible decode-cache/compiler strategy.',
            'The new W8 reference is distinct from original GPU W; do not treat original-capture deltas as rounding-only.'])
    (HERE/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(rows=rows,checks=comparisons,attribution=r['attribution'])),flush=True)


if __name__=='__main__':main()
