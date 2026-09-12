"""One-machine fixed mask ablation; source and full dual consumers stay actual."""
import argparse
import importlib.util
import json
import numpy as np
import integrated as stage

HERE=stage.HERE
common=stage.common
spec=importlib.util.spec_from_file_location('stage_existing_mask_primitives',stage.OPEN/'pruning/execute.py')
masked=importlib.util.module_from_spec(spec);spec.loader.exec_module(masked)


def run(axis,label,name):
    folder=common.FULL/'capture'/axis
    data=common.read_npz(folder/'000_zurich_city_09_a_0001.npz')
    p=common.read_npz(folder/'live_parameters.npz');q=common.read_npz(folder/'parameters.npz')
    if name=='all_keep':
        drop=np.zeros((4,96),bool)
    else:
        path=stage.OPEN/('pruning/paired_phase_masks.json' if name=='row_phase_joint_pair' else 'review/phase_group8_masks.json')
        drop=np.asarray(json.loads(path.read_text())[axis][name]['mask_uint8'],bool)
    m=stage.IntegratedMachine(False)
    blob,base,info=masked.coefficients(p,False)
    m.phase='coefficient_cold_fill';m.dma_input(blob,0,True)
    source_report=masked.source_run(m,data,axis,label)
    geo=json.loads(str(data['window_geometry_json']))[label]
    source=data[label+'_sn1_gate'];words=sum(source[t].astype(np.uint16)<<t for t in range(10))
    m.phase='phase_mask_input';m.dma_input(stage.addresses.metadata_bytes(drop),masked.MASK)
    h,w=geo['gate_shape'];oy,ox=geo['gate_origin'];rows=[]
    for y in range(h):
        for x in range(0,w,2):
            count=min(2,w-x);positions=[(oy+y,ox+x+i) for i in range(count)];start=m.time
            n,occ,live=stage.preview_directory(m,words,geo['source_origin'],positions)
            masked.execute_u(m,n,base,False,count)
            m.phase='phase_mask_decode';keep=masked.kept_groups(m,positions)
            masked.masked_v(m,base,live,keep)
            masked.masked_sn2(m,base,p['preview_A'],keep,common.GATE+(y*w+x)*192)
            rows.append(dict(positions=positions,start=start,gate_ready=m.time))
    actual=np.frombuffer(m.state,'<u2',count=h*w*96,offset=common.GATE).reshape(h,w,96).copy()
    gate=np.stack([(actual>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    saved=data[label+'_sn2_gate'] if name=='all_keep' else common.read_npz(stage.OPEN/'pruning'/f'{axis}_{label}_{name}.npz')['gate']
    assert np.array_equal(gate,saved)
    expected,oracle=common.independent_gold(data,q,label,gate)
    candidate=dict(data)
    for key,name0 in [('updated_I24','updated'),('proj_gate','gate'),('continuous_q24','continuous')]:candidate[label+'_'+key]=expected[name0]
    producer_end=m.time;producer_counts=dict(m.count);producer_stages=dict(m.stages)
    m.uniform_mask=bool(np.all(drop==drop[:1]))
    # Compiler chooses the already-measured stronger static control:
    # uniform all-valid interior needs no offset iterator; boundary/phase
    # cases use the existing fixed iterator. Same rule for either student.
    m.directory_arm='static_predicate' if m.uniform_mask and label=='interior' else 'address_iterator'
    m.forward_i24=True
    stage.integer_chain.directory=stage.addresses.directory
    stage.integer_chain.dense=stage.addresses.resident.dense_adapter
    value,consumer=stage.integer_chain.run(candidate,q,label,None,machine=m,late_v=False)
    assert sum(m.stages.values())==m.time
    assert 'sn2_continuation_input' not in m.stages and 'sn2_gate_egress' not in m.stages
    return dict(axis=axis,window=label,mask=name,service_slots=m.time,producer_end=producer_end,
        consumer_begin=producer_end,consumer_end=m.time,source_program=source_report,
        producer_counts=producer_counts,producer_stages=producer_stages,producer_rows=rows,
        consumer=consumer,counts=dict(m.count),stages=dict(m.stages),
        one_machine_real_gate_handoff=True,mask_metadata_filled_once_then_reused=True,
        actual_directory_arm=m.directory_arm,gold_checks=consumer['checks'],independent_oracle=oracle,
        port_bytes=dict(SR64=8*m.count['SR64_reads'],SW64=8*m.count['SW64_writes'],CR256=32*m.count['CR256_reads'],CW256=32*m.count['CW256_writes']),
        changed_network_delta_vs_original_capture=common.metrics((value['updated'],value['gate'],value['continuous']),
            (data[label+'_updated_I24'],data[label+'_proj_gate'],data[label+'_continuous_q24'])),
        resource=dict(RF96x8_48bit=True,state_bytes=131072,coefficient_bytes=131072,source_ROM_bytes=8192,state_high_water=masked.MASK+32),
        no_R24_combination_no_new_AEE=True,no_native_projection_or_globalBN=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mask',default='global_group2',choices=['all_keep','global_group2','phase_joint','row_phase_joint_pair'])
    ap.add_argument('--window',choices=['corner','interior']);args=ap.parse_args()
    result=dict(scope=__doc__,evidence='CPU payload slot prototype; not RTL/PPA/full-frame.',rows=[])
    out=HERE/('mask_fusion_'+args.mask+('_'+args.window if args.window else '')+'.json')
    for axis in ['ordinary','lifting_raw']:
        for label in ([args.window] if args.window else ['corner','interior']):
            r=run(axis,label,args.mask);result['rows'].append(r);out.write_text(json.dumps(result,indent=2)+'\n')
            print(axis,label,args.mask,r['service_slots'],r['producer_end'],r['consumer']['service_slots'],flush=True)
    print('MASK_FUSION_DONE',out,flush=True)

if __name__=='__main__':main()
