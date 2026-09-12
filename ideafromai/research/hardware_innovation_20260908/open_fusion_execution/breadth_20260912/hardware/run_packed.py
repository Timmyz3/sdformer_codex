"""Same-Machine full local chain, original/expanded/actual packed U weights."""
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import packed_weights as packed

HERE=Path(__file__).resolve().parent
stage=packed.stage
consumer=packed.consumer
OLD=packed.OLD
OLDSTAGE=OLD.parent
MODES=('original32','W8_expanded16','W8_packed','W4_expanded16','W4_packed')


def parameters_and_gold(data,q,label,axis,mode):
    nq={k:v.copy() for k,v in q.items()}
    if mode=='original32':return nq,dict(data),dict(PED_vs_original=stage.common.difference(data[label+'_continuous_q24'],data[label+'_continuous_q24']))
    name=mode.split('_')[0]
    package=stage.common.read_npz(OLDSTAGE/'weight_compensation'/(axis+'_lowbit_gpu_parameters.npz'))
    nq['U_ped_q16']=package[name+'_U'].copy()
    assert np.array_equal(nq['V_ped_q16'],package[name+'_V'])
    nq['_packed_bits']=int(name[1:]) if mode.endswith('_packed') else 0
    nq['_code']=package[name+'_code'].copy();nq['_scale']=package[name+'_scale_q16'].copy()
    deploy=stage.common.read_npz(OLDSTAGE/'algorithm/weight_controls/aee'/axis/name/'deployed_constants.npz')
    equal=[k for k in q if np.array_equal(nq[k],deploy[k])]
    assert len(equal)==len(q),(axis,mode,[k for k in q if k not in equal])
    geo=json.loads(str(data['window_geometry_json']))[label]
    oy,ox=geo['gate_origin'];dy,dx=geo['output_origin']
    updated=data[label+'_updated_I24']
    x=np.stack([updated[:,:,2*(dy+y)-oy,2*(dx+x)-ox] for y in range(4) for x in range(4)]).astype(np.int64)
    coded=x@nq['_code'].astype(np.int64).T
    numerator=coded*nq['_scale'].astype(np.int64)
    assert np.array_equal(numerator,x@nq['U_ped_q16'].astype(np.int64).T)
    u=stage.addresses.resident.rne_sat(numerator,16)
    v=stage.addresses.resident.rne_sat(u@nq['V_ped_q16'].astype(np.int64).T,15)
    v=stage.addresses.resident.rne_sat(v+nq['PED_bias_q24'],0)
    out=v.reshape(4,4,10,96).transpose(2,3,0,1)
    candidate=dict(data);candidate[label+'_continuous_q24']=out
    return nq,candidate,dict(PED_vs_original=stage.common.difference(out,data[label+'_continuous_q24']),
        actual_GPU_deployment_equal_fields=equal,independent_gold='code integer dot -> exact row scale -> original U RNE/sat -> V RNE/sat -> original bias/sat',
        code_dot_maxabs=int(np.max(np.abs(coded))),scaled_numerator_maxabs=int(np.max(np.abs(numerator))))


def run(axis,label,stress=False):
    folder=stage.common.FULL/'capture'/axis
    data=stage.common.read_npz(folder/'000_zurich_city_09_a_0001.npz')
    live=stage.common.read_npz(folder/'live_parameters.npz');q=stage.common.read_npz(folder/'parameters.npz')
    stage.run_windows.Machine=packed.PackedMachine
    stage.run_windows.build_nrv=stage.preview_directory
    _,producer,prefix=stage.run_windows.window(data,live,label,False,stress,True,axis)
    archived=json.loads((stage.common.FULL/'preview_sn2_chain/windows.json').read_text())['axes'][axis][label]['expanded_fp32']['checks']
    assert producer['checks']==archived and producer['checks']['sn2']['differences']==0
    packed.install()
    result=dict(axis=axis,window=label,stress=stress,rows=[],
        scope='Real I24 source -> preview and sn2 -> full K864/BN2/rawI24 -> projection gates + complete U32/V96 PED, same Machine; no native/globalBN.',
        evidence='CPU payload slot prototype; not RTL/PPA.',
        resource=dict(RF_vectors=96,lanes=8,bits_per_lane=48,state_bytes=131072,coefficient_bytes=131072,
            state_ports='1R64/1W64',coefficient_port='1R256',DMA='32B/5slots',source_ROM_bytes=8192,
            common_staging_bytes=64,staging_occupation='24B source gather + 16B weight latch',
            packed_RF='acc0..79, scale80..83, decode84, scale-split85..86, source88..91, header94'),
        producer=producer)
    target=HERE/(axis+'_'+label+('_stress' if stress else '')+'.json')
    for mode in MODES:
        nq,candidate,delta=parameters_and_gold(data,q,label,axis,mode)
        m=copy.deepcopy(prefix);m.forward_i24=True
        boundary=m.time
        values,report=consumer.run(candidate,nq,label,None,stress=stress,machine=m,rank=32)
        assert sum(m.stages.values())==m.time
        if mode=='original32':
            path=OLD/('integrated_r32_ordinary_interior_stress.json' if stress else 'integrated_r32.json')
            ref=next(x for x in json.loads(path.read_text())['rows'] if x['axis']==axis and x['window']==label)
            assert m.time==ref['service_slots'] and dict(m.count)==ref['counts'],('baseline_changed',m.time,ref['service_slots'])
        blob,base,_=packed.coeffs(nq)
        bits=int(nq.get('_packed_bits',0))
        row=dict(mode=mode,service_slots=m.time,consumer_service_slots=m.time-boundary,
            producer_end=boundary,consumer_begin=boundary,same_machine_handoff=True,
            prefix_reuse='Copied full actual Machine state after executed identical producer, not imported gate values or added prior service tables.',
            consumer=report,counts=dict(m.count),stages=dict(m.stages),timeline=m.timeline,
            physical_port_bytes=dict(SR64=8*m.count['SR64_reads'],SW64=8*m.count['SW64_writes'],
                CR256=32*m.count['CR256_reads'],CW256=32*m.count['CW256_writes']),
            U_encoded_bytes=3072*bits//8 if bits else 6144,U_scale_bytes=64 if bits else 0,U_metadata_bytes=32 if bits else 0,
            coefficient_pool_blob_bytes=len(blob),coefficient_layout=base,checks=report['checks'],candidate_delta=delta,
            literal_packed_code_in_memory=bool(bits),original_RNE_and_bias_preserved=True,
            scale_uses_two_16x24_products=bool(bits),free_third_RF_port=False)
        result['rows'].append(row)
        target.write_text(json.dumps(result,indent=2)+'\n')
        print(axis,label,stress,mode,m.time,m.time-boundary,dict(PED=report['checks']['PED']),flush=True)
    # Same-function equality across representation arms is independent of
    # the changed-network comparison to original32.
    for name in ('W8','W4'):
        expanded=next(x for x in result['rows'] if x['mode']==name+'_expanded16')
        code=next(x for x in result['rows'] if x['mode']==name+'_packed')
        assert expanded['candidate_delta']==code['candidate_delta']
        code['service_change_vs_same_function_expanded_percent']=100*(code['service_slots']/expanded['service_slots']-1)
        code['consumer_change_vs_same_function_expanded_percent']=100*(code['consumer_service_slots']/expanded['consumer_service_slots']-1)
    result['complete']=True
    target.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--axis',choices=['ordinary','lifting_raw'],required=True)
    ap.add_argument('--window',choices=['corner','interior'],required=True);ap.add_argument('--stress',action='store_true')
    a=ap.parse_args();assert not a.stress or (a.axis,a.window)==('ordinary','interior')
    run(a.axis,a.window,a.stress)
    print('PACKED_CHAIN_COMPLETE',flush=True)


if __name__=='__main__':main()
