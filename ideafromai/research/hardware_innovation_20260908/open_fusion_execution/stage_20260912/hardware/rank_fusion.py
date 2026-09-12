"""Fork real resident producer state into R32 and two fixed R24 consumers.

The prefix is genuinely executed once, then its complete Machine state/time
is copied for alternative consumers. No service number or gate oracle is
imported at the boundary. This is ordinary deterministic simulation reuse.
"""
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import integrated as stage
import consumer_ranked

HERE=stage.HERE
common=stage.common
REFERENCE=HERE/'integrated_r32.json'


def parameters_and_gold(data,q,label,axis,mode):
    if mode=='original32':
        return dict(q),dict(data),32,dict(PED_q24_differences=0)
    folder=stage.OPEN/'new_interface_selection'
    factors=common.read_npz(folder/(axis+'_rebase_parameters.npz'))
    basis=mode.removesuffix('24')
    nq=dict(q)
    nq['U_ped_q16']=factors[basis+'_U'][:24].copy()
    nq['V_ped_q16']=factors[basis+'_V'][:,:24].copy()
    assert int(nq['U_ped_exponent'])==16 and int(nq['V_ped_exponent'])==15
    geo=json.loads(str(data['window_geometry_json']))[label]
    oy,ox=geo['gate_origin'];dy,dx=geo['output_origin']
    updated=data[label+'_updated_I24']
    x=np.stack([updated[:,:,2*(dy+y)-oy,2*(dx+x)-ox] for y in range(4) for x in range(4)]).astype(np.int64)
    u=stage.addresses.resident.rne_sat(x @ nq['U_ped_q16'].astype(np.int64).T,16)
    v=stage.addresses.resident.rne_sat(u @ nq['V_ped_q16'].astype(np.int64).T,15)
    v=stage.addresses.resident.rne_sat(v+nq['PED_bias_q24'],0)
    out=v.reshape(4,4,10,96).transpose(2,3,0,1)
    candidate=dict(data)
    candidate[label+'_continuous_q24']=out
    delta=common.difference(out,data[label+'_continuous_q24'])
    return nq,candidate,24,dict(PED_vs_original_capture=delta,
        gold='Independent integer U24 RNE/sat -> V96 RNE/sat + unchanged PED bias/sat; new candidate output, not original capture.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true')
    ap.add_argument('--axis',choices=['ordinary','lifting_raw']);ap.add_argument('--window',choices=['corner','interior'])
    args=ap.parse_args()
    old=json.loads(REFERENCE.read_text())
    result=dict(scope=__doc__,evidence='CPU payload slot prototype, not RTL/PPA or whole-frame latency.',
        variants=['original32','original_ordered24','activation_whitened24'],rows=[])
    output=HERE/('rank_fusion'+''.join('_'+x for x in [args.axis,args.window] if x)+('_stress' if args.stress else '')+'.json')
    for axis in ([args.axis] if args.axis else ['ordinary','lifting_raw']):
        folder=common.FULL/'capture'/axis
        data=common.read_npz(folder/'000_zurich_city_09_a_0001.npz')
        p=common.read_npz(folder/'live_parameters.npz');q=common.read_npz(folder/'parameters.npz')
        for label in ([args.window] if args.window else ['corner','interior']):
            stage.run_windows.Machine=stage.IntegratedMachine
            stage.run_windows.build_nrv=stage.preview_directory
            preview,producer,prefix=stage.run_windows.window(data,p,label,False,args.stress,True,axis)
            archived=json.loads((common.FULL/'preview_sn2_chain/windows.json').read_text())['axes'][axis][label]['expanded_fp32']['checks']
            assert producer['checks']==archived and producer['checks']['sn2']['differences']==0
            for mode in result['variants']:
                nq,candidate,rank,delta=parameters_and_gold(data,q,label,axis,mode)
                m=copy.deepcopy(prefix)
                m.forward_i24=True
                producer_end=m.time
                value,consumer=consumer_ranked.run(candidate,nq,label,None,stress=args.stress,machine=m,rank=rank)
                assert sum(m.stages.values())==m.time
                if mode=='original32' and not args.stress:
                    ref=next(r for r in old['rows'] if r['axis']==axis and r['window']==label)
                    assert m.time==ref['service_slots'] and dict(m.count)==ref['counts']
                row=dict(axis=axis,window=label,mode=mode,rank=rank,stress=args.stress,
                    service_slots=m.time,producer_end=producer_end,consumer_begin=producer_end,
                    consumer_end=m.time,consumer_service_slots=m.time-producer_end,
                    source_program=producer['source_program'],producer=producer,consumer=consumer,
                    same_machine_payload_handoff=True,no_gate_DMA_between_producer_and_consumer=True,
                    prefix_reuse='Complete identical simulated Machine state, elapsed time, SRAM/RF and arbitration state copied after actual producer execution.',
                    checks=consumer['checks'],candidate_PED_delta=delta,
                    stages=dict(m.stages),counts=dict(m.count),
                    port_bytes=dict(SR64=8*m.count['SR64_reads'],SW64=8*m.count['SW64_writes'],CR256=32*m.count['CR256_reads'],CW256=32*m.count['CW256_writes']),
                    same_resource_budget=dict(RF96x8_bits48=True,state_bytes=131072,coefficient_bytes=131072,source_ROM_bytes=8192),
                    no_native_projection_or_globalBN=True,new_training=False,new_AEE=False,
                    AEE_scope='Existing separate R24 fixed-diverse10 candidate checks only; full825 R24 validation belongs to phase_aee, not inherited here.')
                result['rows'].append(row)
                output.write_text(json.dumps(result,indent=2)+'\n')
                print(axis,label,mode,m.time,m.time-producer_end,flush=True)
    print('RANK_FUSION_DONE',output,flush=True)

if __name__=='__main__':main()
