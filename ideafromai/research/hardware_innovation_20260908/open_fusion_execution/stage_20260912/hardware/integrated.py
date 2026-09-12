"""One finite machine: compiled source -> resident sn2 -> real dual consumers."""
from pathlib import Path
import argparse
import json
import sys
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1]
sys.path.insert(0,str(OPEN/'pruning/source_address_iterator'))
import execute as addresses
common=addresses.common
integer_chain=addresses.integer_chain
import run_windows
from run_chain import SRC,NRV


class IntegratedMachine(addresses.IteratorMachine):
    def __init__(self,stress=False):
        super().__init__(stress)
        self.directory_arm='coalesced_scan'
        self.uniform_mask=True
        self.source_commit_trace=[]

    def advance(self,*args,**kwargs):
        watch=kwargs.get('tag')=='source_gate_word_store' and kwargs.get('write') is not None
        if watch: requested=self.time; address=kwargs['write'][0]
        super().advance(*args,**kwargs)
        if watch:
            self.source_commit_trace.append(dict(request_slot=requested,accept_slot=self.time-1,
                address=address,wait_slots=self.time-1-requested,bytes=8))


def preview_directory(m,words,source_origin,positions):
    """Same paid coalescer on real producer SRAM, complete K864."""
    assert m.sn1_spatial
    h,w=words.shape[1:]
    geo=dict(gate_shape=[h,w],gate_origin=source_origin)
    n,live=addresses.directory(m,geo,positions,gate_base=SRC,dir_base=NRV,
        phase='preview_complete_K864_coalesced_directory')
    records=np.frombuffer(m.state,'<u4',count=2*n,offset=NRV).reshape(n,2)
    occurrences=sum(int(row[1]).bit_count() for row in records)
    return n,occurrences,live


def run(axis,label,stress):
    folder=common.FULL/'capture'/axis
    data=common.read_npz(folder/'000_zurich_city_09_a_0001.npz')
    p=common.read_npz(folder/'live_parameters.npz')
    q=common.read_npz(folder/'parameters.npz')
    run_windows.Machine=IntegratedMachine
    run_windows.build_nrv=preview_directory
    integer_chain.directory=addresses.directory
    integer_chain.dense=addresses.resident.dense_adapter
    preview,producer,m=run_windows.window(data,p,label,False,stress,True,axis)
    archived=json.loads((common.FULL/'preview_sn2_chain/windows.json').read_text())['axes'][axis][label]['expanded_fp32']['checks']
    assert producer['checks']==archived, ('new_supply_changed_preview',producer['checks'],archived)
    assert producer['checks']['Z']['differences']==0 and producer['checks']['sn2']['differences']==0
    assert producer['source_program']['checks']['differences']==0
    producer_end=m.time
    producer_directories=len(m.directory_trace)
    producer_counts=Counter(m.count)
    # Real sn2 bytes are retained at GATE. No external egress/re-ingress,
    # no value substitution and no fresh machine appears at the boundary.
    m.forward_i24=True
    value,consumer=integer_chain.run(data,q,label,None,stress=stress,machine=m,late_v=False)
    counts=dict(m.count)
    trace=dict(scope='Actual source gate SW64 sink observations and later consumers on this same Machine; batch gate buffer, no invented concurrent consumer.',
        axis=axis,window=label,stress=stress,source_commit_SW64=m.source_commit_trace,
        preview_consumers=producer['P2_rows'],integer_consumers=consumer['rows'])
    trace_dir=HERE/'source_rtl_inputs';trace_dir.mkdir(exist_ok=True)
    trace_file=trace_dir/(axis+'_'+label+('_stress' if stress else '')+'_sink_trace.json')
    trace_file.write_text(json.dumps(trace,indent=2)+'\n')
    assert sum(m.stages.values())==m.time
    assert consumer['service_slots']==m.time-producer_end
    assert not counts.get('DMA_output_slots',0), 'no source/sn2 external spill permitted'
    assert 'sn2_continuation_input' not in m.stages
    report=dict(axis=axis,window=label,stress=stress,service_slots=m.time,
        evidence='CPU payload slot model with actual SRAM/RF arithmetic and shared arbitration; not RTL/PPA or whole-frame latency.',
        scope='Real I24 halo -> compiled T10 source -> complete K864 U32/V+BN1/noncausal sn2 -> resident gate -> complete K864 Conv2 U16/F+BN2+rawI24 -> projection gate and continuous PED U32/V96.',
        source_program=producer['source_program'],producer=producer,consumer=consumer,
        preview_FP_boundary='Existing scalar FP32 execution differs slightly from captured CUDA raw/BN1; complete check statistics reproduce the archived ordinary same-harness baseline, sn2 gates remain exact.',
        actual_sink_trace_file=str(trace_file.relative_to(HERE)),
        source_commit_sink_wait_slots=sum(r['wait_slots'] for r in m.source_commit_trace),
        stages=dict(m.stages),counts=counts,
        physical_port_bytes=dict(state_read=8*m.count['SR64_reads'],state_write=8*m.count['SW64_writes'],
            coefficient_read=32*m.count['CR256_reads'],coefficient_fill=32*m.count['CW256_writes']),
        integration=dict(one_machine=True,resident_sn2_gate_handoff=True,
            sn2_gate_egress_slots=0,sn2_gate_reingress_slots=0,
            producer_begin=0,producer_end=producer_end,consumer_begin=producer_end,consumer_end=m.time,
            actual_preview_coefficient_fill=producer_counts['CW256_writes']*32,
            actual_integer_coefficient_replacement=(m.count['CW256_writes']-producer_counts['CW256_writes'])*32,
            raw_I24_reread_for_consumers_is_paid=True,
            producer_directory_calls=producer_directories,
            consumer_directory_calls=len(m.directory_trace)-producer_directories,
            no_native_projection_or_global_BN=True),
        same_resources=dict(RF_vectors=96,lanes=8,bits_per_lane=48,state_bytes=131072,
            coefficient_bytes=131072,instruction_ROM_bytes=8192,
            state_ports='1R64 / 1W64',coefficient_port='1R256',DMA='32B / 5 slots',
            coalescer_logical_bytes=144,coalescer_existing_RF_vectors=9,geometry_existing_RF_vectors=9,
            source_resident_operand_MAC_common=True),
        actual_directory_trace=m.directory_trace,operator_timeline=m.timeline,
        gold_checks=dict(source=producer['source_program']['checks'],preview=producer['checks'],dual_consumer=consumer['checks']),
        complete_frame=False,new_training=False,new_AEE=False,production_modified=False)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--axis',choices=['ordinary','lifting_raw'])
    ap.add_argument('--window',choices=['corner','interior'])
    ap.add_argument('--stress',action='store_true')
    args=ap.parse_args()
    result=dict(scope=__doc__,rank=32,mask='none',rows=[])
    target=HERE/('integrated_r32'+''.join('_'+x for x in [args.axis,args.window] if x)+('_stress' if args.stress else '')+'.json')
    for axis in ([args.axis] if args.axis else ['ordinary','lifting_raw']):
        for label in ([args.window] if args.window else ['corner','interior']):
            report=run(axis,label,args.stress)
            result['rows'].append(report)
            target.write_text(json.dumps(result,indent=2)+'\n')
            print(axis,label,report['service_slots'],report['integration'],flush=True)
    print('INTEGRATED_R32_DONE',target,flush=True)

if __name__=='__main__':main()
