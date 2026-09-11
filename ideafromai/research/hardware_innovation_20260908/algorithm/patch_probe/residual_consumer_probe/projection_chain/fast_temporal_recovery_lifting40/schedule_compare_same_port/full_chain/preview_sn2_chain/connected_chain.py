"""One machine, resident sn2 handoff, real integer dual consumers.

Only the previously faster common expanded-coefficient control is extended.
No sn2 egress/reload occurs at its resident handoff. --source executes the
compiled source program on the same machine; --native executes full-K native
projection. Dynamic global BN remains outside every current flag combination.
Original I24 input uses the declared spatial/T/C interface.
"""
import argparse
import json
import numpy as np
from run_windows import window,HERE,read_npz
from integer_chain import run as integer_run


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');ap.add_argument('--forward',action='store_true');ap.add_argument('--native',action='store_true');ap.add_argument('--source',action='store_true');args=ap.parse_args()
    result=dict(scope=__doc__,axes={},complete_chain=False,new_training=False,new_quantization=False,
        state_bytes=131072,coefficient_bytes=131072,RF_vector_words=96,RF_lanes=8,RF_bits_per_lane=48,
        state_ports='1R64/1W64',coefficient_port='1R256',DMA='32B / 5 slots',
        stress='fixed SR last8/SW last4 of32 blocked' if args.stress else 'always_ready')
    for axis in ('ordinary','lifting_raw'):
        path=HERE.parent/'capture'/axis
        data=read_npz(path/'000_zurich_city_09_a_0001.npz');p=read_npz(path/'live_parameters.npz');q=read_npz(path/'parameters.npz')
        result['axes'][axis]={}
        for label in ('corner','interior'):
            _,preview,m=window(data,p,label,False,args.stress,True,axis if args.source else None)
            m.forward_i24=args.forward
            value,integer=integer_run(data,q,label,None,args.stress,m,args.native)
            native=None
            if args.native:
                from native_projection import run as native_run
                value['native_projection'],native=native_run(m,data,p,label)
            report=dict(service_slots=m.time,stages=dict(m.stages),counts=dict(m.count),preview=preview,integer=integer,
                handoff='Actual sn2 state SRAM bytes, no oracle replacement and no external round trip',max_pending=m.max_pending,
                complete_chain=False,relative_AEE_gate_passed=False)
            result['axes'][axis][label]=report
            if native is not None:report['native_projection']=native
            print(axis,label,m.time,'integer_differences',sum(x['differences'] for x in integer['checks'].values()),flush=True)
            if not args.stress:np.savez_compressed(HERE/f'{axis}_{label}_connected_output.npz',**value)
    result['common_I24_operand_path']='3B operand collector + ordinary sign extension and next-MAC prefetch' if args.forward else 'explicit register decode reference'
    result['instruction_ROM_bytes']=8192 if args.source else 0
    result['source_program_executed']=args.source
    result['native_projection_executed']=args.native
    name='connected'+('_source' if args.source else '')+('_forward' if args.forward else '')+('_native' if args.native else '')+('_stress' if args.stress else '')+'.json'
    (HERE/name).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
