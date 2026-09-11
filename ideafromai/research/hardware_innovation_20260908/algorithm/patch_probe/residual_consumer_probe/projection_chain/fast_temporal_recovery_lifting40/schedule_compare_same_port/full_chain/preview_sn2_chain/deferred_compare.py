"""Compare both legal PED orders on the same executed local source chain.

Counts are local active service with an unresolved whole-domain BN wait.
The separately executed global BN slots are never added to a local window.
"""
import argparse
import json
import numpy as np
from run_windows import window,HERE,read_npz
from run_chain import difference
from integer_chain import run as integer_run
from native_projection import run as native_run
from deferred_ped import run as completion_run,computed_stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--axis',choices=('ordinary','lifting_raw'),required=True)
    ap.add_argument('--stress',action='store_true');args=ap.parse_args()
    path=HERE.parent/'capture'/args.axis
    data=read_npz(path/'000_zurich_city_09_a_0001.npz');p=read_npz(path/'live_parameters.npz');q=read_npz(path/'parameters.npz')
    stats,stats_path=computed_stats(args.axis,args.stress)
    result=dict(scope=__doc__,axis=args.axis,stress=args.stress,windows={},stats_output_file=stats_path,
        state_bytes=131072,coefficient_bytes=131072,RF_vector_words=96,RF_lanes=8,RF_lane_bits=48,
        state_ports='1R64/1W64',coefficient_port='1R256',DMA='32B/5slots',
        complete_chain=False,BN_full_producer_wait_slots=None,new_training=False,new_AEE=False)
    for label in ('corner','interior'):
        rows={};arms={}
        for late_v in (False,True):
            mode='deferred' if late_v else 'direct'
            print(args.axis,label,mode,'start',flush=True)
            _,preview,m=window(data,p,label,False,args.stress,True,args.axis)
            m.forward_i24=True
            value,integer=integer_run(data,q,label,None,args.stress,m,True,late_v)
            value['native_projection'],native=native_run(m,data,p,label)
            before_completion=m.time
            output,completion=completion_run(m,data,p,q,label,value,stats,late_v)
            rows[mode]=dict(local_active_service_slots=m.time,pre_BN_local_slots=before_completion,
                post_BN_local_slots=completion['service_slots'],counts=dict(m.count),stages=dict(m.stages),
                preview=preview,integer=integer,native=native,completion=completion,
                max_pending=m.max_pending,BN_full_producer_wait_slots=None)
            arms[mode]=output
            # Large numerical arrays remain local, outside Git (existing *.npz ignore).
            np.savez_compressed(HERE/(f'{args.axis}_{label}_{mode}'+('_stress' if args.stress else '')+'_completion.npz'),**output)
            print(args.axis,label,mode,m.time,completion['checks'],flush=True)
        equal={key:difference(arms['direct'][key],arms['deferred'][key]) for key in arms['direct']}
        assert all(x['differences']==0 for x in equal.values()),equal
        bits={k:int(np.count_nonzero(arms['direct'][k].view('u'+str(arms['direct'][k].itemsize))!=
            arms['deferred'][k].view('u'+str(arms['deferred'][k].itemsize)))) for k in arms['direct']}
        assert not any(bits.values()),bits
        rows['arm_equality']=equal
        rows['arm_bitwise_differences']=bits
        rows['deferred_local_service_reduction']=1-rows['deferred']['local_active_service_slots']/rows['direct']['local_active_service_slots']
        result['windows'][label]=rows
    name='deferred_compare_'+args.axis+('_stress' if args.stress else '')+'.json'
    (HERE/name).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
