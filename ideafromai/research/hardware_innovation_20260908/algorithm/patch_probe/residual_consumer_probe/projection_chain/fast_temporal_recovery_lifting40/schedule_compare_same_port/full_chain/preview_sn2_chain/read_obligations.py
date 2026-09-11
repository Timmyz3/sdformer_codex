"""F1 static boundary check on actual matrices, not a pruning experiment."""
import json
import numpy as np
from run_chain import HERE,read_npz


def physical_words(coordinates):
    result=set()
    for tp,c in coordinates:
        address=(tp*96+c)*3
        result.update(range(address//8,(address+2)//8+1))
    return result


def main():
    result=dict(scope=__doc__,axes={},changed_consumers=False,trained_masks=False)
    for axis in ('ordinary','lifting_raw'):
        path=HERE.parent/'capture'/axis
        q=read_npz(path/'parameters.npz');p=read_npz(path/'live_parameters.npz')
        gate_times={int(q['consumer_permutation'][t]) for t in range(10) if int(q['consumer_constant'][t])<0}
        ped_columns=set(np.flatnonzero(np.any(q['U_ped_q16']!=0,axis=0)).tolist())
        gate=physical_words([(p0*10+t,c) for p0 in range(2) for t in gate_times for c in range(96)])
        ped=physical_words([(p0*10+t,c) for p0 in range(2) for t in range(10) for c in ped_columns])
        allocation=set(range(5760//8))
        result['axes'][axis]=dict(updated_I24_P2_bytes=5760,allocation_words64=len(allocation),
            gate_required_words64=len(gate),PED_U_required_words64=len(ped),union_words64=len(gate|ped),
            removable_first_read_words64=len(allocation-(gate|ped)),
            preview_U_live_input_columns=int(np.count_nonzero(np.any(p['preview_u'][:,:32]!=0,axis=1))),
            preview_V_live_latent_rows=int(np.count_nonzero(np.any(p['preview_v'][:32]!=0,axis=1))),
            PED_V_live_latent_rows=int(np.count_nonzero(np.any(q['V_ped_q16']!=0,axis=0))),
            already_common_dead_preview_rows=int(np.count_nonzero(~np.any(p['preview_v']!=0,axis=1))),
            conclusion='No static updated-I24 first-read deletion with unchanged consumers. This does not refute trained joint masks, different production layouts, or fewer repeat reads.')
    (HERE/'read_obligations.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
