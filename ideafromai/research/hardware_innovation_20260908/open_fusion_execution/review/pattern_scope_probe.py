import json
from pathlib import Path
import numpy as np
from collections import OrderedDict
base=Path(__file__).resolve().parents[2]
full=base/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
result={'scope':'One unchanged frame, two existing students. Exact complete K864 pattern opportunity; counts only, not cycle or hardware result. Zero matches excluded from novel opportunity.','cache_entries':16,'axes':{}}
for axis in ['ordinary','lifting_raw']:
    with np.load(full/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz') as d:
        words=d['full_proj_words'];raw=d['proj_bn_full_input_fp32']
    w=np.pad(words,((0,0),(1,1),(1,1)))
    src=np.stack([w[:,ky:ky+240:2,kx:kx+320:2] for ky in range(3) for kx in range(3)],axis=1)
    # The key preserves c,ky,kx; output time-major T,H,W matches full BN vector order.
    src=src.reshape(864,120,160).transpose(1,2,0)
    packed=np.concatenate([np.packbits(((src>>t)&1).astype(np.uint8),axis=-1,bitorder='little').reshape(-1,108) for t in range(10)])
    keys=packed.view('V108').reshape(-1)
    uni,idx,inv,cnt=np.unique(keys,return_index=True,return_inverse=True,return_counts=True)
    zero=np.all(packed==0,axis=1);nz=np.flatnonzero(~zero)
    distinct_nz=int(np.count_nonzero(np.any(packed[idx]!=0,axis=1)))
    exact_x=raw.transpose(0,2,3,1).reshape(-1,96)
    mismatches=int(np.count_nonzero(exact_x != exact_x[idx[inv]]))
    cache=OrderedDict();hits=miss=0
    for i in nz:
        key=keys[i].tobytes()
        if key in cache: hits+=1;cache.move_to_end(key)
        else:
            miss+=1;cache[key]=None
            if len(cache)>16:cache.popitem(last=False)
    # Same timestep scanline-local adjacency; excludes all-zero patterns.
    p=packed.reshape(10,120,160,108)
    equal_x=np.all(p[:,:,:-1]==p[:,:,1:],axis=-1)&np.any(p[:,:,1:]!=0,axis=-1)
    rec=dict(total_vectors=int(len(keys)),zero_vectors=int(zero.sum()),nonzero_vectors=int(len(nz)),distinct_nonzero_patterns=distinct_nz,unbounded_nonzero_reuse=int(len(nz)-distinct_nz),unbounded_fraction_of_all_vectors=(len(nz)-distinct_nz)/len(keys),LRU16_nonzero_hits=hits,LRU16_nonzero_misses=miss,LRU16_fraction_of_nonzero=hits/len(nz),LRU16_fraction_of_all=hits/len(keys),same_row_left_nonzero_hits=int(equal_x.sum()),pattern_implies_same_raw_FP32_mismatched_scalars=mismatches,cache_key_bytes=16*108,cache_H32_result_bytes=16*32*4,charged_metadata_or_timeline=False)
    result['axes'][axis]=rec;print(axis,rec,flush=True)
path=base/'open_fusion_execution/review/pattern_scope_probe.json';path.write_text(json.dumps(result,indent=2)+'\n')
