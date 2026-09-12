"""One fixed two-block directory-reuse interface, same complete suffix."""
from pathlib import Path
import importlib.util
import json
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('native_full_base',HERE.parent/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def main():
    # Reuse the complete existing pipeline, changing only its native mapping.
    source=(HERE.parent/'pipeline.cpp').read_text()
    assert '#include "native.hpp"' in source
    (HERE/'pipeline.cpp').write_text(source.replace('#include "native.hpp"','#include "reuse_native.hpp"'))
    original_work=base.WORK
    base.HERE=HERE;base.WORK=original_work/'directory_reuse';base.WORK.mkdir(exist_ok=True)
    base.main()
    path=HERE/'results_ready.json';r=json.loads(path.read_text())
    old=json.loads((HERE.parent/'results_ready.json').read_text())
    r['layout']='Two complete4x4 source-block directories retained across three sequential true H32 weight loads;600 batches.'
    r['resource']['native_state']=dict(directories=[0,110592],headers=[110592,110720],
        gates=[110720,126272],output_staging=[126272,126304])
    r['resource']['state_high_water']=126304
    r['resource']['single_gate_buffer_reused_between_directory_builds']=True
    r['resource']['simultaneous_weight_tiles']=1
    r['resource']['directory_preservation_checks']='All110720 directory/header bytes compared after each of three H32 consumers, every600 batch; assertion reference is not model storage.'
    r['limits'][1]='Dense BN payload baseline; no source-code0 format or global directory cache. Only fixed two4x4 source blocks are reused.'
    r['baseline']='Original whole-frame weight-stationary native_bn_join/results_ready.json; its result/source preserved.'
    for a,b in zip(r['arms'],old['arms']):
        assert a['mode']==b['mode']
        a['weight_stationary_service_slots']=b['service_slots']
        a['reduction_vs_weight_stationary']=1-a['service_slots']/b['service_slots']
        a['delta_vs_weight_stationary_slots']=a['service_slots']-b['service_slots']
        a['external_delta_vs_weight_stationary']=dict(read=a['external_read_bytes']-b['external_read_bytes'],write=a['external_write_bytes']-b['external_write_bytes'])
    checks={}
    for name in ('native_raw.f32','statistics.f32','materialized_final.f32','fused_final.f32'):
        x=np.memmap(base.WORK/name,dtype='<f4',mode='r');y=np.memmap(original_work/name,dtype='<f4',mode='r')
        checks[name]=base.difference(x,y);assert checks[name]['uint32_bit_differences']==0
    r['exact_same_function_vs_weight_stationary']=checks
    r['native_delta_vs_weight_stationary']={key:r['native'][key]-old['native'][key]
        for key in ('gate_read_bytes','weight_fill_bytes','directory_entries','directory_decode_slots','active_ADD_vector_issues','raw_spill_bytes')}
    path.write_text(json.dumps(r,indent=2)+'\n')
    print('TWO_BLOCK_DIRECTORY_COMPLETE',[(x['mode'],x['service_slots'],x['reduction_vs_weight_stationary']) for x in r['arms']],flush=True)


if __name__=='__main__':main()
