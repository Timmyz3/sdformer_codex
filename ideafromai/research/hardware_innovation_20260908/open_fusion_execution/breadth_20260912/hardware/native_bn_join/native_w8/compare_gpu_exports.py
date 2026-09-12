"""Compare captured new W8 GPU windows/statistics with the measured compact CPU arm."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
BREADTH=HERE.parents[2]
WORK=Path('/tmp/native_bn_join_20260912/native_w8/compact')

def compare(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    return dict(values=int(a.size),bit_differences=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),max_abs=float(np.max(np.abs(a.astype('f8')-b.astype('f8')))))

def main():
    g=np.load(BREADTH/'algorithm/native_w8_aee/first_frame_local_outputs.npz')
    checks={}
    for filename,key in [('native_raw.f32','proj_conv_fp32'),('materialized_BN.f32','proj_norm_fp32'),('fused_final.f32','ped_output_fp32')]:
        a=np.memmap(WORK/filename,dtype='<f4',mode='r',shape=(10,120,160,96))
        for label,(y,x) in {'corner':(0,0),'interior':(60,80)}.items():
            v=a[:,y:y+4,x:x+4,:].transpose(0,3,1,2).copy()
            checks[label+'_'+key]=compare(v,g[label+'_'+key])
    checks['whole_domain_statistics']=compare(np.fromfile(WORK/'statistics.f32','<f4').reshape(5,96),g['proj_bn_onepass_statistics'])
    result=dict(scope='Fresh W8 first-frame two4x4 output windows plus complete-domain onepass statistics. No old original-weight GPU output used.',checks=checks,all_equal=all(x['bit_differences']==0 for x in checks.values()),full_raw_capture=False)
    (HERE/'gpu_alignment.json').write_text(json.dumps(result,indent=2)+'\n')
    print('W8_GPU_CHECK',result['all_equal'],sum(x['values'] for x in checks.values()))

if __name__=='__main__':main()
