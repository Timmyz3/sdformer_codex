"""Execute the real full BN domain, starting from captured raw projection.

An independent full-domain operator test, not a sum with the4x4 local chain.
The other spatial projection values have not been hardware-replayed here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from run_chain import HERE,read_npz,difference


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');ap.add_argument('--pairwise',action='store_true');args=ap.parse_args()
    work=Path('/tmp/tcasii_global_bn_work_20260911')/(('pairwise_' if args.pairwise else '')+('stress' if args.stress else 'ready'));work.mkdir(parents=True,exist_ok=True)
    binary=work/'global_bn'
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off',str(HERE/'global_bn.cpp'),'-o',str(binary)],check=True)
    result=dict(scope=__doc__,axes={},full_layer_chain_closed=False,new_training=False,new_quantization=False,
        state_bytes=131072,coefficient_bytes=131072,RF_vector_words=96,lanes=8,
        arithmetic='FP32 explicit RNE; four interleaved partial sums; centered two-pass variance; bit seed + three Newton rsqrt iterations; no free SFU',
        stress='fixed SR last8/SW last4 of32 blocked' if args.stress else 'always_ready')
    for axis in ('ordinary','lifting_raw'):
        path=HERE.parent/'capture'/axis;data=read_npz(path/'000_zurich_city_09_a_0001.npz');p=read_npz(path/'live_parameters.npz')
        x=data['proj_bn_full_input_fp32'];shape=x.shape
        packed=x.transpose(0,2,3,1).copy().reshape(-1,96)
        input_path=work/(axis+'_input.f32');packed.tofile(input_path)
        constants=np.concatenate([p['proj_bn_gamma'],p['proj_bn_beta'],np.full(8,1/192000),np.full(8,0.5),np.full(8,float(p['proj_bn_eps']))]).astype('<f4')
        coefficient_path=work/(axis+'_coeff.f32');constants.tofile(coefficient_path)
        output_path=work/(axis+'_output.f32');stats_path=work/(axis+'_stats.f32')
        print(axis,'start full domain',list(shape),flush=True)
        proc=subprocess.run([str(binary),str(input_path),str(coefficient_path),str(output_path),str(stats_path),str(int(args.stress)),str(int(args.pairwise))],text=True,capture_output=True,check=True)
        report=json.loads(proc.stdout);stats=np.fromfile(stats_path,'<f4').reshape(3,96)
        output=np.fromfile(output_path,'<f4').reshape(shape[0],shape[2],shape[3],shape[1]).transpose(0,3,1,2)
        reference_rsqrt=np.float32(1)/np.sqrt(np.float32(stats[1])+np.float32(p['proj_bn_eps']))
        report.update(actual_domain=list(shape),values=int(x.size),values_per_channel=int(x.size//96),
            external_input_bytes=int(x.nbytes),external_read_bytes=int(3*x.nbytes),external_output_bytes=int(x.nbytes),
            checks=dict(mean=difference(stats[0],data['proj_bn_actual_domain_mean']),variance=difference(stats[1],data['proj_bn_actual_domain_var']),
                rsqrt_vs_correctly_rounded_FP32=difference(stats[2],reference_rsqrt),normalized=difference(output,data['proj_bn_full_output_fp32'])),
            statistics_input_from_capture=False,projection_input_from_capture=True,AEE_revalidated=False,
            full_layer_chain_closed=False)
        result['axes'][axis]=report
        print(axis,report,flush=True)
    result['partial_sum_reduction']='256-value leaves, four interleaved partials per leaf,10-level SRAM merge tree' if args.pairwise else 'four long interleaved sums'
    result['tree_state_bytes']=3840 if args.pairwise else 0
    name='global_bn'+('_pairwise' if args.pairwise else '')+('_stress' if args.stress else '')+'.json'
    (HERE/name).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
