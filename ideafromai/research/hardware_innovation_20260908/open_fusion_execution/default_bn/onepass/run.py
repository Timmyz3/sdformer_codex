"""Full-domain one-pass BN baseline with explicit changed floating-point order.

This is a BNFF/FlexAcc-inspired common baseline, not a new mechanism. All
statistics are computed from payloads. Dense and tagged variants share the
same two-stripe/256-leaf arithmetic; the old centered tree is a numerical
reference, not an expected bit-exact result of the changed formula.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('complete_default_bn',HERE.parent/'run.py')
source=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def difference(a,b):
    a=np.asarray(a,dtype=np.float32).reshape(-1)
    b=np.asarray(b,dtype=np.float32).reshape(-1)
    assert a.shape==b.shape
    count=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
    delta=a.astype(np.float64)-b.astype(np.float64)
    return dict(values=int(a.size),bit_differences=count,
        max_abs=float(np.max(np.abs(delta))),rms=float(np.sqrt(np.mean(delta*delta))))


def file_difference(a,b):
    return difference(np.memmap(a,dtype='<f4',mode='r'),np.memmap(b,dtype='<f4',mode='r'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stress',action='store_true')
    args=ap.parse_args()
    profile='stress' if args.stress else 'ready'
    work=Path('/tmp/flexacc_onepass_20260912')/profile
    work.mkdir(parents=True,exist_ok=True)
    (work/'global_bn_engine.hpp').write_text((source.OLD/'global_bn.cpp').read_text().split('\nint main(')[0]+'\n')
    (work/'default_stream.hpp').write_text((HERE.parent/'default_bn.cpp').read_text().split('\nint main(')[0]+'\n')
    binary=work/'onepass'
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I',str(work),str(HERE/'onepass.cpp'),'-o',str(binary)],check=True)
    prior=json.loads((HERE.parent/f'results_{profile}.json').read_text())
    result=dict(scope=__doc__,profile=profile,evidence='CPU actual-payload port service model; not RTL cycles, CUDA identity, AEE or PPA.',
        formula='mean=sum(x)/n; variance=sum(x*x)/n - mean*mean; n=192000 including all code0 positions.',
        arithmetic='FP32 two stripes for sum plus two stripes for FMA(x,x,sum_sq); paired256-leaf tree; original rsqrt seed and three Newton iterations; separate affine MUL then ADD.',
        resources=dict(state_bytes=131072,coefficient_bytes=131072,RF_vectors=96,lanes=8,
            sum_RF=list(range(24)),sum_square_RF=list(range(24,48)),tag_RF=list(range(88,92)),
            tree_state_bytes=7680,tree_state_start=16384,tag_state_bytes=32,
            state_ports='1R64/1W64',coefficient_port='1R256',FP_latency=4),
        source_scope='Captured complete-K864 projection and source-derived tags. Native producer and PED costs excluded; all normalized output materialized.',
        axes={})
    path=HERE/f'results_{profile}.json'
    for axis in (['ordinary'] if args.stress else ['ordinary','lifting_raw']):
        aw=work/axis;aw.mkdir(exist_ok=True)
        info=source.prepare(axis,aw)
        result['axes'][axis]=dict(source=info,arms={})
        oldwork=Path('/tmp/default_bn_20260912')/profile/axis
        capture=source.FULL/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz'
        with np.load(capture) as z:
            cuda=z['proj_bn_full_output_fp32'].transpose(0,2,3,1).reshape(-1)
            cuda_mean=z['proj_bn_actual_domain_mean'].reshape(-1)
            cuda_var=z['proj_bn_actual_domain_var'].reshape(-1)
        for tagged,name in ((False,'dense'),(True,'tagged_default')):
            output=aw/f'{name}_output.f32';stats=aw/f'{name}_stats.f32'
            print(axis,name,'start',flush=True);started=time.monotonic()
            cmd=[str(binary),str(aw/'dense.f32'),str(aw/'coeff.f32'),str(aw/'tags.u8'),str(aw/'live.f32'),
                str(output),str(stats),str(int(args.stress)),str(int(tagged))]
            proc=subprocess.run(cmd,text=True,capture_output=True,check=True)
            arm=json.loads(proc.stdout);arm['wall_seconds']=time.monotonic()-started
            arm['stage_names']=['setup','joint_moments','variance_rsqrt_affine','unused','normalize_materialize']
            assert sum(arm['stages'])==arm['service_slots']
            arm['external_input_bytes']=arm['payload_bytes']+arm['tag_bytes']
            arm['external_output_bytes']=192000*96*4
            arm['physical_port_bytes']=dict(state_read=8*arm['SR64_reads'],state_write=8*arm['SW64_writes'],
                coefficient_read=32*arm['CR256_reads'],coefficient_write=32*arm['CW256_writes'])
            arm['same_formula_dense_bits']=dict(output=source.bit_check(output,aw/'dense_output.f32'),
                stats=source.bit_check(stats,aw/'dense_stats.f32'))
            arm['changed_order_vs_centered_tree']=dict(output=file_difference(output,oldwork/'dense_output.f32'),
                stats=file_difference(stats,oldwork/'dense_stats.f32'))
            values=np.memmap(output,dtype='<f4',mode='r');sv=np.fromfile(stats,dtype='<f4').reshape(3,96)
            arm['numeric_vs_cuda_capture']=dict(output=difference(values,cuda),mean=difference(sv[0],cuda_mean),
                variance=difference(sv[1],cuda_var))
            arm['min_variance']=float(sv[1].min())
            oldname='default_ops' if tagged else 'dense'
            oldslots=prior['axes'][axis]['arms'][oldname]['service_slots']
            arm['same_format_centered_tree_service']=oldslots
            arm['service_reduction_vs_same_format_centered_tree']=1-arm['service_slots']/oldslots
            result['axes'][axis]['arms'][name]=arm
            path.write_text(json.dumps(result,indent=2)+'\n')
            print(axis,name,arm['service_slots'],arm['numeric_vs_cuda_capture']['output'],'SAME_FORMULA_BITS_EQUAL',flush=True)


if __name__=='__main__':main()
