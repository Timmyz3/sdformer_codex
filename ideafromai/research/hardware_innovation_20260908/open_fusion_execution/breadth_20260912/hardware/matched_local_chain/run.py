from pathlib import Path
import argparse
import ctypes
import importlib.util
import json
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
BREADTH=HERE.parents[1]
OPEN=BREADTH.parent
sys.path.insert(0,str(HERE.parent))
import packed_weights as shared
stage,consumer=shared.stage,shared.consumer
common,windows=stage.common,stage.run_windows
sys.path.insert(0,str(OPEN/'pruning'))
import consumer_execute as oracle
import source_program


def rne(x,shift):
    x=np.asarray(x,np.int64);assert np.max(np.abs(x))<2**47
    if shift:
        q,rem=np.divmod(x,1<<shift)
        q+=((2*rem>(1<<shift))|((2*rem==(1<<shift))&((q&1)!=0)))
    else:q=x
    return np.clip(q,-2**23,2**23-1)


def source_gold(identity,q,structure):
    shape=identity.shape;x=identity.reshape(10,-1).astype(np.int64)
    if structure=='lifting40':
        x=x.copy()
        for layer in range(4):
            i,j=q['lifting_matchings'][layer].T;a,b=x[i].copy(),x[j].copy()
            first=rne(4096*a+q['lifting_q12'][layer,:,0,None]*b,12)
            second=rne(4096*b+q['lifting_q12'][layer,:,1,None]*first,12)
            x[i],x[j]=first,second
        x=x[q['source_permutation'].astype(int)]
    else:x=rne(q['As_q16'].astype(np.int64)@x,int(q['As_exponent']))
    cutoff=q['source_threshold'][:,None];direction=q['source_direction'][:,None];constant=q['source_constant'][:,None]
    return np.where(constant>=0,constant.astype(bool),np.where(direction>0,x>=cutoff,x<=cutoff)).reshape(shape)


def install_source(program):
    # Reuse the existing source implementation unchanged except the actual
    # program loader. This isolated function namespace never changes old files.
    import inspect
    text=inspect.getsource(source_program.run)
    line=next(s for s in text.splitlines() if 'program=json.loads' in s)
    text=text.replace(line,'    program=NEW_PROGRAM')
    space=dict(source_program.__dict__);space['NEW_PROGRAM']=program
    exec(compile(text,str(HERE/'new_source_binding'), 'exec'),space)
    windows_source=space['run']
    source_program.run=windows_source


def preview_gold(source,p,geo):
    work=Path('/tmp/matched_local_chain_20260912');work.mkdir(exist_ok=True)
    binary=work/'preview.so'
    if not binary.exists() or binary.stat().st_mtime<(HERE/'preview_reference.cpp').stat().st_mtime:
        subprocess.run(['g++','-O3','-march=native','-ffp-contract=off','-shared','-fPIC',str(HERE/'preview_reference.cpp'),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary));f=lib.preview
    fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');bp=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    f.argtypes=[bp]+[ctypes.c_int]*8+[fp]*6+[ctypes.c_float]+[fp]*3+[bp,fp];f.restype=None
    h,w=geo['gate_shape'];sh,sw=source.shape[2:];sy,sx=geo['source_origin'];oy,ox=geo['gate_origin']
    inv=np.float32(1)/np.sqrt(np.float32(p['bn1_var'])+np.float32(p['bn1_eps']))
    scale=np.float32(p['bn1_gamma']*inv);bias=np.float32(p['bn1_beta']-np.float32(p['bn1_mean']*scale))
    args=[p['preview_u'][:,:32],p['preview_v'][:32],scale,bias,p['preview_A'],p['preview_b']]
    z=np.empty((10,32,h,w),np.float32);raw=np.empty((10,96,h,w),np.float32);bn=np.empty_like(raw);gate=np.empty(raw.shape,np.uint8);margin=np.empty_like(raw)
    f(np.ascontiguousarray(source,np.uint8),sh,sw,sy,sx,h,w,oy,ox,*[np.ascontiguousarray(x,np.float32) for x in args],float(p['preview_theta_output']),z,raw,bn,gate,margin)
    return dict(z=z,raw=raw,y=bn,gate=gate.astype(bool)),dict(min_abs_threshold_margin=float(np.abs(margin).min()),
        near_threshold_1e_5=int(np.count_nonzero(np.abs(margin)<1e-5)),
        function='Independent C++ ascending fullK FP32 fmaf, TF32 U/V input completion, fixed BN MUL then ADD, noncausal sn2 time FMA. No Machine-derived gold.')


def run(structure,label):
    folder=common.FULL/'capture/ordinary'
    original=common.read_npz(folder/'000_zurich_city_09_a_0001.npz')
    live=common.read_npz(folder/'live_parameters.npz')
    q=common.read_npz(BREADTH/'algorithm/matched_training'/structure/'stage320/deployed_constants.npz')
    program=json.loads((BREADTH/'source_execution'/structure/'program.json').read_text())
    geo=json.loads(str(original['window_geometry_json']))[label]
    identity=original[label+'_I24'];source=source_gold(identity,q,structure)
    preview,margin=preview_gold(source,live,geo)
    expected,oracle_info=oracle.independent_gold(original,q,label,preview['gate'])
    view=dict(original)
    view.update({label+'_sn1_gate':source,label+'_preview_Z_shared':preview['z'],label+'_preview_shared_raw':preview['raw'],
        label+'_preview_BN1_Y':preview['y'],label+'_sn2_gate':preview['gate'],label+'_updated_I24':expected['updated'],
        label+'_proj_gate':expected['gate'],label+'_continuous_q24':expected['continuous']})
    install_source(program)
    windows.Machine=stage.IntegratedMachine;windows.build_nrv=stage.preview_directory
    value,producer,m=windows.window(view,live,label,False,False,True,structure)
    assert all(x['differences']==0 for x in producer['checks'].values()),producer['checks']
    assert producer['source_program']['checks']['differences']==0
    producer_end=m.time;producer_counts=dict(m.count);m.forward_i24=True
    final,report=consumer.run(view,q,label,None,stress=False,machine=m,rank=24)
    assert all(x['differences']==0 for x in report['checks'].values())
    assert report['service_slots']==m.time-producer_end and sum(m.stages.values())==m.time
    assert 'sn2_continuation_input' not in m.stages and not m.count.get('DMA_output_slots',0)
    np.savez_compressed(HERE/(structure+'_'+label+'_cpu_endpoints.npz'),source_gate=source,preview_gate=preview['gate'],
        cpu_preview_Z=preview['z'],cpu_preview_raw=preview['raw'],cpu_preview_BN1=preview['y'],
        updated=final['updated'],projection_gate=final['gate'],PED=final['continuous'])
    source_check=common.difference(source,original[label+'_sn1_gate'])
    return dict(structure=structure,window=label,service_slots=m.time,stages=dict(m.stages),counts=dict(m.count),
        physical_port_bytes=dict(SR64=8*m.count['SR64_reads'],SW64=8*m.count['SW64_writes'],CR256=32*m.count['CR256_reads'],CW256=32*m.count['CW256_writes']),
        source_program=producer['source_program'],producer=producer,consumer=report,
        integration=dict(one_Machine=True,producer_end=producer_end,consumer_begin=producer_end,consumer_end=m.time,
            resident_sn2_handoff=True,raw_I24_unchanged_and_reread=True,preview_coefficient_bytes=producer_counts['CW256_writes']*32,
            integer_coefficient_bytes=(m.count['CW256_writes']-producer_counts['CW256_writes'])*32),
        cpu_gold=dict(source_literal=True,preview=margin,consumer=oracle_info,source_delta_vs_old_parent=source_check,
            source_fields='Explicit structure, literal As/lifting, RNE/sat, source_threshold/direction/constant; no historical readout recompile.',
            consumer_fields='Actual new U_conv2/F/U_ped/V_ped, exponent, literal cutoff/permutation and BN2/PED bias arrays.'),
        gate_activity=dict(source=int(source.sum()),source_total=int(source.size),sn2=int(preview['gate'].sum()),sn2_total=int(preview['gate'].size),
            projection=int(final['gate'].sum()),projection_total=int(final['gate'].size)),
        resources=dict(RF_vectors=96,lanes=8,word_bits=48,state_bytes=131072,coefficient_bytes=131072,ROM_bytes=8192,
            state_ports='SR64/SW64',coefficient_port='CR256',external='32B/5slots',shared_H4_coalescer=True,shared_resident_source_MAC=True),
        new_stage320_constants=True,old_stage_service_inherited=False,new_GPU_endpoint_capture=False,
        evidence='CPU actual payload same-Machine local service, independent same-arithmetic gold. Source has separate existing RTL checks; this whole chain is not RTL/PPA.',
        limitations=['Frozen preview scalar FP32 order can differ from GPU; no new GPU snippet has been compared.',
            'This candidate view contains newly computed gold, not an original capture replacement.',
            'No native projection/globalBN/join service or full-frame/I24 producer scope. No AEE inherited from old endpoint.'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('structure',choices=['dense','contiguous34','lifting40']);ap.add_argument('window',choices=['corner','interior']);a=ap.parse_args()
    print('MATCHED_LOCAL_START',a.structure,a.window,flush=True)
    result=run(a.structure,a.window)
    (HERE/(a.structure+'_'+a.window+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('MATCHED_LOCAL_COMPLETE',a.structure,a.window,result['service_slots'],result['gate_activity'],flush=True)


if __name__=='__main__':main()
