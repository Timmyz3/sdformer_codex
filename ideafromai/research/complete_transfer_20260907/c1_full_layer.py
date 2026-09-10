"""Complete-K ep34 adapter: original outer CPU model, verified exact fast relation.

This does not implement the paper RTL, physical 1RW service, or trained numerics.
"""
import sys
sys.dont_write_bytecode = True
assert sys.version_info[:2] == (3,12)
from pathlib import Path
import contextlib
import hashlib
import io
import json
import subprocess
import time
from collections import Counter

RUNTIME=Path('/tmp/tcasii_complete_baseline_20260907/runtime')
if RUNTIME.exists():sys.path.insert(0,str(RUNTIME))
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
HW=Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
REPO=HW/'third_party/Prosperity'
LEDGER=HW/'results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901/ep34_c1_support16_rows.memh'
LEDGER_SHA='daa6265115df9c0bae5d96e5a133a4b5fbc9786de75598e53ab2e5812bfdb835'
sys.path.insert(0,str(HW/'scripts'))
from run_prosperity_official_probe import load_official_api

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def restore(plan):
    assert sha(LEDGER)==LEDGER_SHA and LEDGER.stat().st_size==466560000
    sample,op=plan['sample'],plan['operator']
    with LEDGER.open('rb') as f:
        f.seek((sample*4+op)*432*3000*9)
        raw=f.read(432*3000*9)
    chars=np.frombuffer(raw,dtype=np.uint8).reshape(-1,9)
    assert np.all(chars[:,:4]==ord('0')) and np.all(chars[:,8]==10)
    lut=np.full(256,255,dtype=np.uint8)
    for i,c in enumerate(b'0123456789abcdef'):lut[c]=i
    nib=lut[chars[:,4:8]]
    assert np.all(nib<16)
    words=(nib.astype(np.uint16)*np.array([4096,256,16,1],dtype=np.uint16)).sum(1,dtype=np.uint16)
    words=words.reshape(432,3000).T
    matrix=((words[:,:,None] >> np.arange(16,dtype=np.uint16))&1).reshape(3000,6912).astype(bool)
    # Independent capture-plane -> padded im2col reconstruction.
    contract_path=HW/'contracts/m1524_ep34_c1_same_ledger_rebind_source_contract_r1_20260831.json'
    binding=json.loads(contract_path.read_text())
    cap=HW/binding['capture_binding']['directory']
    ordered=cap/'unified_ordered_records.jsonl'
    manifest=cap/'manifest.json'
    assert sha(ordered)==binding['capture_binding']['ordered_records_sha256']
    assert sha(manifest)==binding['capture_binding']['manifest_sha256']
    selected=[json.loads(line) for line in ordered.read_text().splitlines()]
    selected=[r for r in selected if r.get('cohort')=='c1' and r.get('category')=='c1_conv3x3'
              and r.get('global_sample_id')==sample and r.get('name')==plan['module']]
    assert len(selected)==1
    row=selected[0];p=row['payload'];support_path=cap/p['support_sign']
    assert sha(support_path)==p['support_sign_sha256']
    payload=support_path.read_bytes();nb=int(p['positive_plane_bytes'])
    assert len(payload)==nb+int(p['negative_plane_bytes']) and not any(payload[nb:])
    support=np.unpackbits(np.frombuffer(payload[:nb],dtype=np.uint8),bitorder='little').reshape(10,768,15,20).astype(bool)
    assert int(support.sum())==row['input']['active']
    windows=np.lib.stride_tricks.sliding_window_view(np.pad(support,((0,0),(0,0),(1,1),(1,1))),(3,3),axis=(2,3))
    independent=windows.transpose(0,2,3,1,4,5).reshape(3000,6912)
    assert np.array_equal(matrix,independent)
    nonzero=binding['c1_population_binding']['exact_two_code_by_module'][op]
    return matrix,{'ledger_sha256':LEDGER_SHA,'selected_raw_bytes_sha256':hashlib.sha256(raw).hexdigest(),
        'support_payload_sha256':p['support_sign_sha256'],'ordered_records_sha256':sha(ordered),
        'independent_im2col_bits_verified':matrix.size,'input_support_nnz':int(support.sum()),
        'im2col_support_nnz':int(matrix.sum()),'threshold_amplitude_from_sealed_contract':nonzero,
        'matrix_T_then_P_sha256':hashlib.sha256(np.packbits(matrix,bitorder='little').tobytes()).hexdigest()}

class Relation:
    def __init__(self):self.cache={};self.calls=0;self.hits=0
    def entry(self,act):
        assert act.ndim==2 and 0<act.shape[1]<=16
        bits=act.numpy() if isinstance(act,torch.Tensor) else act
        masks=(bits.astype(np.uint16)*(1<<np.arange(bits.shape[1],dtype=np.uint16))).sum(1,dtype=np.uint16)
        key=(len(masks),bits.shape[1],masks.tobytes())
        self.calls+=1
        if key in self.cache:self.hits+=1;return self.cache[key]
        n=len(masks);pc=np.array([int(v).bit_count() for v in masks],dtype=np.int16)
        idx=np.arange(n)
        legal=(masks[:,None]&masks[None,:])==masks[None,:]
        legal &= ~((masks[:,None]==masks[None,:])&(idx[None,:]>=idx[:,None]))
        legal &= pc[None,:]>0
        score=np.where(legal,pc[None,:],0)
        parents=score.argmax(1)
        parents=np.where((score.max(1)>0)&(pc>=2),parents,-1)
        parent_masks=np.where(parents>=0,masks[np.maximum(parents,0)],0).astype(np.uint16)
        residual=masks^parent_masks
        output=((residual[:,None] >> np.arange(bits.shape[1],dtype=np.uint16))&1).astype(bool)
        counts={'input_nnz':int(pc.sum()),'residual_nnz':int(output.sum()),
                'preprocess':int((pc>1).sum())+n//8,
                'equal_rows':int(((residual==0)&(masks!=0)).sum())}
        result=(torch.from_numpy(output),torch.from_numpy(parents.astype(np.float32)),counts)
        self.cache[key]=result
        return result
    def __call__(self,act):
        values,parents,_=self.entry(act)
        return values,parents

def fields(stats):
    return {'total_cycles':int(stats.total_cycles),'compute_cycles':int(stats.compute_cycles),
        'preprocess_stall_cycles':int(stats.preprocess_stall_cycles),'mem_stall_cycles':int(stats.mem_stall_cycles),
        'num_ops':int(stats.num_ops),'reads':{k:int(v) for k,v in stats.reads.items()},
        'writes':{k:int(v) for k,v in stats.writes.items()}}

def reference(ordered,accel,product,relation,N=768,conv=True):
    """Independent closed-form N expansion; K stays inside one layer."""
    M,K=ordered.shape;mt=accel.SpMM_tile_size_M;kt=accel.SpMM_tile_size_K;nt=accel.adder_array_size
    MN=(M+mt-1)//mt;KN=(K+kt-1)//kt;NN=(N+nt-1)//nt
    act_tile=mt*kt;wtile=kt*nt*8
    state_a='all' if M*K<=accel.sram_size['act'] else ('row' if act_tile*KN<=accel.sram_size['act'] else 'tile')
    state_w='all' if K*N*8<=accel.sram_size['wgt'] else ('col' if wtile*KN<=accel.sram_size['wgt'] else 'tile')
    counts=Counter();dram_a=0
    for m in range(0,M,mt):
        for k in range(0,K,kt):
            tile=ordered[m:m+mt,k:k+kt]
            c=relation.entry(tile)[2]
            counts.update(c)
            dram_a+=(tile.size//(9 if conv else 1))*(1 if state_a=='all' else NN)
    nnz=counts['residual_nnz'] if product else counts['input_nnz']
    compute=(nnz+counts['equal_rows'])*NN if product else nnz*NN
    pre=counts['preprocess']*NN if product else 0
    compute_total=max(compute,pre)
    dram_w=K*N*8*(MN if state_w=='tile' else 1)
    reads={'dram':dram_a+dram_w,'g_act':M*K*NN,'g_wgt':nnz*N*8,'g_psum':M*N*KN*8}
    writes={'dram':0,'g_act':M*K*(1 if state_a=='all' else NN),'g_wgt':dram_w,'g_psum':M*N*KN*8}
    init_bits=min(kt,K)*min(nt,N)*8+min(kt,K)*min(mt,M)
    mem=init_bits//accel.mem_if_width+max(0,(reads['dram']-init_bits)//accel.mem_if_width-compute_total)
    return {'total_cycles':compute_total+mem,'compute_cycles':compute_total,'preprocess_stall_cycles':max(0,pre-compute),
            'mem_stall_cycles':mem,'num_ops':nnz*N,'reads':reads,'writes':writes},dict(counts),{'act':state_a,'weight':state_w}

def run(FC,Simulator,accel,matrix,T,P,N,callback=None):
    assert matrix.shape==(T*P,matrix.shape[1])
    op=FC('ep34_resblock0_conv1_kernel3_fc',matrix.shape[1],N,P,1,T)
    op.activation_tensor.sparse_map=torch.from_numpy(matrix.copy()).reshape(T,P,-1)
    sim=Simulator(accel,[op],benchmark_name='ep34_adapter',use_cuda=False)
    scope=Simulator.run_fc.__globals__
    original=scope['find_product_sparsity']
    scope['clear_global_stats']()
    if callback is not None:scope['find_product_sparsity']=callback
    output=io.StringIO();start=time.monotonic()
    try:
        with contextlib.redirect_stdout(output):stats=sim.run_fc(op)
    finally:scope['find_product_sparsity']=original
    # This additionally proves the one intended T/P permutation at entry.
    expected=matrix.reshape(T,P,-1).transpose(1,0,2).reshape(T*P,-1)
    assert np.array_equal(op.activation_tensor.sparse_map.reshape(T*P,-1).numpy(),expected)
    return fields(stats),output.getvalue(),time.monotonic()-start

def main():
    target=ROOT/'c1_full_layer_r1.json';assert not target.exists()
    plan_path=ROOT/'c1_full_layer_plan.json';plan=json.loads(plan_path.read_text());plan_sha=sha(plan_path)
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    assert commit=='6ee1c6f1cb419fcf942f2eda63db84ca28248f4b'
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    source_sha={name:sha(REPO/'simulator'/name) for name in ['simulator.py','accelerator.py','networks.py','utils.py']}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    Accelerator,FC,Simulator,_=load_official_api()
    original=Simulator.run_fc.__globals__['find_product_sparsity']
    matrix,identity=restore(plan)
    ordered=matrix.reshape(10,300,6912).transpose(1,0,2).reshape(3000,6912)
    rng=np.random.default_rng(3406912);relation=Relation();checks=[]
    # Directed relations include zeros, singleton duplicates and signed index ties.
    directed=np.array([0,1,1,3,7,11,3,15],dtype=np.uint16)
    arrays=[((directed[:,None]>>np.arange(16))&1).astype(bool)]
    for rows in [4,8,64,256]:
        arrays.extend(rng.random((rows,16))<p for p in [.02,.15,.5])
    for start in [0,64,256,2944]:
        for k in [0,16,3456,6896]:arrays.append(ordered[start:min(start+64,3000),k:k+16])
    for arr in arrays:
        a,b=original(torch.from_numpy(arr.copy()));x,y=relation(torch.from_numpy(arr))
        assert torch.equal(a,x) and torch.equal(b,y)
        checks.append({'rows':len(arr),'columns':arr.shape[1],'nonzeros':int(arr.sum())})
    print('Input reconstructed and relation checks passed:',len(checks),flush=True)
    outer_checks=[]
    for T,P,K,N,mt,nt in [(2,3,19,129,4,96),(3,5,35,197,8,128),(2,9,32,96,16,96)]:
        a=rng.random((T*P,K))<.23
        for mode in [False,True]:
            acc=Accelerator('Prosperity',nt,32,mt,16,product_sparsity=mode,issue_type=2,mem_if_width=1024)
            direct,_,_=run(FC,Simulator,acc,a,T,P,N)
            fast,_,_=run(FC,Simulator,acc,a,T,P,N,relation)
            ref,_,_=reference(a.reshape(T,P,K).transpose(1,0,2).reshape(T*P,K),acc,mode,relation,N)
            assert direct==fast==ref,(T,P,K,N,mode,direct,ref)
            outer_checks.append({'shape':[T,P,K,N],'tile':[mt,16,nt],'product':mode,'counts':direct})
    print('Original multi-K/multi-N outer path checks passed:',len(outer_checks),flush=True)
    points=[]
    for cfg in plan['configs']:
        relation=Relation();modes={}
        for mode in [False,True]:
            acc=Accelerator('Prosperity',cfg['N_tile'],32,cfg['M_tile'],16,product_sparsity=mode,issue_type=2,mem_if_width=1024)
            got,stdout,elapsed=run(FC,Simulator,acc,matrix,10,300,768,relation)
            want,counts,residency=reference(ordered,acc,mode,relation)
            assert got==want,(cfg['id'],mode,got,want)
            modes['product' if mode else 'bit']={'official_outer_counts':got,'relation_counts_one_N_slice':counts,
                'artifact_residency':residency,'stdout':stdout,'host_elapsed_seconds_not_hardware':elapsed}
            print(cfg['id'],'product' if mode else 'bit','official model cycles',got['total_cycles'],flush=True)
        split={}
        for mode in [False,True]:
            acc=Accelerator('Prosperity',cfg['N_tile'],32,cfg['M_tile'],16,product_sparsity=mode,issue_type=2,mem_if_width=1024)
            totals=Counter();rr=Counter();ww=Counter()
            for k in range(0,6912,16):
                r,_,_=reference(ordered[:,k:k+16],acc,mode,relation)
                for key,value in r.items():
                    if key=='reads':rr.update(value)
                    elif key=='writes':ww.update(value)
                    else:totals[key]+=value
            split['product' if mode else 'bit']={**dict(totals),'reads':dict(rr),'writes':dict(ww)}
        points.append({'config':cfg,'artifact_sram_bits':acc.sram_size,'modes':modes,
                       'incorrectly_split_K16_layer_sum_diagnostic':split,
                       'within_official_model_product_over_bit_speedup':modes['bit']['official_outer_counts']['total_cycles']/modes['product']['official_outer_counts']['total_cycles'],
                       'callback':{'invocations_including_reference_and_split_checks':relation.calls,'memo_hits':relation.hits,'distinct_tiles':len(relation.cache)}})
    assert sha(plan_path)==plan_sha
    assert all(sha(REPO/'simulator'/name)==value for name,value in source_sha.items())
    report={'status':'COMPLETE_SINGLE_OPERATOR_OFFICIAL_OUTER_MODEL_REFERENCE_ONLY','date':'2026-09-07',
        'plan':plan,'plan_sha256':plan_sha,'script_sha256':sha(Path(__file__)),
        'loader_sha256':sha(HW/'scripts/run_prosperity_official_probe.py'),
        'official_commit':commit,'official_source_sha256':source_sha,'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},
        'input_identity':identity,'original_relation_crosschecks':checks,'original_outer_crosschecks':outer_checks,'points':points,
        'limits':['Official default sram sizes and 8-bit outputs are not paper TableIII or local mapped resources.',
                  'Unchanged run_fc with validated exact accelerated subset callback is not the unmodified full official API.',
                  'Sorting issue2 is zero-cost in artifact; max of full-layer preprocessing/compute sums idealizes overlap.',
                  'Artifact conv DRAM division by kernel squared is a heuristic; no exact linebuffer/DMA claim.',
                  'No trained coefficients or numeric convolutions are evaluated, and theta amplitude is recorded rather than approximated as1.',
                  'Only one complete operator/sample; no system, PPA, FPGA/ASIC speedup or acceptance claim.']}
    with target.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')

if __name__=='__main__':main()
