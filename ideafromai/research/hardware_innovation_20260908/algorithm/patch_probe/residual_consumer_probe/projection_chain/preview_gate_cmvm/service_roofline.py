"""Actual preview V-DAG resource accounting, not a scheduler or RTL cycle model.

Fixed first student only; captured f14 train4/diverse10 and real f6 diverse10.
f14 exact whole, f14 exact signed-high / unsigned-low8, f6 whole are compared
under the same ideal bit-ALU/RF interface. f6 is a different quantized student.

Every executed DAG add/sub reads both stored operands and writes one result;
static shifts do not cause new arithmetic nodes. No unbounded forwarding/cache
is credited. This is a declared RF ledger, not a universal minimum-access proof.
Roofline is only a lower bound conditional on that ledger and ideal dispatch.
The source producer/width detector and instruction memories are not simulated.

Exact output sign/shift aliases are folded into directed integer thresholds.
After high8 completes, at most two precompiled integer comparisons per output
prove the full gate. An undecided gate retains its normalized coarse output,
and the needed unsigned-low8 DAG ancestors run once. Fine inputs are unioned
over the real graph roots, not discounted by the fraction of arithmetic nodes.
The already-stored coarse terminal is held across fine execution: not written
a second time. Signed24/f14 source integers are never re-quantized to fabricate f6.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from functools import lru_cache
from collections import Counter
import json
import math
import numpy as np

HERE=Path(__file__).resolve().parent
CHAIN=HERE.parent
DATA=CHAIN/'temporal_structured_recovery'
AXIS='identity_permuted_base'
LOW=8
from scan_gate_certificates import save, minimum_signed_width


def scalar_bits(lo,hi):
    """Exact stored range, unsigned for a nonnegative quantity."""
    lo,hi=int(lo),int(hi)
    if lo==hi==0: return 0
    if lo>=0: return max(1,hi.bit_length())
    return max((-lo-1).bit_length()+1, max(0,hi).bit_length()+1)


def widths(lo,hi):
    return np.array([scalar_bits(l,h) for l,h in zip(lo,hi)],np.int64)


def signed_constant_bits(values):
    x=np.asarray(values)
    return np.array([max(1,(int(v) if v>=0 else ~int(v)).bit_length()+1)
                     for v in x.ravel()],np.int64).reshape(x.shape)


def ceildiv(x,d):
    return -((-int(x))//d)


class ActualDAG:
    def __init__(self,path):
        self.graph=json.loads(Path(path).read_text())
        g=self.graph
        self.nodes=g['nodes']
        self.n=len(self.nodes)
        self.ops=self.nodes[32:]
        self.coeff=np.array([n['coeff'] for n in self.nodes],np.int64)
        self.pos=np.maximum(self.coeff,0).sum(axis=1)
        self.neg=np.minimum(self.coeff,0).sum(axis=1)
        self.out=np.array([o['node'] for o in g['outputs']],np.int64)
        self.shift=np.array([o['shift'] for o in g['outputs']],np.int64)
        self.sign=np.array([o['sign'] for o in g['outputs']],np.int64)
        self.K=np.array(g['numerator_K_output_by_input'],np.int64)
        self.O=self.coeff[self.out]
        assert np.array_equal((self.O*self.sign[:,None])*(1 << self.shift[:,None]),self.K)
        assert len(set(self.out))==96
        assert np.count_nonzero(self.K)==32*96
        self.lhs=np.array([n['lhs'] for n in self.ops],np.int64)
        self.rhs=np.array([n['rhs'] for n in self.ops],np.int64)
        self.lshift=np.array([n['lhs_shift'] for n in self.ops],np.int64)
        self.rshift=np.array([n['rhs_shift'] for n in self.ops],np.int64)
        masks=[int(n['terminal_output_mask_hex'],16) for n in self.nodes]
        self.masklo=np.array([m&((1 << 64)-1) for m in masks],np.uint64)
        self.maskhi=np.array([m >> 64 for m in masks],np.uint64)
        self.powerslo=np.array([1 << h for h in range(64)],np.uint64)
        self.powershi=np.array([1 << h for h in range(32)],np.uint64)

    def active(self,outputs):
        lower=outputs[:,:64].astype(np.uint64)@self.powerslo
        upper=outputs[:,64:].astype(np.uint64)@self.powershi
        return ((lower[:,None]&self.masklo[None,:])!=0)|((upper[:,None]&self.maskhi[None,:])!=0)

    @lru_cache(None)
    def profile(self,bits,unsigned=False,input_storage_bits=None):
        low,high=(0,(1 << bits)-1) if unsigned else (-(1 << (bits-1)),(1 << (bits-1))-1)
        lo=self.pos*low+self.neg*high
        hi=self.pos*high+self.neg*low
        stored=widths(lo,hi)
        if input_storage_bits is not None:
            stored[:32]=input_storage_bits
        left=widths(lo[self.lhs]*(1 << self.lshift),hi[self.lhs]*(1 << self.lshift))
        right=widths(lo[self.rhs]*(1 << self.rshift),hi[self.rhs]*(1 << self.rshift))
        alu=np.maximum.reduce([left,right,widths(lo[32:],hi[32:])])
        read=stored[self.lhs]+stored[self.rhs]
        root_read=np.where(self.lhs<32,stored[self.lhs],0)+np.where(self.rhs<32,stored[self.rhs],0)
        return dict(lo=lo,hi=hi,stored=stored,alu=alu,read=read,write=stored[32:],
                    root_read=root_read,word32=(alu+31)//32)

    def stage(self,outputs,profile):
        active=self.active(outputs)
        keep=active[:,32:].astype(np.int64)
        return dict(dag_nodes=int(keep.sum()),
                    arithmetic_bits=int((keep@profile['alu']).sum()),
                    rf_read_bits=int((keep@profile['read']).sum()),
                    root_operand_read_bits=int((keep@profile['root_read']).sum()),
                    rf_write_bits=int((keep@profile['write']).sum()),
                    arithmetic_word32_ops=int((keep@profile['word32']).sum()),
                    source_union_components=int(active[:,:32].sum()),
                    source_union_read_bits=int((active[:,:32]*profile['stored'][None,:32]).sum()))


def normalized_parameters(a,dag):
    threshold=a['threshold'].astype(np.int64)
    sense=a['sense'].astype(np.int64)
    effective=sense*dag.sign
    tau=np.zeros_like(threshold)
    for t in range(10):
        for h in range(96):
            v=int(dag.sign[h])*int(threshold[t,h]);d=1 << int(dag.shift[h])
            tau[t,h]=ceildiv(v,d) if effective[h]>0 else v//d
    constants=np.asarray(a['constant_gate'],np.int8)
    if constants.ndim==1: constants=np.broadcast_to(constants,(10,96))
    assert np.isin(constants[:,sense==0],[0,1]).all()
    return threshold,sense,tau,effective,constants


def exact_gate(value,tau,sense,constant):
    gate=np.where(sense[None,:]>0,value>=tau,value<=tau)
    return np.where(sense[None,:]==0,constant.astype(bool),gate)


def predicate_nonconstant(lo,hi,threshold,greater):
    return np.where(greater[None,:],(lo<threshold)&(hi>=threshold),
                    (lo<=threshold)&(hi>threshold))


def initial_domain(X,W,tau,sense,constants,profile,dag):
    lo=profile['lo'][dag.out][None,:]
    hi=profile['hi'][dag.out][None,:]
    yes=np.where(sense[None,:]>0,lo>=tau,hi<=tau)
    no=np.where(sense[None,:]>0,hi<tau,lo>tau)
    constant=sense[None,:]==0
    live=~(yes|no|constant)
    zero=np.all(X==0,axis=1)
    live[zero]=False
    return live,zero


def accumulator():
    return Counter()


def add_stage(total,stage,label):
    for key,value in stage.items():
        total[label+'_'+key]+=int(value)
    total['dag_nodes']+=stage['dag_nodes']
    total['arithmetic_bits']+=stage['arithmetic_bits']
    total['rf_read_bits']+=stage['rf_read_bits']
    total['rf_write_bits']+=stage['rf_write_bits']
    total['arithmetic_word32_ops']+=stage['arithmetic_word32_ops']
    total['issued_operations_ideal_dispatch']+=stage['dag_nodes']


def add_comparisons(total,mask,value_bits,threshold,label):
    n=int(mask.sum())
    threshold_bits=signed_constant_bits(threshold)
    comparison_bits=np.maximum(np.broadcast_to(value_bits,mask.shape),threshold_bits)
    bitwork=int((mask*comparison_bits).sum())
    words=int((mask*((comparison_bits+31)//32)).sum())
    total[label+'_comparisons']+=n
    total[label+'_comparison_bitwork']+=bitwork
    total['comparison_count']+=n
    total['comparison_bitwork']+=bitwork
    total['comparison_word32_ops']+=words
    total['threshold_read_bits']+=int((mask*threshold_bits).sum())
    total['rf_read_bits']+=int((mask*threshold_bits).sum())
    total['issued_operations_ideal_dispatch']+=n


def count_group(total,X,times,tau_table,sense,constants,W,dag,split):
    size=len(X)
    tau=tau_table[times]
    constant=constants[times]
    full_profile=dag.profile(W)
    live,zero=initial_domain(X,W,tau,sense,constant,full_profile,dag)
    has_work=live.any(axis=1)
    full=X@dag.O.T
    truth=exact_gate(full,tau,sense,constant)
    total['vectors']+=size
    total['gates']+=size*96
    total['zero_vector_bypass']+=int(zero.sum())
    total['domain_bypass_nonzero_vectors']+=int((~zero&~has_work).sum())
    total['initial_domain_or_zero_gates']+=int((~live).sum())
    total['sum_input_W']+=W*size
    total['header_read_bits']+=5*size
    total['header_write_bits']+=5*size
    total['rf_read_bits']+=5*size+192*size
    total['rf_write_bits']+=5*size+96*size
    total['domain_ROM_read_bits']+=192*size
    total['final_gate_packet_write_bits']+=96*size
    # Ideal producer/header-first payload commit; its completed-vector buffer
    # and leading-sign detector are outside this CMVM conditional ledger.
    ingress=32*W*int(has_work.sum())
    total['input_payload_commit_bits']+=ingress
    total['rf_write_bits']+=ingress
    total['issued_operations_ideal_dispatch']+=2*size
    if not split:
        phase=dag.stage(live,full_profile)
        add_stage(total,phase,'whole')
        value_bits=full_profile['stored'][dag.out][None,:]
        total['rf_read_bits']+=int((live*value_bits).sum())
        total['final_terminal_read_bits']+=int((live*value_bits).sum())
        add_comparisons(total,live,value_bits,tau,'final')
        return truth
    high=X >> LOW
    low=X-(high << LOW)
    assert np.all((low>=0)&(low<256))
    coarse=high@dag.O.T
    fine_reference=low@dag.O.T
    assert np.array_equal((coarse << LOW)+fine_reference,full)
    hb=max(1,W-LOW)
    coarse_profile=dag.profile(hb)
    fine_profile=dag.profile(LOW,True,min(LOW,W))
    stage=dag.stage(live,coarse_profile)
    add_stage(total,stage,'coarse')
    low_min=dag.neg[dag.out]*255
    low_max=dag.pos[dag.out]*255
    true_threshold=np.empty_like(tau)
    false_threshold=np.empty_like(tau)
    for h in range(96):
        if sense[h]>0:
            true_threshold[:,h]=[-((-int(v)+int(low_min[h]))//256) for v in tau[:,h]]
            false_threshold[:,h]=[(int(v)-1-int(low_max[h]))//256 for v in tau[:,h]]
        else:
            true_threshold[:,h]=[(int(v)-int(low_max[h]))//256 for v in tau[:,h]]
            false_threshold[:,h]=[-((-(int(v)+1-int(low_min[h])))//256) for v in tau[:,h]]
    greater=sense>0
    proven_true=np.where(greater[None,:],coarse>=true_threshold,coarse<=true_threshold)
    proven_false=np.where(greater[None,:],coarse<=false_threshold,coarse>=false_threshold)
    assert not np.any(proven_true&proven_false)
    retired=live&(proven_true|proven_false)
    remaining=live&~(proven_true|proven_false)
    total['checkpoint_row_tests']+=int(live.sum())
    total['checkpoint_retired_gates']+=int(retired.sum())
    total['fine_gates']+=int(remaining.sum())
    total['fine_vectors']+=int(remaining.any(axis=1).sum())
    total['wrong_retirements']+=int(np.count_nonzero(proven_true[retired]!=truth[retired]))
    assert total['wrong_retirements']==0
    clo=coarse_profile['lo'][dag.out][None,:]
    chi=coarse_profile['hi'][dag.out][None,:]
    coarse_bits=coarse_profile['stored'][dag.out][None,:]
    test_true=live&predicate_nonconstant(clo,chi,true_threshold,greater)
    test_false=live&predicate_nonconstant(clo,chi,false_threshold,~greater)
    read_coarse=test_true|test_false
    total['rf_read_bits']+=int((read_coarse*coarse_bits).sum())
    total['checkpoint_coarse_read_bits']+=int((read_coarse*coarse_bits).sum())
    add_comparisons(total,test_true,coarse_bits,true_threshold,'checkpoint_true')
    add_comparisons(total,test_false,coarse_bits,false_threshold,'checkpoint_false')
    # The unresolved bits are stored/read. Their DAG transitive closure and
    # compact ready-queue production are not supplied by this optimistic model.
    total['checkpoint_live_mask_write_bits']+=96*int(has_work.sum())
    total['checkpoint_live_mask_read_bits']+=96*int(remaining.any(axis=1).sum())
    total['rf_write_bits']+=96*int(has_work.sum())
    total['rf_read_bits']+=96*int(remaining.any(axis=1).sum())
    stage=dag.stage(remaining,fine_profile)
    add_stage(total,stage,'fine')
    # K is dense: a nonempty unresolved-output set demands all32 low roots.
    assert stage['source_union_components']==32*int(remaining.any(axis=1).sum())
    total['fine_full_program_issue_if_masked_scan']+=len(dag.ops)*int(remaining.any(axis=1).sum())
    final_bits=full_profile['stored'][dag.out][None,:]
    fine_bits=fine_profile['stored'][dag.out][None,:]
    shifted_bits=widths(coarse_profile['lo'][dag.out]*256,
                        coarse_profile['hi'][dag.out]*256)[None,:]
    combine_bits=np.maximum.reduce([np.broadcast_to(final_bits,remaining.shape),
                                    np.broadcast_to(fine_bits,remaining.shape),
                                    np.broadcast_to(shifted_bits,remaining.shape)])
    combine=int(remaining.sum())
    total['fine_recombine_additions']+=combine
    total['arithmetic_bits']+=int((remaining*combine_bits).sum())
    total['arithmetic_word32_ops']+=int((remaining*((combine_bits+31)//32)).sum())
    total['issued_operations_ideal_dispatch']+=combine
    total['rf_read_bits']+=int((remaining*(coarse_bits+fine_bits)).sum())
    total['fine_terminal_plus_saved_coarse_read_bits']+=int((remaining*(coarse_bits+fine_bits)).sum())
    add_comparisons(total,remaining,final_bits,tau,'final')
    total['issued_operations_ideal_dispatch']+=int(remaining.any(axis=1).sum())
    # Required retained state at the coarse/fine boundary, not a full-run peak.
    held_coarse=(remaining*coarse_bits).sum(axis=1)
    held_source=32*min(W,LOW)*remaining.any(axis=1)
    state=held_coarse+held_source+192*remaining.any(axis=1)
    total['retained_coarse_bits_sum']+=int(held_coarse.sum())
    total['checkpoint_required_state_bits_sum']+=int(state.sum())
    total['max_retained_coarse_bits']=max(total['max_retained_coarse_bits'],int(held_coarse.max()))
    total['max_checkpoint_required_state_bits']=max(total['max_checkpoint_required_state_bits'],int(state.max()))
    total['retained_coarse_write_already_in_DAG_bits']+=int(held_coarse.sum())
    reconstructed=np.where(retired,proven_true,truth)
    assert np.array_equal(reconstructed,truth)
    return reconstructed


def finish(total):
    r={k:int(v) for k,v in total.items()}
    bit_alu=r.get('arithmetic_bits',0)+r.get('comparison_bitwork',0)
    ideal_issue=r.get('issued_operations_ideal_dispatch',0)
    scanned_issue=ideal_issue+r.get('fine_full_program_issue_if_masked_scan',0)-r.get('fine_dag_nodes',0)
    roof=dict(bit_ALU=math.ceil(bit_alu/64),RF_read=math.ceil(r.get('rf_read_bits',0)/128),
              RF_write=math.ceil(r.get('rf_write_bits',0)/64),node_issue=math.ceil(ideal_issue/8))
    word_count=r.get('arithmetic_word32_ops',0)+r.get('comparison_word32_ops',0)
    fixed=dict(two_32bit_ALUs=math.ceil(word_count/2),RF_read=roof['RF_read'],
               RF_write=roof['RF_write'],node_issue=roof['node_issue'])
    r.update(roofline_components=roof,roofline_max=max(roof.values()),
             roofline_fine_full_program_issue=math.ceil(scanned_issue/8),
             roofline_with_full_fine_program_scan=max(*roof.values(),math.ceil(scanned_issue/8)),
             fixed32_counterexample_components=fixed,fixed32_counterexample_max=max(fixed.values()),
             fixed32_total_word_ALU_ops=word_count,
             roofline_max_per_vector=max(roof.values())/r['vectors'],
             scope='conditional resource floor for the stated RF ledger, ideal overlap/dispatch; not scheduled cycles or speedup')
    return r


def run_dataset(folder,dag,split,mode):
    traces=sorted((folder/(AXIS+'_preview_capture')).glob('sample_*.npz'))
    assert len(traces)==(4 if split=='train4' else 10)
    total=accumulator()
    per_frame=[]
    for path in traces:
        a=np.load(path,allow_pickle=False)
        raw_tau,raw_sense,tau,sense,constants=normalized_parameters(a,dag)
        x=a['az_q'].astype(np.int64);times=a['time'].astype(np.int64)
        width=minimum_signed_width(x)
        frame=accumulator()
        for W in np.unique(width):
            selected=width==W
            X=x[selected];ts=times[selected]
            got=count_group(frame,X,ts,tau,sense,constants,int(W),dag,split=mode=='coarse_fine')
            original=exact_gate(X@dag.K.T,raw_tau[ts],raw_sense,constants[ts])
            assert np.array_equal(got,original)
            assert np.array_equal(got,a['expected_gate'][selected])
        frame['exact_gate_mismatches']=0
        record=finish(frame);record['trace']=str(path)
        per_frame.append(record)
        for key,value in frame.items():
            if key.startswith('max_'): total[key]=max(total[key],value)
            else: total[key]+=value
    summary=finish(total)
    summary['per_frame']=per_frame
    network=json.loads((folder/(AXIS+'_summary.json')).read_text())
    summary['network_AEE_frame_mean']=network['AEE_frame_mean']
    summary['network_metric_scope']='existing full-frame network AEE; resource counts cover only fixed128 positions x T10 per captured frame'
    summary['source_directory']=str(folder)
    summary['numerical_identity']='f6 quantized new student' if 'f6' in folder.name else 'f14 exact captured integer function'
    return summary


def main():
    dag=ActualDAG(HERE/'whole_integer_dag.json')
    result=dict(
        complete=False,axis=AXIS,low_bits=LOW,formats='f14 whole and exact low8 split; separate actual f6 whole',
        resources=dict(total_adder_comparator_bits_per_cycle=64,RF_read_bits_per_cycle=128,
                       RF_write_bits_per_cycle=64,issued_nodes_or_comparisons_per_cycle=8,
                       fixed_word_counterexample='two32-bit ALUs; ceil(required width/32) word operations, narrow values still cost one'),
        graph=dict(file='whole_integer_dag.json',nodes=len(dag.ops),outputs=96,inputs=32),
        numeric_contract='exact original output sign/shift folded into directed ceil/floor thresholds; high=X>>8, low=X-256*high; no original-V rounding',
        RF_contract='Each executed DAG node reads both stored operands, writes its result. Output tests share one loaded coarse value; fine recombination streams directly into its final comparator. No second write for retained coarse terminal. Static shifts are wiring, instruction/threshold port assumptions remain explicit.',
        header_contract='5bits per vector, reserved0 means all32 inputs zero, otherwise exact signed W; depth0 LUT supplies two bits/output. Header-first payload commit is optimistic and requires producer-side buffering.',
        static_domain_LUT_bits_per_numerical_student=25*10*96*2,
        strong_controls='All modes receive exact W header, whole-vector-zero bypass, signed-W domain certificates and exact output-alias threshold folding. No per-component-zero bitmap or dynamic zero-operand rewiring is assumed.',
        important_unclosed=['producer leading-sign/zero detection and completed-vector buffer',
                           'fine terminal-mask closure and compact dynamic issue construction; ideal dispatch is only a floor',
                           'finite RF capacity/banks, dependency latency, instruction SRAM/broadcast and wire/fanout',
                           'intermediate last-use/peak scheduling and pipeline/bit packing feasibility'],
        results={})
    for split,folder in [('train4','preview_gate_fixed_train4'),('valid10','preview_gate_fixed_diverse10')]:
        result['results'][split]={}
        for mode in ['whole','coarse_fine']:
            value=run_dataset(DATA/folder,dag,split,mode)
            result['results'][split]['f14_'+mode]=value
            print(split,'f14_'+mode,'floor',value['roofline_max'],'fixed32',value['fixed32_counterexample_max'],
                  'nodes',value.get('dag_nodes',0),'finevectors',value.get('fine_vectors',0),flush=True)
    result['results']['valid10']['f6_whole']=run_dataset(DATA/'preview_gate_fixed_f6_diverse10',dag,'valid10','whole')
    result['results']['train4']['f6_whole']=None
    result['missing']='No actual f6 train4 capture; deliberately not synthesized by a second rounding of f14.'
    result['comparisons']={}
    for split,models in result['results'].items():
        candidate=models['f14_coarse_fine']
        result['comparisons'][split]={}
        for baseline in ('f14_whole','f6_whole'):
            old=models[baseline]
            if old is None:
                result['comparisons'][split][baseline]=None
                continue
            result['comparisons'][split][baseline]=dict(
                conditional_roofline_ratio=candidate['roofline_max']/old['roofline_max'],
                fixed32_counterexample_ratio=candidate['fixed32_counterexample_max']/old['fixed32_counterexample_max'],
                DAG_node_ratio=candidate['dag_nodes']/old['dag_nodes'],
                scope='ratio of declared resource-ledger floors, never measured or scheduled speed')
    result['complete']=True
    save(HERE/'service_roofline.json',result)
    rows=[]
    for split,models in result['results'].items():
        for model,r in models.items():
            if r is None:
                rows.append(f'|{split}|{model}|未捕获|—|—|—|—|—|—|')
                continue
            rows.append(f"|{split}|{model}|{r['network_AEE_frame_mean']:.6f}|{r.get('dag_nodes',0)/1e6:.3f}|"
                        f"{(r.get('arithmetic_bits',0)+r.get('comparison_bitwork',0))/1e6:.3f}|"
                        f"{r['rf_read_bits']/1e6:.3f}/{r['rf_write_bits']/1e6:.3f}|"
                        f"{r['issued_operations_ideal_dispatch']/1e6:.3f}|{r['roofline_max']:,}|{r['fixed32_counterexample_max']:,}|")
    md=['# Preview V 普通底座与固定 coarse/fine 资源门',
        '',
        '只读第一轴真实捕获：train4共5120向量，diverse10共12800向量，每向量32输入/96门。'
        '三布局逐门等于各自整数完整函数及捕获门。f6是另一量化学生，其AEE不能称f14无损；f6 train4缺捕获，留空。',
        '',
        '|数据|布局|已有整帧AEE|DAG加减 M|加减+比较 Mbit|RF读/写 Mbit|issue M|条件roofline最大项|32-bit反例最大项|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',*rows,'',
        'Roofline资源固定为64bit/拍总加法/比较、128bit/拍读、64bit/拍写、8条/拍issue。'
        '表中数是各需求除资源后取max的**条件下界**，未给依赖排程或硬件周期，也不是速度比。'
        '32-bit反例用两路固定32-bit ALU，窄节点仍占一字操作，超过32bit分字计。',
        '',
        '共同普通编译包括5bit exact-W/全零header、按W和t预编的depth0域证书、'
        '输出静态sign/shift的有向整数阈值折叠。低8切分为signed high与unsigned low，'
        '每个节点/移位操作数按真实线性系数传播范围与位数，不沿用此前证书depth充当节点算术位宽。'
        'RF合同逐节点两读一写；门阈值读取和至多两次coarse证书比较也收费。',
        '',
        '未决coarse终端已在coarse DAG写入，保留其地址而不重复收费一次写；fine读取和最终组合另计。'
        'K全部32根均非零，只要仍有一个未决门就读取其全部low根需求，并按实际根边累计读流量，'
        '不能按减少的DAG节点比例缩减源码。',
        '',
        'Fine活节点闭包/紧凑dispatch尚未实现，因此同时保存全fine程序扫描的issue敏感性。'
        '当前尚未闭合生产者LZD/零检测与完整向量缓冲、RF峰值/银行、指令ROM和动态依赖控制。'
        '没有额外逐分量zero bitmap/动态zero重连；普通控制仍可继续受益。位级最乐观计数若不胜f6，'
        '只能说明本low8布局缺少合理服务目标，不能否定判门族。','',
        '实测账本结论：valid10候选相对f14 whole的条件下界为0.67155倍，但相对真实f6 whole为1.08132倍；'
        '固定32-bit ALU反例分别为1.05953与1.06079倍。f14/f6已有完整十帧AEE仅相差0.00068936，'
        '当前low8布局未形成胜过更便宜f6控制的服务目标，尚不能据此否定完整825精度差异或整个判门家族。'
        '候选3349个向量仍需fine，共107168个根分量、857344bit根并集；fine实际根边读1378056bit。'
        '最大未决coarse152bit；连low源及两96bit mask/packet的checkpoint必要状态600bit，绝非全程RF峰值。']
    (HERE/'service_roofline.md').write_text('\n'.join(md)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
