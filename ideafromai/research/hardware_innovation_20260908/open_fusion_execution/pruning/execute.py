"""Execute a static 2x2-phase/H8 producer mask on actual preview V and sn2.

This is an intentionally lossy, untrained prototype. The original I24 residual
is untouched. Kept groups follow the exact existing arithmetic; removed groups
emit zero gates without V/BN1/PSN work. No coefficient permutation or duplicate
weight versions are hidden. All original source/U work and gate stores remain.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
FULL=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
sys.path.insert(0,str(FULL/'preview_sn2_chain'))
from run_chain import Machine,coefficients,build_nrv,execute_u,read_npz,difference,SRC,Z,Y,GATE
from source_program import run as source_run
from run_windows import window

MASK=122880


def kept_groups(m, positions):
    word=int.from_bytes(m.read_word(MASK),'little')
    kept=[]
    for y,x in positions:
        phase=2*(y%2)+x%2
        kept.append([not bool(word & (1<<(phase*12+g))) for g in range(12)])
        m.advance(tag='phase_mask_decode')
    return np.array(kept,dtype=bool)


def masked_v(m,base,live,keep):
    m.phase='preview_V32_BN1';start=m.time
    positions=len(keep);nt=positions*10
    for h0 in range(0,96,32):
        active_groups=[];all_groups=[]
        for hg in range(4):
            g=h0//8+hg
            all_tp=[tp for tp in range(nt) if keep[tp//10,g]]
            active=[tp for tp in all_tp if live & (1<<tp)]
            all_groups.append(all_tp);active_groups.append(active)
            m.advance(tag='producer_group_enable')
            for tp in all_tp:m.advance(op=('clear',tp*4+hg,None),tag='V_clear')
        for k in range(32):
            for hg,active in enumerate(active_groups):
                if not active:continue
                m.coefficient(base['V']+(k*96+h0+hg*8)*4)
                m.advance(read=(Z+(active[0]*32+k)*4)//8*8,tag='V_first_source_prefetch')
                for ai,tp in enumerate(active):
                    source=Z+(tp*32+k)*4
                    nxt=Z+(active[ai+1]*32+k)*4 if ai+1<len(active) else None
                    dst=tp*4+hg;m.wait_reg(dst)
                    m.advance(read=nxt//8*8 if nxt is not None else None,
                        op=('FMA',dst,(source,None)),tag='V_FMA')
        m.drain()
        for name,op in [('BN_scale','mul_coef'),('BN_bias','add_coef')]:
            for hg,all_tp in enumerate(all_groups):
                if not all_tp:continue
                m.coefficient(base[name]+(h0+hg*8)*4)
                for tp in all_tp:
                    dst=tp*4+hg;m.wait_reg(dst)
                    m.advance(op=(op,dst,None),tag=name)
        m.drain()
        for tp in range(nt):
            for hg,all_tp in enumerate(all_groups):
                if tp in all_tp:m.store_reg(tp*4+hg,Y+(tp*96+h0+hg*8)*4)
    m.mark('preview_V32_BN1',start)


def masked_sn2(m,base,A,keep,gate_base):
    start=m.time;m.phase='noncausal_T10_sn2'
    positions=len(keep)
    for h in range(0,96,8):
        pp=[p for p in range(positions) if keep[p,h//8]]
        m.advance(tag='PSN_group_enable')
        for p in pp:
            for t in range(10):m.load_reg(p*10+t,Y+((p*10+t)*96+h)*4)
        for p in pp:
            for t in range(10):m.advance(op=('clear',20+p*10+t,None),tag='sn2_clear')
        for s in range(10):
            if not pp:break
            for t in np.flatnonzero(A[:,s]):
                address=base['A']+(int(t)*10+s)*4;m.coefficient(address)
                for p in pp:
                    dst=20+p*10+int(t);m.wait_reg(dst)
                    m.advance(op=('FMA_reg',dst,(p*10+s,address%32)),tag='sn2_nonzero_time_FMA')
        if pp:
            for t in range(10):
                m.coefficient(base['b']+t*32)
                for p in pp:
                    dst=20+p*10+t;m.wait_reg(dst)
                    m.advance(op=('add_coef',dst,None),tag='sn2_bias')
            m.coefficient(base['theta'])
            for p in pp:
                for t in range(10):
                    dst=20+p*10+t;m.wait_reg(dst)
                    m.advance(op=('sub_coef',dst,None),tag='sn2_threshold_subtract')
            for p in pp:
                for t in range(10):
                    dst=20+p*10+t;m.wait_reg(dst)
                    m.advance(op=('compare',dst,None),tag='sn2_compare')
        m.drain()
        for p in range(positions):
            words=np.zeros(8,np.uint16)
            if p in pp:
                for t in range(10):
                    words|=m.rf[20+p*10+t].astype(np.uint16)<<t
                    m.advance(tag='gate_collector_RF_read')
            payload=words.astype('<u2').tobytes()
            for j in (0,8):m.advance(write=(gate_base+(p*96+h)*2+j,payload[j:j+8]),tag='gate_word_store')
    m.mark('noncausal_T10_sn2',start)


def run(data,p,label,axis,drop,stress):
    geo=json.loads(str(data['window_geometry_json']))[label]
    source=data[label+'_sn1_gate'];words=sum(source[t].astype(np.uint16)<<t for t in range(10))
    blob,base,info=coefficients(p,False)
    m=Machine(stress)
    m.phase='coefficient_cold_fill';m.dma_input(blob,0,True)
    source_report=source_run(m,data,axis,label)
    m.phase='phase_mask_input'
    bits=sum(1<<(ph*12+g) for ph in range(4) for g in range(12) if drop[ph,g*8:(g+1)*8].all())
    m.dma_input(bits.to_bytes(8,'little'),MASK)
    h,w=geo['gate_shape'];oy,ox=geo['gate_origin'];rows=[]
    for y in range(h):
        for x in range(0,w,2):
            count=min(2,w-x);positions=[(oy+y,ox+x+i) for i in range(count)]
            start=m.time
            n,occ,live=build_nrv(m,words,geo['source_origin'],positions)
            execute_u(m,n,base,False,count)
            m.phase='phase_mask_decode';keep=kept_groups(m,positions)
            masked_v(m,base,live,keep)
            masked_sn2(m,base,p['preview_A'],keep,GATE+(y*w+x)*192)
            rows.append(dict(y=y,x=x,positions=count,service=m.time-start,kept_groups=keep.sum(axis=1).tolist()))
    m.phase='sn2_gate_egress'
    for off in range(0,h*w*192,32):
        for j in range(0,32,8):m.read_word(GATE+off+j)
        for _ in range(5):m.advance(tag='DMA_output_slots')
    actual=np.frombuffer(m.state,'<u2',count=h*w*96,offset=GATE).reshape(h,w,96).copy()
    gate=np.stack([(actual>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    expected=data[label+'_sn2_gate'].copy()
    for y in range(h):
        for x in range(w):expected[:,drop[2*((oy+y)%2)+(ox+x)%2],y,x]=False
    check=difference(gate,expected);assert check['differences']==0,check
    return gate,dict(service_slots=m.time,stages=dict(m.stages),counts=dict(m.count),
        checks={'actual_masked_gate_vs_unmodified_capture_plus_static_mask':check},
        P2_rows=rows,source_program=source_report,mask_bits=48,mask_stored_bytes=32,
        weight_versions=1,weight_bytes=info['used_bytes'],source_and_U_unchanged=True,
        state_capacity=131072,coefficient_capacity=131072,RF_words=96)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');ap.add_argument('--axis',choices=['ordinary','lifting_raw']);args=ap.parse_args()
    masks=json.loads((HERE.parent/'review/phase_group8_masks.json').read_text())
    result=dict(scope=__doc__,stress=args.stress,lossy=True,new_training=False,new_AEE=False,
        executed_scope='I24 -> actual compiled source -> full K864 U32 -> retained V/BN1/PSN -> materialized gate map. Integer consumers not timed here.',
        axes={})
    for axis in ([args.axis] if args.axis else ['ordinary','lifting_raw']):
        path=FULL/'capture'/axis
        data=read_npz(path/'000_zurich_city_09_a_0001.npz');p=read_npz(path/'live_parameters.npz')
        result['axes'][axis]={}
        for label in ['corner','interior']:
            _,baseline=window(data,p,label,False,args.stress,False,axis)
            entry={'unpruned':baseline}
            # Zero mask validates this separately coded emitter and its small
            # controller overhead. Then both candidates get the same emitter.
            choices={'mask_engine_no_pruning':np.zeros((4,96),bool)}
            choices.update({name:np.array(item['mask_uint8'],bool) for name,item in masks[axis].items()})
            for name,drop in choices.items():
                gate,report=run(data,p,label,axis,drop,args.stress)
                report['service_reduction_vs_existing_unpruned']=1-report['service_slots']/baseline['service_slots']
                entry[name]=report
                print(axis,label,name,report['service_slots'],report['service_reduction_vs_existing_unpruned'],flush=True)
                if not args.stress:np.savez_compressed(HERE/f'{axis}_{label}_{name}.npz',gate=gate)
            result['axes'][axis][label]=entry
    name='execution'+('_'+args.axis if args.axis else '')+('_stress' if args.stress else '')+'.json'
    (HERE/name).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
