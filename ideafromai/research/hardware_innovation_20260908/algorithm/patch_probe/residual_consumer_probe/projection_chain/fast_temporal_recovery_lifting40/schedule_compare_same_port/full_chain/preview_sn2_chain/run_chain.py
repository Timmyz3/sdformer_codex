"""Full-K preview U -> V -> fixed BN1 -> noncausal sn2, finite payload model.

Dead V-tail and low-bit coefficient encodings are EXISTING common structure.
Two arithmetic representations preserve the captured parameter values; this
does not perform quantization or train a new student.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent.parent))
from machine import Machine
from numerical_reference import difference,tf32_round
from consumer_service import read_npz

SRC,NRV,Z,Y,GATE=0,24576,32768,40960,49152


def coefficients(p,compact):
    u=p['preview_u'];v=p['preview_v']
    assert int(p['preview_shared_rank'])==32 and not np.any(v[32:])
    assert float(p['preview_theta_source'])==1.0
    u=u[:,:32]
    exponent=np.ceil(np.log2(np.max(np.abs(u),axis=0)/127)).astype(int)
    code=np.rint(u/np.exp2(exponent)).astype(np.int8)
    assert np.array_equal(u,code*np.exp2(exponent))
    integer_prefix_bound=np.abs(code.astype(np.int64)).sum(0)
    assert integer_prefix_bound.max()<2**24
    vv=v[:32]
    assert np.all(vv!=0)
    shifts=np.rint(np.log2(np.abs(vv))).astype(int)
    assert np.array_equal(np.abs(vv),np.exp2(shifts))
    shift_code=(np.sign(vv)*(shifts+16)).astype(np.int8)
    assert np.all(np.abs(shift_code)>=1) and np.all(np.abs(shift_code)<=15)
    # Fixed-BN constants derived exactly as the earlier numerical reference.
    inv=np.float32(1)/np.sqrt(np.float32(p['bn1_var'])+np.float32(p['bn1_eps']))
    scale=np.float32(p['bn1_gamma']*inv)
    bias=np.float32(p['bn1_beta']-np.float32(p['bn1_mean']*scale))
    blob=bytearray();base={}
    def add(name,data):
        base[name]=len(blob)
        payload=data.tobytes()
        blob.extend(payload);blob.extend(bytes((-len(blob))%32))
    add('U',code if compact else tf32_round(u).astype('<f4'))
    add('V',shift_code if compact else tf32_round(vv).astype('<f4'))
    add('U_scale',np.exp2(exponent).astype('<f4'))
    add('BN_scale',scale.astype('<f4'));add('BN_bias',bias.astype('<f4'))
    add('A',p['preview_A'].astype('<f4'))
    add('b',np.repeat(p['preview_b'][:,None],8,axis=1).astype('<f4'))
    add('theta',np.full(8,float(p['preview_theta_output']),'<f4'))
    assert len(blob)<=131072
    return bytes(blob),base,dict(used_bytes=len(blob),representation='existing U8 + signed exponent V' if compact else 'same coefficients expanded FP32',
        dropped_dead_U_columns=16,V_tail_exact_zeros=int(np.count_nonzero(v[32:]==0)),
        U_integer_prefix_maxabs_bound=int(integer_prefix_bound.max()),U_parameter_reconstruction_mismatches=0,V_parameter_reconstruction_mismatches=0)


def build_nrv(m,words,source_origin,positions):
    start=m.time;m.phase='build_P2_NRVs'
    n=0;occurrences=0;live=0
    height,width=words.shape[1:]
    for k in range(864):
        c,rem=divmod(k,9);ky,kx=divmod(rem,3)
        mask=0
        for p,(y,x) in enumerate(positions):
            sy,sx=y+ky-1,x+kx-1
            if not (0<=sy<240 and 0<=sx<320): continue
            ly,lx=sy-source_origin[0],sx-source_origin[1]
            assert 0<=ly<height and 0<=lx<width
            address=2*((ly*width+lx)*96+c) if getattr(m,'sn1_spatial',False) else 2*((c*height+ly)*width+lx)
            raw=m.read_word(address)
            word=int.from_bytes(raw[address%8:address%8+2],'little')
            mask|=word<<(p*10)
        live |= mask
        m.advance(op=('control',93,[k,mask,live,0,0,0,0,0]),tag='NRV_decode')
        m.wait_reg(93)
        if mask:
            payload=int(m.rf[93,0]).to_bytes(4,'little')+int(m.rf[93,1]).to_bytes(4,'little')
            m.advance(write=(NRV+n*8,payload),tag='NRV_write')
            n+=1;occurrences+=mask.bit_count()
    m.mark('build_P2_NRVs',start)
    return n,occurrences,live


def activate_coefficient(m,address,compact,kind):
    m.coefficient(address)
    if compact:
        m.wait_reg(84)
        m.advance(op=(kind,84,address),tag='coefficient_decode')
        m.wait_reg(84)
    return 84 if compact else None


def execute_u(m,n,base,compact,positions=2):
    start=m.time;m.phase='preview_U32_full_K'
    tp_count=positions*10
    for r in range(tp_count*4): m.advance(op=('clear',r,None),tag='U_clear')
    for record in range(n):
        raw=m.read_word(NRV+record*8)
        k=int.from_bytes(raw[:4],'little');mask=int.from_bytes(raw[4:],'little')
        m.advance(tag='NRV_select')
        for hg in range(4):
            address=base['U']+(k*32+hg*8)*(1 if compact else 4)
            reg=activate_coefficient(m,address,compact,'decode_u8')
            for tp in range(tp_count):
                if mask & (1<<tp):
                    dst=tp*4+hg;m.wait_reg(dst)
                    m.advance(op=('AAC' if compact else 'FMA',dst,(None,reg)),tag='U_active_issue')
    m.drain()
    raw_z=np.empty((tp_count,32),np.float32)
    for hg in range(4):
        if compact: m.coefficient(base['U_scale']+hg*32)
        for tp in range(tp_count):
            dst=tp*4+hg
            if compact:
                m.advance(op=('scale',dst,None),tag='U_exact_dequant')
                m.wait_reg(dst)
            raw_z[tp,hg*8:hg*8+8]=m.rf[dst]
            m.advance(op=('TF32',dst,None),tag='Z_conversion')
            m.store_reg(dst,Z+(tp*32+hg*8)*4)
    m.mark('preview_U32_full_K',start)
    return raw_z


def execute_v_bn(m,base,compact,live,positions=2):
    start=m.time;m.phase='preview_V32_BN1'
    tp_count=positions*10
    active=[tp for tp in range(tp_count) if live&(1<<tp)]
    raw_y=np.empty((tp_count,96),np.float32)
    for h0 in range(0,96,32):
        for r in range(tp_count*4): m.advance(op=('clear',r,None),tag='V_clear')
        for k in range(32):
            for hg in range(4):
                if not active: continue
                h=h0+hg*8
                address=base['V']+(k*96+h)*(1 if compact else 4)
                reg=activate_coefficient(m,address,compact,'decode_shift')
                m.advance(read=(Z+(active[0]*32+k)*4)//8*8,tag='V_first_source_prefetch')
                for ai,tp in enumerate(active):
                    source=Z+(tp*32+k)*4
                    nxt=Z+(active[ai+1]*32+k)*4 if ai+1<len(active) else None
                    m.wait_reg(tp*4+hg)
                    m.advance(read=nxt//8*8 if nxt is not None else None,
                        op=('FMA',tp*4+hg,(source,reg)),tag='V_FMA')
        m.drain()
        for tp in range(tp_count):
            for hg in range(4): raw_y[tp,h0+hg*8:h0+hg*8+8]=m.rf[tp*4+hg]
        for name,op in [('BN_scale','mul_coef'),('BN_bias','add_coef')]:
            for hg in range(4):
                m.coefficient(base[name]+(h0+hg*8)*4)
                for tp in range(tp_count):
                    dst=tp*4+hg;m.wait_reg(dst)
                    m.advance(op=(op,dst,None),tag=name)
        m.drain()
        for tp in range(tp_count):
            for hg in range(4): m.store_reg(tp*4+hg,Y+(tp*96+h0+hg*8)*4)
    m.mark('preview_V32_BN1',start)
    return raw_y


def execute_sn2(m,base,A,positions=2,gate_base=GATE):
    start=m.time;m.phase='noncausal_T10_sn2'
    tp_count=positions*10
    for h in range(0,96,8):
        for tp in range(tp_count): m.load_reg(tp,Y+(tp*96+h)*4)
        for tp in range(tp_count): m.advance(op=('clear',20+tp,None),tag='sn2_clear')
        for s in range(10):
            for t in np.flatnonzero(A[:,s]):
                address=base['A']+(int(t)*10+s)*4
                m.coefficient(address)
                for p in range(positions):
                    dst=20+p*10+int(t);m.wait_reg(dst)
                    m.advance(op=('FMA_reg',dst,(p*10+s,address%32)),tag='sn2_nonzero_time_FMA')
        for t in range(10):
            m.coefficient(base['b']+t*32)
            for p in range(positions):
                dst=20+p*10+t;m.wait_reg(dst)
                m.advance(op=('add_coef',dst,None),tag='sn2_bias')
        m.coefficient(base['theta'])
        for tp in range(tp_count):
            m.wait_reg(20+tp)
            m.advance(op=('sub_coef',20+tp,None),tag='sn2_threshold_subtract')
        for tp in range(tp_count):
            m.wait_reg(20+tp)
            m.advance(op=('compare',20+tp,None),tag='sn2_compare')
        m.drain()
        for p in range(positions):
            words=np.zeros(8,np.uint16)
            for t in range(10):
                words|=m.rf[20+p*10+t].astype(np.uint16)<<t
                m.advance(tag='gate_collector_RF_read')
            payload=words.astype('<u2').tobytes()
            for j in (0,8): m.advance(write=(gate_base+(p*96+h)*2+j,payload[j:j+8]),tag='gate_word_store')
    m.mark('noncausal_T10_sn2',start)
    return np.frombuffer(m.state,'<u2',count=positions*96,offset=gate_base).reshape(positions,96).copy()


def run(data,p,label,compact,stress=False):
    geo=json.loads(str(data['window_geometry_json']))[label]
    source=data[label+'_sn1_gate'];words=np.zeros(source.shape[1:],np.uint16)
    for t in range(10): words|=source[t].astype(np.uint16)<<t
    positions=[tuple(geo['gate_origin']), (geo['gate_origin'][0],geo['gate_origin'][1]+1)]
    blob,base,info=coefficients(p,compact)
    m=Machine(stress)
    m.phase='coefficient_cold_fill';m.dma_input(blob,0,True)
    m.phase='sn1_halo_input';m.dma_input(words.astype('<u2').tobytes(),SRC)
    n,occurrences,live=build_nrv(m,words,geo['source_origin'],positions)
    z=execute_u(m,n,base,compact)
    raw=execute_v_bn(m,base,compact,live)
    y=np.frombuffer(m.state,'<f4',count=1920,offset=Y).reshape(20,96).copy()
    gate_words=execute_sn2(m,base,p['preview_A'])
    m.phase='sn2_gate_egress'
    for off in range(0,384,32):
        payload=b''.join(m.read_word(GATE+off+j) for j in range(0,32,8))
        assert len(payload)==32
        for _ in range(5): m.advance(tag='DMA_output_slots')
    def picked(name):
        return data[label+'_'+name][:,:,0,:2].transpose(2,0,1).reshape(20,-1)
    gold_gate=picked('sn2_gate').reshape(2,10,96)
    gold_word=sum(gold_gate[:,t].astype(np.uint16)<<t for t in range(10))
    checks=dict(Z_vs_CUDA=difference(z,picked('preview_Z_shared')),
        raw_V_vs_CUDA=difference(raw,picked('preview_shared_raw')),
        BN1_vs_CUDA=difference(y,picked('preview_BN1_Y')),
        sn2_gate_bits_different=sum(int(int(a^b).bit_count()) for a,b in zip(gate_words.flat,gold_word.flat)))
    result=dict(service_slots=m.time,stage_slots=dict(m.stages),counts=dict(m.count),timeline=m.timeline,
        parameters=info,NRV_rows=n,source_occurrences=occurrences,checks=checks,
        max_pending=m.max_pending,RF_words=96,accumulator_words=80,coefficient_decode_register=84,NRV_control_register=93,
        state_capacity=131072,coefficient_capacity=131072,staging_bytes=64,
        completion='Actual sn2 P2/C96/T10 gate words stored and accepted by 32B/5slot sink. Source producer and following integer/native BN chain still outside.',
        stress='fixed SR last8/SW last4 of32 blocked' if stress else 'always_ready')
    return dict(z=z,raw=raw,y=y,gates=gate_words),result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stress',action='store_true');args=parser.parse_args()
    result=dict(scope=__doc__,axes={},full_chain_closed=False,relative_AEE_gate_passed=False)
    for axis in ('ordinary','lifting_raw'):
        directory=HERE.parent/'capture'/axis
        data=read_npz(directory/'000_zurich_city_09_a_0001.npz');p=read_npz(directory/'live_parameters.npz')
        result['axes'][axis]={}
        for label in ('corner','interior'):
            values={};rows={}
            for encoding in ('expanded_fp32','existing_u8_shift'):
                value,report=run(data,p,label,encoding=='existing_u8_shift',args.stress)
                values[encoding]=value;rows[encoding]=report
                print(axis,label,encoding,report['service_slots'],report['checks'],flush=True)
            comparisons={key:difference(values['expanded_fp32'][key],values['existing_u8_shift'][key]) for key in values['expanded_fp32']}
            assert all(v['differences']==0 for v in comparisons.values())
            rows['encoding_value_comparison']=comparisons
            result['axes'][axis][label]=rows
    path=HERE/('result_stress.json' if args.stress else 'result.json')
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
