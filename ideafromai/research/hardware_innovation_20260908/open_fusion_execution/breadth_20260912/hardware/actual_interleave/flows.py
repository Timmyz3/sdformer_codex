"""Actual source program and full PED; no independent replay clock."""
import copy
import numpy as np
import packed_weights
from integer_chain import pack24,read24,complete

INPUT_B=90112
GATES_B=98304

def source_flow(m,identity,program,offset):
    height,width=identity.shape[2:]
    mapped=copy.deepcopy(program)
    for ins in mapped:
        if 'dst' in ins:ins['dst']+=offset
        for p in ins.get('operands',[]):p['reg']+=offset
    for y in range(height):
        for x in range(width):
            m.phase='B_source_DMA';m.dma_input(pack24(identity[:,:,y,x]),INPUT_B)
            for h in range(0,96,8):
                m.phase='B_source_program';words=np.zeros(8,np.uint16)
                for ins in mapped:
                    kind=ins['kind'];m.count['source_ROM128_fetches']+=1
                    if kind=='load':m.load_i24(ins['dst'],INPUT_B+(ins['source_t']*96+h)*3)
                    elif kind=='nop':m.advance(tag='source_scheduled_nop')
                    elif kind=='commit':
                        payload=words.astype('<u2').tobytes();address=GATES_B+((y*width+x)*96+h)*2
                        for j in (0,8):m.advance(write=(address+j,payload[j:j+8]),tag='source_gate_word_store')
                    else:
                        for p in ins['operands']:m.wait_reg(p['reg'])
                        dst=95 if kind=='gate' else ins['dst'];m.wait_reg(dst)
                        m.advance(op=('ISOURCE',dst,ins),tag='source_'+kind)
                        if kind=='gate':
                            m.wait_reg(95);words|=m.rf[95].astype(np.uint16)<<ins['output_t']
                            m.advance(tag='source_gate_collector_RF_read')
    m.drain()

def gather_position(m,k,input_base,k_count,ip,regs):
    for t0 in (0,8):
        m.acquire('gather24');values=[]
        for t in range(t0,min(t0+8,10)):
            m.collect_i24(input_base+((ip*10+t)*k_count+k)*3)
            values.append(m.scalar_collector)
        values += [0]*(8-len(values));r=regs[t0//8]
        m.wait_reg(r);m.advance(op=('ILOAD',r,values),tag='common_gather_RF_write')
        m.release('gather24')
    for r in regs:m.wait_reg(r)

def two_rf_dense(m,base,name,k_count,h_count,input_base,output_base,shift,hblock,bias=None):
    m.phase='two_RF_'+name
    for h0 in range(0,h_count,hblock):
        groups=min(hblock,h_count-h0)//8
        for r in range(20*groups):
            m.wait_reg(r);m.advance(op=('clear',r,None),tag='output_clear')
        for k in range(k_count):
            # Second position really revisits CR; no extra cached weight vector.
            for ip in range(2):
                gather_position(m,k,input_base,k_count,ip,(93,94))
                for hg in range(groups):
                    address=base[name]+(k*h_count+h0+hg*8)*2;m.coefficient(address)
                    if not np.frombuffer(m.cword,'<i2',8,address%32).any():continue
                    for t in range(10):
                        src,lane=93+t//8,t%8
                        if m.rf[src,lane]==0:m.advance(tag='ordinary_zero_bypass');continue
                        dst=(ip*10+t)*groups+hg;m.wait_reg(dst)
                        m.advance(op=('IMAC_INDEX',dst,(address%32,src,lane)),tag='ordinary_MAC')
        m.drain()
        for ip in range(2):
            for t in range(10):
                for hg in range(groups):
                    dst=(ip*10+t)*groups+hg;complete(m,dst,shift)
                    if bias:
                        m.coefficient(base[bias]+(h0+hg*8)*4)
                        m.advance(op=('IADD_COEF',dst,None),tag='V_ped_bias');m.wait_reg(dst)
                        m.advance(op=('ISAT',dst,None),tag='V_ped_bias_sat24')
                    m.store_i24(dst,output_base+((ip*10+t)*h_count+h0+hg*8)*3)

def retained_z(m,base,q,updated,ped_v):
    for ip in range(2):
        m.phase='P1_retained_Z_U'
        for r in range(30):
            m.wait_reg(r);m.advance(op=('clear',r,None),tag='U_output_clear')
        for k in range(96):
            packed_weights.gather_k(m,k,updated+ip*10*96*3,96,1)
            for hg in range(3):
                address=base['U_ped']+(k*24+hg*8)*2;m.coefficient(address)
                if not np.frombuffer(m.cword,'<i2',8,address%32).any():continue
                for t in range(10):
                    src,lane=88+t//8,t%8
                    if m.rf[src,lane]==0:m.advance(tag='ordinary_zero_bypass');continue
                    dst=t*3+hg;m.wait_reg(dst)
                    m.advance(op=('IMAC_INDEX',dst,(address%32,src,lane)),tag='ordinary_MAC')
        m.drain()
        for r in range(30):complete(m,r,int(q['U_ped_exponent']))
        for h0 in (0,48):
            m.phase='P1_retained_Z_V'
            for r in range(30,90):
                m.wait_reg(r);m.advance(op=('clear',r,None),tag='V_output_clear')
            for k in range(24):
                for hg in range(6):
                    address=base['V_ped']+(k*96+h0+hg*8)*2;m.coefficient(address)
                    if not np.frombuffer(m.cword,'<i2',8,address%32).any():continue
                    for t in range(10):
                        src,lane=t*3+k//8,k%8;m.wait_reg(src)
                        if m.rf[src,lane]==0:m.advance(tag='ordinary_zero_bypass');continue
                        dst=30+t*6+hg;m.wait_reg(dst)
                        m.advance(op=('IMAC_INDEX',dst,(address%32,src,lane)),tag='ordinary_MAC')
            m.drain()
            for t in range(10):
                for hg in range(6):
                    dst=30+t*6+hg;complete(m,dst,int(q['V_ped_exponent']))
                    m.coefficient(base['PED_bias']+(h0+hg*8)*4)
                    m.advance(op=('IADD_COEF',dst,None),tag='V_ped_bias');m.wait_reg(dst)
                    m.advance(op=('ISAT',dst,None),tag='V_ped_bias_sat24')
                    m.store_i24(dst,ped_v+((ip*10+t)*96+h0+hg*8)*3)

def ped_flow(m,base,q,style,hblock,updated,ped_u,ped_v):
    if style=='P1':retained_z(m,base,q,updated,ped_v)
    elif style=='original':
        dense=packed_weights.ordinary_dense
        dense(m,base,'U_ped',96,24,updated,ped_u,2,int(q['U_ped_exponent']))
        dense(m,base,'V_ped',24,96,ped_u,ped_v,2,int(q['V_ped_exponent']),bias='PED_bias')
    else:
        two_rf_dense(m,base,'U_ped',96,24,updated,ped_u,int(q['U_ped_exponent']),hblock)
        two_rf_dense(m,base,'V_ped',24,96,ped_u,ped_v,int(q['V_ped_exponent']),hblock,bias='PED_bias')
    m.phase='A_actual_PED_egress'
    for off in range(0,2*10*96*3,32):
        m.acquire('DMA32')
        payload=b''.join(m.read_word(ped_v+off+j) for j in (0,8,16,24))
        for _ in range(5):m.advance(tag='PED_DMA_output_slots')
        m.observed_egress.extend(payload) # External test monitor, never a controller input.
        m.release('DMA32')
    m.drain()
