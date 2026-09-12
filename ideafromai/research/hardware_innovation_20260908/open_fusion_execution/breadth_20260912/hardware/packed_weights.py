"""Actual signed4/8 coefficient stream and paid row-scale continuation.

No expanded q16 U is placed in simulated memory on packed arms. Scale uses
two existing signed16 x signed24 products, not an unpriced wide multiplier.
"""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1]
OLD=OPEN/'stage_20260912/hardware'
sys.path.insert(0,str(OLD))
import integrated as stage
import consumer_ranked as consumer
import integer_chain as integer

ordinary_dense=consumer.dense
ordinary_coeffs=consumer.coeffs
gather_k=stage.addresses.resident.resident_mac.__globals__['gather_k']


class PackedMachine(stage.IntegratedMachine):
    def __init__(self,stress=False):
        super().__init__(stress)
        # The previous interface already budgets 64B gather staging. Allocate
        # 24B source gather + 16B weight latch inside that common capacity.
        self.common_staging=bytearray(64)
        self.packed_coeff_valid=False

    def integer_op(self,kind,dst,args):
        if kind=='IUNPACK_SIGNED':
            offset,bits=args
            raw=self.cword[offset:offset+bits]
            assert len(raw)==bits
            if bits==8:value=np.frombuffer(raw,'i1').astype(np.int64)
            else:
                value=np.array([(raw[i//2]>>(4*(i%2)))&15 for i in range(8)],np.int64)
                value=np.where(value>=8,value-16,value)
        elif kind=='IMAC_PACKED_INDEX':
            src,lane=args
            assert self.ready[src]<=self.time and self.packed_coeff_valid
            weight=np.frombuffer(self.common_staging,'<i2',8,24).astype(np.int64)
            value=self.rf[dst].astype(np.int64)+weight*int(self.rf[src,lane])
        elif kind in ('ISPLIT_LOW24','ISPLIT_HIGH24'):
            src=args;assert self.ready[src]<=self.time
            x=self.rf[src].astype(np.int64)
            low=(x+(1<<23))%(1<<24)-(1<<23)
            value=low if kind=='ISPLIT_LOW24' else (x-low)//(1<<24)
        elif kind=='IMUL16X24_REG':
            src=args;assert self.ready[src]<=self.time
            x=self.rf[dst].astype(np.int64);scale=self.rf[src].astype(np.int64)
            assert np.all(x>=-(1<<23)) and np.all(x<(1<<23))
            assert np.all(scale>=-32768) and np.all(scale<=32767)
            value=x*scale
        elif kind=='IJOIN_SCALED24':
            lo,hi=args;assert self.ready[lo]<=self.time and self.ready[hi]<=self.time
            value=self.rf[lo].astype(np.int64)+(self.rf[hi].astype(np.int64)<<24)
        else:return super().integer_op(kind,dst,args)
        assert np.all(value>=-(1<<47)) and np.all(value<(1<<47)),kind
        return value.astype(np.float64)

    def unpack_weight(self,address,bits):
        self.coefficient(address)
        self.wait_reg(84)
        self.advance(op=('IUNPACK_SIGNED',84,(address%32,bits)),tag='packed_code_sign_extend_RF')
        self.wait_reg(84)
        # One explicit RF read to the common coefficient staging region;
        # the subsequent MAC has only acc+source RF reads, no third port.
        payload=self.rf[84].astype('<i2').tobytes()
        self.advance(tag='packed_weight_RF_to_common_staging')
        self.common_staging[24:40]=payload
        self.packed_coeff_valid=True
        return bool(np.any(self.rf[84]))


def pack_codes(code,bits):
    code=code.T.astype(np.int64).reshape(-1)
    if bits==8:return code.astype('i1').tobytes()
    nibbles=(code&15).astype(np.uint8)
    return (nibbles[0::2]|(nibbles[1::2]<<4)).tobytes()


def coeffs(q,late_v=False):
    if not q.get('_packed_bits',0):return ordinary_coeffs(q,late_v)
    assert not late_v
    bits=int(q['_packed_bits']);code=q['_code'];scales=q['_scale']
    assert np.array_equal(code.astype(np.int64)*scales[:,None].astype(np.int64),q['U_ped_q16'])
    _,_,bounds=ordinary_coeffs(q,False)
    blob=bytearray();base={}
    def add(name,payload):
        base[name]=len(blob);blob.extend(payload);blob.extend(bytes((-len(blob))%32))
    for name in ('U_conv2_theta','F','U_ped','V_ped'):
        add(name,pack_codes(code,bits) if name=='U_ped' else q[name+'_q16'].T.astype('<i2').tobytes())
    add('U_scale',scales.astype('<i2').tobytes())
    header=np.array([bits,96,32,int(q['U_ped_exponent']),1,0,0,0],'<i4')
    add('U_header',header.tobytes())
    for name in ('BN2_constant','PED_bias'):add(name,q[name+'_q24'].astype('<i4').tobytes())
    rows=np.zeros((10,8),'<i4')
    for t in range(10):rows[t,:4]=[q['consumer_threshold'][t],q['consumer_direction'][t],q['consumer_constant'][t],q['consumer_permutation'][t]]
    add('compare',rows.tobytes())
    assert len(blob)<=131072
    return bytes(blob),base,bounds


def scale_accumulator(m,dst,scale):
    m.wait_reg(dst)
    for reg,kind in ((85,'ISPLIT_LOW24'),(86,'ISPLIT_HIGH24')):
        m.wait_reg(reg);m.advance(op=(kind,reg,dst),tag='row_scale_signed24_split')
    for reg in (85,86):
        m.wait_reg(reg);m.wait_reg(scale)
        m.advance(op=('IMUL16X24_REG',reg,scale),tag='row_scale_existing_16x24_product')
    m.wait_reg(85);m.wait_reg(86);m.wait_reg(dst)
    m.advance(op=('IJOIN_SCALED24',dst,(85,86)),tag='row_scale_shifted48_merge')


def dense(m,base,name,k_count,h_count,input_base,output_base,positions,shift,live=None,bias=None):
    if name!='U_ped' or 'U_header' not in base:
        return ordinary_dense(m,base,name,k_count,h_count,input_base,output_base,positions,shift,live,bias)
    assert k_count==96 and h_count==32 and 1<=positions<=2 and live is None and bias is None
    start=m.time;m.phase='packed_U_ped'
    m.coefficient(base['U_header']);m.wait_reg(94)
    m.advance(op=('ILOAD',94,np.frombuffer(m.cword,'<i4',8).copy()),tag='packed_header_RF_load')
    m.wait_reg(94);metadata=m.rf[94].astype(np.int64).copy()
    m.advance(tag='packed_header_select')
    bits,k,h,e=map(int,metadata[:4]);assert bits in (4,8) and (k,h,e)==(k_count,h_count,shift)
    for hg in range(4):
        address=base['U_scale']+hg*16;m.coefficient(address)
        m.wait_reg(80+hg)
        m.advance(op=('ILOAD',80+hg,np.frombuffer(m.cword,'<i2',8,address%32).astype(np.int64)),tag='row_scale_RF_load')
    for r in range(positions*10*4):m.advance(op=('clear',r,None),tag='output_clear')
    for k in range(k_count):
        gather_k(m,k,input_base,k_count,positions)
        for hg in range(4):
            address=base['U_ped']+(k*h_count+hg*8)*bits//8
            if not m.unpack_weight(address,bits):continue
            for ip in range(positions):
                for t in range(10):
                    src,lane=88+2*ip+t//8,t%8
                    if m.rf[src,lane]==0:
                        m.advance(tag='ordinary_zero_bypass');continue
                    dst=(ip*10+t)*4+hg;m.wait_reg(dst)
                    m.advance(op=('IMAC_PACKED_INDEX',dst,(src,lane)),tag='packed_small_weight_MAC')
    m.drain()
    for ip in range(positions):
        for t in range(10):
            for hg in range(4):
                dst=(ip*10+t)*4+hg
                scale_accumulator(m,dst,80+hg)
                integer.complete(m,dst,shift)
                m.store_i24(dst,output_base+((ip*10+t)*h_count+hg*8)*3)
    m.mark(m.phase,start)


def install():
    consumer.coeffs=coeffs
    consumer.dense=dense
