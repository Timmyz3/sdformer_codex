"""Shared finite 8-lane preview machine with actual SRAM/RF payload.

Operations issue once/slot. FP32 latency4, integer AAC/decode/compare latency2.
RF lanes have the already budgeted 48-bit integer / FP32 formats. Python
float64 is only a lossless container for signed48 and explicitly rounded FP32;
it does not grant a 64-bit hardware datapath or floating-point accumulation.
"""
from collections import Counter
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from preview_v_schedule import vector_fma
from numerical_reference import tf32_round


class Machine:
    def __init__(self, stress=False):
        self.time=0
        self.rf=np.zeros((96,8),np.float64)
        self.ready=np.zeros(96,np.int64)
        self.pending={}
        self.state=bytearray(131072)
        self.coef=bytearray(131072)
        self.sword=self.saddr=self.sresponse=None
        self.cword=self.caddr=self.cresponse=None
        self.count=Counter()
        self.stages=Counter()
        self.phase='setup'
        self.stress=stress
        self.max_pending=0
        self.timeline=[]
        self.scalar_collector=0

    def advance(self,read=None,write=None,coef=None,cwrite=None,op=None,tag=''):
        latency=2 if op and (op[0].startswith('I') or op[0] in ('AAC','decode_u8','decode_shift','compare','control')) else 4
        while (self.stress and ((read is not None and self.time%32>=24) or
                (write is not None and self.time%32>=28))) or (op and self.time+latency in self.pending):
            self.advance(tag='port_or_writeback_wait')
        if self.time in self.pending:
            dst,value=self.pending.pop(self.time)
            self.rf[dst]=value
            self.count['RF_writebacks']+=1
        if self.sresponse is not None:
            self.saddr,self.sword=self.sresponse
            self.sresponse=None
        if self.cresponse is not None:
            self.caddr,self.cword=self.cresponse
            self.cresponse=None
        if read is not None:
            assert read%8==0 and 0<=read<=131064
            self.sresponse=(read,bytes(self.state[read:read+8]))
            self.count['SR64_reads']+=1
        if write is not None:
            address,payload=write
            assert address%8==0 and len(payload)==8 and 0<=address<=131064
            self.state[address:address+8]=payload
            if self.saddr==address: self.saddr=None
            self.count['SW64_writes']+=1
        if coef is not None:
            assert coef%32==0 and 0<=coef<=131040
            self.cresponse=(coef,bytes(self.coef[coef:coef+32]))
            self.count['CR256_reads']+=1
        if cwrite is not None:
            address,payload=cwrite
            assert address%32==0 and len(payload)==32 and 0<=address<=131040
            self.coef[address:address+32]=payload
            if self.caddr==address: self.caddr=None
            self.count['CW256_writes']+=1
        if op is not None:
            kind,dst,args=op
            assert self.ready[dst]<=self.time,('WAW',kind,dst,self.time,self.ready[dst])
            if kind.startswith('I'):
                value=self.integer_op(kind,dst,args)
            elif kind=='clear': value=np.zeros(8,np.float32)
            elif kind in ('load','control'): value=np.asarray(args,np.float32)
            elif kind in ('decode_u8','decode_shift'):
                address=args
                assert self.caddr==address//32*32
                code=np.frombuffer(self.cword,'i1',count=8,offset=address%32).copy()
                value=code.astype(np.float32) if kind=='decode_u8' else np.where(code==0,0,np.sign(code)*np.exp2(np.abs(code).astype(int)-16)).astype(np.float32)
            elif kind in ('AAC','FMA'):
                address,coefficient_register=args
                coefficient=self.rf[coefficient_register] if coefficient_register is not None else np.frombuffer(self.cword,'<f4',count=8)
                if coefficient_register is not None:
                    assert self.ready[coefficient_register]<=self.time
                if kind=='AAC':
                    value=np.float32(self.rf[dst]+coefficient)
                    assert np.all(value==np.rint(value)) and np.max(np.abs(value))<2**24
                else:
                    if address is None: scalar=np.float32(1)
                    else:
                        assert self.saddr==address//8*8,('source_address',self.saddr,address,self.time)
                        scalar=np.frombuffer(self.sword,'<f4',count=1,offset=address%8)[0]
                    value=vector_fma(coefficient,scalar,self.rf[dst])
            elif kind=='TF32': value=tf32_round(self.rf[dst])
            elif kind in ('scale','mul_coef','add_coef','sub_coef'):
                coefficient=np.frombuffer(self.cword,'<f4',count=8)
                value=np.float32(self.rf[dst]*coefficient) if kind in ('scale','mul_coef') else np.float32(self.rf[dst]+(coefficient if kind=='add_coef' else -coefficient))
            elif kind=='FMA_reg':
                src,coef_offset=args
                assert self.ready[src]<=self.time
                scalar=np.frombuffer(self.cword,'<f4',count=1,offset=coef_offset)[0]
                value=vector_fma(self.rf[src],scalar,self.rf[dst])
            elif kind=='compare': value=(self.rf[dst]>=0).astype(np.float32)
            else: raise ValueError(kind)
            self.pending[self.time+latency]=(dst,value.copy())
            self.ready[dst]=self.time+latency
            self.count[kind+'_issues']+=1
        self.count[tag or 'controller_slots']+=1
        self.stages[self.phase]+=1
        self.max_pending=max(self.max_pending,len(self.pending))
        self.time+=1

    def drain(self):
        while self.pending:
            self.advance(tag='pipeline_drain')

    def wait_reg(self,r):
        while self.time<=self.ready[r] and r in [v[0] for v in self.pending.values()]:
            self.advance(tag='operand_wait')

    def coefficient(self,address):
        word=address//32*32
        if self.caddr!=word:
            self.advance(coef=word,tag='coefficient_prefetch')
            self.advance(tag='coefficient_response')

    def read_word(self,address):
        word=address//8*8
        if self.saddr!=word:
            self.advance(read=word,tag='state_prefetch')
            self.advance(tag='state_response')
        return self.sword

    def dma_input(self,payload,address,coefficient=False):
        payload+=bytes((-len(payload))%32)
        for off in range(0,len(payload),32):
            for _ in range(5): self.advance(tag='DMA_input_slots')
            if coefficient:
                self.advance(cwrite=(address+off,payload[off:off+32]),tag='coefficient_fill')
            else:
                for j in range(0,32,8):
                    self.advance(write=(address+off+j,payload[off+j:off+j+8]),tag='input_store')

    def store_reg(self,r,address):
        self.wait_reg(r)
        payload=self.rf[r].astype('<f4').tobytes()
        for j in range(0,32,8):
            self.advance(write=(address+j,payload[j:j+8]),tag='vector_store')

    def load_reg(self,r,address):
        # One 32B gather region inside common64B staging, plus coefficient32B.
        payload=b''.join(self.read_word(address+j) for j in range(0,32,8))
        self.advance(op=('load',r,np.frombuffer(payload,'<f4').copy()),tag='vector_load')
        self.wait_reg(r)

    def mark(self,name,start):
        self.timeline.append(dict(phase=name,start=start,end=self.time))

    def integer_op(self,kind,dst,args):
        accum=self.rf[dst].astype(np.int64)
        if kind=='ISOURCE':
            ins=args;operands=[]
            for p in ins['operands']:
                assert self.ready[p['reg']]<=self.time
                operands.append((self.rf[p['reg']].astype(np.int64)<<p['shift'])*p['sign'])
            if ins['kind']=='addsub':value=operands[0]+operands[1]
            elif ins['kind']=='norm24':
                divisor=1<<ins['rne_shift'];q,r=np.divmod(operands[0],divisor)
                value=np.clip(q+((2*r>divisor)|((2*r==divisor)&((q&1)!=0))),-(1<<23),(1<<23)-1)
            elif ins['kind']=='gate':
                value=np.full(8,bool(ins['constant'])) if ins['constant']>=0 else (operands[0]>=ins['threshold'] if ins['direction']>0 else operands[0]<=ins['threshold'])
            else:raise ValueError(ins['kind'])
        elif kind=='ILOAD': value=np.asarray(args,np.int64)
        elif kind in ('IAAC','IMAC','IMAC_COLLECTOR'):
            offset,scalar_reg=args
            weight=np.frombuffer(self.cword,'<i2',count=8,offset=offset).astype(np.int64)
            scalar=self.scalar_collector if kind=='IMAC_COLLECTOR' else (1 if scalar_reg is None else int(self.rf[scalar_reg,0]))
            if scalar_reg is not None: assert self.ready[scalar_reg]<=self.time
            value=accum+weight*scalar
        elif kind=='IADD':
            assert self.ready[args]<=self.time
            value=accum+self.rf[args].astype(np.int64)
        elif kind=='IADD_COEF':
            value=accum+np.frombuffer(self.cword,'<i4',count=8).astype(np.int64)
        elif kind=='IRNE':
            shift=int(args)
            if shift>0:
                divisor=1<<shift;quotient,remainder=np.divmod(accum,divisor)
                value=quotient+((remainder*2>divisor)|((remainder*2==divisor)&((quotient&1)!=0)))
            else: value=accum<<(-shift)
        elif kind=='ISAT': value=np.clip(accum,-(1<<23),(1<<23)-1)
        elif kind=='ICMP':
            src=args;assert self.ready[src]<=self.time
            threshold,direction,constant,_=np.frombuffer(self.cword,'<i4',count=4)
            source=self.rf[src].astype(np.int64)
            value=np.full(8,bool(constant)) if constant>=0 else (source>=threshold if direction>0 else source<=threshold)
        else: raise ValueError(kind)
        assert np.all(value>=-(1<<47)) and np.all(value<(1<<47)),kind
        return np.asarray(value,np.float64)

    def store_i24(self,r,address):
        self.wait_reg(r)
        payload=b''.join(int(v).to_bytes(3,'little',signed=True) for v in self.rf[r])
        for j in range(0,24,8):
            self.advance(write=(address+j,payload[j:j+8]),tag='I24_vector_store')

    def load_i24(self,r,address):
        payload=b''.join(self.read_word(address+j) for j in range(0,24,8))
        value=[int.from_bytes(payload[j:j+3],'little',signed=True) for j in range(0,24,3)]
        self.advance(op=('ILOAD',r,value),tag='I24_vector_load')
        self.wait_reg(r)

    def scalar_i24(self,r,address):
        # A three-byte collector spans at most two real 64-bit responses.
        first=self.read_word(address);off=address%8
        payload=first[off:off+3]
        if len(payload)<3: payload+=self.read_word(address//8*8+8)[:3-len(payload)]
        value=int.from_bytes(payload,'little',signed=True)
        self.advance(op=('ILOAD',r,[value]*8),tag='I24_scalar_decode')
        self.wait_reg(r)

    def collect_i24(self,address):
        # Wire selection/sign extension into a 3B operand collector, rather
        # than spending an eight-lane instruction on an ordinary bit slice.
        # Consume a prior MAC's actual prefetch response before inspecting it.
        if self.sresponse is not None:self.advance(tag='I24_prefetch_response')
        first=self.read_word(address);off=address%8
        payload=first[off:off+3]
        if len(payload)<3:payload+=self.read_word(address//8*8+8)[:3-len(payload)]
        self.scalar_collector=int.from_bytes(payload,'little',signed=True)
