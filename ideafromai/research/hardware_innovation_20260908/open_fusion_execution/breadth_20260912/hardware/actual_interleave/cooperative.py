"""Two resumable controllers, exactly one existing payload Machine and clock."""
from collections import Counter
import copy
import numpy as np
from greenlet import greenlet,getcurrent
import packed_weights
Base=packed_weights.stage.IntegratedMachine

class CooperativeMachine(Base):
    def __init__(self,stress=False):
        self.flow_ids={};self.flow_phases={};self.locks={};self.destinations={};self.requests={}
        self.scheduler_counts=Counter();self.active=False
        super().__init__(stress)

    @property
    def phase(self):
        who=self.flow_ids.get(getcurrent())
        return self.flow_phases.get(who,self._phase) if who else self._phase

    @phase.setter
    def phase(self,value):
        who=self.flow_ids.get(getcurrent())
        if who:self.flow_phases[who]=value
        else:self._phase=value

    def who(self):return self.flow_ids.get(getcurrent()) if self.active else None

    def pause(self,event):
        return getcurrent().parent.switch(event)

    def advance(self,read=None,write=None,coef=None,cwrite=None,op=None,tag=''):
        if not self.who():
            return super().advance(read=read,write=write,coef=coef,cwrite=cwrite,op=op,tag=tag)
        if op is not None:
            self.destinations[self.who()].add(op[1])
            allowed=self.allowed[self.who()]
            assert op[1] in allowed,(self.who(),'destination',op)
            reads=[]
            if op[0]=='ISOURCE':reads=[p['reg'] for p in op[2]['operands']]
            elif op[0]=='IMAC_INDEX':reads=[op[1],op[2][1]]
            elif op[0] in ('IRNE','ISAT','IADD_COEF'):reads=[op[1]]
            assert len(reads)<=2 and all(r in allowed for r in reads),(self.who(),'RF ownership',op)
        return self.pause(dict(kind='issue',phase=self.phase,kwargs=dict(read=read,write=write,coef=coef,cwrite=cwrite,op=op,tag=tag)))

    def pending_reg(self,r):return any(dst==r for dst,_ in self.pending.values())

    def wait_reg(self,r):
        if not self.who():return super().wait_reg(r)
        while self.pending_reg(r):self.pause(dict(kind='wait',reason='RF',predicate=lambda r=r:not self.pending_reg(r)))

    def drain(self):
        if not self.who():return super().drain()
        owned=self.destinations[self.who()]
        while any(dst in owned for dst,_ in self.pending.values()):
            self.pause(dict(kind='wait',reason='own_writeback',predicate=lambda owned=owned:not any(dst in owned for dst,_ in self.pending.values())))

    def acquire(self,key):
        who=self.who()
        if not who:return
        while self.locks.get(key) not in (None,who):
            self.pause(dict(kind='wait',reason='lock_'+key,predicate=lambda key=key,who=who:self.locks.get(key) in (None,who)))
        assert self.locks.get(key) is None,(key,'nested lock')
        self.locks[key]=who

    def release(self,key):
        if self.who():
            assert self.locks[key]==self.who();self.locks[key]=None

    def read_word(self,address):
        if not self.who():return super().read_word(address)
        self.acquire('SR_response')
        word=address//8*8
        if self.saddr!=word:
            self.advance(read=word,tag='state_prefetch')
            while self.sresponse is not None:
                self.pause(dict(kind='wait',reason='SR_response',predicate=lambda:self.sresponse is None))
        assert self.saddr==word
        result=self.sword
        self.release('SR_response')
        return result

    def coefficient(self,address):
        if not self.who():return super().coefficient(address)
        word=address//32*32
        if self.caddr!=word:
            self.advance(coef=word,tag='coefficient_prefetch')
            while self.cresponse is not None:
                self.pause(dict(kind='wait',reason='CR_response',predicate=lambda:self.cresponse is None))
        assert self.caddr==word

    def load_i24(self,r,address):
        self.acquire('gather24')
        payload=b''.join(self.read_word(address+j) for j in range(0,24,8))
        values=[int.from_bytes(payload[j:j+3],'little',signed=True) for j in range(0,24,3)]
        self.wait_reg(r)
        self.advance(op=('ILOAD',r,values),tag='I24_vector_load')
        self.release('gather24')
        self.wait_reg(r)

    def collect_i24(self,address):
        first=self.read_word(address);off=address%8;payload=first[off:off+3]
        if len(payload)<3:payload+=self.read_word(address//8*8+8)[:3-len(payload)]
        self.scalar_collector=int.from_bytes(payload,'little',signed=True)

    def dma_input(self,payload,address,coefficient=False):
        if not self.who():return super().dma_input(payload,address,coefficient)
        payload+=bytes((-len(payload))%32)
        for off in range(0,len(payload),32):
            self.acquire('DMA32')
            for _ in range(5):self.advance(tag='DMA_input_slots')
            if coefficient:
                self.advance(cwrite=(address+off,payload[off:off+32]),tag='coefficient_fill')
            else:
                for j in (0,8,16,24):self.advance(write=(address+off+j,payload[off+j:off+j+8]),tag='input_store')
            self.release('DMA32')

    def store_i24(self,r,address):
        self.wait_reg(r)
        # Serialize the still-live RF, not an unbudgeted24B output queue.
        for j in range(0,24,8):
            payload=b''.join(int(v).to_bytes(3,'little',signed=True) for v in self.rf[r])[j:j+8]
            self.advance(write=(address+j,payload),tag='I24_vector_store')

    def issue_ready(self,request):
        q=request['kwargs'];op=q['op']
        if self.stress and ((q['read'] is not None and self.time%32>=24) or (q['write'] is not None and self.time%32>=28)):return False
        if op:
            latency=2 if op[0].startswith('I') or op[0] in ('AAC','decode_u8','decode_shift','compare','control') else 4
            if self.time+latency in self.pending:return False
            if self.ready[op[1]]>self.time:return False
        return True

    def execute_flows(self,source,consumer,source_regs,consumer_regs):
        assert not set(source_regs)&set(consumer_regs)
        self.allowed={'source':set(source_regs),'consumer':set(consumer_regs)}
        self.active=True;self.destinations={'source':set(),'consumer':set()}
        fibers={k:greenlet(fn) for k,fn in [('source',source),('consumer',consumer)]}
        self.flow_ids={fiber:k for k,fiber in fibers.items()};self.flow_phases={k:'interleave_'+k for k in fibers}
        requests={k:fiber.switch() for k,fiber in fibers.items()}
        turn=0;names=['source','consumer'];start=self.time;finish={}
        while not all(f.dead for f in fibers.values()):
            for k in names:
                if fibers[k].dead:
                    finish.setdefault(k,self.time);continue
                e=requests[k]
                while e['kind']=='wait' and e['predicate']():
                    e=fibers[k].switch();requests[k]=e
                    if fibers[k].dead:finish[k]=self.time;break
            chosen=None
            for k in names[turn:]+names[:turn]:
                if fibers[k].dead:continue
                e=requests[k]
                if e['kind']=='issue' and self.issue_ready(e):chosen=k;break
            if chosen is None:
                if all(f.dead for f in fibers.values()):break
                assert self.pending or self.sresponse is not None or self.cresponse is not None or self.stress,('deadlock',requests,self.locks)
                self._phase='shared_wait'
                super().advance(tag='shared_wait_slots');self.scheduler_counts['no_ready_flow']+=1
                continue
            e=requests[chosen];self._phase=e['phase']
            t=self.time;super().advance(**e['kwargs']);assert self.time==t+1
            self.scheduler_counts[chosen+'_issued_slots']+=1
            turn=1-names.index(chosen)
            requests[chosen]=fibers[chosen].switch()
            if fibers[chosen].dead:finish[chosen]=self.time
        self.active=False;self.flow_ids={}
        self.drain()
        assert all(v is None for v in self.locks.values())
        return dict(start=start,end=self.time,flow_finish=finish,counts=dict(self.scheduler_counts),scheduler='One shared issue, fixed round-robin among ready operations; no second RF/clock or merged historical counts.')
