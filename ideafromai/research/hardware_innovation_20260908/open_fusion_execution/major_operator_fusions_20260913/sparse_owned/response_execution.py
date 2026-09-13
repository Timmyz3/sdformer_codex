"""Actual shared Machine port/RAW execution of fixed K4/P2T10 response collector.
All representations get the same response decode and compact coefficient caches.
One new IENC operation uses existing signed48 IMAC and external small gate mux.
No RTL, physical timing, new multiply width, or performance extrapolation.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import probe
HERE=Path(__file__).resolve().parent
BASE=probe.OPEN.parent
MACH=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain/preview_sn2_chain'
sys.path.insert(0,str(MACH));sys.dont_write_bytecode=True
from machine import Machine as OriginalMachine
from phase_channel_probe import downstream,metrics


class Machine(OriginalMachine):
    def __init__(self,stress=False):
        super().__init__(stress)
        self.value_cache=(-1,b'');self.metadata_cache=(-1,b'')
        self.collector=np.zeros((20,4),np.int64)
        self.pop=np.zeros(20,np.int64);self.support=np.zeros((8,4),np.uint8)
        self.component_kind='dense';self.component_slot=0;self.component_offset=0
        self.component_weight=np.zeros(8,np.int64)
    def integer_op(self,kind,dst,args):
        if kind!='IENC':return super().integer_op(kind,dst,args)
        token=args
        if self.component_kind=='base':operand=np.full(8,self.pop[token],np.int64)
        elif self.component_kind=='dense':operand=np.full(8,self.collector[token,self.component_slot],np.int64)
        else:operand=self.collector[token,self.support[:,self.component_slot]]
        weight=np.frombuffer(self.value_cache[1],'<i2',count=8,offset=self.component_offset).astype(np.int64)
        value=self.rf[dst].astype(np.int64)+operand*weight
        assert value.min()>=-(1<<47) and value.max()<(1<<47)
        return value.astype(np.float64)
    def coefficient_bytes(self,address,size,kind):
        cache_name='value_cache' if kind=='value' else 'metadata_cache'
        raw=bytearray()
        end=address+size
        while address<end:
            word=address//32*32;cached,payload=getattr(self,cache_name)
            if cached!=word:
                self.advance(coef=word,tag=kind+'_request')
                self.advance(tag=kind+'_response')
                assert self.caddr==word
                setattr(self,cache_name,(word,self.cword));payload=self.cword
            count=min(end-address,32-address%32)
            raw.extend(payload[address%32:address%32+count]);address+=count
        return bytes(raw)
    def load_collector(self,addresses):
        """One SR64 request per physical word, pipelined, fixed <=64B staging."""
        words=sorted(set(int(a)//8*8 for a in addresses.flat if a>=0));staging={}
        assert len(words)<=8
        for index,address in enumerate(words):
            self.advance(read=address,tag='source_AGU_request')
            if index:staging[self.saddr]=self.sword
        if words:
            self.advance(tag='source_response_drain');staging[self.saddr]=self.sword
        self.collector[:]=0
        for p in range(len(addresses)):
            for j,address in enumerate(addresses[p]):
                if address>=0:
                    raw=staging[int(address)//8*8]
                    word=int.from_bytes(raw[int(address)%8:int(address)%8+2],'little')
                    self.collector[p*10:p*10+10,j]=(word>>np.arange(10))&1
        self.pop=self.collector.sum(1)
        # Registered 4-bit transpose/popcount after collected SR responses.
        # No RF reads; 80 gate bits +60 pop bits +20 live bits <=32B.
        self.advance(tag='collector_transpose_popcount_accept')


def run(f,row,stress=False):
    vb,mb=probe.encode(f);mbase=(len(vb)+31)//32*32
    blob=vb.ljust(mbase,b'\0')+mb
    m=Machine(stress)
    m.phase='cold_coefficient_fill';m.dma_input(blob,0,True)
    m.phase='cold_source_gate_fill';m.dma_input(row['source_blob'],0)
    out=np.zeros((len(row['x']),10,16),np.int64)
    for p0 in range(0,len(row['x']),2):
        m.phase='P2_accumulator_clear'
        for dst in range(40):m.wait_reg(dst);m.advance(op=('ILOAD',dst,np.zeros(8,np.int64)),tag='accumulator_clear')
        for group in range(216):
            m.phase='source_response_collector'
            m.load_collector(row['addresses'][p0:p0+2,group*4:group*4+4])
            expected=row['x'][p0:p0+2,:,group*4:group*4+4].reshape(20,4)
            assert np.array_equal(m.collector,expected)
            if not np.any(m.pop):continue
            for hb in range(2):
                m.phase='compressed_coefficient_decode'
                if mb:
                    bits=f['n']*2;address=mbase+(group*16+hb*8)*bits//8
                    raw=m.coefficient_bytes(address,bits,'metadata');stream=int.from_bytes(raw,'little')
                    support=np.array([[(stream>>(lane*bits+2*j))&3 for j in range(f['n'])] for lane in range(8)],np.uint8)
                    m.support[:,:f['n']]=support
                    m.advance(tag='metadata_support_accept')
                for slot in range(f['n']+int(f['offset'])):
                    m.phase='compressed_coefficient_decode'
                    base=f['offset'] and slot==0
                    j=slot-int(f['offset'])
                    m.component_kind='base' if base else 'dense' if f['mode']=='dense' else 'sparse'
                    m.component_slot=max(j,0)
                    operand=np.broadcast_to(m.pop[:,None],(20,8)) if base else m.collector[:,m.support[:,j]] if mb else np.broadcast_to(m.collector[:,j,None],(20,8))
                    # Runtime mask only; no output/flow oracle. Decoder query is
                    # shared with the instruction issue, both ordinary/candidate.
                    active=np.flatnonzero(np.any(operand,axis=1))
                    m.advance(tag='component_mask_decode_and_select')
                    if not len(active):continue
                    address=((group*2+hb)*(f['n']+int(f['offset']))+slot)*16
                    raw=m.coefficient_bytes(address,16,'value');m.component_offset=address%32
                    assert np.array_equal(np.frombuffer(raw,'<i2'),f['values'][group,hb*8:hb*8+8,slot])
                    m.phase='IENC_shared_IMAC'
                    for token in active:
                        dst=int(token)*2+hb;m.wait_reg(dst)
                        m.advance(op=('IENC',dst,int(token)),tag='encoded_accumulate')
        m.drain()
        m.phase='U_signed48_writeback'
        for token in range(20):
            for hb in range(2):
                value=m.rf[token*2+hb].astype(np.int64)
                out[p0+token//10,token%10,hb*8:hb*8+8]=value
                payload=b''.join(int(v).to_bytes(6,'little',signed=True) for v in value)
                address=32768+(token*16+hb*8)*6
                for k in range(0,48,8):m.advance(write=(address+k,payload[k:k+8]),tag='U_signed48_store')
    gold=row['x']@f['reconstructed'].T
    assert np.array_equal(out,gold)
    return out,dict(service_slots=m.time,stages=dict(m.stages),counts=dict(m.count),
        maximum_pending_RF_writes=m.max_pending,coefficient_bytes=len(blob),
        packed_acc_mismatches=0,RF_count=96,used_accumulator_RF=40,extra_RF_reads=0,
        decoder_storage=dict(source_SR64_staging=64,collector=32,coefficient_value_response=32,metadata_response=32,support_and_dispatch=16),
        integer_MAC='unchanged16-bit coefficient x existing scalar input width to signed48; operand mux now selects0/1 or common popcount0..4; no wider multiply',
        gate_and_PED='same full downstream reference after complete K864, original exponents and RNE/sat')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');args=ap.parse_args()
    q=probe.load(probe.CAP/'deployed_constants.npz')
    rows=[]
    for p in sorted(probe.CAP.glob('[0-9][0-9][0-9]_*.npz')):
        d=probe.load(p)
        for label in ('corner','interior'):
            row=probe.samples(d,label);row['file']=p.name;rows.append(row)
    if args.stress:rows=[rows[-1]]
    result=dict(scope=__doc__,stress=args.stress,layout='natural_K4/P2T10 only',modes={},complete=False,
                common_grants='all modes receive same response collector, dual coefficient response latch, lane-index mux, shared single-issue IMAC and pipeline/ports; no stored packet or uncharged extra RF access')
    for mode in ('dense','ordinary24','ordinary34','shifted14','shifted24'):
        p=probe.load(HERE/f'natural_{mode}.npz')
        f=dict(mode=mode,offset=bool(p['offset']),n=p['indices'].shape[-1],values=p['values'],indices=p['indices'],reconstructed=p['U_conv2_theta_q16'].astype(np.int64))
        records=[];result['modes'][mode]=records
        for row in rows:
            out,report=run(f,row,args.stress)
            report.update(file=row['file'],window=row['label'],numerical=metrics(downstream(out,row['raw'],q),row['gold']))
            records.append(report)
            print(mode,row['file'],row['label'],report['service_slots'],flush=True)
    result['complete']=True
    result['totals']={mode:dict(service_slots=sum(r['service_slots'] for r in records),
        CR256_reads=sum(r['counts'].get('CR256_reads',0) for r in records),
        SR64_reads=sum(r['counts'].get('SR64_reads',0) for r in records),
        SW64_writes=sum(r['counts'].get('SW64_writes',0) for r in records),
        IENC_issues=sum(r['counts'].get('IENC_issues',0) for r in records)) for mode,records in result['modes'].items()}
    (HERE/('response_stress.json' if args.stress else 'response_results.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['totals'],indent=2))

if __name__=='__main__':main()
