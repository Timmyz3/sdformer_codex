"""Full-K subset matching directly on NRV columns, with paid payload execution.

The accompanying RTL implements the additional 20-token relation detector and
residual filter. Arithmetic/RF/SRAM ports stay fixed, but added detector area
and attainable clock are NOT known. Neither arm is an official PR replication.
"""
from pathlib import Path
import json
import argparse
import numpy as np
import compare as old

HERE=Path(__file__).resolve().parent
TOKENS=20


class Relation:
    def __init__(self):
        self.eligible=[(1<<TOKENS)-1]*TOKENS
        self.count=[0]*TOKENS
        self.parents=[-1]*TOKENS
        self.keys=[0]*TOKENS
        self.columns=0

    def column(self,k,mask):
        self.columns+=1
        for i in range(TOKENS):
            if mask&(1<<i):
                self.count[i]+=1
                self.keys[i]|=1<<k  # independent oracle only, no execution storage
            else:
                self.eligible[i]&=~mask

    def select(self,m):
        m.phase='column_parent_selection'
        m.advance(tag='detector_finish')
        packet=[]
        for i in range(TOKENS):
            best=0
            if self.count[i]>=2:
                for j in range(TOKENS):
                    if self.eligible[i]&(1<<j) and self.count[j]>best and \
                       (self.count[j]<self.count[i] or j<i):
                        self.parents[i]=j;best=self.count[j]
            packet.append((self.count[i]<<16)|(self.parents[i]+1))
            payload=None
            if i%2:
                payload=b''.join(v.to_bytes(4,'little') for v in packet)
                packet=[]
            # The RTL emits one parent every accepted cycle. A 32-bit packing
            # latch forms the existing 64-bit SRAM parent table, not extra DMA.
            m.advance(write=(old.PARENTS+(i-1)*4,payload) if payload is not None else None,
                      tag='parent_output_accept')
        expected,counts=old.parent_rule(self.keys)
        assert self.parents==expected and self.count==counts

    def residual(self,mask):
        out=mask
        for i,p in enumerate(self.parents):
            if p>=0 and mask&(1<<p):
                assert mask&(1<<i)
                out&=~(1<<i)
        return out


def build_with_detector(m,words,source_origin,positions,detector):
    """Same source reads/NRV producer as run_chain; accepted NRV write fans out.

    The new always-ready FILL port observes this very mask on that cycle. This
    is parallel hardware, not a free software scan; RTL area/timing are pending.
    """
    m.phase='column_detector_start';m.advance(tag='detector_start')
    m.phase='build_P2_NRVs'
    n=occurrences=live=0
    height,width=words.shape[1:]
    for k in range(864):
        c,rem=divmod(k,9);ky,kx=divmod(rem,3)
        mask=0
        for ip,(y,x) in enumerate(positions):
            sy,sx=y+ky-1,x+kx-1
            if not(0<=sy<240 and 0<=sx<320):continue
            ly,lx=sy-source_origin[0],sx-source_origin[1]
            assert 0<=ly<height and 0<=lx<width
            addr=2*((c*height+ly)*width+lx)
            raw=m.read_word(addr)
            word=int.from_bytes(raw[addr%8:addr%8+2],'little')
            mask|=word<<(ip*10)
        live|=mask
        m.advance(op=('control',93,[k,mask,live,0,0,0,0,0]),tag='NRV_decode')
        m.wait_reg(93)
        if mask:
            payload=int(m.rf[93,0]).to_bytes(4,'little')+int(m.rf[93,1]).to_bytes(4,'little')
            m.advance(write=(old.NRV+n*8,payload),tag='NRV_write_and_relation_accept')
            detector.column(k,mask)
            n+=1;occurrences+=mask.bit_count()
    return n,occurrences,live


def scan_detector(m,n):
    d=Relation()
    m.phase='column_detector_scan';m.advance(tag='detector_start')
    for i in range(n):
        raw=m.read_word(old.NRV+i*8)
        k=int.from_bytes(raw[:4],'little');mask=int.from_bytes(raw[4:],'little')
        m.advance(tag='detector_column_accept')
        d.column(k,mask)
    return d


def filtered_u(m,n,base,rows,d):
    m.phase='column_filtered_GP'
    for r in range(rows*4):m.advance(op=('clear',r,None),tag='U_clear')
    removed=empty_records=0
    for i in range(n):
        raw=m.read_word(old.NRV+i*8)
        k=int.from_bytes(raw[:4],'little');original=int.from_bytes(raw[4:],'little')
        mask=d.residual(original)
        # One registered filter step fits the existing NRV-select step. This
        # is the RTL residual datapath, not a zero-latency software selector.
        m.advance(tag='residual_filter_select')
        removed+=original.bit_count()-mask.bit_count()
        if not mask:
            empty_records+=1
            continue
        for hg in range(4):
            m.coefficient(base['U']+(k*32+hg*8)*4)
            for tp in range(rows):
                if mask&(1<<tp):
                    dst=tp*4+hg;m.wait_reg(dst)
                    m.advance(op=('FMA',dst,(None,None)),tag='U_active_issue')
    m.phase='column_parent_RF_propagation'
    for tp in sorted(range(rows),key=lambda i:(d.count[i],i)):
        if d.parents[tp]<0:continue
        address=old.PARENTS+tp*4
        raw=m.read_word(address)
        parent=(int.from_bytes(raw[address%8:address%8+4],'little')&65535)-1
        assert parent==d.parents[tp]
        for hg in range(4):
            dst=tp*4+hg;src=parent*4+hg
            m.wait_reg(dst);m.wait_reg(src)
            m.advance(op=('add_reg',dst,src),tag='parent_vector_add')
    m.drain()
    z=np.empty((rows,32),np.float32)
    m.phase='common_Z_output'
    # Same order as ordinary execute_u to avoid unrelated writeback schedules.
    for hg in range(4):
        for tp in range(rows):
            r=tp*4+hg;z[tp,hg*8:hg*8+8]=m.rf[r]
            m.advance(op=('TF32',r,None),tag='Z_conversion')
            m.store_reg(r,old.Z+(tp*32+hg*8)*4)
    return z,removed,empty_records


def run(data,p,label,mode,stress):
    geo=json.loads(str(data['window_geometry_json']))[label]
    source=data[label+'_sn1_gate']
    words=sum(source[t].astype(np.uint16)<<t for t in range(10))
    blob,base,info=old.coefficients(p,False)
    m=old.Machine(stress)
    m.phase='common_coefficient_fill';m.dma_input(blob,0,True)
    m.phase='common_source_input';m.dma_input(words.astype('<u2').tobytes(),old.SRC)
    out=[];groups=[]
    h,w=geo['gate_shape'];oy,ox=geo['gate_origin']
    for y in range(h):
        for x in range(0,w,2):
            positions=[(oy+y,ox+x+i) for i in range(min(2,w-x))]
            rows=10*len(positions)
            start=m.time;d=Relation()
            if mode=='producer_coupled':
                n,occ,live=build_with_detector(m,words,geo['source_origin'],positions,d)
            else:
                n,occ,live=old.build_nrv(m,words,geo['source_origin'],positions)
                d=scan_detector(m,n)
            d.select(m)
            z,removed,empty=filtered_u(m,n,base,rows,d)
            out.append(z)
            groups.append(dict(y=y,x=x,logical_tokens=rows,detector_tokens=20,NRV_rows=n,
                source_occurrences=occ,removed_occurrences=removed,
                parents=sum(q>=0 for q in d.parents),empty_residual_rows=empty,service=m.time-start))
    return np.concatenate(out),dict(service_slots=m.time,stages=dict(m.stages),counts=dict(m.count),groups=groups,
        coefficient_bytes=info['used_bytes'],state_capacity=131072,RF_words=96,
        detector_logical_state_bits=dict(eligible=400,counts=200,parents=100,parent_valid=20,
            cursor=5,state=2,residual_output=21,parent_packing_latch=32),
        same_area_or_clock_proven=False,K=864,H=32,T=10,
        source_relation_overlap=mode=='producer_coupled',residual_format='retained original NRV, registered on-read filter')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');args=ap.parse_args()
    result=dict(scope=__doc__,axes={})
    for axis in ('ordinary','lifting_raw'):
        cap=old.FULL/'capture'/axis
        data=old.read_npz(cap/'000_zurich_city_09_a_0001.npz');p=old.read_npz(cap/'live_parameters.npz')
        result['axes'][axis]={}
        for label in ('corner','interior'):
            ref,gp=old.run(data,p,label,'gp',args.stress)
            row={'gp':gp}
            for mode in ('column_scan','producer_coupled'):
                z,report=run(data,p,label,mode,args.stress)
                report['difference_vs_GP']=old.difference(z,ref)
                assert report['difference_vs_GP']['differences']==0
                report['net_service_reduction']=1-report['service_slots']/gp['service_slots']
                row[mode]=report
                print(axis,label,mode,report['service_slots'],report['net_service_reduction'],flush=True)
            result['axes'][axis][label]=row
    (HERE/('column_stress.json' if args.stress else 'column_ready.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
