"""Actual integer dual-consumer continuation on the common finite machine.

sn2 cache -> full K864 U16 -> F -> BN2 + original I24 -> proj gate + PED.
Gate is produced first, then continuous U32/V96. Original I24 is supplied once
per position in an explicitly declared spatial-major T10/C96 packed24 input
interface. Source emission/repacking into that interface remains outside.
Host arrays only check values; every consumed operand comes from modeled
SRAM/RF. No CUDA statistics oracle is used in this integer endpoint.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from run_chain import Machine,read_npz,difference,GATE,HERE

I,DIR,LAT16,UPDATED,PED_U,PED_V,PROJ=0,8192,16384,20480,32768,40960,65536


def pack24(value):
    return b''.join(int(x).to_bytes(3,'little',signed=True) for x in np.asarray(value).flat)


def read24(m,address,shape):
    # Numerical oracle only: never used to feed an operation.
    count=int(np.prod(shape));raw=m.state[address:address+count*3]
    return np.asarray([int.from_bytes(raw[j:j+3],'little',signed=True)
        for j in range(0,len(raw),3)],np.int64).reshape(shape)


def coeffs(q):
    blob=bytearray();base={};bounds={}
    def add(name,payload):
        base[name]=len(blob);blob.extend(payload);blob.extend(bytes((-len(blob))%32))
    for name in ('U_conv2_theta','F','U_ped','V_ped'):
        a=q[name+'_q16'].astype(np.int64)
        add(name,a.T.astype('<i2').tobytes())
        pos=np.maximum(a,0).sum(1);neg=np.minimum(a,0).sum(1)
        lo,hi=(neg,pos) if name=='U_conv2_theta' else (-(1<<23)*pos+((1<<23)-1)*neg,((1<<23)-1)*pos-(1<<23)*neg)
        assert lo.min()>=-(1<<47) and hi.max()<(1<<47)
        bounds[name]=[int(lo.min()),int(hi.max())]
    for name in ('BN2_constant','PED_bias'): add(name,q[name+'_q24'].astype('<i4').tobytes())
    rows=np.zeros((10,8),'<i4')
    for t in range(10):
        rows[t,:4]=[q['consumer_threshold'][t],q['consumer_direction'][t],q['consumer_constant'][t],q['consumer_permutation'][t]]
    add('compare',rows.tobytes())
    assert len(blob)<=131072
    return bytes(blob),base,bounds


def complete(m,r,shift):
    m.wait_reg(r);m.advance(op=('IRNE',r,shift),tag='integer_RNE')
    m.wait_reg(r);m.advance(op=('ISAT',r,None),tag='integer_sat24')
    m.wait_reg(r)


def directory(m,geo,positions,gate_base=GATE,dir_base=DIR,phase='integer_NRVs_from_actual_sn2'):
    m.phase=phase;start=m.time
    h,w=geo['gate_shape'];oy,ox=geo['gate_origin'];n=0;live=0
    for k in range(864):
        c,rem=divmod(k,9);ky,kx=divmod(rem,3);mask=0
        for ip,(y,x) in enumerate(positions):
            sy,sx=y+ky-1,x+kx-1
            if not(0<=sy<240 and 0<=sx<320):continue
            ly,lx=sy-oy,sx-ox
            assert 0<=ly<h and 0<=lx<w
            address=gate_base+((ly*w+lx)*96+c)*2
            raw=m.read_word(address)
            mask|=int.from_bytes(raw[address%8:address%8+2],'little')<<(ip*10)
        live|=mask
        m.advance(op=('control',93,[k,mask,live,0,0,0,0,0]),tag='integer_NRV_decode')
        m.wait_reg(93)
        if mask:
            payload=int(m.rf[93,0]).to_bytes(4,'little')+int(m.rf[93,1]).to_bytes(4,'little')
            m.advance(write=(dir_base+n*8,payload),tag='integer_NRV_write');n+=1
    m.mark(m.phase,start)
    return n,live


def sparse_u(m,n,positions,base,q):
    m.phase='integer_Conv2_U16';start=m.time;tp_count=positions*10
    for r in range(tp_count*2):m.advance(op=('clear',r,None),tag='integer_U_clear')
    for rec in range(n):
        raw=m.read_word(DIR+rec*8);k=int.from_bytes(raw[:4],'little');mask=int.from_bytes(raw[4:],'little')
        m.advance(tag='integer_NRV_select');m.coefficient(base['U_conv2_theta']+k*32)
        for hg in range(2):
            # Read the actual physical group. Static zero groups cost their
            # common directory/word fetch here; no ideal unpriced W index.
            weights=np.frombuffer(m.cword,'<i2',count=8,offset=hg*16)
            if not weights.any():continue
            for tp in range(tp_count):
                if mask&(1<<tp):
                    r=tp*2+hg;m.wait_reg(r)
                    m.advance(op=('IAAC',r,(hg*16,None)),tag='integer_U_AAC')
    m.drain()
    for tp in range(tp_count):
        for hg in range(2):
            r=tp*2+hg;complete(m,r,int(q['U_conv2_theta_exponent'])-14)
            m.store_i24(r,LAT16+(tp*16+hg*8)*3)
    m.mark(m.phase,start)


def dense(m,base,name,k_count,h_count,input_base,output_base,positions,shift,live=None,bias=None):
    m.phase='integer_'+name;start=m.time;tp_count=positions*10
    active=list(range(tp_count)) if live is None else [t for t in range(tp_count) if live&(1<<t)]
    for h0 in range(0,h_count,32):
        width=min(32,h_count-h0);groups=width//8
        for r in range(tp_count*groups):m.advance(op=('clear',r,None),tag=name+'_clear')
        for k in range(k_count):
            for hg in range(groups):
                address=base[name]+(k*h_count+h0+hg*8)*2
                m.coefficient(address)
                weights=np.frombuffer(m.cword,'<i2',count=8,offset=address%32)
                if not weights.any():continue
                forward=getattr(m,'forward_i24',False)
                if forward and active:
                    m.advance(read=(input_base+(active[0]*k_count+k)*3)//8*8,tag='I24_first_source_prefetch')
                for ai,tp in enumerate(active):
                    # The same 3-byte scalar may cross a 64-bit boundary;
                    # both reads and the decode are charged before its MAC.
                    r=tp*groups+hg
                    if forward:
                        m.collect_i24(input_base+(tp*k_count+k)*3)
                        nxt=input_base+(active[ai+1]*k_count+k)*3 if ai+1<len(active) else None
                        m.wait_reg(r)
                        m.advance(read=nxt//8*8 if nxt is not None else None,
                            op=('IMAC_COLLECTOR',r,(address%32,None)),tag=name+'_MAC')
                    else:
                        m.scalar_i24(85,input_base+(tp*k_count+k)*3)
                        m.wait_reg(r);m.advance(op=('IMAC',r,(address%32,85)),tag=name+'_MAC')
        m.drain()
        for tp in range(tp_count):
            for hg in range(groups):
                r=tp*groups+hg;complete(m,r,shift)
                if bias is not None:
                    m.coefficient(base[bias]+(h0+hg*8)*4)
                    m.advance(op=('IADD_COEF',r,None),tag=name+'_bias')
                    m.wait_reg(r);m.advance(op=('ISAT',r,None),tag=name+'_bias_sat24')
                m.store_i24(r,output_base+(tp*h_count+h0+hg*8)*3)
    m.mark(m.phase,start)


def merge(m,base,positions):
    m.phase='integer_BN2_residual_merge';start=m.time
    for tp in range(positions*10):
        for h in range(0,96,8):
            m.load_i24(0,UPDATED+(tp*96+h)*3)
            m.load_i24(1,I+(tp*96+h)*3)
            m.advance(op=('IADD',0,1),tag='residual_I_add')
            m.coefficient(base['BN2_constant']+h*4);m.wait_reg(0)
            m.advance(op=('IADD_COEF',0,None),tag='BN2_constant_add')
            m.wait_reg(0);m.advance(op=('ISAT',0,None),tag='merged_sat24')
            m.store_i24(0,UPDATED+(tp*96+h)*3)
    m.mark(m.phase,start)


def projection_gates(m,base,geo,positions,input_base):
    m.phase='integer_projection_gate';start=m.time
    _,w=geo['gate_shape'];oy,ox=geo['gate_origin']
    for ip,(y,x) in enumerate(positions):
        for h in range(0,96,8):
            for t in range(10):m.load_i24(t,input_base+((ip*10+t)*96+h)*3)
            for t in range(10):
                m.coefficient(base['compare']+t*32)
                perm=int(np.frombuffer(m.cword,'<i4',count=4)[3])
                m.advance(op=('ICMP',20+t,perm),tag='integer_consumer_compare')
            m.drain();words=np.zeros(8,np.uint16)
            for t in range(10):
                words|=m.rf[20+t].astype(np.uint16)<<t
                m.advance(tag='integer_gate_collector_RF_read')
            payload=words.astype('<u2').tobytes();address=PROJ+(((y-oy)*w+x-ox)*96+h)*2
            for j in (0,8):m.advance(write=(address+j,payload[j:j+8]),tag='integer_projection_gate_store')
    m.mark(m.phase,start)


def feed_identity(m,data,label,geo,positions):
    m.phase='original_I24_input';start=m.time;oy,ox=geo['source_origin']
    # Declared interface: spatial-major, contiguous T10/C96, each pixel 2880B.
    # Pixel rows are independently 32-byte aligned, including strided anchors.
    for ip,(y,x) in enumerate(positions):
        m.dma_input(pack24(data[label+'_I24'][:,:,y-oy,x-ox]),I+ip*2880)
    m.mark(m.phase,start)


def run(data,q,label,preview_gate,stress=False,machine=None,handoff_native=False):
    geo=json.loads(str(data['window_geometry_json']))[label]
    h,w=geo['gate_shape'];oy,ox=geo['gate_origin'];out_y,out_x=geo['output_origin']
    m=Machine(stress) if machine is None else machine;begin=m.time
    before=dict(m.count);prior_stages=dict(m.stages)
    if machine is None:
        words=sum(preview_gate[t].astype(np.uint16)<<t for t in range(10)).transpose(1,2,0)
        m.phase='sn2_continuation_input';m.dma_input(words.astype('<u2').tobytes(),GATE)
    blob,base,bounds=coeffs(q)
    m.phase='integer_coefficient_replace';m.dma_input(blob,0,True)
    sy,sx=geo['source_origin']
    updated=data[label+'_I24'][:,:,oy-sy:oy-sy+h,ox-sx:ox-sx+w].astype(np.int64).copy()
    continuous=np.empty((10,96,4,4),np.int64);u_output=np.empty((10,32,4,4),np.int64)
    anchors={(2*(out_y+dy),2*(out_x+dx)) for dy in range(4) for dx in range(4)}
    rows=[]
    for y in range(oy,oy+h):
        for anchor in (True,False):
            xs=[x for x in range(ox,ox+w) if ((y,x) in anchors)==anchor]
            for i in range(0,len(xs),2):
                positions=[(y,x) for x in xs[i:i+2]];count=len(positions);start=m.time
                feed_identity(m,data,label,geo,positions)
                if anchor:
                    n,live=directory(m,geo,positions)
                    sparse_u(m,n,count,base,q)
                    dense(m,base,'F',16,96,LAT16,UPDATED,count,int(q['F_exponent']),live)
                    merge(m,base,count)
                input_base=UPDATED if anchor else I
                actual_updated=read24(m,input_base,(count,10,96))
                projection_gates(m,base,geo,positions,input_base)
                gate_ready=m.time
                if anchor:
                    dense(m,base,'U_ped',96,32,UPDATED,PED_U,count,int(q['U_ped_exponent']))
                    u_ready=m.time;actual_u=read24(m,PED_U,(count,10,32))
                    dense(m,base,'V_ped',32,96,PED_U,PED_V,count,int(q['V_ped_exponent']),bias='PED_bias')
                    actual_ped=read24(m,PED_V,(count,10,96))
                    m.phase='continuous_PED_egress'
                    for off in range(0,count*2880,32):
                        payload=b''.join(m.read_word(PED_V+off+j) for j in range(0,32,8));assert len(payload)==32
                        for _ in range(5):m.advance(tag='PED_DMA_output_slots')
                for ip,(yy,xx) in enumerate(positions):
                    updated[:,:,yy-oy,xx-ox]=actual_updated[ip]
                    if anchor:
                        dy,dx=yy//2-out_y,xx//2-out_x
                        continuous[:,:,dy,dx]=actual_ped[ip];u_output[:,:,dy,dx]=actual_u[ip]
                rows.append(dict(positions=positions,anchor=anchor,start=start,gate_ready=gate_ready,
                    U_ready=u_ready if anchor else None,end=m.time))
    if not handoff_native:
        m.phase='projection_gate_egress'
        for off in range(0,h*w*192,32):
            payload=b''.join(m.read_word(PROJ+off+j) for j in range(0,32,8));assert len(payload)==32
            for _ in range(5):m.advance(tag='gate_DMA_output_slots')
    words=np.frombuffer(m.state,'<u2',count=h*w*96,offset=PROJ).reshape(h,w,96).copy()
    gate=np.stack([(words>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    checks=dict(updated=difference(updated,data[label+'_updated_I24']),PED=difference(continuous,data[label+'_continuous_q24']),
        projection_gate=difference(gate,data[label+'_proj_gate']))
    assert all(v['differences']==0 for v in checks.values()),checks
    report=dict(service_slots=m.time-begin,stages={k:v-prior_stages.get(k,0) for k,v in m.stages.items() if v-prior_stages.get(k,0)},
        counts={k:v-before.get(k,0) for k,v in m.count.items() if v-before.get(k,0)},checks=checks,
        coefficients_bytes=len(blob),legal_accumulator48_bounds=bounds,state_high_water=PROJ+h*w*192,rows=rows,
        input_contract='Each raw I24 pixel is 2880 contiguous bytes, spatial/T/C order; source-side packing not covered.',
        common_baseline='Direct V completion. Delayed V through dynamic BN remains a necessary stronger consumer-order comparator.',
        full_chain_closed=False)
    return dict(updated=updated,continuous=continuous,gate=gate,U_ped=u_output),report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');args=ap.parse_args()
    result=dict(scope=__doc__,axes={},full_chain_closed=False,new_training=False,new_quantization=False)
    for axis in ('ordinary','lifting_raw'):
        path=HERE.parent/'capture'/axis;data=read_npz(path/'000_zurich_city_09_a_0001.npz');q=read_npz(path/'parameters.npz')
        result['axes'][axis]={}
        for label in ('corner','interior'):
            preview=read_npz(HERE/f'{axis}_{label}_preview_output.npz')
            value,report=run(data,q,label,preview['gate'],args.stress)
            result['axes'][axis][label]=report
            print(axis,label,report['service_slots'],report['checks'],flush=True)
            if not args.stress:np.savez_compressed(HERE/f'{axis}_{label}_integer_output.npz',**value)
    name='integer_stress.json' if args.stress else 'integer.json'
    (HERE/name).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
