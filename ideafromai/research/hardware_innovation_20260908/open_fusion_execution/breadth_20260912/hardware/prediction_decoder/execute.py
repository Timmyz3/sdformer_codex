from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import packed_weights as shared
stage=shared.stage
OPEN=HERE.parents[2]
PARAMETERS=OPEN/'breadth_20260912/representation/parameters'
EXPORT=OPEN/'stage_20260912/algorithm/hardware_exports'
INPUT,OUTPUT=0,8192


def pack24(x):return b''.join(int(v).to_bytes(3,'little',signed=True) for v in np.asarray(x).flat)


class DecoderMachine(shared.PackedMachine):
    def __init__(self,stress=False):
        super().__init__(stress);self.predictor_mask=np.zeros(8,bool)
        self.pred_unpack=bytes(24)
    def integer_op(self,kind,dst,args):
        if kind=='IPRED_UNPACK24':
            value=np.array([int.from_bytes(self.pred_unpack[j:j+3],'little',signed=True) for j in range(0,24,3)],np.int64)
        elif kind=='IPRED_BCAST':
            r,l=args;assert self.ready[r]<=self.time
            value=np.full(8,int(self.rf[r,l]),np.int64)
        elif kind=='IPRED_MASKED_CACHED_ADD':
            r,l=args;assert self.ready[r]<=self.time
            value=self.rf[dst].astype(np.int64)+int(self.rf[r,l])*self.predictor_mask.astype(np.int64)
        else:return super().integer_op(kind,dst,args)
        assert np.all(value>=-(1<<47)) and np.all(value<(1<<47))
        return value.astype(np.float64)
    def load_packed24(self,dst,address,n=8):
        payload=bytearray();size=n*3
        while size:
            self.coefficient(address);take=min(size,32-address%32)
            payload.extend(self.cword[address%32:address%32+take]);address+=take;size-=take
        self.pred_unpack=bytes(payload)+bytes(24-len(payload))
        self.wait_reg(dst);self.advance(op=('IPRED_UNPACK24',dst,None),tag='predictor_packed24_decode')
        self.wait_reg(dst)
    def load_gates(self,offset):
        raw=b''.join(self.read_word(offset+j) for j in (0,8))
        self.wait_reg(10);self.advance(op=('ILOAD',10,np.frombuffer(raw,'<u2').astype(np.int64)),tag='gate_T10_word_load')
        self.wait_reg(10)
    def select_mask(self,bit=None,half=None,key=None):
        assert self.ready[10]<=self.time
        g=self.rf[10].astype(np.int64)
        # One paid RF vector read and bounded 8-lane bit/5-bit comparator,
        # followed by an8-bit controller predicate, not a third MAC RF read.
        self.advance(tag='predictor_condition_select' if bit is not None else 'LUT_pattern_match')
        self.predictor_mask=(((g>>bit)&1)!=0) if bit is not None else (((g>>(half*5))&31)==key)


def all_patterns(D,c):
    bits=((np.arange(1024)[:,None]>>np.arange(10))&1).astype(np.int64)
    expected=bits@D.T+c
    halves=[]
    small=((np.arange(32)[:,None]>>np.arange(5))&1).astype(np.int64)
    for half in range(2):halves.append(small@D[:,half*5:(half+1)*5].T+(c if half==0 else 0))
    lut=halves[0][np.arange(1024)&31]+halves[1][np.arange(1024)>>5]
    assert np.array_equal(expected,lut)
    assert all(np.all(x>=-(1<<23)) and np.all(x<(1<<23)) for x in [expected,*halves])
    return halves,dict(patterns=1024,values=10240,differences=0,complete_min=int(expected.min()),
        complete_max=int(expected.max()),half_min=min(int(h.min()) for h in halves),half_max=max(int(h.max()) for h in halves),
        prediction_requires_no_saturation=True)


def constant_image(D,c,method):
    if method=='LUT5plus5':
        halves,check=all_patterns(D,c)
        image=b''.join(pack24(row)+bytes(2) for half in halves for row in half)
        assert len(image)==2048
        return image,dict(zero_rows=[half*32+key for half in range(2) for key in range(32)
            if not np.any(halves[half][key])]),check
    entries=[(s,t,int(D[t,s])) for s in range(10) for t in range(10) if D[t,s]!=0]
    values=np.array(list(c)+[v for _,_,v in entries],np.int64)
    assert np.all(values>=-(1<<23)) and np.all(values<(1<<23))
    assert 16+(len(values)+7)//8<=30
    return pack24(values),dict(entries=entries,count=len(values)),None


def execute(words,D,c,method,stress=False):
    image,meta,check=constant_image(D,c,method)
    m=DecoderMachine(stress)
    raw=words.astype('<u2').tobytes()
    m.state[INPUT:INPUT+len(raw)]=raw # Explicit already-ready input boundary.
    m.phase='predictor_constant_cold_fill';m.dma_input(image,0,True)
    if method=='condition_add':
        m.phase='predictor_constant_RF_cache'
        for i in range(0,meta['count'],8):m.load_packed24(16+i//8,i*3,min(8,meta['count']-i))
    else:
        # Same ordinary c-cache permission as the center/condition controls.
        # Read the actual first-table row0=c once, through CR256/packed24.
        m.phase='predictor_constant_RF_cache'
        m.load_packed24(32,0,8);m.load_packed24(33,24,2)
    setup=m.time;pattern_groups=[]
    for group in range(len(words)//8):
        m.phase='predictor_gate_input'
        need_g=method=='LUT5plus5' or np.any(D)
        if need_g:m.load_gates(INPUT+group*16)
        m.phase='predictor_decode'
        if method=='condition_add':
            for t in range(10):
                m.wait_reg(t);m.advance(op=('IPRED_BCAST',t,(16+t//8,t%8)),tag='predictor_c_broadcast')
            cursor=10
            for s in range(10):
                entries=[(t,v) for col,t,v in meta['entries'] if col==s]
                if not entries:continue
                m.select_mask(bit=s)
                for t,_ in entries:
                    if m.predictor_mask.any():
                        m.wait_reg(t);m.advance(op=('IPRED_MASKED_CACHED_ADD',t,(16+cursor//8,cursor%8)),tag='predictor_conditional_add')
                    cursor+=1
        else:
            for t in range(10):
                m.wait_reg(t);m.advance(op=('ILOAD',t,np.zeros(8,np.int64)),tag='predictor_clear')
            for half in range(2):
                covered=0;reads=0
                while covered!=255:
                    m.advance(tag='LUT_next_uncovered_lane')
                    lane=next(l for l in range(8) if not covered&(1<<l))
                    key=(int(m.rf[10,lane])>>(half*5))&31
                    m.select_mask(half=half,key=key)
                    covered|=sum(int(v)<<l for l,v in enumerate(m.predictor_mask))
                    reads+=1
                    if half*32+key in meta['zero_rows']:
                        m.advance(tag='LUT_static_zero_row_skip')
                        continue
                    address=(half*32+key)*32
                    cache=32 if half==0 and key==0 else 30
                    if cache==32:m.count['LUT_cached_c_row_uses']+=1
                    else:
                        m.load_packed24(30,address,8);m.load_packed24(31,address+24,2)
                        m.count['LUT_row_lookups']+=1
                    for t in range(10):
                        m.wait_reg(t);m.advance(op=('IPRED_MASKED_CACHED_ADD',t,(cache+t//8,t%8)),tag='LUT_prediction_merge')
                pattern_groups.append(reads)
        m.phase='predictor_output_store'
        for t in range(10):m.store_i24(t,OUTPUT+(group*10+t)*24)
    expected=np.array([D@((w>>np.arange(10))&1)+c for w in words],np.int64)
    actual=shared.integer.read24(m,OUTPUT,(len(words)//8,10,8)).transpose(0,2,1).reshape(-1,10)
    assert np.array_equal(actual,expected),(method,np.count_nonzero(actual!=expected))
    assert sum(m.stages.values())==m.time
    return dict(method=method,stress=stress,service_slots=m.time,cold_fill_and_cache_slots=setup,
        post_setup_service_slots=m.time-setup,stages=dict(m.stages),counts=dict(m.count),
        gate_words=len(words),H8_groups=len(words)//8,output_values=int(actual.size),differences=0,
        coefficient_body_bytes=len(image),coefficient_padded_bytes=(len(image)+31)//32*32,
        RF_constant_cache_vectors=(meta['count']+7)//8 if method=='condition_add' else 2,
        RF_used_highest=33 if method=='LUT5plus5' else max(10,15+(meta['count']+7)//8),
        port_bytes=dict(SR64=m.count['SR64_reads']*8,SW64=m.count['SW64_writes']*8,
            CR256=m.count['CR256_reads']*32,CW256=m.count['CW256_writes']*32),
        LUT_half_pattern_groups=pattern_groups,all_gate_patterns_check=check)


def actual_words(axis,label):
    with np.load(EXPORT/axis/'000_zurich_city_09_a_0001.npz') as z:
        geo=json.loads(str(z['window_geometry_json']))[label]
        y,x=geo['gate_origin'];oy,ox=geo['output_origin'];dy,dx=2*oy-y,2*ox-x
        g=z[label+'_proj_gate'][:,:,dy:dy+8:2,dx:dx+8:2]
    # Spatial/H8 with all T bits in each actual channel word.
    words=sum(g[t].astype(np.uint16)<<t for t in range(10)).transpose(1,2,0).reshape(-1)
    assert len(words)==1536
    return words


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');args=ap.parse_args()
    result=dict(scope='Already-ready actual projection-g SRAM -> Dg+c predictor SRAM, four final R24-parent windows.',
        input_gate_generation=False,residual_encoding=False,PED_or_full_chain=False,
        evidence='CPU payload slot prototype, not RTL/PPA/AEE.',
        constants_source='New train-only fit from thun_00_a_0012,4096points; no old D substituted.',
        coefficient_cache='Common128KiB coefficient pool and existing32B CR response. Condition-add constants additionally fit paid existing RF; LUT64 padded rows do not fit the same straightforward2RF/row representation.',
        budget=dict(RF_vectors=96,lanes=8,word_bits=48,state_bytes=131072,coefficient_bytes=131072,
            SRAM_ports='SR64/SW64',coefficient_port='CR256',control_predicate_bits=8,existing_unpack_staging_bytes=24),rows=[])
    path=HERE/('stress.json' if args.stress else 'ready.json')
    for axis in (['ordinary'] if args.stress else ['ordinary','lifting_raw']):
        params=dict(np.load(PARAMETERS/(axis+'.npz')))
        for label in (['interior'] if args.stress else ['corner','interior']):
            words=actual_words(axis,label)
            for mode in ('fixed_q8','center_q8','affine_q8','diagonal_g_q8','full_g_q8'):
                D=params[mode+'_D'].astype(np.int64);c=params[mode+'_c'].astype(np.int64)
                for method in (['condition_add','LUT5plus5'] if mode=='full_g_q8' else ['condition_add']):
                    row=execute(words,D,c,method,args.stress)
                    row.update(axis=axis,window=label,mode=mode,D_nonzero=int(np.count_nonzero(D)))
                    result['rows'].append(row);path.write_text(json.dumps(result,indent=2)+'\n')
                    print(axis,label,mode,method,row['service_slots'],row['port_bytes'],flush=True)
    result['complete']=True;path.write_text(json.dumps(result,indent=2)+'\n')
    print('PREDICTOR_DECODE_COMPLETE',flush=True)


if __name__=='__main__':main()
