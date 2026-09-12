"""Finite H4 occupancy metadata, generated from actual committed gate words."""
import numpy as np
import run_binding as binding
common=binding.pair.common
stage=binding.pair.matched.stage
addresses=stage.addresses
SOURCE=stage.SRC
META=112640
SN2_META=114688
META_RF=94

class MetadataMachine(stage.IntegratedMachine):
    def integer_op(self,kind,dst,args):
        if kind in ('IMETA_GATE','IMETA_WORD'):
            value=self.rf[dst].astype(np.int64).copy()
            if kind=='IMETA_GATE':
                payload,index=args;words=np.frombuffer(payload,'<u2')
                bits=int(np.any(words[:4]&1023))|(int(np.any(words[4:]&1023))<<1)
            else:
                index=args;words=np.frombuffer(self.sword,'<u2');bits=int(np.any(words&1023))
            value[0]|=bits<<index
        elif kind=='IMETA_GEOMETRY':
            value=self.rf[dst].astype(np.int64).copy();value[4+args]=int.from_bytes(self.sword,'little')&0xffffff
        elif kind=='IMETA_PERMISSION':
            value=self.rf[dst].astype(np.int64).copy();valid=int(value[6]);permission=0
            for ip in range(2):
                if valid&(1<<ip) and (int(value[4+ip])>>args)&1:permission|=1<<ip
            value[7]=permission
        else:return super().integer_op(kind,dst,args)
        return value.astype(np.float64)

def producer_summary(m,payload,h):
    m.wait_reg(META_RF)
    m.advance(op=('IMETA_GATE',META_RF,(payload,h//4)),tag='producer_H4_summary_update')
    m.wait_reg(META_RF)

def write_summary(m,pixel):
    m.wait_reg(META_RF)
    m.advance(write=(META+pixel*8,int(m.rf[META_RF,0]).to_bytes(8,'little')),tag='H4_summary_store')

def scan_summary(m,count,source_base,meta_base,phase):
    """Strong ordinary scan: pair words and overlap next read with summary op."""
    m.phase=phase
    for pixel in range(count):
        address=source_base+pixel*192
        m.wait_reg(META_RF)
        m.advance(read=address,op=('ILOAD',META_RF,[0]*8),tag='summary_clear_and_first_prefetch')
        m.advance(tag='summary_prefetch_response')
        for group in range(12):
            assert m.saddr==address+16*group
            first=m.sword # Existing16B collector holds this and the next word.
            m.advance(read=address+16*group+8,tag='summary_second_word_prefetch')
            m.advance(tag='summary_prefetch_response')
            payload=first+m.sword;m.wait_reg(META_RF)
            next_address=address+16*(group+1) if group<11 else None
            m.advance(read=next_address,op=('IMETA_GATE',META_RF,(payload,2*group)),tag='consumer_H4_summary_update_and_prefetch')
            if group<11:m.advance(tag='summary_prefetch_response')
        m.wait_reg(META_RF)
        m.advance(write=(meta_base+pixel*8,int(m.rf[META_RF,0]).to_bytes(8,'little')),tag='H4_summary_store')

def source(m,data,axis,label,program,mode):
    start=m.time;before=dict(m.count)
    identity=data[label+'_I24'];height,width=identity.shape[2:];m.sn1_spatial=True
    for y in range(height):
        for x in range(width):
            m.phase='source_I24_input'
            if mode=='producer':m.advance(op=('ILOAD',META_RF,[0]*8),tag='H4_summary_clear')
            m.dma_input(binding.pair.consumer.pack24(identity[:,:,y,x]),90112)
            for h in range(0,96,8):
                m.phase='source_compiled_temporal_program';words=np.zeros(8,np.uint16)
                for ins in program:
                    kind=ins['kind'];m.count['source_ROM128_fetches']+=1
                    if kind=='load':m.load_i24(ins['dst'],90112+(ins['source_t']*96+h)*3)
                    elif kind=='nop':m.advance(tag='source_scheduled_nop')
                    elif kind=='commit':
                        payload=words.astype('<u2').tobytes();address=SOURCE+((y*width+x)*96+h)*2
                        for j in (0,8):m.advance(write=(address+j,payload[j:j+8]),tag='source_gate_word_store')
                        if mode=='producer':producer_summary(m,payload,h)
                    else:
                        for p in ins['operands']:m.wait_reg(p['reg'])
                        dst=95 if kind=='gate' else ins['dst'];m.wait_reg(dst)
                        m.advance(op=('ISOURCE',dst,ins),tag='source_'+kind)
                        if kind=='gate':
                            m.wait_reg(dst);words|=m.rf[dst].astype(np.uint16)<<ins['output_t']
                            m.advance(tag='source_gate_collector_RF_read')
            if mode=='producer':write_summary(m,y*width+x)
    if mode=='consumer':
        scan_summary(m,height*width,SOURCE,META,'consumer_H4_summary_build')
    m.drain()
    words=np.frombuffer(m.state,'<u2',count=height*width*96,offset=SOURCE).reshape(height,width,96).copy()
    gate=np.stack([(words>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    check=common.difference(gate,data[label+'_sn1_gate']);assert check['differences']==0
    if mode!='baseline':
        actual=np.frombuffer(m.state,'<u8',count=height*width,offset=META)
        occupied=(words.reshape(height*width,24,4)&1023).any(2)
        target=sum(occupied[:,g].astype(np.uint64)<<g for g in range(24))
        assert np.array_equal(actual,target)
    return dict(service_slots=m.time-start,checks=check,compiled_instructions=len(program),instruction_ROM_bytes=8192,
        counts={k:v-before.get(k,0) for k,v in m.count.items() if v-before.get(k,0)},
        summary_slots=height*width if mode!='baseline' else 0,summary_bytes=height*width*8 if mode!='baseline' else 0,
        metadata_from_actual_commits=mode=='producer',mode=mode)

def directory(m,words,source_origin,positions,source_base=SOURCE,meta_base=META,dir_base=binding.pair.windows.NRV,phase='preview_metadata_K864_directory'):
    start=m.time;m.phase=phase
    height,width=words.shape[1:];oy,ox=source_origin
    for rem in range(9):
        ky,kx=divmod(rem,3);bases=[0,0];phases=[0,0];valid=0;pixels=[0,0]
        for ip,(y,x) in enumerate(positions):
            m.advance(tag='source_coordinate_bounds_phase');sy,sx=y+ky-1,x+kx-1
            m.advance(tag='source_address_generation')
            if 0<=sy<240 and 0<=sx<320:
                ly,lx=sy-oy,sx-ox;assert 0<=ly<height and 0<=lx<width
                pixels[ip]=ly*width+lx;bases[ip]=source_base+pixels[ip]*192
                phases[ip]=2*(sy%2)+sx%2;valid|=1<<ip
        r=addresses.GEOMETRY+rem;m.wait_reg(r)
        m.advance(op=('ILOAD',r,bases+phases+[0,0,valid,valid]),tag='source_geometry_RF_write')
        for ip in range(len(positions)):
            if valid&(1<<ip):
                m.read_word(meta_base+pixels[ip]*8);m.wait_reg(r)
                m.advance(op=('IMETA_GEOMETRY',r,ip),tag='H4_summary_to_geometry')
    m.drain();m.wait_reg(addresses.NRV)
    m.advance(op=('ILOAD',addresses.NRV,[0]*8),tag='directory_live_init');m.wait_reg(addresses.NRV)
    n=0
    for c0 in range(0,96,4):
        m.advance(tag='H4_channel_counter')
        for rem in range(9):
            r=addresses.GEOMETRY+rem;m.wait_reg(r)
            m.advance(op=('IMETA_PERMISSION',r,c0//4),tag='H4_summary_permissions')
        m.drain()
        for rem in range(9):
            r=addresses.GEOMETRY+rem;permit=int(m.rf[r,7])
            m.advance(tag='H4_offset_word_dispatch')
            if not permit:continue
            # Always clear stale absent-position lanes from the previous group.
            reset=True
            for ip in range(len(positions)):
                if not permit&(1<<ip):continue
                m.read_word(int(m.rf[r,ip])+2*c0)
                target=addresses.CACHE+rem;m.wait_reg(target)
                m.advance(op=('IWORD_PAIR',target,(ip,reset)),tag='SR64_to_H4_cache_decode')
                m.wait_reg(target);reset=False
        for c in range(c0,c0+4):
            for rem in range(9):
                m.advance(tag='K864_scan_select_and_predicate')
                if not int(m.rf[addresses.GEOMETRY+rem,7]):continue
                k=c*9+rem;m.wait_reg(addresses.NRV)
                m.advance(op=('INRV_CACHED',addresses.NRV,(k,addresses.CACHE+rem)),tag='integer_NRV_decode')
                m.wait_reg(addresses.NRV);mask=int(m.rf[addresses.NRV,1])
                if mask:
                    payload=int(m.rf[addresses.NRV,0]).to_bytes(4,'little')+mask.to_bytes(4,'little')
                    m.advance(write=(dir_base+n*8,payload),tag='integer_NRV_write');n+=1
    live=int(m.rf[addresses.NRV,2]);m.mark(m.phase,start)
    records=np.frombuffer(m.state,'<u4',count=n*2,offset=dir_base).reshape(n,2)
    assert n==0 or np.all(records[1:,0]>records[:-1,0])
    return n,sum(int(row[1]).bit_count() for row in records),live

def integer_directory(m,geo,positions):
    shape_only=np.empty((0,*geo['gate_shape']),np.uint8)
    n,_,live=directory(m,shape_only,geo['gate_origin'],positions,
        source_base=binding.pair.consumer.GATE,meta_base=SN2_META,dir_base=binding.pair.consumer.DIR,
        phase='integer_metadata_K864_directory')
    return n,live

def check_sn2_summary(m,geo):
    h,w=geo['gate_shape'];count=h*w
    gate=np.frombuffer(m.state,'<u2',count=count*96,offset=binding.pair.consumer.GATE).reshape(count,24,4)
    target=sum((gate[:,g]&1023).any(1).astype(np.uint64)<<g for g in range(24))
    actual=np.frombuffer(m.state,'<u8',count=count,offset=SN2_META)
    assert np.array_equal(target,actual)

def build_sn2_summary(m,geo):
    scan_summary(m,int(np.prod(geo['gate_shape'])),binding.pair.consumer.GATE,SN2_META,'consumer_sn2_H4_summary_build')

def sn2_with_summary(original):
    import inspect
    code=inspect.getsource(original)
    code=code.replace('    for h in range(0,96,8):',"    for p in range(positions):m.advance(op=('ILOAD',93+p,[0]*8),tag='sn2_H4_summary_clear')\n    for h in range(0,96,8):")
    commit="            for j in (0,8): m.advance(write=(gate_base+(p*96+h)*2+j,payload[j:j+8]),tag='gate_word_store')"
    code=code.replace(commit,commit+"\n            m.wait_reg(93+p);m.advance(op=('IMETA_GATE',93+p,(payload,h//4)),tag='sn2_H4_summary_update');m.wait_reg(93+p)")
    code=code.replace("    m.mark('noncausal_T10_sn2',start)","    for p in range(positions):\n        address=SN2_META+((gate_base-GATE)//192+p)*8\n        m.advance(write=(address,int(m.rf[93+p,0]).to_bytes(8,'little')),tag='H4_summary_store')\n    m.mark('noncausal_T10_sn2',start)")
    namespace=dict(original.__globals__);namespace['SN2_META']=SN2_META
    exec(compile(code,'sn2_H4_summary_binding','exec'),namespace)
    return namespace['execute_sn2']
