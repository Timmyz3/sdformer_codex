"""Natural-K4 nonzero-fill cancellation and complete fixed integer continuation.
Same Machine runs source gate reads -> full U -> RNE/sat -> F/raw merge -> gate
and U24/V96/PED including nonanchor gate and real gate/PED egress. Source PSN,
preview, native projection convolution and global BN remain outside this scope.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import response_execution as response
import probe
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(probe.OPEN/'breadth_20260912/hardware'))
# The legacy resident helper also names its module probe; retain our explicit
# reference and let its own sys.path load that different helper.
sys.modules.pop('probe',None)
import packed_weights as shared
integer,consumer=shared.integer,shared.consumer


class Machine(response.Machine,shared.PackedMachine):
    def integer_op(self,kind,dst,args):
        if kind=='IENC' and self.component_kind=='fillbase':
            operand=self.pop[args]-self.collector[args,self.support[:,0]]
            weight=np.frombuffer(self.value_cache[1],'<i2',count=8,offset=self.component_offset).astype(np.int64)
            value=self.rf[dst].astype(np.int64)+operand*weight
            assert value.min()>=-(1<<47) and value.max()<(1<<47)
            return value.astype(np.float64)
        return super().integer_op(kind,dst,args)


def factor(mode,fill=False):
    p=probe.load(HERE/f'natural_{mode}.npz')
    f=dict(mode=mode,offset=bool(p['offset']),n=p['indices'].shape[-1],values=p['values'].copy(),indices=p['indices'],reconstructed=p['U_conv2_theta_q16'].astype(np.int64),fill=fill)
    if fill:
        assert mode=='shifted14'
        values=f['values'].astype(np.int64)
        values[:,:,1]+=values[:,:,0]
        assert values.min()>=-32768 and values.max()<=32767
        f['values']=values.astype(np.int16)
    return f


def pair_u(m,f,addresses,mbase):
    """Real compressed-coefficient/collector execution, final RF U16 live."""
    m.phase='U_clear'
    for r in range(40):m.wait_reg(r);m.advance(op=('ILOAD',r,np.zeros(8,np.int64)),tag='accumulator_clear')
    live=0
    for group in range(216):
        m.phase='source_response_collector';m.load_collector(addresses[:,group*4:group*4+4])
        live|=sum(int(v!=0)<<t for t,v in enumerate(m.pop))
        if not np.any(m.pop):continue
        for hb in range(2):
            m.phase='compressed_coefficient_decode'
            if f['mode']!='dense':
                bits=f['n']*2;address=mbase+(group*16+hb*8)*bits//8
                stream=int.from_bytes(m.coefficient_bytes(address,bits,'metadata'),'little')
                m.support[:,:f['n']]=np.array([[(stream>>(lane*bits+2*j))&3 for j in range(f['n'])] for lane in range(8)])
                m.advance(tag='metadata_support_accept')
            for slot in range(f['n']+int(f['offset'])):
                m.phase='compressed_coefficient_decode';is_base=f['offset'] and slot==0
                j=slot-int(f['offset'])
                m.component_kind='fillbase' if is_base and f['fill'] else 'base' if is_base else 'dense' if f['mode']=='dense' else 'sparse'
                m.component_slot=max(j,0)
                if m.component_kind=='fillbase':operand=m.pop[:,None]-m.collector[:,m.support[:,0]]
                elif is_base:operand=np.broadcast_to(m.pop[:,None],(20,8))
                elif f['mode']=='dense':operand=np.broadcast_to(m.collector[:,j,None],(20,8))
                else:operand=m.collector[:,m.support[:,j]]
                m.advance(tag='component_mask_decode_and_select')
                if not np.any(operand):continue
                address=((group*2+hb)*(f['n']+int(f['offset']))+slot)*16
                raw=m.coefficient_bytes(address,16,'value');m.component_offset=address%32
                weight=np.frombuffer(raw,'<i2').astype(np.int64)
                # All arms can cancel exact zero weights and skip zero products;
                # fillbase also excludes selected inputs BEFORE its coefficient request.
                active=np.flatnonzero(np.any(operand*weight,axis=1))
                m.phase='U_encoded_IMAC'
                for token in active:
                    dst=int(token)*2+hb;m.wait_reg(dst);m.advance(op=('IENC',dst,int(token)),tag='encoded_accumulate')
    m.drain();m.stream_live=live
    return m.rf[:40].astype(np.int64).reshape(20,16).copy()


def full_run(data,q,label,mode,fill,stress=False):
    f=factor(mode,fill);vb,mb=probe.encode(f);mbase=(len(vb)+31)//32*32
    compressed=vb.ljust(mbase,b'\0')+mb
    # Drop the full U region from the common image. Rebase unchanged successors
    # above compressed U, with actual cold fill and no expanded-U shadow in SRAM.
    bound_q=dict(q,U_conv2_theta_q16=f['reconstructed'].astype(np.int16))
    oldblob,oldbase,bounds=shared.ordinary_coeffs(bound_q,False)
    cut=oldbase['F'];tailbase=(len(compressed)+31)//32*32
    blob=compressed.ljust(tailbase,b'\0')+oldblob[cut:]
    base={k:tailbase+v-cut for k,v in oldbase.items() if k!='U_conv2_theta'}
    base['U_conv2_theta']=0
    m=Machine(stress);m.forward_i24=True
    source=probe.samples(data,label)
    m.phase='source_gate_cold_fill';m.dma_input(source['source_blob'],integer.GATE)
    m.phase='compressed_and_successor_coefficient_fill';m.dma_input(blob,0,True)
    expected=data.copy();geo=json.loads(str(data['window_geometry_json']))[label]
    gold=probe.downstream(source['x']@f['reconstructed'].T,source['raw'],q)
    expected[label+'_updated_I24']=data[label+'_updated_I24'].copy();expected[label+'_proj_gate']=data[label+'_proj_gate'].copy();expected[label+'_continuous_q24']=data[label+'_continuous_q24'].copy()
    oy,ox=geo['output_origin'];gy,gx=geo['gate_origin']
    for iy in range(4):
        for ix in range(4):
            i=iy*4+ix;y,x=2*(oy+iy)-gy,2*(ox+ix)-gx
            expected[label+'_updated_I24'][:,:,y,x]=gold[0][i]
            expected[label+'_proj_gate'][:,:,y,x]=gold[1][i]
            expected[label+'_continuous_q24'][:,:,iy,ix]=gold[2][i]
    old=(consumer.coeffs,consumer.directory,consumer.sparse_u,consumer.dense)
    # consumer.run normally replaces constants; keep the already-paid image and
    # stop that duplicate DMA by a one-use header-only fill shim.
    original_dma=m.dma_input;skip={'next':True}
    def dma(payload,address,coefficient=False):
        if coefficient and skip['next']:
            assert payload==blob and address==0;skip['next']=False;return
        return original_dma(payload,address,coefficient)
    m.dma_input=dma
    def directory(machine,geometry,positions):machine.current_positions=positions;machine.current_geometry=geometry;return 0,None
    checks=[]
    def sparse_u(machine,n,count,b,constants):
        assert count==2
        addresses=np.full((2,864),-1,np.int64)
        h,w=machine.current_geometry['gate_shape'];gy,gx=machine.current_geometry['gate_origin']
        for ip,(y,x) in enumerate(machine.current_positions):
            for c in range(96):
                for ky in range(3):
                    for kx in range(3):
                        sy,sx=y+ky-1,x+kx-1
                        if 0<=sy<240 and 0<=sx<320:addresses[ip,c*9+ky*3+kx]=integer.GATE+2*(((sy-gy)*w+sx-gx)*96+c)
        actual=pair_u(machine,f,addresses,mbase)
        # Only after U execution, independent dense reconstruction checks it.
        indices=[(y//2-oy)*4+(x//2-ox) for y,x in machine.current_positions]
        target=(source['x'][indices]@f['reconstructed'].T).reshape(20,16)
        assert np.array_equal(actual,target);checks.append(dict(positions=machine.current_positions,U_values=actual.size,differences=0))
        for token in range(20):
            for hb in range(2):
                r=token*2+hb;integer.complete(machine,r,int(constants['U_conv2_theta_exponent'])-14)
                machine.store_i24(r,integer.LAT16+(token*16+hb*8)*3)
    def dense(machine,b,name,k_count,h_count,input_base,output_base,positions,shift,live=None,bias=None):
        return shared.ordinary_dense(machine,b,name,k_count,h_count,input_base,output_base,positions,shift,machine.stream_live if name=='F' else live,bias)
    try:
        consumer.coeffs=lambda constants,late:(blob,base,bounds)
        consumer.directory=directory;consumer.sparse_u=sparse_u;consumer.dense=dense
        value,report=consumer.run(expected,q,label,None,machine=m,rank=24)
    finally:consumer.coeffs,consumer.directory,consumer.sparse_u,consumer.dense=old
    assert sum(m.stages.values())==m.time
    return dict(mode=mode,nonzero_fill_override=fill,service_slots=m.time,consumer_report=report,
        stages=dict(m.stages),counts=dict(m.count),coefficient_bytes=len(blob),compressed_U_bytes=len(compressed),
        U_checks=checks,checks=report['checks'],
        quality=probe.metrics((value['updated'][:,:,2*(oy)-gy:2*(oy)-gy+8:2,2*ox-gx:2*ox-gx+8:2].transpose(2,3,0,1).reshape(16,10,96),
                              value['gate'][:,:,2*oy-gy:2*oy-gy+8:2,2*ox-gx:2*ox-gx+8:2].transpose(2,3,0,1).reshape(16,10,96),
                              value['continuous'].transpose(2,3,0,1).reshape(16,10,96)),source['gold']),
        extra_storage_bytes=dict(SR64_staging=64,source_collector=32,value_response=32,metadata_response=32,support=16),
        new_datapath='8 lane-local 3-bit operands select n-e or1bit; existing 16x24 multiply width; no additional RF read. Physical mux area/timing unmeasured.',
        full_native_projection_and_dynamic_BN=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');a=ap.parse_args()
    data=probe.load(probe.CAP/'003_zurich_city_09_a_0004.npz');q=probe.load(probe.CAP/'deployed_constants.npz')
    result=dict(scope=__doc__,frame='003_zurich_city_09_a_0004.npz',window='interior',stress=a.stress,rows=[])
    for mode,fill in [('dense',False),('ordinary24',False),('shifted14',False),('shifted14',True)]:
        row=full_run(data,q,'interior',mode,fill,a.stress);result['rows'].append(row)
        print(mode,fill,row['service_slots'],row['checks'],flush=True)
        (HERE/('fill_chain_stress.json' if a.stress else 'fill_chain_results.json')).write_text(json.dumps(result,indent=2)+'\n')
    result['complete']=True
    (HERE/('fill_chain_stress.json' if a.stress else 'fill_chain_results.json')).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
