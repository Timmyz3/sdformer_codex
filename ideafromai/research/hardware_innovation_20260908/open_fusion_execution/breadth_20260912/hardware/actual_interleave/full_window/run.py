"""Actual full A local consumer and B complete source, one shared Machine."""
from pathlib import Path
import argparse,copy,json,sys,types
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run as pair
from flows import source_flow,ped_flow,GATES_B
consumer=pair.consumer
ANCHORS=32768
SN2=98304
PROJECTION=79872
PED_U=81920
PED_V=83968

def relocated_projection(m,base,geo,positions,input_base):
    original=consumer.projection_gates
    namespace=dict(original.__globals__);namespace['PROJ']=PROJECTION
    types.FunctionType(original.__code__,namespace)(m,base,geo,positions,input_base)

def copy_state(m,src,dst,size,tag):
    # One read response feeds one write. No host tensor feeds execution.
    m.phase=tag
    for off in range(0,size,8):
        payload=m.read_word(src+off)
        m.advance(write=(dst+off,payload),tag=tag+'_SW64')

def egress(m,address,size,tag):
    m.phase=tag;result=bytearray()
    for off in range(0,size,32):
        m.acquire('DMA32')
        payload=b''.join(m.read_word(address+off+j) for j in (0,8,16,24))
        for _ in range(5):m.advance(tag=tag+'_DMA_slots')
        result.extend(payload) # External observer, never used as an operand.
        m.release('DMA32')
    return bytes(result)

def producer(axis,stress):
    structure,q,program,data,live,geo,expected=pair.inputs(axis)
    pair.matched.install_source(program)
    pair.windows.Machine=pair.CooperativeMachine
    pair.windows.build_nrv=pair.matched.stage.preview_directory
    _,report,m=pair.windows.window(data,live,'corner',False,stress,True,structure)
    assert all(x['differences']==0 for x in report['checks'].values())
    assert report['source_program']['checks']['differences']==0
    m.forward_i24=True
    blob,base,_=consumer.coeffs(q,False)
    m.phase='A_integer_coefficients';m.dma_input(blob,0,True)
    return m,q,program,data,base,geo,expected,report

def prepare_integer(m,q,data,base,geo,expected,style,hblock,batch):
    oy,ox=geo['gate_origin'];out_y,out_x=geo['output_origin'];h,w=geo['gate_shape']
    anchor_order=[(2*(out_y+y),2*(out_x+x)) for y in range(4) for x in range(4)]
    anchor_index={position:i for i,position in enumerate(anchor_order)}
    if batch:copy_state(m,consumer.GATE,SN2,h*w*192,'A_sn2_relocation')
    gate_base=SN2 if batch else consumer.GATE
    m.observed_egress=bytearray();updated_checks=0;ped_pairs=0
    for y in range(oy,oy+h):
        for anchor in (True,False):
            xs=[x for x in range(ox,ox+w) if ((y,x) in anchor_index)==anchor]
            for i in range(0,len(xs),2):
                positions=[(y,x) for x in xs[i:i+2]];count=len(positions)
                consumer.feed_identity(m,data,'corner',geo,positions)
                if anchor:
                    assert count==2
                    n,live=consumer.directory(m,geo,positions,gate_base=gate_base)
                    consumer.sparse_u(m,n,count,base,q)
                    consumer.dense(m,base,'F',16,96,consumer.LAT16,consumer.UPDATED,count,int(q['F_exponent']),live)
                    consumer.merge(m,base,count)
                address=consumer.UPDATED if anchor else consumer.I
                actual=consumer.read24(m,address,(count,10,96))
                gold=np.stack([expected['updated'][:,:,yy-oy,xx-ox] for yy,xx in positions])
                assert np.array_equal(actual,gold)
                updated_checks+=actual.size
                relocated_projection(m,base,geo,positions,address)
                if anchor:
                    if batch:
                        first=anchor_index[positions[0]]
                        assert anchor_index[positions[1]]==first+1
                        copy_state(m,consumer.UPDATED,ANCHORS+first*2880,5760,'A_anchor_preserve')
                    else:
                        ped_flow(m,base,q,style,hblock,consumer.UPDATED,consumer.PED_U,consumer.PED_V)
                        ped_pairs+=1
    m.drain()
    words=np.frombuffer(m.state,'<u2',count=h*w*96,offset=PROJECTION).reshape(h,w,96).copy()
    gate=np.stack([(words>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    assert np.array_equal(gate,expected['gate'])
    projection_bytes=egress(m,PROJECTION,h*w*192,'A_projection_egress')
    assert projection_bytes==words.astype('<u2').tobytes()
    expected_ped=np.stack([expected['continuous'][:,:,yy//2-out_y,xx//2-out_x] for yy,xx in anchor_order])
    if batch:
        saved=consumer.read24(m,ANCHORS,(16,10,96))
        target=np.stack([expected['updated'][:,:,yy-oy,xx-ox] for yy,xx in anchor_order])
        assert np.array_equal(saved,target)
    else:
        assert bytes(m.observed_egress)==consumer.pack24(expected_ped)
    return expected_ped,dict(updated_values=updated_checks,updated_differences=0,
        projection_values=gate.size,projection_differences=0,projection_actual_DMA_bytes=len(projection_bytes),
        saved_anchor_bytes=46080 if batch else 0,direct_PED_pairs=ped_pairs)

def execute(axis,stress,producer_bundle,mode):
    template,q,full,data,base,geo,expected,producer_report=producer_bundle
    m=copy.deepcopy(template);start=m.time;before=Counter(m.count)
    batch=mode['batch'];joint=mode['joint']
    expected_ped,checks=prepare_integer(m,q,data,base,geo,expected,mode['style'],mode['hblock'],batch)
    boundary=m.time
    p=mode['program'];offset=mode['offset'];peak=pair.registers(p)
    source=lambda:source_flow(m,data['interior_I24'],p,offset)
    def all_ped():
        for pos in range(0,16,2):
            ped_flow(m,base,q,mode['style'],mode['hblock'],ANCHORS+pos*2880,PED_U,PED_V)
    if joint:
        schedule=m.execute_flows(source,all_ped,list(range(offset,offset+peak))+[95],list(range(offset))+[93,94])
    else:
        source();source_end=m.time
        if batch:all_ped()
        schedule=dict(start=boundary,end=m.time,source_end=source_end)
    assert bytes(m.observed_egress)==consumer.pack24(expected_ped)
    bh,bw=data['interior_I24'].shape[2:]
    words=np.frombuffer(m.state,'<u2',count=bh*bw*96,offset=GATES_B).reshape(bh,bw,96)
    gate=np.stack([(words>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    assert np.array_equal(gate,data['interior_sn1_gate'])
    checks.update(PED_values=expected_ped.size,PED_differences=0,PED_actual_DMA_bytes=len(m.observed_egress),
        B_gate_values=gate.size,B_gate_differences=0)
    assert sum(m.stages.values())==m.time
    rom=len(full)+(len(p) if offset else 0)
    assert rom<=512
    return dict(axis=axis,mode=mode['name'],stress=stress,service_slots=m.time,
        A_source_preview_integer_coeff_prefix=start,A_integer_gate_end=boundary,post_integer_slots=m.time-boundary,
        stages=dict(m.stages),counts=dict(m.count),post_producer_counts=dict(Counter(m.count)-before),
        checks=checks,producer_checks=producer_report['checks'],source_checks=producer_report['source_program']['checks'],
        scheduling=schedule,source_work_RF=peak,source_offset=offset,PED_hblock=mode['hblock'],
        combined_source_ROM_words=rom,state_high_water=121536,state_capacity=131072,coefficient_capacity=131072,
        RF=[96,8,48],single_issue=True,ports=['SR64','SW64','CR256'],batch_anchor_bytes=46080 if batch else 0,
        scope='Complete A corner local source/preview/8x8 updated/projection and4x4 PED actual egress, plus B full11x11 source. No B preview/native/globalBN/full layer/network.',
        new_RTL=False,new_PPA=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('axis',choices=['dense_twopot','lifting_twopot','contiguous34']);ap.add_argument('--stress',action='store_true');ap.add_argument('--mode');args=ap.parse_args()
    print('PRODUCER',args.axis,args.stress,flush=True)
    bundle=producer(args.axis,args.stress);full=bundle[2]
    candidates=pair.configurations(full,None)
    modes=[]
    for orig in candidates:
        if orig['name'].startswith('serial_CSE_'):
            m=dict(orig);m['batch']=False;m['name']=m['name'].replace('serial_','direct_');modes.append(m)
        else:
            m=dict(orig);m['batch']=True;m['name']=m['name'].replace('serial_same_','batch_serial_').replace('joint_','batch_joint_');modes.append(m)
    for mode in modes:
        if args.mode and mode['name'] not in args.mode.split(','):continue
        r=execute(args.axis,args.stress,bundle,mode)
        name=args.axis+'_'+mode['name']+('_stress' if args.stress else '')
        pair.dump(HERE/(name+'.json'),r);print('COMPLETE',name,r['service_slots'],r['checks'],flush=True)

if __name__=='__main__':main()
