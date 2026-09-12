"""Fixed common-prefix, actual A-PED plus complete B-source workload."""
from pathlib import Path
import argparse,copy,importlib.util,json,sys,subprocess
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
BREADTH=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
spec=importlib.util.spec_from_file_location('matched_interleave_prefix',HERE.parent/'matched_local_chain/run.py')
matched=importlib.util.module_from_spec(spec);spec.loader.exec_module(matched)
from cooperative import CooperativeMachine
from flows import source_flow,ped_flow,GATES_B
consumer,common,windows=matched.consumer,matched.common,matched.windows

def dump(path,x):path.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')

def inputs(axis):
    structure='dense' if axis=='dense_twopot' else 'lifting40' if axis=='lifting_twopot' else 'contiguous34'
    folder=(BREADTH/'source_constant_probe'/structure if axis.endswith('twopot') else BREADTH/'algorithm/matched_training'/structure/'stage320')
    q=common.read_npz(folder/'deployed_constants.npz')
    pf=folder/'program.json' if axis.endswith('twopot') else BREADTH/'source_execution'/structure/'program.json'
    program=json.loads(pf.read_text())
    old=common.read_npz(common.FULL/'capture/ordinary/000_zurich_city_09_a_0001.npz')
    live=common.read_npz(common.FULL/'capture/ordinary/live_parameters.npz')
    geo=json.loads(str(old['window_geometry_json']))['corner']
    source=matched.source_gold(old['corner_I24'],q,structure)
    preview,margin=matched.preview_gold(source,live,geo)
    expected,oracle=matched.oracle.independent_gold(old,q,'corner',preview['gate'])
    data=dict(old)
    for key,value in [('sn1_gate',source),('preview_Z_shared',preview['z']),('preview_shared_raw',preview['raw']),('preview_BN1_Y',preview['y']),('sn2_gate',preview['gate']),('updated_I24',expected['updated']),('proj_gate',expected['gate']),('continuous_q24',expected['continuous'])]:data['corner_'+key]=value
    data['interior_sn1_gate']=matched.source_gold(old['interior_I24'],q,structure)
    return structure,q,program,data,live,geo,expected

def prepare_prefix(axis,stress):
    structure,q,program,data,live,geo,expected=inputs(axis)
    matched.install_source(program)
    windows.Machine=CooperativeMachine;windows.build_nrv=matched.stage.preview_directory
    _,producer,m=windows.window(data,live,'corner',False,stress,True,structure)
    assert all(x['differences']==0 for x in producer['checks'].values())
    assert producer['source_program']['checks']['differences']==0
    m.forward_i24=True
    blob,base,bounds=consumer.coeffs(q,False)
    m.phase='A_integer_coefficients';m.dma_input(blob,0,True)
    oy,ox=geo['gate_origin'];out_y,out_x=geo['output_origin']
    positions=[(2*out_y,2*out_x),(2*out_y,2*out_x+2)]
    consumer.feed_identity(m,data,'corner',geo,positions)
    n,live_mask=consumer.directory(m,geo,positions)
    consumer.sparse_u(m,n,2,base,q)
    consumer.dense(m,base,'F',16,96,consumer.LAT16,consumer.UPDATED,2,int(q['F_exponent']),live_mask)
    consumer.merge(m,base,2)
    updated=consumer.read24(m,consumer.UPDATED,(2,10,96))
    target=np.stack([expected['updated'][:,:,y-oy,x-ox] for y,x in positions])
    assert np.array_equal(updated,target)
    consumer.projection_gates(m,base,geo,positions,consumer.UPDATED)
    m.drain();m.ctrace=[]
    h,w=geo['gate_shape']
    words=np.frombuffer(m.state,'<u2',count=h*w*96,offset=consumer.PROJ).reshape(h,w,96)
    observed_gate=np.stack([np.stack([(words[y-oy,x-ox]>>t)&1 for t in range(10)]) for y,x in positions]).astype(bool)
    expected_gate=np.stack([expected['gate'][:,:,y-oy,x-ox] for y,x in positions])
    assert np.array_equal(observed_gate,expected_gate)
    ped_expected=np.stack([expected['continuous'][:,:,y//2-out_y,x//2-out_x] for y,x in positions])
    return m,q,program,data,base,ped_expected,dict(prefix_slots=m.time,source_and_preview=producer['checks'],source=producer['source_program']['checks'],updated_values=updated.size,updated_differences=0,projection_gate_values=observed_gate.size,projection_gate_differences=0,positions=positions,scope='Actual A full source/preview followed by first anchorP2 Conv2/merge/projection gate; other A integer positions not computed.')

def low_program(axis,q):
    if axis=='dense_twopot':return json.loads((BREADTH/'source_constant_probe/dense_low_state/program.json').read_text())
    if axis!='contiguous34':return None
    compiler=BREADTH.parent.parent/'psn/cmvm_20260909/.venv/bin/python'
    subprocess.run([str(compiler),str(HERE/'compile_low34.py')],check=True)
    program=json.loads((HERE/'contiguous34_low_program.json').read_text())
    assert len(program)<=512
    return program

def registers(program):return max(i.get('dst',-1) for i in program)+1

def configurations(full,low):
    modes=[dict(name='serial_CSE_P2',program=full,offset=0,style='original',hblock=32,joint=False),dict(name='serial_CSE_P1_Z',program=full,offset=0,style='P1',hblock=48,joint=False)]
    for label,p in [('CSE',full),('low_CSD',low)]:
        if p is None:continue
        peak=registers(p);groups=min(4,(95-peak-2)//20)
        assert groups>=1
        for joint in (False,True):
            modes.append(dict(name=('joint_' if joint else 'serial_same_')+label,program=p,offset=20*groups,style='two_RF',hblock=8*groups,joint=joint))
    return modes

def run_case(axis,prefix,q,full,data,base,ped_expected,prefix_report,mode):
    m=copy.deepcopy(prefix);m.observed_egress=bytearray();start=m.time;old_counts=Counter(m.count)
    p=mode['program'];offset=mode['offset'];peak=registers(p)
    rom_words=len(full)+(len(p) if offset or p is not full else 0)
    assert rom_words<=512,(axis,mode['name'],rom_words)
    source=lambda:source_flow(m,data['interior_I24'],p,offset)
    ped=lambda:ped_flow(m,base,q,mode['style'],mode['hblock'],consumer.UPDATED,consumer.PED_U,consumer.PED_V)
    if mode['joint']:
        scheduling=m.execute_flows(source,ped,range(offset,offset+peak),range(offset))
    else:
        source();source_end=m.time;ped();scheduling=dict(start=start,end=m.time,source_end=source_end)
    # Byte-for-byte public boundary checks, not execution-time oracle operands.
    h,w=data['interior_I24'].shape[2:]
    words=np.frombuffer(m.state,'<u2',count=h*w*96,offset=GATES_B).reshape(h,w,96)
    gate=np.stack([(words>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    actual=consumer.read24(m,consumer.PED_V,(2,10,96))
    checks=dict(B_source_gate=common.difference(gate,data['interior_sn1_gate']),A_PED=common.difference(actual,ped_expected))
    expected_bytes=consumer.pack24(ped_expected)
    assert bytes(m.observed_egress)==expected_bytes
    checks['A_PED_actual_DMA_bytes']=dict(values=len(expected_bytes),differences=0)
    assert all(c['differences']==0 for c in checks.values()),checks
    assert sum(m.stages.values())==m.time
    return dict(axis=axis,mode=mode['name'],stress=m.stress,service_slots=m.time,tail_slots=m.time-start,common_prefix=prefix_report,
        checks=checks,scheduling=scheduling,counts=dict(m.count),tail_counts=dict(Counter(m.count)-old_counts),stages=dict(m.stages),
        source_program_words=len(p),combined_source_ROM_words=rom_words,source_work_RF=peak,source_offset=offset,PED_hblock=mode['hblock'],
        SRAM_high_water=GATES_B+h*w*192,resource=dict(RF=[96,8,48],SR64=True,SW64=True,CR256=True,state_bytes=131072,coefficient_bytes=131072,source_ROM_words=512,shared_issue=1,shared_pending_table=True),
        scope='One real A first-anchorP2 fullPED plus B complete source halo, actual A producer prefix. Not full two-window inference or a network speed.',
        new_RTL=False,new_PPA=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('axis',choices=['dense_twopot','lifting_twopot','contiguous34']);ap.add_argument('--mode');ap.add_argument('--stress',action='store_true');a=ap.parse_args()
    print('PREFIX_START',a.axis,a.stress,flush=True)
    prefix,q,full,data,base,gold,pr=prepare_prefix(a.axis,a.stress)
    print('PREFIX_DONE',pr['prefix_slots'],flush=True)
    low=low_program(a.axis,q)
    for mode in configurations(full,low):
        if a.mode and mode['name']!=a.mode:continue
        # Gate RF95 and scalar source RF93/94 are assigned explicitly.
        if mode['joint']:
            original=CooperativeMachine.execute_flows
            def assigned(self,source,ped,sregs,cregs):
                return original(self,source,ped,list(sregs)+[95],list(cregs)+[93,94])
            CooperativeMachine.execute_flows=assigned
        try:r=run_case(a.axis,prefix,q,full,data,base,gold,pr,mode)
        finally:
            if mode['joint']:CooperativeMachine.execute_flows=original
        name=a.axis+'_'+mode['name']+('_stress' if a.stress else '')
        dump(HERE/(name+'.json'),r)
        print('CASE_DONE',name,r['service_slots'],r['tail_slots'],flush=True)

if __name__=='__main__':main()
