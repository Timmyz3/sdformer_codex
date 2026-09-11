"""Run existing complete compiled source programs on the common SRAM port.

Both axes inherit the same fused two-stage signed48 source ALU, register
allocator and512x128 instruction ROM. Real pixel-major packed I24 is loaded
through the same32B/5slot DMA and1R64/1W64 state port used by all followers.
No previously measured eight-bank source cycle count is imported.
"""
import json
import numpy as np
from run_chain import HERE,SRC,difference
from integer_chain import pack24

INPUT=90112


def run(m,data,axis,label):
    start=m.time;counts=dict(m.count);stages=dict(m.stages)
    program=json.loads((HERE.parent.parent/'two_stage_writeback'/f'{axis}_fused_program.json').read_text())
    assert len(program)<=512 and max(i.get('dst',0) for i in program)<95
    # Both have a fixed ROM allocation, not an uncharged runtime instruction
    # stream fetched from host memory. Fetch is combinational and broadcast.
    identity=data[label+'_I24'];_,_,height,width=identity.shape
    m.sn1_spatial=True
    for y in range(height):
        for x in range(width):
            m.phase='source_I24_input';m.dma_input(pack24(identity[:,:,y,x]),INPUT)
            for h in range(0,96,8):
                m.phase='source_compiled_temporal_program';words=np.zeros(8,np.uint16)
                for ins in program:
                    kind=ins['kind'];m.count['source_ROM128_fetches']+=1
                    if kind=='load':
                        m.load_i24(ins['dst'],INPUT+(ins['source_t']*96+h)*3)
                    elif kind=='nop':m.advance(tag='source_scheduled_nop')
                    elif kind=='commit':
                        payload=words.astype('<u2').tobytes();address=SRC+((y*width+x)*96+h)*2
                        for j in (0,8):m.advance(write=(address+j,payload[j:j+8]),tag='source_gate_word_store')
                    else:
                        for p in ins['operands']:m.wait_reg(p['reg'])
                        dst=95 if kind=='gate' else ins['dst'];m.wait_reg(dst)
                        m.advance(op=('ISOURCE',dst,ins),tag='source_'+kind)
                        if kind=='gate':
                            m.wait_reg(dst)
                            words|=m.rf[dst].astype(np.uint16)<<ins['output_t']
                            m.advance(tag='source_gate_collector_RF_read')
    actual=np.frombuffer(m.state,'<u2',count=height*width*96,offset=SRC).reshape(height,width,96).copy()
    gate=np.stack([(actual>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    checks=difference(gate,data[label+'_sn1_gate']);assert checks['differences']==0,checks
    report=dict(service_slots=m.time-start,counts={k:v-counts.get(k,0) for k,v in m.count.items() if v-counts.get(k,0)},
        stages={k:v-stages.get(k,0) for k,v in m.stages.items() if v-stages.get(k,0)},checks=checks,
        compiled_instructions=len(program),instruction_ROM_bytes=8192,source_halo_shape=list(identity.shape),
        interface='Spatial/T/C packed24 input; spatial/C packedT10 gate output. Original I24 remains external and is reread later; no full-window I24 retention claimed.',
        cross_operator_strip_fusion=False)
    return report
