"""Real I24 -> source34 gates on the unchanged finite integrated Machine.

Only the producer stage is exercised. No old sn2/consumer gold or downstream
service is reused for the changed source function.
"""
from pathlib import Path
import sys
import json
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import integrated
from integer_chain import pack24
from run_chain import SRC,difference

INPUT = 90112


def run(label,stress=False):
    fixture = dict(np.load(HERE/'fixture.npz'))
    identity = fixture[label+'_I24']
    _,_,height,width = identity.shape
    program = json.loads((HERE/'source34_program.json').read_text())
    m = integrated.IntegratedMachine(stress)
    m.sn1_spatial = True
    for y in range(height):
        for x in range(width):
            m.phase='source_I24_input';m.dma_input(pack24(identity[:,:,y,x]),INPUT)
            for h in range(0,96,8):
                m.phase='source_compiled_temporal_program';words=np.zeros(8,np.uint16)
                for ins in program:
                    kind=ins['kind'];m.count['source_ROM128_fetches']+=1
                    if kind=='load':m.load_i24(ins['dst'],INPUT+(ins['source_t']*96+h)*3)
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
    gates=np.stack([(actual>>t)&1 for t in range(10)]).transpose(0,3,1,2).astype(bool)
    checks=difference(gates,fixture[label+'_gate'])
    assert checks['differences']==0 and sum(m.stages.values())==m.time
    return dict(window=label,stress=stress,service_slots=m.time,counts=dict(m.count),
        stages=dict(m.stages),checks=checks,source_shape=list(identity.shape),
        physical_port_bytes=dict(state_read=8*m.count['SR64_reads'],state_write=8*m.count['SW64_writes'],
            coefficient_read=32*m.count['CR256_reads'],coefficient_fill=32*m.count['CW256_writes']),
        program_instructions=len(program),instruction_ROM_bytes=8192,
        state_highest_touched_byte=max(INPUT+2880,SRC+height*width*96*2),
        source_commit_trace=m.source_commit_trace,
        scope='Actual I24 halo cold DMA -> complete T10 source gate SRAM; no preview/sn2/PED consumers.',
        evidence='CPU payload slot prototype with actual RF/SRAM arithmetic; not RTL or PPA.')


def export_rtl():
    out=HERE/'rtl_inputs';out.mkdir(exist_ok=True)
    fixture=dict(np.load(HERE/'fixture.npz'))
    program=json.loads((HERE/'source34_program.json').read_text())
    (out/'source34_program.json').write_text(json.dumps(program,indent=2)+'\n')
    manifest=dict(input_layout='tile=(y*width+x)*12+hgroup; [tile,T10,lanes8], little-endian int32 holding signed24',
        output_layout='[tile,lanes8], little-endian uint16, low10 gate bits',
        arithmetic='same original source ISA, same RNE15/saturate24 composed into cutoff',cases=[])
    for label in ('corner','interior'):
        x=fixture[label+'_I24'];_,_,height,width=x.shape
        tiles=x.transpose(2,3,1,0).reshape(height,width,12,8,10).transpose(0,1,2,4,3).reshape(-1,10,8)
        gate=fixture[label+'_gate'];words=sum(gate[t].astype(np.uint16)<<t for t in range(10))
        gold=words.transpose(1,2,0).reshape(-1,8)
        prefix='source34_'+label
        tiles.astype('<i4').tofile(out/(prefix+'_inputs_i24.bin'))
        gold.astype('<u2').tofile(out/(prefix+'_gates_u16.bin'))
        manifest['cases'].append(dict(axis='source34',window=label,height=height,width=width,tiles=len(tiles),
            program_file='source34_program.json',input_file=prefix+'_inputs_i24.bin',gate_gold_file=prefix+'_gates_u16.bin',
            source_values=int(tiles.size),gate_bits=int(gold.size*10),instruction_count=len(program),
            operation_counts={kind:sum(i['kind']==kind for i in program) for kind in sorted({i['kind'] for i in program})}))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    export_rtl()
    result=dict(resource=dict(RF_vectors=96,lanes=8,bits_per_lane=48,state_bytes=131072,
        coefficient_bytes=131072,instruction_ROM_bytes=8192,state_ports='1R64 / 1W64',
        coefficient_port='1R256',DMA='32B / 5 slots'),rows=[])
    for label,stress in [('corner',False),('interior',False),('interior',True)]:
        row=run(label,stress);result['rows'].append(row)
        (HERE/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
        print(label,stress,row['service_slots'],row['checks'],flush=True)
    print('SOURCE34_COMPLETE',flush=True)


if __name__=='__main__':main()
