from pathlib import Path
import json
import numpy as np
from integrated import common,HERE

OUT=HERE/'source_rtl_inputs'
OUT.mkdir(exist_ok=True)
PROGRAM=common.FULL.parent/'two_stage_writeback'
manifest=dict(input_layout='tile=(y*width+x)*12+hgroup; [tile, T10, lanes8], little-endian signed int32 container of original signed24 values',
    output_layout='[tile,lanes8], little-endian uint16, low10 bits are gates for t0..t9',
    instruction_arithmetic='Original fused_program JSON; no new quantization, thresholds or operation ordering.',
    machine_semantics='ISOURCE latency2, one issue and one RF writeback per slot; source arithmetic signed48; norm24 performs floor quotient + ties-even increment then signed24 saturation; gate collects RF95 after writeback; commit performs two SW64 stores.',
    cases=[])
for axis in ['ordinary','lifting_raw']:
    program=json.loads((PROGRAM/f'{axis}_fused_program.json').read_text())
    (OUT/f'{axis}_program.json').write_text(json.dumps(program,indent=2)+'\n')
    data=common.read_npz(common.FULL/'capture'/axis/'000_zurich_city_09_a_0001.npz')
    for label in ['corner','interior']:
        x=data[label+'_I24'].astype(np.int32)
        _,_,h,w=x.shape
        assert np.all(x>=-(1<<23)) and np.all(x<(1<<23))
        tiles=x.transpose(2,3,1,0).reshape(h,w,12,8,10).transpose(0,1,2,4,3).reshape(-1,10,8)
        gates=data[label+'_sn1_gate']
        words=sum(gates[t].astype(np.uint16)<<t for t in range(10))
        gold=words.transpose(1,2,0).reshape(-1,8)
        prefix=f'{axis}_{label}'
        tiles.astype('<i4').tofile(OUT/f'{prefix}_inputs_i24.bin')
        gold.astype('<u2').tofile(OUT/f'{prefix}_gates_u16.bin')
        manifest['cases'].append(dict(axis=axis,window=label,height=h,width=w,tiles=len(tiles),
            input_file=f'{prefix}_inputs_i24.bin',gate_gold_file=f'{prefix}_gates_u16.bin',program_file=f'{axis}_program.json',
            source_values=int(tiles.size),gate_bits=int(gold.size*10),instruction_count=len(program),
            operation_counts={kind:sum(i['kind']==kind for i in program) for kind in sorted({i['kind'] for i in program})}))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
