"""Prepare original program fields and observed boundary wait traces for RTL."""
from pathlib import Path
import json
import struct

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'source_rtl_inputs'
OUT=HERE/'generated'
OUT.mkdir(exist_ok=True)
KINDS={'nop':0,'load':1,'addsub':2,'norm24':3,'gate':4,'commit':5}
for axis in ('ordinary','lifting_raw'):
    program=json.loads((INPUT/f'{axis}_program.json').read_text())
    lines=[]
    for ins in program:
        ops=ins.get('operands',[])+[dict(reg=0,shift=0,sign=1)]*2
        values=[KINDS[ins['kind']],ins.get('dst',95 if ins['kind']=='gate' else 0),
                ops[0]['reg'],ops[1]['reg'],ops[0]['shift'],ops[1]['shift'],
                int(ops[0]['sign']<0),int(ops[1]['sign']<0),ins.get('rne_shift',0),
                ins.get('output_t',ins.get('source_t',0)),ins.get('threshold',0),
                int(ins.get('direction',1)<0),ins.get('constant',-1)+1]
        lines.append(' '.join(map(str,values)))
    (OUT/f'{axis}.txt').write_text('\n'.join(lines)+'\n')
for path in sorted(INPUT.glob('*sink_trace.json')):
    trace=json.loads(path.read_text())
    waits=[r['wait_slots'] for r in trace['source_commit_SW64']]
    (OUT/(path.stem+'.waits')).write_text('\n'.join(map(str,waits))+'\n')
# A single directed tile checks negative ties, even rounding, both saturations,
# both threshold directions, and the two constant-gate encodings.
directed=[
    [1,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,1,0,0,0],
    [2,2,0,1,13,0,0,0,0,0,0,0,0],
    [3,3,2,0,0,0,0,0,12,0,0,0,0],
]
thresholds=[(0,0,0),(0,1,0),(-2,1,0),(2,0,0),(0,0,2),(0,0,1),(8388607,0,0),(-8388608,1,0)]
for t,(th,negative,constant) in enumerate(thresholds):
    directed.append([4,95,3,0,0,0,0,0,0,t,th,negative,constant])
directed.append([5,0,0,0,0,0,0,0,0,0,0,0,0])
(OUT/'directed.txt').write_text('\n'.join(' '.join(map(str,row)) for row in directed)+'\n')
x=[8388607,-8388608,1,-1,3,-3,0,0]+[0,0,2048,2048,2048,2048,-2048,2048]+[0]*64
y=[8388607,-8388608,2,-2,6,-6,0,0]
gold=[]
for v in y:
    bits=[v>=0,v<=0,v<=-2,v>=2,True,False,v>=8388607,v<=-8388608]
    gold.append(sum(int(b)<<t for t,b in enumerate(bits)))
(OUT/'directed_i24.bin').write_bytes(struct.pack('<80i',*x))
(OUT/'directed_gate.bin').write_bytes(struct.pack('<8H',*gold))
print(json.dumps({'programs':2,'wait_traces':len(list(OUT.glob('*.waits')))}))
