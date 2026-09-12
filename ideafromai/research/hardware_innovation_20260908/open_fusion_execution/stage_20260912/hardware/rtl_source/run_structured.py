"""Replay the ordinary 34-support control on the existing common source RTL."""
from pathlib import Path
import json
import subprocess

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'structured_source_control/rtl_inputs'
OUT=HERE/'generated'
OUT.mkdir(exist_ok=True)
kinds={'nop':0,'load':1,'addsub':2,'norm24':3,'gate':4,'commit':5}
program=json.loads((INPUT/'source34_program.json').read_text())
lines=[]
for i in program:
    ops=i.get('operands',[])+[dict(reg=0,shift=0,sign=1)]*2
    values=[kinds[i['kind']],i.get('dst',95 if i['kind']=='gate' else 0),
        ops[0]['reg'],ops[1]['reg'],ops[0]['shift'],ops[1]['shift'],
        int(ops[0]['sign']<0),int(ops[1]['sign']<0),i.get('rne_shift',0),
        i.get('output_t',i.get('source_t',0)),i.get('threshold',0),
        int(i.get('direction',1)<0),i.get('constant',-1)+1]
    lines.append(' '.join(map(str,values)))
fields=OUT/'source34.txt';fields.write_text('\n'.join(lines)+'\n')
rows=[]
for case in json.loads((INPUT/'manifest.json').read_text())['cases']:
    for mode in ('ready','stress'):
        row=json.loads(subprocess.run([str(HERE/'obj_dir/Vtemporal_source'),str(fields),
            str(INPUT/case['input_file']),str(INPUT/case['gate_gold_file']),mode],
            check=True,capture_output=True,text=True).stdout)
        row.update(axis='source34',window=case['window']);rows.append(row)
        print(json.dumps(row),flush=True)
(HERE/'structured_control_results.json').write_text(json.dumps(dict(
    scope=__doc__,rows=rows,hardware_unchanged=True,source_only=True,
    gate_gold='CPU new34-support matrix, not original dense gate. Actual GPU helper checks are reported separately under algorithm/source34.',
    AEE_scope='See algorithm/source34. No inherited old row34 quality.',
    caveat='Compared programs compute different gate functions. AEE and downstream work must be reported separately. Not a full-chain speedup or PPA.'),indent=2)+'\n')
