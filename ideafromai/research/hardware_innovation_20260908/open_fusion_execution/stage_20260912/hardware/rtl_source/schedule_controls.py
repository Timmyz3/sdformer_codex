"""Same arithmetic DAG, fixed output-order DFS controls for register lifetime.

No learned parameter, CSE node, shift, threshold or RNE is changed. This tests
whether the original high RF footprint was a scheduling artifact before using
the footprint as a new source/consumer-overlap argument.
"""
from collections import Counter
from pathlib import Path
import json
import subprocess
import prepare

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'source_rtl_inputs'
OUT=HERE/'generated'

def compile_order(program,order):
    nodes={i['logical_node']:i for i in program if 'logical_node' in i}
    roots=[i for i in program if i['kind']=='gate']
    if order=='time_order':roots=sorted(roots,key=lambda i:i['output_t'])
    refs=Counter(o['node'] for i in nodes.values() for o in i.get('operands',[]))
    scheduled=[];seen=set();register={};free=set(range(95));peak=0
    def visit(node):
        nonlocal peak
        if node in seen:return
        ins=nodes[node]
        for operand in ins.get('operands',[]):visit(operand['node'])
        copied={k:v for k,v in ins.items() if k not in ('operands','dst','issue_slot')}
        operands=[]
        for operand in ins.get('operands',[]):
            operands.append(dict(operand,reg=register[operand['node']]))
        if operands:copied['operands']=operands
        for operand in ins.get('operands',[]):
            n=operand['node'];refs[n]-=1
            if refs[n]==0:free.add(register.pop(n))
        if ins['kind']!='gate':
            r=min(free);free.remove(r);register[node]=r;copied['dst']=r
            peak=max(peak,95-len(free))
        copied['issue_slot']=len(scheduled);scheduled.append(copied);seen.add(node)
    for root in roots:visit(root['logical_node'])
    assert seen==set(nodes), 'no unused arithmetic silently removed'
    scheduled.append(dict(kind='nop',issue_slot=len(scheduled)))
    scheduled.append(dict(kind='commit',issue_slot=len(scheduled)))
    assert Counter(i['kind'] for i in scheduled)==Counter(i['kind'] for i in program)
    return scheduled,peak

def write_fields(program,path):
    lines=[]
    for i in program:
        ops=i.get('operands',[])+[dict(reg=0,shift=0,sign=1)]*2
        values=[prepare.KINDS[i['kind']],i.get('dst',95 if i['kind']=='gate' else 0),
                ops[0]['reg'],ops[1]['reg'],ops[0]['shift'],ops[1]['shift'],
                int(ops[0]['sign']<0),int(ops[1]['sign']<0),i.get('rne_shift',0),
                i.get('output_t',i.get('source_t',0)),i.get('threshold',0),
                int(i.get('direction',1)<0),i.get('constant',-1)+1]
        lines.append(' '.join(map(str,values)))
    path.write_text('\n'.join(lines)+'\n')

rows=[]
cases=json.loads((INPUT/'manifest.json').read_text())['cases']
for axis in ('ordinary','lifting_raw'):
    original=json.loads((INPUT/f'{axis}_program.json').read_text())
    for order in ('original_gate_order','time_order'):
        program,peak=compile_order(original,order)
        stem=axis+'_'+order
        fields=OUT/(stem+'.txt');write_fields(program,fields)
        (OUT/(stem+'.json')).write_text(json.dumps(program,indent=2)+'\n')
        for case in (c for c in cases if c['axis']==axis):
            for mode in ('ready','stress'):
                measured=json.loads(subprocess.run([str(HERE/'obj_dir/Vtemporal_source'),str(fields),
                    str(INPUT/case['input_file']),str(INPUT/case['gate_gold_file']),mode],
                    check=True,capture_output=True,text=True).stdout)
                measured.update(axis=axis,window=case['window'],order=order,peak_live_general_RF=peak,
                    maximum_general_RF=max(i.get('dst',0) for i in program),instruction_count=len(program))
                rows.append(measured);print(json.dumps(measured),flush=True)
(HERE/'schedule_controls.json').write_text(json.dumps(dict(
    scope=__doc__,rows=rows,same_DAG=True,same_RNE_and_parameters=True,
    hardware_unchanged=True,register_capacity=96,configuration_cost_excluded=True,
    actual_recomputation=False,actual_source_consumer_overlap=False),indent=2)+'\n')
