"""One fixed destination-controller change, measured by Verilator."""
from pathlib import Path
import json
import subprocess

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'native_sparse'
obj=HERE/'obj_dir'
with (HERE/'build.log').open('w') as log:
    subprocess.run(['verilator','--cc','--exe','-Wall','--top-module','native_sparse',
                    '--Mdir',str(obj),str(HERE/'native_sparse.sv'),str(HERE/'tb.cpp')],
                   check=True,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run(['make','-C',str(obj),'-f','Vnative_sparse.mk','-j2'],
                   check=True,stdout=log,stderr=subprocess.STDOUT)
control=json.loads((BASE/'control_results.json').read_text())
real=json.loads((BASE/'real_results.json').read_text())
fixtures=sorted({r['fixture'] for r in control+real})
rows=[]
for fixture in fixtures:
    for mode in (1,3):
        for stall in (0,1):
            proc=subprocess.run([str(obj/'Vnative_sparse'),str(BASE/'fixtures'/fixture),str(mode),str(stall)],
                                check=True,text=True,capture_output=True)
            for line in proc.stdout.splitlines():
                row=json.loads(line);row['fixture']=fixture;rows.append(row)
    print('RTL_PASS',fixture,flush=True)
(HERE/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={}
for arm in ('dense','physical25','magnitude25'):
    summary[arm]={}
    for mode in (1,3):
        summary[arm][str(mode)]={}
        for stall in (0,1):
            rs=[r for r in rows if r['fixture'].startswith(f'real_{arm}_tile') and r['mode']==mode and r['stall']==stall and r['command']==0]
            fields=('cycles','source_words','weight_words','psum_reads','psum_writes','sum_issues','update_issues','source_stalls','weight_stalls','output_stalls')
            s={field:sum(r[field] for r in rs) for field in fields}
            s['tiles']=len(rs)
            s['cycles_with_fresh_source_origin']=s['cycles']+1537*len(rs)
            summary[arm][str(mode)][str(stall)]=s
# Both controllers must have identical requested work and final values.
for fixture in fixtures:
    group=[r for r in rows if r['fixture']==fixture]
    for field in ('source_words','weight_words','psum_reads','psum_writes','sum_issues','update_issues','outputs'):
        assert len({r[field] for r in group})==1,(fixture,field)
old={(r['fixture'],r['mode'],r['stall'],r['command']):r for r in control+real}
for r in rows:
    if r['mode']==1:
        reference=old[(r['fixture'],1,r['stall'],r['command'])]
        assert r['cycles']==reference['cycles'],(r,reference)
result=dict(runs=len(rows),checked_outputs=sum(r['outputs'] for r in rows),
            identical_requested_work=True,unchanged_mode1_cycles=True,summary=summary,
            additional_control='48bit pending destination register, 48way priority/decode and bounded 12x4 predicate; same module budget for modes1/3, not area/Fmax equivalence',
            scope='Same full C96/N96/T10 linear leaf; same calibration8tiles and masks as native_sparse. No additional model approximation.')
(HERE/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
