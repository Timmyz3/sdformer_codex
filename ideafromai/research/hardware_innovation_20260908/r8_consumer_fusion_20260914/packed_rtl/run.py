from pathlib import Path
import json,subprocess,concurrent.futures
H=Path(__file__).resolve().parent
with (H/'build.log').open('w') as log:
    for cmd in (['verilator','-Wall','--cc','--exe','--top-module','packed_r8','packed_r8.sv','tb.cpp','-CFLAGS','-O3'],['make','-C','obj_dir','-f','Vpacked_r8.mk','-j2']):
        subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
cases=json.loads((H/'definition.json').read_text())['fixtures']
def run(task):
    case,mode,stall=task
    r=subprocess.run([str(H/'obj_dir/Vpacked_r8'),str(H/'fixtures'/case),str(mode),str(stall)],capture_output=True,text=True)
    if r.returncode:raise RuntimeError((task,r.returncode,r.stdout,r.stderr))
    a=[dict(json.loads(x),fixture=case) for x in r.stdout.splitlines()]
    print('PASS',case,mode,stall,a[0]['cycles'],flush=True);return a
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(run,[(x['name'],m,s) for x in cases for m in [14,15] for s in [0,1]]):rows.extend(result)
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
real={}
fields=[k for k in rows[0] if k not in ['mode','stall','command','fixture','state_cycles']]
for mode in [14,15]:
    real[str(mode)]={}
    for stall in [0,1]:
        a=[r for r in rows if r['mode']==mode and r['stall']==stall and r['command']==0 and r['fixture'].startswith('real_')]
        real[str(mode)][str(stall)]={k:sum(r[k] for r in a) for k in fields}
summary={'runs':len(rows),'outputs_compared':sum(r['outputs'] for r in rows),'real_first_commands':real,
         'claim':'Measured full linear tile kernels; no Fmax/PPA/full-frame/consumer claim; both modes use same shared19x13 multiplier resources'}
aa=real['14']['0'];bb=real['15']['0']
summary['real_exact_no_stall_delta']={'scalar_first_issues':aa['first_issues'],'dual_first_issues':bb['first_issues'],
 'merged_dual_incidence':bb['dual_updates'],'cycles_per_merged_incidence':3,'cycles_saved':aa['cycles']-bb['cycles'],
 'percent_cycles_saved':100*(aa['cycles']-bb['cycles'])/aa['cycles']}
summary['same_ports_note']='Both mode14/15 share packed26 bank and208bit vectorport; not equalport to oldmode12'
summary['author_novelty_self_assessment']='Ordinary SWAR/precision packing/window/cachedOS A; local measured positive increment, X remains thin'
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
