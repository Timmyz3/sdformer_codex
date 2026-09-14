from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess
H=Path(__file__).resolve().parent
s=(H/'tb.cpp').read_text().replace('if(argc!=4)','if(argc!=4&&argc!=5)')
a=s.index(' for(unsigned i=0;i<src.size();i++)');b=s.index(' d.cfg_valid=0;unsigned checked=0;',a)
body=s[a:b]
reload=''' src=read(path+"/source.hex");gold=read(path+"/gold.hex");origin=read(path+"/origin.hex");
 q1=read(path+"/q1.hex");q2=read(path+"/q2.hex");klive=read(path+"/k_live.hex");
'''
body=body.replace('read(dir+','read(path+').replace('return 21;', 'throw std::runtime_error("bad metadata shape");')
s=s[:a]+' auto load=[&](const std::string& path)->void{\n'+reload+body+' d.cfg_valid=0;};\n load(dir);\n'+s[b:]
s=s.replace('for(unsigned command=0;command<2;command++){','for(unsigned command=0;command<2;command++){\n  if(command==1){cfg_cycles=0;if(argc==5)load(argv[4]);}')
s=s.replace('(command==0?cfg_cycles:0)','cfg_cycles')
(H/'reconfig_tb.cpp').write_text(s)
with (H/'build_reconfigure.log').open('w') as log:
 for cmd in [['verilator','-Wall','--cc','--exe','--top-module','decomp_core','--Mdir','obj_reconfigure','decomp_core.sv','reconfig_tb.cpp','-CFLAGS','-O3'],['make','-C','obj_reconfigure','-f','Vdecomp_core.mk','-j2']]:subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
pairs=[('mixed_class_255_256','max_positive'),('max_positive','edge_low_255'),('edge_low_255','q2_zero_overlay'),('q2_zero_overlay','random_padding_poison')]
def run(job):
 a,b,m,stall=job
 proc=subprocess.run([str(H/'obj_reconfigure/Vdecomp_core'),str(H/'fixtures'/a),str(m),str(stall),str(H/'fixtures'/b)],text=True,capture_output=True)
 if proc.returncode:raise RuntimeError((job,proc.returncode,proc.stdout,proc.stderr))
 rows=[dict(json.loads(line),fixture=(a if i==0 else b),first_fixture=a,second_fixture=b) for i,line in enumerate(proc.stdout.splitlines())]
 assert len(rows)==2
 print('PASS',job,[r['cycles'] for r in rows],flush=True);return rows
rows=[]
with ThreadPoolExecutor(max_workers=3) as pool:
 for a in pool.map(run,[(a,b,m,s) for a,b in pairs for m in [14,20] for s in [0,1]]):rows+=a
standalone=json.loads((H/'results.json').read_text())
for r in rows:
 base=next(x for x in standalone if x['fixture']==r['fixture'] and x['mode']==r['mode'] and x['stall']==r['stall'] and x['command']==0)
 for key,value in base.items():
  if key not in ['command','reference']:assert r[key]==value,(key,r[key],value)
(H/'results_reconfigure.json').write_text(json.dumps(rows,indent=2)+'\n')
(H/'reconfigure_checks.json').write_text(json.dumps(dict(passed=True,commands=len(rows),raw_values=sum(r['outputs'] for r in rows),all_counters_match_standalone=True,second_command_full_source_Q1_Q2_class_reconfigured=len(rows)//2),indent=2)+'\n')
