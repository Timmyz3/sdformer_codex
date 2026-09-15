from pathlib import Path
import subprocess,sys,csv,json
H=Path(__file__).resolve().parent
D=H/'producer_rtl'
OLD=H.parent/'psn/rtl/gp_slice'
SRC=H.parent/'consumer_transfer_20260914/gustav_intersection'
subprocess.run([sys.executable,str(H/'build_producer.py')],check=True)
subprocess.run([sys.executable,str(H/'prepare_producer_cases.py')],check=True)
cmds=[['verilator','--cc','--exe','--top-module','gp_slice','-Wno-fatal','--Mdir','obj','-CFLAGS','-std=c++17 -O2','gp_slice.sv',str(OLD/'gp_slice_pe.sv'),'scoreboard.cpp'],
      ['make','-C','obj','-f','Vgp_slice.mk','-j2'],
      [str(D/'obj/Vgp_slice'),str(D/'cases.bin'),str(D/'results.tsv')]]
with (D/'run.log').open('w') as log:
    for cmd in cmds:
        result=subprocess.run(cmd,cwd=D,stdout=log,stderr=subprocess.STDOUT)
        print(cmd[0],result.returncode,flush=True)
        if result.returncode:sys.exit(result.returncode)
rows=[{k:v if k=='case' else int(v) for k,v in r.items()} for r in csv.DictReader((D/'results.tsv').open(),delimiter='\t')]
groups={}
for r in rows:
    key=(r['case'],r['time'],r['reduce'],r['stress'],r['producer_calendar'])
    groups.setdefault(key,{})[r['intersection']]=r
equivalent=all(v[5]['service_from_first_write']==v[6]['service_from_first_write'] and v[5]['w_reads']==v[6]['w_reads'] for v in groups.values())
out={'tasks':len(rows),'gate_bits':len(rows)*320,'seen_vs_simple_flag_cycle_and_bytes_equal':equivalent,'source_writes_per_task':sorted({r['source_writes'] for r in rows}),'comparisons':[]}
for cal in (0,1):
    for stress in (0,1):
        for family in ('dense','pruned24','pruned_c16','other'):
            chosen=[]
            for v in groups.values():
                r=v[3]
                typ=('pruned24' if 'row_2of4_trained' in r['case'] else 'pruned_c16' if 'broadcast8_C16_half_trained' in r['case'] else 'dense') if r['real'] else 'other'
                if r['time']==0 and r['reduce']==0 and r['stress']==stress and r['producer_calendar']==cal and typ==family:chosen.append(v)
            if chosen:
                out['comparisons'].append({'producer_calendar':cal,'stress':stress,'family':family,'cases':[v[3]['case'] for v in chosen],
                    'mean_service':{str(m):sum(v[m]['service_from_first_write'] for v in chosen)/len(chosen) for m in range(3,7)},
                    'mean_rank_catchup':{str(m):sum(v[m]['rank_catchup_pe_beats'] for v in chosen)/len(chosen) for m in range(3,7)}})
(D/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
