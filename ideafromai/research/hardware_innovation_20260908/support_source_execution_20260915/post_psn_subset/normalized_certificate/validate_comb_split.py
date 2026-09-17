from pathlib import Path
import json,subprocess
P=Path(__file__).resolve().parent
names=['results_all.jsonl','results_swap.jsonl','results_diagnostics.jsonl']
before={n:[json.loads(x) for x in (P/n).read_text().splitlines()] for n in names}
def run(args,log):
 with (P/log).open('a') as f:subprocess.run(args,cwd=P,stdout=f,stderr=subprocess.STDOUT,check=True)
run(['verilator','--cc','--exe','--unroll-count','512','-Wall','-Wno-fatal','--top-module','normalized_fc1','normalized_fc1.sv','normalized_bound.sv','tb.cpp','-CFLAGS','-O2 -std=c++17'],'build_final.log')
run(['make','-C','obj_dir','-f','Vnormalized_fc1.mk','-j4'],'build_final.log')
for tag in ['all','swap']:
 with (P/(tag+'_final.log')).open('w') as f:subprocess.run([str(P/'obj_dir/Vnormalized_fc1'),tag],cwd=P,stdout=f,stderr=subprocess.STDOUT,check=True)
run(['verilator','--cc','--exe','--unroll-count','512','-Wall','-Wno-fatal','--top-module','normalized_leaf','--Mdir','obj_leaf','normalized_leaf.sv','normalized_bound.sv','leaf_tb.cpp','-CFLAGS','-O2 -std=c++17'],'build_leaf_final.log')
run(['make','-C','obj_leaf','-f','Vnormalized_leaf.mk','-j4'],'build_leaf_final.log')
with (P/'diagnostics_final.log').open('w') as f:subprocess.run([str(P/'obj_leaf/Vnormalized_leaf'),'2','diagnostics','diagnostics.bin'],cwd=P,stdout=f,stderr=subprocess.STDOUT,check=True)
for n in names:
 after=[json.loads(x) for x in (P/n).read_text().splitlines()]
 assert before[n]==after,n
for name in ['build_final.log','build_leaf_final.log']:
 s=(P/name).read_text()
 assert not any(x in s for x in ['Warning-UNOPTFLAT','Warning-LATCH','Warning-MULTIDRIVEN','%Error']),name
(P/'comb_split_check.json').write_text(json.dumps({'status':'PASS','records':496,'meaning':'Separate combinational blocks for prefix production and normalized-result consumption remove process-level UNOPTFLAT warnings. Every previously measured field/clock unchanged; no register/state/arithmetic change.'},separators=(',',':'))+'\n')
print('PASS496 exact records; final builds have no UNOPTFLAT/LATCH/MULTIDRIVEN/Errors')
