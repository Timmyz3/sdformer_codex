"""Fresh instrumented RTL: authorization, denial holding, cross-mode, real contention."""
from pathlib import Path
import ast
import json
import shutil
import subprocess
import numpy as np

H = Path(__file__).resolve().parent
B = H.parents[2]
P = B / 'fusion_ten_trials_20260914/phase_borrow'
D = B / 'r8_consumer_fusion_20260914/data'
A = H / 'audit_rtl'
A.mkdir(exist_ok=True)

# Reuse the existing independent review's full z/state denial monitor. Reading
# only literal assertion declarations avoids running/importing its old audit.
tree = ast.parse((B / 'fusion_review_followup_20260914/audit_rr/rtl_audit.py').read_text())
monitors = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.endswith('_ASSERTIONS')}
monitors['TOP_ASSERTIONS'] += '''
 always_ff @(posedge clk)if(reset_n)begin
  if(cons_add_grant && borrow_active)$fatal(1,"producer stole consumer wide chain");
  for(integer a=0;a<2;a=a+1)begin
   if(request[a][5] && request[a]!=6'b100100)$fatal(1,"non-atomic wide/z request");
   if(grant[a] && request[a][5] && cons_add_grant)$fatal(1,"borrow bypassed consumer");
  end
 end
'''
monitors['CONTEXT_ASSERTIONS'] += '''
 always_ff @(posedge clk)if(reset_n)begin
  if(!resource_grant && z_read_bus!=0)$fatal(1,"ungranted z read");
  if(state==ZADD && mode_q==3 && resource_request!=6'b100100)$fatal(1,"wide update lost request");
 end
'''
for name, key in [('interleave_stream.sv','TOP_ASSERTIONS'), ('rr_context.sv','CONTEXT_ASSERTIONS'),
                  ('i24_consumer.sv',None), ('wide_phase_alu.sv',None)]:
    s = (H / name).read_text()
    if name=='i24_consumer.sv':
        s=s.replace('endmodule', '\n logic audit_wide_blocked;logic [3:0] audit_wide_state;logic [511:0] audit_wide_hold;\n always_ff @(posedge clk)begin\n  if(!reset_n)audit_wide_blocked<=0;\n  else begin\n   if(audit_wide_blocked)begin\n    if(state!=audit_wide_state)$fatal(1,"denied consumer advanced");\n    for(integer l=0;l<8;l=l+1)if(wide_hold[l]!=audit_wide_hold[l*64+:64])$fatal(1,"denied consumer sum changed");\n   end\n   audit_wide_blocked<=add_req&&!add_grant;audit_wide_state<=state;\n   for(integer l=0;l<8;l=l+1)audit_wide_hold[l*64+:64]<=wide_hold[l];\n  end\n end\nendmodule\n')
    if key:
        s = s.replace('endmodule', monitors[key] + '\nendmodule')
    (A / name).write_text(s)

s = (H / 'tb.cpp').read_text()
s = 'double sc_time_stamp(){return 0.0;}\n' + s
s = s.replace('if(src.size()!=1536 || raw.size()!=3840 || id.size()!=3840 || gold.size()!=3840)return 3;',
              '''bool pair=src.size()==96*4*6;
 if(src.size()!=(pair?96*4*6:1536) || raw.size()!=(pair?7680:3840) || id.size()!=raw.size() || gold.size()!=raw.size())return 3;''')
s = s.replace('if(count==1)d.source_data=', 'if(pair)d.source_data=src.at(c*24+ly*6+lx);\n    else if(count==1)d.source_data=')
s = s.replace('id.at(d.identity_address*8+l)', 'id.at(((pair?d.identity_tile-tile:0)*480+d.identity_address)*8+l)')
for name, counter in [('jgold','js'),('raw','raws'),('gold','outputs')]:
    s = s.replace(f'{name}.at(({counter}%480)*8+l)', f'{name}.at((pair?{counter}:({counter}%480))*8+l)')
(A / 'tb.cpp').write_text(s)

def prepare_pair():
    out = H / 'pair_fixture'
    out.mkdir(exist_ok=True)
    for k in [4,5,6,7]:
        shutil.copyfile(P / f'fixtures/one/param{k}.bin', out / f'param{k}.bin')
    # Native adjacent tiles share two source columns. The first four columns are
    # zero; the last two are one. Context 0 retires while context 1 still has Q1.
    word = np.zeros((96,4,6), dtype=np.uint16)
    word[:,:,4:] = 1023
    word.tofile(out / 'source.bin')
    c = np.load(D / 'consumer_coefficients.npz')
    q1,q2,a,b = [c[k].astype(np.int64) for k in ['q1','q2','a_q40','b_q20']]
    values = []
    for tile in range(2):
        w = word[:,:,tile*2:tile*2+4]
        spike = ((w[None] >> np.arange(10)[:,None,None,None]) & 1).astype(np.int64)
        patch = np.stack([spike[:,:,p//2:p//2+3,p%2:p%2+3].reshape(10,864) for p in range(4)])
        raw = patch @ q1.T @ q2.T
        # (P,T,N) -> the actual (N8,P,T,lane) output rows.
        raw = raw.reshape(4,10,12,8).transpose(2,0,1,3).reshape(480,8)
        wide = raw * np.repeat(a.reshape(12,1,8),40,axis=1).reshape(480,8)
        wide += np.repeat(b.reshape(12,1,8),40,axis=1).reshape(480,8) * (1<<20)
        quotient = wide >> 26
        rem = wide & ((1<<26)-1)
        rounded = quotient + ((rem>(1<<25))|((rem==(1<<25))&((quotient&1)!=0)))
        gold = np.clip(rounded,-(1<<23),(1<<23)-1)
        values.append((raw,gold))
    for i,name in [(0,'raw'),(1,'gold')]:
        np.concatenate([v[i] for v in values]).astype('<i4').tofile(out / f'{name}.bin')
    np.zeros((960,8),'<i4').tofile(out / 'identity.bin')
    np.zeros((960,8),'<f4').tofile(out / 'identity_fp32.bin')
    return out

pair = prepare_pair()
with (A / 'build.log').open('w') as log:
    subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','interleave_stream','--Mdir','obj',
                    'interleave_stream.sv','rr_context.sv','i24_consumer.sv','wide_phase_alu.sv','tb.cpp',
                    '-CFLAGS','-O1 -std=c++14'],cwd=A,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['make','-C','obj','-f','Vinterleave_stream.mk','-j2'],cwd=A,stdout=log,stderr=subprocess.STDOUT,check=True)
rows=[]
for f, tile, count in [(pair,6460,2), *[(P/f'fixtures/{n}',6460,2) for n in ['one','extreme_1','extreme_-1']],
                        (P/'fixtures/padding_poison',0,1), (P/'fixtures/zero',6460,1)]:
    for mode, nextmode in [(1,4),(4,3),(2,4),(4,2)]:
        name=f'{f.name}_{mode}_{nextmode}'
        r=subprocess.run([str(A/'obj/Vinterleave_stream'),str(f),str(mode),'1',str(tile),str(count),str(nextmode)],text=True,capture_output=True)
        (A/f'{name}.log').write_text(r.stdout+r.stderr)
        assert r.returncode==0,(name,r.returncode,r.stdout[-2000:],r.stderr)
        rows += [dict(json.loads(line),test=name,fixture=f.name,tiles=count,first_tile=tile) for line in r.stdout.splitlines() if line.startswith('{')]
(H/'audit_results.json').write_text(json.dumps(rows,indent=2)+'\n')
summary=dict(passed=True,commands=len(rows),values_each_stage=sum(r['outputs'] for r in rows),
             wide_conflict_cycles=sum(r['wide_conflict_cycles'] for r in rows),
             consumer_wide_waits=sum(r['consumer_wide_waits'] for r in rows),
             borrow_consumer_stalls=sum(r['borrow_consumer_stalls'] for r in rows),
             repair_denied_cycles=sum(r['core_repair_arbitration_stalls'] for r in rows),
             normalization_denied_cycles=sum(r['core_normalization_arbitration_stalls'] for r in rows))
assert summary['wide_conflict_cycles']>0 and summary['borrow_consumer_stalls']>0
(H/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
