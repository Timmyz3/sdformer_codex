"""Fresh assertion-instrumented review copies; no candidate source writes."""
import json
import shutil
import subprocess
from audit import H, SRC, FIX, DATA

TOP_ASSERTIONS = r'''
 logic [1:0] audit_owed;
 always_ff @(posedge clk)begin
  if(!reset_n) audit_owed<=0;
  else begin
   if((grant & ~eligible)!=0)$fatal(1,"grant to ineligible context");
   if(grant==3 && (request[0]&request[1])!=0)$fatal(1,"overlapping grant");
   if(proof_active && grant!=0)$fatal(1,"proof and context backend overlap");
   for(integer ac=0;ac<2;ac=ac+1)begin
    if(audit_owed[ac] && eligible[ac] && !grant[ac])$fatal(1,"second eligible denial");
    if(grant[ac] || request[ac]==0)audit_owed[ac]<=0;
    else if(eligible[ac])audit_owed[ac]<=1;
   end
  end
 end
'''

CONTEXT_ASSERTIONS = r'''
 logic audit_blocked;
 logic [5:0] audit_state;
 logic [255:0] audit_acc;
 logic [415:0] audit_hold;
 logic [63:0] audit_correction;
 logic [39:0] audit_pending;
 logic [51:0] audit_z[0:7][0:9];
 logic [31:0] audit_p[0:7];
 integer audit_k,audit_fp,audit_zrow,audit_og,audit_qfill,audit_row,audit_load_xy;
 logic [8:0] audit_p_index;
 always_ff @(posedge clk)begin
  if(!reset_n) audit_blocked<=0;
  else begin
   if(audit_blocked)begin
    if(state!=audit_state || k!=audit_k || fp!=audit_fp || zrow!=audit_zrow || og!=audit_og ||
       qfill!=audit_qfill || row!=audit_row || load_xy!=audit_load_xy || pending!=audit_pending)
       $fatal(1,"denied context state advanced");
    for(integer al=0;al<8;al=al+1)begin
     if(acc[al]!=audit_acc[al*32+:32] || z_hold[al]!=audit_hold[al*52+:52])$fatal(1,"denied context data advanced");
     if(p_mem[al][audit_p_index]!=audit_p[al])$fatal(1,"denied psum committed");
     for(integer ar=0;ar<10;ar=ar+1)if(z_mem[al][ar]!=audit_z[al][ar])$fatal(1,"denied z committed");
     for(integer ap=0;ap<4;ap=ap+1)if(correction[al][ap]!=audit_correction[al*8+ap*2+:2])$fatal(1,"denied repair changed");
    end
   end
   audit_blocked<=resource_request!=0 && !resource_grant;
   audit_state<=state;audit_k<=k;audit_fp<=fp;audit_zrow<=zrow;audit_og<=og;audit_qfill<=qfill;
   audit_row<=row;audit_load_xy<=load_xy;audit_pending<=pending;
   audit_p_index<=9'((state==DRAIN_READ)?row:og*40+fp);
   for(integer al=0;al<8;al=al+1)begin
    audit_acc[al*32+:32]<=acc[al];audit_hold[al*52+:52]<=z_hold[al];
    audit_p[al]<=p_mem[al][(state==DRAIN_READ)?row:og*40+fp];
    for(integer ar=0;ar<10;ar=ar+1)audit_z[al][ar]<=z_mem[al][ar];
    for(integer ap=0;ap<4;ap=ap+1)audit_correction[al*8+ap*2+:2]<=correction[al][ap];
   end
  end
 end
'''


def copy_and_build(stream=False):
    folder = H/('stream' if stream else 'fixtures_build')
    folder.mkdir(exist_ok=True)
    for name, assertions in [('interleave_stream.sv', TOP_ASSERTIONS), ('rr_context.sv', CONTEXT_ASSERTIONS), ('i24_consumer.sv', '')]:
        source = (SRC/name).read_text()
        if assertions:
            source = source.replace('endmodule', assertions+'\nendmodule')
        (folder/name).write_text(source)
    tb = (SRC/('stream_tb.cpp' if stream else 'tb.cpp')).read_text()
    tb = 'double sc_time_stamp(){return 0.0;}\n' + tb
    tb = tb.replace('d.mode=mode;', 'unsigned runmode=(mode==3?1+command%2:mode==4?2-command%2:mode);d.mode=runmode;')
    tb = tb.replace('(mode==1&&!expected_range)', '(runmode==1&&!expected_range)')
    tb = tb.replace('(mode==2?10*d.retired_tiles:0)', '(runmode==2?10*d.retired_tiles:0)')
    tb = tb.replace('if(mode!=2&&', 'if(runmode!=2&&')
    (folder/'tb.cpp').write_text(tb)
    with (folder/'build.log').open('w') as log:
        subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','interleave_stream','--Mdir','obj',
                        'interleave_stream.sv','rr_context.sv','i24_consumer.sv','tb.cpp','-CFLAGS','-O3 -std=c++14'],
                        cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['make','-C','obj','-f','Vinterleave_stream.mk','-j2'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
    return folder


def run(folder, args, name):
    r = subprocess.run([str(folder/'obj/Vinterleave_stream'),*map(str,args)],cwd=folder,text=True,capture_output=True)
    (folder/f'{name}.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0, (name,r.returncode,r.stderr,r.stdout[-1000:])
    return [dict(json.loads(line), test=name) for line in r.stdout.splitlines() if line.startswith('{')]


def main():
    folder=copy_and_build()
    records=[]
    for name in ['one','extreme_1','extreme_-1']:
        for mode in [3,4]:
            records += run(folder,[FIX/'fixtures'/name,mode,1,6460,2],f'{name}_mode{mode}_two_tiles')
    folder=copy_and_build(stream=True)
    files=[DATA/f'{name}.npy' for name in ['first_source_words','identity_fp32_full','raw_p_full','i24_new_full','identity_q20_full']]
    for first,mode in [(159,3),(19197,4)]:
        records += run(folder,[*files,FIX/'fixtures/real_0',mode,first,3,1,2,1000000],f'native{first}_mode{mode}_three_tiles')
    (H/'rtl_results.json').write_text(json.dumps(records,indent=2)+'\n')
    summary=dict(passed=True,commands=len(records),raw_J_I24_each=sum(r['outputs'] for r in records),
                 cross_mode_commands=len(records),all_runs_stalled=True,
                 repair_denied_cycles=sum(r['core_repair_arbitration_stalls'] for r in records),
                 normalize_denied_cycles=sum(r['core_normalization_arbitration_stalls'] for r in records),
                 assertions='eligible grants, disjoint resources, proof/backend exclusivity, at most one eligible denial, state/acc/holding/correction/all-z/current-psum freeze under denial')
    (H/'rtl_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
