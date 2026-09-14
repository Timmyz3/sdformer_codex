from pathlib import Path
import shutil
H=Path(__file__).resolve().parent;B=H.parents[1]
O=B/'fusion_ten_trials_20260914/dataflow/d1_forward'
AUX='aux_reads aux_writes aux_issues aux_weight_words aux_events bitmap_native_reads bitmap_native_issues cache_reads cache_writes'.split()
shutil.copyfile(H.parent/'bitmap_block_resident/decomp_core.sv',H/'decomp_core.sv')
shutil.copyfile(O/'i24_consumer.sv',H/'i24_consumer.sv')
s=(O/'consumer_stream.sv').read_text()
s=s.replace('logic [1:0] mode_q;','logic [3:0] mode_q;').replace('mode_q<=mode[1:0];','mode_q<=mode;')
s=s.replace('forward_core leaf (','decomp_core leaf (').replace(".mode(4'd15),.flow_mode(mode_q[1:0])",'.mode(mode_q)')
s=s.replace('(mode>2)','(mode!=14&&mode!=9&&mode!=8)')
s=s.replace(' output logic [63:0] core_cycles,',''.join(' output logic [63:0] core_'+x+',\n' for x in AUX)+' output logic [63:0] core_cycles,')
s=s.replace(' logic [31:0] l_cycles;',''.join(' logic [31:0] l_'+x+';\n' for x in AUX)+' logic [31:0] l_cycles;')
s=s.replace('  .cycles(l_cycles),',''.join('  .'+x+'(l_'+x+'),\n' for x in AUX)+'  .cycles(l_cycles),')
s=s.replace('   core_cycles<=0;',''.join('   core_'+x+'<=0;\n' for x in AUX)+'   core_cycles<=0;')
s=s.replace("      core_cycles<=core_cycles+64'(l_cycles);",''.join("      core_"+x+"<=core_"+x+"+64'(l_"+x+");\n" for x in AUX)+"      core_cycles<=core_cycles+64'(l_cycles);")
(H/'consumer_stream.sv').write_text(s)
# Use the mature full-gold harness; count_rr adds cross-mode, detailed raw errors and finite identity checks.
for name in ['tb.cpp','stream_tb.cpp']:
 t=(H.parent/'count_rr'/name).read_text().replace('Vinterleave_stream','Vconsumer_stream')
 t=t.replace('param[12];for(int k:{4,5,6,7,8,9,10,11})','param[8];for(int k:{4,5,6,7})')
 t=t.replace('bool class_loaded=false,permutation_loaded=false;','')
 a=t.index('  unsigned expected_static=');b=t.index('  d.mode=active_mode;',a);t=t[:a]+'  unsigned expected_static=command?0:1848;\n'+t[b:]
 a=t.index('    if(d.proof_issues');b=t.index('    if(outputs!=',a);t=t[:a]+t[b:]
 t=t.replace('d.window_cycles+d.launch_cycles+d.static_words+d.parameter_stalls+d.source_load_words+d.origin_words+d.source_load_stalls+1','d.consumer_cycles+d.static_words+d.parameter_stalls+d.source_load_words+d.origin_words+d.source_load_stalls+2ULL*count+1')
 a=t.index('    SHOW(shared_wide_grants)');b=t.index('    SHOW(core_cycles);',a)
 t=t[:a]+'''    SHOW(total_cycles);SHOW(static_words);SHOW(parameter_stalls);SHOW(source_load_words);SHOW(external_source_words);SHOW(padding_words);SHOW(origin_words);SHOW(source_load_stalls);SHOW(output_beats);SHOW(retired_tiles);
    '''+''.join('SHOW(core_'+x+');' for x in AUX)+'\n'+t[b:]
 t=t.replace('SHOW(core_merged_updates)','SHOW(core_dual_updates)').replace('SHOW(core_arbitration_stalls);','')
 (H/name).write_text(t)
print('Copied bitmap leaf unchanged; attached existing single-context FP consumer')
