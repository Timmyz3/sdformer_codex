from pathlib import Path
import re
H=Path(__file__).resolve().parent
core=['cycles','source_words','weight_words','second_weight_words','local_source_reads','z_vector_reads','z_scalar_reads','z_writes','first_issues','dual_updates','psum_reads','psum_writes','mac_issues','source_stalls','weight_stalls','output_stalls']
p=H/'generate_wrapper.py';s=p.read_text();s=re.sub(r'^core=\[.*\]$',f'core={core!r}',s,flags=re.M)
s=s.replace('r8_fusion leaf (','packed_r8 leaf (').replace('.cfg_addr(core_cfg_addr),','.cfg_addr(core_cfg_addr[10:0]),')
s=s.replace(' logic cons_done, cons_error, cons_identity_request, cons_result_valid;',' logic cons_done, cons_error, cons_identity_request, cons_result_valid;\n logic [5:0] unused_debug_state;')
s=s.replace('.result_data(raw_data),.done(leaf_done),','.result_data(raw_data),.done(leaf_done),.debug_state(unused_debug_state),')
s=s.replace('mode_q<=6','mode_q<=14').replace('param_kind<=1','param_kind<=4').replace('(mode!=6 && mode!=7 && mode!=9 && mode!=11)','(mode!=14 && mode!=15)')
s=s.replace('if((param_kind==1 && param_row==10367) || (param_kind==2 && param_row==287) ||\n        ((param_kind==4 || param_kind==6)','if(((param_kind==4 || param_kind==6)')
s=s.replace('1:param_kind<=2;2:param_kind<=4;4:param_kind<=5;','4:param_kind<=5;');p.write_text(s)
p=H/'tb.cpp';s=p.read_text().replace('for(int k:{1,2,4,5,6,7})','for(int k:{4,5,6,7})').replace('12504','1848')
s=re.sub(r'^\s*SHOW\(core_cycles\).*$', '    '+''.join('SHOW(core_'+k+');' for k in core),s,flags=re.M);p.write_text(s)
p=H/'make_stream_tb.py';s=p.read_text().replace('for(int k:{1,2,4,5,6,7})','for(int k:{4,5,6,7})');p.write_text(s)
for f in ['run.py','run_stream.py']:
 p=H/f;s=p.read_text().replace('../fusion_rtl/r8_fusion.sv','../packed_rtl/packed_r8.sv').replace('for m in [6,7,9,11]','for m in [14,15]').replace('for m in [9,11]','for m in [14,15]');p.write_text(s)
p=H/'prepare.py';s=p.read_text().replace(' return p\n',' return {k:v for k,v in p.items() if k not in [1,2]}\n');p.write_text(s)
