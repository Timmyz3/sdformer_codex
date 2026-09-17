from pathlib import Path
import re
P=Path(__file__).resolve().parent

def normalize(s,joined):
 group='FGROUP' if joined else 'GROUP';plane='FPLANE' if joined else 'PLANE_SUM';hg='hg' if joined else 'hgroup'
 if joined:
  s=s.replace('module joined_fc1(','module normalized_fc1(')
  s=s.replace('output logic[3839:0] mon_group_u,mon_bound_lo,mon_bound_hi,\n output logic[959:0] mon_tail,mon_tail_delta,','output logic[3839:0] mon_group_u,\n output logic[3919:0] mon_norm_n,mon_norm_p,mon_q_n,mon_q_p,mon_shift_n,mon_shift_p,')
  s=s.replace('pos[10],neg[10],tail_pos[10],tail_neg[10],v[80]','pos[10],neg[10],v[80]')
  s=s.replace('logic signed[47:0] bound_a[80],bound_lo[80],bound_hi[80],tail_a[20],tail_b[20],tail_delta[20];','logic signed[48:0] norm_n[80],norm_p[80],q_n[80],q_p[80],shift_n[80],shift_p[80];')
  s=s.replace('logic[79:0] lower_hit,upper_hit,lo_less,lo_equal,hi_less,hi_equal;','wire[79:0] lower_hit,upper_hit;')
  s=s.replace('mon_bound_lo=0;mon_bound_hi=0;mon_tail=0;mon_tail_delta=0;','mon_norm_n=0;mon_norm_p=0;mon_q_n=0;mon_q_p=0;mon_shift_n=0;mon_shift_p=0;')
  start=s.index('   for(integer t=0;t<10;t++)begin\n    tail_a');end=s.index('   locked_next=locked;',start)
  s=s[:start]+'''   for(integer t=0;t<10;t++)mon_table_data[t*16+:16]=code==0?16'd0:16'(add_y[t]);
'''+s[end:]
  start=s.index('    bound_a[i]=');end=s.index('    if(!locked[i])',start);s=s[:start]+s[end:]
  s=s.replace('mon_bound_lo[i*48+:48]=bound_lo[i];mon_bound_hi[i*48+:48]=bound_hi[i];', 'mon_norm_n[i*49+:49]=norm_n[i];mon_norm_p[i*49+:49]=norm_p[i];mon_q_n[i*49+:49]=q_n[i];mon_q_p[i*49+:49]=q_p[i];mon_shift_n[i*49+:49]=shift_n[i];mon_shift_p[i*49+:49]=shift_p[i];')
  s=s.replace('for(integer t=0;t<10;t++)begin pos[t]<=0;neg[t]<=0;end', "for(integer t=0;t<10;t++)begin pos[t]<=-48'sd1;neg[t]<=-48'sd1;end")
  s=s.replace('       for(integer t=0;t<10;t++)begin tail_pos[t]<=tail_delta[2*t];tail_neg[t]<=tail_delta[2*t+1];end\n','')
  s=s.replace('       if(cert&&m!=0)for(integer t=0;t<10;t++)begin tail_pos[t]<=$signed(tail_delta[2*t])>>>1;tail_neg[t]<=$signed(tail_delta[2*t+1])>>>1;end\n','')
 else:
  s=s.replace('module subset_psn(','module normalized_leaf(')
  s=s.replace('output logic mon_lower,mon_upper,output logic[3839:0] mon_bound,mon_bound_hi,\n output logic[959:0] mon_tail,mon_tail_delta,','output logic mon_lower,mon_upper,output logic[3919:0] mon_norm_n,mon_norm_p,mon_q_n,mon_q_p,mon_shift_n,mon_shift_p,')
  s=s.replace('pos[10],neg[10],tail_pos[10],tail_neg[10]','pos[10],neg[10]')
  s=s.replace('logic signed[47:0] bound_a[80],bound_lo[80],bound_hi[80];\n logic signed[47:0] tail_a[20],tail_b[20],tail_delta[20];','logic signed[48:0] norm_n[80],norm_p[80],q_n[80],q_p[80],shift_n[80],shift_p[80];')
  s=s.replace('logic[79:0] lower_hit,upper_hit,locked_next,gate_next,lo_less,lo_equal,hi_less,hi_equal;', 'logic[79:0] locked_next,gate_next;wire[79:0] lower_hit,upper_hit;')
  s=s.replace('mon_bound=0;mon_bound_hi=0;mon_tail=0;mon_tail_delta=0;', 'mon_norm_n=0;mon_norm_p=0;mon_q_n=0;mon_q_p=0;mon_shift_n=0;mon_shift_p=0;')
  start=s.index('   for(integer t=0;t<10;t++)begin\n     tail_a');end=s.index('   lower_hit=0;',start)
  s=s[:start]+s[end:];s=s.replace('lower_hit=0;upper_hit=0;','')
  start=s.index('     // A further two parallel 48-bit bound');end=s.index('     if(!locked[i])',start);s=s[:start]+s[end:]
  s=s.replace('mon_bound[i*48+:48]=bound_lo[i];mon_bound_hi[i*48+:48]=bound_hi[i];', 'mon_norm_n[i*49+:49]=norm_n[i];mon_norm_p[i*49+:49]=norm_p[i];mon_q_n[i*49+:49]=q_n[i];mon_q_p[i*49+:49]=q_p[i];mon_shift_n[i*49+:49]=shift_n[i];mon_shift_p[i*49+:49]=shift_p[i];')
  s=s.replace('for(integer t=0;t<10;t++)begin pos[t]<=0;neg[t]<=0;end', "for(integer t=0;t<10;t++)begin pos[t]<=-48'sd1;neg[t]<=-48'sd1;end")
  s=s.replace('       for(integer t=0;t<10;t++)begin tail_pos[t]<=tail_delta[2*t];tail_neg[t]<=tail_delta[2*t+1];end\n','')
  start=s.index('       if(cert && m!=0)begin');end=s.index('       if(m==0)',start);s=s[:start]+s[end:]
 instances=f''' for(genvar lane=0;lane<80;lane++)begin: normalized_lanes
  normalized_bound bound_unit(.clk,.group_enable(st=={group}),
   .positive(positive[int'({hg})*8+lane%8]),.tau(tau[lane/8][int'({hg})*8+lane%8]),
   .nv(nv[lane]),.n_minus_one(neg[lane/8]),.p_minus_one(pos[lane/8]),.m,
   .norm_n(norm_n[lane]),.norm_p(norm_p[lane]),.q_n(q_n[lane]),.q_p(q_p[lane]),
   .shift_n(shift_n[lane]),.shift_p(shift_p[lane]),.lower_hit(lower_hit[lane]),.upper_hit(upper_hit[lane]));
 end
'''
 s=s.replace(' always_comb begin',instances+' always_comb begin',1)
 # Separate cell-result consumption from nv production. This removes a
 # Verilator process-level feedback dependency without changing any data edge/register.
 hit=re.search(r'if\(!locked\[i\]\)begin\s*if\(lower_hit\[i\]\).*?gate_next\[i\]=!positive\[h\];end\s*end',s,re.S)
 assert hit
 hitcode=hit.group(0);s=s[:hit.start()]+s[hit.end():]
 monitor='mon_norm_n[i*49+:49]=norm_n[i];mon_norm_p[i*49+:49]=norm_p[i];mon_q_n[i*49+:49]=q_n[i];mon_q_p[i*49+:49]=q_p[i];mon_shift_n[i*49+:49]=shift_n[i];mon_shift_p[i*49+:49]=shift_p[i];'
 zeros='mon_norm_n=0;mon_norm_p=0;mon_q_n=0;mon_q_p=0;mon_shift_n=0;mon_shift_p=0;'
 assert monitor in s and zeros in s
 s=s.replace(monitor,'').replace(zeros,'').replace('locked_next=locked;gate_next=gate;','')
 observer=" always_comb begin\n  locked_next=locked;gate_next=gate;"+zeros+"\n  for(integer i=0;i<80;i++)begin\n   integer h;h=int'("+hg+")*8+i%8;\n   "+hitcode+"\n   "+monitor+"\n  end\n end\n"
 s=s.replace(' always_ff',observer+' always_ff',1)
 assert 'tail_' not in s and 'bound_a' not in s and '<<<m' not in s
 return s
(P/'normalized_fc1.sv').write_text(normalize((P.parent/'joined_fc1/joined_fc1.sv').read_text(),True))
(P/'normalized_leaf.sv').write_text(normalize((P.parent/'fused_datapath/subset_psn.sv').read_text(),False))
print('generated shared normalized_bound users: joined + direct Y diagnosis leaf')
