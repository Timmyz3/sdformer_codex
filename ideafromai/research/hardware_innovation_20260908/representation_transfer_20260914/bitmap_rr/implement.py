"""Attach the final K16 bitmap layout to the existing shared two-context consumer."""
from pathlib import Path
import shutil
H=Path(__file__).resolve().parent;B=H.parents[1]
OLD=B/'consumer_transfer_20260914/count_rr'
COUNTERS=['bitmap_native_reads','bitmap_native_issues','cache_reads','cache_writes',
          'bitmap_z_arbitration_stalls','bitmap_alu_arbitration_stalls','bitmap_weight_arbitration_stalls',
          'bitmap_hold_writes','bitmap_hold_reads','bitmap_row_z_arbitration_stalls']
def replace(s,a,b):
 assert a in s,a
 return s.replace(a,b)
s=(OLD/'rr_context.sv').read_text()
s=replace(s,'output logic [1:0] weight_kind','output logic [2:0] weight_kind')
s=replace(s,' input logic [39:0] time_permutation', ''' input logic [161:0] plane_live,
 output logic alu_pop,output logic [15:0] pop_mask,output logic [127:0] pop_plane,output logic [1:0] pop_shift,
 output logic [31:0] '''+','.join(COUNTERS)+''',
 input logic [39:0] time_permutation''')
s=replace(s,'G_ZREAD,G_MAC} state_t;','G_ZREAD,G_MAC,BM_GROUP,BM_WFILL,BM_POS,BM_SCAN,BM_CACHED_POP,BM_NATIVE_QREAD,BM_NATIVE_ADD,BM_STORE} state_t;')
s=replace(s,' logic [9:0] source_mem', ''' wire bitmap_mode=(mode_q==8||mode_q==7);
 logic [15:0] bitmap[0:39],bm_hold,bitmap_read_word;
 logic [39:0] bitmap_live,bm_pending,bm_after;
 logic signed [31:0] bm_acc[0:7];logic [51:0] bm_write[0:7];
 integer bm_block,bm_plane,first_live,next_live,onehot_index;
 logic bitmap_onehot,bm_plane_live,bm_same_row;
 logic [2:0] qblock_read_index;logic qblock_read_enable;logic [127:0] qblock_read_bus;
 task advance_block;
 begin
 if(k==863)begin fp<=0;state<=ZSCAN;end
 else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end
 end endtask
 logic [9:0] source_mem''')
s=replace(s,' aux_reads<=0;', ''.join(f' {c}<=0;' for c in COUNTERS)+'\n aux_reads<=0;')
s=replace(s,'z_read_address=(state==ZSCAN || state==BASE_MAC || state==G_ZREAD || state==G_MAC)?','z_read_address=(state==ZSCAN || state==BASE_MAC || state==G_ZREAD || state==G_MAC || state==BM_POS || state==BM_STORE)?')
s=replace(s,'  G_MAC:resource_request=6\'b010100;', '''  G_MAC:resource_request=6'b010100;
  BM_POS:resource_request=6'b000100;
  BM_STORE:if(mode_q!=7||!bm_same_row)resource_request=6'b000100;
  BM_WFILL:if(bm_plane_live)resource_request=6'b000010;
  BM_CACHED_POP:if(bm_plane_live)resource_request=6'b010000;
  BM_NATIVE_QREAD:resource_request=6'b000010;
  BM_NATIVE_ADD:resource_request=6'b010000;''')
s=replace(s," weight_kind=(state==VLOAD)?2'd1:(state==MREAD)?2'd2:(state==G_QREAD)?2'd3:2'd0;", " weight_kind=(state==BM_WFILL)?3'd4:(state==VLOAD)?3'd1:(state==MREAD)?3'd2:(state==G_QREAD)?3'd3:3'd0;")
s=replace(s," weight_address=(state==VLOAD)?", " weight_address=(state==BM_WFILL)?10'(bm_plane*54+bm_block):(state==BM_NATIVE_QREAD)?10'(bm_block*16+onehot_index):(state==VLOAD)?")
s=replace(s,' else if(state==G_MAC&&packed_retire)alu_format=5;', ''' else if(state==G_MAC&&packed_retire)alu_format=5;
 else if(state==BM_CACHED_POP&&bm_plane==2)alu_format=6;''')
s=replace(s,'state==NORMALIZE_READ || state==G_ZREAD ||','state==NORMALIZE_READ || state==G_ZREAD || state==BM_POS ||')
s=replace(s,'  multiply_coefficient[l]={{3{qblock[selected_rank][l][15]}},qblock[selected_rank][l]};',
            '  multiply_coefficient[l]={{3{qblock_read_bus[l*16+15]}},qblock_read_bus[l*16+:16]};')
s=replace(s,'  alu_lhs[l*32+:32]=lhs[l];', '''  if(state==BM_CACHED_POP)begin lhs[l]=bm_acc[l];rhs[l]=0;end
  if(state==BM_NATIVE_ADD)begin lhs[l]=bm_acc[l];rhs[l]={{29{q_hold[l][2]}},q_hold[l]};end
  alu_lhs[l*32+:32]=lhs[l];''')
s=replace(s,' always_comb begin\n  for(integer lane=0;lane<8;lane=lane+1)add_y', ''' always_comb begin
  bm_plane_live=(bm_plane<3)?plane_live[bm_plane*54+bm_block]:1'b0;
  bitmap_read_word=(state==BM_SCAN)?bitmap[fp]:16'd0;
  bitmap_onehot=(bitmap_read_word!=0)&&((bitmap_read_word&(bitmap_read_word-16'd1))==0);
  onehot_index=0;for(integer j=15;j>=0;j=j-1)if(bm_hold[j])onehot_index=j;
  bm_after=bm_pending&~(40'd1<<fp);first_live=0;next_live=0;
  for(integer j=39;j>=0;j=j-1)begin
   if(bitmap_live[j])first_live=j;
   if(bm_after[j])next_live=j;
  end
  if(mode_q==7)for(integer t=9;t>=0;t=t-1)for(integer p=3;p>=0;p=p-1)begin
   if(bitmap_live[p*10+t])first_live=p*10+t;
   if(bm_after[p*10+t])next_live=p*10+t;
  end
  bm_same_row=(bm_after!=0)&&(next_live%10==fp%10);
 end
 always_comb begin
  alu_pop=(state==BM_CACHED_POP)&&bm_plane_live;
  pop_mask=bm_hold;pop_shift=2'(bm_plane);
  qblock_read_index=(state==BM_CACHED_POP)?3'(bm_plane):selected_rank;
  qblock_read_enable=resource_grant&&((state==BASE_MAC)||(state==BM_CACHED_POP&&bm_plane_live));
  for(integer i=0;i<8;i=i+1)begin
   qblock_read_bus[i*16+:16]=qblock_read_enable?qblock[qblock_read_index][i]:16'd0;
   bm_write[i]=z_hold[i];bm_write[i][13*(fp/10)+:13]=bm_acc[i][12:0];
  end
  pop_plane=qblock_read_bus;
 end
 always_comb begin
  for(integer lane=0;lane<8;lane=lane+1)add_y''')
s=replace(s,' if(!reset_n)begin\n  block_pending', ''' if(!reset_n)begin
  bm_block<=0;bm_plane<=0;bm_hold<=0;bm_pending<=0;bitmap_live<=0;
  for(integer i=0;i<8;i=i+1)bm_acc[i]<=0;
  block_pending''')
s=replace(s,'   else arbitration_stalls<=arbitration_stalls+1;', '''   else begin
    arbitration_stalls<=arbitration_stalls+1;
    if(bitmap_mode)begin
     if(resource_request[2])bitmap_z_arbitration_stalls<=bitmap_z_arbitration_stalls+1;
     if(resource_request[4])bitmap_alu_arbitration_stalls<=bitmap_alu_arbitration_stalls+1;
     if(resource_request[1])bitmap_weight_arbitration_stalls<=bitmap_weight_arbitration_stalls+1;
     if(state==BM_POS||state==BM_STORE)bitmap_row_z_arbitration_stalls<=bitmap_row_z_arbitration_stalls+1;
    end
   end''')
s=replace(s,'   L_GATHER:if(!k_live[k])state<=KNEXT;', '''   L_GATHER:if(bitmap_mode)begin
    if(k_live[k])local_source_reads<=local_source_reads+1;
    for(integer i=0;i<40;i=i+1)begin
     bitmap[i][k%16]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
     if(k%16==0)bitmap_live[i]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
     else bitmap_live[i]<=bitmap_live[i]||(k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10]);
    end
    aux_writes<=aux_writes+1;state<=KNEXT;
   end else if(!k_live[k])state<=KNEXT;''')
s=replace(s,'   KNEXT:if(k==863)begin','   KNEXT:if(bitmap_mode&&k%16==15)state<=BM_GROUP;\n   else if(k==863)begin')
bitmap_states='''
 BM_GROUP:begin
  bm_pending<=bitmap_live;bm_block<=k/16;fp<=first_live;bm_plane<=0;
  if(bitmap_live==0)advance_block();else state<=BM_WFILL;
 end
 BM_WFILL:begin
  for(integer i=0;i<8;i=i+1)qblock[bm_plane][i]<=bm_plane_live?$signed(weight_data[i*32+:16]):16'sd0;
  cache_writes<=cache_writes+1;
  if(bm_plane_live)begin weight_words<=weight_words+1;aux_weight_words<=aux_weight_words+1;end
  if(bm_plane==2)begin bm_plane<=0;state<=BM_POS;end else bm_plane<=bm_plane+1;
 end
 BM_POS:begin
  for(integer i=0;i<8;i=i+1)begin
   z_hold[i]<=z_read_bus[i*52+:52];
   bm_acc[i]<=32'($signed(z_read_bus[i*52+13*(fp/10)+:13]));
  end
  z_vector_reads<=z_vector_reads+1;state<=BM_SCAN;
 end
 BM_SCAN:begin
  if(mode_q==7)begin
   for(integer i=0;i<8;i=i+1)bm_acc[i]<=32'($signed(z_hold[i][13*(fp/10)+:13]));
   bitmap_hold_reads<=bitmap_hold_reads+1;
  end
  bm_hold<=bitmap_read_word;aux_reads<=aux_reads+1;bm_plane<=0;
  state<=(bitmap_read_word==0)?BM_STORE:(bitmap_onehot?BM_NATIVE_QREAD:BM_CACHED_POP);
 end
 BM_CACHED_POP:begin
  if(bm_plane_live)begin
   for(integer i=0;i<8;i=i+1)bm_acc[i]<=add_y[i];
   first_issues<=first_issues+1;aux_issues<=aux_issues+1;cache_reads<=cache_reads+1;
  end
  if(bm_plane==2)state<=BM_STORE;else bm_plane<=bm_plane+1;
 end
 BM_NATIVE_QREAD:begin
  for(integer i=0;i<8;i=i+1)q_hold[i]<=weight_data[i*32+:3];
  weight_words<=weight_words+1;bitmap_native_reads<=bitmap_native_reads+1;state<=BM_NATIVE_ADD;
 end
 BM_NATIVE_ADD:begin
  for(integer i=0;i<8;i=i+1)bm_acc[i]<=add_y[i];
  first_issues<=first_issues+1;bitmap_native_issues<=bitmap_native_issues+1;state<=BM_STORE;
 end
 BM_STORE:begin
  bm_pending<=bm_after;
  if(mode_q==7&&bm_same_row)begin
   for(integer i=0;i<8;i=i+1)z_hold[i]<=bm_write[i];
   bitmap_hold_writes<=bitmap_hold_writes+1;fp<=next_live;state<=BM_SCAN;
  end else begin
   for(integer i=0;i<8;i=i+1)z_mem[i][read_zrow]<=bm_write[i];
   z_writes<=z_writes+1;
   if(bm_after==0)advance_block();else begin fp<=next_live;state<=BM_POS;end
  end
 end
'''
s=replace(s,' MREAD:begin',bitmap_states+' MREAD:begin')
s=replace(s,'  if(state==ZSCAN)check_retired<=1;', '''  if(state==ZSCAN)check_retired<=1;
  if(state==BM_WFILL||state==BM_CACHED_POP||state==BM_NATIVE_ADD)begin
   if(check_retired)$fatal(1,"bitmap qblock use after Q2 starts");
   if(bm_block>=54||bm_plane>=3)$fatal(1,"bitmap address range");
  end
  if(!resource_grant&&qblock_read_bus!=0)$fatal(1,"ungranted qblock read");''')
s=replace(s,' logic check_retired;integer check_stores;', ' logic check_retired,check_bm_owned;logic [3:0] check_bm_row;integer check_stores;')
s=replace(s,'if(!reset_n)begin check_retired<=0;check_stores<=0;end','if(!reset_n)begin check_retired<=0;check_stores<=0;check_bm_owned<=0;check_bm_row<=0;end')
s=replace(s,'   check_retired<=0;check_stores<=0;','   check_retired<=0;check_stores<=0;check_bm_owned<=0;')
s=replace(s,'  if(state==ZSCAN)check_retired<=1;', '''  if(mode_q==7)begin
   if(state==BM_POS&&resource_grant)begin
    if(check_bm_owned)$fatal(1,"bitmap holding overwrite before row store");
    check_bm_owned<=1;check_bm_row<=read_zrow;
   end
   if(state==BM_SCAN||state==BM_STORE)begin
    if(!check_bm_owned||check_bm_row!=read_zrow)$fatal(1,"bitmap holding row ownership");
   end
   if(state==BM_STORE&&!bm_same_row&&resource_grant)check_bm_owned<=0;
   if(state==ZSCAN&&check_bm_owned)$fatal(1,"Q2 before held bitmap row store");
  end
  if(state==ZSCAN)check_retired<=1;''')
(H/'rr_context.sv').write_text(s)
t=(OLD/'interleave_stream.sv').read_text()
t=replace(t,' output logic [63:0] core_aux_reads,',''.join(f' output logic [63:0] core_{c},\n' for c in COUNTERS)+' output logic [63:0] core_aux_reads,')
t=replace(t,' logic [1:0] weight_kind[0:1];',' logic [2:0] weight_kind[0:1];')
t=replace(t,' logic weight_owner,alu_owner,alu_active,alu_mac,proof_active;', ''' logic [15:0] bp_q[0:2][0:7][0:53];logic [161:0] plane_live;logic [2:0] cfg_plane_live;
 logic [1:0] child_pop;logic [15:0] child_pop_mask[0:1];logic [127:0] child_pop_plane[0:1];logic [1:0] child_pop_shift[0:1];
 logic alu_pop,alu_sub;logic [4:0] pop_count[0:7];logic [15:0] pop_input[0:7];
 function automatic [4:0] pop16(input logic [15:0] a);
 logic [1:0] x[0:7];logic [2:0] y[0:3];logic [3:0] z[0:1];
 begin
 for(integer i=0;i<8;i=i+1)x[i]={1'b0,a[2*i]}+{1'b0,a[2*i+1]};
 for(integer i=0;i<4;i=i+1)y[i]={1'b0,x[2*i]}+{1'b0,x[2*i+1]};
 for(integer i=0;i<2;i=i+1)z[i]={1'b0,y[2*i]}+{1'b0,y[2*i+1]};
 pop16={1'b0,z[0]}+{1'b0,z[1]};
 end endfunction
 logic weight_owner,alu_owner,alu_active,alu_mac,proof_active;''')
t=replace(t,' logic [31:0] l_aux_reads[0:1];',''.join(f' logic [31:0] l_{c}[0:1];\n' for c in COUNTERS)+' logic [31:0] l_aux_reads[0:1];')
t=replace(t,"  alu_format=proof_active?3'd1:(alu_active?child_format[alu_owner]:3'd0);", """  alu_format=proof_active?3'd1:(alu_active?child_format[alu_owner]:3'd0);
  alu_pop=alu_active&&!proof_active&&child_pop[alu_owner];alu_sub=alu_active&&!proof_active&&(alu_format==6);
  cfg_plane_live=0;
  for(integer b=0;b<3;b=b+1)for(integer l=0;l<8;l=l+1)cfg_plane_live[b]=cfg_plane_live[b]|parameter_data[l*32+b];""")
t=replace(t,"    else if(weight_kind[weight_owner]==3)shared_weight_data[l*32+:3]=representative[weight_addr[weight_owner][4:0]][l*3+:3];", """    else if(weight_kind[weight_owner]==3)shared_weight_data[l*32+:3]=representative[weight_addr[weight_owner][4:0]][l*3+:3];
    else if(weight_kind[weight_owner]==4)shared_weight_data[l*32+:16]=bp_q[weight_addr[weight_owner]/54][l][weight_addr[weight_owner]%54];""")
t=replace(t,"   if(alu_mac&&alu_format==5)rhs[l]=", """   pop_input[l]=alu_pop?(child_pop_mask[alu_owner]&child_pop_plane[alu_owner][l*16+:16]):16'd0;
   pop_count[l]=pop16(pop_input[l]);
   if(alu_pop)rhs[l]=$signed({27'd0,pop_count[l]})<<<child_pop_shift[alu_owner];
   if(alu_mac&&alu_format==5)rhs[l]=""")
t=replace(t,"   if(gb==0)assign cin=1'b0;","   if(gb==0)assign cin=alu_sub;")
t=replace(t,'lhs[gl][gb]^rhs[gl][gb]^cin','lhs[gl][gb]^(rhs[gl][gb]^alu_sub)^cin')
t=replace(t,'(lhs[gl][gb]&rhs[gl][gb])|((lhs[gl][gb]^rhs[gl][gb])&cin)','(lhs[gl][gb]&(rhs[gl][gb]^alu_sub))|((lhs[gl][gb]^(rhs[gl][gb]^alu_sub))&cin)')
t=replace(t,'   .time_permutation(time_permutation),.ngroups(ngroups),', '''   .plane_live(plane_live),.alu_pop(child_pop[c]),.pop_mask(child_pop_mask[c]),.pop_plane(child_pop_plane[c]),.pop_shift(child_pop_shift[c]),
'''+''.join(f'   .{c}(l_{c}[c]),\n' for c in COUNTERS)+'''   .time_permutation(time_permutation),.ngroups(ngroups),''')
t=replace(t,'   core_aux_reads<=0;',''.join(f'   core_{c}<=0;\n' for c in COUNTERS)+'   core_aux_reads<=0;')
t=replace(t,'    if(|leaf_done)core_aux_reads<=', ''.join(f"    if(|leaf_done)core_{c}<=core_{c}+(leaf_done[0]?64'(l_{c}[0]):64'd0)+(leaf_done[1]?64'(l_{c}[1]):64'd0);\n" for c in COUNTERS)+'    if(|leaf_done)core_aux_reads<=')
t=replace(t,'    if(param_kind==4)for(integer l=0;l<8;l=l+1)q_mem[l][param_row[9:0]]<=parameter_data[l*32+:3];', '''    if(param_kind==4)begin
     for(integer l=0;l<8;l=l+1)begin
      q_mem[l][param_row[9:0]]<=parameter_data[l*32+:3];
      for(integer b=0;b<3;b=b+1)bp_q[b][l][param_row[9:4]][param_row[3:0]]<=parameter_data[l*32+b];
     end
     for(integer b=0;b<3;b=b+1)
      if(param_row[3:0]==0)plane_live[b*54+int'(param_row[9:4])]<=cfg_plane_live[b];
      else plane_live[b*54+int'(param_row[9:4])]<=plane_live[b*54+int'(param_row[9:4])]|cfg_plane_live[b];
    end''')
t=replace(t,'mode!=4&&mode!=20','mode!=4&&mode!=7&&mode!=8&&mode!=20')
(H/'interleave_stream.sv').write_text(t)
for name in ['i24_consumer.sv','wide_phase_alu.sv']:shutil.copyfile(OLD/name,H/name)
for name in ['tb.cpp','stream_tb.cpp']:
 t=(OLD/name).read_text()
 t=replace(t,'+d.core_aux_issues)return 36;','+((active_mode==20||active_mode==21)?d.core_aux_issues:0))return 36;')
 t=replace(t,'+d.core_aux_reads+d.core_aux_writes ||','+((active_mode==20||active_mode==21)?d.core_aux_reads+d.core_aux_writes:0) ||')
 t=replace(t,'    SHOW(core_aux_reads);','    '+''.join(f'SHOW(core_{c});' for c in COUNTERS)+'\n    SHOW(core_aux_reads);')
 if name=='stream_tb.cpp':
  t=replace(t,'if(argc!=13 && argc!=14)return 2;','if(argc!=13 && argc!=14 && argc!=15)return 2;')
  t=replace(t,'  unsigned active_mode=command && argc>13?std::stoul(argv[13]):mode;',
            '  unsigned active_mode=command && argc>13?std::stoul(argv[13]):mode;\n  if(command && argc>14)first=std::stoul(argv[14]);')
 (H/name).write_text(t)
shutil.copyfile(OLD/'fixtures.json',H/'fixtures.json')
if not (H/'fixtures').exists():(H/'fixtures').symlink_to('../../consumer_transfer_20260914/count_rr/fixtures',target_is_directory=True)
t=(OLD/'run.py').read_text().replace('modes=[0,1,2,3,4,20,21]','modes=[2,4,21,8,7]').replace('for mode in [0,2,4,20,21]','for mode in [2,4,21,8,7]').replace('[(0,20),(20,0),(20,21),(21,20),(4,21),(21,4)]','[(2,8),(8,2),(4,8),(8,4),(21,8),(8,21),(2,7),(7,2),(4,7),(7,4),(21,7),(7,21),(8,7),(7,8)]')
(H/'run.py').write_text(t)
print('Wrote shared bitmap modes8/7, preserved modes2/4/21 and existing fixture references')
