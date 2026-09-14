from pathlib import Path

H=Path(__file__).resolve().parent
B=H.parents[1]
OLD=B/'fusion_ten_trials_20260914/decompositions/q1_bitplanes'
s=(OLD/'decomp_core.sv').read_text()
s=s.replace('BM_POP,BM_STORE} state_t;', 'BM_POP,BM_STORE,BM_PIPE} state_t;')
s=s.replace('integer bm_block,bm_plane;', '''integer bm_block,bm_plane;
 logic [53:0] bm_live[0:39],bm_pending,bm_after;
 integer first_live,next_live;
 logic pipe_valid;logic [1:0] pipe_plane,execute_plane;
 logic pop_execute;
''')
s=s.replace('if(bm_block==53)state<=BM_STORE;else begin bm_block<=bm_block+1;state<=BM_SCAN;end', '''if(mode_q==13)begin
  bm_pending<=bm_after;
  if(bm_after==0)state<=BM_STORE;
  else begin bm_block<=next_live;state<=BM_SCAN;end
 end else if(bm_block==53)state<=BM_STORE;else begin bm_block<=bm_block+1;state<=BM_SCAN;end''')
s=s.replace('sub_alu=(state==BM_POP&&bm_plane==2);', '''pop_execute=(state==BM_POP)||(state==BM_PIPE&&pipe_valid);
 execute_plane=(state==BM_PIPE)?pipe_plane:2'(bm_plane);
 sub_alu=pop_execute&&execute_plane==2;
 bm_after=bm_pending&(bm_pending-54'd1);first_live=0;next_live=0;
 for(integer j=53;j>=0;j=j-1)begin
  if(bm_live[fp][j])first_live=j;
  if(bm_after[j])next_live=j;
 end''')
s=s.replace('if(state==BM_POP)begin lhs[l]=bm_acc[l];rhs[l]=$signed({27\'d0,pc[l]})<<<bm_plane;end', "if(pop_execute)begin lhs[l]=bm_acc[l];rhs[l]=$signed({27'd0,pc[l]})<<<execute_plane;end")
s=s.replace('bm_block<=0;bm_plane<=0;bm_hold<=0;', 'bm_block<=0;bm_plane<=0;bm_hold<=0;bm_pending<=0;pipe_valid<=0;pipe_plane<=0;')
old=''' L_GATHER:if(mode_q==15)begin
 for(integer i=0;i<4;i=i+1)src_masks[i]<=k_live[k]?local_source[(i/2+(k%9)/3)*4+i%2+k%3]:10'd0;
 if(k_live[k])local_source_reads<=local_source_reads+1;
 state<=BM_PACK;
 end else if(!k_live[k])state<=KNEXT;'''
new=''' L_GATHER:if(mode_q!=14)begin
 for(integer i=0;i<4;i=i+1)src_masks[i]<=k_live[k]?local_source[(i/2+(k%9)/3)*4+i%2+k%3]:10'd0;
 if(k_live[k])local_source_reads<=local_source_reads+1;
 if(mode_q==15)state<=BM_PACK;
 else begin
  // Use the original forty bit-write banks in the native gather cycle.
  for(integer i=0;i<40;i=i+1)begin
   bitmap[i][k/16][k%16]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
   if(mode_q==13)begin
    if(k%16==0)bm_live[i][k/16]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
    else bm_live[i][k/16]<=bm_live[i][k/16]||(k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10]);
   end
  end
  aux_writes<=aux_writes+1;state<=KNEXT;
 end
 end else if(!k_live[k])state<=KNEXT;'''
assert old in s
s=s.replace(old,new)
s=s.replace('bm_block<=0;state<=BM_SCAN;', '''if(mode_q==13)begin
 bm_pending<=bm_live[fp];bm_block<=first_live;state<=(bm_live[fp]==0)?BM_STORE:BM_SCAN;
 end else begin bm_block<=0;state<=BM_SCAN;end''')
s=s.replace('else begin bm_plane<=0;state<=BM_QREAD;end', 'else begin bm_plane<=0;pipe_valid<=0;state<=(mode_q==12||mode_q==13)?BM_PIPE:BM_QREAD;end')
needle=' BM_STORE:begin'
pipe=''' BM_PIPE:begin
  // One plane read can overlap one older pop/ALU issue. The original
  // bpq_hold is a single response register; no second memory read is added.
  if(pipe_valid)begin
   for(integer i=0;i<8;i=i+1)bm_acc[i]<=add_y[i];
   first_issues<=first_issues+1;aux_issues<=aux_issues+1;
  end
  pipe_valid<=0;
  if(bm_plane<3)begin
   if(!bp_live[bm_plane][bm_block])bm_plane<=bm_plane+1;
   else if(weight_allow)begin
    for(integer i=0;i<8;i=i+1)bpq_hold[i]<=bp_q[bm_plane][i][bm_block];
    pipe_plane<=2'(bm_plane);pipe_valid<=1;bm_plane<=bm_plane+1;
    weight_words<=weight_words+1;aux_weight_words<=aux_weight_words+1;
    if(pipe_valid)aux_events<=aux_events+1;
   end else weight_stalls<=weight_stalls+1;
  end else bm_next();
 end
'''
assert needle in s
s=s.replace(needle,pipe+needle)
s=s.replace('state<=(mode_q==15)?BM_POS:ZSCAN;', 'state<=(mode_q!=14)?BM_POS:ZSCAN;')
(H/'decomp_core.sv').write_text(s)
# Keep the raw harness counters identical to the old implementation.
(H/'tb.cpp').write_text((OLD/'tb.cpp').read_text())

# Second adaptation: one-hot K16 words are already a direct signed3 vector.
s=(H/'decomp_core.sv').read_text()
s=s.replace('output logic [5:0] debug_state','output logic [31:0] bitmap_native_reads,bitmap_native_issues,\n output logic [5:0] debug_state')
s=s.replace('BM_STORE,BM_PIPE} state_t;', 'BM_STORE,BM_PIPE,BM_NATIVE_QREAD,BM_NATIVE_ADD} state_t;')
s=s.replace('logic pop_execute;', 'logic pop_execute;integer onehot_index;logic bitmap_onehot;')
s=s.replace('mode_q==13', '(mode_q==13||mode_q==10)')
s=s.replace("pop_execute=(state==BM_POP)", "bitmap_onehot=(bitmap[fp][bm_block]!=0)&&((bitmap[fp][bm_block]&(bitmap[fp][bm_block]-16'd1))==0);\n onehot_index=0;for(integer j=15;j>=0;j=j-1)if(bm_hold[j])onehot_index=j;\n pop_execute=(state==BM_POP)")
s=s.replace('if(state==ZADD)begin\n lhs[l]', "if(state==BM_NATIVE_ADD)begin lhs[l]=bm_acc[l];rhs[l]={{29{q_hold[l][2]}},q_hold[l]};end\n if(state==ZADD)begin\n lhs[l]")
s=s.replace('aux_reads<=0;aux_writes<=0;', 'bitmap_native_reads<=0;bitmap_native_issues<=0;aux_reads<=0;aux_writes<=0;')
s=s.replace('state<=(mode_q==12||(mode_q==13||mode_q==10))?BM_PIPE:BM_QREAD;', 'state<=(mode_q==10&&bitmap_onehot)?BM_NATIVE_QREAD:((mode_q==12||(mode_q==13||mode_q==10))?BM_PIPE:BM_QREAD);')
s=s.replace(' BM_STORE:begin', ''' BM_NATIVE_QREAD:if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)q_hold[i]<=q_mem[i][bm_block*16+onehot_index];
 weight_words<=weight_words+1;bitmap_native_reads<=bitmap_native_reads+1;state<=BM_NATIVE_ADD;
 end else weight_stalls<=weight_stalls+1;
 BM_NATIVE_ADD:begin
 for(integer i=0;i<8;i=i+1)bm_acc[i]<=add_y[i];
 first_issues<=first_issues+1;bitmap_native_issues<=bitmap_native_issues+1;bm_next();
 end
 BM_STORE:begin''')
(H/'decomp_core.sv').write_text(s)
s=(H/'tb.cpp').read_text().replace('<<",\\"state_cycles\\":[";', '<<",\\"bitmap_native_reads\\":"<<d.bitmap_native_reads<<",\\"bitmap_native_issues\\":"<<d.bitmap_native_issues<<",\\"state_cycles\\":[";')
(H/'tb.cpp').write_text(s)

# Make mutually exclusive fetch paths one explicit physical read expression.
s=(H/'decomp_core.sv').read_text()
s=s.replace('logic bitmap_onehot;', '''logic bitmap_onehot;
 logic [9:0] q_read_addr;logic q_read_enable,bp_read_enable;
 logic signed [2:0] q_read_data[0:7];logic [15:0] bp_read_data[0:7],bitmap_read_word;''')
s=s.replace("bitmap_onehot=(bitmap[fp][bm_block]!=0)&&((bitmap[fp][bm_block]&(bitmap[fp][bm_block]-16'd1))==0);", '''q_read_enable=(state==QREAD||state==BM_NATIVE_QREAD);
 q_read_addr=(state==BM_NATIVE_QREAD)?10'(bm_block*16+onehot_index):10'(k);
 bp_read_enable=(state==BM_QREAD||state==BM_PIPE)&&(bm_plane<3);
 bitmap_read_word=(state==BM_SCAN)?bitmap[fp][bm_block]:16'd0;
 bitmap_onehot=(bitmap_read_word!=0)&&((bitmap_read_word&(bitmap_read_word-16'd1))==0);''')
# onehot_index must be derived before its use in the common read address.
line='onehot_index=0;for(integer j=15;j>=0;j=j-1)if(bm_hold[j])onehot_index=j;'
s=s.replace(line,'')
s=s.replace('q_read_enable=(state==QREAD||state==BM_NATIVE_QREAD);',line+'\n q_read_enable=(state==QREAD||state==BM_NATIVE_QREAD);')
s=s.replace("for(integer l=0;l<8;l=l+1)begin\n multiply_coefficient", "for(integer l=0;l<8;l=l+1)begin\n q_read_data[l]=q_read_enable?q_mem[l][q_read_addr]:3'sd0;\n bp_read_data[l]=bp_read_enable?bp_q[bm_plane][l][bm_block]:16'd0;\n multiply_coefficient")
s=s.replace('bm_hold<=bitmap[fp][bm_block];','bm_hold<=bitmap_read_word;').replace('if(bitmap[fp][bm_block]==0)','if(bitmap_read_word==0)')
s=s.replace('bpq_hold[i]<=bp_q[bm_plane][i][bm_block];','bpq_hold[i]<=bp_read_data[i];')
s=s.replace('q_hold[i]<=q_mem[i][bm_block*16+onehot_index];','q_hold[i]<=q_read_data[i];').replace('q_hold[i]<=q_mem[i][k];','q_hold[i]<=q_read_data[i];')
(H/'decomp_core.sv').write_text(s)
