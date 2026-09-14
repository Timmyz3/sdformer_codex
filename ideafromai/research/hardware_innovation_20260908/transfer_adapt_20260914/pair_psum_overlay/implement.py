from pathlib import Path
H=Path(__file__).resolve().parent
S=H.parent/'pair_sparse'
s=(S/'decomp_core.sv').read_text()
s=s.replace('logic [19:0] count_mem[0:7][0:319],count_hold[0:7];\n logic [8:0] count_rd_addr[0:7];logic [19:0] count_rd_data[0:7];logic [7:0] count_rd_enable;', '''logic [31:0] count_hold[0:7];
 logic [8:0] p_addr[0:7];logic [31:0] p_read_data[0:7],p_write_data[0:7];
 logic [7:0] p_read_enable,p_write_enable;''')
s=s.replace('logic [319:0] count_live[0:3];','logic [159:0] count_live[0:3];')
s=s.replace('logic [9:0] block_pending,source_blocks,retire_blocks;','logic [4:0] block_pending,source_blocks,retire_blocks;')
s=s.replace('logic [9:0] block_after,retire_after;','logic [4:0] block_after,retire_after;')
s=s.replace('integer next_block,next_retire,valid_groups,read_banks;','integer next_block,next_retire,read_banks;')
s=s.replace('integer cblock,cclear,group_iter,gselected,gpair,gtarget,active_groups;','integer cblock,group_iter,gselected,gtarget,active_groups;')
s=s.replace('logic [3:0] group_pending;logic [9:0] count_scalar[0:3];','logic [7:0] group_pending;logic [9:0] count_scalar[0:3];')
s=s.replace('logic [1:0] clo,chi;logic [3:0] gp_encoded;','logic [7:0] gp_encoded;logic [31:0] count_increment[0:7];')
a=s.index(' source_blocks=0;');b=s.index(' cfg_v_live=',a)
s=s[:a]+''' source_blocks=0;retire_blocks=0;read_banks=0;
 for(integer t=0;t<5;t=t+1)begin
  for(integer j=0;j<2;j=j+1)for(integer p=0;p<4;p=p+1)
   source_blocks[t]=source_blocks[t]||src_masks[p][t*2+j];
  for(integer g=0;g<4;g=g+1)
   if(group_iter<32)retire_blocks[t]=retire_blocks[t]||count_live[g][group_iter*5+t];
 end
 block_after=block_pending&~(5'd1<<cblock);
 retire_after=retire_blocks&~((5'd1<<(cblock+1))-5'd1);
 next_block=0;next_retire=0;
 for(integer t=4;t>=0;t=t-1)begin
  if(block_after[t])next_block=t;
  if(retire_after[t])next_retire=t;
 end
 // One physical p_mem address and at most one read OR write per bank.
 // Count phase uses four independent class addresses; output phase broadcasts its row.
 for(integer i=0;i<8;i=i+1)begin
  count_increment[i]=0;
  for(integer p=0;p<4;p=p+1)count_increment[i][p*8]=src_masks[p][cblock*2+i%2];
  p_addr[i]=0;p_read_enable[i]=0;p_write_enable[i]=0;p_write_data[i]=0;
  if(state==C_CHECK&&class_hold[i/2]>0&&class_hold[i/2]<=32)begin
   p_read_enable[i]=count_live[i/2][(int'(class_hold[i/2])-1)*5+cblock];
   p_addr[i]=9'((int'(class_hold[i/2])-1)*5+cblock);
  end else if(state==G_READ)begin
   p_read_enable[i]=count_live[i/2][group_iter*5+cblock];p_addr[i]=9'(group_iter*5+cblock);
  end else if(state==C_ADD&&class_hold[i/2]>0&&class_hold[i/2]<=32)begin
   p_write_enable[i]=1;p_addr[i]=9'((int'(class_hold[i/2])-1)*5+cblock);p_write_data[i]=add_y[i];
  end else if(state==STORE)begin
   p_write_enable[i]=1;p_addr[i]=9'(og*40+fp);p_write_data[i]=acc[i];
  end else if(state==DRAIN_READ)begin
   p_read_enable[i]=1;p_addr[i]=9'(row);
  end
  p_read_data[i]=p_read_enable[i]?p_mem[i][p_addr[i]]:32'd0;
  if((state==C_CHECK||state==G_READ)&&p_read_enable[i])read_banks=read_banks+1;
 end
 gp_encoded=0;
 for(integer g=0;g<4;g=g+1)for(integer j=0;j<2;j=j+1)for(integer p=0;p<4;p=p+1)
  gp_encoded[j*4+p]=gp_encoded[j*4+p]||(p_read_data[2*g+j][p*8+:8]!=0);
 gselected=0;for(integer i=7;i>=0;i=i-1)if(group_pending[i])gselected=i;
 gtarget=(gselected%4)*10+cblock*2+gselected/4;ghalf=(gselected%2)!=0;
 packed_safe=1;
 for(integer g=0;g<4;g=g+1)begin
  count_low[g]={2'd0,count_hold[2*g+gselected/4][((gselected/2)%2)*16+:8]};
  count_high[g]={2'd0,count_hold[2*g+gselected/4][((gselected/2)%2)*16+8+:8]};
  if(count_high[g]>127)packed_safe=0;
 end
 packed_retire=(mode_q==20)&&packed_safe;
 for(integer g=0;g<4;g=g+1)count_scalar[g]=ghalf?count_high[g]:count_low[g];
'''+s[b:]
s=s.replace("lhs[l]={12'd0,count_hold[l]};rhs[l]={12'd0,9'd0,chi[l%2],9'd0,clo[l%2]};",'lhs[l]=count_hold[l];rhs[l]=count_increment[l];')
s=s.replace("else if(b==10)assign cin=(state==C_ADD)?1'b0:BIT[b-1].CARRY.cout;", "else if(b==8||b==16||b==24)assign cin=(state==C_ADD)?1'b0:BIT[b-1].CARRY.cout;")
s=s.replace('cblock<=0;cclear<=0;','cblock<=0;').replace('group_live<=0;cclear<=0;','group_live<=0;')
s=s.replace('if(zrow==19)begin k<=0;state<=(mode_q>=15&&mode_q<17&&ngroups!=0)?DCLEAR:L_START;end','if(zrow==19)begin k<=0;state<=L_START;end')
s=s.replace('for(integer i=9;i>=0;i=i-1)if(source_blocks[i])','for(integer i=4;i>=0;i=i-1)if(source_blocks[i])')
s=s.replace('for(integer i=9;i>=0;i=i-1)if(block_pending[i])','for(integer i=4;i>=0;i=i-1)if(block_pending[i])')
s=s.replace('for(integer i=9;i>=0;i=i-1)if(retire_blocks[i])','for(integer i=4;i>=0;i=i-1)if(retire_blocks[i])')
a=s.index(' DCLEAR:begin');b=s.index(' G_START:',a)
s=s[:a]+''' C_CHECK:begin
 count_checks<=count_checks+1;
 for(integer i=0;i<8;i=i+1)count_hold[i]<=p_read_data[i];
 if(read_banks!=0)aux_reads<=aux_reads+1;
 count_bank_reads<=count_bank_reads+32'(read_banks);state<=C_ADD;
 end
 C_ADD:begin
 for(integer g=0;g<4;g=g+1)if(class_hold[g]>0&&class_hold[g]<=32)
 count_live[g][(int'(class_hold[g])-1)*5+cblock]<=1;
 aux_writes<=aux_writes+1;aux_issues<=aux_issues+1;count_bank_writes<=count_bank_writes+32'(2*active_groups);
 block_pending<=block_after;
 if(block_after==0)state<=KNEXT;else begin cblock<=next_block;state<=C_CHECK;end
 end
'''+s[b:]
s=s.replace('count_hold[i]<=count_rd_data[i];','count_hold[i]<=p_read_data[i];')
s=s.replace("group_pending&~(4'd3<<((gselected/2)*2))","group_pending&~(8'd3<<((gselected/2)*2))").replace("group_pending-4'd1","group_pending-8'd1")
s=s.replace(' for(integer i=0;i<8;i=i+1)p_mem[i][og*40+fp]<=acc[i];\n','')
s=s.replace('result_data[i*32+:32]<=p_mem[i][row]','result_data[i*32+:32]<=p_read_data[i]')
s=s.replace('done<=0;\n if(cfg_valid', 'done<=0;\n for(integer i=0;i<8;i=i+1)if(p_write_enable[i])p_mem[i][p_addr[i]]<=p_write_data[i];\n if(cfg_valid')
# Eliminate legacy scheduling branches; only mode14 and20 are supported here.
s=s.replace('(mode_q>=15)','(mode_q==20)').replace('mode_q>=15&&','mode_q==20&&').replace('mode_q<15||','mode_q==14||')
s=s.replace('if(mode_q>=16)for','for').replace('if(mode_q>=17)for','for')
a=s.index(' G_SELECT:if(group_pending==0)begin');b=s.index(' G_ZREAD:',a)
s=s[:a]+''' G_SELECT:if(group_pending==0)begin
 if(retire_after==0)begin group_iter<=group_iter+1;state<=G_START;end
 else begin cblock<=next_retire;state<=G_READ;end
 end else begin fp<=gtarget;state<=G_ZREAD;end
'''+s[b:]
# Verification state is compiled only by Verilator, not part of candidate hardware.
s=s.replace('endmodule', '''`ifdef VERILATOR
 logic check_retired;integer check_stores;
 always_ff @(posedge clk)begin
 if(!reset_n)begin check_retired<=0;check_stores<=0;end
 else begin
  if(state==IDLE&&start)begin check_retired<=0;check_stores<=0;end
  if(state==ZSCAN)check_retired<=1;
  if(state==C_CHECK||state==C_ADD||state==G_READ)begin
   if(check_retired)$fatal(1,"count access after retirement");
   for(integer i=0;i<8;i=i+1)if((p_read_enable[i]||p_write_enable[i])&&p_addr[i]>=160)$fatal(1,"count address range");
  end
  if(state==STORE)begin
   if(!check_retired||og*40+fp!=check_stores)$fatal(1,"output overwrite ordering");
   check_stores<=check_stores+1;
  end
  if(state==DRAIN_READ&&check_stores!=480)$fatal(1,"drain before all outputs overwritten");
  for(integer i=0;i<8;i=i+1)begin
   if(p_read_enable[i]&&p_write_enable[i])$fatal(1,"psum bank read/write conflict");
   if((p_read_enable[i]||p_write_enable[i])&&p_addr[i]>=480)$fatal(1,"psum address range");
   if(state==C_ADD&&p_write_enable[i])for(integer p=0;p<4;p=p+1)
    if(count_hold[i][p*8+:8]==8'd255&&count_increment[i][p*8])$fatal(1,"count8 overflow");
  end
 end
 end
`endif
endmodule''')
assert 'count_mem' not in s
assert s.count('p_mem[i][p_addr[i]]')==2
(H/'decomp_core.sv').write_text(s)
for f in ['tb.cpp','stream_tb.cpp']:(H/f).write_text((S/f).read_text().replace('int main(int argc,char** argv)', 'double sc_time_stamp(){return 0;}\nint main(int argc,char** argv)'))
print('generated overlay core and unchanged handshake harnesses')
