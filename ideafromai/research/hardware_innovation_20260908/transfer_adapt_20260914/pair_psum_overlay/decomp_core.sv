module decomp_core(
 input logic clk,reset_n,cfg_valid,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic start,input logic [4:0] mode,input logic source_allow,weight_allow,
 output logic result_valid,input logic result_ready,
 output logic [8:0] result_addr,output logic [255:0] result_data,output logic done,
 output logic [31:0] cycles,source_words,weight_words,second_weight_words,local_source_reads,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,first_issues,dual_updates,
 output logic [31:0] psum_reads,psum_writes,mac_issues,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [31:0] aux_reads,aux_writes,aux_issues,aux_weight_words,aux_events,
 output logic [31:0] metadata_reads,count_checks,count_bank_reads,count_bank_writes,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,L_START,L_LOAD,L_GATHER,CHECK,QREAD,TIMESEL,
 ZREAD,ZADD,KNEXT,ZSCAN,VLOAD,POSLOAD,BASE_MAC,STORE,DRAIN_READ,DRAIN_SEND,FINISH,MREAD,DCLEAR,C_CHECK,C_READ,C_ADD,G_START,G_QREAD,G_READ,G_SELECT,G_ZREAD,G_MAC} state_t;
 state_t state;logic [4:0] mode_q;
 logic [5:0] k_class[0:3][0:863],class_hold[0:3],ngroups;
 logic signed [2:0] representative[0:7][0:31];
 logic [31:0] count_hold[0:7];
 logic [8:0] p_addr[0:7];logic [31:0] p_read_data[0:7],p_write_data[0:7];
 logic [7:0] p_read_enable,p_write_enable;
 logic [31:0] group_live;
 logic [159:0] count_live[0:3];
 logic [4:0] block_pending,source_blocks,retire_blocks;
 logic [4:0] block_after,retire_after;
 integer next_block,next_retire,read_banks;
 integer cblock,group_iter,gselected,gtarget,active_groups;
 logic [7:0] group_pending;logic [9:0] count_scalar[0:3];
 logic [7:0] gp_encoded;logic [31:0] count_increment[0:7];
 logic packed_retire,packed_safe;
 logic [9:0] count_low[0:3],count_high[0:3];
 logic signed [12:0] multiply_scalar[0:7];
 logic [7:0] correction_carry;
 logic ghalf,has_direct,has_count,count_active;
 logic [9:0] source_mem[0:1535],local_source[0:15],src_masks[0:3];
 logic signed [2:0] q_mem[0:7][0:863],q_hold[0:7];
 logic signed [15:0] v_mem[0:7][0:95],qblock[0:7][0:7];
 logic k_live[0:863],v_live[0:95];
 logic [25:0] z_mem[0:7][0:19],z_hold[0:7];
 logic signed [31:0] p_mem[0:7][0:479],acc[0:7];
 logic [7:0] position_live[0:39],rank_live,block_live,remaining,scan_mask;
 logic [39:0] pending;logic signed [15:0] oy,ox;
 integer k,fp,zrow,og,qfill,row,load_xy,selected_time;
 logic [10:0] source_addr;logic [4:0] read_zrow;
 logic in_bounds,cfg_v_live,active_lo,active_hi,read_half;
 logic [2:0] selected_rank;
 logic signed [12:0] scalar;
 logic signed [18:0] multiply_coefficient[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7],add_y[0:7];

 task clear_counters;
 begin
 aux_reads<=0;aux_writes<=0;aux_issues<=0;aux_weight_words<=0;aux_events<=0;
 metadata_reads<=0;count_checks<=0;count_bank_reads<=0;count_bank_writes<=0;
 cycles<=0;source_words<=0;weight_words<=0;second_weight_words<=0;local_source_reads<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;first_issues<=0;dual_updates<=0;
 psum_reads<=0;psum_writes<=0;mac_issues<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 end
 endtask

 always_comb begin
 source_addr=11'((k/9)*16+load_xy);
 in_bounds=(int'(oy)+load_xy/4>=0 && int'(oy)+load_xy/4<240 &&
 int'(ox)+load_xy%4>=0 && int'(ox)+load_xy%4<320);
 has_direct=0;has_count=0;active_groups=0;
 for(integer g=0;g<4;g=g+1)begin
  has_direct=has_direct||(k_class[g][k]==63);
  has_count=has_count||(k_class[g][k]>0&&k_class[g][k]<=32);
  if(class_hold[g]>0&&class_hold[g]<=32)active_groups=active_groups+1;
 end
 source_blocks=0;retire_blocks=0;read_banks=0;
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
 cfg_v_live=0;for(integer i=0;i<8;i=i+1)cfg_v_live=cfg_v_live||(cfg_data[i*32+:16]!=0);
 selected_time=0;for(integer i=39;i>=0;i=i-1)if(pending[i])selected_time=i;
 selected_rank=0;for(integer i=7;i>=0;i=i-1)if(remaining[i])selected_rank=3'(i);
 read_zrow=5'((fp/20)*10+fp%10);read_half=((fp/10)%2)!=0;
 scan_mask=0;
 if(state==ZSCAN)for(integer i=0;i<8;i=i+1)
 scan_mask[i]=read_half?(z_mem[i][read_zrow][25:13]!=0):(z_mem[i][read_zrow][12:0]!=0);
 scalar=read_half?$signed(z_mem[selected_rank][read_zrow][25:13]):$signed(z_mem[selected_rank][read_zrow][12:0]);
 correction_carry=0;
 for(integer l=0;l<8;l=l+1)begin
 multiply_coefficient[l]={{3{qblock[selected_rank][l][15]}},qblock[selected_rank][l]};
 if(state==G_MAC)multiply_coefficient[l]={{16{q_hold[l][2]}},q_hold[l]};
 multiply_scalar[l]=(state==G_MAC)?$signed({3'd0,count_scalar[l/2]}):scalar;
 if(state==G_MAC&&packed_retire)begin
 multiply_coefficient[l]=$signed({1'b0,count_high[l/2][6:0],3'd0,count_low[l/2][7:0]});
 multiply_scalar[l]={{10{q_hold[l][2]}},q_hold[l]};
 correction_carry[l]=q_hold[l][2]&&(count_low[l/2]!=0);
 end
 product[l]=$signed(multiply_coefficient[l])*$signed(multiply_scalar[l]);
 lhs[l]=acc[l];rhs[l]=product[l];
 if(state==C_ADD)begin
 lhs[l]=count_hold[l];rhs[l]=count_increment[l];
 end else if(state==G_MAC)begin
 if(packed_retire)begin
 lhs[l]={6'd0,z_hold[l]};
 rhs[l]={6'd0,product[l][23:11],{2{product[l][10]}},product[l][10:0]};
 end else lhs[l]=ghalf?{{19{z_hold[l][25]}},z_hold[l][25:13]}:{{19{z_hold[l][12]}},z_hold[l][12:0]};
 end
 if(state==ZADD)begin
 lhs[l]={6'b0,z_hold[l]};
 rhs[l]={6'b0,(active_hi?{{10{q_hold[l][2]}},q_hold[l]}:13'd0),
                    (active_lo?{{10{q_hold[l][2]}},q_hold[l]}:13'd0)};
 end
 end
 result_valid=(state==DRAIN_SEND);result_addr=9'(row);debug_state=state;
 end
 genvar l,b;
 generate for(l=0;l<8;l=l+1)begin:ALU
 for(b=0;b<32;b=b+1)begin:BIT
 wire cin;
 if(b==0)assign cin=1'b0;
 else if(b==8||b==16||b==24)assign cin=(state==C_ADD)?1'b0:BIT[b-1].CARRY.cout;
 else if(b==13)assign cin=(state==ZADD)?1'b0:((state==G_MAC&&packed_retire)?correction_carry[l]:BIT[b-1].CARRY.cout);
 else assign cin=BIT[b-1].CARRY.cout;
 assign add_y[l][b]=lhs[l][b]^rhs[l][b]^cin;
 if(b<31)begin:CARRY
 wire cout;assign cout=(lhs[l][b]&rhs[l][b])|((lhs[l][b]^rhs[l][b])&cin);
 end
 end
 end endgenerate

 always_ff @(posedge clk)begin
 if(!reset_n)begin
 block_pending<=0;for(integer g=0;g<4;g=g+1)count_live[g]<=0;
 count_active<=0;cblock<=0;group_iter<=0;group_pending<=0;group_live<=0;ngroups<=0;
 state<=IDLE;mode_q<=14;done<=0;oy<=0;ox<=0;k<=0;fp<=0;zrow<=0;og<=0;qfill<=0;row<=0;load_xy<=0;
 rank_live<=0;block_live<=0;remaining<=0;pending<=0;result_data<=0;active_lo<=0;active_hi<=0;clear_counters();
 for(integer i=0;i<4;i=i+1)begin src_masks[i]<=0;class_hold[i]<=0;end
 for(integer i=0;i<8;i=i+1)begin q_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;count_hold[i]<=0;end
 end else begin
 done<=0;
 for(integer i=0;i<8;i=i+1)if(p_write_enable[i])p_mem[i][p_addr[i]]<=p_write_data[i];
 if(cfg_valid&&state==IDLE)case(cfg_kind)
 0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
 3:begin oy<=cfg_data[15:0];ox<=cfg_data[31:16];end
 4:for(integer i=0;i<8;i=i+1)q_mem[i][cfg_addr[9:0]]<=cfg_data[i*32+:3];
 5:begin
 for(integer i=0;i<8;i=i+1)v_mem[i][cfg_addr[6:0]]<=cfg_data[i*32+:16];
 v_live[cfg_addr[6:0]]<=cfg_v_live;
 end
 6:k_live[cfg_addr[9:0]]<=cfg_data[0];
 7:if(cfg_addr<32)begin
 for(integer i=0;i<8;i=i+1)representative[i][cfg_addr[4:0]]<=cfg_data[i*32+:3];
 end else if(cfg_addr==1023)ngroups<=cfg_data[5:0];
 else for(integer g=0;g<4;g=g+1)k_class[g][10'(cfg_addr-11'd32)]<=cfg_data[g*6+:6];
 default:begin end
 endcase
 if(state!=IDLE)cycles<=cycles+1;
 case(state)
 IDLE:if(start)begin
 mode_q<=mode;block_pending<=0;for(integer g=0;g<4;g=g+1)count_live[g]<=0;zrow<=0;rank_live<=0;block_live<=0;group_live<=0;state<=ZCLEAR;clear_counters();
 end
 ZCLEAR:begin
 for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=0;
 z_writes<=z_writes+1;
 if(zrow==19)begin k<=0;state<=L_START;end else zrow<=zrow+1;
 end
 L_START:begin load_xy<=0;state<=L_LOAD;end
 L_LOAD:if(!in_bounds||source_allow)begin
 local_source[load_xy]<=in_bounds?source_mem[source_addr]:10'd0;
 if(in_bounds)source_words<=source_words+1;
 if(load_xy==15)state<=L_GATHER;else load_xy<=load_xy+1;
 end else source_stalls<=source_stalls+1;
 L_GATHER:if(!k_live[k])state<=KNEXT;
 else begin
 for(integer i=0;i<4;i=i+1)src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];
 local_source_reads<=local_source_reads+1;state<=CHECK;
 end
 CHECK:begin
 pending<={20'd0,(src_masks[2]|src_masks[3]),(src_masks[0]|src_masks[1])};
 if((src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3])==0)state<=KNEXT;
 else state<=(mode_q==20)?MREAD:QREAD;
 end
 MREAD:begin
 for(integer g=0;g<4;g=g+1)begin
  class_hold[g]<=k_class[g][k];
  if(k_class[g][k]>0&&k_class[g][k]<=32)group_live[int'(k_class[g][k])-1]<=1;
 end
 metadata_reads<=metadata_reads+1;count_active<=has_count;cblock<=0;block_pending<=source_blocks;
 for(integer i=4;i>=0;i=i-1)if(source_blocks[i])cblock<=i;
 state<=has_direct?QREAD:(has_count?C_CHECK:KNEXT);
 end
 QREAD:if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)q_hold[i]<=(mode_q==14||class_hold[i/2]==63)?q_mem[i][k]:3'sd0;
 weight_words<=weight_words+1;state<=TIMESEL;
 end else weight_stalls<=weight_stalls+1;
 TIMESEL:if(pending!=0)begin
 begin
 zrow<=selected_time;
 active_lo<=src_masks[(selected_time/10)*2][selected_time%10];
 active_hi<=src_masks[(selected_time/10)*2+1][selected_time%10];
 end
 state<=ZREAD;
 end else if(mode_q==20&&count_active)begin
 cblock<=0;for(integer i=4;i>=0;i=i-1)if(block_pending[i])cblock<=i;
 state<=C_CHECK;end
 else state<=KNEXT;
 ZREAD:begin
 for(integer i=0;i<8;i=i+1)z_hold[i]<=z_mem[i][zrow];
 z_vector_reads<=z_vector_reads+1;state<=ZADD;
 end
 ZADD:begin
 for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=add_y[i][25:0];
 first_issues<=first_issues+1;z_writes<=z_writes+1;
 if(active_lo&&active_hi)dual_updates<=dual_updates+1;
 pending<=pending&(pending-40'd1);state<=TIMESEL;
 end
 KNEXT:if(k==863)begin fp<=0;group_iter<=0;state<=(mode_q==20&&ngroups!=0)?G_START:ZSCAN;end
 else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end
 C_CHECK:begin
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
 G_START:if(group_iter>=int'(ngroups))begin fp<=0;state<=ZSCAN;end
 else if(!group_live[group_iter])group_iter<=group_iter+1;
 else begin cblock<=0;
 for(integer i=4;i>=0;i=i-1)if(retire_blocks[i])cblock<=i;
 state<=G_QREAD;end
 G_QREAD:if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)q_hold[i]<=representative[i][group_iter];
 weight_words<=weight_words+1;aux_weight_words<=aux_weight_words+1;state<=G_READ;
 end else weight_stalls<=weight_stalls+1;
 G_READ:begin
 for(integer i=0;i<8;i=i+1)count_hold[i]<=p_read_data[i];
 group_pending<=gp_encoded;aux_reads<=aux_reads+1;count_bank_reads<=count_bank_reads+32'(read_banks);state<=G_SELECT;
 end
 G_SELECT:if(group_pending==0)begin
 if(retire_after==0)begin group_iter<=group_iter+1;state<=G_START;end
 else begin cblock<=next_retire;state<=G_READ;end
 end else begin fp<=gtarget;state<=G_ZREAD;end
 G_ZREAD:begin
 for(integer i=0;i<8;i=i+1)z_hold[i]<=z_mem[i][read_zrow];
 z_vector_reads<=z_vector_reads+1;state<=G_MAC;
 end
 G_MAC:begin
 for(integer i=0;i<8;i=i+1)
 if(packed_retire)z_mem[i][read_zrow]<=add_y[i][25:0];
 else if(ghalf)z_mem[i][read_zrow][25:13]<=add_y[i][12:0];else z_mem[i][read_zrow][12:0]<=add_y[i][12:0];
 z_writes<=z_writes+1;first_issues<=first_issues+1;aux_events<=aux_events+1;
 if(packed_retire)group_pending<=group_pending&~(8'd3<<((gselected/2)*2));
 else group_pending<=group_pending&(group_pending-8'd1);state<=G_SELECT;
 end
 ZSCAN:begin
 z_vector_reads<=z_vector_reads+1;position_live[fp]<=scan_mask;rank_live<=rank_live|scan_mask;
 if(fp==39)begin og<=0;qfill<=0;state<=VLOAD;end else fp<=fp+1;
 end
 VLOAD:begin
 if(!rank_live[qfill]||!v_live[og*8+qfill]||weight_allow)begin
 block_live[qfill]<=rank_live[qfill]&&v_live[og*8+qfill];
 for(integer i=0;i<8;i=i+1)qblock[qfill][i]<=(rank_live[qfill]&&v_live[og*8+qfill])?v_mem[i][og*8+qfill]:16'sd0;
 if(rank_live[qfill]&&v_live[og*8+qfill])begin weight_words<=weight_words+1;second_weight_words<=second_weight_words+1;end
 if(qfill==7)begin fp<=0;state<=POSLOAD;end else qfill<=qfill+1;
 end else weight_stalls<=weight_stalls+1;
 end
 POSLOAD:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=0;
 remaining<=position_live[fp]&block_live;
 state<=((position_live[fp]&block_live)==0)?STORE:BASE_MAC;
 end
 BASE_MAC:begin
 for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
 mac_issues<=mac_issues+1;z_scalar_reads<=z_scalar_reads+1;
 remaining<=remaining&(remaining-8'd1);
 if((remaining&(remaining-8'd1))==0)state<=STORE;
 end
 STORE:begin
 psum_writes<=psum_writes+1;
 if(fp<39)begin fp<=fp+1;state<=POSLOAD;end
 else if(og<11)begin og<=og+1;qfill<=0;state<=VLOAD;end
 else begin row<=0;state<=DRAIN_READ;end
 end
 DRAIN_READ:begin
 for(integer i=0;i<8;i=i+1)result_data[i*32+:32]<=p_read_data[i];
 psum_reads<=psum_reads+1;state<=DRAIN_SEND;
 end
 DRAIN_SEND:if(result_ready)begin
 if(row==479)state<=FINISH;else begin row<=row+1;state<=DRAIN_READ;end
 end else output_stalls<=output_stalls+1;
 FINISH:begin done<=1;state<=IDLE;end
 default:state<=IDLE;
 endcase
 end
 end
`ifdef VERILATOR
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
endmodule
