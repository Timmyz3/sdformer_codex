module rr_context(
 input logic clk,reset_n,cfg_valid,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic start,input logic [4:0] mode,input logic source_allow,weight_allow,
 input logic range_ok,resource_grant,
 output logic fallback_used,compute_done,
 output logic [5:0] resource_request,
 output logic [511:0] borrow_lhs,borrow_rhs,input logic [415:0] borrow_y,
 output logic [2:0] weight_kind,output logic [9:0] weight_address,input logic [255:0] weight_data,
 output logic [255:0] alu_lhs,alu_rhs,output logic [151:0] alu_coefficient,
 output logic [103:0] alu_scalar,output logic [7:0] alu_correction,output logic alu_mac,output logic [2:0] alu_format,
 input logic [255:0] alu_result,
 input logic [161:0] plane_live,
 output logic alu_pop,output logic [15:0] pop_mask,output logic [127:0] pop_plane,output logic [1:0] pop_shift,
 output logic [31:0] bitmap_native_reads,bitmap_native_issues,cache_reads,cache_writes,bitmap_z_arbitration_stalls,bitmap_alu_arbitration_stalls,bitmap_weight_arbitration_stalls,bitmap_hold_writes,bitmap_hold_reads,bitmap_row_z_arbitration_stalls,
 input logic [39:0] time_permutation,input logic [5:0] ngroups,
 output logic [31:0] aux_reads,aux_writes,aux_issues,aux_weight_words,aux_events,metadata_reads,count_checks,count_bank_reads,count_bank_writes,
 output logic [31:0] arbitration_stalls,repair_arbitration_stalls,normalization_arbitration_stalls,
 output logic result_valid,input logic result_ready,
 output logic [8:0] result_addr,output logic [255:0] result_data,output logic done,
 output logic [31:0] cycles,source_words,weight_words,second_weight_words,local_source_reads,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,first_issues,merged_updates,
 output logic [31:0] repair_issues,repair_fields,normalization_issues,
 output logic [31:0] psum_reads,psum_writes,mac_issues,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,L_START,L_LOAD,L_GATHER,CHECK,QREAD,TIMESEL,
 ZREAD,ZADD,REPAIR_READ,REPAIR_ADD,KNEXT,NORMALIZE_READ,NORMALIZE_ADD,
 ZSCAN,VLOAD,POSLOAD,BASE_MAC,STORE,DRAIN_READ,DRAIN_SEND,FINISH,MREAD,C_CHECK,C_ADD,G_START,G_QREAD,G_READ,G_SELECT,G_ZREAD,G_MAC,BM_GROUP,BM_WFILL,BM_POS,BM_SCAN,BM_CACHED_POP,BM_NATIVE_QREAD,BM_NATIVE_ADD,BM_STORE} state_t;
 state_t state;logic [4:0] mode_q;wire count_mode=(mode_q==20||mode_q==21);
 wire bitmap_mode=(mode_q==8||mode_q==7);
 logic [15:0] bitmap[0:39],bm_hold,bitmap_read_word;
 logic [39:0] bitmap_live,bm_pending,bm_after;
 logic signed [31:0] bm_acc[0:7];logic [51:0] bm_write[0:7];
 integer bm_block,bm_plane,first_live,next_live,onehot_index;
 logic bitmap_onehot,bm_plane_live,bm_same_row;
 logic [4:0] qblock_read_index;logic qblock_read_enable;logic [127:0] qblock_read_bus;
 task advance_block;
 begin
 if(k==863)begin fp<=0;state<=ZSCAN;end
 else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end
 end endtask
 logic [9:0] source_mem[0:1535],local_source[0:15],src_masks[0:3];
 logic signed [2:0] q_hold[0:7];
 logic signed [15:0] qblock[0:23][0:7];
 logic k_live[0:863],v_live[0:95];
 logic [51:0] z_mem[0:7][0:39],z_hold[0:7];
 logic signed [31:0] p_mem[0:7][0:479],acc[0:7];
 logic signed [1:0] correction[0:7][0:3],next_correction[0:7][0:3];
 logic [7:0] position_live[0:39],rank_live,block_live,remaining,scan_mask;
 logic [39:0] pending;logic signed [15:0] oy,ox;
 integer k,fp,zrow,og,qfill,row,load_xy,selected_time;
 logic [10:0] source_addr;logic [5:0] read_zrow;
 logic in_bounds,cfg_v_live,pair_sel;logic [3:0] active;
 logic [2:0] selected_rank;logic signed [12:0] scalar;
 logic signed [18:0] multiply_coefficient[0:7];
 logic signed [31:0] lhs[0:7],rhs[0:7],add_y[0:7];
 logic [415:0] z_read_bus;
 logic [5:0] z_read_address;
 logic [5:0] class_hold[0:3];logic [3:0] inverse_time;logic [51:0] retire_write[0:7];
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

 logic [7:0] correction_carry;
 logic ghalf,has_direct,has_count,count_active;
 logic repair_needed;
 logic [5:0] correction_count;

 task clear_counters;
 begin
 bitmap_native_reads<=0; bitmap_native_issues<=0; cache_reads<=0; cache_writes<=0; bitmap_z_arbitration_stalls<=0; bitmap_alu_arbitration_stalls<=0; bitmap_weight_arbitration_stalls<=0; bitmap_hold_writes<=0; bitmap_hold_reads<=0; bitmap_row_z_arbitration_stalls<=0;
 aux_reads<=0;aux_writes<=0;aux_issues<=0;aux_weight_words<=0;aux_events<=0;metadata_reads<=0;count_checks<=0;count_bank_reads<=0;count_bank_writes<=0;
 cycles<=0;arbitration_stalls<=0;repair_arbitration_stalls<=0;normalization_arbitration_stalls<=0;source_words<=0;weight_words<=0;second_weight_words<=0;local_source_reads<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;first_issues<=0;merged_updates<=0;
 repair_issues<=0;repair_fields<=0;normalization_issues<=0;
 psum_reads<=0;psum_writes<=0;mac_issues<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 end
 endtask

 // Each bank has one explicit shared read address/expression. Scalar MAC reads
 // only its selected bank; scans and updates use the common 416-bit vector.
 always_comb begin
 source_addr=11'((k/9)*16+load_xy);
 in_bounds=(int'(oy)+load_xy/4>=0 && int'(oy)+load_xy/4<240 &&
 int'(ox)+load_xy%4>=0 && int'(ox)+load_xy%4<320);
 cfg_v_live=0;for(integer i=0;i<8;i=i+1)cfg_v_live=cfg_v_live||(cfg_data[i*32+:16]!=0);
 selected_time=0;for(integer i=39;i>=0;i=i-1)if(pending[i])selected_time=i;
 selected_rank=0;for(integer i=7;i>=0;i=i-1)if(remaining[i])selected_rank=3'(i);
 read_zrow=6'(fp%10);
 z_read_address=(state==ZSCAN || state==BASE_MAC || state==G_ZREAD || state==G_MAC || state==BM_POS || state==BM_STORE)?read_zrow:6'(zrow);
 resource_request=0;
 case(state)
  ZCLEAR,ZREAD,ZSCAN,REPAIR_READ,NORMALIZE_READ:resource_request=6'b000100;
  L_LOAD:if(in_bounds)resource_request=6'b000001;
  QREAD,MREAD,G_QREAD:resource_request=6'b000010;
  VLOAD:if(rank_live[qfill]&&v_live[og*8+qfill])resource_request=6'b000010;
  ZADD:resource_request=(mode_q==3)?6'b100100:6'b010100;
  REPAIR_ADD,NORMALIZE_ADD,BASE_MAC:resource_request=6'b010100;
  STORE,DRAIN_READ,G_READ:resource_request=6'b001000;
  C_CHECK:if(read_banks!=0)resource_request=6'b001000;
  C_ADD:resource_request=6'b011000;
  G_ZREAD:resource_request=6'b000100;
  G_MAC:resource_request=6'b010100;
  BM_POS:resource_request=6'b000100;
  BM_STORE:if(mode_q!=7||!bm_same_row)resource_request=6'b000100;
  BM_WFILL:if(bm_plane_live)resource_request=6'b000010;
  BM_CACHED_POP:if(bm_plane_live)resource_request=6'b010000;
  BM_NATIVE_QREAD:resource_request=6'b000010;
  BM_NATIVE_ADD:resource_request=6'b010000;
  default:begin end
 endcase
 weight_kind=(state==BM_WFILL)?3'd4:(state==VLOAD)?3'd1:(state==MREAD)?3'd2:(state==G_QREAD)?3'd3:3'd0;
 weight_address=(state==BM_WFILL)?10'(bm_plane*54+bm_block):(state==BM_NATIVE_QREAD)?10'(bm_block*16+onehot_index):(state==VLOAD)?10'(og*8+qfill):(state==G_QREAD)?10'(group_iter):10'(k);
 alu_mac=(state==BASE_MAC||state==G_MAC);alu_format=0;
 if(state==REPAIR_ADD || state==NORMALIZE_ADD)alu_format=4;
 else if(state==ZADD)alu_format=count_mode?3'd1:(mode_q[2:0]+3'd1);
 else if(state==C_ADD)alu_format=3;
 else if(state==G_MAC&&packed_retire)alu_format=5;
 else if(state==BM_CACHED_POP&&bm_plane==2)alu_format=6;
 end
 always_comb begin
 z_read_bus=0;
 for(integer i=0;i<8;i=i+1)
  if(resource_grant && (state==ZSCAN || state==ZREAD || state==REPAIR_READ || state==NORMALIZE_READ || state==G_ZREAD || state==BM_POS ||
     (state==BASE_MAC && selected_rank==3'(i))))
   z_read_bus[i*52+:52]=z_mem[i][z_read_address];
 scan_mask=0;
 if(state==ZSCAN)for(integer i=0;i<8;i=i+1)
  scan_mask[i]=(z_read_bus[i*52+13*(fp/10)+:13]!=0);
 scalar=(state==BASE_MAC)?$signed(z_read_bus[int'(selected_rank)*52+13*(fp/10)+:13]):13'sd0;
 alu_correction=correction_carry;
 for(integer l=0;l<8;l=l+1)begin
  borrow_lhs[l*64+:64]={12'd0,z_hold[l]};borrow_rhs[l*64+:64]=0;
  for(integer p=0;p<4;p=p+1)
   borrow_rhs[l*64+p*13+:13]=active[p]?{{10{q_hold[l][2]}},q_hold[l]}:13'd0;
  multiply_coefficient[l]={{3{qblock_read_bus[l*16+15]}},qblock_read_bus[l*16+:16]};
  lhs[l]=acc[l];rhs[l]=0;
  if(state==REPAIR_ADD || state==NORMALIZE_ADD)begin
   lhs[l]=0;rhs[l]=0;
   for(integer p=0;p<4;p=p+1)begin
    lhs[l][p*5+:5]=z_hold[l][p*13+8+:5];
    if(state==REPAIR_ADD)rhs[l][p*5+:5]={{3{correction[l][p][1]}},correction[l][p]};
    else rhs[l][p*5+:5]=z_hold[l][p*13+7]?5'b11111:5'd0;
   end
  end else if(state==ZADD)begin
   lhs[l]=0;rhs[l]=0;
   if(mode_q==0||count_mode)begin
    lhs[l]={6'b0,z_hold[l][26*int'(pair_sel)+:26]};
    rhs[l]={6'b0,(active[2*int'(pair_sel)+1]?{{10{q_hold[l][2]}},q_hold[l]}:13'd0),
                     (active[2*int'(pair_sel)]?{{10{q_hold[l][2]}},q_hold[l]}:13'd0)};
   end else if(mode_q==1)begin
    if(pair_sel)begin
     lhs[l][9:0]=z_hold[l][39+:10];
     rhs[l][9:0]=active[3]?{{7{q_hold[l][2]}},q_hold[l]}:10'd0;
    end else for(integer p=0;p<3;p=p+1)begin
     lhs[l][p*10+:10]=z_hold[l][p*13+:10];
     rhs[l][p*10+:10]=active[p]?{{7{q_hold[l][2]}},q_hold[l]}:10'd0;
    end
   end else for(integer p=0;p<4;p=p+1)begin
    lhs[l][p*8+:8]=z_hold[l][p*13+:8];
    rhs[l][p*8+:8]=active[p]?{{5{q_hold[l][2]}},q_hold[l]}:8'd0;
   end
  end
  alu_scalar[l*13+:13]=scalar;
  if(state==C_ADD)begin lhs[l]=count_hold[l];rhs[l]=count_increment[l];end
  if(state==G_MAC)begin
   multiply_coefficient[l]={{16{q_hold[l][2]}},q_hold[l]};
   alu_scalar[l*13+:13]={3'd0,count_scalar[l/2]};
   lhs[l]={{19{z_hold[l][13*(fp/10)+12]}},z_hold[l][13*(fp/10)+:13]};
   if(packed_retire)begin
    multiply_coefficient[l]=$signed({1'b0,count_high[l/2][6:0],3'd0,count_low[l/2][7:0]});
    alu_scalar[l*13+:13]={{10{q_hold[l][2]}},q_hold[l]};
    lhs[l]={6'd0,z_hold[l][26*(fp/20)+:26]};
   end
  end
  if(state==BM_CACHED_POP)begin lhs[l]=bm_acc[l];rhs[l]=0;end
  if(state==BM_NATIVE_ADD)begin lhs[l]=bm_acc[l];rhs[l]={{29{q_hold[l][2]}},q_hold[l]};end
  alu_lhs[l*32+:32]=lhs[l];alu_rhs[l*32+:32]=rhs[l];
  alu_coefficient[l*19+:19]=multiply_coefficient[l];
 end
 result_valid=(state==DRAIN_SEND);result_addr=9'(row);debug_state=state;
 end

 always_comb begin
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
  qblock_read_index=(state==BM_CACHED_POP)?5'(bm_plane):{2'd0,selected_rank};
  qblock_read_enable=resource_grant&&((state==BASE_MAC)||(state==BM_CACHED_POP&&bm_plane_live));
  for(integer i=0;i<8;i=i+1)begin
   qblock_read_bus[i*16+:16]=qblock_read_enable?qblock[qblock_read_index][i]:16'd0;
   bm_write[i]=z_hold[i];bm_write[i][13*(fp/10)+:13]=bm_acc[i][12:0];
  end
  pop_plane=qblock_read_bus;
 end
 always_comb begin
  for(integer lane=0;lane<8;lane=lane+1)add_y[lane]=alu_result[lane*32+:32];
 end
 always_comb begin
  correction_count=0;repair_needed=0;
  for(integer lane=0;lane<8;lane=lane+1)for(integer p=0;p<4;p=p+1)begin
   next_correction[lane][p]=0;
   if(state==ZADD && mode_q==2 && active[p] &&
      (z_hold[lane][p*13+7]==q_hold[lane][2]) &&
      (z_hold[lane][p*13+7]!=add_y[lane][p*8+7]))begin
    next_correction[lane][p]=z_hold[lane][p*13+7]?-2'sd1:2'sd1;
    repair_needed=1;
    correction_count=correction_count+6'd1;
   end
  end
 end

 always_comb begin
 inverse_time=4'(row%10);
 if(mode_q==21)for(integer t=0;t<10;t=t+1)
  if(time_permutation[t*4+:4]==4'(row%10))inverse_time=4'(t);
 has_direct=0;has_count=0;active_groups=0;
 for(integer g=0;g<4;g=g+1)begin
  has_direct=has_direct||(weight_data[g*6+:6]==63);
  has_count=has_count||(weight_data[g*6+:6]>0&&weight_data[g*6+:6]<=32);
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
   p_read_enable[i]=resource_grant&&count_live[i/2][(int'(class_hold[i/2])-1)*5+cblock];
   p_addr[i]=9'((int'(class_hold[i/2])-1)*5+cblock);
  end else if(state==G_READ)begin
   p_read_enable[i]=resource_grant&&count_live[i/2][group_iter*5+cblock];p_addr[i]=9'(group_iter*5+cblock);
  end else if(state==C_ADD&&class_hold[i/2]>0&&class_hold[i/2]<=32)begin
   p_write_enable[i]=resource_grant;p_addr[i]=9'((int'(class_hold[i/2])-1)*5+cblock);p_write_data[i]=add_y[i];
  end else if(state==STORE)begin
   p_write_enable[i]=resource_grant;p_addr[i]=9'(og*40+fp);p_write_data[i]=acc[i];
  end else if(state==DRAIN_READ)begin
   p_read_enable[i]=resource_grant;p_addr[i]=9'((row/10)*10+int'(inverse_time));
  end
  p_read_data[i]=p_read_enable[i]?p_mem[i][p_addr[i]]:32'd0;
  if(state==C_CHECK&&class_hold[i/2]>0&&class_hold[i/2]<=32&&count_live[i/2][(int'(class_hold[i/2])-1)*5+cblock])read_banks=read_banks+1;
  if(state==G_READ&&count_live[i/2][group_iter*5+cblock])read_banks=read_banks+1;
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
 packed_retire=((mode_q==20||mode_q==21))&&packed_safe;
 for(integer g=0;g<4;g=g+1)count_scalar[g]=ghalf?count_high[g]:count_low[g];

 correction_carry=0;
 for(integer l=0;l<8;l=l+1)correction_carry[l]=(state==G_MAC&&packed_retire)&&q_hold[l][2]&&(count_low[l/2]!=0);
 end

 always_comb begin
  // Full52bit write from the previously authorized row read; no bit-write memory port.
  for(integer lane=0;lane<8;lane=lane+1)begin
   retire_write[lane]=z_hold[lane];
   if(packed_retire)retire_write[lane][26*(fp/20)+:26]=add_y[lane][25:0];
   else retire_write[lane][13*(fp/10)+:13]=add_y[lane][12:0];
  end
 end
 always_ff @(posedge clk)begin
 if(!reset_n)begin
  bm_block<=0;bm_plane<=0;bm_hold<=0;bm_pending<=0;bitmap_live<=0;
  for(integer i=0;i<8;i=i+1)bm_acc[i]<=0;
  block_pending<=0;for(integer g=0;g<4;g=g+1)begin count_live[g]<=0;class_hold[g]<=0;end
  count_active<=0;cblock<=0;group_iter<=0;group_pending<=0;group_live<=0;
  for(integer i=0;i<8;i=i+1)count_hold[i]<=0;
  state<=IDLE;mode_q<=0;done<=0;compute_done<=0;fallback_used<=0;oy<=0;ox<=0;
  k<=0;fp<=0;zrow<=0;og<=0;qfill<=0;row<=0;load_xy<=0;
  rank_live<=0;block_live<=0;remaining<=0;pending<=0;result_data<=0;active<=0;pair_sel<=0;clear_counters();
  for(integer i=0;i<4;i=i+1)src_masks[i]<=0;
  for(integer i=0;i<8;i=i+1)begin
   q_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;
   for(integer p=0;p<4;p=p+1)correction[i][p]<=0;
  end
 end else begin
  done<=0;
  for(integer i=0;i<8;i=i+1)if(p_write_enable[i])p_mem[i][p_addr[i]]<=p_write_data[i];
  if(cfg_valid&&state==IDLE)case(cfg_kind)
   0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
   3:begin oy<=cfg_data[15:0];ox<=cfg_data[31:16];end
   4:begin end
   5:begin
    v_live[cfg_addr[6:0]]<=cfg_v_live;
   end
   6:k_live[cfg_addr[9:0]]<=cfg_data[0];
   default:begin end
  endcase
  if(state!=IDLE)cycles<=cycles+1;
  if(resource_request!=0 && !resource_grant)begin
   if(resource_request[0]&&!source_allow)source_stalls<=source_stalls+1;
   else if(resource_request[1]&&!weight_allow)weight_stalls<=weight_stalls+1;
   else begin
    arbitration_stalls<=arbitration_stalls+1;
    if(bitmap_mode)begin
     if(resource_request[2])bitmap_z_arbitration_stalls<=bitmap_z_arbitration_stalls+1;
     if(resource_request[4])bitmap_alu_arbitration_stalls<=bitmap_alu_arbitration_stalls+1;
     if(resource_request[1])bitmap_weight_arbitration_stalls<=bitmap_weight_arbitration_stalls+1;
     if(state==BM_POS||state==BM_STORE)bitmap_row_z_arbitration_stalls<=bitmap_row_z_arbitration_stalls+1;
    end
   end
   if(state==REPAIR_READ || state==REPAIR_ADD)repair_arbitration_stalls<=repair_arbitration_stalls+1;
   if(state==NORMALIZE_READ || state==NORMALIZE_ADD)normalization_arbitration_stalls<=normalization_arbitration_stalls+1;
  end
  if(resource_request==0 || resource_grant)case(state)
   IDLE:if(start)begin
    compute_done<=0;block_pending<=0;group_live<=0;for(integer g=0;g<4;g=g+1)count_live[g]<=0;
    mode_q<=(mode==1&&!range_ok)?5'd0:mode;fallback_used<=(mode==1&&!range_ok);
    zrow<=0;rank_live<=0;block_live<=0;state<=ZCLEAR;clear_counters();
   end
   ZCLEAR:begin
    for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=0;
    z_writes<=z_writes+1;
    if(zrow==9)begin k<=0;state<=L_START;end else zrow<=zrow+1;
   end
   L_START:begin load_xy<=0;state<=L_LOAD;end
   L_LOAD:if(!in_bounds||source_allow)begin
    local_source[load_xy]<=in_bounds?source_mem[source_addr]:10'd0;
    if(in_bounds)source_words<=source_words+1;
    if(load_xy==15)state<=L_GATHER;else load_xy<=load_xy+1;
   end else source_stalls<=source_stalls+1;
   L_GATHER:if(bitmap_mode)begin
    if(k_live[k])local_source_reads<=local_source_reads+1;
    for(integer i=0;i<40;i=i+1)begin
     bitmap[i][k%16]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
     if(k%16==0)bitmap_live[i]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
     else bitmap_live[i]<=bitmap_live[i]||(k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10]);
    end
    aux_writes<=aux_writes+1;state<=KNEXT;
   end else if(!k_live[k])state<=KNEXT;
   else begin
    for(integer i=0;i<4;i=i+1)begin
     if(mode_q==21)for(integer t=0;t<10;t=t+1)src_masks[i][t]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3][time_permutation[t*4+:4]];
     else src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];
    end
    local_source_reads<=local_source_reads+1;state<=CHECK;
   end
   CHECK:begin
    if((mode_q==2||mode_q==3))pending<={30'd0,(src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3])};
    else if(mode_q==1)pending<={20'd0,src_masks[3],(src_masks[0]|src_masks[1]|src_masks[2])};
    else pending<={20'd0,(src_masks[2]|src_masks[3]),(src_masks[0]|src_masks[1])};
    state<=((src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3])==0)?KNEXT:(count_mode?MREAD:QREAD);
   end
   QREAD:if(weight_allow)begin
    for(integer i=0;i<8;i=i+1)q_hold[i]<=(!count_mode||class_hold[i/2]==63)?weight_data[i*32+:3]:3'sd0;
    weight_words<=weight_words+1;state<=TIMESEL;
   end else weight_stalls<=weight_stalls+1;
   TIMESEL:if(pending!=0)begin
    zrow<=selected_time%10;pair_sel<=(selected_time>=10);
    for(integer p=0;p<4;p=p+1)
     active[p]<=src_masks[p][selected_time%10] &&
       ((mode_q==2||mode_q==3) || (mode_q==1 ? (p==3)==(selected_time>=10) : p/2==selected_time/10));
    state<=ZREAD;
   end else if(count_mode&&count_active)begin
    cblock<=0;for(integer i=4;i>=0;i=i-1)if(block_pending[i])cblock<=i;state<=C_CHECK;
   end else state<=KNEXT;
   ZREAD:begin
    for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_bus[i*52+:52];
    z_vector_reads<=z_vector_reads+1;state<=ZADD;
   end
   ZADD:begin
    for(integer i=0;i<8;i=i+1)begin
     if(mode_q==3)z_mem[i][zrow]<=borrow_y[i*52+:52];
     else if(mode_q==0||count_mode)begin
      if(pair_sel)z_mem[i][zrow]<={add_y[i][25:0],z_hold[i][25:0]};
      else z_mem[i][zrow]<={z_hold[i][51:26],add_y[i][25:0]};
     end else if(mode_q==1)begin
      if(pair_sel)z_mem[i][zrow]<={{{3{add_y[i][9]}},add_y[i][9:0]},z_hold[i][38:0]};
      else z_mem[i][zrow]<={z_hold[i][51:39],{{3{add_y[i][29]}},add_y[i][29:20]},
                          {{3{add_y[i][19]}},add_y[i][19:10]},{{3{add_y[i][9]}},add_y[i][9:0]}};
     end else begin
      z_mem[i][zrow]<={z_hold[i][51:47],add_y[i][31:24],z_hold[i][38:34],add_y[i][23:16],
                      z_hold[i][25:21],add_y[i][15:8],z_hold[i][12:8],add_y[i][7:0]};
      for(integer p=0;p<4;p=p+1)correction[i][p]<=next_correction[i][p];
     end
    end
    first_issues<=first_issues+1;z_writes<=z_writes+1;
    merged_updates<=merged_updates+32'(active[0])+32'(active[1])+32'(active[2])+32'(active[3])-32'd1;
    pending<=pending&(pending-40'd1);
    if(mode_q==2 && repair_needed)begin
     repair_fields<=repair_fields+32'(correction_count);state<=REPAIR_READ;
    end else state<=TIMESEL;
   end
   REPAIR_READ,NORMALIZE_READ:begin
    for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_bus[i*52+:52];
    z_vector_reads<=z_vector_reads+1;
    state<=(state==REPAIR_READ)?REPAIR_ADD:NORMALIZE_ADD;
   end
   REPAIR_ADD,NORMALIZE_ADD:begin
    for(integer i=0;i<8;i=i+1)
     z_mem[i][zrow]<={add_y[i][19:15],z_hold[i][46:39],add_y[i][14:10],z_hold[i][33:26],
                     add_y[i][9:5],z_hold[i][20:13],add_y[i][4:0],z_hold[i][7:0]};
    z_writes<=z_writes+1;
    if(state==REPAIR_ADD)begin repair_issues<=repair_issues+1;state<=TIMESEL;end
    else begin
     normalization_issues<=normalization_issues+1;
     if(zrow==9)begin fp<=0;state<=ZSCAN;end else begin zrow<=zrow+1;state<=NORMALIZE_READ;end
    end
   end
   KNEXT:if(bitmap_mode&&k%16==15)state<=BM_GROUP;
   else if(k==863)begin
    if(mode_q==2)begin zrow<=0;state<=NORMALIZE_READ;end else begin fp<=0;group_iter<=0;state<=(count_mode&&ngroups!=0)?G_START:ZSCAN;end
   end else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end

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
 MREAD:begin
 for(integer g=0;g<4;g=g+1)begin
  class_hold[g]<=weight_data[g*6+:6];
  if(weight_data[g*6+:6]>0&&weight_data[g*6+:6]<=32)group_live[int'(weight_data[g*6+:6])-1]<=1;
 end
 metadata_reads<=metadata_reads+1;count_active<=has_count;cblock<=0;block_pending<=source_blocks;
 for(integer i=4;i>=0;i=i-1)if(source_blocks[i])cblock<=i;
 state<=has_direct?QREAD:(has_count?C_CHECK:KNEXT);
 end
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
 for(integer i=0;i<8;i=i+1)q_hold[i]<=weight_data[i*32+:3];
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
 for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_bus[i*52+:52];
 z_vector_reads<=z_vector_reads+1;state<=G_MAC;
 end
 G_MAC:begin
 for(integer i=0;i<8;i=i+1)
 z_mem[i][read_zrow]<=retire_write[i];
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
     for(integer i=0;i<8;i=i+1)qblock[qfill][i]<=(rank_live[qfill]&&v_live[og*8+qfill])?weight_data[i*32+:16]:16'sd0;
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
    else begin compute_done<=1;row<=0;state<=DRAIN_READ;end
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
 logic check_retired,check_bm_owned;logic [5:0] check_bm_row;integer check_stores;
 always_ff @(posedge clk)begin
 if(!reset_n)begin check_retired<=0;check_stores<=0;check_bm_owned<=0;check_bm_row<=0;end
 else begin
  if(state==IDLE&&start)begin
   check_retired<=0;check_stores<=0;check_bm_owned<=0;
   if(mode==21)for(integer t=0;t<10;t=t+1)begin
    if(time_permutation[t*4+:4]>=10)$fatal(1,"permutation range");
    for(integer u=0;u<t;u=u+1)if(time_permutation[t*4+:4]==time_permutation[u*4+:4])$fatal(1,"permutation duplicate");
   end
  end
  if(mode_q==7)begin
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
  if(state==ZSCAN)check_retired<=1;
  if(state==BM_WFILL||state==BM_CACHED_POP||state==BM_NATIVE_ADD)begin
   if(check_retired)$fatal(1,"bitmap qblock use after Q2 starts");
   if(bm_block>=54||bm_plane>=3)$fatal(1,"bitmap address range");
  end
  if(!resource_grant&&qblock_read_bus!=0)$fatal(1,"ungranted qblock read");
  if(state==C_CHECK||state==C_ADD||state==G_READ)begin
   if(check_retired)$fatal(1,"count access after retirement");
   for(integer i=0;i<8;i=i+1)if((p_read_enable[i]||p_write_enable[i])&&p_addr[i]>=160)$fatal(1,"count address range");
  end
  if(state==STORE&&resource_grant)begin
   if(!check_retired||og*40+fp!=check_stores)$fatal(1,"output overwrite ordering");
   check_stores<=check_stores+1;
  end
  if(state==DRAIN_READ&&resource_grant&&check_stores!=480)$fatal(1,"drain before all outputs overwritten");
  if(resource_request!=0&&!resource_grant)begin
   if(p_read_enable!=0||p_write_enable!=0||z_read_bus!=0)$fatal(1,"ungranted memory access");
  end
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
