module decomp_core(
 input logic clk,reset_n,cfg_valid,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic start,input logic [3:0] mode,input logic source_allow,weight_allow,
 output logic result_valid,input logic result_ready,
 output logic [8:0] result_addr,output logic [255:0] result_data,output logic done,
 output logic [31:0] cycles,source_words,weight_words,second_weight_words,local_source_reads,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,first_issues,dual_updates,
 output logic [31:0] psum_reads,psum_writes,mac_issues,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [31:0] direct_issues,endpoint_issues,selection_issues,prefix_issues,merge_issues,
 output logic [31:0] selected_direct_columns,selected_endpoint_columns,direct_pair_work,endpoint_pair_work,
 output logic [31:0] prefix_reads,merge_reads,zclear_writes,falling_fields,
 output logic [31:0] preclassified_direct_columns,empty_columns_skipped,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,L_START,L_LOAD,L_GATHER,CHECK,QREAD,TIMESEL,
 ZREAD,ZADD,KNEXT,ZSCAN,VLOAD,POSLOAD,BASE_MAC,STORE,DRAIN_READ,DRAIN_SEND,FINISH,
 SELECT,PREFIX_READ,PREFIX_ADD,MERGE_READ,MERGE_ADD,LATE_CLEAR} state_t;
 state_t state;logic [3:0] mode_q;
 logic [9:0] source_mem[0:1535],local_source[0:15],src_masks[0:3],delta_masks[0:3];
 logic signed [2:0] q_mem[0:7][0:863],q_hold[0:7];
 logic signed [15:0] v_mem[0:7][0:95],qblock[0:7][0:7];
 logic k_live[0:863],v_live[0:95];
 // One 208-bit port over a union of two 520B domains. No second read port.
 logic [25:0] z_mem[0:7][0:39],z_hold[0:7];
 logic [207:0] z_read_bus;
 logic [5:0] z_read_address;
 logic signed [31:0] p_mem[0:7][0:479],acc[0:7];
 logic [7:0] position_live[0:39],rank_live,block_live,remaining,scan_mask;
 logic [19:0] pending,direct_mask,endpoint_mask;
 logic signed [15:0] oy,ox;
 integer k,fp,zrow,og,qfill,row,load_xy,selected_time;
 logic [10:0] source_addr;logic [4:0] read_zrow;
 logic in_bounds,cfg_v_live,active_lo,active_hi,negative_lo,negative_hi,read_half;
 logic k_endpoint,any_endpoint,has_adjacent;
 logic [3:0] time_order[0:9],inverse_time;
 logic [8:0] drain_row;
 logic [5:0] direct_count,endpoint_count;
 logic [2:0] selected_rank;
 logic signed [12:0] scalar;
 logic signed [18:0] multiply_coefficient[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7],add_y[0:7];
 logic split_fields,sub_lo,sub_hi;
 logic mixed_mode,endpoint_mode,permuted_mode;

 task clear_counters;
 begin
 cycles<=0;source_words<=0;weight_words<=0;second_weight_words<=0;local_source_reads<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;first_issues<=0;dual_updates<=0;
 psum_reads<=0;psum_writes<=0;mac_issues<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 direct_issues<=0;endpoint_issues<=0;selection_issues<=0;prefix_issues<=0;merge_issues<=0;
 selected_direct_columns<=0;selected_endpoint_columns<=0;direct_pair_work<=0;endpoint_pair_work<=0;
 prefix_reads<=0;merge_reads<=0;zclear_writes<=0;falling_fields<=0;
 preclassified_direct_columns<=0;empty_columns_skipped<=0;
 end
 endtask

 always_comb begin
 mixed_mode=(mode_q==2 || mode_q==3 || mode_q==4);
 endpoint_mode=(mode_q==1 || mode_q==5);permuted_mode=(mode_q==4 || mode_q==5);
 source_addr=11'((k/9)*16+load_xy);
 in_bounds=(int'(oy)+load_xy/4>=0 && int'(oy)+load_xy/4<240 &&
 int'(ox)+load_xy%4>=0 && int'(ox)+load_xy%4<320);
 cfg_v_live=0;for(integer i=0;i<8;i=i+1)cfg_v_live=cfg_v_live||(cfg_data[i*32+:16]!=0);
 inverse_time=0;
 for(integer t=0;t<10;t=t+1)if(time_order[t]==4'(row%10))inverse_time=4'(t);
 drain_row=permuted_mode?9'(row-row%10+int'(inverse_time)):9'(row);
 for(integer p=0;p<4;p=p+1)delta_masks[p]=src_masks[p]^{src_masks[p][8:0],1'b0};
 has_adjacent=0;
 for(integer p=0;p<4;p=p+1)has_adjacent=has_adjacent||((src_masks[p]&{src_masks[p][8:0],1'b0})!=0);
 direct_mask={(src_masks[2]|src_masks[3]),(src_masks[0]|src_masks[1])};
 endpoint_mask={(delta_masks[2]|delta_masks[3]),(delta_masks[0]|delta_masks[1])};
 direct_count=0;endpoint_count=0;
 for(integer t=0;t<20;t=t+1)begin
  direct_count=direct_count+6'(direct_mask[t]);endpoint_count=endpoint_count+6'(endpoint_mask[t]);
 end
 selected_time=0;for(integer i=19;i>=0;i=i-1)if(pending[i])selected_time=i;
 selected_rank=0;for(integer i=7;i>=0;i=i-1)if(remaining[i])selected_rank=3'(i);
 read_zrow=5'((fp/20)*10+fp%10);read_half=((fp/10)%2)!=0;
 z_read_address=6'(zrow);
 if(state==ZSCAN || state==BASE_MAC)z_read_address={1'b0,read_zrow};
 if(state==PREFIX_READ && mixed_mode)z_read_address=6'(zrow+20);
 z_read_bus=0;
 for(integer l=0;l<8;l=l+1)
  if(state==ZREAD || state==PREFIX_READ || state==MERGE_READ || state==ZSCAN ||
     (state==BASE_MAC && selected_rank==3'(l)))z_read_bus[l*26+:26]=z_mem[l][z_read_address];
 scan_mask=0;
 if(state==ZSCAN)for(integer i=0;i<8;i=i+1)
  scan_mask[i]=read_half?(z_read_bus[i*26+13+:13]!=0):(z_read_bus[i*26+:13]!=0);
 scalar=read_half?$signed(z_read_bus[int'(selected_rank)*26+13+:13]):$signed(z_read_bus[int'(selected_rank)*26+:13]);
 split_fields=(state==ZADD || state==PREFIX_ADD || state==MERGE_ADD);
 sub_lo=(state==ZADD && negative_lo);sub_hi=(state==ZADD && negative_hi);
 for(integer l=0;l<8;l=l+1)begin
  multiply_coefficient[l]={{3{qblock[selected_rank][l][15]}},qblock[selected_rank][l]};
  product[l]=$signed(multiply_coefficient[l])*$signed(scalar);
  lhs[l]=acc[l];rhs[l]=product[l];
  if(state==ZADD)begin
   lhs[l]={6'b0,z_hold[l]};
   // Negation is a 13-bit XOR/carry-in on the same ALU, including -(-4)=+4.
   rhs[l]={6'b0,((active_hi?{{10{q_hold[l][2]}},q_hold[l]}:13'd0)^{13{sub_hi}}),
                     ((active_lo?{{10{q_hold[l][2]}},q_hold[l]}:13'd0)^{13{sub_lo}})};
  end else if(state==PREFIX_ADD || state==MERGE_ADD)begin
   lhs[l]={6'd0,z_hold[l]};rhs[l]=acc[l];
  end
 end
 result_valid=(state==DRAIN_SEND);result_addr=9'(row);debug_state=state;
 end
 genvar l,b;
 generate for(l=0;l<8;l=l+1)begin:ALU
 for(b=0;b<32;b=b+1)begin:BIT
 wire cin;
 if(b==0)assign cin=sub_lo;
 else if(b==13)assign cin=split_fields?sub_hi:BIT[b-1].CARRY.cout;
 else assign cin=BIT[b-1].CARRY.cout;
 assign add_y[l][b]=lhs[l][b]^rhs[l][b]^cin;
 if(b<31)begin:CARRY
 wire cout;assign cout=(lhs[l][b]&rhs[l][b])|((lhs[l][b]^rhs[l][b])&cin);
 end
 end
 end endgenerate

 always_ff @(posedge clk)begin
 if(!reset_n)begin
 state<=IDLE;mode_q<=0;done<=0;oy<=0;ox<=0;k<=0;fp<=0;zrow<=0;og<=0;qfill<=0;row<=0;load_xy<=0;
 rank_live<=0;block_live<=0;remaining<=0;pending<=0;result_data<=0;active_lo<=0;active_hi<=0;
 negative_lo<=0;negative_hi<=0;k_endpoint<=0;any_endpoint<=0;clear_counters();
 for(integer i=0;i<4;i=i+1)src_masks[i]<=0;
 for(integer t=0;t<10;t=t+1)time_order[t]<=4'(t);
 for(integer i=0;i<8;i=i+1)begin q_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;end
 end else begin
 done<=0;
 if(cfg_valid&&state==IDLE)case(cfg_kind)
 0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
 3:begin oy<=cfg_data[15:0];ox<=cfg_data[31:16];end
 4:for(integer i=0;i<8;i=i+1)q_mem[i][cfg_addr[9:0]]<=cfg_data[i*32+:3];
 5:begin
  for(integer i=0;i<8;i=i+1)v_mem[i][cfg_addr[6:0]]<=cfg_data[i*32+:16];
  v_live[cfg_addr[6:0]]<=cfg_v_live;
 end
 6:k_live[cfg_addr[9:0]]<=cfg_data[0];
 7:for(integer t=0;t<10;t=t+1)time_order[t]<=cfg_data[t*4+:4];
 default:begin end
 endcase
 if(state!=IDLE)cycles<=cycles+1;
 case(state)
 IDLE:if(start)begin
  mode_q<=mode;zrow<=0;rank_live<=0;block_live<=0;any_endpoint<=0;
  state<=ZCLEAR;clear_counters();
 end
 ZCLEAR:begin
  for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=0;
  z_writes<=z_writes+1;zclear_writes<=zclear_writes+1;
  if(zrow==((mode_q==2)?39:19))begin k<=0;state<=L_START;end else zrow<=zrow+1;
 end
 L_START:begin load_xy<=0;state<=L_LOAD;end
 L_LOAD:if(!in_bounds||source_allow)begin
  local_source[load_xy]<=in_bounds?source_mem[source_addr]:10'd0;
  if(in_bounds)source_words<=source_words+1;
  if(load_xy==15)state<=L_GATHER;else load_xy<=load_xy+1;
 end else source_stalls<=source_stalls+1;
 L_GATHER:if(!k_live[k])state<=KNEXT;
 else begin
  // The source words stay in their native order at the external interface.
  // Only execution after the complete gate word is available is permuted.
  for(integer i=0;i<4;i=i+1)for(integer t=0;t<10;t=t+1)
   src_masks[i][t]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3][permuted_mode?int'(time_order[t]):t];
  local_source_reads<=local_source_reads+1;state<=CHECK;
 end
 CHECK:begin
  direct_pair_work<=direct_pair_work+32'(direct_count);endpoint_pair_work<=endpoint_pair_work+32'(endpoint_count);
  pending<=endpoint_mode?endpoint_mask:direct_mask;k_endpoint<=endpoint_mode;
  if(mode_q==2)state<=SELECT;
  else if(mode_q==3 || mode_q==4)begin
   if(direct_mask==0)begin empty_columns_skipped<=empty_columns_skipped+1;state<=KNEXT;end
   else if(!has_adjacent)begin
    selected_direct_columns<=selected_direct_columns+1;
    preclassified_direct_columns<=preclassified_direct_columns+1;state<=QREAD;
   end else state<=SELECT;
  end
  else state<=(direct_mask==0)?KNEXT:QREAD;
 end
 SELECT:begin
  selection_issues<=selection_issues+1;
  k_endpoint<=endpoint_count<direct_count;
  pending<=(endpoint_count<direct_count)?endpoint_mask:direct_mask;
  if(direct_mask!=0)begin
   if(endpoint_count<direct_count)begin
    any_endpoint<=1;selected_endpoint_columns<=selected_endpoint_columns+1;
   end else selected_direct_columns<=selected_direct_columns+1;
  end
  if(mode_q>=3 && endpoint_count<direct_count && !any_endpoint)begin zrow<=20;state<=LATE_CLEAR;end
  else state<=(direct_mask==0)?KNEXT:QREAD;
 end
 LATE_CLEAR:begin
  // Preserve current k, masks, pending and chosen representation. The first
  // endpoint column waits while the shared port clears its full second domain.
  for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=0;
  z_writes<=z_writes+1;zclear_writes<=zclear_writes+1;
  if(zrow==39)state<=QREAD;else zrow<=zrow+1;
 end
 QREAD:if(weight_allow)begin
  for(integer i=0;i<8;i=i+1)q_hold[i]<=q_mem[i][k];
  weight_words<=weight_words+1;state<=TIMESEL;
 end else weight_stalls<=weight_stalls+1;
 TIMESEL:if(pending!=0)begin
  zrow<=selected_time+((mixed_mode&&k_endpoint)?20:0);
  active_lo<=k_endpoint?delta_masks[(selected_time/10)*2][selected_time%10]:src_masks[(selected_time/10)*2][selected_time%10];
  active_hi<=k_endpoint?delta_masks[(selected_time/10)*2+1][selected_time%10]:src_masks[(selected_time/10)*2+1][selected_time%10];
  negative_lo<=k_endpoint&&delta_masks[(selected_time/10)*2][selected_time%10]&&!src_masks[(selected_time/10)*2][selected_time%10];
  negative_hi<=k_endpoint&&delta_masks[(selected_time/10)*2+1][selected_time%10]&&!src_masks[(selected_time/10)*2+1][selected_time%10];
  state<=ZREAD;
 end else state<=KNEXT;
 ZREAD:begin
  for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_bus[i*26+:26];
  z_vector_reads<=z_vector_reads+1;state<=ZADD;
 end
 ZADD:begin
  for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=add_y[i][25:0];
  first_issues<=first_issues+1;z_writes<=z_writes+1;
  if(k_endpoint)endpoint_issues<=endpoint_issues+1;else direct_issues<=direct_issues+1;
  falling_fields<=falling_fields+32'(negative_lo)+32'(negative_hi);
  if(active_lo&&active_hi)dual_updates<=dual_updates+1;
  pending<=pending&(pending-20'd1);state<=TIMESEL;
 end
 KNEXT:if(k==863)begin
  fp<=0;zrow<=0;for(integer i=0;i<8;i=i+1)acc[i]<=0;
  state<=(endpoint_mode || (mixed_mode&&any_endpoint))?PREFIX_READ:ZSCAN;
 end else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end
 PREFIX_READ:begin
  for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_bus[i*26+:26];
  prefix_reads<=prefix_reads+1;z_vector_reads<=z_vector_reads+1;state<=PREFIX_ADD;
 end
 PREFIX_ADD:begin
  for(integer i=0;i<8;i=i+1)begin
   z_mem[i][zrow+(mixed_mode?20:0)]<=add_y[i][25:0];
   acc[i]<=(endpoint_mode&&zrow==9)?32'd0:{6'd0,add_y[i][25:0]};
  end
  prefix_issues<=prefix_issues+1;z_writes<=z_writes+1;
  if(mixed_mode)state<=MERGE_READ;
  else if(zrow==19)begin fp<=0;state<=ZSCAN;end
  else begin zrow<=zrow+1;state<=PREFIX_READ;end
 end
 MERGE_READ:begin
  for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_bus[i*26+:26];
  merge_reads<=merge_reads+1;z_vector_reads<=z_vector_reads+1;state<=MERGE_ADD;
 end
 MERGE_ADD:begin
  for(integer i=0;i<8;i=i+1)begin
   z_mem[i][zrow]<=add_y[i][25:0];
   if(zrow==9)acc[i]<=0;
  end
  merge_issues<=merge_issues+1;z_writes<=z_writes+1;
  if(zrow==19)begin fp<=0;state<=ZSCAN;end
  else begin zrow<=zrow+1;state<=PREFIX_READ;end
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
  remaining<=position_live[fp]&block_live;state<=((position_live[fp]&block_live)==0)?STORE:BASE_MAC;
 end
 BASE_MAC:begin
  for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
  mac_issues<=mac_issues+1;z_scalar_reads<=z_scalar_reads+1;
  remaining<=remaining&(remaining-8'd1);
  if((remaining&(remaining-8'd1))==0)state<=STORE;
 end
 STORE:begin
  for(integer i=0;i<8;i=i+1)p_mem[i][og*40+fp]<=acc[i];
  psum_writes<=psum_writes+1;
  if(fp<39)begin fp<=fp+1;state<=POSLOAD;end
  else if(og<11)begin og<=og+1;state<=VLOAD;qfill<=0;end
  else begin row<=0;state<=DRAIN_READ;end
 end
 DRAIN_READ:begin
  // All Q2 outputs are materialized before drain. Restore the original T
  // address using the existing psum read; no extra copy or read port.
  for(integer i=0;i<8;i=i+1)result_data[i*32+:32]<=p_mem[i][drain_row];
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
endmodule
